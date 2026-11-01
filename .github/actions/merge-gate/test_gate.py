"""The gate builds every N merges or H hours, never twice for one head, and fails open."""

import unittest
from datetime import datetime, timedelta, timezone

import gate

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
GATE = "Merge gate"


def run(run_id, sha, event="push", status="completed", hours_ago=1.0):
    created = (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"id": run_id, "head_sha": sha, "event": event, "status": status,
            "created_at": created}


def job(name, conclusion=None, status="completed"):
    return {"name": name, "status": status, "conclusion": conclusion}


class BuiltTests(unittest.TestCase):
    def test_gate_only_run_is_not_a_build(self):
        self.assertFalse(gate.built([job(GATE, "success"), job("HIP", "skipped")], GATE))

    def test_finished_or_running_build_job_counts(self):
        self.assertTrue(gate.built([job(GATE, "success"), job("HIP", "failure")], GATE))
        self.assertTrue(gate.built([job(GATE, "success"), job("HIP", None, "in_progress")], GATE))

    def test_cancelled_build_does_not_count(self):
        self.assertFalse(gate.built([job(GATE, "success"), job("HIP", "cancelled")], GATE))


class LastBuildTests(unittest.TestCase):
    def test_skips_current_gated_out_and_pr_runs(self):
        runs = [run(5, "e" * 40), run(4, "d" * 40), run(3, "c" * 40, event="pull_request"),
                run(2, "b" * 40, status="queued"), run(1, "a" * 40)]
        jobs = {4: [job(GATE, "success"), job("HIP", "skipped")],
                3: [job("HIP", "success")], 1: [job(GATE, "success"), job("HIP", "success")]}
        last = gate.last_build(runs, lambda i: jobs.get(i, []), GATE, current_run=5)
        self.assertEqual(last["id"], 1)

    def test_none_without_builds(self):
        self.assertIsNone(gate.last_build([run(1, "a" * 40)], lambda i: [job(GATE, "success")],
                                          GATE, current_run=9))


class DecideTests(unittest.TestCase):
    last = run(7, "a" * 40)

    def test_pull_request_and_dispatch_always_build(self):
        for event in ("pull_request", "workflow_dispatch"):
            self.assertTrue(gate.decide(event, "b" * 40, None, None, None, 5, 6)[0])

    def test_first_build(self):
        self.assertEqual(gate.decide("push", "b" * 40, None, None, None, 5, 6),
                         (True, "no earlier build on this branch"))

    def test_same_head_never_rebuilds(self):
        ok, reason = gate.decide("schedule", "a" * 40, self.last, 0, 30.0, 5, 6)
        self.assertFalse(ok)
        self.assertIn("already built by run 7", reason)

    def test_enough_merges(self):
        self.assertTrue(gate.decide("push", "b" * 40, self.last, 5, 0.5, 5, 6)[0])
        self.assertFalse(gate.decide("push", "b" * 40, self.last, 4, 0.5, 5, 6)[0])

    def test_enough_hours(self):
        self.assertTrue(gate.decide("schedule", "b" * 40, self.last, 1, 6.0, 5, 6)[0])
        self.assertFalse(gate.decide("schedule", "b" * 40, self.last, 1, 5.9, 5, 6)[0])

    def test_unknown_distance_builds(self):
        self.assertTrue(gate.decide("push", "b" * 40, self.last, None, None, 5, 6)[0])


class EvaluateTests(unittest.TestCase):
    env = {"GITHUB_EVENT_NAME": "push", "GITHUB_REPOSITORY": "o/r", "GATE_WORKFLOW": "hip.yml",
           "GITHUB_REF_NAME": "development", "GITHUB_SHA": "b" * 40, "GATE_JOB": GATE,
           "GATE_MERGES": "5", "GATE_HOURS": "6", "GITHUB_RUN_ID": "9"}

    def fetch_with(self, runs, jobs, compare):
        def fetch(url):
            if "/workflows/hip.yml/runs?branch=development" in url:
                return {"workflow_runs": runs}
            if "/jobs?" in url:
                return {"jobs": jobs[int(url.split("/runs/")[1].split("/")[0])]}
            if "/compare/" in url:
                return compare
            raise AssertionError(url)
        return fetch

    def test_counts_from_the_last_real_build(self):
        runs = [run(8, "c" * 40, hours_ago=0.2), run(7, "a" * 40, hours_ago=2.0)]
        jobs = {8: [job(GATE, "success"), job("HIP", "skipped")],
                7: [job(GATE, "success"), job("HIP", "success")]}
        fetch = self.fetch_with(runs, jobs, {"ahead_by": 3})
        self.assertEqual(gate.evaluate(self.env, fetch, NOW),
                         (False, "3/5 merges and 2.0/6 h since aaaaaaaaa"))
        fetch = self.fetch_with(runs, jobs, {"ahead_by": 6})
        self.assertTrue(gate.evaluate(self.env, fetch, NOW)[0])

    def test_pull_request_makes_no_api_calls(self):
        env = dict(self.env, GITHUB_EVENT_NAME="pull_request")
        def fetch(url):
            raise AssertionError("no API call expected")
        self.assertTrue(gate.evaluate(env, fetch, NOW)[0])


if __name__ == "__main__":
    unittest.main()
