"""Decide whether a development push or scheduled run should build.

A build happens when at least ``GATE_MERGES`` commits landed since the last run of this
workflow that built, or when that build is at least ``GATE_HOURS`` old; a head that is
already built never rebuilds. Pull requests and manual runs are never gated. Any error
lets the build run, so the gate can only save work, never hide a commit.
"""

import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

Fetch = Callable[[str], Any]

BUILT = ("success", "failure", "timed_out")
GATED_EVENTS = ("push", "schedule")


def github(token: str) -> Fetch:
    def fetch(url: str) -> Any:
        request = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    return fetch


def built(jobs: List[Dict[str, Any]], gate_job: str) -> bool:
    """True when a job other than the gate ran, is running, or finished for real."""
    for job in jobs:
        if job.get("name") == gate_job:
            continue
        if job.get("status") == "in_progress" or job.get("conclusion") in BUILT:
            return True
    return False


def last_build(runs: List[Dict[str, Any]], jobs_of: Callable[[int], List[Dict[str, Any]]],
               gate_job: str, current_run: int) -> Optional[Dict[str, Any]]:
    """Newest gated-event run on this branch whose build jobs actually ran."""
    for run in runs:
        if run.get("id") == current_run or run.get("event") not in GATED_EVENTS:
            continue
        if run.get("status") not in ("completed", "in_progress"):
            continue
        if built(jobs_of(run["id"]), gate_job):
            return run
    return None


def decide(event: str, head: str, last: Optional[Dict[str, Any]], ahead_by: Optional[int],
           age_hours: Optional[float], merges: int, hours: float) -> Tuple[bool, str]:
    if event not in GATED_EVENTS:
        return True, f"{event} runs are not gated"
    if last is None:
        return True, "no earlier build on this branch"
    sha = str(last.get("head_sha", ""))[:9]
    if last.get("head_sha") == head:
        return False, f"{head[:9]} already built by run {last.get('id')}"
    if ahead_by is None or age_hours is None:
        return True, f"cannot compare with run {last.get('id')} ({sha})"
    if ahead_by >= merges:
        return True, f"{ahead_by} merges since {sha} (limit {merges})"
    if age_hours >= hours:
        return True, f"{age_hours:.1f} h since {sha} (limit {hours:g} h), {ahead_by} merges"
    return False, f"{ahead_by}/{merges} merges and {age_hours:.1f}/{hours:g} h since {sha}"


def age(created_at: str, now: datetime) -> float:
    created = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return (now - created).total_seconds() / 3600


def evaluate(env: Dict[str, str], fetch: Fetch, now: datetime) -> Tuple[bool, str]:
    event = env["GITHUB_EVENT_NAME"]
    if event not in GATED_EVENTS:
        return decide(event, "", None, None, None, 0, 0)
    api = env.get("GITHUB_API_URL", "https://api.github.com")
    repo = env["GITHUB_REPOSITORY"]
    workflow = env["GATE_WORKFLOW"]
    branch = env["GITHUB_REF_NAME"]
    head = env["GITHUB_SHA"]
    gate_job = env["GATE_JOB"]
    merges = int(env["GATE_MERGES"])
    hours = float(env["GATE_HOURS"])
    current_run = int(env.get("GITHUB_RUN_ID", "0"))

    runs = fetch(f"{api}/repos/{repo}/actions/workflows/{workflow}/runs"
                 f"?branch={branch}&per_page=20")["workflow_runs"]

    def jobs_of(run_id: int) -> List[Dict[str, Any]]:
        return list(fetch(f"{api}/repos/{repo}/actions/runs/{run_id}/jobs?per_page=100")["jobs"])

    last = last_build(runs, jobs_of, gate_job, current_run)
    ahead_by: Optional[int] = None
    age_hours: Optional[float] = None
    if last is not None and last.get("head_sha") != head:
        try:
            compare = fetch(f"{api}/repos/{repo}/compare/{last['head_sha']}...{head}")
            ahead_by = int(compare["ahead_by"])
            age_hours = age(str(last["created_at"]), now)
        except (urllib.error.HTTPError, KeyError, ValueError):
            pass
    return decide(event, head, last, ahead_by, age_hours, merges, hours)


def main() -> int:
    env = dict(os.environ)
    try:
        run, reason = evaluate(env, github(env.get("GATE_TOKEN", "")), datetime.now(timezone.utc))
    except Exception as error:  # noqa: BLE001 - fail open, never hide a commit
        run, reason = True, f"gate error, building anyway: {error}"
    verdict = "build" if run else "skip"
    print(f"merge gate: {verdict}: {reason}")
    with open(env["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"run={'true' if run else 'false'}\nreason={reason}\n")
    summary = env.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"Merge gate: **{verdict}** ({reason})\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
