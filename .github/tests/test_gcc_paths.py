"""Check GCC build selection and its required-check contract without GitHub API calls."""

import json
from pathlib import Path
import subprocess
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = yaml.load((ROOT / ".github/workflows/gcc.yml").read_text(), Loader=yaml.BaseLoader)


class GccPathsTests(unittest.TestCase):
    def select(self, paths, event="pull_request", before="abc", error=False, forced=False):
        files = [p if isinstance(p, dict) else {"filename": p} for p in paths]
        script = WORKFLOW["jobs"]["changes"]["steps"][0]["with"]["script"]
        harness = """
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const context = {repo: {owner: 'example', repo: 'ERF'}, eventName: input.event,
  payload: {pull_request: {number: 1}, before: input.before, after: 'def', forced: input.forced}};
const response = async () => {
  if (input.error) throw new Error('API unavailable');
  return input.files;
};
const github = {paginate: response, rest: {pulls: {listFiles: {}}, repos: {
  compareCommitsWithBasehead: async () => ({data: {files: await response()}})
}}};
const core = {setOutput: (key, value) => process.stdout.write(value)};
(async () => { SCRIPT })().catch(() => {process.exitCode = 1;});
""".replace("SCRIPT", script)
        return subprocess.run(
            ["node", "-e", harness],
            input=json.dumps(dict(files=files, event=event, before=before, error=error, forced=forced)),
            text=True, capture_output=True,
        )

    def test_build_selection(self):
        ignored = ["Build/spack/check_pins.py", "Build/spack/spack_repo/erf/packages/erf/package.py",
                   "paper/figure.png", "Docs/sphinx_doc/conf.py", "README.md", "license.txt",
                   ".github/workflows/cuda-ci.yml"]
        required = ["Source/ERF.cpp", "CMakeLists.txt", "Tests/Unit/test.cpp",
                    ".github/workflows/gcc.yml", ".github/actions/ccache-trunk/cache.py"]
        for event in ("pull_request", "push"):
            for path in ignored + required:
                with self.subTest(event=event, path=path):
                    result = self.select([path], event)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, str(path in required).lower())
            self.assertEqual(self.select(ignored, event).stdout, "false")
            self.assertEqual(self.select(ignored + required[:1], event).stdout, "true")

    def test_renamed_source_still_builds(self):
        result = self.select([{"filename": "paper/old.cpp", "previous_filename": "Source/old.cpp"}])
        self.assertEqual(result.stdout, "true")

    def test_conservative_fallbacks_and_api_failure(self):
        self.assertEqual(self.select([]).stdout, "true")
        self.assertEqual(self.select(["README.md"], "workflow_dispatch").stdout, "true")
        self.assertEqual(self.select(["README.md"], "push", before="000000").stdout, "true")
        self.assertEqual(self.select(["README.md"], "push", forced=True).stdout, "true")
        for event, limit in (("push", 300), ("pull_request", 3000)):
            self.assertEqual(self.select(["README.md"] * limit, event).stdout, "true")
        result = self.select([], error=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_required_check_and_job_wiring(self):
        for event in ("push", "pull_request"):
            self.assertNotIn("paths", WORKFLOW["on"][event])
            self.assertNotIn("paths-ignore", WORKFLOW["on"][event])
        jobs = WORKFLOW["jobs"]
        self.assertEqual(jobs["library"]["needs"], "changes")
        self.assertIn("needs.changes.outputs.build == 'true'", jobs["library"]["if"])
        self.assertIn("needs.changes.outputs.build == 'true'", jobs["gnumake"]["if"])
        gate = jobs["ci-required"]
        self.assertEqual(gate["needs"], ["changes", "library"])
        self.assertEqual(gate["if"], "${{ !cancelled() }}")
        script = gate["steps"][0]["run"]
        for changes, build, matrix, passes in (
            ("success", "false", "skipped", True),
            ("success", "true", "success", True),
            ("success", "true", "failure", False),
            ("success", "true", "skipped", False),
            ("failure", "", "skipped", False),
            ("success", "", "skipped", False),
        ):
            with self.subTest(changes=changes, build=build, matrix=matrix):
                result = subprocess.run(["bash", "-c", script], capture_output=True,
                                        env={"CHANGES_RESULT": changes, "BUILD": build,
                                             "MATRIX_RESULT": matrix})
                self.assertEqual(result.returncode == 0, passes)


if __name__ == "__main__":
    unittest.main()
