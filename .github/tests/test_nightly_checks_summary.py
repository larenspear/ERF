"""Test the nightly checks summary on small synthetic logs; never runs a build."""

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "summary", ROOT / ".github/scripts/nightly_checks_summary.py"
)
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)

SRC = "/home/runner/work/ERF/ERF"

JUNIT = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="Linux" tests="3" failures="1" disabled="0" skipped="1">
  <testcase name="Bubble" classname="Bubble" time="2.0" status="run">
    <system-out>step 1
Unused ParmParse Variables:
  [TOP]::erf.no_such_option(nvals = 1)  :: [1]
  [TOP]::erf.typo_cfl(nvals = 1)  :: [0.5]
</system-out>
  </testcase>
  <testcase name="Squall" classname="Squall" time="3.0" status="fail">
    <failure message="Failed"/>
    <system-out>step 1
Erroneous arithmetic operation
See Backtrace.0 file for details
</system-out>
  </testcase>
  <testcase name="Off" classname="Off" time="0" status="notrun">
    <skipped message="Disabled"/>
  </testcase>
</testsuite>
"""

ASAN_LOG = f"""=================================================================
==123==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602 at pc 0x55 bp 0x7f sp 0x7f
READ of size 8 at 0x602 thread T0
    #0 0x55aa in operator[] /usr/include/c++/11/bits/stl_vector.h:1046
    #1 0x55ab in init_zphys {SRC}/Source/Utils/ERF_TerrainMetrics.cpp:704:31
    #2 0x55ac in main {SRC}/Exec/main.cpp:12
"""

UBSAN_LOG = (
    f"{SRC}/Source/ERF_Advance.cpp:360:12: runtime error: reference binding to null pointer of type 'MultiFab'\n"
    f"{SRC}/Source/ERF_Advance.cpp:360:12: runtime error: reference binding to null pointer of type 'MultiFab'\n"
    f"{SRC}/Submodules/AMReX/Src/Base/AMReX_Box.H:10:3: runtime error: signed integer overflow: 2147483647 + 1\n"
)

BUILD_LOG = (
    f"{SRC}/Source/ERF.cpp:12:5: warning: unused variable 'x' [-Wunused-variable]\n"
    f"{SRC}/Source/ERF.cpp:12:5: warning: unused variable 'x' [-Wunused-variable]\n"
    f"{SRC}/Submodules/AMReX/Src/Base/AMReX.cpp:1:1: warning: ignored [-Wfoo]\n"
    "/usr/include/x.h:1:1: warning: system [-Wbar]\n"
)


class SummaryTest(unittest.TestCase):
    def run_summary(self, extra=(), **files):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            logs = tmp / "sanitizer-logs"
            logs.mkdir()
            for name, text in files.get("sanitizer", {}).items():
                (logs / name).write_text(text)
            (tmp / "build.txt").write_text(files.get("build", ""))
            (tmp / "junit.xml").write_text(files.get("junit", JUNIT))
            tests = tmp / "Tests" / "Restart"
            tests.mkdir(parents=True)
            (tests / "Restart.simulation.log").write_text("  [TOP]::erf.no_such_option(nvals = 1)  :: [1]\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = summary.main([
                    "--variant", "sanitizers", "--build-log", str(tmp / "build.txt"),
                    "--junit", str(tmp / "junit.xml"), "--sanitizer-logs", str(logs),
                    "--test-dir", str(tmp / "Tests"), "--source-root", SRC,
                    "--exit-code", "ctest=8", *extra,
                ])
            self.assertEqual(code, 0, "findings must never change the exit status")
            return out.getvalue()

    def test_counts_and_failure_reason(self):
        text = self.run_summary()
        self.assertIn("| Tests failed / run | 1 / 2 (1 skipped) |", text)
        self.assertIn("| Squall | floating point: Erroneous arithmetic operation |", text)
        self.assertIn("CTest exit codes: ctest 8", text)

    def test_unused_inputs_from_output_and_log_files(self):
        text = self.run_summary()
        self.assertIn("| erf.no_such_option | 2 | Bubble |", text)
        self.assertIn("| erf.typo_cfl | 1 | Bubble |", text)

    def test_sanitizer_findings_are_grouped_by_project_location(self):
        text = self.run_summary(sanitizer={"asan.1": ASAN_LOG, "ubsan.2": UBSAN_LOG})
        self.assertIn("| Source/Utils/ERF_TerrainMetrics.cpp:704 | ASan: heap-buffer-overflow | 1 |", text)
        self.assertIn("| Source/ERF_Advance.cpp:360 | UB: reference binding to null pointer of type 'MultiFab' | 2 |", text)
        self.assertIn("Submodules/AMReX/Src/Base/AMReX_Box.H:10", text)
        self.assertIn("| Sanitizer findings (unique locations) | 3 |", text)

    def test_warnings_only_from_erf_sources(self):
        text = self.run_summary(build=BUILD_LOG)
        self.assertIn("| Compiler warnings in ERF sources (unique) | 1 |", text)
        self.assertIn("| Source/ERF.cpp:12:5 | -Wunused-variable |", text)
        self.assertNotIn("AMReX.cpp", text)

    def test_missing_reports_are_reported_not_fatal(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = summary.main(["--variant", "x", "--build-log", "/nonexistent",
                                 "--junit", "/nonexistent.xml"])
        self.assertEqual(code, 0)
        self.assertIn("No test results were found", out.getvalue())

    def test_tests_cut_off_by_stop_time_are_counted(self):
        text = self.run_summary(extra=["--expected-tests", "10"])
        self.assertIn("| Tests not reached before the time limit | 7 |", text)
        self.assertNotIn("not reached", self.run_summary(extra=["--expected-tests", "3"]))

    def test_table_cells_escape_pipes(self):
        rows = summary.table(["a"], [["x|y"]])
        self.assertIn("x\\|y", rows[-1])


if __name__ == "__main__":
    unittest.main()
