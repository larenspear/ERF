#!/usr/bin/env python3
"""Summarize one Nightly checks variant as Markdown for the GitHub job summary.

Reads only local files: the build log, CTest JUnit reports, sanitizer log files
(ASAN_OPTIONS/UBSAN_OPTIONS log_path) and the test working directories. Report only:
findings never change the exit status.
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

# GitHub rejects step summaries over 1 MiB; leave room for the other steps.
SUMMARY_LIMIT = 600_000
TABLE_ROWS = 50

WARNING = re.compile(r"^(?P<loc>[^\s:][^:]*:\d+(?::\d+)?): warning: (?P<msg>.*?)(?: \[(?P<flag>-W[^\]]+)\])?$")
UBSAN = re.compile(r"^(?P<loc>[^\s:][^:]*:\d+):\d+: runtime error: (?P<msg>.*)$")
ASAN = re.compile(r"ERROR: AddressSanitizer: (?P<kind>[\w-]+)")
FRAME = re.compile(r"^\s*#\d+ 0x[0-9a-f]+ in (?P<func>.+?) (?P<loc>/\S+?:\d+)(?::\d+)?$")
# AMReX ParmParse::Finalize prints "  [TOP]::erf.cfl(nvals = 1)  :: [0.5]".
UNUSED = re.compile(r"^\s*\[TOP\]::(?P<key>[^\s(]+)\(nvals = \d+\)")

# First line of a failed test's output that says why, in priority order.
REASONS = [
    ("sanitizer", re.compile(r"ERROR: AddressSanitizer: .*|runtime error: .*")),
    ("bounds", re.compile(r".*(?:out of bound|bound check).*", re.IGNORECASE)),
    ("floating point", re.compile(r".*(?:SIGFPE|Erroneous arithmetic operation|floating point exception).*", re.IGNORECASE)),
    ("libstdc++ debug", re.compile(r".*(?:Error: attempt to|_GLIBCXX_DEBUG|Debug.*assertion).*")),
    ("assertion", re.compile(r".*(?:Assertion|AMREX_ASSERT).*failed.*", re.IGNORECASE)),
    ("abort", re.compile(r".*(?:amrex::Abort|Abort|SIGSEGV|SIGABRT|Segmentation fault).*")),
    ("timeout", re.compile(r".*Timeout.*", re.IGNORECASE)),
]


@dataclass
class TestResult:
    name: str
    status: str  # passed, failed, skipped
    output: str = ""


@dataclass
class Report:
    variant: str
    warnings: Counter[tuple[str, str]] = field(default_factory=Counter)
    tests: list[TestResult] = field(default_factory=list)
    sanitizer: Counter[tuple[str, str]] = field(default_factory=Counter)
    sanitizer_example: dict[tuple[str, str], str] = field(default_factory=dict)
    unused: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    exit_codes: dict[str, str] = field(default_factory=dict)
    expected_tests: int | None = None


def relative(path: str, roots: list[str]) -> str:
    for root in roots:
        if root and path.startswith(root.rstrip("/") + "/"):
            return path[len(root.rstrip("/")) + 1:]
    return path


def is_project(path: str) -> bool:
    """ERF's own sources: not system headers or submodules."""
    return not path.startswith(("/", "Submodules/", "build/_deps/"))


def read_warnings(build_log: Path, roots: list[str], report: Report) -> None:
    if not build_log.is_file():
        return
    for line in build_log.read_text(errors="replace").splitlines():
        match = WARNING.match(line.strip())
        if not match:
            continue
        loc = relative(match["loc"], roots)
        if is_project(loc):
            report.warnings[(loc, match["flag"] or match["msg"][:80])] += 1


def read_junit(paths: list[Path], report: Report) -> None:
    for path in paths:
        if not path.is_file():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            print(f"warning: unreadable JUnit report {path}: {exc}", file=sys.stderr)
            continue
        for case in root.iter("testcase"):
            status = "passed"
            if case.find("failure") is not None or case.find("error") is not None:
                status = "failed"
            elif case.find("skipped") is not None or case.get("status") == "notrun":
                status = "skipped"
            output = "\n".join(
                (node.text or "") for node in case
                if node.tag in ("system-out", "system-err", "failure", "error"))
            output += "\n".join((node.get("message") or "") for node in case if node.tag == "failure")
            report.tests.append(TestResult(case.get("name", "?"), status, output))


def failure_reason(output: str) -> str:
    lines = output.splitlines()
    for label, pattern in REASONS:
        for line in lines:
            if pattern.search(line):
                return f"{label}: {line.strip()[:160]}"
    tail = [line.strip() for line in lines if line.strip()]
    return tail[-1][:160] if tail else "no output captured"


def first_project_frame(lines: list[str], roots: list[str]) -> str:
    fallback = ""
    for line in lines:
        match = FRAME.match(line)
        if not match:
            continue
        loc = relative(match["loc"], roots)
        if is_project(loc):
            return loc
        if not fallback and "sanitizer" not in loc:
            fallback = loc
    return fallback or "unknown location"


def read_sanitizer_logs(log_dir: Path, roots: list[str], report: Report) -> None:
    if not log_dir.is_dir():
        return
    for path in sorted(log_dir.iterdir()):
        lines = path.read_text(errors="replace").splitlines()
        for index, line in enumerate(lines):
            ubsan = UBSAN.match(line.strip())
            if ubsan:
                message = re.sub(r"0x[0-9a-f]+|-?\d+(?:\.\d+)?(?:e[-+]?\d+)?", "N", ubsan["msg"])
                key = (relative(ubsan["loc"], roots), "UB: " + message[:100])
            else:
                asan = ASAN.search(line)
                if not asan:
                    continue
                block = lines[index:index + 40]
                key = (first_project_frame(block, roots), "ASan: " + asan["kind"])
            report.sanitizer[key] += 1
            report.sanitizer_example.setdefault(key, "\n".join(lines[index:index + 25]))


def read_unused_inputs(report: Report, test_dirs: list[Path]) -> None:
    for test in report.tests:
        for line in test.output.splitlines():
            match = UNUSED.match(line)
            if match:
                report.unused[match["key"]].add(test.name)
    # Regression drivers write the simulation output to their own log files.
    for test_dir in test_dirs:
        if not test_dir.is_dir():
            continue
        for log in test_dir.rglob("*.log"):
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                match = UNUSED.match(line)
                if match:
                    report.unused[match["key"]].add(log.parent.name)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    def cell(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", " ")
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows[:TABLE_ROWS]]
    if len(rows) > TABLE_ROWS:
        out.append(f"\n_{len(rows) - TABLE_ROWS} more not shown._")
    return out


def render(report: Report) -> str:
    counts = Counter(t.status for t in report.tests)
    failed = [t for t in report.tests if t.status == "failed"]
    # CTest's --stop-time leaves tests it never started out of the JUnit report.
    not_reached = 0
    if report.expected_tests is not None and report.tests:
        not_reached = max(report.expected_tests - len(report.tests), 0)
    lines = [f"## Nightly checks: `{report.variant}`", ""]
    lines += table(["Finding", "Count"], [
        ["Tests failed / run", f"{counts['failed']} / {counts['failed'] + counts['passed']}"
         + (f" ({counts['skipped']} skipped)" if counts["skipped"] else "")],
        *([["Tests not reached before the time limit", str(not_reached)]] if not_reached else []),
        ["Sanitizer findings (unique locations)", str(len(report.sanitizer))],
        ["Unused input parameters (unique keys)", str(len(report.unused))],
        ["Compiler warnings in ERF sources (unique)", str(len(report.warnings))],
    ])
    if report.exit_codes:
        lines += ["", "CTest exit codes: " + ", ".join(
            f"{k} {v}" for k, v in sorted(report.exit_codes.items()))]
    lines.append("")
    if report.tests == []:
        lines += ["**No test results were found.** The build or the test step may have "
                  "failed before CTest wrote its reports.", ""]
    if failed:
        lines += ["### Failed tests", ""]
        lines += table(["Test", "Reason (first matching line)"],
                       [[t.name, failure_reason(t.output)] for t in failed])
        lines.append("")
    if report.sanitizer:
        lines += ["### Sanitizer findings", ""]
        ordered = report.sanitizer.most_common()
        lines += table(["Location", "Kind", "Reports"],
                       [[loc, kind, str(n)] for (loc, kind), n in ordered])
        lines += ["", "<details><summary>First report for each location</summary>", ""]
        for key, _ in ordered[:TABLE_ROWS]:
            lines += [f"**{key[0]}** ({key[1]})", "", "```", report.sanitizer_example[key], "```", ""]
        lines += ["</details>", ""]
    if report.unused:
        lines += ["### Unused input parameters", "",
                  "Set in an inputs file but never read: a typo, or a parameter that no longer does anything.", ""]
        rows = sorted(report.unused.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        lines += table(["Parameter", "Tests", "Example"],
                       [[k, str(len(v)), sorted(v)[0]] for k, v in rows])
        lines.append("")
    if report.warnings:
        lines += ["### Compiler warnings in ERF sources", ""]
        warnings = sorted(report.warnings)
        lines += table(["Location", "Warning"], [[loc, flag] for loc, flag in warnings])
        lines.append("")
    text = "\n".join(lines)
    if len(text) > SUMMARY_LIMIT:
        text = text[:SUMMARY_LIMIT] + "\n\n_Summary truncated at the job-summary size limit._\n"
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--build-log", type=Path, required=True)
    parser.add_argument("--junit", type=Path, nargs="*", default=[])
    parser.add_argument("--sanitizer-logs", type=Path)
    parser.add_argument("--test-dir", type=Path, action="append", default=[])
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--expected-tests", type=int,
                        help="tests CTest selected; the difference from the report is not reached")
    parser.add_argument("--exit-code", action="append", default=[],
                        help="label=code for each CTest invocation")
    args = parser.parse_args(argv)

    report = Report(args.variant, expected_tests=args.expected_tests)
    roots = [str(Path(r).resolve()) for r in args.source_root] + args.source_root
    read_warnings(args.build_log, roots, report)
    read_junit(args.junit, report)
    if args.sanitizer_logs:
        read_sanitizer_logs(args.sanitizer_logs, roots, report)
    read_unused_inputs(report, args.test_dir)
    for item in args.exit_code:
        label, _, code = item.partition("=")
        report.exit_codes[label] = code
    print(render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
