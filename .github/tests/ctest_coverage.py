#!/usr/bin/env python3
"""Fail when a registered CTest test runs in no leg of a matrix job that registers it.

Each leg saves `ctest --show-only=json-v1` as registry.json and `toJSON(matrix)` as
matrix.json. A test passes when one of the job's ctest steps selects it in a leg that
registers it; the step's `if:` is evaluated against that leg's matrix. Tests selected only
by another workflow's ctest steps are listed but not failed: that match ignores the other
workflow's conditions and build options.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import yaml  # type: ignore[import-untyped]

SELECTOR_OPTIONS = {
    "-L": "labels", "--label-regex": "labels",
    "-LE": "exclude_labels", "--label-exclude": "exclude_labels",
    "-R": "names", "--tests-regex": "names",
    "-E": "exclude_names", "--exclude-regex": "exclude_names",
}
# Selection options this model does not implement; a call using one fails the check.
UNSUPPORTED = re.compile(
    r"^(-U|--union|-I|--tests-information|--rerun-failed|--tests-from-file|"
    r"--exclude-from-file|-FA|-FS|-FC|--fixture-exclude-.*|--label-.*=.*|--tests-regex=.*|"
    r"--exclude-regex=.*|--label-regex=.*)$")
ALWAYS_TRUE = {"success()", "always()", "!cancelled()"}
MATRIX_TERM = re.compile(r"matrix\.(\w+)\s*(==|!=)\s*'([^']*)'")
CTEST_CALL = re.compile(r"(?:^|[;&|(]\s*)ctest\s+(.*)")


@dataclass
class Selector:
    """One `ctest` invocation and the step it runs in."""

    workflow: str
    job: str
    step: str
    condition: str
    labels: List[str] = field(default_factory=list)
    exclude_labels: List[str] = field(default_factory=list)
    names: List[str] = field(default_factory=list)
    exclude_names: List[str] = field(default_factory=list)

    def selects(self, name: str, labels: Sequence[str]) -> bool:
        """CTest semantics: a label regex matches if it matches any label, and repeated
        -L (or -LE) options must all match."""
        def matches_a_label(pattern: str) -> bool:
            return any(re.search(pattern, label) for label in labels)
        if not all(matches_a_label(p) for p in self.labels):
            return False
        if self.exclude_labels and all(matches_a_label(p) for p in self.exclude_labels):
            return False
        if self.names and not re.search(self.names[0], name):
            return False
        return not (self.exclude_names and re.search(self.exclude_names[0], name))


@dataclass
class Leg:
    source: str
    matrix: Dict[str, str]
    tests: Dict[str, List[str]]


def ctest_arguments(script: str) -> Iterator[str]:
    """Argument text of each `ctest` call in a run: block."""
    for line in script.replace("\\\n", " ").splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        match = CTEST_CALL.search(stripped)
        if match:
            yield match.group(1)


def parse_selector(arguments: str, workflow: str, job: str, step: str,
                   condition: str) -> Optional[Selector]:
    """Selector for one ctest call; None for calls that only list tests."""
    where = f"{workflow} / {job} / {step}"
    try:
        tokens = shlex.split(arguments)
    except ValueError:
        tokens = arguments.split()
    if any(t in ("-N", "--show-only") or t.startswith("--show-only=") for t in tokens):
        return None
    selector = Selector(workflow, job, step, condition)
    for option, value in zip(tokens, tokens[1:] + [""]):
        if UNSUPPORTED.match(option):
            raise ValueError(f"{where}: ctest option {option} is not supported by this check")
        attribute = SELECTOR_OPTIONS.get(option)
        if attribute is not None:
            if not value:
                raise ValueError(f"{where}: {option} has no value")
            getattr(selector, attribute).append(value)
    if len(selector.names) > 1 or len(selector.exclude_names) > 1:
        raise ValueError(f"{where}: repeated -R or -E is not supported by this check")
    return selector


def workflow_selectors(path: Path) -> List[Selector]:
    """Every test-running ctest call in a workflow file."""
    document = yaml.safe_load(path.read_text())
    selectors: List[Selector] = []
    for job_id, job in (document.get("jobs") or {}).items():
        for index, step in enumerate(job.get("steps") or []):
            script = step.get("run")
            if not isinstance(script, str):
                continue
            name = str(step.get("name") or f"step {index + 1}")
            for arguments in ctest_arguments(script):
                selector = parse_selector(arguments, path.name, job_id, name,
                                          str(step.get("if") or ""))
                if selector is not None:
                    selectors.append(selector)
    return selectors


def condition_holds(condition: str, matrix: Dict[str, str]) -> bool:
    """Evaluate `matrix.<axis> ==/!= '<value>'` terms joined by &&; other forms raise."""
    text = condition.strip()
    if text.startswith("${{") and text.endswith("}}"):
        text = text[3:-2].strip()
    if not text:
        return True
    result = True
    for term in (t.strip() for t in text.split("&&")):
        if term in ALWAYS_TRUE:
            continue
        match = MATRIX_TERM.fullmatch(term)
        if match is None:
            raise ValueError(f"unsupported step condition {condition!r} (term {term!r})")
        axis, operator, value = match.groups()
        if axis not in matrix:
            raise ValueError(f"step condition {condition!r} uses unknown matrix axis {axis!r}")
        equal = matrix[axis] == value
        result = result and (equal if operator == "==" else not equal)
    return result


def registered_tests(registry: Dict[str, Any]) -> Dict[str, List[str]]:
    """Test name -> labels from `ctest --show-only=json-v1` output."""
    tests: Dict[str, List[str]] = {}
    for test in registry.get("tests", []):
        labels: List[str] = []
        for prop in test.get("properties", []):
            if prop.get("name") == "LABELS":
                labels = [str(label) for label in prop.get("value", [])]
        tests[str(test["name"])] = labels
    return tests


def load_legs(directory: Path) -> List[Leg]:
    legs = []
    for registry in sorted(directory.glob("*/registry.json")):
        matrix = json.loads((registry.parent / "matrix.json").read_text())
        legs.append(Leg(registry.parent.name,
                        {str(k): str(v) for k, v in matrix.items()},
                        registered_tests(json.loads(registry.read_text()))))
    if not legs:
        raise SystemExit(f"no */registry.json under {directory}")
    return legs


def coverage(legs: Sequence[Leg], own: Sequence[Selector], others: Sequence[Selector]
             ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, Set[str]]]:
    """(not run anywhere: labels, run only by another workflow: steps, registering legs)."""
    registered_in: Dict[str, Set[str]] = {}
    labels_of: Dict[str, List[str]] = {}
    covered: Set[str] = set()
    for leg in legs:
        active = [s for s in own if condition_holds(s.condition, leg.matrix)]
        for name, labels in leg.tests.items():
            registered_in.setdefault(name, set()).add(leg.source)
            labels_of.setdefault(name, labels)
            if any(s.selects(name, labels) for s in active):
                covered.add(name)
    uncovered: Dict[str, List[str]] = {}
    elsewhere: Dict[str, List[str]] = {}
    for name in sorted(set(registered_in) - covered):
        steps = [f"{s.workflow} / {s.step}" for s in others if s.selects(name, labels_of[name])]
        if steps:
            elsewhere[name] = steps
        else:
            uncovered[name] = labels_of[name]
    return uncovered, elsewhere, registered_in


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--registries", type=Path, required=True)
    args = parser.parse_args(argv)

    own = [s for s in workflow_selectors(args.workflow) if s.job == args.job]
    if not own:
        raise SystemExit(f"no ctest steps found in {args.workflow} job {args.job}")
    others = [s for path in sorted(args.workflow.parent.glob("*.yml"))
              if path.resolve() != args.workflow.resolve()
              for s in workflow_selectors(path)]
    legs = load_legs(args.registries)
    uncovered, elsewhere, registered_in = coverage(legs, own, others)

    lines = [f"{len(registered_in)} registered tests across {len(legs)} legs of "
             f"{args.workflow.name} job {args.job}; {len(uncovered)} run nowhere."]
    for name, steps in elsewhere.items():
        lines.append(f"run only by another workflow: {name} ({'; '.join(steps)})")
    for name, labels in uncovered.items():
        lines.append(f"NOT RUN: {name} labels={';'.join(labels) or '(none)'} "
                     f"registered in {', '.join(sorted(registered_in[name]))}")
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as handle:
            handle.write("### CTest coverage\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
    if uncovered:
        print(f"::error title=Registered tests that CI never runs::{', '.join(uncovered)}. "
              f"Add or widen a ctest step in {args.workflow.name} that selects each of them "
              "in a leg that registers it.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
