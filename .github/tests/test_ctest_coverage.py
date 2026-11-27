"""Tests for ctest_coverage.py: selection rules, step conditions and leg coverage."""

import json
from pathlib import Path
import sys
import tempfile
import textwrap
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctest_coverage as cc  # noqa: E402

WORKFLOW = textwrap.dedent("""\
    jobs:
      library:
        steps:
          - name: Record
            run: ctest --show-only=json-v1 > registry.json
          - name: Unit
            if: matrix.precision == 'DOUBLE'
            run: |
              ctest -L unit -j 4 -LE parallel \\
                    --output-on-failure
          - name: Single
            if: ${{ matrix.precision == 'SINGLE' && matrix.particles != 'OFF' }}
            run: ctest -R '^(Foo|Bar)\\.' --no-tests=error
          - name: Comment only
            run: |
              # ctest -L forgotten
              echo done
    """)
OTHER = textwrap.dedent("""\
    jobs:
      other:
        steps:
          - name: Stress
            run: ctest -L stress --output-on-failure
    """)


def registry(*tests: tuple) -> dict:
    return {"tests": [{"name": name, "properties": [{"name": "LABELS", "value": labels}]}
                      for name, labels in tests]}


class SelectorTest(unittest.TestCase):
    def test_label_regex_matches_any_label_as_substring(self) -> None:
        selector = cc.Selector("w", "j", "s", "", labels=["parallel"])
        self.assertTrue(selector.selects("x", ["regression", "parallel"]))
        self.assertTrue(selector.selects("x", ["integration;parallel-io"]))
        self.assertFalse(selector.selects("x", ["regression"]))

    def test_exclusions_and_name_regex(self) -> None:
        selector = cc.Selector("w", "j", "s", "", labels=["unit"], exclude_labels=["parallel"],
                               names=[r"^SBM"], exclude_names=["Slow$"])
        self.assertTrue(selector.selects("SBMFast", ["unit"]))
        self.assertFalse(selector.selects("SBMFast", ["unit", "parallel"]))
        self.assertFalse(selector.selects("SBMSlow", ["unit"]))
        self.assertFalse(selector.selects("Other", ["unit"]))

    def test_repeated_label_options_must_all_match(self) -> None:
        both = cc.Selector("w", "j", "s", "", labels=["unit", "sbm"])
        self.assertTrue(both.selects("x", ["unit", "sbm"]))
        self.assertFalse(both.selects("x", ["unit"]))
        # Repeated -LE excludes only tests matching every pattern.
        exclude_both = cc.Selector("w", "j", "s", "", exclude_labels=["slow", "gpu"])
        self.assertTrue(exclude_both.selects("x", ["slow"]))
        self.assertFalse(exclude_both.selects("x", ["slow", "gpu"]))

    def test_no_selector_options_selects_everything(self) -> None:
        self.assertTrue(cc.Selector("w", "j", "s", "").selects("anything", []))


class WorkflowFixture(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        (self.directory / "gcc.yml").write_text(WORKFLOW)
        (self.directory / "other.yml").write_text(OTHER)


class ParseTest(WorkflowFixture):

    def test_reads_continuations_and_skips_listing_and_comments(self) -> None:
        selectors = cc.workflow_selectors(self.directory / "gcc.yml")
        self.assertEqual([s.step for s in selectors], ["Unit", "Single"])
        unit, single = selectors
        self.assertEqual((unit.labels, unit.exclude_labels), (["unit"], ["parallel"]))
        self.assertEqual(single.names, [r"^(Foo|Bar)\."])

    def test_missing_option_value_fails(self) -> None:
        with self.assertRaises(ValueError):
            cc.parse_selector("-L", "w", "j", "s", "")

    def test_unmodelled_selection_options_fail(self) -> None:
        for arguments in ("-U -R a -L b", "-I 1,5", "--label-regex=unit", "-R a -R b",
                          "--rerun-failed"):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                cc.parse_selector(arguments, "w", "j", "s", "")
        self.assertIsNotNone(cc.parse_selector(
            "--test-dir build -L unit -j 4 --output-on-failure --no-tests=error",
            "w", "j", "s", ""))

    def test_conditions(self) -> None:
        leg = {"precision": "SINGLE", "particles": "ON"}
        self.assertTrue(cc.condition_holds("", leg))
        self.assertTrue(cc.condition_holds("always() && matrix.precision == 'SINGLE'", leg))
        self.assertTrue(cc.condition_holds("${{ matrix.particles != 'OFF' }}", leg))
        self.assertFalse(cc.condition_holds("matrix.precision == 'DOUBLE'", leg))
        with self.assertRaises(ValueError):
            cc.condition_holds("matrix.precision == 'SINGLE' || failure()", leg)
        with self.assertRaises(ValueError):
            cc.condition_holds("matrix.compiler == 'gcc'", leg)


class CoverageTest(WorkflowFixture):
    def write_leg(self, name: str, matrix: dict, tests: dict) -> None:
        leg = self.directory / "registries" / name
        leg.mkdir(parents=True)
        (leg / "matrix.json").write_text(json.dumps(matrix))
        (leg / "registry.json").write_text(json.dumps(registry(*tests.items())))

    def run_main(self) -> int:
        return cc.main(["--workflow", str(self.directory / "gcc.yml"), "--job", "library",
                        "--registries", str(self.directory / "registries")])

    def test_test_registered_only_where_no_step_selects_it_fails(self) -> None:
        # unit runs only in DOUBLE; SingleOnly is registered only in SINGLE.
        self.write_leg("double", {"precision": "DOUBLE", "particles": "ON"},
                       {"UnitA": ["unit"]})
        self.write_leg("single", {"precision": "SINGLE", "particles": "ON"},
                       {"UnitA": ["unit"], "SingleOnly": ["unit"], "Foo.case": ["unit"]})
        legs = cc.load_legs(self.directory / "registries")
        own = cc.workflow_selectors(self.directory / "gcc.yml")
        others = cc.workflow_selectors(self.directory / "other.yml")
        uncovered, elsewhere, _ = cc.coverage(legs, own, others)
        self.assertEqual(list(uncovered), ["SingleOnly"])
        self.assertEqual(elsewhere, {})
        self.assertEqual(self.run_main(), 1)

    def test_other_workflow_coverage_is_reported_not_failed(self) -> None:
        self.write_leg("double", {"precision": "DOUBLE", "particles": "ON"},
                       {"UnitA": ["unit"], "Shuffle": ["stress"]})
        legs = cc.load_legs(self.directory / "registries")
        uncovered, elsewhere, _ = cc.coverage(
            legs, cc.workflow_selectors(self.directory / "gcc.yml"),
            cc.workflow_selectors(self.directory / "other.yml"))
        self.assertEqual(uncovered, {})
        self.assertEqual(elsewhere, {"Shuffle": ["other.yml / Stress"]})
        self.assertEqual(self.run_main(), 0)

    def test_no_registries_fails(self) -> None:
        (self.directory / "registries").mkdir()
        with self.assertRaises(SystemExit):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
