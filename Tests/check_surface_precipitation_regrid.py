#!/usr/bin/env python3
"""Check cumulative-rain history through actual new-level creation and remakes.

CPU double-precision fixture. Checkpoint edits are confined to this test's
copies and seed liquid rain plus a consistent native/SLM history pair.
"""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--checkpoint-tool", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--working-directory", type=Path, required=True)
    args = parser.parse_args()
    args.working_directory.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="precip-regrid-", dir=args.working_directory.resolve()))
    exe = args.executable.resolve()
    tool = args.checkpoint_tool.resolve()
    report = {"work_directory": str(work), "runs": [], "checks": []}
    sequence = 0

    def execute(command, cwd, label):
        nonlocal sequence
        sequence += 1
        stem = work / f"{sequence:04d}-{label}"
        command = [str(a) for a in command]
        stem.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
        run = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=90)
        output = run.stdout + run.stderr
        stem.with_suffix(".log").write_text(output)
        if run.returncode:
            raise AssertionError(f"{label}: exit {run.returncode}; see {stem}.log")
        return output, [json.loads(line) for line in output.splitlines() if line.startswith("{")]

    def create_case(name):
        folder = work / name
        folder.mkdir()
        shutil.copyfile(args.input_dir / "SLM_Restart.i", folder / "inputs")
        shutil.copyfile(args.input_dir / "input_sounding", folder / "input_sounding")
        return folder

    common = [exe, "inputs", "erf.check_int=1", "erf.check_file=chk",
              "erf.plot_int_1=-1", "erf.plot_lsm=false", "erf.fixed_dt=0.01",
              "amrex.throw_exception=1", "amrex.signal_handling=0",
              "amrex.fpe_trap_invalid=0", "amr.max_grid_size=4",
              "amr.max_grid_size_z=16", "amr.blocking_factor=2"]
    try:
        initial = create_case("initial")
        execute(common + ["max_step=0"], initial, "initial")
        for history in [0, 12]:
            seed = work / f"seed-{history}"
            shutil.copytree(initial / "chk00000", seed)
            execute([tool, "seed", seed, history, "0.001"], work, f"seed-{history}")
            for scenario in ["newlevel", "moving"]:
                case = create_case(f"{scenario}-{history}")
                command = common + ["max_step=5", f"erf.restart={seed}",
                    "amr.n_error_buf=0", "amr.max_level=1", "amr.ref_ratio_vect=2 2 1",
                    "erf.regrid_int=1"]
                if scenario == "newlevel":
                    command += ["erf.refinement_indicators=patch", "erf.patch.max_level=1",
                        "erf.patch.in_box_lo_indices_crse=0 0 0",
                        "erf.patch.in_box_hi_indices_crse=3 7 15", "erf.patch.start_time=0.025"]
                else:
                    command += ["erf.refinement_indicators=old_patch new_patch",
                        "erf.old_patch.max_level=1", "erf.old_patch.in_box_lo_indices_crse=0 0 0",
                        "erf.old_patch.in_box_hi_indices_crse=3 7 15", "erf.old_patch.start_time=0.005",
                        "erf.old_patch.end_time=0.025", "erf.new_patch.max_level=1",
                        "erf.new_patch.in_box_lo_indices_crse=2 0 0",
                        "erf.new_patch.in_box_hi_indices_crse=5 7 15", "erf.new_patch.start_time=0.025"]
                output, _ = execute(command, case, f"{scenario}-{history}")
                assert "NEW BA FROM COARSE AT LEVEL 1" in output, "new level was not created"
                assert "Coarse STEP 5 ends" in output, "run did not complete five steps"
                if scenario == "moving":
                    assert "REMAKING WITH NEW BA AT LEVEL 1" in output, "patch did not move"
                report["runs"].append({"scenario": scenario, "history": history, "passed": True})

        # An additive constant in both cumulative fields cannot affect interval
        # rain. The native/baseline offset tolerance permits tens of double
        # rounding errors, and the physical-field tolerance permits accumulated
        # roundoff through five coupled steps. Neither permits a missing interval.
        epsilon = sys.float_info.epsilon
        private_fields = ["SLM_mw", "SLM_mws", "SLM_t_canop", "SLM_t_skin",
                          "SLM_t_ground_skin", "SLM_t_cas", "SLM_q_cas"]
        for scenario in ["newlevel", "moving"]:
            for step in range(1, 6):
                for level in [0, 1]:
                    left = work / f"{scenario}-0" / f"chk{step:05d}" / f"Level_{level}"
                    right = work / f"{scenario}-12" / f"chk{step:05d}" / f"Level_{level}"
                    expect_fine = step >= (4 if scenario == "newlevel" else 2)
                    if level == 1 and not expect_fine:
                        assert not left.exists() and not right.exists(), "fine level appeared early"
                        continue
                    assert left.is_dir() and right.is_dir(), "expected checkpoint level is missing"
                    fields = ["RainAccum", "SlmPrecip"]
                    if step == 5:
                        fields += ["Cell"] + private_fields
                        fields += sorted(p.name[:-2] for p in left.glob("LsmData*_H"))
                    for field in fields:
                        offset = 12 if field in ["RainAccum", "SlmPrecip"] else 0
                        label = f"{scenario}-{step}-{level}-{field}"
                        _, records = execute([tool, "compare", left / field, right / field, offset], work, label)
                        assert records, f"empty comparison: {label}"
                        for record in records:
                            error = record["max_abs_diff"]
                            if offset:
                                tolerance = 64 * epsilon * offset
                            else:
                                tolerance = 4096 * epsilon * max(1, record["left_norm"])
                            assert math.isfinite(error) and error <= tolerance, (label, record, tolerance)
                        report["checks"].append({"case": label, "records": records})
                    for prefix in [left, right]:
                        _, records = execute([tool, "baseline", prefix.parent, level], work,
                                             f"{scenario}-{step}-{level}-{prefix.parent.parent.name}-baseline")
                        assert records and all(r["max_abs_diff"] == 0 and r["surface_cells"] > 0 for r in records), "SLM baseline differs from native rain"
                    _, stats = execute([tool, "stats", left / "SlmPrecip"], work,
                                       f"{scenario}-{step}-{level}-rain")
                    assert stats and math.isfinite(stats[0]["min"]) and stats[0]["min"] > 0, "fixture did not rain"
        report["passed"] = True
    except (AssertionError, subprocess.TimeoutExpired, OSError, ValueError) as error:
        report["passed"] = False
        report["error"] = str(error)
    (work / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "work_directory": str(work),
                      "checks": len(report["checks"]), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
