#!/usr/bin/env python3
"""Verify surface checkpoints after AMR removal, restart, and level recreation.

The CPU double-precision fixture starts with a fine patch, removes it for two
coarse steps, then recreates it. Checkpoint headers and fields must follow that
active hierarchy. Restarting while the fine level is absent must reproduce the
continuation through recreation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
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
    work = Path(tempfile.mkdtemp(prefix="surface-derefinement-", dir=args.working_directory.resolve()))
    executable = args.executable.resolve()
    tool = args.checkpoint_tool.resolve()
    report = {"work_directory": str(work), "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
              "hierarchies": {}, "comparisons": [], "passed": False}

    def invoke(command, directory, label):
        command = [str(arg) for arg in command]
        (directory / (label + ".command.json")).write_text(json.dumps(command, indent=2) + "\n")
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=90)
        output = result.stdout + result.stderr
        (directory / (label + ".log")).write_text(output)
        if result.returncode:
            raise RuntimeError(f"{label}: exit {result.returncode}; see {directory / (label + '.log')}")
        return output

    def prepare(name):
        directory = work / name
        directory.mkdir()
        shutil.copyfile(args.input_dir / "SLM_Restart.i", directory / "inputs")
        shutil.copyfile(args.input_dir / "input_sounding", directory / "input_sounding")
        return directory

    def check_hierarchy(directory, step, nlevels):
        checkpoint = directory / f"chk{step:05d}"
        lines = (checkpoint / "SurfaceModel_Header").read_text().splitlines()
        if lines[0] != "Checkpoint file for SurfaceModel" or int(lines[1]) != nlevels:
            raise RuntimeError(f"{checkpoint}: surface header has the wrong active level count")
        actual = {path.name for path in checkpoint.glob("Level_*") if path.is_dir()}
        expected = {f"Level_{lev}" for lev in range(nlevels)}
        if actual != expected:
            raise RuntimeError(f"{checkpoint}: checkpoint level directories {actual} != {expected}")
        for lev in range(nlevels):
            for field in ("ustar", "wavg", "tstar", "qstar", "tsurf", "_f_olen"):
                path = checkpoint / f"Level_{lev}" / f"SurfaceModel_{field}_H"
                if not path.is_file():
                    raise RuntimeError(f"missing active surface field {path}")
        report["hierarchies"][f"{directory.name}/{step}"] = nlevels

    common = [executable, "inputs", "max_step=6", "erf.check_int=1", "erf.check_file=chk",
              "erf.plot_int_1=-1", "erf.plot_lsm=false", "erf.fixed_dt=0.01",
              "amrex.throw_exception=1", "amrex.signal_handling=0", "amrex.fpe_trap_invalid=0",
              "amr.max_grid_size=4", "amr.max_grid_size_z=16", "amr.blocking_factor=2",
              "amr.n_error_buf=0", "amr.max_level=1", "amr.ref_ratio_vect=2 2 1",
              "erf.regrid_int=1", "erf.refinement_indicators=old_patch new_patch",
              "erf.old_patch.max_level=1", "erf.old_patch.in_box_lo_indices_crse=0 0 0",
              "erf.old_patch.in_box_hi_indices_crse=3 7 15", "erf.old_patch.end_time=0.015",
              "erf.new_patch.max_level=1", "erf.new_patch.in_box_lo_indices_crse=0 0 0",
              "erf.new_patch.in_box_hi_indices_crse=3 7 15", "erf.new_patch.start_time=0.035"]
    try:
        direct = prepare("direct")
        log = invoke(common, direct, "erf")
        levels = {int(step): int("[Level 1 step " in part) for step, part in re.findall(
            r"Coarse STEP (\d+) starts \.\.\.(.*?)(?=Coarse STEP \d+ starts|\Z)", log, re.S)}
        if levels != {1: 1, 2: 1, 3: 0, 4: 0, 5: 1, 6: 1}:
            raise RuntimeError(f"fixture did not remove and recreate level 1: {levels}")
        for step in range(7):
            check_hierarchy(direct, step, 1 if step in (3, 4) else 2)

        restarted = prepare("restarted")
        log = invoke(common + ["erf.restart=" + str(direct / "chk00003")], restarted, "erf")
        if "NEW BA FROM COARSE AT LEVEL 1" not in log or "Coarse STEP 6 ends" not in log:
            raise RuntimeError("restart did not recreate the fine level and complete")
        for step in (4, 5, 6):
            check_hierarchy(restarted, step, 1 if step == 4 else 2)

        # Compare exposed fields, private SLM state, atmosphere, and the surface
        # interface through both absence and recreation. The tolerance permits
        # accumulated double roundoff across six steps, scaled by each field.
        for step in (4, 5, 6):
            for lev in range(1 if step == 4 else 2):
                left = direct / f"chk{step:05d}/Level_{lev}"
                right = restarted / f"chk{step:05d}/Level_{lev}"
                names = {"Cell", "RainAccum", "SlmPrecip"}
                for pattern in ("SLM*_H", "LsmData*_H", "SurfaceModel*_H"):
                    names.update(path.name[:-2] for path in left.glob(pattern))
                for name in sorted(names):
                    label = f"compare-{step}-{lev}-{name}"
                    output = invoke([tool, "compare", left / name, right / name, 0], restarted, label)
                    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
                    if not records:
                        raise RuntimeError(f"{label}: no numerical comparison records")
                    for record in records:
                        error = record["max_abs_diff"]
                        tolerance = 4096 * sys.float_info.epsilon * max(1, record["left_norm"])
                        if not math.isfinite(error) or error > tolerance:
                            raise RuntimeError(f"{label}: {error} exceeds {tolerance}")
                    report["comparisons"].append({"step": step, "level": lev, "field": name, "records": records})
        report["passed"] = True
    except (OSError, RuntimeError, ValueError, IndexError, subprocess.TimeoutExpired) as error:
        report["error"] = str(error)
    (work / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "work_directory": str(work),
                      "comparisons": len(report["comparisons"]), "error": report.get("error")}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
