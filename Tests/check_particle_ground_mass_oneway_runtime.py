#!/usr/bin/env python3
"""CPU-double serial OneWay ground mass through fine removal and recreation.

The initial fine grid covers the whole domain, so a sum on the finest level
is a composite kg total without counting covered coarse cells. A prescribed
fine-only offset must survive both transitions, while generated mass remains
nondecreasing. Ground history must not change atmosphere or particle state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from particle_checkpoint_compare import compare as particle_compare

FIELDS = ["GroundSpecies_water", "GroundSpecies_agua",
          "GroundAerosol_0_NaCl", "GroundAerosol_1_NH42SO4"]
PHYSICAL = ["Cell", "XFace", "YFace", "ZFace", "RainAccum", "SnowAccum"]
OPTIONS = ["amr.max_level=1", "amr.ref_ratio_vect=2 2 2", "amr.max_grid_size=8",
           "amr.max_grid_size_z=16", "amr.n_error_buf=0", "erf.regrid_int=1",
           "erf.coupling_type=OneWay", "erf.cf_width=0", "erf.refinement_indicators=patch restored",
           "erf.patch.max_level=1", "erf.patch.end_time=0.0015",
           "erf.patch.in_box_lo_indices_crse=0 0 0", "erf.patch.in_box_hi_indices_crse=7 7 7",
           "erf.restored.max_level=1", "erf.restored.start_time=0.0065",
           "erf.restored.in_box_lo_indices_crse=0 0 0", "erf.restored.in_box_hi_indices_crse=7 7 7"]
EPS = sys.float_info.epsilon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erf", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).parent /
                        "test_files/SDM_GroundMass_Restart/SDM_GroundMass_Restart.i")
    args = parser.parse_args()
    assert sys.byteorder == "little", "native particle reader currently supports little-endian hosts"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="particle-ground-oneway-", dir=args.work_dir.resolve()))
    exe = root / "erf_exec"
    shutil.copy2(args.erf, exe)
    tool = args.tool.resolve()
    results = {"binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
               "runs": [], "checks": [], "passed": False}

    def save():
        (root / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    def check(name, passed, values):
        results["checks"].append({"name": name, "passed": bool(passed), "values": values})
        if not passed:
            print("FAILED", name, values, flush=True)
        save()

    def run(name, steps, checkpoint=None):
        out = root / name
        out.mkdir()
        shutil.copy2(args.inputs, out / "inputs")
        command = [str(exe), "inputs", *OPTIONS, f"max_step={steps}"]
        if checkpoint:
            shutil.copytree(checkpoint, out / "input-checkpoint")
            command += ["erf.restart=input-checkpoint"]
        (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (out / "run.log").open("w") as log:
            proc = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=90)
        results["runs"].append({"name": name, "returncode": proc.returncode})
        save()
        assert proc.returncode == 0, str(out / "run.log")
        return out

    def field(action, *arguments):
        proc = subprocess.run([str(tool), action, *map(str, arguments)],
                              text=True, capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        rows = [json.loads(line) for line in proc.stdout.splitlines() if line.startswith("{")]
        assert rows, "checkpoint tool returned no records"
        return rows

    def levels(chk):
        return sorted(p.name for p in chk.glob("Level_*") if p.is_dir())

    def total(chk, value):
        rows = field("stats", chk / levels(chk)[-1] / value)
        assert len(rows) == 1
        return rows[0]["sum"]

    def compare(name, left, right, offset):
        rows = field("compare", left, right, offset)
        tolerance = 512 * EPS * abs(offset)
        check(name, all(row["max_abs_diff"] <= tolerance for row in rows),
              {"rows": rows, "offset": offset, "tolerance": tolerance})

    try:
        initial = run("initial", 1) / "chk00001"
        assert levels(initial) == ["Level_0", "Level_1"]
        history = root / "history-checkpoint"
        shutil.copytree(initial, history)
        for n, value in enumerate(FIELDS, 1):
            field("add-bottom-history", history / "Level_1" / value, n)
            compare("prescribed-fine-" + value, initial / "Level_1" / value, history / "Level_1" / value, n)
            compare("unchanged-coarse-" + value, initial / "Level_0" / value, history / "Level_0" / value, 0)
        reference = run("reference", 8, initial)
        modified = run("history", 8, history)
        for step in range(2, 9):
            name = f"chk{step:05d}"
            left, right = reference / name, modified / name
            expected_levels = ["Level_0", "Level_1"] if step in (2, 8) else ["Level_0"]
            check(f"topology-step{step}", levels(left) == levels(right) == expected_levels, levels(left))
            for level in levels(left):
                for value in PHYSICAL:
                    compare(f"physical-step{step}-{level}-{value}", left / level / value, right / level / value, 0)
                for n, value in enumerate(FIELDS, 1):
                    # Four horizontal fine cells comprise one coarse surface cell.
                    expected = n * (4 if level == "Level_0" else 1)
                    compare(f"preserved-step{step}-{level}-{value}", left / level / value, right / level / value, expected)
            diff = particle_compare(left / "super_droplets_moisture", right / "super_droplets_moisture")
            check(f"particles-step{step}", not diff["changed_real_fields"] and not diff["changed_level_or_grid"] and
                  not diff["changed_integer_records"] and diff["next_id_equal"], diff)
            for n, value in enumerate(FIELDS, 1):
                # 8x8 coarse surface cells, each with four fine children.
                expected = 256 * n
                actual = total(right, value) - total(left, value)
                check(f"composite-step{step}-{value}", abs(actual-expected) <= 512*EPS*expected,
                      {"kg": actual, "expected_kg": expected})
        for value in FIELDS:
            generated = [total(initial, value)] + [total(reference / f"chk{step:05d}", value) for step in range(2, 9)]
            check("generated-nondecreasing-" + value, generated[0] > 0 and
                  all(b >= a for a, b in zip(generated, generated[1:])), generated)
        results["passed"] = all(row["passed"] for row in results["checks"])
        assert results["passed"], "OneWay ground history comparisons failed"
    except BaseException as error:
        results["error"] = repr(error)
        raise
    finally:
        save()
        print(root)


if __name__ == "__main__":
    main()
