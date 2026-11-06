#!/usr/bin/env python3
"""CPU-double serial ground-species and aerosol checkpoint regression.

Generated kg histories must survive an immediate recheckpoint and the first
resumed step. Particle and atmospheric state comparisons detect collateral
changes. These fixed-grid tests do not establish AMR lifecycle conservation.
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
FINE = ["amr.max_level=1", "amr.ref_ratio_vect=2 2 2", "amr.max_grid_size=8",
        "amr.max_grid_size_z=16", "amr.n_error_buf=0", "erf.regrid_int=-1",
        "erf.coupling_type=TwoWay", "erf.cf_width=0", "erf.refinement_indicators=patch",
        "erf.patch.max_level=1", "erf.patch.in_box_lo_indices_crse=0 0 0",
        "erf.patch.in_box_hi_indices_crse=3 7 7"]
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
    root = Path(tempfile.mkdtemp(prefix="particle-ground-restart-", dir=args.work_dir.resolve()))
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

    def run(name, options, checkpoint=None, error=None):
        out = root / name
        out.mkdir()
        shutil.copy2(args.inputs, out / "inputs")
        command = [str(exe), "inputs", *options]
        if checkpoint:
            shutil.copytree(checkpoint, out / "input-checkpoint")
            command += ["erf.restart=input-checkpoint"]
        (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (out / "run.log").open("w") as log:
            proc = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        results["runs"].append({"name": name, "returncode": proc.returncode, "expected_error": error})
        save()
        if error:
            check(name, proc.returncode != 0 and error in (out / "run.log").read_text(),
                  {"returncode": proc.returncode, "expected_error": error})
        else:
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

    def manifest(chk):
        expected = "ERF ground mass v1\n" + str(len(FIELDS)) + "\n" + "".join(n + "\n" for n in FIELDS)
        for level in levels(chk):
            actual = (chk / level / "GroundMassFields").read_text()
            check(str(chk.relative_to(root)) + "-" + level + "-manifest", actual == expected, actual)

    def fields_equal(name, a, b, field_names, exact):
        assert levels(a) == levels(b), "checkpoint levels differ"
        for level in levels(a):
            for value in field_names:
                data = field("compare", a / level / value, b / level / value, 0)
                for row in data:
                    # A one-kg floor would mask complete loss of small aerosol masses.
                    floor = 1e-30 if value.startswith("Ground") else 1
                    row["tolerance"] = 0 if exact else 64 * EPS * max(floor, row["left_norm"])
                check(name + "-" + level + "-" + value,
                      all(row["max_abs_diff"] <= row["tolerance"] for row in data), data)

    def particles_equal(name, a, b):
        data = particle_compare(a / "super_droplets_moisture", b / "super_droplets_moisture")
        passed = (not data["changed_real_fields"] and data["changed_level_or_grid"] ==
                  data["changed_integer_records"] == 0 and data["next_id_equal"])
        check(name, passed, data)

    try:
        for cold, fine in [(True, False), (True, True), (False, False), (False, True)]:
            tag = ("cold" if cold else "warm") + ("-fine" if fine else "-base")
            options = ([] if cold else ["super_droplets_moisture.include_cold_processes=false"]) + (FINE if fine else [])
            continuous = run(tag + "-continuous", options)
            chk1, chk2 = continuous / "chk00001", continuous / "chk00002"
            assert len(levels(chk1)) == (2 if fine else 1)
            immediate = run(tag + "-immediate", [*options, "max_step=1", "erf.check_int=-1", "erf.check_per=1"], chk1)
            resumed = run(tag + "-resumed", options, chk1)
            for chk in (chk1, chk2, immediate / "chk00001", resumed / "chk00002"):
                manifest(chk)
            names = [*FIELDS, "RainAccum", *(["SnowAccum"] if cold else []), "Cell", "XFace", "YFace", "ZFace"]
            fields_equal(tag + "-immediate", chk1, immediate / "chk00001", names, True)
            fields_equal(tag + "-resumed", chk2, resumed / "chk00002", names, False)
            particles_equal(tag + "-immediate-particles", chk1, immediate / "chk00001")
            particles_equal(tag + "-resumed-particles", chk2, resumed / "chk00002")
            for level in levels(chk1):
                for value in FIELDS:
                    before, after = field("stats", chk1 / level / value), field("stats", chk2 / level / value)
                    assert len(before) == len(after) == 1
                    check(tag + "-generated-" + level + "-" + value,
                          after[0]["sum"] > before[0]["sum"] > 0 and before[0]["min"] >= 0 and after[0]["min"] >= 0,
                          {"step1": before[0], "step2": after[0]})
            if cold and fine:
                for missing in ("GroundMassFields", FIELDS[-1] + "_H"):
                    bad = root / ("missing-" + missing)
                    shutil.copytree(chk1, bad)
                    (bad / "Level_1" / missing).unlink()
                    diagnostic = "missing ground mass manifest" if missing == "GroundMassFields" else "missing ground mass field"
                    run("reject-" + missing, options, bad, error=diagnostic)
        results["passed"] = all(row["passed"] for row in results["checks"])
        assert results["passed"], "ground mass checkpoint comparisons failed"
    except BaseException as error:
        results["error"] = repr(error)
        raise
    finally:
        save()
        print(root)


if __name__ == "__main__":
    main()
