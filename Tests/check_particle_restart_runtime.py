#!/usr/bin/env python3
"""CPU-double serial full ERF particle restart invariance.

Tests actual generated precipitation and particle attributes. Same-step output
must restore native and physical fields exactly. First-step comparisons allow
64 double epsilons at each field's native scale, never discretization error.
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

FINE = ["amr.max_level=1", "amr.ref_ratio_vect=2 2 2", "amr.max_grid_size=8",
        "amr.max_grid_size_z=16", "amr.n_error_buf=0", "erf.regrid_int=-1",
        "erf.cf_width=0", "erf.refinement_indicators=patch", "erf.patch.max_level=1",
        "erf.patch.in_box_lo_indices_crse=0 0 0", "erf.patch.in_box_hi_indices_crse=3 7 7"]
NONWATER = ["super_droplets_moisture.species=water",
            "super_droplets_moisture.initial_species_distribution_type_water=mass_constant",
            "super_droplets_moisture.initial_species_mean_mass_water=1e-9",
            "super_droplets_moisture.include_phase_change=true"]
CONFIGS = {
    "cold-base": [],
    "warm-base": ["super_droplets_moisture.include_cold_processes=false"],
    "cold-twoway": [*FINE, "erf.coupling_type=TwoWay"],
    "warm-twoway": [*FINE, "erf.coupling_type=TwoWay", "super_droplets_moisture.include_cold_processes=false"],
    "cold-oneway": [*FINE, "erf.coupling_type=OneWay"],
    "warm-oneway": [*FINE, "erf.coupling_type=OneWay", "super_droplets_moisture.include_cold_processes=false"],
    "density-scaling": ["super_droplets_moisture.density_scaling=true"],
    "condensate-density": ["super_droplets_moisture.distribution_type=condensate_density"],
    "phase-relaxation": ["super_droplets_moisture.include_phase_change=true",
                         "super_droplets_moisture.initial_phase_change_relaxation=true",
                         "super_droplets_moisture.initial_phase_change_relaxation_time=0.0001"],
    "nonwater-base": NONWATER,
    "nonwater-twoway": [*FINE, "erf.coupling_type=TwoWay", *NONWATER],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erf", type=Path, required=True)
    parser.add_argument("--tool", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, default=Path(__file__).parent / "test_files/SDM_Restart/SDM_Restart.i")
    parser.add_argument("--case", choices=CONFIGS, action="append")
    args = parser.parse_args()
    assert sys.byteorder == "little", "native particle reader currently supports little-endian hosts"
    args.work_dir.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="particle-restart-", dir=args.work_dir.resolve()))
    exe = root / "erf_exec"
    shutil.copy2(args.erf, exe)
    tool = args.tool.resolve()
    manifest = {"binary_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
                "cases": [], "checks": [], "passed": False}

    def save():
        (root / "results.json").write_text(json.dumps(manifest, indent=2) + "\n")

    def run(name, overrides, checkpoint=None):
        out = root / name
        out.mkdir()
        shutil.copy2(args.inputs, out / "inputs")
        command = [str(exe), "inputs", *overrides]
        if checkpoint:
            shutil.copytree(checkpoint, out / "input-checkpoint")
            command += ["erf.restart=input-checkpoint"]
        (out / "command.json").write_text(json.dumps(command, indent=2) + "\n")
        with (out / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=out, stdout=log, stderr=subprocess.STDOUT, timeout=60)
        manifest["cases"].append({"name": name, "returncode": result.returncode})
        save()
        assert result.returncode == 0, str(out / "run.log")
        return out

    def field(action, *options):
        result = subprocess.run([str(tool), action, *map(str, options)], capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        return [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]

    def add(name, passed, values):
        manifest["checks"].append({"name": name, "passed": passed, "values": values})
        if not passed:
            print("FAILED", name, values, flush=True)
        save()

    def compare_field(name, left, right, exact):
        values = field("compare", left, right, 0)
        for v in values:
            scale = v.get("reference_norm", v.get("left_norm"))
            v["tolerance"] = 0 if exact else 64 * sys.float_info.epsilon * max(1, scale)
        add(name, all(v["max_abs_diff"] <= v["tolerance"] for v in values), values)

    def compare_particles(name, left, right, exact):
        values = particle_compare(left / "super_droplets_moisture", right / "super_droplets_moisture")
        passed = values["changed_level_or_grid"] == values["changed_integer_records"] == 0 and values["next_id_equal"]
        for field_name, v in values["changed_real_fields"].items():
            # Mass and particle dimensions use their own physical scale; a metre
            # or kilogram floor would hide meaningful fractional particle changes.
            tiny = ("mass" in field_name or field_name in ("radius", "a", "c"))
            scale = max(1e-30 if tiny else 1, v["max_abs_reference"])
            v["tolerance"] = 0 if exact or field_name == "uid" else 64 * sys.float_info.epsilon * scale
            passed = passed and v["max_abs_diff"] <= v["tolerance"]
        add(name, passed, values)

    try:
        for name in args.case or CONFIGS:
            overrides = CONFIGS[name]
            native = ["RainAccum"] + ([] if name.startswith("warm") else ["SnowAccum"])
            continuous = run(name + "-continuous", overrides)
            chk1, chk2 = continuous / "chk00001", continuous / "chk00002"
            levels = sorted(p.name for p in chk1.glob("Level_*") if p.is_dir())
            assert len(levels) == (2 if any(x == "amr.max_level=1" for x in overrides) else 1)
            immediate = run(name + "-immediate", [*overrides, "max_step=1", "erf.check_int=-1", "erf.check_per=1"], chk1)
            resumed = run(name + "-resumed", overrides, chk1)
            for level in levels:
                for value in native:
                    start = field("stats", chk1 / level / value)[0]
                    end = field("stats", chk2 / level / value)[0]
                    add(name + "-generated-" + level + "-" + value,
                        start["sum"] > 0 and end["sum"] > start["sum"], {"step1": start, "step2": end})
                for value in [*native, "Cell", "XFace", "YFace", "ZFace"]:
                    compare_field(name + "-immediate-" + level + "-" + value,
                                  chk1 / level / value, immediate / "chk00001" / level / value, True)
                    compare_field(name + "-first-step-" + level + "-" + value,
                                  chk2 / level / value, resumed / "chk00002" / level / value, False)
                if name.startswith("nonwater"):
                    # Water alias is a distinct nonwater species; cold fixtures
                    # have six water slots, then its vapor and condensate slots.
                    cell = field("stats", chk1 / level / "Cell")
                    add(name + "-nonwater-present-" + level, cell[11]["max"] > 0, cell[11])
            compare_particles(name + "-immediate-particles", chk1, immediate / "chk00001", True)
            compare_particles(name + "-first-step-particles", chk2, resumed / "chk00002", False)
        manifest["passed"] = all(check["passed"] for check in manifest["checks"])
        assert manifest["passed"], "particle restart comparisons failed"
    except BaseException as error:
        manifest["error"] = repr(error)
        raise
    finally:
        save()
        print(root)


if __name__ == "__main__":
    main()
