#!/usr/bin/env python3
"""Check that a fine-level restart preserves SLM's precipitation baseline.

CPU double-precision runtime regression. A common shift of the native cumulative
rain field and its SLM baseline must not change the resumed physical solution.
The fixture uses actual ERF checkpoint I/O and nonzero Kessler sedimentation.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


LIMIT = 1e-10
FIELDS = ("Cell", "RainAccum", "SlmPrecip", "SLM_mw", "SLM_mws",
          "SLM_wet_canop", "SLM_t_canop")


def invoke(command, directory, label):
    (directory / (label + "-command.json")).write_text(json.dumps(command, indent=2) + "\n")
    run = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=60)
    output = run.stdout + run.stderr
    (directory / (label + ".log")).write_text(output)
    if run.returncode:
        raise RuntimeError(f"{label} exited with {run.returncode}; see {directory / (label + '.log')}")
    return output


def tool_records(tool, directory, label, *arguments):
    output = invoke([str(tool), *map(str, arguments)], directory, label)
    records = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    if not records:
        raise RuntimeError(f"{label} returned no numerical records")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erf", type=Path, required=True)
    parser.add_argument("--checkpoint-tool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    erf = args.erf.resolve()
    tool = args.checkpoint_tool.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="slm-precip-restart-", dir=args.output.resolve()))
    fixtures = Path(__file__).resolve().parent / "test_files/SLM_Restart"
    common = [str(erf), "inputs", "erf.check_int=1", "erf.check_file=chk",
              "erf.plot_int_1=-1", "erf.plot_lsm=false", "erf.fixed_dt=0.01",
              "amrex.throw_exception=1", "amrex.signal_handling=0", "amrex.fpe_trap_invalid=0",
              "amr.max_grid_size=4", "amr.max_grid_size_z=16", "amr.blocking_factor=2",
              "amr.max_level=1", "amr.ref_ratio_vect=2 2 1", "erf.regrid_int=-1",
              "erf.refinement_indicators=patch", "erf.patch.max_level=1",
              "erf.patch.in_box_lo_indices_crse=0 0 0",
              "erf.patch.in_box_hi_indices_crse=3 7 15", "amr.n_error_buf=0"]
    report = {"erf_sha256": hashlib.sha256(erf.read_bytes()).hexdigest(),
              "checkpoint_tool_sha256": hashlib.sha256(tool.read_bytes()).hexdigest(),
              "absolute_tolerance": LIMIT, "work": str(work), "cases": [], "passes": False}

    def prepare(name):
        directory = work / name
        directory.mkdir()
        shutil.copyfile(fixtures / "SLM_Restart.i", directory / "inputs")
        shutil.copyfile(fixtures / "input_sounding", directory / "input_sounding")
        return directory

    try:
        initial = prepare("initial")
        invoke(common + ["max_step=0"], initial, "erf")
        baseline = None
        for name, history, legacy in (("aligned", 12.0, False),
                                       ("fine-higher", 12.01, False),
                                       ("fine-lower", 11.99, False),
                                       ("legacy-fine-higher", 12.01, True)):
            directory = prepare(name)
            checkpoint = directory / "input-checkpoint"
            shutil.copytree(initial / "chk00000", checkpoint)
            for lev, level_history in ((0, 12.0), (1, history)):
                tool_records(tool, directory, f"seed-{lev}", "seed", checkpoint,
                             level_history, "0.001", lev)
            if legacy:
                header = checkpoint / "Level_1/SlmPrecip_H"
                header.rename(header.with_name("SlmPrecip_H.hidden"))
            log = invoke(common + ["max_step=1", "erf.restart=" + str(checkpoint)], directory, "erf")
            current = directory / "chk00001/Level_1"
            row = {"name": name, "native_history": history, "legacy": legacy, "checks": []}
            if legacy:
                row["checks"].append("Checkpoint has no SLM precipitation baseline" in log)
            if baseline is None:
                baseline = current
            rain = tool_records(tool, directory, "rain-stats", "stats", current / "RainAccum")[0]
            slm = tool_records(tool, directory, "baseline-stats", "stats", current / "SlmPrecip")[0]
            puddle = tool_records(tool, directory, "puddle-stats", "stats", current / "SLM_mws")[0]
            increment = rain["max"] - history
            row["rain_increment"] = increment
            row["maximum_puddle_water"] = puddle["max"]
            row["checks"].append(math.isfinite(increment) and increment > 1e-8)
            row["checks"].append(abs(slm["max"] - rain["max"]) <= LIMIT)
            alignment = tool_records(tool, directory, "baseline-alignment", "baseline",
                                     directory / "chk00001", 1)[0]
            row["checks"].append(alignment["surface_cells"] > 0 and
                                 alignment["max_abs_diff"] <= LIMIT)
            # Initial puddle water is zero and new rain is about 6.6e-5 mm.
            # A coarse baseline of zero would replay the 12 mm history here.
            row["checks"].append(math.isfinite(puddle["max"]) and puddle["max"] < 1e-3)
            row["differences"] = {}
            for field in FIELDS:
                offset = history - 12.0 if field in ("RainAccum", "SlmPrecip") else 0.0
                differences = tool_records(tool, directory, "compare-" + field,
                                           "compare", baseline / field, current / field, offset)
                maximum = max(record["max_abs_diff"] for record in differences)
                row["differences"][field] = maximum
                row["checks"].append(math.isfinite(maximum) and maximum <= LIMIT)
            row["passes"] = all(row["checks"])
            report["cases"].append(row)
        # Continue an actual evolved checkpoint as well: its canopy has already
        # initialized, and its native rain was produced by Kessler sedimentation.
        uninterrupted = prepare("evolved-direct")
        invoke(common + ["max_step=2", "erf.restart=" + str(work / "aligned/input-checkpoint")],
               uninterrupted, "erf")
        continued = prepare("evolved-restarted")
        invoke(common + ["max_step=2", "erf.restart=" + str(work / "aligned/chk00001")],
               continued, "erf")
        row = {"name": "evolved-restart-parity", "checks": [], "differences": {}}
        for field in FIELDS:
            differences = tool_records(tool, continued, "compare-" + field, "compare",
                                       uninterrupted / "chk00002/Level_1" / field,
                                       continued / "chk00002/Level_1" / field, 0)
            maximum = max(record["max_abs_diff"] for record in differences)
            row["differences"][field] = maximum
            row["checks"].append(math.isfinite(maximum) and maximum <= LIMIT)
        row["passes"] = all(row["checks"])
        report["cases"].append(row)
        report["passes"] = all(case["passes"] for case in report["cases"])
    except (OSError, RuntimeError, subprocess.TimeoutExpired, ValueError) as error:
        report["error"] = str(error)
    (work / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
