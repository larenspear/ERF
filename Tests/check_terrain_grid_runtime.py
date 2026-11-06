#!/usr/bin/env python3
"""Reject terrain layers that collapse after translation; preserve valid controls.

The three small startup cases require an ERF executable with FFT support. The
invalid case previously exited successfully with NaNs in its initial projection.
This is executable regression evidence, not a universal floating-point proof.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import resource
import shutil
import subprocess


BASE = """erf.prob_name = ABL
max_step = 0
amrex.throw_exception = 1
amrex.signal_handling = 0
geometry.prob_lo = 0 0 0
geometry.prob_hi = 160 80 2048
amr.n_cell = 16 8 8
geometry.is_periodic = 1 1 0
amr.max_level = 0
amr.max_grid_size = 64
amr.blocking_factor = 2
erf.anelastic = 1
erf.fixed_dt = 0.01
erf.init_type = Uniform
erf.use_gravity = false
erf.use_coriolis = false
erf.molec_diff_type = None
erf.les_type = None
erf.pbl_type = None
erf.v = 1
erf.mg_v = 2
amr.v = 1
erf.check_int = -1
erf.plot_int_1 = -1
erf.sum_interval = -1
prob.rho_0 = 1.0
prob.T_0 = 300.0
prob.W_0_Pert_Mag = 0.0
erf.deterministic_ic_pert = 1
erf.fix_random_seed = 1
erf.poisson_abstol = 1e-12
erf.poisson_reltol = 1e-12
zlo.type = SlipWall
zhi.type = SlipWall
erf.use_fft = true
erf.terrain_type = StaticFittedMesh
prob.custom_terrain_type = RaisedFlat
"""
REJECTION = "Physical vertical grid must have finite, strictly positive stored layer"
NORMS = re.compile(
    r"Max/L2 norm of divergence (?:before solve in subdomain \d+ at level \d+|"
    r"after\s+solve at level \d+) : (\S+) (\S+) and volume-weighted sum (\S+)"
)


def check_case(executable, output, name, surface, first, reject):
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    deck = BASE + f"prob.z_offset = {surface}\n"
    deck += f"erf.terrain_z_levels = 0 {first} 10 100 250 500 1000 1500 2048\n"
    (directory / "inputs").write_text(deck)
    row = {"name": name, "passes": False, "expected_rejection": reject}
    try:
        run = subprocess.run([str(executable), "inputs"], cwd=directory,
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as error:
        row["error"] = str(error)
        return row
    log = run.stdout + run.stderr
    (directory / "output.log").write_text(log)
    norms = NORMS.findall(log)
    row.update(returncode=run.returncode, projection_norms=norms)
    if reject:
        row["passes"] = (run.returncode != 0 and REJECTION in log
                         and "Projecting initial velocity" not in log)
    else:
        # The controls start with zero velocity. All three reported quantities
        # must be finite; checking the maximum alone used to miss the NaN L2 norm.
        row["passes"] = (run.returncode == 0 and len(norms) == 2
                         and all(math.isfinite(float(value)) and abs(float(value)) < 1e-12
                                 for values in norms for value in values))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Isolate execution from a concurrent rebuild of the supplied executable.
    executable = output / "erf_exec_snapshot"
    if args.erf.resolve() != executable:
        shutil.copy2(args.erf.resolve(), executable)
    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    rows = []
    for case in (("translated-collapse", "1000", "1e-14", True),
                 ("ordinary-translated-control", "1000", "1", False),
                 ("tiny-zero-surface-control", "0", "1e-14", False)):
        row = check_case(executable, output, *case)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (output / "results.json").write_text(json.dumps(
        {"executable_sha256": digest, "cases": rows}, indent=2) + "\n")
    return int(any(not row["passes"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
