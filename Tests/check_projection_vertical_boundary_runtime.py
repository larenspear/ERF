#!/usr/bin/env python3
"""Check full ERF projection behavior at vertical Outflow and wall faces.

Run with a CPU, double-precision ERF executable built with FFT support. The
small deterministic initial perturbation makes a failed projection observable
even when ERF itself exits successfully. This is numerical regression evidence,
not a proof of projection convergence for arbitrary inputs.
"""

import argparse
import json
import math
from pathlib import Path
import re
import resource
import subprocess


BASE = """erf.prob_name = ABL
max_step = 0
amrex.throw_exception = 1
amrex.signal_handling = 0
geometry.prob_lo = 0 0 0
geometry.prob_hi = 160 80 80
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
prob.W_0_Pert_Mag = 0.1
erf.deterministic_ic_pert = 1
erf.fix_random_seed = 1
erf.poisson_abstol = 1e-12
erf.poisson_reltol = 1e-12
"""
BEFORE = re.compile(r"divergence before solve in subdomain \d+ at level \d+ : (\S+)")
AFTER = re.compile(r"divergence after\s+solve at level \d+ : (\S+)")
TERRAIN_REJECTION = "Terrain projection does not support lower-z Outflow/Open boundaries"
LIMIT = 5e-11


def cases():
    for stretched in (False, True):
        for lower, upper in ((True, False), (False, True), (True, True), (False, False)):
            for backend in (("fft",) if stretched else ("fft", "mlmg")):
                name = f'{"stretched" if stretched else "constant"}-{backend}-{int(lower)}{int(upper)}'
                extra = "erf.terrain_z_levels = 0 5 11 18 26 35 45 57 80\n" if stretched else ""
                yield name, lower, upper, backend, extra, False
    for lower in (True, False):
        extra = "erf.terrain_type = StaticFittedMesh\nprob.custom_terrain_type = WoA\nprob.hmax = 1\nprob.L = 20\n"
        yield f'terrain-{"lower-outflow" if lower else "walls"}', lower, False, "fft", extra, lower


def check_case(executable, output, case):
    name, lower, upper, backend, extra, reject = case
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    deck = BASE + f'zlo.type = {"Outflow" if lower else "SlipWall"}\nzhi.type = {"Outflow" if upper else "SlipWall"}\n'
    deck += f'erf.use_fft = {str(backend == "fft").lower()}\n' + extra
    (directory / "inputs").write_text(deck)
    row = {"name": name, "passes": False, "expected_rejection": reject}
    try:
        run = subprocess.run([str(executable), "inputs"], cwd=directory,
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        row["error"] = str(error)
        return row
    log = run.stdout + run.stderr
    (directory / "output.log").write_text(log)
    before = [float(value) for value in BEFORE.findall(log)]
    after = [float(value) for value in AFTER.findall(log)]
    means = re.findall(r"Subtracting (\S+) from rhs", log)
    row.update(returncode=run.returncode, before=before, after=after,
               subtracted_means=means, absolute_limit=LIMIT)
    if reject:
        row["passes"] = run.returncode != 0 and TERRAIN_REJECTION in log
    else:
        row["passes"] = (run.returncode == 0 and len(before) == len(after) == 1
                         and math.isfinite(before[0]) and before[0] > 1e-3
                         and math.isfinite(after[0]) and 0 <= after[0] < LIMIT
                         and (not (lower or upper) or not means))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # The intentional unsupported-terrain case must not leave a core dump.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    executable = args.erf.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases():
        row = check_case(executable, args.output, case)
        rows.append(row)
        print(json.dumps(row), flush=True)
    (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return int(any(not row["passes"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
