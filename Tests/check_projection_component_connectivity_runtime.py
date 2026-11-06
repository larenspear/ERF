#!/usr/bin/env python3
"""Check independent pressure components with full CPU, DOUBLE ERF execution.

Each regular constant-spacing component needs its own solvability mean. Patches
touching only at edges or corners must not acquire a shared pressure nullspace.
The deterministic velocity perturbation supplies nonzero projection right-hand
sides. These numerical regressions do not prove convergence for arbitrary grids.
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
geometry.prob_hi = 240 240 80
amr.n_cell = 24 24 8
geometry.is_periodic = 0 0 0
xlo.type = SlipWall
xhi.type = SlipWall
ylo.type = SlipWall
yhi.type = SlipWall
zlo.type = SlipWall
zhi.type = SlipWall
amr.max_level = 1
amr.ref_ratio_vect = 2 2 1
amr.max_grid_size = 64
amr.blocking_factor = 2
amr.n_error_buf = 0
amr.no_box_split_dir = -1
amr.grid_eff = 1
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
prob.U_0_Pert_Mag = 0.1
prob.V_0_Pert_Mag = 0.1
prob.W_0_Pert_Mag = 0.1
erf.deterministic_ic_pert = 1
erf.fix_random_seed = 1
erf.poisson_abstol = 1e-12
erf.poisson_reltol = 1e-12
"""
# This allows 50 times the requested absolute solver tolerance, covering the
# subsequent flux update and divergence reduction while remaining far below the
# original failed solve's residual (~1e-7).
LIMIT = 5.e-11
BEFORE = re.compile(
    r"Max/L2 norm of divergence before solve in subdomain (\d+) at level (\d+) : "
    r"(\S+) (\S+) and volume-weighted sum (\S+)"
)
AFTER = re.compile(
    r"Max/L2 norm of divergence after\s+solve at level (\d+) : "
    r"(\S+) (\S+) and volume-weighted sum (\S+)"
)
SOLVES = re.compile(r"Solving in subdomain (\d+) of (\d+) bins at level (\d+)")


def cases():
    full = ((4, 4, 0), (11, 11, 7))
    lower = ((4, 4, 0), (11, 11, 3))
    layouts = (
        ("edge_xy", (full, ((12, 12, 0), (19, 19, 7))), 2),
        ("edge_xz", (lower, ((12, 4, 4), (19, 11, 7))), 2),
        ("corner_xyz", (lower, ((12, 12, 4), (19, 19, 7))), 2),
        ("face", (full, ((12, 4, 0), (19, 11, 7))), 1),
        ("partial_face", (full, ((12, 8, 0), (19, 15, 7))), 1),
        ("separate", (full, ((13, 13, 0), (20, 20, 7))), 2),
        ("single", (full,), 1),
    )
    for name, boxes, components in layouts:
        for fft in (False, True):
            yield name + ("-auto" if fft else "-mlmg"), boxes, components, fft


def run_case(executable, output, case):
    name, boxes, components, fft = case
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    text = BASE + f"erf.use_fft = {str(fft).lower()}\n"
    text += "erf.refinement_indicators = " + " ".join(f"patch{i}" for i in range(len(boxes))) + "\n"
    for i, (lower, upper) in enumerate(boxes):
        text += f"erf.patch{i}.max_level = 1\n"
        text += f"erf.patch{i}.in_box_lo_indices_crse = " + " ".join(map(str, lower)) + "\n"
        text += f"erf.patch{i}.in_box_hi_indices_crse = " + " ".join(map(str, upper)) + "\n"
    (directory / "inputs").write_text(text)
    row = dict(name=name, expected_components=components, passes=False)
    try:
        run = subprocess.run([str(executable), "inputs"], cwd=directory,
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        row["error"] = str(error)
        if isinstance(error, subprocess.TimeoutExpired):
            (directory / "output.log").write_bytes((error.stdout or b"") + (error.stderr or b""))
        return row
    log = run.stdout + run.stderr
    (directory / "output.log").write_text(log)
    before_diagnostics = [
        (int(sub), int(level), float(maximum), float(l2), float(volume_sum))
        for sub, level, maximum, l2, volume_sum in BEFORE.findall(log)
    ]
    after_diagnostics = [
        (int(level), float(maximum), float(l2), float(volume_sum))
        for level, maximum, l2, volume_sum in AFTER.findall(log)
    ]
    before = [(sub, level, maximum) for sub, level, maximum, _, _ in before_diagnostics]
    after = [(level, maximum) for level, maximum, _, _ in after_diagnostics]
    solves = [(int(sub), int(count), int(level)) for sub, count, level in SOLVES.findall(log)]
    fine = [(sub, count) for sub, count, level in solves if level == 1]
    row.update(returncode=run.returncode, before=before, after=after, solves=solves,
               before_diagnostics=before_diagnostics, after_diagnostics=after_diagnostics,
               absolute_limit=LIMIT)
    row["passes"] = (
        run.returncode == 0
        and len(before) == 1 + components
        and all(math.isfinite(value) for diagnostics in before_diagnostics
                for value in diagnostics[2:])
        and all(math.isfinite(value) for diagnostics in after_diagnostics
                for value in diagnostics[1:])
        and all(math.isfinite(value) and value > 1.e-3 for _, _, value in before)
        and [level for level, _ in after] == [0, 1]
        and all(math.isfinite(value) and 0 <= value < LIMIT for _, value in after)
        and fine == [(sub, components) for sub in range(components)]
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    executable = args.erf.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases():
        row = run_case(executable, args.output, case)
        rows.append(row)
        print(json.dumps(row), flush=True)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
    return int(any(not row["passes"] for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
