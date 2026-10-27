#!/usr/bin/env python3
"""Inject bugs in the normal-grid momentum divergence helpers and require detection."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_momentum_divergence.py"
MOM_N_H = "Source/Advection/ERF_AdvectionSrcForMom_N.H"

# name -> (old text, replacement, occurrence index, total old occurrences)
MUTANTS = {
    "baseline (no change)": None,
    "x momentum uses the wrong map factors": (
        "1 / (mf_ux_inv(i,j,0) * mf_uy_inv(i,j,0))",
        "1 / (mf_ux_inv(i,j,0) * mf_ux_inv(i,j,0))", 0, 1),
    "y momentum uses the wrong map factors": (
        "1 / (mf_vx_inv(i,j,0) * mf_vy_inv(i,j,0))",
        "1 / (mf_vx_inv(i,j,0) * mf_vx_inv(i,j,0))", 0, 1),
    "z momentum uses the wrong horizontal map factor": (
        "mf_mx(i,j,0) * mf_my(i,j,0)", "mf_mx(i,j,0) * mf_mx(i,j,0)", 0, 1),
    "reverse x-flux orientation in x-momentum": (
        "(xflux_hi - xflux_lo) * dxInv * mfsq",
        "(xflux_lo - xflux_hi) * dxInv * mfsq", 0, 3),
    "omit mfsq from the y term in x-momentum": (
        "(yflux_hi - yflux_lo) * dyInv * mfsq",
        "(yflux_hi - yflux_lo) * dyInv", 0, 3),
    "include mfsq in the z term in x-momentum": (
        "(zflux_hi - zflux_lo) * dzInv;",
        "(zflux_hi - zflux_lo) * dzInv * mfsq;", 0, 3),
    "x wrapper writes the positive helper result": (
        "rho_u_rhs(i, j, k) = -AdvectionSrcForXMom_N",
        "rho_u_rhs(i, j, k) = AdvectionSrcForXMom_N", 0, 1),
    "y wrapper calls the x-momentum helper": (
        "rho_v_rhs(i, j, k) = -AdvectionSrcForYMom_N",
        "rho_v_rhs(i, j, k) = -AdvectionSrcForXMom_N", 0, 1),
    "z wrapper uses the wrong boundary dz branch": (
        "auto dzInv = (k == 0) ? one / dz_ptr[k] : two/(dz_ptr[k] + dz_ptr[k-1]);",
        "auto dzInv = (k == 1) ? one / dz_ptr[k] : two/(dz_ptr[k] + dz_ptr[k-1]);", 0, 1),
    "z helper returns the wrong sign": (
        "return advectionSrc;", "return -advectionSrc;", 2, 3),
}


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="erf-momentum-divergence-mutants-"))
    original = (ERF / MOM_N_H).read_text()
    print(f"mutants in {work}\n")
    unexpected = 0
    for index, (name, edit) in enumerate(MUTANTS.items()):
        root = work / f"m{index}"
        target = root / MOM_N_H
        target.parent.mkdir(parents=True, exist_ok=True)
        source = original
        if edit is not None:
            old, new, occurrence, expected_count = edit
            count = source.count(old)
            if count != expected_count:
                raise AssertionError(f"{name}: found {count} pattern(s), expected {expected_count}")
            starts = [match.start() for match in re.finditer(re.escape(old), source)]
            start = starts[occurrence]
            source = source[:start] + new + source[start + len(old):]
        target.write_text(source)

        result = subprocess.run([sys.executable, str(VERIFY), str(root)],
                                capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout.strip() or "checker skipped because z3-solver is unavailable")
            return 77
        failures = [line.strip() for line in result.stdout.splitlines()
                    if line.strip().startswith("FAIL")]
        if result.returncode and not failures:
            failures = [(result.stderr.strip().splitlines() or ["checker crashed"])[-1]]
        caught = bool(result.returncode)
        unexpected += caught == (edit is None)
        print(f"[{'CAUGHT' if caught else 'passes'}] {name}")
        for failure in failures[:3]:
            print(f"           {failure}")

    print(f"\n{unexpected} unexpected outcome(s)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
