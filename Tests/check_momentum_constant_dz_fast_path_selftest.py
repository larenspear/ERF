#!/usr/bin/env python3
"""Inject defects into the constant-dz momentum path and require detection."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from momentum_flux_cases import flux_cases

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_momentum_constant_dz_fast_path.py"
MOM_CPP = "Source/Advection/ERF_AdvectionSrcForMom_ConstantDz.cpp"

# name -> (old text, replacement, occurrence index, total old occurrences[, source path])
MUTANTS = {
    "baseline (no change)": None,
    "inline path omits the vertical scheme gate": (
        "horiz_adv_type == AdvType::Centered_2nd && vert_adv_type == AdvType::Centered_2nd",
        "horiz_adv_type == AdvType::Centered_2nd", 0, 1),
    "x momentum uses the wrong map factors": (
        "mf_ux(i,j,0) * mf_uy(i,j,0)", "mf_ux(i,j,0) * mf_ux(i,j,0)", 0, 1),
    "y momentum uses the wrong map factors": (
        "mf_vx(i,j,0) * mf_vy(i,j,0)", "mf_vx(i,j,0) * mf_vx(i,j,0)", 0, 1),
    "z momentum uses the wrong horizontal map factor": (
        "mf_mx(i,j,0) * mf_my(i,j,0)", "mf_mx(i,j,0) * mf_mx(i,j,0)", 0, 1),
    "reverse x-flux orientation in x-momentum": (
        "(xflux_hi - xflux_lo) * dxInv * mfsq",
        "(xflux_lo - xflux_hi) * dxInv * mfsq", 0, 3),
    "omit mfsq from the y term in y-momentum": (
        "(yflux_hi - yflux_lo) * dyInv * mfsq",
        "(yflux_hi - yflux_lo) * dyInv", 1, 3),
    "include mfsq in the z term in z-momentum": (
        "(zflux_hi - zflux_lo) * dzInv;",
        "(zflux_hi - zflux_lo) * dzInv * mfsq;", 2, 3),
    "x wrapper writes the positive source": (
        "rho_u_rhs(i, j, k) = -advectionSrc;",
        "rho_u_rhs(i, j, k) = advectionSrc;", 0, 1),
    "z high flux uses the wrong boundary index": (
        "(k == hi_z_face) ? omega(i,j,k) * w(i,j,k) :",
        "(k == lo_z_face) ? omega(i,j,k) * w(i,j,k) :", 0, 1),
    "z top flux is averaged instead of using the wall value": (
        "(k == hi_z_face) ? omega(i,j,k) * w(i,j,k) :",
        "(k == hi_z_face) ? fourth * (omega(i,j,k) + omega(i,j,k+1)) * "
        "(w(i,j,k) + w(i,j,k+1)) :", 0, 1),
    "z interior high flux uses the wrong neighbor": (
        "omega(i,j,k) + omega(i,j,k+1)",
        "omega(i,j,k) + omega(i,j,k+2)", 0, 1),
    'component launch boxes are swapped': ('ParallelFor(bxx, bxy, bxz,', 'ParallelFor(bxy, bxx, bxz,', 0, 1),
    'update tendency after the checked assignment': ('rho_u_rhs(i, j, k) = -advectionSrc;',
         'rho_u_rhs(i, j, k) = -advectionSrc;\n                rho_u_rhs(i,j,k) *= -1;',
         0,
         1),
    'source fourth constant is doubled': ('fourth       = amrex::Real(0.25)',
         'fourth       = amrex::Real(0.5)',
         0,
         1,
         'Source/ERF_NumericalConstants.H'),
    'vertical spacing uses horizontal axis': ('dzInv = cellSizeInv[2]', 'dzInv = cellSizeInv[0]', 0, 1),
    'spacing alias is scaled after the indexed read': ('dxInv = cellSizeInv[0]', 'dxInv = cellSizeInv[0] * 2', 0, 1),
}


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="erf-momentum-constant-dz-mutants-"))
    originals = {name: (ERF / name).read_text() for name in (MOM_CPP, CONST_H)}
    flux_mutants, controls = flux_cases(originals[MOM_CPP])
    cases = {**MUTANTS, **flux_mutants, **controls}
    print(f"mutants in {work}\n")
    unexpected = 0
    for index, (name, edit) in enumerate(cases.items()):
        root = work / f"m{index}"
        for relative, original in originals.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original)
        relative = edit[4] if edit is not None and len(edit) == 5 else MOM_CPP
        target = root / relative
        source = originals[relative]
        if edit is not None:
            old, new, occurrence, expected_count = edit[:4]
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
        caught, expected, failures = mutation_outcome(result, edit is not None and name not in controls)
        unexpected += not expected
        print(f"[{'CAUGHT' if caught else 'passes'}] {name}")
        for failure in failures[:3]:
            print(f"           {failure}")

    print(f"\n{unexpected} unexpected outcome(s)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
