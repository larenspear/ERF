#!/usr/bin/env python3
"""Inject scalar-advection routing and divergence bugs and require detection."""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_scalar_flux_divergence.py"
SCALAR_H = "Source/Advection/ERF_AdvectionSrcForScalars.H"
STATE_CPP = "Source/Advection/ERF_AdvectionSrcForState.cpp"

# name -> (source path, old text, replacement, expected old occurrences)
MUTANTS = {
    "baseline (no change)": None,
    "reverse x-face difference": (
        SCALAR_H,
        "flx_arr[0](i+1,j,k,flux_comp) - flx_arr[0](i,j,k,flux_comp)",
        "flx_arr[0](i,j,k,flux_comp) - flx_arr[0](i+1,j,k,flux_comp)", 1),
    "read the wrong y-high face": (
        SCALAR_H, "flx_arr[1](i,j+1,k,flux_comp)", "flx_arr[1](i,j+2,k,flux_comp)", 1),
    "use the wrong metric factor": (
        SCALAR_H, "mf_mx(i,j,0) * mf_my(i,j,0)", "mf_mx(i,j,0) * mf_mx(i,j,0)", 1),
    "include cells with zero Jacobian": (
        SCALAR_H, "if (detJ(i,j,k) > zero)", "if (detJ(i,j,k) >= zero)", 1),
    "write nonzero tendency for a nonpositive Jacobian": (
        SCALAR_H, "rhs(i,j,k,rhs_comp) = zero;", "rhs(i,j,k,rhs_comp) = one;", 1),
    "use y velocity for the x-face flux": (
        SCALAR_H,
        "flx_arr[0](i,j,k,flux_comp) = avg_xmom(i,j,k) * interpx;",
        "flx_arr[0](i,j,k,flux_comp) = avg_ymom(i,j,k) * interpx;", 1),
    "shift the primitive scalar component": (
        STATE_CPP, "const int scalar_comp = cons_comp - 1;",
        "const int scalar_comp = cons_comp;", 1),
    "shift the flux component away from its conserved component": (
        STATE_CPP, "const int flux_comp = cons_comp;",
        "const int flux_comp = cons_comp + 1;", 1),
    "write divergence into the primitive instead of conserved component": (
        STATE_CPP,
        "advectionSrc, cons_comp, detJ,",
        "advectionSrc, scalar_comp, detJ,", 1),
    'update tendency after nonpositive branch assignment': ('Source/Advection/ERF_AdvectionSrcForScalars.H',
         'rhs(i,j,k,rhs_comp) = zero;',
         'rhs(i,j,k,rhs_comp) = zero;\n            rhs(i,j,k,rhs_comp) += 1;',
         1),
    'source zero constant is changed': ('Source/ERF_NumericalConstants.H',
         'zero         = amrex::Real(0.0)',
         'zero         = amrex::Real(1.0)',
         1),
    'source one constant is changed': ('Source/ERF_NumericalConstants.H',
         'one          = amrex::Real(1.0)',
         'one          = amrex::Real(2.0)',
         1),
}


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="erf-scalar-divergence-mutants-"))
    originals = {
        CONST_H: (ERF / CONST_H).read_text(),
        SCALAR_H: (ERF / SCALAR_H).read_text(),
        STATE_CPP: (ERF / STATE_CPP).read_text(),
    }
    print(f"mutants in {work}\n")
    unexpected = 0
    for index, (name, edit) in enumerate(MUTANTS.items()):
        root = work / f"m{index}"
        for relative, original in originals.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original)

        if edit is not None:
            relative, old, new, expected_count = edit
            target = root / relative
            text = target.read_text()
            count = text.count(old)
            if count != expected_count:
                raise AssertionError(f"{name}: found {count} pattern(s), expected {expected_count}")
            target.write_text(text.replace(old, new, 1))

        result = subprocess.run([sys.executable, str(VERIFY), str(root)],
                                capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout.strip() or "checker skipped because z3-solver is unavailable")
            return 77
        caught, expected, failures = mutation_outcome(result, edit is not None)
        unexpected += not expected
        print(f"[{'CAUGHT' if caught else 'passes'}] {name}")
        for failure in failures[:3]:
            print(f"           {failure}")

    print(f"\n{unexpected} unexpected outcome(s)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
