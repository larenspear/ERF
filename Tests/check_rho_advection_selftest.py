#!/usr/bin/env python3
"""Inject bugs in AdvectionSrcForRho and require the source checker to catch them."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_rho_advection.py"
STATE_CPP = "Source/Advection/ERF_AdvectionSrcForState.cpp"

# name -> (old text, new text, occurrence index, total old occurrences)
MUTANTS = {
    "baseline (no change)": None,
    "x mass flux uses the wrong map factor": (
        "ax_arr(i,j,k) * rho_u(i,j,k) / mf_uy(i,j,0)",
        "ax_arr(i,j,k) * rho_u(i,j,k) / mf_mx(i,j,0)", 0, 1),
    "y mass flux uses the wrong area fraction": (
        "ay_arr(i,j,k) * rho_v(i,j,k) / mf_vx(i,j,0)",
        "ax_arr(i,j,k) * rho_v(i,j,k) / mf_vx(i,j,0)", 0, 1),
    "vertical mass flux multiplies by mfsq": (
        "az_arr(i,j,k) * Omega(i,j,k) / mfsq",
        "az_arr(i,j,k) * Omega(i,j,k) * mfsq", 0, 1),
    "x averaged momentum copies the y flux": (
        "avg_xmom(i,j,k) = (flx_arr[0])(i,j,k,0);",
        "avg_xmom(i,j,k) = (flx_arr[1])(i,j,k,0);", 0, 1),
    "fixed_rho branch writes nonzero tendency": (
        "advectionSrc(i,j,k,0) = zero;", "advectionSrc(i,j,k,0) = one;", 0, 2),
    "x inverse spacing uses the y spacing": (
        "auto dxInv = cellSizeInv[0], dyInv = cellSizeInv[1], dzInv = cellSizeInv[2];",
        "auto dxInv = cellSizeInv[1], dyInv = cellSizeInv[1], dzInv = cellSizeInv[2];", 0, 1),
    "reverse the x mass-flux difference": (
        "( (flx_arr[0])(i+1,j,k,0) - (flx_arr[0])(i  ,j,k,0) )",
        "( (flx_arr[0])(i  ,j,k,0) - (flx_arr[0])(i+1,j,k,0) )", 0, 1),
    "include zero-Jacobian cells in the divergence": (
        "if (detJ(i,j,k) > zero)", "if (detJ(i,j,k) >= zero)", 0, 1),
    "nonpositive Jacobian branch writes nonzero tendency": (
        "advectionSrc(i,j,k,0) = zero;", "advectionSrc(i,j,k,0) = one;", 1, 2),
    "vertical divergence reads the wrong high face": (
        "(flx_arr[2])(i,j,k+1,0)", "(flx_arr[2])(i,j,k+2,0)", 0, 1),
    'update density after the fixed-rho assignment': ('advectionSrc(i,j,k,0) = zero;',
         'advectionSrc(i,j,k,0) = zero;\n            advectionSrc(i,j,k,0) += 1;',
         0,
         2),
    'source zero constant is changed': ('zero         = amrex::Real(0.0)',
         'zero         = amrex::Real(1.0)',
         0,
         1,
         'Source/ERF_NumericalConstants.H'),
}


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="erf-rho-advection-mutants-"))
    originals = {name: (ERF / name).read_text() for name in (STATE_CPP, CONST_H)}
    print(f"mutants in {work}\n")
    unexpected = 0
    for index, (name, edit) in enumerate(MUTANTS.items()):
        root = work / f"m{index}"
        for relative, original in originals.items():
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(original)
        relative = edit[4] if edit is not None and len(edit) == 5 else STATE_CPP
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
        caught, expected, failures = mutation_outcome(result, edit is not None)
        unexpected += not expected
        print(f"[{'CAUGHT' if caught else 'passes'}] {name}")
        for failure in failures[:3]:
            print(f"           {failure}")

    print(f"\n{unexpected} unexpected outcome(s)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
