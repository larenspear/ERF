#!/usr/bin/env python3
"""Inject WENO candidate and weight bugs and require the checker to catch them.

Mutants live in temporary copies of the two WENO headers; the repository files
are never changed.

usage: check_weno_candidates_selftest.py [WORK_DIR]
exit:  0 every outcome as expected, 1 otherwise, 77 z3 is not importable
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_weno_candidates.py"
HEADERS = (
    "Source/Utils/ERF_Interpolation_WENO.H",
    "Source/Utils/ERF_Interpolation_WENO_Z.H",
)

MUTANTS = {
    "baseline (no change)": None,
    "WENO3 candidate loses linear exactness": (
        HEADERS[0], "-sm1 + amrex::Real(3) * s", "-sm1 + amrex::Real(2) * s"),
    "WENO5 candidate coefficient typo": (
        HEADERS[0], "amrex::Real(2) * sm2 - amrex::Real(7.0) * sm1",
        "amrex::Real(3) * sm2 - amrex::Real(7.0) * sm1"),
    "WENO7 candidate coefficient typo": (
        HEADERS[0], "amrex::Real(23.0)/amrex::Real(12.0)",
        "amrex::Real(22.0)/amrex::Real(12.0)"),
    "WENO-MZQ3 third-order candidate typo": (
        HEADERS[1], "amrex::Real(5.0) * s + amrex::Real(2) * sp1) / amrex::Real(6.0)",
        "amrex::Real(5.0) * s + amrex::Real(3) * sp1) / amrex::Real(6.0)"),
    "negative WENO7 unnormalized weight": (
        HEADERS[0], "amrex::Real w4 = g4 / ( (eps + b4) * (eps + b4) );",
        "amrex::Real w4 = -g4 / ( (eps + b4) * (eps + b4) );"),
    "WENO5 normalizer drops a weight": (
        HEADERS[0], "amrex::Real wsum = w1 + w2 + w3;",
        "amrex::Real wsum = w1 + w2;"),
    "negative WENO-MZQ3 unnormalized weight": (
        HEADERS[1], "amrex::Real a3 = g3 * ( amrex::Real(1) +",
        "amrex::Real a3 = -g3 * ( amrex::Real(1) +"),
    'WENO3 return subtracts second candidate': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v1 - w2 * v2) / (amrex::Real(2) * wsum) );'),
    'WENO3 return duplicates second candidate': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v2 + w2 * v2) / (amrex::Real(2) * wsum) );'),
    'WENO3 return drops second candidate': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v1) / (amrex::Real(2) * wsum) );'),
    'WENO3 return divides by wrong normalizer': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * (wsum + w1)) );'),
    'WENO-MZQ3 return reverses correction': ('Source/Utils/ERF_Interpolation_WENO_Z.H',
         '(v3 - g1 * v1 - g2 * v2)',
         '(v3 + g1 * v1 - g2 * v2)'),
    'WENO3 weight is updated after initialization': ('Source/Utils/ERF_Interpolation_WENO.H',
         'amrex::Real wsum = w1 + w2;',
         'w1 *= -1;\n        amrex::Real wsum = w1 + w2;'),
}


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="erf-weno-candidate-mutants-"))
    originals = {header: (ERF / header).read_text() for header in HEADERS}
    print(f"mutants in {work}\n")
    unexpected = 0
    for i, (name, edit) in enumerate(MUTANTS.items()):
        root = work / f"m{i}"
        for header in HEADERS:
            target = root / header
            target.parent.mkdir(parents=True, exist_ok=True)
            text = originals[header]
            if edit and header == edit[0]:
                old, new = edit[1:]
                found = text.count(old)
                if found != 1:
                    raise AssertionError(f"{name}: pattern found {found}x in {header}, wanted 1")
                text = text.replace(old, new, 1)
            target.write_text(text)
        result = subprocess.run([sys.executable, str(VERIFY), str(root)],
                                capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout.strip() or "check_weno_candidates.py skipped (exit 77)")
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
