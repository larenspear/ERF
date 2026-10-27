#!/usr/bin/env python3
"""Inject six UPWIND3SL mutations and require the limiter checker to catch them.

Every mutation is written to a scratch copy of the two headers read by the
checker. The repository files are never modified.

usage: check_upw3sl_limiter_selftest.py [WORK_DIR]
exit:  0 every outcome as expected, 1 otherwise, 77 z3 is not importable
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_upw3sl_limiter.py"

PHI = "amrex::Real phi = amrex::max(amrex::Real(0), amrex::min(amrex::Real(2)*rmh, amrex::min(amrex::Real(2), K)));"
INNER = "amrex::min(amrex::Real(2)*rmh, amrex::min(amrex::Real(2), K))"
MUTANTS = {
    "baseline (no change)": [],
    "swap outer max arguments (NaN escapes)": [
        (PHI, f"amrex::Real phi = amrex::max({INNER}, amrex::Real(0));", 1)],
    "halve the Koren slope (smooth branch no longer equals UPWIND3)": [
        ("l2 + amrex::Real(2)*l2*rmh", "l2 + l2*rmh", 1)],
    "change 2r to 3r (violates Sweby bound)": [
        ("amrex::Real(2)*rmh", "amrex::Real(3)*rmh", 1)],
    "drop myhalf from the face correction (unbounded face value)": [
        ("myhalf * phi * (sm1 - sm2)", "phi * (sm1 - sm2)", 1)],
    "change one direction's opposite-face Evaluate arguments": [
        ("Evaluate(sm1,s  ,sp1)", "Evaluate(sm1,s  ,sm2)", 3)],
    "delete outer max (negative limiter values escape)": [
        (PHI, f"amrex::Real phi = {INNER};", 1)],
}


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="erf-upw3sl-mutants-"))
    hdr_path = ERF / "Source/Utils/ERF_Interpolation_UPW.H"
    const_path = ERF / "Source/ERF_NumericalConstants.H"
    original = hdr_path.read_text()
    print(f"mutants in {work}\n")
    unexpected = 0
    for i, (name, edits) in enumerate(MUTANTS.items()):
        root = work / f"m{i}"
        (root / "Source/Utils").mkdir(parents=True, exist_ok=True)
        shutil.copy(const_path, root / "Source/ERF_NumericalConstants.H")
        text = original
        for old, new, expected_count in edits:
            found = text.count(old)
            if found != expected_count:
                raise AssertionError(f"{name}: pattern found {found}x, wanted {expected_count}")
            # The direction mutation has three identical occurrences; mutate only
            # the first direction after verifying the exact expected count.
            text = text.replace(old, new, 1 if expected_count > 1 else -1)
        (root / "Source/Utils/ERF_Interpolation_UPW.H").write_text(text)
        result = subprocess.run([sys.executable, str(VERIFY), "--fast", str(root)],
                                capture_output=True, text=True)
        if result.returncode == 77:
            print(result.stdout.strip() or "check_upw3sl_limiter.py skipped (exit 77)")
            return 77
        failures = [line.strip() for line in result.stdout.splitlines()
                    if line.strip().startswith("FAIL")]
        if result.returncode and not failures:
            failures = [(result.stderr.strip().splitlines() or ["checker crashed"])[-1]]
        caught = bool(result.returncode)
        unexpected += caught == (not edits)
        print(f"[{'CAUGHT' if caught else 'passes'}] {name}")
        for failure in failures[:3]:
            print(f"           {failure}")
    print(f"\n{unexpected} unexpected outcome(s)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
