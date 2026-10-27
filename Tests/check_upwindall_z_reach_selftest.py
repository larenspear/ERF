#!/usr/bin/env python3
"""Inject UPWINDALL Z-reach bugs and require the source checker to catch them."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_upwindall_z_reach.py"
HEADER = "Source/Utils/ERF_Interpolation_UPW.H"
GATE = "adv_type == AdvType::Upwind_5th || adv_type == AdvType::Centered_6th"

# name -> (old text, new text, occurrence index in UPWINDALL, total old occurrences)
MUTANTS = {
    "baseline (no change)": None,
    "sp2 gate reads for Upwind_3rd": (
        GATE, "adv_type == AdvType::Upwind_3rd || adv_type == AdvType::Upwind_5th || "
              "adv_type == AdvType::Centered_6th", 0, 2),
    "sm3 gate misses Centered_6th": (
        GATE, "adv_type == AdvType::Upwind_5th", 1, 2),
    "sp2 true arm uses k+1": (
        "k+2, qty_index): amrex::Real(0);", "k+1, qty_index): amrex::Real(0);", 0, 1),
    "sm3 true arm uses k-2": (
        "k-3, qty_index) : amrex::Real(0);", "k-2, qty_index) : amrex::Real(0);", 0, 1),
    "sp2 false arm is nonzero": (
        ": amrex::Real(0);", ": amrex::Real(1);", 0, 2),
    "common sm2 read is too shallow": (
        "sm2 = m_phi(i  , j  , k-2, qty_index);",
        "sm2 = m_phi(i  , j  , k-1, qty_index);", 0, 1),
    "Centered_2nd early return also catches Centered_4th": (
        "if (adv_type == AdvType::Centered_2nd)",
        "if (adv_type == AdvType::Centered_2nd || adv_type == AdvType::Centered_4th)", 0, 2),
    "Centered_2nd early return reads the wrong cell": (
        "m_phi(i,j,k-1,qty_index)", "m_phi(i,j,k,qty_index)", 0, 1),
}


def replace_in_upwindall(text, edit, name):
    old, new, occurrence, expected_count = edit
    match = re.search(r"^struct UPWINDALL\s*\n\{", text, flags=re.M)
    if not match:
        raise AssertionError("UPWINDALL struct not found in mutation source")
    end = re.search(r"^\};", text[match.end():], flags=re.M)
    if not end:
        raise AssertionError("UPWINDALL struct end not found in mutation source")
    body_start = match.end()
    body_end = body_start + end.start()
    body = text[body_start:body_end]
    count = body.count(old)
    if count != expected_count:
        raise AssertionError(f"{name}: pattern found {count}x in UPWINDALL, wanted {expected_count}")
    starts = [m.start() for m in re.finditer(re.escape(old), body)]
    pos = starts[occurrence]
    body = body[:pos] + new + body[pos + len(old):]
    return text[:body_start] + body + text[body_end:]


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="erf-upwindall-z-reach-mutants-"))
    original = (ERF / HEADER).read_text()
    print(f"mutants in {work}\n")
    unexpected = 0
    for i, (name, edit) in enumerate(MUTANTS.items()):
        root = work / f"m{i}"
        target = root / HEADER
        target.parent.mkdir(parents=True, exist_ok=True)
        source = original if edit is None else replace_in_upwindall(original, edit, name)
        target.write_text(source)
        result = subprocess.run([sys.executable, str(VERIFY), str(root)],
                                capture_output=True, text=True)
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
