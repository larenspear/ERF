#!/usr/bin/env python3
"""Break ERF_Interpolation_UPW.H in realistic ways and confirm check_upw_stencils.py objects.

Each mutant is written to its own copy of the two headers the checker reads; the real
tree is never touched.  Expect `passes` for the baseline and `CAUGHT` for every mutant.

usage: check_upw_stencils_selftest.py [WORK_DIR]     default: a fresh temporary directory
exit:  0 every outcome as expected, 1 otherwise, 77 z3 is not importable (CTest skip)
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_upw_stencils.py"

R5 = "upw * g3 * (d3 - amrex::Real(5.0)*d2 + amrex::Real(10.0)*d1)"
R5ALL = "upw * g3_5_6 * (d3 - amrex::Real(5.0)*d2 + amrex::Real(10.0)*d1)"
C6_TAIL = ("\n\nprivate:\n    amrex::Array4<const amrex::Real> m_phi;   // Quantity to interpolate\n"
           "    static constexpr amrex::Real g1=(amrex::Real(37.0)/amrex::Real(60.0));\n"
           "    static constexpr amrex::Real g2=(amrex::Real(2)/amrex::Real(15.0));\n"
           "    static constexpr amrex::Real g3=(amrex::Real(1)/amrex::Real(60.0));\n};\n\n/**\n"
           " * Interpolation operators used for all central")
U5_Y = ("amrex::Real sm3 = m_phi(i  , j-{n}, k  , qty_index);\n\n        // Upwinding flags\n"
        "        if (upw_lo != amrex::Real(0)) upw_lo = (upw_lo > 0) ? amrex::Real(1) : -amrex::Real(1);\n\n"
        "        // Add blending\n        upw_lo *= m_upw_frac;\n\n        // Interpolate lo\n"
        "        val_lo = Evaluate(sp2,sp1,s,sm1,sm2,sm3,upw_lo);")
U3_Z = ("val_lo = Evaluate({args});\n    }}\n\n    AMREX_GPU_DEVICE\n    AMREX_FORCE_INLINE\n"
        "    amrex::Real\n    Evaluate (const amrex::Real& sp1,")
U5_G1 = ("g1=(amrex::Real({n}.0)/amrex::Real(60.0));\n    static constexpr amrex::Real g2=(amrex::Real(2)/amrex::Real(15.0));\n"
         "    static constexpr amrex::Real g3=(amrex::Real(1)/amrex::Real(60.0));\n}};\n\n/**\n"
         " * Interpolation operators used for 6th order centered scheme")

# name -> list of (old, new, expected occurrences of old)
MUTANTS = {
    "baseline (no change)": [],
    "UPWIND5 InterpolateInY: sm3 read at j-2 instead of j-3":
        [(U5_Y.format(n=3), U5_Y.format(n=2), 1)],
    "UPWIND5 coefficient g1 = 38/60 (one copy only)":
        [(U5_G1.format(n=37), U5_G1.format(n=38), 1)],
    "3rd/4th-order coefficients 7/12,1/12 -> 8/12,2/12 in ALL copies (weights still sum to 1)":
        [("amrex::Real(7.0)/amrex::Real(12.0)", "amrex::Real(8.0)/amrex::Real(12.0)", 3),
         ("amrex::Real(1)/amrex::Real(12.0)", "amrex::Real(2)/amrex::Real(12.0)", 3)],
    "UPWIND3 InterpolateInZ: Evaluate() args swapped (s,sp1,...)":
        [(U3_Z.format(args="sp1,s,sm1,sm2,upw_lo"), U3_Z.format(args="s,sp1,sm1,sm2,upw_lo"), 1)],
    "UPWIND3 upwind term sign flipped (upwind becomes downwind)":
        [("return ( g1*a1 - g2*a2 + upw * g2 * (d2 - three*d1) );",
          "return ( g1*a1 - g2*a2 - upw * g2 * (d2 - three*d1) );", 1)],
    "UPWIND5 upwind term doubled in BOTH copies (order unchanged)":
        [(R5, "amrex::Real(2.0) * " + R5, 1), (R5ALL, "amrex::Real(2.0) * " + R5ALL, 1)],
    "CENTERED6 GetUpwindCellNumber() returns 2 instead of 3":
        [("int GetUpwindCellNumber () const {return 3;}" + C6_TAIL,
          "int GetUpwindCellNumber () const {return 2;}" + C6_TAIL, 1)],
    "UPWINDALL Upwind_5th branch drops the a3 term":
        [("iv = g1_5_6*a1 - g2_5_6*a2 + g3_5_6*a3 - upw * g3_5_6 * (d3",
          "iv = g1_5_6*a1 - g2_5_6*a2 - upw * g3_5_6 * (d3", 1)],
    "UPWINDALL Centered_2nd inline early return reads k+1 instead of k-1":
        [("val_lo = myhalf * ( m_phi(i,j,k,qty_index) + m_phi(i,j,k-1,qty_index) );",
          "val_lo = myhalf * ( m_phi(i,j,k,qty_index) + m_phi(i,j,k+1,qty_index) );", 1)],
}


def main():
    work =Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="erf-upw-mutants-"))
    hdr = (ERF / "Source/Utils/ERF_Interpolation_UPW.H").read_text()
    print(f"mutants in {work}\n")
    bad = 0
    for i, (name, edits) in enumerate(MUTANTS.items()):
        root = work / f"m{i}"
        (root / "Source/Utils").mkdir(parents=True, exist_ok=True)
        shutil.copy(ERF / "Source/ERF_NumericalConstants.H", root / "Source/ERF_NumericalConstants.H")
        text = hdr
        for old, new, n in edits:
            assert text.count(old) == n, f"{name}: pattern found {text.count(old)}x, wanted {n}"
            text = text.replace(old, new)
        (root / "Source/Utils/ERF_Interpolation_UPW.H").write_text(text)
        r = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        if r.returncode == 77:                  # the checker skipped itself: this interpreter has no z3
            print(r.stdout.strip() or "check_upw_stencils.py skipped (exit 77)")
            return 77
        fails =[l.strip() for l in r.stdout.splitlines() if l.strip().startswith("FAIL")]
        if r.returncode and not fails:
            fails = [(r.stderr.strip().splitlines() or ["crash"])[-1]]
        caught = bool(r.returncode)
        bad += caught == (not edits)                       # baseline must pass, mutants must fail
        print(f"[{'CAUGHT' if caught else 'passes'}] {name}")
        for f in fails[:3]:
            print(f"           {f}")
    print(f"\n{bad} unexpected outcome(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
