#!/usr/bin/env python3
"""Check ERF's source-level scalar advection flux and mapped divergence.

The checker parses the live scalar flux builder, component adapter, and divergence
kernel. It checks the directional interpolation/flux routing and uses Z3 real
arithmetic to prove that the positive-Jacobian branch is the negative mapped
finite-volume divergence of the selected face fluxes. The nonpositive-Jacobian
branch must write zero.

usage: check_scalar_flux_divergence.py [ERF_ROOT]
exit:  0 all checks pass, 1 a check fails, 77 z3 is not importable (CTest skip)
"""
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

try:
    import z3
except ImportError:
    print("check_scalar_flux_divergence.py: z3-solver is not importable; skipping (exit 77)")
    sys.exit(77)

SCALAR_H = "Source/Advection/ERF_AdvectionSrcForScalars.H"
STATE_CPP = "Source/Advection/ERF_AdvectionSrcForState.cpp"
failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def compact(text):
    return re.sub(r"\s+", "", text)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def brace_block(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start + 1:i]
    raise ValueError("unbalanced braces in advection source")


def function_body(text, name):
    match = re.search(rf"\b{name}\s*\([^)]*\)\s*\{{", text, flags=re.S)
    if not match:
        raise ValueError(f"{name}() definition not found")
    return brace_block(text, match.end() - 1)


def call_args(text, name):
    matches = list(re.finditer(rf"\b{name}\s*\(([^)]*)\)\s*;", text, flags=re.S))
    if len(matches) != 1:
        raise ValueError(f"expected one {name}() call, found {len(matches)}")
    return [compact(arg) for arg in matches[0].group(1).split(",")]


def number(value):
    fraction = Fraction(value)
    return z3.RealVal(f"{fraction.numerator}/{fraction.denominator}")


def replace_metric_reads(expression):
    expected = {
        ("detJ", "i,j,k"): "J",
        ("mf_mx", "i,j,0"): "mx",
        ("mf_my", "i,j,0"): "my",
    }

    def replace(match):
        key = (match.group(1), compact(match.group(2)))
        if key not in expected:
            raise ValueError(f"unexpected metric read: {match.group(0)}")
        return expected[key]

    return re.sub(r"\b(detJ|mf_mx|mf_my)\s*\(([^)]*)\)", replace, expression)


def replace_flux_reads(expression):
    expected = {
        (0, "i+1,j,k,flux_comp"): "Fx_hi",
        (0, "i,j,k,flux_comp"): "Fx_lo",
        (1, "i,j+1,k,flux_comp"): "Fy_hi",
        (1, "i,j,k,flux_comp"): "Fy_lo",
        (2, "i,j,k+1,flux_comp"): "Fz_hi",
        (2, "i,j,k,flux_comp"): "Fz_lo",
    }
    seen = Counter()

    def replace(match):
        args = [compact(arg) for arg in match.group(2).split(",")]
        if len(args) != 4:
            raise ValueError(f"unexpected face-flux read: {match.group(0)}")
        key = (int(match.group(1)), ",".join(args))
        if key not in expected:
            raise ValueError(f"unexpected face-flux read: {match.group(0)}")
        seen[key] += 1
        return expected[key]

    translated = re.sub(r"\bflx_arr\s*\[\s*(\d+)\s*\]\s*\(([^)]*)\)", replace, expression)
    if any(count != 1 for count in seen.values()) or len(seen) != 6:
        raise ValueError(f"expected each low/high face flux once, found {dict(seen)}")
    return translated


def to_z3(expression, env, flux_reads=False):
    expr = expression.replace("amrex::Real", "")
    if flux_reads:
        expr = replace_flux_reads(expr)
    expr = replace_metric_reads(expr)
    if re.search(r"[^\w\s.+\-*/()]", expr):
        raise ValueError(f"unsupported C++ arithmetic expression: {expression}")
    expr = re.sub(r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?)", r"_N('\1')", expr)
    return eval(expr, {"__builtins__": {}}, {**env, "_N": number})


def find_assignment(block):
    matches = list(re.finditer(r"\brhs\s*\(([^)]*)\)\s*=\s*(.*?);", block, flags=re.S))
    if len(matches) != 1:
        raise ValueError(f"expected one rhs assignment in branch, found {len(matches)}")
    lhs = f"rhs({matches[0].group(1)})"
    return compact(lhs), matches[0].group(2).strip()


def check_flux_builder(body):
    check(bool(re.search(r"InterpType_H\s+interp_h\s*\(\s*scalar\s*,\s*horiz_upw_frac\s*\)", body)),
          "horizontal interpolation policy uses the horizontal upwind fraction")
    check(bool(re.search(r"InterpType_V\s+interp_v\s*\(\s*scalar\s*,\s*vert_upw_frac\s*\)", body)),
          "vertical interpolation policy uses the vertical upwind fraction")

    directions = (
        ("X", "interp_h", "interpx", "avg_xmom", 0, "xbx"),
        ("Y", "interp_h", "interpy", "avg_ymom", 1, "ybx"),
        ("Z", "interp_v", "interpz", "avg_zmom", 2, "zbx"),
    )
    for axis, interp, face_value, velocity, dim, box in directions:
        call = (rf"\b{interp}\.InterpolateIn{axis}\s*\(\s*i\s*,\s*j\s*,\s*k\s*,\s*"
                rf"scalar_comp\s*,\s*{face_value}\s*,\s*{velocity}\s*\(\s*i\s*,\s*j\s*,\s*k\s*\)\s*\)\s*;")
        assign = (rf"flx_arr\s*\[\s*{dim}\s*\]\s*\(\s*i\s*,\s*j\s*,\s*k\s*,\s*flux_comp\s*\)\s*=\s*"
                  rf"{velocity}\s*\(\s*i\s*,\s*j\s*,\s*k\s*\)\s*\*\s*{face_value}\s*;")
        box_def = rf"const\s+amrex::Box\s+{box}\s*=\s*amrex::surroundingNodes\s*\(\s*bx\s*,\s*{dim}\s*\)\s*;"
        check(bool(re.search(call, body)),
              f"{axis}: interpolates the scalar with its matching face-normal mass flux")
        check(bool(re.search(assign, body)),
              f"{axis}: writes mass flux times reconstructed scalar to the matching face component")
        check(bool(re.search(box_def, body)),
              f"{axis}: computes fluxes on the {axis}-nodal box")


def check_component_adapter(body):
    check(bool(re.search(r"const\s+int\s+cons_comp\s*=\s*icomp\s*\+\s*n\s*;", body)),
          "scalar loop maps n to conserved component icomp+n")
    check(bool(re.search(r"const\s+int\s+scalar_comp\s*=\s*cons_comp\s*-\s*1\s*;", body)),
          "primitive scalar source component is one below its conserved component")
    check(bool(re.search(r"const\s+int\s+flux_comp\s*=\s*cons_comp\s*;", body)),
          "face flux component matches the conserved destination component")
    check(call_args(body, "BuildScalarAdvectionFluxes") == [
        "bx", "cell_prim", "scalar_comp", "flx_arr", "flux_comp", "avg_xmom", "avg_ymom",
        "avg_zmom", "horiz_adv_type", "vert_adv_type", "horiz_upw_frac", "vert_upw_frac"],
        "flux builder receives the primitive source component, selected flux component, and all three velocities")
    check(call_args(body, "ApplyScalarAdvectionFluxDivergence") == [
        "bx", "flx_arr", "flux_comp", "advectionSrc", "cons_comp", "detJ", "cellSizeInv", "mf_mx", "mf_my"],
        "divergence reads that flux component and writes the matching conserved component")


def check_divergence(body):
    for alias, dim in (("dxInv", 0), ("dyInv", 1), ("dzInv", 2)):
        pattern = rf"const\s+auto\s+{alias}\s*=\s*cellSizeInv\s*\[\s*{dim}\s*\]\s*;"
        check(bool(re.search(pattern, body)), f"{alias} uses cellSizeInv[{dim}]")

    if_match = re.search(r"\bif\s*\((.*?)\)\s*\{", body, flags=re.S)
    if not if_match:
        raise ValueError("positive-Jacobian branch not found")
    check(compact(if_match.group(1)) == "detJ(i,j,k)>zero",
          "mapped divergence runs only when the cell Jacobian is positive")
    positive = brace_block(body, if_match.end() - 1)
    else_match = re.search(r"\belse\s*\{", body[if_match.end():])
    if not else_match:
        raise ValueError("nonpositive-Jacobian branch not found")
    else_open = if_match.end() + else_match.end() - 1
    nonpositive = brace_block(body, else_open)

    lhs, rhs = find_assignment(positive)
    check(lhs == "rhs(i,j,k,rhs_comp)", "positive-Jacobian branch writes the requested cell and component")
    else_lhs, else_rhs = find_assignment(nonpositive)
    check(else_lhs == lhs, "nonpositive-Jacobian branch writes the same cell and component")
    check(compact(else_rhs) == "zero", "nonpositive-Jacobian branch writes zero")

    env = {
        "zero": number("0"), "one": number("1"),
        "J": z3.Real("J"), "mx": z3.Real("mx"), "my": z3.Real("my"),
        "dxInv": z3.Real("dxInv"), "dyInv": z3.Real("dyInv"), "dzInv": z3.Real("dzInv"),
        "Fx_hi": z3.Real("Fx_hi"), "Fx_lo": z3.Real("Fx_lo"),
        "Fy_hi": z3.Real("Fy_hi"), "Fy_lo": z3.Real("Fy_lo"),
        "Fz_hi": z3.Real("Fz_hi"), "Fz_lo": z3.Real("Fz_lo"),
    }
    base_env = dict(env)
    for name, expected in (
        ("invdetJ", "one / detJ(i,j,k)"),
        ("mfsq", "mf_mx(i,j,0) * mf_my(i,j,0)"),
    ):
        match = re.search(rf"\bconst\s+amrex::Real\s+{name}\s*=\s*(.*?);", positive, flags=re.S)
        if not match:
            raise ValueError(f"{name} local not found")
        actual_term = to_z3(match.group(1), base_env)
        expected_term = to_z3(expected, base_env)
        check(z3.simplify(actual_term == expected_term) == z3.BoolVal(True),
              f"{name} equals its source-level metric expression")
        env[name] = actual_term

    actual = to_z3(rhs, env, flux_reads=True)
    div = ((env["Fx_hi"] - env["Fx_lo"]) * env["dxInv"] +
           (env["Fy_hi"] - env["Fy_lo"]) * env["dyInv"] +
           (env["Fz_hi"] - env["Fz_lo"]) * env["dzInv"])
    expected = -env["invdetJ"] * env["mfsq"] * div
    solver = z3.Solver()
    solver.add(env["J"] > 0, actual != expected)
    result = solver.check()
    check(result == z3.unsat,
          "Z3 proves the positive-Jacobian RHS is the negative mapped x/y/z face-flux divergence")
    if result == z3.sat:
        print(f"       counterexample: {solver.model()}")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    scalar = strip_comments((root / SCALAR_H).read_text())
    state = strip_comments((root / STATE_CPP).read_text())
    check_flux_builder(function_body(scalar, "BuildScalarAdvectionFluxesWithPolicy"))
    check_component_adapter(function_body(state, "AdvectionSrcForScalars"))
    check_divergence(function_body(scalar, "ApplyScalarAdvectionFluxDivergence"))
    print(f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, NameError, SyntaxError) as exc:
        print(f"FAIL checker could not parse source: {exc}")
        sys.exit(1)
