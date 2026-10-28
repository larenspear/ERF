#!/usr/bin/env python3
"""Check ERF's density-advection face mass fluxes and source branches.

The checker reads AdvectionSrcForRho() from the live C++ source. It checks the
three area/metric-weighted face mass fluxes and their averaged-momentum copies,
the fixed-density zero source, and the positive-Jacobian mapped divergence. Z3
proves the latter is the negative x/y/z finite-volume divergence of component-0
mass fluxes.

usage: check_rho_advection.py [ERF_ROOT]
exit:  0 all checks pass, 1 a check fails, 77 z3 is not importable (CTest skip)
"""
import re
import sys
from pathlib import Path

from advection_source import Source, LAMBDA, checked_constants

from check_scalar_flux_divergence import (
    check, compact, failures, function_body, number, strip_comments, to_z3, z3,
)

STATE_CPP = "Source/Advection/ERF_AdvectionSrcForState.cpp"


def parallel_body(body, box):
    match = re.search(rf"\bParallelFor\s*\(\s*{box}\s*,", body)
    if not match:
        raise ValueError(f"ParallelFor({box}, ...) not found")
    return _lambda_block(body, match.start())


def _lambda_block(text, start):
    open_brace = text.find("{", start)
    if open_brace < 0:
        raise ValueError("ParallelFor lambda body not found")
    depth = 0
    for i in range(open_brace, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[open_brace + 1:i]
    raise ValueError("unbalanced ParallelFor lambda braces")


def check_face_mass_fluxes(body):
    axes = (
        ("X", "xbx", 0, "avg_xmom",
         "(flx_arr[0])(i,j,k,0)=ax_arr(i,j,k)*rho_u(i,j,k)/mf_uy(i,j,0);"),
        ("Y", "ybx", 1, "avg_ymom",
         "(flx_arr[1])(i,j,k,0)=ay_arr(i,j,k)*rho_v(i,j,k)/mf_vx(i,j,0);"),
        ("Z", "zbx", 2, "avg_zmom",
         "(flx_arr[2])(i,j,k,0)=az_arr(i,j,k)*Omega(i,j,k)/mfsq;"),
    )
    for axis, box, dim, avg, flux in axes:
        box_pattern = rf"const\s+Box\s+{box}\s*=\s*surroundingNodes\s*\(\s*bx\s*,\s*{dim}\s*\)\s*;"
        check(bool(re.search(box_pattern, body)), f"{axis}: mass flux is computed on the matching nodal box")
        block = compact(parallel_body(body, box))
        avg_copy = f"{avg}(i,j,k)=(flx_arr[{dim}])(i,j,k,0);"
        check(flux in block,
              f"{axis}: component-0 face mass flux uses the expected area and metric factors")
        check(avg_copy in block, f"{axis}: averaged momentum is copied from the face mass flux")

    check(bool(re.search(
        r"Real\s+mfsq\s*=\s*mf_mx\s*\(\s*i\s*,\s*j\s*,\s*0\s*\)\s*\*\s*"
        r"mf_my\s*\(\s*i\s*,\s*j\s*,\s*0\s*\)\s*;", body)),
        "vertical mass flux uses the product of the two horizontal map factors")


def find_assignment(block):
    matches = list(re.finditer(r"\badvectionSrc\s*\(([^)]*)\)\s*=\s*(.*?);", block, flags=re.S))
    if len(matches) != 1:
        raise ValueError(f"expected one advectionSrc assignment in branch, found {len(matches)}")
    return compact(f"advectionSrc({matches[0].group(1)})"), matches[0].group(2).strip()


def check_source_branches(body):
    fixed_if = re.search(r"\bif\s*\(\s*fixed_rho\s*\)\s*\{", body)
    if not fixed_if:
        raise ValueError("fixed_rho branch not found")
    fixed_block = _lambda_block(body, fixed_if.end() - 1)
    fixed_lhs, fixed_rhs = find_assignment(fixed_block)
    check(fixed_lhs == "advectionSrc(i,j,k,0)" and compact(fixed_rhs) == "zero",
          "fixed_rho branch writes zero tendency to density component 0")

    else_match = re.search(r"\belse\s*\{", body[fixed_if.end():])
    if not else_match:
        raise ValueError("advecting-density branch not found")
    else_open = fixed_if.end() + else_match.end() - 1
    advecting = _lambda_block(body, else_open)
    jac_if = re.search(r"\bif\s*\((.*?)\)\s*\{", advecting, flags=re.S)
    if not jac_if:
        raise ValueError("positive-Jacobian branch not found")
    check(compact(jac_if.group(1)) == "detJ(i,j,k)>zero",
          "density divergence runs only when the cell Jacobian is positive")
    positive = _lambda_block(advecting, jac_if.end() - 1)
    inner_else = re.search(r"\belse\s*\{", advecting[jac_if.end():])
    if not inner_else:
        raise ValueError("nonpositive-Jacobian branch not found")
    inner_else_open = jac_if.end() + inner_else.end() - 1
    nonpositive = _lambda_block(advecting, inner_else_open)
    lhs, rhs = find_assignment(positive)
    else_lhs, else_rhs = find_assignment(nonpositive)
    check(lhs == "advectionSrc(i,j,k,0)",
          "positive-Jacobian branch writes density component 0 at the cell")
    check(else_lhs == lhs, "nonpositive-Jacobian branch writes the same density component")
    check(compact(else_rhs) == "zero", "nonpositive-Jacobian branch writes zero")

    aliases_match = re.search(r"auto\s+dxInv\s*=.*?;", body, flags=re.S)
    if not aliases_match:
        raise ValueError("inverse-spacing aliases not found")
    aliases = compact(aliases_match.group(0))
    check(aliases == "autodxInv=cellSizeInv[0],dyInv=cellSizeInv[1],dzInv=cellSizeInv[2];",
          "density divergence uses the matching inverse spacing in x, y, and z")
    mfsq_match = re.search(r"\bReal\s+mfsq\s*=\s*(.*?);", positive, flags=re.S)
    if not mfsq_match:
        raise ValueError("positive-branch mfsq local not found")

    env = {
        "zero": number("0"), "J": z3.Real("J"), "mx": z3.Real("mx"), "my": z3.Real("my"),
        "dxInv": z3.Real("dxInv"), "dyInv": z3.Real("dyInv"), "dzInv": z3.Real("dzInv"),
        "Fx_hi": z3.Real("Fx_hi"), "Fx_lo": z3.Real("Fx_lo"),
        "Fy_hi": z3.Real("Fy_hi"), "Fy_lo": z3.Real("Fy_lo"),
        "Fz_hi": z3.Real("Fz_hi"), "Fz_lo": z3.Real("Fz_lo"),
    }
    mfsq = to_z3(mfsq_match.group(1), env)
    check(z3.simplify(mfsq == env["mx"] * env["my"]) == z3.BoolVal(True),
          "positive-branch mfsq equals the product of horizontal map factors")
    env["mfsq"] = mfsq

    actual = to_z3(rhs, env, flux_reads=True, flux_component="0")
    div = ((env["Fx_hi"] - env["Fx_lo"]) * env["dxInv"] +
           (env["Fy_hi"] - env["Fy_lo"]) * env["dyInv"] +
           (env["Fz_hi"] - env["Fz_lo"]) * env["dzInv"])
    expected = -mfsq / env["J"] * div
    solver = z3.Solver()
    solver.add(env["J"] > 0, actual != expected)
    result = solver.check()
    check(result == z3.unsat,
          "Z3 proves the density source is the negative mapped divergence of component-0 mass fluxes")
    if result == z3.sat:
        print(f"       counterexample: {solver.model()}")


def validate_statements(body):
    """Consume the full density routine; arithmetic/routing is checked below."""
    source = Source(body)
    source.take(r'BL_PROFILE_VAR\("AdvectionSrcForRho",\s*AdvectionSrcForRho\)\s*;')
    source.take(r"auto\s+dxInv\s*=\s*[^;{}]+;")
    for box in ('xbx', 'ybx', 'zbx'):
        source.take(rf"const\s+Box\s+{box}\s*=\s*surroundingNodes\([^()]+\)\s*;")
    for dim, axis in enumerate('xyz'):
        source.take(rf"ParallelFor\s*\(\s*{axis}bx\s*," + LAMBDA)
        if dim == 2:
            source.take(r"Real\s+mfsq\s*=\s*[^;{}]+;")
        source.take(rf"\(flx_arr\[{dim}\]\)\([^()]+\)\s*=\s*[^;{{}}]+;")
        source.take(rf"avg_{axis}mom\([^()]+\)\s*=\s*[^;{{}}]+;")
        source.take(r"\}\s*\)\s*;")
    source.take(r"if\s*\(\s*fixed_rho\s*\)\s*\{")
    source.take(r"ParallelFor\s*\(\s*bx\s*," + LAMBDA)
    source.take(r"advectionSrc\([^()]+\)\s*=\s*[^;{}]+;")
    source.take(r"\}\s*\)\s*;\s*\}\s*else\s*\{")
    source.take(r"ParallelFor\s*\(\s*bx\s*," + LAMBDA)
    source.take(r"if\s*\([^{}]+\)\s*\{")
    source.take(r"Real\s+mfsq\s*=\s*[^;{}]+;")
    source.take(r"advectionSrc\([^()]+\)\s*=\s*[^;{}]+;")
    source.take(r"\}\s*else\s*\{")
    source.take(r"advectionSrc\([^()]+\)\s*=\s*[^;{}]+;")
    source.take(r"\}\s*\}\s*\)\s*;\s*\}")
    source.done()


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    checked_constants(root, {"zero": "0"})
    source = strip_comments((root / STATE_CPP).read_text())
    body = function_body(source, "AdvectionSrcForRho")
    validate_statements(body)
    check_face_mass_fluxes(body)
    check_source_branches(body)
    print(f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, NameError, SyntaxError) as exc:
        print(f"FAIL checker could not parse source: {exc}")
        sys.exit(1)
