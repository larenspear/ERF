#!/usr/bin/env python3
"""Check the inline centered-2nd momentum operator in the constant-dz path.

The checker reads the three live ParallelFor lambdas, proves each RHS is the
negative mapped x/y/z face-flux divergence, checks component map factors, and
checks all staggered flux reconstructions and inverse-map initialization. It also
proves shared interior face consistency and volume-weighted cancellation, and
checks the top z-face flux special case. The higher-order templated path,
memory/halo validity, and floating-point rounding remain outside this proof.

usage: check_momentum_constant_dz_fast_path.py [ERF_ROOT]
exit:  0 all checks pass, 1 a check fails, 77 z3 is not importable (CTest skip)
"""
import re
import sys
from collections import Counter
from pathlib import Path

from advection_source import (
    checked_constants, momentum_component, momentum_lambdas as get_lambdas, spacing_aliases,
)

from check_scalar_flux_divergence import (
    brace_block, check, compact, failures, function_body, number, strip_comments, z3,
)
from momentum_centered import check_fluxes, check_setup, reconstructed_rhs
from momentum_balance import check_shared_faces

MOM_CPP = "Source/Advection/ERF_AdvectionSrcForMom_ConstantDz.cpp"
ARRAYS = {
    "mf_ux", "mf_uy", "mf_vx", "mf_vy", "mf_mx", "mf_my",
    "mf_ux_inv", "mf_uy_inv", "mf_vx_inv", "mf_vy_inv",
    "rho_u", "rho_v", "omega", "u", "v", "w",
}
array_reads = {}


def replace_array_reads(expression):
    def replace(match):
        name, args = match.groups()
        if name not in ARRAYS:
            raise ValueError(f"unexpected call in arithmetic expression: {match.group(0)}")
        key = f"{name}({compact(args)})"
        if key not in array_reads:
            array_reads[key] = z3.Real(f"array_read_{len(array_reads)}")
        return array_reads[key].decl().name()

    return re.sub(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)", replace, expression)


def to_z3(expression, env):
    expr = replace_array_reads(compact(expression))
    if re.search(r"[^\w\s.+\-*/()]", expr):
        raise ValueError(f"unsupported C++ arithmetic expression: {expression}")
    expr = re.sub(r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?)", r"_N('\1')", expr)
    reads = {symbol.decl().name(): symbol for symbol in array_reads.values()}
    return eval(expr, {"__builtins__": {}}, {**env, **reads, "_N": number})


def prove_equal(actual, expected, description, assumptions=()):
    solver = z3.Solver()
    solver.set(timeout=10000)
    solver.add(*assumptions, actual != expected)
    result = solver.check()
    check(result == z3.unsat, f'{description} [{result}]')
    if result == z3.sat:
        print(f"       counterexample: {solver.model()}")
    elif result == z3.unknown:
        print(f"       inconclusive: {solver.reason_unknown()}")


def parse_single_assignment(body, name, lhs_pattern=None):
    lhs = lhs_pattern or rf"\b{name}\b"
    matches = list(re.finditer(rf"{lhs}\s*=\s*(.*?);", body, flags=re.S))
    if len(matches) != 1:
        raise ValueError(f"expected one assignment to {name}, found {len(matches)}")
    return matches[0].group(1).strip()


def check_component(body, component, expected_metric, constants):
    rhs_name = {"X": "rho_u_rhs", "Y": "rho_v_rhs", "Z": "rho_w_rhs"}[component]
    momentum_component(body, rhs_name, stretched=False)
    actual_fluxes, expected_fluxes = check_fluxes(body, component, constants, to_z3, prove_equal)
    flux_names = [f"{axis}flux_{edge}" for axis in "xyz" for edge in ("hi", "lo")]
    declared = re.findall(r"\bReal\s+([xyz]flux_(?:hi|lo))\s*=", body)
    check(Counter(declared) == Counter(flux_names),
          f"{component}: defines each low/high directional flux once")

    env = {
        **constants,
        "dxInv": z3.Real("dxInv"), "dyInv": z3.Real("dyInv"), "dzInv": z3.Real("dzInv"),
        **{name: z3.Real(name) for name in flux_names},
    }
    metric_expr = parse_single_assignment(body, "mfsq", r"\bReal\s+mfsq")
    actual_metric = to_z3(metric_expr, env)
    expected_metric_term = to_z3(expected_metric, env)
    prove_equal(actual_metric, expected_metric_term,
                f"{component}: inline map factor matches the component metric")
    env["mfsq"] = actual_metric

    source_expr = parse_single_assignment(body, "advectionSrc", r"\bReal\s+advectionSrc")
    actual_source = to_z3(source_expr, env)
    rhs_expr = parse_single_assignment(
        body, f"{rhs_name}(i,j,k)", rf"\b{rhs_name}\s*\(\s*i\s*,\s*j\s*,\s*k\s*\)")
    rhs_env = {**env, "advectionSrc": actual_source}
    actual_rhs = to_z3(rhs_expr, rhs_env)
    expected_divergence = (
        (env["xflux_hi"] - env["xflux_lo"]) * env["dxInv"] * expected_metric_term +
        (env["yflux_hi"] - env["yflux_lo"]) * env["dyInv"] * expected_metric_term +
        (env["zflux_hi"] - env["zflux_lo"]) * env["dzInv"])
    prove_equal(actual_source, expected_divergence,
                f"{component}: Z3 proves the inline source is the mapped face-flux divergence")
    prove_equal(actual_rhs, -expected_divergence,
                f"{component}: RHS stores the negative mapped divergence")
    reconstructed_rhs(source_expr, rhs_expr, env, actual_fluxes, expected_fluxes,
                      expected_metric_term, component, to_z3, prove_equal)
    check_shared_faces(component, actual_fluxes, actual_rhs, env, expected_metric_term,
                       env['dzInv'], array_reads, to_z3, prove_equal)


def check_top_z_flux(z_body, constants):
    expression = parse_single_assignment(z_body, "zflux_hi", r"\bReal\s+zflux_hi")
    match = re.fullmatch(r"\s*\(\s*(k\s*==\s*hi_z_face)\s*\)\s*\?\s*(.*?)\s*:\s*(.*?)\s*", expression, re.S)
    if not match:
        raise ValueError("zflux_hi no longer has the expected top-face conditional")
    condition, top_expr, interior_expr = match.groups()
    check(compact(condition) == "k==hi_z_face",
          "Z high flux selects the wall value only at hi_z_face")

    env = dict(constants)
    actual_top = to_z3(top_expr, env)
    expected_top = to_z3("omega(i,j,k) * w(i,j,k)", env)
    prove_equal(actual_top, expected_top, "top Z flux is omega times w")

    actual_interior = to_z3(interior_expr, env)
    expected_interior = to_z3(
        "(1/4) * (omega(i,j,k) + omega(i,j,k+1)) * (w(i,j,k) + w(i,j,k+1))", env)
    prove_equal(actual_interior, expected_interior,
                "interior Z high flux averages omega and w across the face")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    constants = {name: number(value) for name, value in
                 checked_constants(root, {"one": "1", "fourth": "1/4"}).items()}
    text = strip_comments((root / MOM_CPP).read_text())
    body = function_body(text, "AdvectionSrcForMom_ConstantDz")
    branch_match = re.search(
        r"\bif\s*\(\s*(.*?)\s*\)\s*\{", body, flags=re.S)
    if not branch_match:
        raise ValueError("Centered_2nd inline path not found")
    branch_condition = compact(branch_match.group(1))
    check(branch_condition ==
          "horiz_adv_type==AdvType::Centered_2nd&&vert_adv_type==AdvType::Centered_2nd",
          "inline path is gated on centered second order in both directions")
    spacing_aliases(body[:branch_match.start()], 3)
    check_setup(body[:branch_match.start()], False, constants, to_z3, prove_equal)
    branch = brace_block(body, branch_match.end() - 1)
    x_body, y_body, z_body = get_lambdas(branch)

    check_component(x_body, "X", "mf_ux(i,j,0) * mf_uy(i,j,0)", constants)
    check_component(y_body, "Y", "mf_vx(i,j,0) * mf_vy(i,j,0)", constants)
    check_component(z_body, "Z", "mf_mx(i,j,0) * mf_my(i,j,0)", constants)
    check_top_z_flux(z_body, constants)
    print(f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, NameError, SyntaxError) as exc:
        print(f"FAIL checker could not parse source: {exc}")
        sys.exit(1)
