#!/usr/bin/env python3
"""Check the inline centered-2nd momentum divergence on a stretched-z mesh.

The checker parses the three live ParallelFor lambdas. It proves the mapped
face-flux divergence and component metrics, checks the X/Y and Z inverse-z
spacing rules, and checks the top z-face flux special case. It does not prove
the flux reconstruction formulas or higher-order templated path.

usage: check_momentum_stretched_dz_fast_path.py [ERF_ROOT]
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

MOM_CPP = "Source/Advection/ERF_AdvectionSrcForMom_StretchedDz.cpp"
ARRAYS = {
    "mf_ux", "mf_uy", "mf_vx", "mf_vy", "mf_mx", "mf_my",
    "mf_ux_inv", "mf_uy_inv", "mf_vx_inv", "mf_vy_inv",
    "rho_u", "rho_v", "omega", "u", "v", "w", "dz_ptr",
}
array_reads = {}


def replace_array_reads(expression):
    def replace_bracket(match):
        name, index = match.groups()
        key = f"{name}[{compact(index)}]"
        if key not in array_reads:
            array_reads[key] = z3.Real(f"array_read_{len(array_reads)}")
        return array_reads[key].decl().name()

    def replace(match):
        name, args = match.groups()
        if name not in ARRAYS:
            raise ValueError(f"unexpected call in arithmetic expression: {match.group(0)}")
        key = f"{name}({compact(args)})"
        if key not in array_reads:
            array_reads[key] = z3.Real(f"array_read_{len(array_reads)}")
        return array_reads[key].decl().name()

    expression = re.sub(r"\b(dz_ptr)\s*\[([^\]]+)\]", replace_bracket, expression)
    return re.sub(r"\b([A-Za-z_]\w*)\s*\(([^()]*)\)", replace, expression)


def to_z3(expression, env):
    expr = replace_array_reads(compact(expression))
    if re.search(r"[^\w\s.+\-*/()]", expr):
        raise ValueError(f"unsupported C++ arithmetic expression: {expression}")
    expr = re.sub(r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?)", r"_N('\1')", expr)
    reads = {symbol.decl().name(): symbol for symbol in array_reads.values()}
    return eval(expr, {"__builtins__": {}}, {**env, **reads, "_N": number})


def prove_equal(actual, expected, description):
    solver = z3.Solver()
    solver.add(actual != expected)
    result = solver.check()
    check(result == z3.unsat, description)
    if result == z3.sat:
        print(f"       counterexample: {solver.model()}")


def parse_single_assignment(body, name, lhs_pattern=None):
    lhs = lhs_pattern or rf"\b{name}\b"
    matches = list(re.finditer(rf"{lhs}\s*=\s*(.*?);", body, flags=re.S))
    if len(matches) != 1:
        raise ValueError(f"expected one assignment to {name}, found {len(matches)}")
    return matches[0].group(1).strip()


def parse_select(expression, expected_condition, expected_true, expected_false, env, description):
    match = re.fullmatch(r"\((.*?)\)\?(.*?):(.*)", compact(expression))
    if not match:
        raise ValueError(f"{description}: expected a conditional expression")
    condition, true_expr, false_expr = match.groups()
    check(condition == expected_condition,
          f"{description}: uses condition {expected_condition}")
    prove_equal(to_z3(true_expr, env), to_z3(expected_true, env),
                f"{description}: true branch has the expected value")
    prove_equal(to_z3(false_expr, env), to_z3(expected_false, env),
                f"{description}: false branch has the expected value")

    int_condition = re.fullmatch(r"k==(-?\d+|[A-Za-z_]\w*)", condition)
    if not int_condition:
        raise ValueError(f"{description}: unsupported condition {condition}")
    value = int_condition.group(1)
    rhs = z3.IntVal(int(value)) if re.fullmatch(r"-?\d+", value) else z3.Int(value)
    return z3.If(z3.Int("k") == rhs, to_z3(true_expr, env), to_z3(false_expr, env))


def check_vertical_spacing(body, component, env):
    expression = parse_single_assignment(body, "dzInv", r"\bReal\s+dzInv")
    if component in ("X", "Y"):
        actual = to_z3(expression, env)
        expected = to_z3("one / dz_ptr[k]", env)
        prove_equal(actual, expected,
                    f"{component}: vertical inverse spacing is one over dz[k]")
        return actual

    return parse_select(
        expression, "k==0", "one / dz_ptr[k]", "two / (dz_ptr[k] + dz_ptr[k-1])",
        env, "Z: vertical inverse spacing")


def check_component(body, component, expected_metric, constants):
    rhs_name = {"X": "rho_u_rhs", "Y": "rho_v_rhs", "Z": "rho_w_rhs"}[component]
    momentum_component(body, rhs_name, stretched=True)
    flux_names = [f"{axis}flux_{edge}" for axis in "xyz" for edge in ("hi", "lo")]
    declared = re.findall(r"\bReal\s+([xyz]flux_(?:hi|lo))\s*=", body)
    check(Counter(declared) == Counter(flux_names),
          f"{component}: defines each low/high directional flux once")

    env = {
        **constants,
        "dxInv": z3.Real("dxInv"), "dyInv": z3.Real("dyInv"),
        **{name: z3.Real(name) for name in flux_names},
    }
    dz_inv = check_vertical_spacing(body, component, env)

    metric_expr = parse_single_assignment(body, "mfsq", r"\bReal\s+mfsq")
    actual_metric = to_z3(metric_expr, env)
    expected_metric_term = to_z3(expected_metric, env)
    prove_equal(actual_metric, expected_metric_term,
                f"{component}: inline map factor matches the component metric")

    source_expr = parse_single_assignment(body, "advectionSrc", r"\bReal\s+advectionSrc")
    actual_source = to_z3(source_expr, {**env, "mfsq": actual_metric, "dzInv": dz_inv})
    rhs_expr = parse_single_assignment(
        body, f"{rhs_name}(i,j,k)", rf"\b{rhs_name}\s*\(\s*i\s*,\s*j\s*,\s*k\s*\)")
    actual_rhs = to_z3(rhs_expr, {**env, "mfsq": actual_metric,
                                  "dzInv": dz_inv, "advectionSrc": actual_source})
    expected_divergence = (
        (env["xflux_hi"] - env["xflux_lo"]) * env["dxInv"] * expected_metric_term +
        (env["yflux_hi"] - env["yflux_lo"]) * env["dyInv"] * expected_metric_term +
        (env["zflux_hi"] - env["zflux_lo"]) * dz_inv)
    prove_equal(actual_source, expected_divergence,
                f"{component}: Z3 proves the stretched-z mapped face-flux divergence")
    prove_equal(actual_rhs, -expected_divergence,
                f"{component}: RHS stores the negative stretched-z divergence")


def check_top_z_flux(z_body, constants):
    expression = parse_single_assignment(z_body, "zflux_hi", r"\bReal\s+zflux_hi")
    env = dict(constants)
    parse_select(
        expression, "k==hi_z_face", "omega(i,j,k) * w(i,j,k)",
        "(1/4) * (omega(i,j,k) + omega(i,j,k+1)) * (w(i,j,k) + w(i,j,k+1))",
        env, "Z high flux")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    constants = {name: number(value) for name, value in
                 checked_constants(root, {"one": "1", "two": "2", "fourth": "1/4"}).items()}
    text = strip_comments((root / MOM_CPP).read_text())
    body = function_body(text, "AdvectionSrcForMom_StretchedDz")
    branch_match = re.search(r"\bif\s*\(\s*(.*?)\s*\)\s*\{", body, flags=re.S)
    if not branch_match:
        raise ValueError("Centered_2nd inline path not found")
    condition = compact(branch_match.group(1))
    check(condition ==
          "horiz_adv_type==AdvType::Centered_2nd&&vert_adv_type==AdvType::Centered_2nd",
          "inline path is gated on centered second order in both directions")
    spacing_aliases(body[:branch_match.start()], 2)
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
