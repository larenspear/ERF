#!/usr/bin/env python3
"""Machine-check ERF's UPWIND3SL slope limiter against its C++ source.

The expression for Evaluate() and the left-face blend are parsed from
ERF_Interpolation_UPW.H. The same parsed expression is evaluated in exact-real
and IEEE binary64 models. `--fast` omits the two expensive end-to-end FP proofs
and is intended for the mutation self-test.

usage: check_upw3sl_limiter.py [--fast] [ERF_ROOT]
exit:  0 all checks pass, 1 a check failed, 77 z3 is not importable (CTest skip)
"""
import argparse
import re
import sys
import time
from fractions import Fraction
from pathlib import Path

try:
    import z3
except ImportError:
    print("check_upw3sl_limiter.py: the z3-solver package is not importable; skipping (exit 77)")
    sys.exit(77)

import check_upw_stencils as linear_checker

UPW_H = "Source/Utils/ERF_Interpolation_UPW.H"
CONST_H = "Source/ERF_NumericalConstants.H"
T0 = time.time()
FAILURES = []
NUMBER = re.compile(r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?)")


def stamp():
    return f"{time.time() - T0:6.1f}s"


def report_result(name, actual, expected):
    ok = actual == expected
    print(f"  {'ok  ' if ok else 'FAIL'} {name}  [{actual}, expected {expected}]  {stamp()}")
    if not ok:
        FAILURES.append(name)
    sys.stdout.flush()
    return ok


def report_bool(ok, name):
    return report_result(name, bool(ok), True)


def solver_result(constraints, timeout_ms=None):
    solver = z3.Solver()
    if timeout_ms:
        solver.set("timeout", timeout_ms)
    solver.add(*constraints)
    result = solver.check()
    return result, solver.model() if result == z3.sat else None


def validity(claim, assumptions=(), timeout_ms=None):
    return solver_result([*assumptions, z3.Not(claim)], timeout_ms)[0]


def real_number(value):
    f = Fraction(value)
    return z3.RealVal(f"{f.numerator}/{f.denominator}")


def fp_number(value, sort):
    # Python's decimal-to-float conversion and IEEE binary64 both round to
    # nearest-even; FPVal then preserves that already-rounded binary value.
    return z3.FPVal(float(value), sort)


def parse_constants(text, fp_mode, sort=None):
    make_number = (lambda x: fp_number(x, sort)) if fp_mode else real_number
    constants = {}
    pattern = r"constexpr\s+amrex::Real\s+(\w+)\s*=\s*amrex::Real\(\s*([-+0-9.eE]+)\s*\)\s*;"
    for m in re.finditer(pattern, text):
        constants[m.group(1)] = make_number(m.group(2))
    return constants


def min_max(fp_mode):
    if fp_mode:
        return (lambda a, b: z3.If(z3.fpLT(b, a), b, a),
                lambda a, b: z3.If(z3.fpLT(a, b), b, a))
    return (lambda a, b: z3.If(b < a, b, a),
            lambda a, b: z3.If(a < b, b, a))


def expression(expr, env, fp_mode, sort=None):
    """Replay the arithmetic expression with a restricted eval namespace."""
    e = expr.replace("amrex::min", "_min").replace("amrex::max", "_max")
    e = e.replace("amrex::Real", "")
    if re.search(r"[^\w\s.+\-*/(),]", e):
        raise ValueError(f"unsupported expression: {expr}")
    e = NUMBER.sub(lambda m: f"_N('{m.group(1)}')", e)
    make_number = (lambda x: fp_number(x, sort)) if fp_mode else real_number
    mn, mx = min_max(fp_mode)
    namespace = {**env, "_N": make_number, "_min": mn, "_max": mx}
    return eval(e, {"__builtins__": {}}, namespace)


def parse_evaluate(body):
    sig = re.search(r"Evaluate\s*\(([^)]*)\)\s*const\s*\{", body)
    if not sig:
        raise ValueError("UPWIND3SL::Evaluate signature was not found")
    params = []
    for p in sig.group(1).split(","):
        pm = re.fullmatch(r"\s*const\s+amrex::Real&\s*(\w+)\s*", p)
        if not pm:
            raise ValueError(f"unsupported Evaluate parameter: {p}")
        params.append(pm.group(1))
    block = linear_checker.brace_block(body, sig.end() - 1)
    constants = re.findall(
        r"static\s+constexpr\s+amrex::Real\s+(\w+)\s*=\s*(.+?);", body)
    assignments = re.findall(r"amrex::Real\s+(\w+)\s*=\s*([^;]+);", block)
    ret = re.search(r"return\s*(.+?);", block, flags=re.S)
    if not ret:
        raise ValueError("UPWIND3SL::Evaluate return expression was not found")
    names = [name for name, _ in assignments]
    if not {"rmh", "K", "phi"} <= set(names):
        raise ValueError(f"limiter locals missing from Evaluate: {names}")
    return params, constants, assignments, ret.group(1).strip()


def build_evaluate(parsed, constants0, values, fp_mode, sort=None, overrides=None):
    params, constants, assignments, ret = parsed
    env = dict(constants0)
    for name, val in (overrides or {}).items():
        if name not in {n for n, _ in constants}:
            raise ValueError(f"cannot override missing C++ constant {name}")
    for name, expr in constants:
        env[name] = expression(expr, env, fp_mode, sort)
        if overrides and name in overrides:
            env[name] = overrides[name]
    if not {"eps", "l1", "l2"} <= {n for n, _ in constants}:
        raise ValueError("UPWIND3SL constants eps, l1 and l2 were not all parsed")
    for name in params:
        if name not in values:
            raise ValueError(f"no model value supplied for Evaluate parameter {name}")
        env[name] = values[name]
    for name, expr in assignments:
        env[name] = expression(expr, env, fp_mode, sort)
    return expression(ret, env, fp_mode, sort), env


def build_phi_function(parsed, constants0, rmh, fp_mode, sort=None):
    _, constants, assignments, _ = parsed
    env = dict(constants0)
    for name, expr in constants:
        env[name] = expression(expr, env, fp_mode, sort)
    env["rmh"] = rmh
    for name, expr in assignments:
        if name == "rmh":
            continue
        env[name] = expression(expr, env, fp_mode, sort)
        if name == "phi":
            return env[name]
    raise ValueError("could not replay the phi assignment")


def parse_interpolations(body):
    out = {}
    for m in re.finditer(r"InterpolateIn([XYZ])\s*\(", body):
        direction = m.group(1)
        block = linear_checker.brace_block(body, body.index("{", m.end()))
        calls = [tuple(x.strip() for x in args.split(","))
                 for args in re.findall(r"\bEvaluate\s*\(([^)]*)\)", block)]
        blends = re.findall(r"\bval_lo\s*=\s*([^;]+);", block)
        if len(blends) != 1:
            raise ValueError(f"InterpolateIn{direction}: expected one val_lo assignment, got {len(blends)}")
        out[direction] = (calls, blends[0].strip(), block)
    return out


def make_blend(blend_text, fhi, flo, upw, fp_mode, sort=None):
    return expression(blend_text, {"Fhi": fhi, "Flo": flo, "upw_lo": upw}, fp_mode, sort)


def safe_finite(value):
    return z3.And(z3.Not(z3.fpIsNaN(value)), z3.Not(z3.fpIsInf(value)))


def run(root, fast):
    source = linear_checker.strip_comments((root / UPW_H).read_text())
    ctext = linear_checker.strip_comments((root / CONST_H).read_text())
    structs = linear_checker.split_structs(source)
    if "UPWIND3SL" not in structs or "UPWIND3" not in structs:
        raise ValueError("UPWIND3SL and UPWIND3 were not both found in the parsed header")
    body = structs["UPWIND3SL"][0]
    parsed = parse_evaluate(body)
    reals = parse_constants(ctext, fp_mode=False)
    s, sm1, sm2, sp1 = z3.Reals("s sm1 sm2 sp1")
    upw = z3.Real("upw")
    eps = real_number("1e-16")
    r = z3.Real("r")
    mn_real, mx_real = min_max(False)

    print("== structure")
    report_bool(parsed[0] == ["s", "sm1", "sm2"],
                "Evaluate parameters are s, sm1, sm2 in declaration order")
    reach = re.search(r"GetUpwindCellNumber\s*\(\s*\)\s*const\s*\{\s*return\s+(\d+)\s*;", body)
    report_bool(reach is not None and int(reach.group(1)) == 2,
                "GetUpwindCellNumber returns the two-cell reach")
    interp = parse_interpolations(body)
    report_bool(set(interp) == {"X", "Y", "Z"},
                f"Interpolate directions present: {''.join(sorted(interp))}")
    linear_interp = linear_checker.parse_interp(body)
    expected_calls = [("s", "sm1", "sm2"), ("sm1", "s", "sp1")]
    for d in "XYZ":
        if d not in interp:
            continue
        calls, blend, block = interp[d]
        report_bool(calls == expected_calls,
                    f"InterpolateIn{d}: calls Evaluate(s,sm1,sm2) then Evaluate(sm1,s,sp1)")
        if d in linear_interp:
            offsets = linear_interp[d][0]
            axis = "XYZ".index(d)
            bad = [(name, offsets[name]) for name in offsets
                   if offsets[name] != tuple(linear_checker.offset_of(name) if i == axis else 0
                                              for i in range(3))]
            report_bool(not bad, f"InterpolateIn{d}: data reads match the variable offsets")
        normalized = re.search(
            r"if\s*\(\s*upw_lo\s*!=\s*amrex::Real\(0\)\s*\)\s*"
            r"upw_lo\s*=\s*\(\s*upw_lo\s*>\s*0\s*\)\s*\?\s*"
            r"amrex::Real\(1\)\s*:\s*-\s*amrex::Real\(1\)\s*;", block)
        report_bool(normalized is not None,
                    f"InterpolateIn{d}: normalizes momentum to -1, 0 or +1")
        report_bool("m_upw_frac" not in block,
                    f"InterpolateIn{d}: does not reference m_upw_frac")
    blend_texts = [interp[d][1] for d in "XYZ" if d in interp]
    report_bool(len(blend_texts) == 3 and len(set(re.sub(r"\s+", "", x) for x in blend_texts)) == 1,
                "X/Y/Z val_lo blend expressions are identical")
    phi_source = next(expr for name, expr in parsed[2] if name == "phi")
    max_first = bool(re.search(
        r"amrex::max\s*\(\s*amrex::Real\s*\(\s*0(?:\.0*)?\s*\)\s*,", phi_source))
    report_bool(max_first, "outer amrex::max has literal zero as its first argument")
    l1_uses = len(re.findall(r"\bl1\b", body))
    print(f"  note l1 references beyond declaration: {max(0, l1_uses - 1)}\n")

    print("== exact reals")
    phi_r = build_phi_function(parsed, reals, r, False)
    report_result("limiter function: 0 <= phi(r) <= 2 for every r",
                  validity(z3.And(phi_r >= 0, phi_r <= 2)), z3.unsat)
    report_result("limiter function: phi(r) <= 2r for r >= 0 (Sweby TVD region)",
                  validity(z3.Implies(r >= 0, phi_r <= 2 * r)), z3.unsat)
    report_result("limiter function: phi(r) = 0 for r <= 0 (extrema)",
                  validity(z3.Implies(r <= 0, phi_r == 0)), z3.unsat)
    report_result("limiter function: phi(1) = 1",
                  validity(z3.Implies(r == 1, phi_r == 1)), z3.unsat)
    report_result("limiter function: phi(r) = (1+2r)/3 exactly on [1/4, 5/2]",
                  validity(z3.Implies(z3.And(r >= real_number("1/4"), r <= real_number("5/2")),
                                      phi_r == real_number("1/3") + real_number("2/3") * r)), z3.unsat)

    real_values = {"s": s, "sm1": sm1, "sm2": sm2}
    zero = real_number(0)
    F0, _ = build_evaluate(parsed, reals, real_values, False, overrides={"eps": zero})
    report_result("eps=0: face value lies between sm1 and s (upwind and downwind cells)",
                  validity(z3.And(F0 >= mn_real(sm1, s), F0 <= mx_real(sm1, s)),
                           [sm1 != sm2]), z3.unsat)

    V = {linear_checker.name_of(d): z3.Real(linear_checker.name_of(d)) for d in range(-5, 6)}
    V["upw"] = z3.Real("upw_stencil")
    gconst = parse_constants(ctext, fp_mode=False)
    _, _, up3_terms = linear_checker.parse_evaluate(structs["UPWIND3"][0], gconst, V)
    up3_smooth = z3.substitute(up3_terms[""], (V["upw"], real_number(1)))
    F0_up3, _ = build_evaluate(parsed, reals,
                               {"s": V["s"], "sm1": V["sm1"], "sm2": V["sm2"]},
                               False, overrides={"eps": zero})
    rmh0 = (V["s"] - V["sm1"]) / (V["sm1"] - V["sm2"])
    report_result("eps=0: smooth branch equals parsed UPWIND3 at upw=+1",
                  validity(z3.Implies(z3.And(rmh0 >= real_number("1/4"),
                                              rmh0 <= real_number("5/2")),
                                      F0_up3 == up3_smooth),
                           [V["sm1"] != V["sm2"]]), z3.unsat)

    a, b = z3.Reals("a b")
    Flinear, _ = build_evaluate(parsed, reals,
                                {"s": a + b * real_number("1/2"),
                                 "sm1": a - b * real_number("1/2"),
                                 "sm2": a - b * real_number("3/2")},
                                False, overrides={"eps": zero})
    report_result("eps=0: linear data a+b*x is reproduced exactly at the face (x=0)",
                  validity(Flinear == a, [b != 0]), z3.unsat)

    F_eps, _ = build_evaluate(parsed, reals, real_values, False)
    absd = mx_real(sm1 - sm2, sm2 - sm1)
    report_result("eps=1e-16: |F-sm1| <= |sm1-sm2| for all real inputs",
                  validity(z3.And(F_eps - sm1 <= absd, sm1 - F_eps <= absd)), z3.unsat)
    report_result("eps=1e-16: face value lies within eps of [min(sm1,s), max(sm1,s)]",
                  validity(z3.And(F_eps >= mn_real(sm1, s) - eps,
                                  F_eps <= mx_real(sm1, s) + eps)), z3.unsat)
    half_eps = eps / 2
    res_tight = solver_result([z3.Or(F_eps < mn_real(sm1, s) - half_eps,
                                     F_eps > mx_real(sm1, s) + half_eps)])[0]
    report_result("eps=1e-16: the eps/2 containment bound has a counterexample", res_tight, z3.sat)
    nonnegative = z3.And(s >= 0, sm1 >= 0, sm2 >= 0)
    report_result("eps=1e-16: non-negative inputs give F >= -eps",
                  validity(F_eps >= -eps, [nonnegative]), z3.unsat)

    interp_values = {"s": s, "sm1": sm1, "sm2": sm2}
    fhi, _ = build_evaluate(parsed, reals, interp_values, False)
    flo, _ = build_evaluate(parsed, reals, {"s": sm1, "sm1": s, "sm2": sp1}, False)
    blend = interp["X"][1]
    value = make_blend(blend, fhi, flo, upw, False)
    reflected = make_blend(blend, flo, fhi, -upw, False)
    report_result("InterpolateIn*: mirror symmetry under reflected data and flipped upw",
                  validity(value == reflected), z3.unsat)
    report_result("InterpolateIn*: upw=+1 gives Fhi, -1 gives Flo, 0 gives their mean",
                  validity(z3.And(z3.substitute(value, (upw, real_number(1))) == fhi,
                                  z3.substitute(value, (upw, real_number(-1))) == flo,
                                  z3.substitute(value, (upw, real_number(0))) == (fhi + flo) / 2)),
                  z3.unsat)

    print("\n== IEEE binary64 (round-to-nearest-even, std::min/std::max semantics)")
    sort, rm = z3.Float64(), z3.RNE()
    fps = parse_constants(ctext, fp_mode=True, sort=sort)
    sf, sm1f, sm2f = z3.FPs("s_f sm1_f sm2_f", sort)
    F_fp, e_fp = build_evaluate(parsed, fps,
                                {"s": sf, "sm1": sm1f, "sm2": sm2f}, True, sort)
    phi_any = z3.FP("rmh_any", sort)
    phi_fp = build_phi_function(parsed, fps, phi_any, True, sort)
    zero_fp, two_fp = fp_number(0.0, sort), fp_number(2.0, sort)
    report_result("phi is never NaN and lies in [0,2] for every rmh, including NaN and infinities",
                  validity(z3.And(z3.Not(z3.fpIsNaN(phi_fp)),
                                  z3.fpGEQ(phi_fp, zero_fp), z3.fpLEQ(phi_fp, two_fp))), z3.unsat)

    # Confirm the specific argument-order hazard described in the handoff.
    # Preserve the nested min expression while swapping only the outer max arguments.
    if "amrex::max" in phi_source:
        _, inner_expr = phi_source.split("amrex::max", 1)
        inner_expr = inner_expr.strip()
        if inner_expr.startswith("(") and inner_expr.endswith(")"):
            outer_args = inner_expr[1:-1]
            depth = 0
            comma_at = None
            for i, char in enumerate(outer_args):
                depth += (char == "(") - (char == ")")
                if char == "," and depth == 0:
                    comma_at = i
                    break
            if comma_at is not None:
                first = outer_args[:comma_at]
                second = outer_args[comma_at + 1:]
                swapped = f"amrex::max({second}, {first})"
                l2_fp = z3.fpDiv(rm, fps["one"], fps["three"])
                local_exprs = dict(parsed[2])
                k_fp = expression(local_exprs["K"], {"rmh": phi_any, "l2": l2_fp}, True, sort)
                swapped_expr = expression(swapped,
                                          {"rmh": phi_any, "l2": l2_fp, "K": k_fp},
                                          True, sort)
                res_nan = solver_result([z3.fpIsNaN(phi_any), z3.fpIsNaN(swapped_expr)])[0]
                report_result("DEMO: swapping max arguments can let NaN through", res_nan, z3.sat)
            else:
                report_bool(False, "DEMO: could parse outer max arguments")
        else:
            report_bool(False, "DEMO: could parse outer max expression")
    else:
        report_bool(False, "DEMO: outer max expression exists")

    den = e_fp["sm1"] - e_fp["sm2"] + e_fp["eps"]
    finite_inputs = [safe_finite(x) for x in (sf, sm1f, sm2f)]
    res_den = solver_result([*finite_inputs, z3.fpIsZero(den)])[0]
    report_result("finite inputs can make (sm1-sm2)+eps exactly zero", res_den, z3.sat)

    print("       concrete evaluation, sm1=0, sm2=1e-16:")
    concrete_solver = z3.Solver()
    for sv in (1.0, 0.0, -1.0):
        concrete_solver.push()
        concrete_solver.add(sf == fp_number(sv, sort),
                            sm1f == fp_number(0.0, sort),
                            sm2f == fp_number(1e-16, sort))
        if concrete_solver.check() != z3.sat:
            raise RuntimeError("concrete FP values unexpectedly became unsatisfiable")
        model = concrete_solver.model()
        print(f"         s={sv:+.1f}: den={model.eval(den, model_completion=True)} "
              f"rmh={model.eval(e_fp['rmh'], model_completion=True)} "
              f"phi={model.eval(e_fp['phi'], model_completion=True)} "
              f"out={model.eval(F_fp, model_completion=True)}")
        concrete_solver.pop()

    if not fast:
        limit = fp_number(2.0 ** 100, sort)
        bounded = [*finite_inputs, *(z3.fpLEQ(z3.fpAbs(x), limit) for x in (sf, sm1f, sm2f))]
        report_result("end to end: output is finite for finite inputs with |x| <= 2^100",
                      validity(safe_finite(F_fp), bounded, timeout_ms=240000), z3.unsat)
        rounded_difference = z3.fpSub(rm, sm1f, sm2f)
        alternate = z3.fpAdd(rm, sm1f, rounded_difference)
        possible = z3.Or(z3.fpEQ(F_fp, sm1f), z3.fpEQ(F_fp, alternate))
        report_result("end to end: zero divisor output is sm1 or sm1+(sm1-sm2)",
                      validity(possible, bounded + [z3.fpIsZero(den)], timeout_ms=240000), z3.unsat)
    else:
        print("  note skipped the two expensive end-to-end FP proofs (--fast)\n")

    print(f"done {stamp()}\n{len(FAILURES)} check(s) failed")
    return 1 if FAILURES else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="skip the two slow end-to-end FP proofs")
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    print(f"z3 {z3.get_version_string()}   source: {args.root / UPW_H}\n")
    try:
        return run(args.root, args.fast)
    except Exception as exc:
        print(f"FAIL checker error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    sys.exit(main())
