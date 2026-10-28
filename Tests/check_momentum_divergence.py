#!/usr/bin/env python3
"""Check normal-grid higher-order momentum divergence helpers from live C++.

For each staggered momentum component, the checker parses the source's mapped
divergence and metric factor, then uses Z3 real arithmetic to prove it equals
the expected negative-orientation x/y/z face-flux divergence. It also checks
that the wrapper writes the negative helper result to each matching RHS and
uses the intended vertical inverse spacing.

This covers the normal-grid N.H helpers; the constant-dz inline fast path,
terrain-fitted path, and embedded-boundary path are outside this check.

usage: check_momentum_divergence.py [ERF_ROOT]
exit:  0 all checks pass, 1 a check fails, 77 z3 is not importable (CTest skip)
"""
import re
import sys
from pathlib import Path

from advection_source import Source, LAMBDA, checked_constants, spacing_aliases

from check_scalar_flux_divergence import (
    check, compact, failures, function_body, number, strip_comments, to_z3, z3,
)

MOM_N_H = "Source/Advection/ERF_AdvectionSrcForMom_N.H"


def replace_inverse_map_reads(expression):
    expected = {
        ("mf_ux_inv", "i,j,0"): "ux_inv",
        ("mf_uy_inv", "i,j,0"): "uy_inv",
        ("mf_vx_inv", "i,j,0"): "vx_inv",
        ("mf_vy_inv", "i,j,0"): "vy_inv",
    }

    def replace(match):
        key = (match.group(1), compact(match.group(2)))
        if key not in expected:
            raise ValueError(f"unexpected inverse map-factor read: {match.group(0)}")
        return expected[key]

    return re.sub(r"\b(mf_ux_inv|mf_uy_inv|mf_vx_inv|mf_vy_inv)\s*\(([^)]*)\)",
                  replace, expression)


def check_component(text, component, mfsq_expression, constants):
    helper = f"AdvectionSrcFor{component}Mom_N"
    body = function_body(text, helper)
    mfsq_match = re.search(r"\bamrex::Real\s+mfsq\s*=\s*(.*?);", body, flags=re.S)
    if not mfsq_match:
        raise ValueError(f"{helper}: mfsq local not found")

    env = {
        **constants,
        "ux_inv": z3.Real("ux_inv"), "uy_inv": z3.Real("uy_inv"),
        "vx_inv": z3.Real("vx_inv"), "vy_inv": z3.Real("vy_inv"),
        "mx": z3.Real("mx"), "my": z3.Real("my"),
        "dxInv": z3.Real("dxInv"), "dyInv": z3.Real("dyInv"), "dzInv": z3.Real("dzInv"),
        "xflux_hi": z3.Real("xflux_hi"), "xflux_lo": z3.Real("xflux_lo"),
        "yflux_hi": z3.Real("yflux_hi"), "yflux_lo": z3.Real("yflux_lo"),
        "zflux_hi": z3.Real("zflux_hi"), "zflux_lo": z3.Real("zflux_lo"),
    }
    source_mfsq = replace_inverse_map_reads(mfsq_match.group(1))
    mfsq = to_z3(source_mfsq, env)
    expected_mfsq = to_z3(replace_inverse_map_reads(mfsq_expression), env)
    check(z3.simplify(mfsq == expected_mfsq) == z3.BoolVal(True),
          f"{component}: mfsq uses the matching component map factors")
    env["mfsq"] = mfsq

    # Flux reconstruction remains abstract. No protected spacing/result variable
    # may be changed in that prefix, and the entire divergence tail is consumed.
    prefix = body[:mfsq_match.start()]
    prefix = re.sub(r"amrex::Real\s+advectionSrc\s*;", "", prefix, count=1)
    if (re.search(r"\b(?:advectionSrc|mfsq|dxInv|dyInv|dzInv|return)\b", prefix)
            or prefix.count('{') != prefix.count('}')):
        raise ValueError(f"{helper}: unmodeled change or control flow before divergence")
    tail = Source(body[mfsq_match.start():])
    tail.take(r"amrex::Real\s+mfsq\s*=\s*([^;{}]+);")
    source = tail.take(r"advectionSrc\s*=\s*([^;{}]+);").group(1)
    returned = tail.take(r"return\s+([^;{}]+);").group(1)
    tail.done()
    actual = to_z3(compact(source), env)
    actual_return = to_z3(returned, {**env, "advectionSrc": actual})
    check(z3.simplify(actual_return == actual) == z3.BoolVal(True),
          f"{component}: helper returns its computed divergence")

    divergence = ((env["xflux_hi"] - env["xflux_lo"]) * env["dxInv"] * env["mfsq"] +
                  (env["yflux_hi"] - env["yflux_lo"]) * env["dyInv"] * env["mfsq"] +
                  (env["zflux_hi"] - env["zflux_lo"]) * env["dzInv"])
    solver = z3.Solver()
    solver.add(actual != divergence)
    result = solver.check()
    check(result == z3.unsat,
          f"{component}: Z3 proves the source equals the mapped x/y/z face-flux divergence")
    if result == z3.sat:
        print(f"       counterexample: {solver.model()}")


def check_wrapper(text):
    body = function_body(text, "AdvectionSrcForMomWrapper_N")
    expected = (
        "rho_u_rhs(i,j,k)=-AdvectionSrcForXMom_N(i,j,k,rho_u,rho_v,rho_w,"
        "interp_u_h,interp_u_v,mf_ux_inv,mf_uy_inv,mf_vx_inv,dxInv,dyInv,dzInv);",
        "rho_v_rhs(i,j,k)=-AdvectionSrcForYMom_N(i,j,k,rho_u,rho_v,rho_w,"
        "interp_v_h,interp_v_v,mf_uy_inv,mf_vx_inv,mf_vy_inv,dxInv,dyInv,dzInv);",
        "rho_w_rhs(i,j,k)=-AdvectionSrcForZMom_N(i,j,k,rho_u,rho_v,rho_w,w,"
        "interp_w_h,interp_w_v,interp_w_wall,mf_mx,mf_my,mf_uy_inv,mf_vx_inv,"
        "vert_adv_type,lo_z_face,hi_z_face,dxInv,dyInv,dzInv);",
    )
    source = Source(body)
    # Constructor bodies/reconstruction are outside this divergence proof, but
    # no other executable statements may be hidden between setup and launches.
    for component in 'uvw':
        for direction, policy in (('h', 'H'), ('v', 'V')):
            source.take(rf"InterpType_{policy}\s+interp_{component}_{direction}"
                        rf"\s*\(\s*{component}\s*,\s*upw_frac_{direction}\s*\)\s*;")
    source.take(r"WallInterpType\s+interp_w_wall\s*\(\s*w\s*,\s*upw_frac_v\s*\)\s*;")
    start = source.pos
    source.take(r"auto\s+dxInv\s*=\s*[^;]+;\s*auto\s+dyInv\s*=\s*[^;]+;")
    spacing_aliases(body[start:source.pos], 2)
    source.take(r"auto\s+dz_ptr\s*=\s*stretched_dz_d\.data\(\s*\)\s*;")
    for component, box, assignment in zip('XYZ', ('bxx', 'bxy', 'bxz'), expected):
        source.take(rf"amrex::ParallelFor\s*\(\s*{box}\s*," + LAMBDA)
        block = source.take(r"([^{}]*)\}\s*\)\s*;").group(1)
        kernel = Source(block)
        spacing = kernel.take(r"auto\s+dzInv\s*=\s*([^;{}]+);").group(1)
        wanted_spacing = ('one/dz_ptr[k]' if component != 'Z' else
                          '(k==0)?one/dz_ptr[k]:two/(dz_ptr[k]+dz_ptr[k-1])')
        check(compact(spacing) == wanted_spacing,
              f"wrapper uses the matching vertical spacing for {component}-momentum")
        actual = kernel.take(r"([^;{}]+;)").group(1)
        kernel.done()
        check(compact(actual) == assignment,
              f"wrapper writes {component}-momentum RHS as the negative matching helper result")
    source.done()


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    constants = {name: number(value) for name, value in
                 checked_constants(root, {"one": "1", "two": "2"}).items()}
    text = strip_comments((root / MOM_N_H).read_text())
    check_component(text, "X", "1 / (mf_ux_inv(i,j,0) * mf_uy_inv(i,j,0))", constants)
    check_component(text, "Y", "1 / (mf_vx_inv(i,j,0) * mf_vy_inv(i,j,0))", constants)
    check_component(text, "Z", "mf_mx(i,j,0) * mf_my(i,j,0)", constants)
    check_wrapper(text)
    print(f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, NameError, SyntaxError) as exc:
        print(f"FAIL checker could not parse source: {exc}")
        sys.exit(1)
