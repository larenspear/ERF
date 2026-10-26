#!/usr/bin/env python3
"""Machine-check the linear face-reconstruction stencils in ERF.

Reads Source/Utils/ERF_Interpolation_UPW.H straight from the tree (no hand-copied
coefficients), rebuilds each Evaluate() as a Z3 real-arithmetic term, and proves:

  structure   InterpolateIn{X,Y,Z} read the offsets their variable names promise,
              Evaluate() is called with arguments in declaration order, and
              GetUpwindCellNumber() equals the stencil's true half-width
  algebra     Evaluate() is linear in the data and affine in the upwind blend `upw`
  consistency constants are reproduced exactly for every `upw`;
              the scheme has at least its declared order for every `upw`;
              mirroring the stencil and flipping `upw` is the identity;
              at upw=+1 the farthest downstream cell drops out (upwind, not downwind);
              UPWINDALL's Evaluate branches, and its inline early return in
              InterpolateInZ (the live path for Centered_2nd), equal the standalone structs

Order of accuracy is measured on the flux-divergence operator D = F_hi - F_lo applied to
monomials sampled at cell centres: exact through degree p  <=>  order p.  This is the
same number whether the data are read as point values or cell averages (the two differ
only in how the face value itself is interpreted), so no interpretation is assumed.
Proofs are in exact real arithmetic; the C++ rounds each coefficient once to double.

UPWIND3SL (flux limiter) is nonlinear and only gets the structural checks.

usage: verify_upw_stencils.py [ERF_ROOT]       exit status 1 if any check fails
"""
import re
import sys
from fractions import Fraction
from pathlib import Path

import z3

UPW_H = "Source/Utils/ERF_Interpolation_UPW.H"
CONST_H = "Source/ERF_NumericalConstants.H"
MAX_DEG = 8

# UPWINDALL branch -> standalone struct it must agree with
ALL_BRANCHES = {"Centered_2nd": "CENTERED2", "Upwind_3rd": "UPWIND3",
                "Centered_4th": "CENTERED4", "Upwind_5th": "UPWIND5",
                "Centered_6th": "CENTERED6"}
NONLINEAR = {"UPWIND3SL"}
EXPECTED = {"CENTERED2", "UPWIND3", "UPWIND3SL", "CENTERED4", "UPWIND5", "CENTERED6", "UPWINDALL"}
PHI =r"m_phi\(\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*([^,]+?)\s*,\s*qty_index\s*\)"

failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


# ---------------------------------------------------------------- C++ scraping
def strip_comments(t):
    t = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), t, flags=re.S)
    return re.sub(r"//[^\n]*", "", t)


def brace_block(s, start):
    depth = 0
    for i in range(start, len(s)):
        depth += (s[i] == "{") - (s[i] == "}")
        if depth == 0:
            return s[start + 1:i]
    raise ValueError("unbalanced braces")


def split_structs(text):
    return {m.group(1): (m.group(2), text.count("\n", 0, m.start()) + 1)
            for m in re.finditer(r"^struct (\w+)\s*\n\{(.*?)^\};", text, flags=re.S | re.M)}


def num(x):
    f = Fraction(x)
    return z3.RealVal(f"{f.numerator}/{f.denominator}")


def to_z3(expr, env):
    e = expr.replace("amrex::Real", "")
    if re.search(r"[^\w\s.+\-*/()]", e):
        raise ValueError(f"unsupported expression: {expr}")
    e = re.sub(r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?)", r"_N('\1')", e)
    return eval(e, {"__builtins__": {}}, {**env, "_N": num})


def struct_env(body, env0):
    env = dict(env0)
    for m in re.finditer(r"static constexpr amrex::Real\s+(\w+)\s*=\s*(.+?);\s*$", body, flags=re.M):
        env[m.group(1)] = to_z3(m.group(2), env)
    return env


def offset_of(name):
    if name == "s":
        return 0
    m = re.fullmatch(r"s([mp])(\d)", name)
    return int(m.group(2)) * (1 if m.group(1) == "p" else -1)


def name_of(d):
    return "s" if d == 0 else (f"sp{d}" if d > 0 else f"sm{-d}")


def index_off(expr, base):
    m = re.fullmatch(rf"{base}\s*(?:([+-])\s*(\d+))?", expr.strip())
    if not m:
        raise ValueError(f"unexpected index expression '{expr}' (base {base})")
    return 0 if not m.group(1) else int(m.group(2)) * (1 if m.group(1) == "+" else -1)


def phi_to_names(expr, axis):
    """Rewrite m_phi(i,j,k+2,qty_index) -> sp2 so an inline formula can go through to_z3."""
    def repl(m):
        offs = [index_off(a, b) for a, b in zip(m.groups(), "ijk")]
        if any(o for a, o in enumerate(offs) if a != axis):
            raise ValueError(f"off-axis read in '{m.group(0)}'")
        return name_of(offs[axis])
    return re.sub(PHI, repl, expr)


def parse_interp(body):
    """direction -> (offsets {var: (di,dj,dk)}, [Evaluate call args], {branch: inline early-return expr})"""
    out = {}
    for m in re.finditer(r"InterpolateIn([XYZ])\s*\(", body):
        blk = brace_block(body, body.index("{", m.end()))
        offs = {}
        for st in blk.split(";"):
            mm = re.search(r"amrex::Real\s+(\w+)\s*=.*?" + PHI, st, re.S)
            if mm:
                offs[mm.group(1)] = tuple(index_off(a, b) for a, b in zip(mm.groups()[1:], "ijk"))
        call = re.search(r"=\s*Evaluate\s*\(([^)]*)\)", blk)
        inline = dict(re.findall(r"AdvType::(\w+)\)\s*\{\s*val_lo\s*=\s*([^;]+);\s*return;", blk))
        out[m.group(1)] = (offs, [a.strip() for a in call.group(1).split(",")] if call else [], inline)
    return out


def parse_evaluate(body, env0, V):
    """-> (real param names, ordered all-param names, {variant: z3 term})"""
    m = re.search(r"Evaluate\s*\(([^)]*)\)\s*const\s*\{", body)
    params = []
    for p in m.group(1).split(","):
        pm = re.fullmatch(r"\s*const amrex::Real&\s*(\w+)\s*", p)
        params.append((pm.group(1) if pm else p.split()[-1], bool(pm)))
    blk = brace_block(body, m.end() - 1)
    env = struct_env(body, env0)
    for name, is_real in params:
        if is_real:
            env[name] = V[name]
    for nm, ex in re.findall(r"amrex::Real\s+(\w+)\s*=\s*([^;]+);", blk):
        env[nm] = to_z3(ex, env)
    variants = {b: to_z3(ex, env) for b, ex in
                re.findall(r"AdvType::(\w+)\)\s*\{\s*iv\s*=\s*([^;]+);", blk)}
    if not variants:                       # no adv_type switch: a plain `return expr;`
        variants[""] = to_z3(re.search(r"return\s*(.+?);", blk, flags=re.S).group(1), env)
    return [n for n, r in params if r], [n for n, _ in params], variants


# ---------------------------------------------------------------- proofs
def valid(claim):
    s = z3.Solver()
    s.add(z3.Not(claim))
    return s.check() == z3.unsat


def frac(e):
    e = z3.simplify(e)
    return Fraction(e.numerator_as_long(), e.denominator_as_long())


def declared_order(label):
    return int(re.search(r"\d", label).group())


def analyze(label, f, names, V):
    upw = V["upw"]
    A, B = {}, {}
    for v in names:
        unit = [(V[u], z3.RealVal(int(u == v))) for u in names]
        a = frac(z3.substitute(f, *unit, (upw, z3.RealVal(0))))
        b = frac(z3.substitute(f, *unit, (upw, z3.RealVal(1)))) - a
        A[v], B[v] = a, b
    w = {v: num(A[v]) + num(B[v]) * upw for v in names}
    check(valid(f == z3.Sum([w[v] * V[v] for v in names])),
          f"{label}: linear in data, affine in upw (Z3: f == sum_v (A_v + B_v*upw) * v)")

    active = [v for v in names if (A[v], B[v]) != (0, 0)]
    if any(B[v] for v in names):
        hi, lo = max(active, key=offset_of), min(active, key=offset_of)
        check(A[hi] + B[hi] == 0 and A[lo] + B[lo] != 0,
              f"{label}: at upw=+1 the farthest downstream cell ({hi}) drops out and the farthest "
              f"upstream ({lo}) stays; callers pass the lo-face momentum, so upw>0 is flow toward +x")

    mirror = [(V[n], V[name_of(-1 - offset_of(n))]) for n in names] + [(upw, -upw)]
    check(valid(f == z3.substitute(f, *mirror)),
          f"{label}: mirror symmetry (swap sides, flip upw) is the identity")

    check(valid(z3.Sum([w[v] for v in names]) == 1),
          f"{label}: reproduces constants exactly for every upw (weights sum to 1)")

    # Flux divergence of cell i, D = F_hi - F_lo, on point data x_k = k (cell centres, h=1):
    # F_lo = sum_d w_d phi(d), F_hi = sum_d w_d phi(d+1).  D[x^n] == (x^n)'(0) through n=p <=> order p.
    def order(upw_val=None):
        ww = w if upw_val is None else {v: num(A[v] + B[v] * upw_val) for v in names}
        for n in range(1, MAX_DEG + 2):
            lhs = z3.Sum([ww[v] * z3.RealVal((offset_of(v) + 1) ** n - offset_of(v) ** n) for v in names])
            if not valid(lhs == (1 if n == 1 else 0)):
                return n - 1
        return MAX_DEG + 1

    p_all, want = order(), declared_order(label)
    check(p_all >= want, f"{label}: flux-divergence order {p_all} >= declared {want} for every upw (Z3)")
    return dict(name=label, want=want, p_all=p_all, p0=order(0), p1=order(1), A=A, B=B, names=names)


def weights_str(row, upw):
    items = sorted(row["names"], key=offset_of)
    return " ".join(f"{v}={row['A'][v] + row['B'][v] * upw}" for v in items
                    if row["A"][v] + row["B"][v] * upw != 0)


# ---------------------------------------------------------------- main
def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    text = strip_comments((root / UPW_H).read_text())
    ctext = strip_comments((root / CONST_H).read_text())
    gconst = {m.group(1): num(m.group(2)) for m in re.finditer(
        r"constexpr\s+amrex::Real\s+(\w+)\s*=\s*amrex::Real\(\s*([-+0-9.eE]+)\s*\)\s*;", ctext)}
    V = {name_of(d): z3.Real(name_of(d)) for d in range(-5, 6)}
    V["upw"] = z3.Real("upw")
    print(f"z3 {z3.get_version_string()}   source: {root / UPW_H}\n")

    structs = split_structs(text)
    check(EXPECTED <= set(structs),
          f"every expected struct was found by the parser (missing: {sorted(EXPECTED - set(structs)) or 'none'})")
    print()

    rows, fns, inline_fns = [], {}, {}
    for name, (body, line) in structs.items():
        print(f"{name}  ({UPW_H}:{line})")
        interp = parse_interp(body)
        dirs = "Z" if name == "UPWINDALL" else "XYZ"
        check(set(interp) == set(dirs), f"Interpolate directions present: {''.join(sorted(interp))}")
        support = None
        for d in dirs:
            if d not in interp:
                continue
            offs, _, inline = interp[d]
            axis = "XYZ".index(d)
            bad = [v for v, o in offs.items()
                   if o != tuple(offset_of(v) if a == axis else 0 for a in range(3))]
            check(not bad, f"Interpolate{d}: offsets match variable names"
                  + (f"  MISMATCH {[(v, offs[v]) for v in bad]}" if bad else ""))
            if support is None:
                support = set(offs)
            check(set(offs) == support, f"Interpolate{d}: same stencil as the first direction")
            for b, ex in inline.items():
                inline_fns[(name, b)] = to_z3(phi_to_names(ex, axis), struct_env(body, {**gconst, **V}))
        ds = sorted(offset_of(v) for v in support)
        check(ds == list(range(ds[0], ds[-1] + 1)), f"stencil support contiguous: {ds}")
        half = max(-ds[0], ds[-1] + 1)
        m = re.search(r"GetUpwindCellNumber\s*\(\)\s*const\s*\{\s*return\s*(\d+)\s*;", body)
        if m:
            check(int(m.group(1)) == half, f"GetUpwindCellNumber()={m.group(1)} equals half-width {half}")

        if name in NONLINEAR:
            print("  --   nonlinear limiter: arithmetic checks not applied\n")
            continue

        reals, allp, variants = parse_evaluate(body, gconst, V)
        names = [n for n in reals if n != "upw"]
        check(set(names) == support, "Evaluate() parameters are exactly the cells read")
        for d in dirs:
            args = interp[d][1]
            ok = len(args) == len(allp) and all(
                a == ("upw_lo" if p == "upw" else p) for p, a in zip(allp, args))
            check(ok, f"Interpolate{d}: Evaluate() called in declaration order")
        for var, f in variants.items():
            label = name if not var else f"{name}[{var}]"
            rows.append(analyze(label, f, names, V))
            fns[label] = f
        print()

    print("UPWINDALL vs standalone structs")
    for b, s in ALL_BRANCHES.items():
        a, c = fns.get(f"UPWINDALL[{b}]"), fns.get(s)
        dead = ("UPWINDALL", b) in inline_fns
        check(a is not None and c is not None and valid(a == c),
              f"UPWINDALL[{b}] Evaluate branch"
              f"{' (unreachable: InterpolateInZ returns early)' if dead else ''} == {s} (Z3)")
        if dead:
            check(c is not None and valid(inline_fns[("UPWINDALL", b)] == c),
                  f"UPWINDALL[{b}] inline early return in InterpolateInZ (the live path) == {s} (Z3)")

    print("\norder of accuracy: flux divergence exact for polynomials through degree p (Z3, exact arithmetic)")
    print(f"  {'scheme':<12}{'declared':>9}{'all upw':>9}{'upw=0':>7}{'|upw|=1':>9}")
    for r in rows:
        if not r["name"].startswith("UPWINDALL"):
            print(f"  {r['name']:<12}{r['want']:>9}{r['p_all']:>9}{r['p0']:>7}{r['p1']:>9}")
    print("\neffective weights")
    for r in rows:
        if r["name"].startswith("UPWINDALL"):
            continue
        print(f"  {r['name']:<10} upw= 0: {weights_str(r, 0)}")
        if any(r["B"][v] != 0 for v in r["names"]):
            print(f"  {'':<10} upw=+1: {weights_str(r, 1)}")

    print(f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
