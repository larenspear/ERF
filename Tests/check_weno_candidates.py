#!/usr/bin/env python3
"""Check ERF's source-defined WENO candidates and nonlinear weights.

The checker reads candidate and weight expressions from the WENO headers. It proves
finite-volume candidate moments in exact rational arithmetic, smoothness-indicator
positive semidefiniteness by rational LDLᵀ, and nonlinear-weight normalization with Z3.

usage: check_weno_candidates.py [ERF_ROOT]
exit:  0 all checks pass, 1 a check failed, 77 z3 is not importable (CTest skip)
"""
import re
import sys
from fractions import Fraction
from pathlib import Path

from advection_source import evaluation

try:
    import z3
except ImportError:
    print("check_weno_candidates.py: the z3-solver package is not importable; skipping (exit 77)")
    sys.exit(77)

HEADERS = (
    "Source/Utils/ERF_Interpolation_WENO.H",
    "Source/Utils/ERF_Interpolation_WENO_Z.H",
)

# Expected exact polynomial degree of each candidate at x=+1/2, using cell
# averages on unit cells centered at integer positions. The WENO-MZQ3 blend
# intentionally includes two second-order candidates and one third-order one.
EXPECTED = {
    "WENO3": {"v1": 2, "v2": 2},
    "WENO5": {"v1": 3, "v2": 3, "v3": 3},
    "WENO7": {"v1": 4, "v2": 4, "v3": 4, "v4": 4},
    "WENO_Z3": {"v1": 2, "v2": 2},
    "WENO_MZQ3": {"v1": 2, "v2": 2, "v3": 3},
    "WENO_Z5": {"v1": 3, "v2": 3, "v3": 3},
    "WENO_Z7": {"v1": 4, "v2": 4, "v3": 4, "v4": 4},
}
EXPECTED_SUPPORT = {
    "WENO3": {"v1": {"sm1", "s"}, "v2": {"s", "sp1"}},
    "WENO5": {"v1": {"sm2", "sm1", "s"}, "v2": {"sm1", "s", "sp1"},
              "v3": {"s", "sp1", "sp2"}},
    "WENO7": {"v1": {"sm3", "sm2", "sm1", "s"},
              "v2": {"sm2", "sm1", "s", "sp1"},
              "v3": {"sm1", "s", "sp1", "sp2"},
              "v4": {"s", "sp1", "sp2", "sp3"}},
    "WENO_Z3": {"v1": {"sm1", "s"}, "v2": {"s", "sp1"}},
    "WENO_MZQ3": {"v1": {"sm1", "s"}, "v2": {"s", "sp1"},
                  "v3": {"sm1", "s", "sp1"}},
    "WENO_Z5": {"v1": {"sm2", "sm1", "s"}, "v2": {"sm1", "s", "sp1"},
                "v3": {"s", "sp1", "sp2"}},
    "WENO_Z7": {"v1": {"sm3", "sm2", "sm1", "s"},
                "v2": {"sm2", "sm1", "s", "sp1"},
                "v3": {"sm1", "s", "sp1", "sp2"},
                "v4": {"s", "sp1", "sp2", "sp3"}},
}
EXPECTED_BY_HEADER = {
    HEADERS[0]: {"WENO3", "WENO5", "WENO7"},
    HEADERS[1]: {"WENO_Z3", "WENO_MZQ3", "WENO_Z5", "WENO_Z7"},
}

failures = []


def check(ok, what):
    print(f"  {'ok  ' if ok else 'FAIL'} {what}")
    if not ok:
        failures.append(what)


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def brace_block(text, start):
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            return text[start + 1:i]
    raise ValueError("unbalanced braces while reading C++ block")


def split_structs(text):
    return {m.group(1): m.group(2) for m in re.finditer(
        r"^struct (\w+)\s*\n\{(.*?)^\};", text, flags=re.S | re.M)}


def num(value):
    value = Fraction(value)
    return z3.RealVal(f"{value.numerator}/{value.denominator}")


def to_z3(expr, env):
    expr = " ".join(expr.split()).replace("amrex::Real", "")
    expr = expr.replace("std::abs", "abs")
    if re.search(r"[^\w\s.+\-*/()]", expr):
        raise ValueError(f"unsupported candidate expression: {expr.strip()}")
    expr = re.sub(r"(?<![\w.])(\d+\.?\d*(?:[eE][+-]?\d+)?)", r"_N('\1')", expr)
    abs_real = lambda value: z3.If(value >= 0, value, -value)
    return eval(expr, {"__builtins__": {}}, {**env, "_N": num, "abs": abs_real})


def rational(term):
    term = z3.simplify(term)
    if not z3.is_rational_value(term):
        raise ValueError(f"expected an exact rational, got {term}")
    return Fraction(term.numerator_as_long(), term.denominator_as_long())


def prove(claim, assumptions=(), timeout_ms=60000):
    solver = z3.SolverFor("QF_NRA")
    solver.set(timeout=timeout_ms)
    solver.add(*assumptions)
    solver.add(z3.Not(claim))
    return solver.check()


def polynomial(expr, variables):
    """Expand an exact Z3 polynomial into monomial -> Fraction coefficients."""
    expr = z3.simplify(expr)
    zero_power = (0,) * len(variables)
    if z3.is_rational_value(expr):
        value = rational(expr)
        return {} if value == 0 else {zero_power: value}
    for i, variable in enumerate(variables):
        if expr.eq(variable):
            power = [0] * len(variables)
            power[i] = 1
            return {tuple(power): Fraction(1)}

    kind = expr.decl().kind()
    args = list(expr.children())
    if kind == z3.Z3_OP_ADD:
        result = {}
        for arg in args:
            for monomial, coefficient in polynomial(arg, variables).items():
                result[monomial] = result.get(monomial, Fraction(0)) + coefficient
        return {m: c for m, c in result.items() if c}
    if kind == z3.Z3_OP_SUB:
        result = polynomial(args[0], variables)
        for arg in args[1:]:
            for monomial, coefficient in polynomial(arg, variables).items():
                result[monomial] = result.get(monomial, Fraction(0)) - coefficient
        return {m: c for m, c in result.items() if c}
    if kind == z3.Z3_OP_UMINUS:
        return {m: -c for m, c in polynomial(args[0], variables).items()}
    if kind == z3.Z3_OP_MUL:
        result = {zero_power: Fraction(1)}
        for arg in args:
            factor = polynomial(arg, variables)
            product = {}
            for left_power, left_coefficient in result.items():
                for right_power, right_coefficient in factor.items():
                    power = tuple(a + b for a, b in zip(left_power, right_power))
                    product[power] = product.get(power, Fraction(0)) + left_coefficient * right_coefficient
            result = product
        return {m: c for m, c in result.items() if c}
    if kind == z3.Z3_OP_DIV:
        denominator = polynomial(args[1], variables)
        if set(denominator) - {zero_power}:
            raise ValueError(f"non-constant divisor in smoothness polynomial: {args[1]}")
        divisor = denominator.get(zero_power, Fraction(0))
        if divisor == 0:
            raise ValueError("zero divisor in smoothness polynomial")
        return {m: c / divisor for m, c in polynomial(args[0], variables).items()}
    raise ValueError(f"unsupported smoothness polynomial term: {expr}")


def quadratic_is_psd(expr, variables):
    """Decide PSD exactly by converting the homogeneous quadratic to LDL^T."""
    terms = polynomial(expr, variables)
    if not terms or any(sum(power) != 2 for power in terms):
        raise ValueError("smoothness indicator is not a homogeneous quadratic")
    size = len(variables)
    matrix = [[Fraction(0) for _ in range(size)] for _ in range(size)]
    for powers, coefficient in terms.items():
        indices = [i for i, exponent in enumerate(powers) for _ in range(exponent)]
        if len(indices) != 2:
            raise ValueError("smoothness polynomial term is not quadratic")
        i, j = indices
        if i == j:
            matrix[i][i] += coefficient
        else:
            matrix[i][j] += coefficient / 2
            matrix[j][i] += coefficient / 2

    # Schur-complement elimination: a symmetric matrix is PSD iff a positive
    # diagonal pivot has a PSD Schur complement, or the remaining block is zero.
    while matrix:
        if any(matrix[i][i] < 0 for i in range(len(matrix))):
            return False
        pivot_index = next((i for i in range(len(matrix)) if matrix[i][i] > 0), None)
        if pivot_index is None:
            return all(value == 0 for row in matrix for value in row)
        pivot = matrix[pivot_index][pivot_index]
        rest = [i for i in range(len(matrix)) if i != pivot_index]
        matrix = [[matrix[i][j] - matrix[i][pivot_index] * matrix[pivot_index][j] / pivot
                   for j in rest] for i in rest]
    return True


def candidate_scale(name, body):
    """Read the common candidate divisor where the WENO formula has one."""
    evaluate = re.search(r"\bEvaluate\s*\([^)]*\)\s*const\s*\{", body)
    if not evaluate:
        raise ValueError(f"{name}: Evaluate() not found")
    block = brace_block(body, evaluate.end() - 1)
    ret = re.search(r"\breturn\s*(.*?);", block, flags=re.S)
    if not ret:
        raise ValueError(f"{name}: Evaluate() return not found")
    expr = ret.group(1)
    scaled = re.search(
        r"/\s*\(\s*amrex::Real\(\s*([-+0-9.eE]+)\s*\)\s*\*\s*wsum\s*\)", expr)
    if scaled:
        return Fraction(scaled.group(1))
    if re.search(r"/\s*\(?\s*wsum\s*\)?", expr):
        return Fraction(1)
    if name == "WENO_MZQ3":
        # Its alpha weights are normalized by `asum`; candidate values are already
        # normalized and the correction reduces to v3 when alpha_i = g_i.
        return Fraction(1)
    raise ValueError(f"{name}: unrecognized candidate normalization in return expression")


def parse_evaluate(name, body):
    signature = re.search(r"\bEvaluate\s*\(([^)]*)\)\s*const\s*\{", body, flags=re.S)
    if not signature:
        raise ValueError(f"{name}: Evaluate() not found")
    params = []
    for param in signature.group(1).split(","):
        match = re.fullmatch(r"\s*const\s+amrex::Real\s*&\s*(\w+)\s*", param)
        if not match:
            raise ValueError(f"{name}: unexpected Evaluate() parameter {param!r}")
        params.append(match.group(1))
    block = brace_block(body, signature.end() - 1)
    env = {param: z3.Real(param) for param in params}
    env["eps"] = z3.Real("eps")
    # Resolve the ideal weights from static constants in this source struct.
    for match in re.finditer(
            r"static constexpr amrex::Real\s+(\w+)\s*=\s*(.+?);\s*$", body, flags=re.M):
        env[match.group(1)] = to_z3(match.group(2), env)
    assignments = {}
    raw_assignments = {}
    declarations, returned = evaluation(block)
    for local, source_expr in declarations.items():
        value = to_z3(source_expr, env)
        assignments[local] = value
        raw_assignments[local] = source_expr
        env[local] = value
    return params, env, assignments, raw_assignments, returned


def cell_average_monomial(center, degree):
    """Average x^degree over [center-1/2, center+1/2]."""
    return (Fraction(2 * center + 1, 2) ** (degree + 1)
            - Fraction(2 * center - 1, 2) ** (degree + 1)) / (degree + 1)


def offset(param):
    if param == "s":
        return 0
    match = re.fullmatch(r"s([mp])(\d+)", param)
    if not match:
        raise ValueError(f"cannot determine cell offset for {param}")
    return int(match.group(2)) * (1 if match.group(1) == "p" else -1)


def prove_candidate(name, label, expr, env, params, order, scale, support):
    variables = [env[param] for param in params]
    zeros = [(var, z3.RealVal(0)) for var in variables]
    constant = rational(z3.substitute(expr, *zeros))
    coeffs = {}
    for param, var in zip(params, variables):
        replacements = [(item, z3.RealVal(int(item.eq(var)))) for item in variables]
        coeffs[param] = rational(z3.substitute(expr, *replacements)) - constant
    linear_form = num(constant) + z3.Sum([num(coeffs[p]) * env[p] for p in params])
    solver = z3.Solver()
    solver.add(expr != linear_form)
    check(solver.check() == z3.unsat, f"{name}.{label}: source candidate is linear in its cell averages")
    check(constant == 0, f"{name}.{label}: source candidate has no additive constant")

    active = {param for param in params if coeffs[param] != 0}
    check(active == support,
          f"{name}.{label}: uses exactly its candidate cells {sorted(support)} (found {sorted(active)})")

    coeff_sum = sum(coeffs.values(), Fraction(0))
    check(coeff_sum == scale,
          f"{name}.{label}: constant-data candidate scale {coeff_sum} matches Evaluate() divisor {scale}")

    # A polynomial in cell averages is checked against its exact value at the
    # right face of the local center cell, x=+1/2. Positive-flow callers shift
    # Evaluate()'s center left to the target lo face; the opposite-flow call
    # mirrors the arguments. This is the finite-volume WENO
    # convention used by these candidate formulas.
    for degree in range(order):
        got = sum((coeffs[p] * cell_average_monomial(offset(p), degree) for p in params), Fraction(0))
        expected = scale * Fraction(1, 2) ** degree
        check(got == expected,
              f"{name}.{label}: reproduces cell-average monomials through degree {order - 1} at x=+1/2"
              if degree == order - 1 else
              f"{name}.{label}: reproduces cell-average monomial degree {degree} at x=+1/2")

    next_degree = order
    got = sum((coeffs[p] * cell_average_monomial(offset(p), next_degree)
               for p in params), Fraction(0))
    expected = scale * Fraction(1, 2) ** next_degree
    check(got != expected,
          f"{name}.{label}: candidate has exactly order {order} (degree {next_degree} is not exact)")


def prove_weights(name, body, assignments, raw_assignments, env, params,
                  return_expr, candidate_count, scale):
    """Prove source smoothness indicators and normalized nonlinear weights."""
    is_mzq = "a1" in assignments
    raw_prefix = "a" if is_mzq else "w"
    normalizer_name = "asum" if is_mzq else "wsum"
    raw_names = [f"{raw_prefix}{i}" for i in range(1, candidate_count + 1)]
    weight_names = [f"w{i}" for i in range(1, candidate_count + 1)]
    beta_names = [f"b{i}" for i in range(1, candidate_count + 1)]
    required = raw_names + weight_names + beta_names + [normalizer_name]
    missing = [item for item in required if item not in assignments]
    check(not missing, f"{name}: parsed all source weights and smoothness indicators")
    if missing:
        return

    beta = [assignments[item] for item in beta_names]
    eps = z3.Real(f"{name}_eps")
    eps_values = [Fraction(value) for value in re.findall(
        r"const\s+amrex::Real\s+eps\s*=\s*amrex::Real\(\s*([-+0-9.eE]+)\s*\)", body)]
    check(len(eps_values) == 2 and all(value > 0 for value in eps_values),
          f"{name}: both source precision branches use positive eps values {eps_values}")
    beta_symbols = [z3.Real(f"{name}_{item}") for item in beta_names]
    for label, indicator in zip(beta_names, beta):
        psd = quadratic_is_psd(indicator, [env[param] for param in params])
        check(psd,
              f"{name}.{label}: smoothness indicator is non-negative for all real cell data "
              "(exact rational LDLᵀ)")

    # Abstract the already-proved beta_i expressions as independent nonnegative
    # reals. The alpha_i source formulas then reduce to small rational identities,
    # avoiding a large nonlinear solve over the expanded WENO7 quadratics.
    alpha_env = {"eps": eps, **{label: symbol for label, symbol in zip(beta_names, beta_symbols)}}
    if "t5" in raw_assignments:
        alpha_env["t5"] = z3.Real(f"{name}_t5")
    for i in range(1, candidate_count + 1):
        alpha_env[f"g{i}"] = env[f"g{i}"]
    alpha = [to_z3(raw_assignments[label], alpha_env) for label in raw_names]
    eps_positive = eps > 0
    beta_facts = [item >= 0 for item in beta_symbols]
    for i, (label, value) in enumerate(zip(raw_names, alpha), start=1):
        ideal = rational(env[f"g{i}"]) > 0
        check(ideal, f"{name}.g{i}: source ideal weight is positive")
        result = prove(value > 0, [eps_positive, *beta_facts])
        check(result == z3.unsat,
              f"{name}.{label}: unnormalized nonlinear weight is positive for eps>0"
              if result != z3.unknown else
              f"{name}.{label}: unnormalized weight proof returned unknown")

    alpha_symbols = [z3.Real(f"{name}_alpha{i}") for i in range(1, candidate_count + 1)]
    raw_sum = z3.Sum(alpha_symbols)
    normalizer_env = {label: symbol for label, symbol in zip(raw_names, alpha_symbols)}
    normalizer_source = to_z3(raw_assignments[normalizer_name], normalizer_env)
    result = prove(normalizer_source == raw_sum)
    check(result == z3.unsat,
          f"{name}.{normalizer_name}: source normalizer equals the sum of source alphas")

    alpha_positive = [item > 0 for item in alpha_symbols]
    normalizer_positive_facts = [*alpha_positive]
    result = prove(raw_sum > 0, normalizer_positive_facts)
    check(result == z3.unsat,
          f"{name}.{normalizer_name}: normalizer is positive")

    if is_mzq:
        normalized_env = {**normalizer_env, normalizer_name: raw_sum}
        normalized = [to_z3(raw_assignments[item], normalized_env) for item in weight_names]
        for weight_name, weight in zip(weight_names, normalized):
            equation = weight == alpha_symbols[weight_names.index(weight_name)] / raw_sum
            result = prove(equation)
            check(result == z3.unsat,
                  f"{name}.{weight_name}: source normalized weight is alpha/normalizer")
    else:
        normalized = [item / raw_sum for item in alpha_symbols]

    # Connect the weight proof to the entire source return, with independent
    # candidate values. This catches signs, pairings, dropped terms and divisors
    # without expanding large smoothness polynomials in the solver.
    candidates = [z3.Real(f"{name}_candidate{i}") for i in range(1, candidate_count + 1)]
    return_env = {f"v{i}": value for i, value in enumerate(candidates, 1)}
    if is_mzq:
        return_env.update(zip(weight_names, normalized))
        return_env.update({f"g{i}": env[f"g{i}"] for i in range(1, candidate_count + 1)})
        expected_return = (normalized[2] / env["g3"] *
                           (candidates[2] - env["g1"] * candidates[0] - env["g2"] * candidates[1]) +
                           normalized[0] * candidates[0] + normalized[1] * candidates[1])
    else:
        return_env.update(normalizer_env)
        return_env[normalizer_name] = normalizer_source
        expected_return = z3.Sum([weight * candidate for weight, candidate
                                 in zip(normalized, candidates)]) / num(scale)
    actual_return = to_z3(return_expr, return_env)
    result = prove(actual_return == expected_return, alpha_positive)
    check(result == z3.unsat,
          f"{name}: complete source return equals the normalized candidate reconstruction")

    proof_facts = alpha_positive
    result = prove(z3.Sum(normalized) == 1, proof_facts)
    check(result == z3.unsat,
          f"{name}: normalized nonlinear weights sum to one"
          if result != z3.unknown else
          f"{name}: normalized-weight sum proof returned unknown")
    for i, weight in enumerate(normalized, start=1):
        result = prove(weight >= 0, proof_facts)
        check(result == z3.unsat,
              f"{name}.weight{i}: normalized nonlinear weight is non-negative"
              if result != z3.unknown else
              f"{name}.weight{i}: non-negativity proof returned unknown")


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    total_structs = set()
    print(f"z3 {z3.get_version_string()}   source: {root}\n")
    for header in HEADERS:
        text = strip_comments((root / header).read_text())
        structs = split_structs(text)
        wanted = EXPECTED_BY_HEADER[header]
        check(set(structs) == wanted,
              f"{header}: found expected WENO structs {sorted(wanted)} (found {sorted(structs)})")
        total_structs.update(structs)
        for name in sorted(wanted):
            if name not in structs:
                continue
            params, env, assignments, raw_assignments, return_expr = parse_evaluate(name, structs[name])
            candidates = {key: value for key, value in assignments.items()
                          if re.fullmatch(r"v\d+", key)}
            expected_candidates = EXPECTED[name]
            check(set(candidates) == set(expected_candidates),
                  f"{name}: found candidates {sorted(expected_candidates)} (found {sorted(candidates)})")
            if set(candidates) != set(expected_candidates):
                continue
            scale = candidate_scale(name, structs[name])
            print(f"\n{name}: Evaluate() candidate normalization scale {scale}")
            for label, order in expected_candidates.items():
                prove_candidate(name, label, candidates[label], env, params, order, scale,
                                EXPECTED_SUPPORT[name][label])
            prove_weights(name, structs[name], assignments, raw_assignments, env, params, return_expr,
                          len(expected_candidates), scale)
    check(total_structs == set(EXPECTED),
          f"WENO header set complete (found {sorted(total_structs)})")
    print(f"\n{len(failures)} check(s) failed")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, NameError, TypeError) as exc:
        print(f"FAIL checker could not parse source: {exc}")
        sys.exit(1)
