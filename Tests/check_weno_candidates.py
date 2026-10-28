#!/usr/bin/env python3
"""Check ERF's source-defined WENO candidates and nonlinear weights.

The checker reads candidate and weight expressions from the WENO headers. It proves
finite-volume candidate moments in exact rational arithmetic, smoothness-indicator
identities against independent derivative integrals, optimal ideal-weight ratios
and full-stencil ideal reconstruction, positive semidefiniteness by rational LDLᵀ,
and nonlinear-weight normalization with Z3. Independent policies specify squared
global smoothness and normalized nonlinear alpha ratios. A modular replay proves
every source division defined for arbitrary real data and positive epsilon; a
separate actual nonlinear replay proves constant/affine exactness. Polynomial identities use exact
rational coefficient comparison, not sample testing. Paired source proofs establish
offset covariance and amplitude covariance with epsilon scaled by amplitude squared.
Exact fixed-epsilon counterexamples describe the latter theorem's scope. Nonlinear convergence order,
critical-point behavior, nonuniform-grid accuracy, and floating-point rounding
remain outside these uniform-grid, cell-average checks.

usage: check_weno_candidates.py [ERF_ROOT]
exit:  0 all checks pass, 1 a check failed, 77 z3 is not importable (CTest skip)
"""
import ast
import re
import sys
from fractions import Fraction
from pathlib import Path

from advection_source import Source, evaluation
import weno_polynomial as reference
import weno_weights as weight_reference

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


def to_z3(expr, env, divisors=None):
    """Read pure scalar arithmetic, retaining C++ integer/real operand types.

    In particular, 1/3 must not become real division just because its result
    is assigned to a Real. Unsupported integer division fails closed. Optional
    divisor collection happens during AST traversal, before simplification can
    erase divisions in canceled terms, zero products, or unused declarations.
    """
    expr = " ".join(expr.split()).replace('amrex::Real', 'Real').replace('std::abs', 'abs')

    def visit(node):
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            spelling = ast.get_source_segment(expr, node)
            if not re.fullmatch(r'(?:0|[1-9]\d*)', spelling) and type(node.value) is int:
                raise ValueError('unsupported integer spelling in WENO expression')
            return z3.IntVal(spelling) if type(node.value) is int else num(spelling)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if z3.is_int(left) and z3.is_int(right):
                    raise ValueError('integer division is unsupported in WENO expressions')
                if divisors is not None:
                    divisors.append(z3.ToReal(right) if z3.is_int(right) else right)
                return left / right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) == 1 and not node.keywords:
            value = visit(node.args[0])
            if node.func.id == 'Real':
                return z3.ToReal(value) if z3.is_int(value) else value
            if node.func.id == 'abs':
                return z3.If(value >= 0, value, -value)
        raise ValueError(f'unsupported WENO expression: {expr}')

    result = visit(ast.parse(expr, mode='eval').body)
    return z3.ToReal(result) if z3.is_int(result) else result


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
    signatures = list(re.finditer(r"\bEvaluate\s*\(([^)]*)\)\s*const\s*\{", body, flags=re.S))
    if len(signatures) != 1:
        raise ValueError(f"{name}: expected one Evaluate() definition")
    signature = signatures[0]
    params = []
    for param in signature.group(1).split(","):
        match = re.fullmatch(r"\s*const\s+amrex::Real\s*&\s*(\w+)\s*", param)
        if not match:
            raise ValueError(f"{name}: unexpected Evaluate() parameter {param!r}")
        params.append(match.group(1))
    expected_params = set().union(*EXPECTED_SUPPORT[name].values())
    if len(params) != len(set(params)) or set(params) != expected_params:
        raise ValueError(f"{name}: missing, duplicate or unexpected Evaluate() samples")
    block = brace_block(body, signature.end() - 1)
    # Consume the complete supported member-data region. Constants cannot be
    # silently duplicated, omitted by line-oriented matching, or shadow samples.
    private = list(re.finditer(r'\bprivate\s*:', body))
    if len(private) != 1:
        raise ValueError(f'{name}: expected one private member-data region')
    source = Source(body[private[0].end():])
    source.take(r'amrex::Array4\s*<\s*const\s+amrex::Real\s*>\s+m_phi\s*;')
    source.take(r'#ifdef\s+AMREX_USE_FLOAT\s+')
    eps_pattern = r'const\s+amrex::Real\s+eps\s*=\s*amrex::Real\(\s*([-+0-9.eE]+)\s*\)\s*;'
    source.take(eps_pattern)
    source.take(r'#else\s+')
    source.take(eps_pattern)
    source.take(r'#endif\s+')
    constants = {}
    while source.text[source.pos:].strip():
        local, expr = source.take(r'static\s+constexpr\s+amrex::Real\s+(\w+)\s*=\s*([^;{}]+);').groups()
        if local in constants or local in expected_params | {'eps', 'm_phi'}:
            raise ValueError(f'{name}: duplicate or shadowing member constant {local}')
        divisors = []
        constants[local] = to_z3(expr, constants, divisors)
        if any(rational(divisor) == 0 for divisor in divisors):
            raise ValueError(f'{name}.{local}: zero divisor in member constant')
        rational(constants[local])  # No symbolic, sample-dependent constants.
    source.done()
    env = {**constants, **{param: z3.Real(param) for param in params}, 'eps': z3.Real('eps')}
    assignments = {}
    raw_assignments = {}
    declarations, returned = evaluation(block)
    for local, source_expr in declarations.items():
        if local in env:
            raise ValueError(f'{name}: Evaluate() shadows an input/member: {local}')
        value = to_z3(source_expr, env)
        assignments[local] = value
        raw_assignments[local] = source_expr
        env[local] = value
    return params, env, assignments, raw_assignments, returned


def polynomial_identity(actual, expected, variables, description):
    """An exact coefficient identity is universal over all real cell averages."""
    got, wanted = polynomial(actual, variables), polynomial(expected, variables)
    check(got == wanted, description)
    if got != wanted:
        power = next(power for power in sorted(set(got) | set(wanted))
                     if got.get(power, 0) != wanted.get(power, 0))
        print(f'       coefficient at powers {power}: source {got.get(power, 0)}, expected {wanted.get(power, 0)}')


def prove_polynomial_specification(name, assignments, raw_assignments, env, params, return_expr):
    """Derive beta and ideal-blend specifications without copying ERF formulas."""
    variables = [env[param] for param in params]
    supports = [tuple(sorted(support, key=offset)) for support in EXPECTED_SUPPORT[name].values()]
    stencils = tuple(tuple(offset(param) for param in support) for support in supports)
    for index, (support, stencil) in enumerate(zip(supports, stencils), 1):
        matrix = reference.smoothness(stencil)
        beta = z3.Sum([num(matrix[row][column]) * env[left] * env[right]
                       for row, left in enumerate(support) for column, right in enumerate(support)])
        polynomial_identity(assignments[f'b{index}'], beta, variables,
                            f'{name}.b{index}: equals derivative-integral smoothness indicator')

    ideals = tuple(rational(env[f'g{i}']) for i in range(1, len(stencils) + 1))
    total = sum(ideals, Fraction(0))
    # Existing weight proofs diagnose nonpositive g. Do not infer an ideal
    # reconstruction from invalid/zero normalizers.
    if any(value <= 0 for value in ideals):
        check(False, f'{name}: independent ideal reconstruction needs positive weights')
        return
    if name == 'WENO_MZQ3':
        # MZQ's corrected third candidate permits ANY positive unit-sum triple.
        # Its 1/3 choices are not determined by polynomial accuracy alone.
        check(total == 1, f'{name}: corrected ideal weights sum to one')
    else:
        wanted = reference.ideal_weights(stencils)
        got = tuple(value / total for value in ideals)
        check(got == wanted, f'{name}: normalized ideal weights match full-stencil reconstruction')
        if got != wanted:
            print(f'       source ratios {got}, expected {wanted}')

    # Replay the source declarations/return with only the RAW nonlinear weights
    # replaced by g. This is the ideal-weight operator, not an assertion that
    # the actual nonlinear weights equal g for arbitrary or smooth data.
    ideal_env = {key: value for key, value in env.items() if key not in assignments}
    prefix = 'a' if name == 'WENO_MZQ3' else 'w'
    overrides = {f'{prefix}{i}': num(value) for i, value in enumerate(ideals, 1)}
    for local, expr in raw_assignments.items():
        ideal_env[local] = overrides[local] if local in overrides else to_z3(expr, ideal_env)
    actual = to_z3(return_expr, ideal_env)
    full = tuple(sorted(set().union(*stencils)))
    samples = {offset(param): env[param] for param in params}
    expected = z3.Sum([num(value) * samples[cell] for cell, value in zip(full, reference.face(full))])
    polynomial_identity(actual, expected, variables,
                        f'{name}: source ideal return matches full-stencil reconstruction (order {len(full)})')


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


def prove_nonlinear_fields(name, assignments, raw_assignments, env, params, return_expr):
    """Replay every source declaration on symbolic constant/affine cell averages.

    No beta, tau, weight, normalizer, or return is replaced by its specification.
    At each expression, prove its divisors nonzero before simplifying the value.
    Nested divisors are visited inside out. A failed definedness obligation stops
    that field's replay: we must not claim accuracy on an empty/partial domain by
    assuming away a zero denominator, or rely on Z3's totalized division at zero.
    """
    intercept, slope = z3.Reals(f'{name}_intercept {name}_slope')
    for field, gradient in [('constant', num(0)), ('affine', slope)]:
        actual_env = {key: value for key, value in env.items() if key not in assignments}
        actual_env.update({param: intercept + gradient * offset(param) for param in params})
        assumptions = [actual_env['eps'] > 0]
        defined = True
        checked = set()
        for local, expression in [*raw_assignments.items(), ('return', return_expr)]:
            divisors = []
            value = to_z3(expression, actual_env, divisors)
            for index, divisor in enumerate(divisors, 1):
                divisor = z3.simplify(divisor, som=True)
                # Repeated divisors and nonzero constants need no solver call.
                if divisor in checked or (z3.is_rational_value(divisor) and rational(divisor) != 0):
                    continue
                result = prove(divisor != 0, assumptions)
                if result != z3.unsat:
                    check(False, f'{name}: {field} source divisions are defined for eps>0 '
                          f'({local} divisor {index}) [{result}]')
                    defined = False
                    break
                checked.add(divisor)
            if not defined:
                break
            # Expanding polynomial locals first keeps high-order beta formulas
            # small after the affine substitution, without assuming beta=m^2.
            actual_env[local] = z3.simplify(value, som=True)
        if not defined:
            continue
        check(True, f'{name}: {field} source divisions are defined for eps>0')
        result = prove(actual_env['return'] == intercept + gradient / num(2), assumptions)
        check(result == z3.unsat,
              f'{name}: actual nonlinear return reproduces {field} fields for eps>0 [{result}]')


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


def divisors_defined(divisors, assumptions):
    """Check inside-out AST divisors; never assume away a singularity."""
    for divisor in divisors:
        divisor = z3.simplify(divisor)
        if z3.is_rational_value(divisor) and rational(divisor) != 0:
            continue
        result = prove(divisor != 0, assumptions)
        if result != z3.unsat:
            return result
    return z3.unsat


def prove_nonlinear_policy(name, raw, alpha_env, raw_names, beta_symbols, positivity_proved):
    """Check tau squared and normalized alpha ratios on independent beta>=0.

    The separate PSD proof connects these beta symbols to the actual cell data.
    Alpha ratios are checked for EVERY real tau, so the subsequent substitution
    of a defined source tau with the specified square preserves the policy.
    Positive alphas and correct normalization (checked separately) make these
    cross-product equalities equivalent to equality of normalized weights.
    """
    facts = [alpha_env['eps'] > 0, *[beta >= 0 for beta in beta_symbols]]
    if name not in ('WENO3', 'WENO5', 'WENO7'):
        if 't5' not in raw:
            check(False, f'{name}: missing global smoothness expression')
            return
        divisors = []
        actual = to_z3(raw['t5'], {key: value for key, value in alpha_env.items() if key != 't5'}, divisors)
        status = divisors_defined(divisors, facts)
        check(status == z3.unsat, f'{name}: global smoothness divisions are defined [{status}]')
        if status == z3.unsat:
            expected = weight_reference.tau_squared(name, beta_symbols,
                                                    lambda x: z3.If(x >= 0, x, -x))
            status = prove(actual * actual == expected, facts)
            check(status == z3.unsat,
                  f'{name}: squared global smoothness matches independent policy [{status}]')

    divisors = []
    actual = [to_z3(raw[local], alpha_env, divisors) for local in raw_names]
    status = divisors_defined(divisors, facts)
    check(status == z3.unsat, f'{name}: nonlinear weight divisions are defined [{status}]')
    if status != z3.unsat:
        return
    if not positivity_proved:
        check(False, f'{name}: normalized nonlinear policy requires proved beta/alpha positivity')
        return
    ideals = [alpha_env[f'g{i}'] for i in range(1, len(raw_names) + 1)]
    expected = weight_reference.alphas(name, ideals, beta_symbols, alpha_env['eps'],
                                       alpha_env.get('t5'))
    # Anchor to alpha_1, whose strict positivity is a separate prerequisite.
    # A shared, positive, even data-dependent factor cancels in normalization.
    status = prove(z3.And(*[actual[i] * expected[0] == actual[0] * expected[i]
                            for i in range(1, len(actual))]), facts)
    check(status == z3.unsat,
          f'{name}: normalized nonlinear weights match independent policy [{status}]')


def prove_all_data_definedness(name, assignments, raw, env, return_expr,
                              beta_names, raw_names, positivity_proved):
    """Consume every actual declaration/return, with sound modular abstractions.

    Once its own divisions are proved defined, replace a source beta by a fresh
    nonnegative real, a raw alpha by a fresh positive real, and tau by a fresh
    unconstrained real. The separate source PSD/alpha proofs justify those
    overapproximations. All other locals, including unused ones, keep their source
    expressions. No relation between independent beta/alpha symbols is assumed.
    """
    if not positivity_proved:
        check(False, f'{name}: all-data definedness requires proved beta/alpha positivity')
        return False
    actual_env = {key: value for key, value in env.items() if key not in assignments}
    facts = [actual_env['eps'] > 0]
    for local, expression in [*raw.items(), ('return', return_expr)]:
        divisors = []
        value = to_z3(expression, actual_env, divisors)
        status = divisors_defined(divisors, facts)
        if status != z3.unsat:
            check(False, f'{name}: all-data source divisions are defined for eps>0 '
                  f'({local}) [{status}]')
            return False
        if local in beta_names or local in raw_names or local == 't5':
            value = z3.Real(f'{name}_defined_{local}')
            if local in beta_names:
                facts.append(value >= 0)
            elif local in raw_names:
                facts.append(value > 0)
        actual_env[local] = z3.simplify(value)
    check(True, f'{name}: all-data source divisions are defined for eps>0')
    return True


def prove_covariance(name, assignments, raw, env, params, returned, raw_names, scale, defined):
    """Compose source-derived offset and epsilon-aware amplitude transformations.

    Each abstraction below is introduced only after proving its paired relation:
    beta/candidates from actual sample polynomials, tau and alpha ratios from raw
    declarations, and finally the actual normalizer/normalized weights/return.
    Positive kappa is independent in the weight proof; instantiate kappa=lambda^2
    for any nonzero real lambda, including negative amplitudes. No fixed raw-alpha
    scaling is imposed, since a common positive source factor may cancel.
    """
    if not defined:
        check(False, f'{name}: covariance requires proved source division definedness')
        return
    delta, amplitude, kappa, factor = z3.Reals(f'{name}_offset {name}_amplitude {name}_kappa {name}_factor')
    beta_names = [f'b{i}' for i in range(1, len(raw_names) + 1)]
    candidate_names = list(EXPECTED[name])
    variables = [env[param] for param in params] + [delta, amplitude]
    constants = {key: value for key, value in env.items() if key not in assignments and key not in params}
    beta = {local: z3.Real(f'{name}_cov_{local}') for local in beta_names}
    tau = z3.Real(f'{name}_cov_tau')
    alpha = {local: z3.Real(f'{name}_cov_{local}') for local in raw_names}
    candidates = {local: z3.Real(f'{name}_cov_{local}') for local in candidate_names}
    for mode in ('offset', 'amplitude'):
        multiplier = num(1) if mode == 'offset' else amplitude
        shift = delta if mode == 'offset' else num(0)
        substitutions = [(env[param], multiplier * env[param] + shift) for param in params]
        beta_ok = all(polynomial(z3.substitute(assignments[local], *substitutions), variables) ==
                      polynomial(multiplier * multiplier * assignments[local], variables)
                      for local in beta_names)
        check(beta_ok, f'{name}: {mode} covariance of source beta polynomials')
        candidate_ok = all(polynomial(z3.substitute(assignments[local], *substitutions), variables) ==
                           polynomial(multiplier * assignments[local] + num(scale) * shift, variables)
                           for local in candidate_names)
        check(candidate_ok, f'{name}: {mode} covariance of source candidate polynomials')
        if not (beta_ok and candidate_ok):
            continue

        variance = num(1) if mode == 'offset' else kappa
        base = {**constants, **beta, 't5': tau}
        transformed = {**constants, **{local: variance * value for local, value in beta.items()},
                       'eps': variance * constants['eps'], 't5': variance * tau}
        facts = [constants['eps'] > 0, *[value >= 0 for value in beta.values()]]
        if mode == 'amplitude':
            facts.append(kappa > 0)
        if 't5' in raw:
            original_tau = to_z3(raw['t5'], {key: value for key, value in base.items() if key != 't5'})
            changed_tau = to_z3(raw['t5'], {key: value for key, value in transformed.items() if key != 't5'})
            status = prove(changed_tau == variance * original_tau, facts)
            check(status == z3.unsat, f'{name}: {mode} covariance of source global smoothness [{status}]')
            if status != z3.unsat:
                continue

        divisors = []
        original_alpha = [to_z3(raw[local], base, divisors) for local in raw_names]
        changed_alpha = [to_z3(raw[local], transformed, divisors) for local in raw_names]
        domain = divisors_defined(divisors, facts)
        status = (prove(z3.And(*[changed_alpha[i] * original_alpha[0] ==
                                 original_alpha[i] * changed_alpha[0]
                                 for i in range(1, len(raw_names))]), facts)
                  if domain == z3.unsat else domain)
        check(domain == z3.unsat and status == z3.unsat,
              f'{name}: {mode} covariance of normalized source weights [{status}]')
        if domain != z3.unsat or status != z3.unsat:
            continue

        # Positive source alphas and the ratio identities justify one arbitrary
        # positive common factor between the two raw-alpha vectors. Replay all
        # remaining locals/return, without substituting a reference reconstruction.
        originals = {**beta, 't5': tau, **alpha, **candidates}
        changes = {**{local: multiplier * multiplier * value for local, value in beta.items()},
                   't5': multiplier * multiplier * tau,
                   **{local: factor * value for local, value in alpha.items()},
                   **{local: multiplier * value + num(scale) * shift for local, value in candidates.items()}}
        original_env = {**constants, **{param: env[param] for param in params}}
        changed_env = {**constants, **{param: multiplier * env[param] + shift for param in params},
                       'eps': multiplier * multiplier * constants['eps']}
        divisors = []
        for local, expression in [*raw.items(), ('return', returned)]:
            original_env[local] = originals[local] if local in originals else to_z3(expression, original_env, divisors)
            changed_env[local] = changes[local] if local in changes else to_z3(expression, changed_env, divisors)
        facts = [constants['eps'] > 0, factor > 0,
                 *[value >= 0 for value in beta.values()], *[value > 0 for value in alpha.values()]]
        if mode == 'amplitude':
            facts.append(amplitude != 0)
        domain = divisors_defined(divisors, facts)
        status = (prove(changed_env['return'] == multiplier * original_env['return'] + shift, facts)
                  if domain == z3.unsat else domain)
        check(domain == z3.unsat and status == z3.unsat,
              f'{name}: {mode} covariance of actual nonlinear return [{status}]')


def show_fixed_epsilon_witness(name, env, params, returned, eps_values, defined):
    """Exact counterexamples to unqualified amplitude scaling, not sample proofs.

    Use both epsilon constants actually compiled by ERF, with the same non-affine
    cell averages j^5+j^2 and amplitude two. This checks the parameterized source
    family; the compiled kernel itself has a fixed epsilon, not an epsilon input.
    These are scope diagnostics, not a requirement that every admissible future
    ideal-weight choice must fail scaling at this particular stencil. The baseline
    self-test requires these witnesses for the current unmodified source.
    """
    if not defined or any(eps <= 0 for eps in eps_values):
        return
    actual = to_z3(returned, env)
    for eps in eps_values:
        def evaluate(multiplier, epsilon):
            replacements = [(env[param], num(multiplier * (offset(param)**5 + offset(param)**2)))
                            for param in params] + [(env['eps'], num(epsilon))]
            return rational(z3.substitute(actual, *replacements))
        original = evaluate(1, eps)
        if evaluate(2, eps) != 2 * original:
            print(f'  witness {name}: fixed-epsilon scaling fails at eps={eps} (exact rational)')


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

    prove_polynomial_specification(name, assignments, raw_assignments, env, params, return_expr)
    prove_nonlinear_fields(name, assignments, raw_assignments, env, params, return_expr)

    beta = [assignments[item] for item in beta_names]
    eps = z3.Real(f"{name}_eps")
    eps_values = [Fraction(value) for value in re.findall(
        r"const\s+amrex::Real\s+eps\s*=\s*amrex::Real\(\s*([-+0-9.eE]+)\s*\)", body)]
    check(len(eps_values) == 2 and all(value > 0 for value in eps_values),
          f"{name}: both source precision branches use positive eps values {eps_values}")
    beta_symbols = [z3.Real(f"{name}_{item}") for item in beta_names]
    positivity_proved = True
    for label, indicator in zip(beta_names, beta):
        psd = quadratic_is_psd(indicator, [env[param] for param in params])
        positivity_proved &= psd
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
        positivity_proved &= ideal and result == z3.unsat
        check(result == z3.unsat,
              f"{name}.{label}: unnormalized nonlinear weight is positive for eps>0"
              if result != z3.unknown else
              f"{name}.{label}: unnormalized weight proof returned unknown")

    prove_nonlinear_policy(name, raw_assignments, alpha_env, raw_names, beta_symbols, positivity_proved)
    defined = prove_all_data_definedness(name, assignments, raw_assignments, env, return_expr,
                                        beta_names, raw_names, positivity_proved)

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

    prove_covariance(name, assignments, raw_assignments, env, params, return_expr, raw_names, scale, defined)
    show_fixed_epsilon_witness(name, env, params, return_expr, eps_values, defined)


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
