#!/usr/bin/env python3
"""Verify exact-real contracts for ERF's bounded geometric grid construction.

The complete supported source region is consumed. SMT checks connect the Horner
polynomial, bracket midpoint, capped tolerance and nodal recurrence. Finite input
checks, bounded loops, adjacency selection and the stored-node guard are checked
structurally. IEEE rounding, closest-ratio selection accuracy and convergence are
covered by compiled tests, not by these exact-real obligations.

An accepted tolerance-based return has a positive real last-layer margin. An
adjacent-ratio return may exceed the requested tolerance and makes no such claim;
the builder independently rejects nonfinite or nonincreasing stored nodes.
Assumptions: valid nonaliasing vector, ordinary assert/vector semantics, no size
or integer overflow, no macro rewriting. No global call-chain/lifetime proof.

usage: check_geometric_grid.py [ERF_ROOT]
exit: 0 success, 1 failure (including unknown), 77 z3 unavailable
"""
import ast
import re
import sys
from pathlib import Path

from advection_source import pure_expression
from check_momentum_dispatch import require
from geometric_grid_source import FIELDS, GRID, read_program

try:
    import z3
except ImportError:
    print('check_geometric_grid.py: z3-solver is not importable; skipping (exit 77)')
    sys.exit(77)


class Proof:
    def __init__(self):
        self.count = 0
        self.failures = []

    def prove(self, claim, label, assumptions=()):
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*assumptions, z3.Not(claim))
        answer = solver.check()
        self.count += 1
        print(f"  {'ok  ' if answer == z3.unsat else 'FAIL'} {label} [{answer}]")
        if answer != z3.unsat:
            self.failures.append(label)
            print('       ' + str(solver.model() if answer == z3.sat else solver.reason_unknown()))

    def value(self, text, env, label, assumptions=(), powers=None, read=None):
        pure_expression(text)
        require(not re.search(r'\b__(?:real|pow|abs|min)\b', text), 'source uses an internal expression symbol')
        for old, new in [('amrex::Real', '__real'), ('std::pow', '__pow'), ('amrex::Math::abs', '__abs'), ('amrex::min', '__min')]:
            text = re.sub(r'\s*::\s*'.join(map(re.escape, old.split('::'))), new, text)
        text = re.sub(r'\s+', ' ', text).strip()
        divisors = []

        def arithmetic(*values):
            require(all(z3.is_arith(value) for value in values), 'numeric expression requires arithmetic operands')

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return (z3.IntVal if type(node.value) is int else z3.RealVal)(ast.get_source_segment(text, node))
            if isinstance(node, ast.Name):
                if node.id in ('true', 'false'):
                    return z3.BoolVal(node.id == 'true')
                if node.id in env:
                    return env[node.id]
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                key = node.value.id + '.' + node.attr
                if key in env:
                    return env[key]
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == 'zlevels' and read is not None:
                index = visit(node.slice)
                require(z3.is_int(index), 'builder index must be integer')
                self.prove(z3.And(index >= 0, index < env['k']), label + ' initialized read', assumptions)
                return read(index)
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                arithmetic(value)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                arithmetic(left, right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    require(z3.is_real(left) or z3.is_real(right), 'integer division unsupported')
                    divisors.append(right)
                    return left / right
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = visit(node.left), visit(node.comparators[0])
                arithmetic(left, right)
                if isinstance(node.ops[0], ast.LtE):
                    return left <= right
                if isinstance(node.ops[0], ast.Gt):
                    return left > right
                if isinstance(node.ops[0], ast.GtE):
                    return left >= right
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                args = [visit(arg) for arg in node.args]
                arithmetic(*args)
                if node.func.id == '__real' and len(args) == 1:
                    return z3.ToReal(args[0]) if z3.is_int(args[0]) else args[0]
                if node.func.id == '__abs' and len(args) == 1:
                    return z3.Abs(args[0])
                if node.func.id == '__min' and len(args) == 2:
                    return z3.If(args[0] <= args[1], args[0], args[1])
                if node.func.id == '__pow' and len(args) == 2:
                    base, exponent = args
                    exponent = z3.simplify(exponent)
                    if z3.is_int_value(exponent) or z3.is_rational_value(exponent):
                        require(exponent.denominator_as_long() == 1 if z3.is_rational_value(exponent) else True,
                                'power exponent must be an integer')
                        power = exponent.as_long()
                        require(0 <= power <= 64, 'literal power outside supported range')
                        product = z3.RealVal(1)
                        for _ in range(power):
                            product *= base
                        return product
                    # Integer powers are abstracted ONLY at the source ratio and
                    # the two exponents justified by the induction certificates.
                    if powers is not None:
                        r, n, previous = powers
                        if z3.is_true(z3.simplify(base == r)):
                            if z3.is_true(z3.simplify(exponent == n)):
                                return r * previous
                            if z3.is_true(z3.simplify(exponent == n-1)):
                                return previous
                    raise ValueError('unsupported power in ' + label)
            raise ValueError('unsupported or out-of-scope expression: ' + text)

        result = visit(ast.parse(text, mode='eval').body)
        for index, divisor in enumerate(divisors):
            self.prove(divisor != 0, label + f' divisor {index} nonzero', assumptions)
        return result

    def equal(self, text, expected, env, label, assumptions=(), **kwargs):
        actual = self.value(text, env, label, assumptions, **kwargs)
        require(z3.is_bool(actual) == z3.is_bool(expected), label + ': boolean type changed')
        self.prove(actual == expected, label, assumptions)
        return actual


def check_solver(p, m):
    n = z3.Int('nz')
    d, top, tol, r, depth, low, high, residual, largest = z3.Reals(
        'dz0 top tol ratio depth lower upper residual largest')
    env = dict(dz0=d, z_top=top, nz=n, Nz=z3.ToReal(n), tol=tol)
    positive = [n >= 1, d > 0, top > 0, tol > 0]
    for name, expected in zip(FIELDS, (0, 1, 0, False, False)):
        wanted = z3.BoolVal(expected) if type(expected) is bool else z3.RealVal(expected)
        p.equal(m['defaults'][name], wanted, {}, 'invalid default ' + name)
    uniform = dict(dz0=top/n, ratio=z3.RealVal(1), residual=z3.RealVal(0),
                   uniform=z3.BoolVal(True), converged=(top/n > 0))
    for name in FIELDS:
        if name != 'converged':
            p.equal(m['uniform'][name], uniform[name], env, 'uniform return ' + name, positive)
    # Do not assume a stored quotient is positive: IEEE division can underflow.
    # Check the guard for arbitrary stored values as well as exact-real inputs.
    stored_first = z3.Real('stored_uniform_first')
    p.equal(m['uniform']['converged'], stored_first > 0,
            {'gs.dz0': stored_first}, 'uniform return converged')
    p.prove(top/n > 0, 'uniform thickness positive', positive)
    p.prove(top - (n-1)*(top/n) == top/n, 'uniform forced top preserves thickness', positive)
    p.prove(n >= 2, 'nonuniform branch has at least two layers', [*positive, n != 1, d < top/n])
    p.equal(m['depth_seed'], d, env, 'depth seed')
    step = p.equal(m['depth_step'], depth*r+d, dict(env, depth=depth, ratio=r), 'Horner recurrence')
    p.prove(step > 0, 'depth recurrence stays positive', [d > 0, depth > 0, r >= 1])
    p.prove(step >= depth+d, 'depth recurrence at least uniform', [d > 0, depth > 0, r >= 1])
    other = z3.Real('other_ratio')
    p.prove(depth*other+d > step, 'depth recurrence increases in ratio',
            [depth > 0, other > r])
    target = p.equal(m['target'], z3.If(tol <= d/2, tol, d/2), env, 'capped tolerance', positive)
    p.prove(z3.And(target > 0, target <= tol, target <= d/2), 'target bounds', positive)
    p.equal(m['lower'], z3.RealVal(1), env, 'initial lower ratio')
    p.equal(m['upper'], z3.RealVal(2), env, 'initial upper ratio')
    lower_return = dict(dz0=d, ratio=low, residual=residual, uniform=z3.BoolVal(False),
                        converged=z3.Abs(residual) <= largest)
    for name in FIELDS:
        p.equal(m['lower_return'][name], lower_return[name],
                dict(env, lower=low, lower_residual=residual, largest=largest),
                'lower-bound return ' + name)
    midpoint = p.equal(m['midpoint'], low+(high-low)/2, dict(lower=low, upper=high), 'bracket midpoint')
    p.prove(z3.And(midpoint > low, midpoint < high), 'exact midpoint inside bracket', [low < high])
    p.prove(high-midpoint == (high-low)/2, 'upper half contracts bracket', [low < high])
    p.prove(midpoint-low == (high-low)/2, 'lower half contracts bracket', [low < high])
    # For a monotone depth, retaining the midpoint with the same residual sign
    # keeps the root bracketed. IEEE adjacency and overflow are not real claims.
    root = z3.Real('root')
    p.prove(z3.And(midpoint <= root, root <= high), 'negative midpoint bracket update',
            [low <= root, root <= high, midpoint <= root])
    p.prove(z3.And(low <= root, root <= midpoint), 'positive midpoint bracket update',
            [low <= root, root <= high, root <= midpoint])
    returned = dict(dz0=d, ratio=r, residual=residual, uniform=z3.BoolVal(False),
                    converged=z3.Abs(residual) <= largest)
    for name in FIELDS:
        p.equal(m['returned'][name], returned[name], dict(env, ratio=r, residual=residual, largest=largest),
                'nonuniform return ' + name)
    p.prove(d-residual > 0, 'tolerance acceptance leaves final-layer margin',
            [*positive, z3.Abs(residual) <= target])


def check_induction(p):
    # n is real here: proving these polynomial identities for all reals also
    # proves the induction steps for positive integer n. No variable-exponent
    # transcendental solver or uninterpreted pow axioms are used.
    n, r, power, total = z3.Reals('count r P S')
    positive = [n >= 1, r > 1, power > 0]
    p.prove(z3.RealVal(1) > 0, 'power base positive')
    p.prove(z3.And(r*power > 0, r*power >= power), 'power positive and increasing step', positive)
    p.prove((r-1)*1 == r-1, 'geometric sum base')
    p.prove((r-1)*(total+power) == r*power-1, 'geometric sum induction step',
            [(r-1)*total == power-1])
    p.prove((r-1)*(r*total+1) == r*power-1, 'Horner geometric sum induction step',
            [(r-1)*total == power-1])
    # Horner evaluation is strictly increasing in r from the two-layer case.
    high_r, low_sum, high_sum, first = z3.Reals('high_ratio low_sum high_sum first')
    p.prove(first*high_r+first > first*r+first, 'depth monotonicity two-layer base',
            [first > 0, high_r > r])
    p.prove(high_sum*high_r+first > low_sum*r+first, 'depth monotonicity induction step',
            [first > 0, high_r > r, r >= 1, high_sum >= low_sum, low_sum > 0])
    # S_n <= n*H_n: every earlier layer is at most the last. The recurrence
    # below establishes the sum bound without assuming that whole property.
    total, last, first = z3.Reals('raw_sum last first')
    p.prove(first <= first, 'last-layer sum bound base')
    p.prove(total+last*r <= (n+1)*(last*r), 'last-layer sum bound induction step',
            [n >= 1, r >= 1, last > 0, total <= n*last])
    p.prove(last*r >= first, 'last layer at least first induction step', [r >= 1, last >= first, first > 0])


def check_builder(p, m):
    n, k = z3.Ints('builder_n builder_k')
    first, ratio, dz, top = z3.Reals('first ratio dz requested_top')
    z = z3.Function('node', z3.IntSort(), z3.RealSort())
    env = {'nz': n, 'k': k, 'gs.dz0': first, 'gs.ratio': ratio, 'dz': dz, 'z_top': top}
    domain = [n >= 1, k >= 2, k <= n]
    p.equal(m['builder_seed'], first, {'gs.dz0': first, 'gs.ratio': ratio}, 'builder thickness seed')
    p.equal(m['bottom'], z3.RealVal(0), {}, 'builder bottom')
    p.equal(m['first_node'], first, {'dz': first}, 'builder first node')
    actual = p.equal(m['node'], z(k-1)+dz, env, 'builder node recurrence', domain, read=z)
    factor = p.equal(m['factor'], ratio, env, 'builder ratio')
    p.prove(actual-z(k-1) > 0, 'unadjusted layer positive', [*domain, dz > 0])
    p.prove(dz*factor > 0, 'builder next thickness positive', [*domain, dz > 0, ratio >= 1])
    forced = p.equal(m['top'], top, {'z_top': top, 'nz': n}, 'builder forced top')
    # The first-node assignment and unit-step loop cover every 1<=k<=n.
    # Only k>=2 multiplies a thickness. IEEE exception behavior is tested in C++.
    # Its raw sum is the solver's geometric sum by the certificates above.
    total, last, residual, tol = z3.Reals('raw_total last residual tol')
    common = [n >= 1, first > 0, last >= first, tol > 0, top > 0,
              residual == total-top, z3.Abs(residual) <= tol]
    adjusted = forced-(total-last)
    p.prove(adjusted == last-residual, 'final thickness equals last minus residual', common)
    p.prove((adjusted > 0) == (last > residual), 'exact final positivity condition', common)
    p.prove(adjusted > 0, 'first-thickness tolerance margin sufficient', [*common, tol < first])
    p.prove(adjusted > 0, 'column-depth tolerance margin sufficient',
            [*common, total <= n*last, top > (n-1)*tol])
    p.prove(top + z3.Real('surface')-(total-last+z3.Real('surface')) == adjusted,
            'surface translation preserves final thickness')


def check_witness(p, m):
    # The old tiny-grid witness no longer meets the source-derived tolerance.
    d, top, tol = map(z3.RealVal, ('1e-9', '1.01e-8', '1e-8'))
    ratio = z3.RealVal('1.03')
    env = dict(dz0=d, z_top=top, tol=tol)
    depth = p.value(m['depth_seed'], env, 'witness seed')
    for _ in range(1, 10):
        depth = z3.simplify(p.value(m['depth_step'], dict(env, depth=depth, ratio=ratio), 'witness polynomial'))
    target = p.value(m['target'], env, 'witness target')
    p.prove(depth-top > target, 'old inverted-grid witness fails capped tolerance')
    # The builder recurrence is tied to the same polynomial by induction, but
    # stored-node validation is still needed for rounding and adjacency returns.
    print(f'  old witness residual={z3.simplify(depth-top)}; target={z3.simplify(target)}')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    model = read_program((root / GRID).read_text())
    p = Proof()
    check_solver(p, model)
    check_induction(p)
    check_builder(p, model)
    check_witness(p, model)
    if p.failures:
        print(f'{len(p.failures)} failed obligations; no geometric-grid contract claim.')
        return 1
    print(f'{p.count} proved obligations; geometric-grid conditional contracts passed.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, SyntaxError, OSError) as error:
        print(f'FAIL source contract: {error}')
        sys.exit(1)
