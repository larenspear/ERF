#!/usr/bin/env python3
"""Verify exact-real contracts for ERF's geometric grid builder.

The complete supported struct/solver/builder region is consumed. Induction
certificates connect integer powers, the geometric sum and its derivative;
Newton's ratio clamp preserves r>1 when tol>0. The 50-step solve need not
converge. Properties of accepted nonuniform returns remain conditional.

The forced top node leaves final thickness H-F, where H is the unadjusted last
layer and F the returned residual. Increasing nodes require H>F. Two sufficient
margins are tol<returned dz0 or z_top>(nz-1)*tol. Converged alone is insufficient:
an exact source-replayed, zero-iteration counterexample is checked explicitly.

Assumptions: valid nonaliasing vector with nz+1 entries, nz>=1, positive finite
dz0/z_top/tol, ordinary vector/assert semantics, no integer overflow or macro
rewriting. Arithmetic is real, not IEEE; even a positive real margin is not a
floating-point guarantee. No full application call-chain/lifetime proof is made.
The positivity margins can discharge check_stretched_dz.py's increasing-node
assumption, but init_zlevels currently asserts only gs.converged, not a margin.

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
        require(not re.search(r'\b__(?:real|pow|abs)\b', text), 'source uses an internal expression symbol')
        for old, new in [('amrex::Real', '__real'), ('std::pow', '__pow'), ('amrex::Math::abs', '__abs')]:
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
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
                args = [visit(arg) for arg in node.args]
                arithmetic(*args)
                if node.func.id == '__real' and len(args) == 1:
                    return z3.ToReal(args[0]) if z3.is_int(args[0]) else args[0]
                if node.func.id == '__abs' and len(args) == 1:
                    return z3.Abs(args[0])
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
    d, top, tol, r, previous, f, derivative = z3.Reals('dz0 top tol ratio power_previous residual derivative')
    env = dict(dz0=d, z_top=top, nz=n, Nz=z3.ToReal(n), tol=tol)
    positive = [n >= 1, d > 0, top > 0, tol > 0]
    for name, expected in zip(FIELDS, (0, 1, 0, False, False)):
        wanted = z3.BoolVal(expected) if type(expected) is bool else z3.RealVal(expected)
        p.equal(m['defaults'][name], wanted, {}, 'invalid default ' + name)
    uniform = dict(dz0=top/n, ratio=z3.RealVal(1), residual=z3.RealVal(0),
                   uniform=z3.BoolVal(True), converged=z3.BoolVal(True))
    for name in FIELDS:
        p.equal(m['uniform'][name], uniform[name], env, 'uniform return ' + name, positive)
    p.prove(top/n > 0, 'uniform thickness positive', positive)
    p.prove(top - (n-1)*(top/n) == top/n, 'uniform forced top preserves thickness', positive)
    p.prove(n >= 2, 'nonuniform branch has at least two layers', [*positive, n != 1, d < top/n])
    seed = p.equal(m['seed'], z3.RealVal('1.03'), env, 'initial ratio', positive)
    p.prove(seed > 1, 'initial ratio invariant', positive)
    current = dict(env, SFact=r)
    invariant = [*positive, n >= 2, r > 1, previous > 0]
    geometric_sum = (r*previous-1)/(r-1)
    derivative_sum = (n*previous*(r-1)-r*previous+1)/((r-1)*(r-1))
    for field in ('initial_residual', 'updated_residual'):
        p.equal(m[field], d*geometric_sum-top, current, field, invariant, powers=(r, n, previous))
    actual_derivative = p.equal(m['derivative'], d*derivative_sum, current, 'analytic derivative',
                                invariant, powers=(r, n, previous))
    p.prove(actual_derivative > 0, 'Newton derivative positive from induction',
            [*invariant, derivative_sum > 0])
    step = p.equal(m['correction'], f/derivative, dict(current, F=f, dFdSF=derivative),
                   'Newton correction', [*invariant, derivative > 0])
    minimum = p.equal(m['minimum'], 1+tol, current, 'ratio minimum', invariant)
    clamped = z3.If(r-step > minimum, r-step, minimum)
    p.prove(clamped > 1, 'Newton step preserves ratio invariant', [*invariant, derivative > 0])
    returned = dict(dz0=d, ratio=r, residual=f, uniform=z3.BoolVal(False), converged=z3.Abs(f) <= tol)
    for name in FIELDS:
        p.equal(m['returned'][name], returned[name], dict(current, F=f), 'nonuniform return ' + name, invariant)
    p.prove(f <= tol, 'accepted residual upper bound', [tol > 0, z3.Abs(f) <= tol])
    p.prove(f >= -tol, 'accepted residual lower bound', [tol > 0, z3.Abs(f) <= tol])


def check_induction(p):
    # n is real here: proving these polynomial identities for all reals also
    # proves the induction steps for positive integer n. No variable-exponent
    # transcendental solver or uninterpreted pow axioms are used.
    n, r, power, total, ds = z3.Reals('count r P S D')
    positive = [n >= 1, r > 1, power > 0]
    p.prove(z3.RealVal(1) > 0, 'power base positive')
    p.prove(z3.And(r*power > 0, r*power >= power), 'power positive and increasing step', positive)
    p.prove((r-1)*1 == r-1, 'geometric sum base')
    p.prove((r-1)*(total+power) == r*power-1, 'geometric sum induction step',
            [(r-1)*total == power-1])
    # D_n is the derivative of sum_{j=0}^{n-1} r^j, and P=r^(n-1).
    numerator = n*power*(r-1)-r*power+1
    next_numerator = (n+1)*(r*power)*(r-1)-r*r*power+1
    p.prove((1*(r-1)-r+1) == 0, 'derivative single-layer base')
    p.prove((2*r*(r-1)-r*r+1) == (r-1)*(r-1), 'derivative two-layer base')
    p.prove(next_numerator == numerator + n*power*(r-1)*(r-1), 'derivative induction identity')
    p.prove(ds+n*power > 0, 'derivative positive induction step', [*positive, ds >= 0])
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
    domain = [n >= 1, k >= 1, k <= n]
    p.equal(m['builder_seed'], first, {'gs.dz0': first, 'gs.ratio': ratio}, 'builder thickness seed')
    p.equal(m['bottom'], z3.RealVal(0), {}, 'builder bottom')
    actual = p.equal(m['node'], z(k-1)+dz, env, 'builder node recurrence', domain, read=z)
    factor = p.equal(m['factor'], ratio, env, 'builder ratio')
    p.prove(actual-z(k-1) > 0, 'unadjusted layer positive', [*domain, dz > 0])
    p.prove(dz*factor > 0, 'builder next thickness positive', [*domain, dz > 0, ratio >= 1])
    forced = p.equal(m['top'], top, {'z_top': top, 'nz': n}, 'builder forced top')
    # The consumed unit-step loop gives the recurrence for every 1<=k<=n.
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
    # Replay the parsed expressions using exact rationals, including the zero-
    # iteration decision and the actual overwrite. This is an expected defect
    # witness, not a claim that all converged grids are monotone.
    n = 10
    d, top, tol = map(z3.RealVal, ('1e-9', '1.01e-8', '1e-8'))
    env = dict(dz0=d, z_top=top, nz=z3.IntVal(n), Nz=z3.RealVal(n), tol=tol)
    p.prove(z3.And(d > 0, top > 0, n != 1, d < top/n), 'witness enters nonuniform branch')
    ratio = p.value(m['seed'], env, 'witness initial ratio')
    residual = z3.simplify(p.value(m['initial_residual'], dict(env, SFact=ratio), 'witness residual'))
    p.prove(z3.Abs(residual) <= tol, 'witness skips Newton loop')
    result = {key: z3.simplify(p.value(value, dict(env, SFact=ratio, F=residual), 'witness return '+key))
              for key, value in m['returned'].items()}
    p.prove(result['converged'], 'witness accepted return')
    builder = {**env, **{'gs.'+key: value for key, value in result.items()}}
    dz = p.value(m['builder_seed'], builder, 'witness builder seed')
    nodes = [p.value(m['bottom'], builder, 'witness bottom')]
    for k in range(1, n+1):
        step = dict(builder, k=z3.IntVal(k), dz=dz)

        def read(index):
            index = z3.simplify(index)
            require(z3.is_int_value(index) and 0 <= index.as_long() < len(nodes), 'witness read outside initialized prefix')
            return nodes[index.as_long()]

        nodes.append(z3.simplify(p.value(m['node'], step, f'witness node {k}', read=read)))
        dz = z3.simplify(dz*p.value(m['factor'], step, f'witness factor {k}'))
    nodes[-1] = p.value(m['top'], builder, 'witness forced top')
    thickness = z3.simplify(nodes[-1]-nodes[-2])
    p.prove(thickness < 0, 'known converged nonmonotone witness')
    print(f'  witness residual={residual}; final thickness={thickness}')


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
