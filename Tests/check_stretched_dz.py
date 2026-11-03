#!/usr/bin/env python3
"""Verify source-derived stretched-dz construction, refinement and device copy.

The source reader consumes both complete z-level routines and the supported 1D
refiner. Z3 checks loop domains, sizes, indices and exact-real arithmetic. Unit
loop increments plus proved pointwise coverage give induction contracts, not
bounded executions. Source order guards ensure every generated node used in a
difference has already received its surface offset. Both override paths precede
the complete host-to-device transfer. The live normal momentum wrapper supplies
the consumer spacing expressions and data() binding.

Assumptions: valid nonaliasing containers, nonnegative level indices, at least
one cell per level, consistent positive integer refinement ratios and domain
sizes, successful vector/GPU operations with their usual semantics, no integer
overflow or macro rewriting. Geometry dz is positive. Explicit stretching needs
dz0>0 and ratio>0. Supplied levels must be strictly increasing. The geometric
solver must return a same-sized strictly increasing nodal vector on its accepted
path; check_geometric_grid.py supplies sufficient exact-real tolerance margins
for this property. The caller's convergence assertion alone is insufficient.
Its Newton convergence and floating-point positivity are NOT proved here. Abort
and ALWAYS_ASSERT failures do not return. ParmParse has its ordinary semantics.

The consumer indexing theorem requires a zero-based vertical domain and the
corresponding freshly copied device vector as input. X/Y cells have 0<=k<n and
interior Z nodes have 0<k<n. The allocation-to-consumer lifetime/call chain is
not proved. Neither input validation of monotonicity nor floating-point safety,
external boundary budgets, physical volumes or whole-update conservation is
established. Real-arithmetic identities need not hold exactly after rounding.

usage: check_stretched_dz.py [ERF_ROOT]
exit: 0 success, 1 failure (including unknown), 77 z3 unavailable
"""
import ast
import re
import sys
from pathlib import Path

from advection_source import checked_constants, pure_expression
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require
from stretched_dz_source import read_program

try:
    import z3
except ImportError:
    print('check_stretched_dz.py: z3-solver is not importable; skipping (exit 77)')
    sys.exit(77)

from momentum_tendency import read_wrapper

WRAPPER = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'


class Proof:
    def __init__(self):
        self.n, self.k, self.r, self.q, self.levels = z3.Ints('n k r q max_level')
        self.z = z3.Function('node', z3.IntSort(), z3.RealSort())
        self.spacing, self.dz, self.ratio, self.dz0 = z3.Reals('spacing dz ratio dz0')
        self.env = dict(nz=self.n, Nz=self.n+1, nz_zlevs=self.n+1, nodes=self.n+1,
                        host_size=self.n, k=self.k, koff=self.q, refine_fac=self.r,
                        max_level=self.levels, spacing=self.spacing, dz=self.dz,
                        grid_stretching_ratio=self.ratio, dz0=self.dz0, one=z3.RealVal(1), two=z3.RealVal(2))
        self.assumptions = [self.n > 0, self.r > 0, self.levels >= 0]
        self.failures = []
        self.count = 0
        self.scope = set()
        self.arrays = set()

    def prove(self, claim, label, assumptions=()):
        solver = z3.Solver()
        solver.set(timeout=5000)
        solver.add(*self.assumptions, *assumptions, z3.Not(claim))
        answer = solver.check()
        self.count += 1
        print(f"  {'ok  ' if answer == z3.unsat else 'FAIL'} {label} [{answer}]")
        if answer != z3.unsat:
            self.failures.append(label)
            print('       ' + str(solver.model() if answer == z3.sat else solver.reason_unknown()))

    def expression(self, text, env=None):
        pure_expression(text)
        require(not re.search(r'\b(?:nodes|host_size|spacing|Z|D)\b', text),
                'source uses an internal proof symbol')
        for name in ('zlevels_stag', 'zorig', 'dz_ptr'):
            require(name in self.arrays or not re.search(r'\b' + name + r'\b', text),
                    'array dependency outside the current function: ' + name)
        text = re.sub(r'static_cast\s*<\s*int\s*>\s*\(\s*zlevels_stag\s*\[\s*lev\s*\]\s*\.\s*size\s*\(\s*\)\s*\)', 'nodes', text)
        text = re.sub(r'zlevels_stag\s*\[\s*lev\s*\]\s*\.\s*size\s*\(\s*\)', 'nodes', text)
        text = re.sub(r'stretched_dz_h\s*\[\s*lev\s*\]\s*\.\s*size\s*\(\s*\)', 'host_size', text)
        text = re.sub(r'domain\s*\.\s*length\s*\(\s*2\s*\)', 'nz', text)
        text = re.sub(r'zlevels_stag\s*\[\s*lev\s*\]', 'Z', text)
        text = re.sub(r'\bzorig\b', 'Z', text)
        text = re.sub(r'\bdz_ptr\b', 'D', text)
        text = re.sub(r'\bdx\s*\[\s*2\s*\]', 'spacing', text)
        text = text.replace('amrex::Real', 'Real').strip()
        bindings = {**{key: value for key, value in self.env.items() if key in self.scope}, **(env or {})}
        reads, divisors = [], []

        def visit(node):
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return z3.IntVal(node.value)
            if isinstance(node, ast.Constant) and type(node.value) is float:
                return z3.RealVal(ast.get_source_segment(text, node))
            if isinstance(node, ast.Name) and node.id in bindings:
                return bindings[node.id]
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in ('Z', 'D'):
                index = visit(node.slice)
                require(z3.is_int(index), 'array index must be integer')
                reads.append((node.value.id, index))
                return self.z(index) if node.value.id == 'Z' else self.z(index+1)-self.z(index)
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
                    require(z3.is_real(left) or z3.is_real(right), 'integer division is unsupported')
                    divisors.append(right)
                    return left / right
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Real'
                    and len(node.args) == 1 and not node.keywords):
                value = visit(node.args[0])
                return z3.ToReal(value) if z3.is_int(value) else value
            raise ValueError('unsupported spacing expression: ' + text)

        return visit(ast.parse(text, mode='eval').body), reads, divisors

    def value(self, text, label, assumptions=(), env=None, initialized=None, no_reads=False):
        value, reads, divisors = self.expression(text, env)
        require(not no_reads or not reads, label + ': unexpected array dependency')
        for number, (array, index) in enumerate(reads):
            bound = self.n+1 if array == 'Z' else self.n
            safe = z3.And(index >= 0, index < bound)
            if initialized is not None:
                safe = z3.And(safe, index <= initialized)
            self.prove(safe, label + f' read {number} in initialized range', assumptions)
        for number, divisor in enumerate(divisors):
            self.prove(divisor != 0, label + f' divisor {number} nonzero', assumptions)
        return value

    def scalar(self, text, expected, label, assumptions=(), env=None):
        value = self.value(text, label, assumptions, env, no_reads=True)
        require(z3.is_int(value) == z3.is_int(expected), label + ': scalar type changed')
        self.prove(value == expected, label, assumptions)
        return value

    def loop(self, bounds, variable, lower, upper, label):
        start = self.value(bounds.start, label, no_reads=True)
        stop = self.value(bounds.stop, label, no_reads=True)
        require(z3.is_int(start) and z3.is_int(stop), 'loop bounds must be integer')
        require(not z3.is_true(z3.simplify(start == variable)) and
                variable.decl() not in [v.decl() for v in z3.z3util.get_vars(start)],
                'loop start depends on its induction variable')
        require(variable.decl() not in [v.decl() for v in z3.z3util.get_vars(stop)],
                'loop bound depends on its induction variable')
        domain = z3.And(variable >= start, variable <= stop if bounds.comparison == '<=' else variable < stop)
        self.prove(domain == z3.And(variable >= lower, variable < upper), label + ' exact loop domain')

    def difference(self, target, expression, wanted, label, assumptions, initialized=None):
        self.scalar(target, wanted, label + ' destination', assumptions)
        actual = self.value(expression, label, assumptions, initialized=initialized)
        expected = self.z(wanted+1)-self.z(wanted)
        self.prove(actual == expected, label + ' adjacent-level difference', assumptions)
        self.prove(actual > 0, label + ' positive for increasing levels',
                   [*assumptions, self.z(wanted+1) > self.z(wanted)])


def check_constructors(p, update, init):
    n, k = p.n, p.k
    cell = [k >= 0, k < n]
    node = [k >= 0, k <= n]
    upper = [k >= 1, k <= n]
    p.arrays = {'zlevels_stag'}
    p.scope = {'nodes'}
    p.scalar(update['size'], n, 'update cell count')
    p.scope |= {'nz', 'k'}
    p.scalar(update['host_size'], n, 'update host extent')
    p.scope.add('host_size')
    p.loop(update['bounds'], k, 0, n, 'update')
    p.difference(update['target'], update['value'], k, 'update', cell)
    p.scalar(update['device_size'], n, 'update device extent')
    p.scope = {'max_level'}
    for key, lower in (('levels', 0), ('override_refine_levels', 1), ('override_levels', 0), ('copy_levels', 0)):
        p.loop(init[key], z3.Int('lev'), lower, p.levels+1, key)
    p.scope = {'nz', 'k', 'spacing', 'dz0', 'grid_stretching_ratio'}
    p.scalar(init['z_size'], n+1, 'constructor nodal extent')
    p.scope.add('nodes')
    p.scalar(init['host_size'], n, 'constructor host extent')
    p.scope = {'host_size', 'max_level'}
    p.scalar(init['device_size'], n, 'constructor device extent')
    p.scope = {'nz', 'nodes', 'k', 'spacing', 'dz0', 'grid_stretching_ratio', 'host_size'}
    p.loop(init['uniform_z_loop'], k, 0, n+1, 'uniform nodes')
    p.loop(init['uniform_h_loop'], k, 0, n, 'uniform thicknesses')
    p.scalar(init['uniform_z'][0], k, 'uniform node destination', node)
    zlo = p.value(init['uniform_z'][1], 'uniform node', node, no_reads=True)
    zhi = p.value(init['uniform_z'][1], 'uniform next node', cell, {'k': k+1}, no_reads=True)
    p.prove(zlo == k*p.spacing, 'uniform node coordinate', node)
    p.scalar(init['uniform_h'][0], k, 'uniform thickness destination', cell)
    height = p.value(init['uniform_h'][1], 'uniform thickness', cell, no_reads=True)
    p.prove(height == p.spacing, 'uniform Geometry spacing', cell)
    p.prove(height == zhi-zlo, 'uniform adjacent-level difference', cell)
    p.prove(height > 0, 'uniform positive for positive Geometry dz', [*cell, p.spacing > 0])
    p.loop(init['generated_loop'], k, 0, n+1, 'generated nodes')
    p.difference(*init['generated_h'], k-1, 'generated', upper, initialized=k)
    # Source order + unit step proves both referenced nodes already have zsurf.
    g0, g1, offset = z3.Reals('generated_lower generated_upper surface_offset')
    p.prove((g1+offset)-(g0+offset) == g1-g0, 'generated surface shift preserves thickness')
    p.loop(init['geometric_loop'], k, 1, n+1, 'geometric recurrence')
    seed = p.scalar(init['geometric_seed'], p.dz0, 'geometric seed')
    p.scope.add('dz')
    p.scalar(init['geometric_h'][0], k-1, 'geometric thickness destination', upper)
    height = p.value(init['geometric_h'][1], 'geometric thickness', upper, no_reads=True)
    p.prove(height == p.dz, 'geometric stored current thickness', upper)
    p.scalar(init['geometric_z'][0], k, 'geometric node destination', upper)
    next_z = p.value(init['geometric_z'][1], 'geometric node', upper, initialized=k-1)
    p.prove(next_z == p.z(k-1)+p.dz, 'geometric coordinate recurrence', upper)
    p.prove(next_z-p.z(k-1) == height, 'geometric adjacent-level difference', upper)
    factor = p.value(init['geometric_factor'], 'geometric next factor', upper, no_reads=True)
    p.prove(factor == p.ratio, 'geometric prescribed ratio', upper)
    p.prove(z3.And(height > 0, p.dz*factor > 0), 'geometric positive induction step',
            [*upper, p.dz > 0, p.ratio > 0])
    p.prove(seed > 0, 'geometric positive induction base', [p.dz0 > 0])
    p.scope.discard('dz')
    for label in ('refined', 'override'):
        if label == 'override':
            p.scope = {'k', 'nz_zlevs', 'nodes', 'host_size', 'dz0', 'grid_stretching_ratio'}
        p.loop(init[label+'_loop'], k, 0, n, label)
        p.difference(*init[label+'_h'], k, label, cell)
    print('  ok   complete constructor bodies and per-level copy bindings consumed')


def check_refinement(p, model):
    n, r, k, q = p.n, p.r, p.k, p.q
    cells = [k >= 0, k < n]
    nodes = [k >= 0, k <= n]
    interior = [*cells, q >= 1, q < r]
    p.arrays = {'zorig'}
    p.scope = {'Nz', 'refine_fac', 'k', 'koff'}
    p.scalar(model['size'], r*n+1, 'refinement output extent')
    p.loop(model['nodes_loop'], k, 0, n+1, 'refinement coarse nodes')
    p.loop(model['cells_loop'], k, 0, n, 'refinement coarse cells')
    p.loop(model['offset_loop'], q, 1, r, 'refinement interior offsets')
    dst = p.scalar(model['node_target'], r*k, 'refinement coarse-node destination', nodes)
    p.prove(z3.And(dst >= 0, dst < r*n+1), 'refinement coarse-node destination in range', nodes)
    value = p.value(model['node_value'], 'refinement copied node', nodes)
    p.prove(value == p.z(k), 'refinement preserves coarse nodes', nodes)
    dst = p.scalar(model['inner_target'], r*k+q, 'refinement interior destination', interior)
    p.prove(z3.And(dst >= 0, dst < r*n+1), 'refinement interior destination in range', interior)
    left = p.value(model['left'], 'refinement left coordinate', cells)
    width = p.value(model['width'], 'refinement width', cells, {'z1': left})
    p.prove(left == p.z(k), 'refinement left coordinate binding', cells)
    p.prove(width == p.z(k+1)-p.z(k), 'refinement coarse width', cells)
    env = {'z1': left, 'dz': width}
    value = p.value(model['inner_value'], 'refinement interpolant', cells, env)
    expected = p.z(k) + z3.ToReal(q)/r*(p.z(k+1)-p.z(k))
    p.prove(value == expected, 'refinement linear interpolation', cells)
    # Extend the parsed interpolating polynomial to the two already-copied
    # endpoints; these lemmas justify the endpoint/interior composition.
    low = p.value(model['inner_value'], 'refinement low endpoint', cells, {**env, 'koff': z3.IntVal(0)})
    high = p.value(model['inner_value'], 'refinement high endpoint', cells, {**env, 'koff': r})
    next_value = p.value(model['inner_value'], 'refinement adjacent interpolant', cells, {**env, 'koff': q+1})
    p.prove(low == p.z(k), 'refinement polynomial matches low node', cells)
    p.prove(high == p.z(k+1), 'refinement polynomial matches high node', cells)
    p.prove(next_value-value == (p.z(k+1)-p.z(k))/r, 'refinement thickness equals coarse thickness divided by ratio', cells)
    p.prove(next_value-value > 0, 'refinement preserves increasing levels', [*cells, p.z(k+1) > p.z(k)])
    j = z3.Int('fine_cell')
    block, off = j/r, j % r
    p.prove(z3.And(block >= 0, block < n, off >= 0, off < r, j == r*block+off),
            'refinement intervals cover every fine cell', [j >= 0, j < r*n])
    print('  ok   complete refiner body and destag=false result binding consumed')


def check_consumer(p, wrappers):
    k, n = p.k, p.n
    p.arrays = {'dz_ptr'}
    p.scope = {'k', 'one', 'two'}
    for component, (source, _) in wrappers.items():
        source = compact(source)
        if component == 'Z':
            require(source.startswith('(k==0)?') and source.count('?') == source.count(':') == 1,
                    'Z consumer spacing conditional changed')
            _, arms = source.split('?', 1)
            yes, no = arms.split(':')
            # Z momentum only runs at 0<k<n; the bottom fallback is unreachable.
            expression = no
            assumptions = [k > 0, k < n, p.z(k+1) > p.z(k), p.z(k) > p.z(k-1)]
            wanted = 2/(p.z(k+1)-p.z(k-1))
            # Still syntax-check the excluded branch without claiming its use.
            p.expression(yes)
        else:
            expression = source
            assumptions = [k >= 0, k < n, p.z(k+1) > p.z(k)]
            wanted = 1/(p.z(k+1)-p.z(k))
        actual = p.value(expression, component+' consumer', assumptions)
        p.prove(actual == wanted, component+' consumer inverse nodal layer spacing', assumptions)
        p.prove(actual > 0, component+' consumer inverse spacing positive', assumptions)
    print('  ok   complete consumer wrapper and device data() binding consumed')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    checked_constants(root, {'one': 1, 'two': 2})
    update, init, refinement = read_program(root)
    wrappers = read_wrapper(strip_comments((root / WRAPPER).read_text()))
    p = Proof()
    check_constructors(p, update, init)
    check_refinement(p, refinement)
    check_consumer(p, wrappers)
    if p.failures:
        print(f'{len(p.failures)} failed obligations; no composed construction claim.')
        return 1
    print(f'{p.count} proved obligations; stretched-dz source contracts passed.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, SyntaxError, OSError) as error:
        print(f'FAIL source contract: {error}')
        sys.exit(1)
