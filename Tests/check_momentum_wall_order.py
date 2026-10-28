#!/usr/bin/env python3
"""Prove normal Z-momentum's wall policy and vertical flux routing from live C++.

For arbitrary integer lo < k < hi and all 12 supported vertical schemes, prove
the nearest-wall distance rule, one interpolation call per face, in-domain
reads for reduced UPWINDALL stencils, and equal shared-face flux expressions.
Both external direct-flux branches are checked separately as dormant paths.

Interpolation results are uninterpreted pure functions of their policy, full
call coordinates, component, and mass-flux argument. This proves call/flux
wiring, not higher-order reconstruction arithmetic. Wall-policy instantiation
and wrapper-to-helper bindings are checked by the dispatch/divergence suites.
The load-bound check covers InterpolateInZ's explicit m_phi loads; Evaluate
arithmetic is covered separately by the linear-stencil suite.
The interior-domain precondition follows the caller's ShrinkZmomBoxAtDomainEnds
usage, but this checker does not prove that caller or storage/halo validity.
Integers and reals are mathematical values: overflow and rounding are excluded.

usage: check_momentum_wall_order.py [ERF_ROOT]
exit: 0 success, 1 a check fails, 77 z3 is not importable (CTest skip)
"""
import ast
import re
import sys
from pathlib import Path

from advection_source import Source, checked_constants
from check_momentum_dispatch import SCHEMES, enum_values, require, split_parameters
import check_upwindall_z_reach as reach

try:
    import z3
except ImportError:
    print('check_momentum_wall_order.py: z3-solver is not importable; skipping (exit 77)')
    sys.exit(77)

MOM = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'
ENUM = 'Source/ERF_IndexDefines.H'
failures = []


def prove(claim, description, assumptions):
    solver = z3.Solver()
    solver.set(timeout=5000)
    solver.add(*assumptions, z3.Not(claim))
    result = solver.check()
    print(f"  {'ok  ' if result == z3.unsat else 'FAIL'} {description} [{result}]")
    if result != z3.unsat:
        failures.append(description)
        print(f'       {solver.model() if result == z3.sat else solver.reason_unknown()}')


def literal(source, text):
    """Match a supported statement, allowing whitespace between its tokens."""
    tokens = re.findall(r'\w+|::|[^\s]', text)
    source.take(r'\s*'.join(re.escape(token) for token in tokens))


def parse_if(source):
    condition = source.take(r'if\s*\(([^{};]+?)\)\s*\{').group(1)
    yes = parse_block(source)
    source.take(r'else\s*')
    if re.match(r'\s*if\b', source.text[source.pos:]):
        no = [parse_if(source)]
    else:
        source.take(r'\{')
        no = parse_block(source)
    return ('if', condition, yes, no)


def parse_block(source):
    statements = []
    while not re.match(r'\s*\}', source.text[source.pos:]):
        if re.match(r'\s*if\b', source.text[source.pos:]):
            statements.append(parse_if(source))
        elif re.match(r'\s*interp_w_', source.text[source.pos:]):
            obj, direction, args = source.take(
                r'(interp_w_\w+)\s*\.\s*InterpolateIn([XYZ])\s*\(([^();]+)\)\s*;').groups()
            statements.append(('call', obj, direction, [arg.strip() for arg in args.split(',')]))
        else:
            name, expr = source.take(r'(\w+)\s*=\s*([^;{}]+);').groups()
            statements.append(('assign', name, expr))
    source.take(r'\}')
    return statements


def read_helper(text):
    pattern = (r'template\s*<([^>]+)>\s*AMREX_GPU_DEVICE\s*AMREX_FORCE_INLINE\s*'
               r'amrex::Real\s+AdvectionSrcForZMom_N\s*\(([^()]*)\)\s*\{')
    matches = list(re.finditer(pattern, text))
    require(len(matches) == 1, 'expected one supported Z-momentum helper definition')
    match = matches[0]
    compact = lambda value: re.sub(r'\s+', '', value).replace('amrex::', '')
    require(compact(match[1]) == 'typenameInterpType_H,typenameInterpType_V,typenameWallInterpType',
            'Z helper template bindings changed')
    params = ['int ' + name for name in ('i', 'j', 'k')]
    params += ['const Array4<const Real>& ' + name for name in ('rho_u', 'rho_v', 'rho_w', 'w')]
    params += ['InterpType_H interp_w_h', 'InterpType_V interp_w_v', 'WallInterpType interp_w_wall']
    params += ['const Array4<const Real>& ' + name for name in ('mf_mx', 'mf_my', 'mf_uy_inv', 'mf_vx_inv')]
    params += ['const AdvType vert_adv_type', 'const int lo_z_face', 'const int hi_z_face']
    params += ['const Real ' + name for name in ('dxInv', 'dyInv', 'dzInv')]
    require([compact(p) for p in split_parameters(match[2])] == [compact(p) for p in params],
            'Z helper parameter bindings/types changed')
    source = Source(reach.brace_block(text, match.end() - 1))
    literal(source, 'amrex::Real advectionSrc;')
    for axis in 'uvw':
        literal(source, f'amrex::Real rho_{axis}_avg_lo, rho_{axis}_avg_hi;')
    for axis in 'xyz':
        literal(source, f'amrex::Real {axis}flux_hi; amrex::Real {axis}flux_lo;')
    literal(source, 'amrex::Real interp_hi(zero), interp_lo(zero);')
    # Horizontal arithmetic is abstracted, but the complete prefix is consumed.
    # RHS expressions are subsequently checked as pure arithmetic/array reads.
    horizontal = []
    for axis, component, high in [('X', 'u', True), ('X', 'u', False),
                                  ('Y', 'v', True), ('Y', 'v', False)]:
        edge = 'hi' if high else 'lo'
        horizontal.append(source.take(rf'rho_{component}_avg_{edge}\s*=\s*([^;{{}}]+);').group(1))
        indices = 'i+1,j,k' if axis == 'X' and high else 'i,j+1,k' if high else 'i,j,k'
        literal(source, f'interp_w_h.InterpolateIn{axis}({indices},0,interp_{edge},rho_{component}_avg_{edge});')
        horizontal.append(source.take(rf'{axis.lower()}flux_{edge}\s*=\s*([^;{{}}]+);').group(1))
    high, low = parse_if(source), parse_if(source)
    # No later writes or early returns may change the selected vertical fluxes.
    tail = [source.take(r'amrex::Real\s+mfsq\s*=\s*([^;{}]+);').group(1),
            source.take(r'advectionSrc\s*=\s*([^;{}]+);').group(1),
            source.take(r'return\s+([^;{}]+);').group(1)]
    source.done()
    return high, low, horizontal + tail


class Model:
    def __init__(self, enums, constants):
        self.enums = enums
        self.env = {name: z3.Int(name) for name in ('i', 'j', 'k', 'lo_z_face', 'hi_z_face', 'vert_adv_type')}
        self.env.update({name: z3.RealVal(str(value)) for name, value in constants.items()})
        self.locals = {'interp_hi', 'interp_lo', 'zflux_hi', 'zflux_lo', 'rho_w_avg_hi', 'rho_w_avg_lo'}
        for name in self.locals | {'mfsq', 'advectionSrc', 'dxInv', 'dyInv', 'dzInv',
                                   'xflux_hi', 'xflux_lo', 'yflux_hi', 'yflux_lo',
                                   'rho_u_avg_hi', 'rho_u_avg_lo', 'rho_v_avg_hi', 'rho_v_avg_lo'}:
            self.env[name] = z3.Real('initial_' + name)
        self.arrays = {name: z3.Function(name, *([z3.IntSort()] * 3), z3.RealSort())
                       for name in ('rho_u', 'rho_v', 'rho_w', 'w', 'mf_mx', 'mf_my', 'mf_uy_inv', 'mf_vx_inv')}
        self.interp = {obj: z3.Function(obj, *([z3.IntSort()] * 4), z3.RealSort(), z3.IntSort(), z3.RealSort())
                       for obj in ('interp_w_v', 'interp_w_wall')}

    def expression(self, text, env, condition=False):
        text = text.replace('AdvType::', 'Adv_').replace('&&', ' and ').replace('||', ' or ')
        text = re.sub(r'!(?!=)', ' not ', text).strip()

        def visit(node):
            if isinstance(node, ast.Name):
                if node.id.startswith('Adv_') and node.id[4:] in self.enums:
                    return z3.IntVal(self.enums[node.id[4:]])
                require(node.id in env, f'unknown expression symbol: {node.id}')
                value = env[node.id]
                require(not condition or z3.is_int(value), 'noninteger symbol in branch condition')
                return value
            if isinstance(node, ast.Constant) and type(node.value) is int:
                return z3.IntVal(node.value)
            if isinstance(node, ast.UnaryOp):
                value = visit(node.operand)
                if isinstance(node.op, ast.USub):
                    return -value
                if isinstance(node.op, ast.UAdd):
                    return value
                if isinstance(node.op, ast.Not) and z3.is_bool(value):
                    return z3.Not(value)
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
            if isinstance(node, ast.BoolOp):
                values = [visit(value) for value in node.values]
                require(all(z3.is_bool(value) for value in values), 'nonboolean branch operand')
                return z3.And(*values) if isinstance(node.op, ast.And) else z3.Or(*values)
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                left, right = visit(node.left), visit(node.comparators[0])
                operations = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                              ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                              ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right}
                if type(node.ops[0]) in operations:
                    return operations[type(node.ops[0])]()
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not condition:
                require(node.func.id in self.arrays and not node.keywords, 'unsupported call in arithmetic')
                args = [visit(arg) for arg in node.args]
                require(len(args) == 3 and all(z3.is_int(arg) for arg in args), 'array read must use three integer indices')
                return self.arrays[node.func.id](*args)
            raise ValueError(f'unsupported expression: {text}')

        value = visit(ast.parse('(' + text + ')', mode='eval').body)
        require(z3.is_bool(value) if condition else z3.is_arith(value), 'wrong expression type')
        return value

    def execute(self, statements, env, events, guard=z3.BoolVal(True)):
        env = dict(env)
        for statement in statements:
            kind = statement[0]
            if kind == 'if':
                _, condition, yes, no = statement
                test = self.expression(condition, env, condition=True)
                yes_env = self.execute(yes, env, events, z3.And(guard, test))
                no_env = self.execute(no, env, events, z3.And(guard, z3.Not(test)))
                for name in self.locals:
                    env[name] = z3.If(test, yes_env[name], no_env[name])
            elif kind == 'assign':
                _, name, expr = statement
                require(name in self.locals, f'write to protected or unsupported variable: {name}')
                env[name] = self.expression(expr, env)
            else:
                _, obj, direction, args = statement
                require(obj in self.interp and direction == 'Z', 'vertical flux must use a vertical interpolation object/method')
                require(len(args) == (7 if obj == 'interp_w_wall' else 6), 'wrong interpolation argument count')
                require(args[4] in ('interp_hi', 'interp_lo'), 'wrong interpolation destination')
                xyzq = [self.expression(arg, env) for arg in args[:4]]
                require(all(z3.is_int(arg) for arg in xyzq), 'interpolation indices must be integers')
                upw = self.expression(args[5], env)
                scheme = self.expression(args[6], env) if len(args) == 7 else env['vert_adv_type']
                require(z3.is_int(scheme), 'interpolation scheme must be an enum/integer')
                env[args[4]] = self.interp[obj](*xyzq, upw, scheme)
                events.append((guard, obj, xyzq[2], scheme))
        return env


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    enums = enum_values(reach.strip_comments((root / ENUM).read_text()))
    constants = checked_constants(root, {'zero': '0', 'myhalf': '1/2'})
    high, low, pure_regions = read_helper(reach.strip_comments((root / MOM).read_text()))
    model = Model(enums, constants)
    for expr in pure_regions:
        model.expression(expr, model.env)
    print('  ok   complete Z helper consumed; signatures, protected inputs, and pure arithmetic checked')
    structs = reach.split_structs(reach.strip_comments((root / reach.UPW_H).read_text()))
    require(set(reach.BRANCHES.values()) | {'UPWINDALL'} <= set(structs), 'missing UPW interpolation struct')
    signatures = list(re.finditer(r'void\s+InterpolateInZ\s*\(([^()]*)\)\s*const\s*\{', structs['UPWINDALL']))
    require(len(signatures) == 1, 'expected one UPWINDALL Z interpolation definition')
    signature = re.sub(r'\s+|amrex::', '', signatures[0][1])
    require(signature == 'constint&i,constint&j,constint&k,constint&qty_index,Real&val_lo,Realupw_lo,constAdvTypeadv_type',
            'UPWINDALL Z parameter bindings/types changed')
    standalones = {name: reach.standalone_stencil(structs[name]) for name in reach.BRANCHES.values()}
    offsets = reach.analyze_upwindall(structs['UPWINDALL'], standalones)
    i, j, k, lo, hi, scheme = [model.env[name] for name in ('i', 'j', 'k', 'lo_z_face', 'hi_z_face', 'vert_adv_type')]
    supported = z3.Or(*[scheme == enums[name] for name in SCHEMES])
    interior = [supported, lo < k, k < hi]
    print('\nWall routing: all 12 schemes, arbitrary integer lo < k < hi')
    # Preserve execution order: the low-face block must not silently overwrite
    # a high-face result already proved from the preceding block.
    events_by_edge = {'hi': [], 'lo': []}
    high_env = model.execute([high], model.env, events_by_edge['hi'])
    final_env = model.execute([low], high_env, events_by_edge['lo'])
    results = {edge: final_env['zflux_' + edge] for edge in ('hi', 'lo')}
    all_events = events_by_edge['hi'] + events_by_edge['lo']
    for edge, q in [('hi', k + 1), ('lo', k)]:
        events = events_by_edge[edge]
        actual = results[edge]
        distance = z3.If(q - lo < hi - q + 1, q - lo, hi - q + 1)
        selected = z3.If(distance == 1, enums['Centered_2nd'],
                         z3.If(z3.Or(scheme == enums['Centered_2nd'], scheme == enums['Upwind_3rd']),
                               scheme, enums['Centered_4th']))
        average = (model.arrays['rho_w'](i, j, q - 1) + model.arrays['rho_w'](i, j, q)) / 2
        wanted = average * z3.If(distance <= 2,
                                 model.interp['interp_w_wall'](i, j, q, 0, average, selected),
                                 model.interp['interp_w_v'](i, j, q, 0, average, scheme))
        prove(actual == wanted, f'{edge}: nearest-wall policy, call bindings, and flux product', interior)
        count = z3.Sum(*[z3.If(guard, 1, 0) for guard, *_ in events])
        prove(count == 1, f'{edge}: exactly one interpolation call on every interior path', interior)
        boundary = hi if edge == 'hi' else lo
        direct = model.arrays['rho_w'](i, j, k) * model.arrays['w'](i, j, k)
        prove(z3.And(actual == direct, count == 0), f'{edge}: external direct flux without interpolation',
              [supported, lo < hi, k == boundary])
    safe = []
    for guard, obj, q, selected in all_events:
        if obj == 'interp_w_wall':
            choices = [z3.And(selected == enums[name], q + min(reads) >= lo, q + max(reads) <= hi)
                       for name, reads in offsets.items()]
            safe.append(z3.Implies(guard, z3.Or(*choices)))
    prove(z3.And(*safe), 'reduced wall stencils read only within [lo, hi]', interior)
    neighbor = z3.substitute(results['lo'], (k, k + 1))
    prove(results['hi'] == neighbor, 'adjacent interior cells compute the same shared vertical flux',
          [supported, lo < k, k + 1 < hi])
    total = len(failures) + len(reach.failures)
    print(f'\n{total} check(s) failed')
    return 1 if total else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, z3.Z3Exception) as exc:
        print(f'FAIL momentum wall-order source contract: {exc}')
        sys.exit(1)
