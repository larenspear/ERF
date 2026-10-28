"""Source execution and independent stagger geometry for horizontal momentum flux.

Consume the complete helper, retaining execution order across horizontal and
vertical blocks. Vertical interpolation is a pure arbitrary function; a later
write to a horizontal result therefore cannot escape the final flux checks.
The divergence tail is consumed here and checked by check_momentum_divergence.
"""
import re

from advection_source import Source, pure_expression
from check_momentum_dispatch import compact, require, split_parameters
from check_momentum_wall_order import Model, literal, z3
from check_upwindall_z_reach import brace_block


def statements(source, nested=False):
    result = []
    stop = r'\s*\}' if nested else r'\s*amrex::Real\s+mfsq\b'
    while not re.match(stop, source.text[source.pos:]):
        if re.match(r'\s*if\b', source.text[source.pos:]):
            result.extend(conditional(source))
        elif re.match(r'\s*interp_', source.text[source.pos:]):
            obj, direction, args = source.take(
                r'(interp_[uvw]_\w+)\s*\.\s*InterpolateIn([XYZ])\s*\(([^();]+)\)\s*;').groups()
            result.append(('call', obj, direction, [arg.strip() for arg in args.split(',')]))
        else:
            name, expression = source.take(r'(\w+)\s*=\s*([^;{}]+);').groups()
            pure_expression(expression)
            result.append(('assign', name, expression))
    if nested:
        source.take(r'\}')
    return result


def block(source):
    source.take(r'\{')
    return statements(source, True)


def conditional(source):
    condition = source.take(r'if\s*\(([^{};]+?)\)\s*\{').group(1)
    yes = statements(source, True)
    source.take(r'else\s*')
    no = conditional(source) if re.match(r'\s*if\b', source.text[source.pos:]) else block(source)
    return [('if', condition, yes, no)]


def read_helper(text, component, with_tail=False):
    """Consume the helper; optionally return its metric/divergence/return tail."""
    velocity = 'uvw'['XYZ'.index(component)]
    pattern = (r'template\s*<([^>]+)>\s*AMREX_GPU_DEVICE\s*AMREX_FORCE_INLINE\s*'
               rf'amrex::Real\s+AdvectionSrcFor{component}Mom_N\s*\(([^()]*)\)\s*\{{')
    matches = list(re.finditer(pattern, text))
    require(len(matches) == 1, f'{component}: expected one momentum helper')
    match = matches[0]
    template = 'typenameInterpType_H,typenameInterpType_V'
    if component == 'Z':
        template += ',typenameWallInterpType'
    require(compact(match[1]) == template, f'{component}: helper template bindings changed')
    params = ['int ' + name for name in 'ijk']
    arrays = ['rho_u', 'rho_v', 'rho_w'] + (['w'] if component == 'Z' else [])
    params += ['const Array4<const Real>& ' + name for name in arrays]
    params += [f'InterpType_H interp_{velocity}_h', f'InterpType_V interp_{velocity}_v']
    if component == 'Z':
        params += ['WallInterpType interp_w_wall']
    metrics = {'X': ('mf_ux_inv', 'mf_uy_inv', 'mf_vx_inv'),
               'Y': ('mf_uy_inv', 'mf_vx_inv', 'mf_vy_inv'),
               'Z': ('mf_mx', 'mf_my', 'mf_uy_inv', 'mf_vx_inv')}[component]
    params += ['const Array4<const Real>& ' + name for name in metrics]
    if component == 'Z':
        params += ['const AdvType vert_adv_type', 'const int lo_z_face', 'const int hi_z_face']
    params += ['const Real ' + name for name in ('dxInv', 'dyInv', 'dzInv')]
    require([compact(p).replace('amrex::', '') for p in split_parameters(match[2])] ==
            [compact(p) for p in params], f'{component}: helper parameter bindings changed')
    source = Source(brace_block(text, match.end() - 1))
    literal(source, 'amrex::Real advectionSrc;')
    for axis in 'uvw':
        literal(source, f'amrex::Real rho_{axis}_avg_lo, rho_{axis}_avg_hi;')
    for axis in 'xyz':
        literal(source, f'amrex::Real {axis}flux_hi; amrex::Real {axis}flux_lo;')
    literal(source, 'amrex::Real interp_hi(zero), interp_lo(zero);')
    result = statements(source)
    tail = [source.take(r'amrex::Real\s+mfsq\s*=\s*([^;{}]+);').group(1),
            source.take(r'advectionSrc\s*=\s*([^;{}]+);').group(1),
            source.take(r'return\s+([^;{}]+);').group(1)]
    for expression in tail:
        pure_expression(expression)
    source.done()
    return (result, tail) if with_tail else result


class HorizontalModel(Model):
    def __init__(self, enums, constants, component, field, reconstruct):
        super().__init__(enums, constants)
        self.component, self.field, self.reconstruct = component, field, reconstruct
        self.locals |= {f'{axis}flux_{edge}' for axis in 'xyz' for edge in ('hi', 'lo')}
        self.locals |= {f'rho_{axis}_avg_{edge}' for axis in 'uvw' for edge in ('hi', 'lo')}
        # The source initializes only the interpolation temporaries. Other
        # locals remain independent symbols until their source assignment.
        self.env['interp_hi'] = self.env['interp_lo'] = self.env['zero']
        self.arrays['w'] = lambda i, j, k: field(i, j, k, 0)
        velocity = 'uvw'['XYZ'.index(component)]
        self.horizontal = f'interp_{velocity}_h'
        self.vertical = {f'interp_{velocity}_v'} | ({'interp_w_wall'} if component == 'Z' else set())
        self.vertical_functions = {(obj, direction): z3.Function(
            obj + '_' + direction, *([z3.IntSort()] * 4), z3.RealSort(), z3.IntSort(), z3.RealSort())
            for obj in self.vertical for direction in 'XYZ'}

    def execute(self, sequence, env, events, guard=z3.BoolVal(True)):
        env = dict(env)
        for statement in sequence:
            if statement[0] != 'call':
                env = super().execute([statement], env, events, guard)
                continue
            _, obj, direction, args = statement
            require(obj == self.horizontal or obj in self.vertical, 'unknown interpolation object')
            require(len(args) == (7 if obj == 'interp_w_wall' else 6), 'wrong interpolation arity')
            require(args[4] in ('interp_hi', 'interp_lo'), 'unsupported interpolation destination')
            point = [self.expression(arg, env) for arg in args[:4]]
            require(all(z3.is_int(arg) for arg in point), 'interpolation indices must be integers')
            flow = self.expression(args[5], env)
            selected = self.expression(args[6], env) if len(args) == 7 else env['vert_adv_type']
            require(z3.is_int(selected), 'interpolation selector must be an integer')
            if obj == self.horizontal:
                require(direction in 'XY', 'horizontal policy used for a vertical call')
                value = self.reconstruct(direction, *point, flow)
            else:
                value = self.vertical_functions[obj, direction](*point, flow, selected)
            env[args[4]] = value
            events.append((guard, obj, direction, point, flow))
        return env


def mass_average(model, component, direction, high):
    """Average the d-mass-flux along c onto the shared staggered face."""
    c, d = 'XYZ'.index(component), 'XY'.index(direction)
    point = [model.env[name] for name in 'ijk']
    point[d] += int(high)
    other = point.copy()
    other[c] -= 1

    def sample(p):
        return (model.arrays[('rho_u', 'rho_v')[d]](*p) *
                model.arrays[('mf_uy_inv', 'mf_vx_inv')[d]](p[0], p[1], 0))
    return (sample(point) + sample(other)) / 2, point
