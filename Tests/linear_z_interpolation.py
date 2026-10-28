"""Source-derived linear interpolation and an independent rational-weight oracle.

The source model retains constructors, signatures, load offsets, Evaluate call
bindings, normalization/blending, and the complete returned expression. The
oracle derives uniform-index finite-volume reconstruction weights by solving
moment equations over exact rationals. This algebraic specification does not
assert how ERF's velocity samples should be interpreted on a stretched grid.
"""
import re
from fractions import Fraction

from advection_source import NORMALIZE, Source, pure_expression
from check_momentum_dispatch import compact, require
from check_upw_stencils import num, parse_evaluate, struct_env, to_z3, z3
from check_upwindall_z_reach import brace_block, cell_offset, enum_set

LINEAR = {'Centered_2nd': 'CENTERED2', 'Upwind_3rd': 'UPWIND3',
          'Centered_4th': 'CENTERED4', 'Upwind_5th': 'UPWIND5', 'Centered_6th': 'CENTERED6'}


def weights(offsets):
    """Reconstruct p(0) from averages of p over [offset, offset+1]."""
    size = len(offsets)
    matrix = [[Fraction((offset + 1) ** (degree + 1) - offset ** (degree + 1), degree + 1)
               for offset in offsets] + [Fraction(degree == 0)] for degree in range(size)]
    for col in range(size):
        pivot = next(row for row in range(col, size) if matrix[row][col])
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        divisor = matrix[col][col]
        matrix[col] = [entry / divisor for entry in matrix[col]]
        for row in range(size):
            if row != col:
                factor = matrix[row][col]
                matrix[row] = [entry - factor * other for entry, other in zip(matrix[row], matrix[col])]
    return tuple((offset, matrix[index][-1]) for index, offset in enumerate(offsets))


WEIGHTS = {order: weights(tuple(range(-order // 2, order // 2))) for order in (2, 4, 6)}
UPWIND_WEIGHTS = {order: (weights(tuple(range(-(order + 1) // 2, (order - 1) // 2))),
                          weights(tuple(range(-(order - 1) // 2, (order + 1) // 2))))
                  for order in (3, 5)}


def sign(value):
    return z3.If(value > 0, num(1), z3.If(value < 0, num(-1), num(0)))


def reference(scheme, field, i, j, q, component, flow, blend, direction='Z'):
    axis = 'XYZ'.index(direction)

    def reconstruct(row):
        result = num(0)
        for offset, coefficient in row:
            point = [i, j, q]
            point[axis] += offset
            result += num(coefficient) * field(*point, component)
        return result
    if scheme.startswith('Centered_'):
        return reconstruct(WEIGHTS[int(scheme.split('_')[1][0])])
    left, right = UPWIND_WEIGHTS[int(scheme.split('_')[1][0])]
    # The entrypoint separately proves this normalization equals +blend,
    # -blend, or zero according to flow direction. Keeping that established
    # term canonical avoids costly nonlinear search in composed flux proofs.
    alpha = sign(flow) * blend
    return (1 + alpha) / 2 * reconstruct(left) + (1 - alpha) / 2 * reconstruct(right)


class LinearZ:
    # Keep the original name/default for the existing vertical checkers.
    def __init__(self, name, body, constants, enums, direction='Z'):
        require(direction in ('X', 'Y', 'Z'), 'unsupported interpolation direction')
        self.name, self.enums = name, enums
        self.dispatcher = name == 'UPWINDALL'
        self.upwind = name in ('UPWIND3', 'UPWIND5', 'UPWINDALL')
        # Constructor initializers are part of the source-to-model binding.
        constructor = re.findall(rf'\b{name}\s*\(([^()]*)\)\s*:\s*([^{{}}]*)\{{([^{{}}]*)\}}', body)
        require(len(constructor) == 1, f'{name}: expected one supported constructor')
        params, initializers, constructor_body = constructor[0]
        frac_param = 'constamrex::Realupw_frac' if self.upwind else 'constamrex::Real'
        require(compact(params) == 'constamrex::Array4<constamrex::Real>&phi,' + frac_param,
                f'{name}: constructor parameter bindings changed')
        expected_init = 'm_phi(phi)' + (',m_upw_frac(upw_frac)' if self.upwind else '')
        require(compact(initializers) == expected_init and not constructor_body.strip(),
                f'{name}: constructor field bindings/body changed')

        matches = list(re.finditer(rf'void\s+InterpolateIn{direction}\s*\(([^()]*)\)\s*const\s*\{{', body))
        require(len(matches) == 1, f'{name}: expected one {direction} interpolation definition')
        signature = compact(matches[0][1])
        expected = ('constint&i,constint&j,constint&k,constint&qty_index,amrex::Real&val_lo,amrex::Real'
                    + ('upw_lo' if self.upwind else '') + (',constAdvTypeadv_type' if self.dispatcher else ''))
        require(signature == expected, f'{name}: {direction} interpolation parameter bindings changed')
        source = Source(brace_block(body, matches[0].end() - 1))
        self.early = None
        if self.dispatcher:
            source.take(r'if\s*\(\s*adv_type\s*==\s*AdvType::Centered_2nd\s*\)\s*\{')
            self.early = source.take(r'val_lo\s*=\s*([^;{}]+);\s*return\s*;\s*\}\s*else\s*\{').group(1)
            pure_expression(self.early)
        self.loads = []
        for local, expression in source.initializers().items():
            require(re.fullmatch(r's(?:[mp]\d+)?', local) is not None, f'{name}: unsupported interpolation local {local}')
            gate = None
            if '?' in expression:
                require(self.dispatcher, 'conditional loads only supported in UPWINDALL')
                condition, arms = expression.split('?', 1)
                read, separator, false = arms.partition(':')
                require(separator and compact(false) == 'amrex::Real(0)', 'conditional load must have a zero false arm')
                gate = enum_set(condition)
                require(gate <= set(LINEAR), 'unknown load-gate enum')
            else:
                read = expression
            self.loads.append((local, self.read_offsets(read), gate))
        require(bool(self.loads), f'{name}: missing stencil loads')
        if self.upwind:
            source.take(NORMALIZE)
            source.take(r'upw_lo\s*\*=\s*m_upw_frac\s*;')
        call = source.take(r'val_lo\s*=\s*Evaluate\s*\((\w+(?:\s*,\s*\w+)*)\)\s*;').group(1)
        self.args = [arg.strip() for arg in call.split(',')]
        if self.dispatcher:
            source.take(r'\}')
        source.done()

        matches = list(re.finditer(r'amrex::Real\s+Evaluate\s*\(([^()]*)\)\s*const\s*\{', body))
        require(len(matches) == 1, f'{name}: expected one Evaluate definition')
        params = []
        for param in matches[0][1].split(','):
            match = re.fullmatch(r'\s*const\s+amrex::Real\s*&\s*(\w+)\s*', param)
            if match:
                params.append(match[1])
            else:
                require(self.dispatcher and compact(param) == 'constAdvTypeadv_type', 'unsupported Evaluate parameter')
                params.append('adv_type')
        require(len(params) == len(set(params)) and len(params) == len(self.args), 'Evaluate arity/parameter uniqueness changed')
        if self.dispatcher:
            require(params.count('adv_type') == 1 and self.args[params.index('adv_type')] == 'adv_type',
                    'Evaluate enum binding changed')
        self.symbols = {param: z3.Real(name + '_parameter_' + param) for param in params if param != 'adv_type'}
        env = {key: num(value) for key, value in constants.items()}
        try:
            self.constants = struct_env(body, env)
            _, actual_params, self.variants = parse_evaluate(body, env, self.symbols)
        except (NameError, KeyError) as exc:
            raise ValueError(f'{name}: unsupported scalar dependency in Evaluate: {exc}') from exc
        require(actual_params == params, 'Evaluate parser lost parameter order')
        self.params = params

    @staticmethod
    def read_offsets(expression):
        match = re.fullmatch(r'\s*m_phi\s*\(([^()]*)\)\s*', expression)
        require(match is not None, f'unsupported field read: {expression}')
        args = [arg.strip() for arg in match[1].split(',')]
        require(len(args) == 4 and args[3] == 'qty_index', 'field read component binding changed')
        return tuple(cell_offset(arg, axis) for arg, axis in zip(args[:3], 'ijk'))

    def formula(self, field, i, j, q, component, flow, blend, selected=None):
        def read(offsets):
            return field(i + offsets[0], j + offsets[1], q + offsets[2], component)

        def branch(scheme):
            env = {'upw_lo': sign(flow) * blend}
            for local, offsets, gate in self.loads:
                env[local] = read(offsets) if gate is None or scheme in gate else num(0)
            if self.dispatcher and scheme == 'Centered_2nd':
                early_env = dict(self.constants)
                def replace(match):
                    local = '_read_' + str(len(early_env))
                    early_env[local] = read(self.read_offsets(match[0]))
                    return local
                expression = re.sub(r'm_phi\s*\([^()]*\)', replace, self.early)
                try:
                    return to_z3(expression, early_env)
                except NameError as exc:
                    raise ValueError(f'unsupported early-return dependency: {exc}') from exc
            substitutions = []
            for param, arg in zip(self.params, self.args):
                if param != 'adv_type':
                    require(arg in env, f'unknown Evaluate argument {arg}')
                    substitutions.append((self.symbols[param], env[arg]))
            return z3.substitute(self.variants[scheme if self.dispatcher else ''], *substitutions)

        if not self.dispatcher:
            return z3.simplify(branch(None))
        require(selected is not None, 'UPWINDALL needs an explicit scheme')
        # UPWINDALL initializes iv to zero for an unsupported selector. The
        # momentum selector proof separately prohibits those reachable choices.
        result = num(0)
        for scheme in LINEAR:
            result = z3.If(selected == self.enums[scheme], branch(scheme), result)
        return z3.simplify(result)
