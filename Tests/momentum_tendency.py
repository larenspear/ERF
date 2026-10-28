"""Read and compose complete normal momentum helper/wrapper arithmetic.

Reuse the full flux statement interpreter, then evaluate the consumed metric,
divergence and return expressions in its final environment. The wrapper reader
preserves its sign and spacing arithmetic while enforcing component/argument,
constructor and launch bindings. This remains a deliberately narrow C++ reader.
"""
import ast
import re

from advection_source import LAMBDA, Source, pure_expression, spacing_aliases
from check_momentum_dispatch import SHARED_ARGS, compact, function, require
from linear_z_interpolation import num
from momentum_horizontal import HorizontalModel, z3


class TendencyModel(HorizontalModel):
    def __init__(self, *args):
        super().__init__(*args)
        for name in ('mf_ux_inv', 'mf_vy_inv'):
            self.arrays[name] = z3.Function(name, *([z3.IntSort()] * 3), z3.RealSort())
        self.dz = z3.Function('dz', z3.IntSort(), z3.RealSort())

    def arithmetic(self, expression, env):
        """Evaluate real arithmetic; array indices retain integer semantics."""
        pure_expression(expression)
        bindings = dict(env)

        def read(match):
            name = '_read_' + str(len(bindings))
            bindings[name] = self.expression(match[0], env)
            return name

        expression = re.sub(r'\b(?:' + '|'.join(self.arrays) + r')\s*\([^()]*\)', read, expression)
        expression = '(' + expression.replace('amrex::Real', 'Real') + ')'

        def visit(node):
            if isinstance(node, ast.Name):
                require(node.id in bindings and z3.is_arith(bindings[node.id]),
                        f'unsupported tendency dependency: {node.id}')
                return bindings[node.id]
            if isinstance(node, ast.Constant):
                if type(node.value) is int:
                    return z3.IntVal(node.value)
                if type(node.value) is float:
                    return num(ast.get_source_segment(expression, node))
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                value = visit(node.operand)
                return value if isinstance(node.op, ast.UAdd) else -value
            if isinstance(node, ast.BinOp):
                left, right = visit(node.left), visit(node.right)
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div):
                    # Do not silently turn C++ integer truncation into real
                    # division. No live helper/wrapper needs integer division.
                    require(z3.is_real(left) or z3.is_real(right), 'integer division is unsupported')
                    return left / right
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'Real'
                    and len(node.args) == 1 and not node.keywords):
                value = visit(node.args[0])
                return z3.ToReal(value) if z3.is_int(value) else value
            raise ValueError(f'unsupported tendency arithmetic: {expression}')

        return visit(ast.parse(expression, mode='eval').body)

    def spacing(self, expression):
        """Read the wrapper's real division and optional integer conditional."""
        pure_expression(expression)
        if '?' in expression:
            condition, arms = expression.split('?', 1)
            yes, separator, no = arms.partition(':')
            require(bool(separator) and '?' not in arms, 'unsupported spacing conditional')
            return z3.If(self.expression(condition, self.env, condition=True),
                         self.spacing(yes), self.spacing(no))
        bindings = dict(self.env)

        def read(match):
            index = self.expression(match[1], self.env)
            require(z3.is_int(index), 'spacing index must be integer')
            name = '_dz_' + str(len(bindings))
            bindings[name] = self.dz(index)
            return name

        expression = re.sub(r'\bdz_ptr\s*\[([^\[\]]+)\]', read, expression)
        return self.arithmetic(expression, bindings)

    def finish(self, final, tail):
        env = dict(final)
        env['mfsq'] = self.arithmetic(tail[0], env)
        env['advectionSrc'] = self.arithmetic(tail[1], env)
        return self.arithmetic(tail[2], env)


def read_wrapper(text):
    body = function(text, 'AdvectionSrcForMomWrapper_N', SHARED_ARGS,
                    ('InterpType_H', 'InterpType_V', 'WallInterpType'))
    source = Source(body)
    for velocity in 'uvw':
        for direction, policy in (('h', 'H'), ('v', 'V')):
            source.take(rf'InterpType_{policy}\s+interp_{velocity}_{direction}\s*\('
                        rf'\s*{velocity}\s*,\s*upw_frac_{direction}\s*\)\s*;')
    source.take(r'WallInterpType\s+interp_w_wall\s*\(\s*w\s*,\s*upw_frac_v\s*\)\s*;')
    start = source.pos
    source.take(r'auto\s+dxInv\s*=\s*[^;]+;\s*auto\s+dyInv\s*=\s*[^;]+;')
    spacing_aliases(body[start:source.pos], 2)
    source.take(r'auto\s+dz_ptr\s*=\s*stretched_dz_d\.data\(\s*\)\s*;')
    result = {}
    for component, velocity in zip('XYZ', 'uvw'):
        source.take(rf'amrex::ParallelFor\s*\(\s*bx{component.lower()}\s*,' + LAMBDA)
        kernel = Source(source.take(r'([^{}]*)\}\s*\)\s*;').group(1))
        spacing = kernel.take(r'auto\s+dzInv\s*=\s*([^;{}]+);').group(1)
        value = kernel.take(rf'rho_{velocity}_rhs\s*\(\s*i\s*,\s*j\s*,\s*k\s*\)'
                            r'\s*=\s*([^;{}]+);').group(1)
        kernel.done()
        pattern = rf'\bAdvectionSrcFor{component}Mom_N\s*\(([^()]*)\)'
        calls = list(re.finditer(pattern, value))
        require(len(calls) == 1, f'{component}: expected one matching helper call')
        args = ['i', 'j', 'k', 'rho_u', 'rho_v', 'rho_w']
        args += ['w'] if component == 'Z' else []
        args += [f'interp_{velocity}_h', f'interp_{velocity}_v']
        args += ['interp_w_wall'] if component == 'Z' else []
        args += {'X': ['mf_ux_inv', 'mf_uy_inv', 'mf_vx_inv'],
                 'Y': ['mf_uy_inv', 'mf_vx_inv', 'mf_vy_inv'],
                 'Z': ['mf_mx', 'mf_my', 'mf_uy_inv', 'mf_vx_inv',
                       'vert_adv_type', 'lo_z_face', 'hi_z_face']}[component]
        args += ['dxInv', 'dyInv', 'dzInv']
        require([compact(arg) for arg in calls[0][1].split(',')] == args,
                f'{component}: wrapper helper argument bindings changed')
        value = re.sub(pattern, 'helper_result', value)
        pure_expression(value)
        result[component] = spacing, value
    source.done()
    return result


def geometry(model, component):
    """Independent component metric, spacing and physical preconditions."""
    i, j, k = [model.env[key] for key in 'ijk']
    if component in 'XY':
        velocity = 'uv'['XY'.index(component)]
        mx = model.arrays[f'mf_{velocity}x_inv'](i, j, 0)
        my = model.arrays[f'mf_{velocity}y_inv'](i, j, 0)
        metric, spacing = 1 / (mx * my), 1 / model.dz(k)
        assumptions = [mx > 0, my > 0, model.dz(k) > 0]
    else:
        mx, my = [model.arrays['mf_m' + axis](i, j, 0) for axis in 'xy']
        metric = mx * my
        spacing = z3.If(k == 0, 1 / model.dz(k), 2 / (model.dz(k) + model.dz(k - 1)))
        assumptions = [mx > 0, my > 0, model.dz(k) > 0, model.dz(k - 1) > 0,
                       model.env['lo_z_face'] < k, k < model.env['hi_z_face']]
    return metric, spacing, assumptions
