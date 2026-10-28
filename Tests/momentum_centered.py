"""Shared exact-real specification for the two centered momentum fast paths.

For component c and flux direction d, average the d-mass-flux samples along c
and the c-velocity samples along d to their shared control-volume face. This
geometric rule generates all eighteen fluxes without copying their C++ formulas.
The top Z/Z face uses the prescribed point product instead of an average.

Reciprocal identities assume positive map factors. Inverse-map construction and
flux reconstruction are proved separately, then the
caller substitutes the reconstructed fluxes into its parsed RHS. Array storage,
halo validity, launch synchronization, and floating-point rounding are not modeled.
"""
import re

from advection_source import Source, momentum_setup
from check_scalar_flux_divergence import compact, number, z3


def check_setup(prefix, stretched, constants, to_z3, prove_equal):
    """Consume setup through the dispatch gate and verify inverse-map writes."""
    for name, expression in momentum_setup(prefix, stretched).items():
        metric = to_z3(f'{name}(i,j,0)', {})
        prove_equal(to_z3(expression, constants), 1 / metric,
                    f'{name}_inv: setup stores the reciprocal of its positive map factor',
                    assumptions=[metric > 0])


def centered_flux(component, direction, high, to_z3):
    """Build the face product from stagger locations, independent of C++ locals."""
    c, d = 'XYZ'.index(component), 'xyz'.index(direction)

    def sample(name, offset, metric=False):
        indices = [base if step == 0 else f'{base}{step:+d}'
                   for base, step in zip('ijk', offset)]
        if metric:
            indices[2] = '0'
        return to_z3(f'{name}({",".join(indices)})', {})

    def mass(offset):
        value = sample(('rho_u', 'rho_v', 'omega')[d], offset)
        if d < 2:
            value *= sample(('mf_uy_inv', 'mf_vx_inv')[d], offset, metric=True)
        return value

    # A velocity component lives on the low c-face. Low d-flux therefore samples
    # mass at I and I-e_c; the high d-flux shifts both samples by +e_d.
    first = [0, 0, 0]
    first[d] = int(high)
    second = first.copy()
    second[c] -= 1
    neighbor = [0, 0, 0]
    neighbor[d] = 1 if high else -1
    velocity = 'uvw'[c]
    return ((mass(first) + mass(second)) / 2 *
            (sample(velocity, [0, 0, 0]) + sample(velocity, neighbor)) / 2)


def check_fluxes(body, component, constants, to_z3, prove_equal):
    """Parse every flux initializer, prove its geometry, and return both models."""
    declarations = Source(body).initializers()
    actual, expected = {}, {}
    for name, expression in declarations.items():
        match = re.fullmatch(r'([xyz])flux_(hi|lo)', name)
        if match is None:
            break
        direction, edge = match.groups()
        wanted = centered_flux(component, direction, edge == 'hi', to_z3)
        env = {**constants, **actual}
        if component == 'Z' and name == 'zflux_hi':
            select = re.fullmatch(r'\(k==hi_z_face\)\?([^?:]+):([^?:]+)', compact(expression))
            if select is None:
                raise ValueError('Z high flux must select its top-face product at hi_z_face')
            condition = z3.Int('k') == z3.Int('hi_z_face')
            value = z3.If(condition, to_z3(select[1], env), to_z3(select[2], env))
            wanted = z3.If(condition, to_z3('omega(i,j,k) * w(i,j,k)', {}), wanted)
        else:
            value = to_z3(expression, env)
        prove_equal(value, wanted,
                    f'{component}.{name}: flux matches the staggered centered face product')
        actual[name], expected[name] = value, wanted
    required = {f'{axis}flux_{edge}' for axis in 'xyz' for edge in ('hi', 'lo')}
    if set(actual) != required:
        raise ValueError('all six flux definitions must precede the metric and divergence')
    return actual, expected


def reconstructed_rhs(source_expr, rhs_expr, env, actual_fluxes, expected_fluxes,
                      expected_metric, component, to_z3, prove_equal):
    """Connect parsed reconstruction, divergence, and the final RHS assignment."""
    actual_env = {**env, **actual_fluxes}
    actual_env['advectionSrc'] = to_z3(source_expr, actual_env)
    rhs = to_z3(rhs_expr, actual_env)
    expected = number(0)
    for direction in 'xyz':
        difference = expected_fluxes[f'{direction}flux_hi'] - expected_fluxes[f'{direction}flux_lo']
        expected -= difference * env[f'd{direction}Inv'] * (expected_metric if direction != 'z' else 1)
    prove_equal(rhs, expected,
                f'{component}: complete inline RHS equals the reconstructed centered divergence')
