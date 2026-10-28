#!/usr/bin/env python3
"""Compose vertical reconstruction with normal X/Y-momentum fluxes.

For all five linear and seven WENO vertical policies, check both final Z fluxes
and shared-face equality. Independent stagger geometry averages rho_w along
the momentum component; velocity reconstruction acts along Z. Execute the
entire helper so earlier interpolation temporaries and later writes are visible.
Horizontal interpolators are arbitrary pure functions. Existing helper tails
and wrapper bindings are rechecked, but the full reconstructed RHS is not yet
composed here.

Linear formulas use independent rational moments. WENO scalar prerequisites
and universal interface identities supply its modular proof. All real samples,
flow signs and blend fractions, and all integer coordinates are covered under
valid-array/pure-kernel assumptions. There is no special wall-order branch in
these two helpers. This does not establish boundary fill, storage/halo validity,
floating-point behavior, convergence on stretched grids, or whole-update balance.

usage: check_momentum_xy_vertical_flux.py [ERF_ROOT] [--scheme NAME] [--component X|Y]
Selectors restrict mutation runs; the registered baseline covers all cases.
exit: 0 success, 1 check failed, 77 z3 unavailable (CTest skip)
"""
import argparse
import sys
from pathlib import Path

from advection_source import checked_constants
import check_momentum_divergence as divergence
from check_momentum_dispatch import enum_values, require
from check_momentum_horizontal_flux import prove
from check_momentum_wall_order import ENUM, MOM, failures, z3
from check_upwindall_z_reach import UPW_H, split_structs, strip_comments
from linear_z_interpolation import LINEAR, LinearZ, num, reference, sign
from momentum_horizontal import HorizontalModel, read_helper
from weno_z_interpolation import WENO, WenoZ, scalar


def vertical_mass(model, component, high):
    """Place vertical mass flux on the c-momentum control-volume Z face."""
    point = [model.env[name] for name in 'ijk']
    point[2] += int(high)
    other = point.copy()
    other['XY'.index(component)] -= 1
    return (model.arrays['rho_w'](*point) + model.arrays['rho_w'](*other)) / 2, point


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--scheme', choices=tuple(LINEAR) + tuple(WENO))
    parser.add_argument('--component', choices=tuple('XY'))
    args = parser.parse_args()
    enums = enum_values(strip_comments((args.root / ENUM).read_text()))
    constants = checked_constants(args.root, {'zero': '0', 'myhalf': '1/2', 'three': '3', 'one': '1', 'two': '2'})
    text = strip_comments((args.root / MOM).read_text())
    components = (args.component,) if args.component else tuple('XY')
    bodies = {component: read_helper(text, component) for component in components}
    for component in components:
        velocity = 'uv'['XY'.index(component)]
        metric = f'1 / (mf_{velocity}x_inv(i,j,0) * mf_{velocity}y_inv(i,j,0))'
        divergence.check_component(text, component, metric, {key: num(value) for key, value in constants.items()})
    divergence.check_wrapper(text)
    structs = split_structs(strip_comments((args.root / UPW_H).read_text()))
    require(set(LINEAR.values()) <= set(structs), 'missing linear interpolation struct')
    for header in scalar.HEADERS:
        found = split_structs(strip_comments((args.root / header).read_text()))
        require(set(found) == scalar.EXPECTED_BY_HEADER[header], 'WENO struct set changed')
        structs.update(found)
    schemes = {**LINEAR, **WENO}
    if args.scheme:
        schemes = {args.scheme: schemes[args.scheme]}
    field = z3.Function('velocity', *([z3.IntSort()] * 4), z3.RealSort())
    i, j, k, n = z3.Ints('sample_i sample_j sample_k sample_component')
    flow, blend = z3.Reals('mass_flux blend')
    prove(sign(flow) * blend == z3.If(flow == 0, 0, z3.If(flow > 0, blend, -blend)),
          'vertical XY flow normalization yields signed blend', [])
    for scheme, name in schemes.items():
        linear = scheme in LINEAR
        policy = LinearZ(name, structs[name], constants, enums) if linear else WenoZ(name, structs[name])

        def actual(x, y, z, component, mass, selected):
            return (policy.formula(field, x, y, z, component, mass, blend) if linear else
                    policy.formula(field, x, y, z, component, mass))

        def wanted(point, component, mass, canonical=False):
            if linear:
                return reference(scheme, field, *point, component, mass, blend)
            if not canonical:
                return policy.reference(field, *point, component, mass)
            left = z3.simplify(policy.reference(field, *point, component, num(1)))
            right = z3.simplify(policy.reference(field, *point, component, num(-1)))
            alpha = sign(mass)
            return (1 + alpha) / 2 * left + (1 - alpha) / 2 * right

        prove(actual(i, j, k, n, flow, None) == wanted([i, j, k], n, flow),
              f'{scheme}/Z: source interpolation matches reconstruction', [])
        if not linear:
            prove(wanted([i, j, k], n, flow, True) == wanted([i, j, k], n, flow),
                  f'{scheme}/Z: canonical signed reconstruction matches sign cases', [])
        for component in components:
            horizontal = {direction: z3.Function('horizontal_' + component + '_' + direction,
                          *([z3.IntSort()] * 4), z3.RealSort(), z3.RealSort()) for direction in 'XY'}
            model = HorizontalModel(enums, constants, component, field,
                                    lambda direction, *values: horizontal[direction](*values))
            velocity = 'uv'['XY'.index(component)]
            model.env['vert_adv_type'] = z3.IntVal(enums[scheme])
            # Only the actual vertical method gets the parsed reconstruction.
            # A wrong method/object still denotes a distinct arbitrary function
            # or an unsupported binding and cannot inherit this implementation.
            model.vertical_functions[f'interp_{velocity}_v', 'Z'] = actual
            final = model.execute(bodies[component], model.env, [])
            for edge in ('hi', 'lo'):
                mass, point = vertical_mass(model, component, edge == 'hi')
                expected = mass * wanted(point, 0, mass, True)
                prove(final[f'zflux_{edge}'] == expected,
                      f'{scheme}/{component}/{edge}: vertical flux matches reconstruction', [])
            coordinate = model.env['k']
            neighbor = z3.substitute(final['zflux_lo'], (coordinate, coordinate + 1))
            prove(final['zflux_hi'] == neighbor,
                  f'{scheme}/{component}: shared vertical flux agrees', [])
    total = len(failures) + len(scalar.failures) + len(divergence.failures)
    print(f'\n{total} check(s) failed')
    return 1 if total else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, z3.Z3Exception) as exc:
        print(f'FAIL XY-momentum vertical-flux source contract: {exc}')
        sys.exit(1)
