#!/usr/bin/env python3
"""Compose X/Y reconstruction with all three normal momentum components.

Check all five linear and seven WENO horizontal policies. The independent
specification averages mass flux times the matching inverse map factor along
the momentum component, then reconstructs velocity along the flux direction.
Prove each final horizontal flux and shared-face equality. Complete helper
execution order is retained; vertical reconstruction is an arbitrary pure
function. Rerun the existing divergence/wrapper checks to bind the helpers.

Linear arithmetic uses independent rational moments; WENO uses checked scalar
kernels through universal interface identities. This is exact-real/integer
algebra, not a convergence, floating-point, halo-validity, or whole-update proof.
Vertical reconstruction and boundary policy are checked in separate suites.

usage: check_momentum_horizontal_flux.py [ERF_ROOT] [--scheme NAME] [--component X|Y|Z]
The selectors restrict mutation runs; the registered baseline covers all cases.
exit: 0 success, 1 check failed, 77 z3 unavailable (CTest skip)
"""
import argparse
import sys
from pathlib import Path

from advection_source import checked_constants
import check_momentum_divergence as divergence
from check_momentum_dispatch import enum_values, require
from check_momentum_wall_order import ENUM, MOM, failures, prove as check_claim, z3
from check_upwindall_z_reach import UPW_H, split_structs, strip_comments
from linear_z_interpolation import LINEAR, LinearZ, num, reference, sign
from momentum_horizontal import HorizontalModel, mass_average, read_helper
from weno_z_interpolation import WENO, WenoZ, scalar


def prove(claim, description, assumptions):
    # Canonicalize polynomial products before asking the solver. In particular,
    # the Z helper factors a shared map multiplier out of its mass average;
    # the geometric oracle distributes it over samples. Ring normalization
    # keeps equivalent flow-sign tests identical without adding assumptions.
    check_claim(z3.simplify(claim, som=True), description, assumptions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--scheme', choices=tuple(LINEAR) + tuple(WENO))
    parser.add_argument('--component', choices=tuple('XYZ'))
    args = parser.parse_args()
    enums = enum_values(strip_comments((args.root / ENUM).read_text()))
    constants = checked_constants(args.root, {'zero': '0', 'myhalf': '1/2', 'three': '3', 'one': '1', 'two': '2'})
    text = strip_comments((args.root / MOM).read_text())
    components = (args.component,) if args.component else tuple('XYZ')
    bodies = {component: read_helper(text, component) for component in components}
    metrics = {'X': '1 / (mf_ux_inv(i,j,0) * mf_uy_inv(i,j,0))',
               'Y': '1 / (mf_vx_inv(i,j,0) * mf_vy_inv(i,j,0))',
               'Z': 'mf_mx(i,j,0) * mf_my(i,j,0)'}
    for component in components:
        divergence.check_component(text, component, metrics[component],
                                   {key: num(value) for key, value in constants.items()})
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
          'horizontal flow normalization yields signed blend', [])
    for scheme, name in schemes.items():
        linear = scheme in LINEAR
        policies = {direction: LinearZ(name, structs[name], constants, enums, direction) if linear else
                    WenoZ(name, structs[name], direction) for direction in 'XY'}

        def actual(direction, x, y, z, component, mass):
            policy = policies[direction]
            return (policy.formula(field, x, y, z, component, mass, blend) if linear else
                    policy.formula(field, x, y, z, component, mass))

        def wanted(direction, point, component, mass, canonical=False):
            if linear:
                return reference(scheme, field, *point, component, mass, blend, direction)
            policy = policies[direction]
            if not canonical:
                return policy.reference(field, *point, component, mass)
            # The independently stated sign-case oracle is proved equal to
            # this form below. Reuse that lemma in the larger flux queries.
            left = z3.simplify(policy.reference(field, *point, component, num(1)))
            right = z3.simplify(policy.reference(field, *point, component, num(-1)))
            alpha = sign(mass)
            return (1 + alpha) / 2 * left + (1 - alpha) / 2 * right

        for direction in 'XY':
            prove(actual(direction, i, j, k, n, flow) == wanted(direction, [i, j, k], n, flow),
                  f'{scheme}/{direction}: source interpolation matches reconstruction', [])
            if not linear:
                prove(wanted(direction, [i, j, k], n, flow, True) ==
                      wanted(direction, [i, j, k], n, flow),
                      f'{scheme}/{direction}: canonical signed reconstruction matches sign cases', [])
        for component in components:
            model = HorizontalModel(enums, constants, component, field, actual)
            final = model.execute(bodies[component], model.env, [])
            assumptions = ([model.env['lo_z_face'] < model.env['k'],
                            model.env['k'] < model.env['hi_z_face']] if component == 'Z' else [])
            for direction in 'XY':
                axis = direction.lower()
                for edge in ('hi', 'lo'):
                    mass, point = mass_average(model, component, direction, edge == 'hi')
                    expected = mass * wanted(direction, point, 0, mass, True)
                    prove(final[f'{axis}flux_{edge}'] == expected,
                          f'{scheme}/{component}.{axis}/{edge}: horizontal flux matches reconstruction', assumptions)
                coordinate = model.env['i' if direction == 'X' else 'j']
                neighbor = z3.substitute(final[f'{axis}flux_lo'], (coordinate, coordinate + 1))
                prove(final[f'{axis}flux_hi'] == neighbor,
                      f'{scheme}/{component}.{axis}: shared horizontal flux agrees', assumptions)
    total = len(failures) + len(scalar.failures) + len(divergence.failures)
    print(f'\n{total} check(s) failed')
    return 1 if total else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, z3.Z3Exception) as exc:
        print(f'FAIL momentum horizontal-flux source contract: {exc}')
        sys.exit(1)
