#!/usr/bin/env python3
"""Compose all six reconstructed faces with the normal momentum tendency.

For each of 12x12 H/V policies and all three components, execute the complete
helper, including its metric/divergence/return, and apply the parsed wrapper's
spacing and RHS arithmetic. Compare to the negative divergence of independently
specified staggered face reconstructions. H/V blend fractions are independent.
Each final face identity is proved first. The parsed tail is then checked for
six arbitrary face values; universal substitution composes these two results.
Only identical terms/assumptions reuse a proved face lemma across scheme pairs.
The centered/centered helper instance is also checked, although production
dispatch selects the separately verified inline fast path for that pair.

Linear weights come from rational moments; WENO uses checked scalar kernels
through universal interface identities. Z-momentum assumes lo < k < hi.
Local map factors and required dz samples are positive. Array validity, caller
domain/dispatch, inverse-map buffer construction, floating point, convergence,
external boundary budgets and whole-update conservation are outside this proof.

usage: check_momentum_tendency.py [ERF_ROOT] [--horizontal NAME] [--vertical NAME]
       [--component X|Y|Z]
Selectors restrict mutation runs; the registered baseline covers every pair.
exit: 0 success, 1 failed, 77 z3 unavailable (CTest skip)
"""
import argparse
import sys
from itertools import product
from pathlib import Path

from advection_source import checked_constants
from check_momentum_dispatch import SCHEMES, enum_values, require
from check_momentum_horizontal_flux import prove
from check_momentum_wall_order import ENUM, MOM, failures, z3
from check_upwindall_z_reach import UPW_H, split_structs, strip_comments
from linear_z_interpolation import LINEAR, LinearZ, num, reference, sign
from momentum_horizontal import mass_average, read_helper
from momentum_tendency import TendencyModel, geometry, read_wrapper
from weno_z_interpolation import WenoZ, scalar


class Reconstruction:
    def __init__(self, scheme, direction, structs, constants, enums, field, blend):
        self.scheme, self.direction, self.field, self.blend = scheme, direction, field, blend
        name = SCHEMES[scheme]
        self.linear = scheme in LINEAR
        self.policy = (LinearZ(name, structs[name], constants, enums, direction) if self.linear else
                       WenoZ(name, structs[name], direction))
        i, j, k, n = z3.Ints('sample_i sample_j sample_k sample_component')
        flow = z3.Real('sample_flow')
        prove(self.actual(i, j, k, n, flow) == self.wanted(i, j, k, n, flow),
              f'{scheme}/{direction}: interpolation prerequisite', [])
        if not self.linear:
            prove(self.wanted(i, j, k, n, flow, canonical=True) == self.wanted(i, j, k, n, flow),
                  f'{scheme}/{direction}: signed-form prerequisite', [])

    def actual(self, i, j, k, n, flow):
        args = (self.field, i, j, k, n, flow)
        return self.policy.formula(*args, self.blend) if self.linear else self.policy.formula(*args)

    def wanted(self, i, j, k, n, flow, canonical=False):
        if self.linear:
            return reference(self.scheme, self.field, i, j, k, n, flow, self.blend, self.direction)
        if not canonical:
            return self.policy.reference(self.field, i, j, k, n, flow)
        left = z3.simplify(self.policy.reference(self.field, i, j, k, n, num(1)))
        right = z3.simplify(self.policy.reference(self.field, i, j, k, n, num(-1)))
        alpha = sign(flow)
        return (1 + alpha) / 2 * left + (1 - alpha) / 2 * right


def vertical_flux(model, component, high, reconstruction):
    point = [model.env[key] for key in 'ijk']
    point[2] += int(high)
    other = list(point)
    other['XYZ'.index(component)] -= 1
    mass = (model.arrays['rho_w'](*point) + model.arrays['rho_w'](*other)) / 2
    value = reconstruction.wanted(*point, 0, mass, canonical=True)
    if component == 'Z':
        lo, hi = [model.env[key] for key in ('lo_z_face', 'hi_z_face')]
        distance = z3.If(point[2] - lo < hi - point[2] + 1, point[2] - lo, hi - point[2] + 1)
        cap = reconstruction.scheme if reconstruction.scheme in ('Centered_2nd', 'Upwind_3rd') else 'Centered_4th'
        wall = lambda scheme: reference(scheme, reconstruction.field, *point, 0, mass, reconstruction.blend)
        value = z3.If(distance == 1, wall('Centered_2nd'), z3.If(distance == 2, wall(cap), value))
    return mass * value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--horizontal', choices=tuple(SCHEMES))
    parser.add_argument('--vertical', choices=tuple(SCHEMES))
    parser.add_argument('--component', choices=tuple('XYZ'))
    args = parser.parse_args()
    enums = enum_values(strip_comments((args.root / ENUM).read_text()))
    constants = checked_constants(args.root, {'zero': '0', 'myhalf': '1/2', 'three': '3', 'one': '1', 'two': '2'})
    text = strip_comments((args.root / MOM).read_text())
    components = (args.component,) if args.component else tuple('XYZ')
    bodies = {component: read_helper(text, component, with_tail=True) for component in components}
    wrappers = read_wrapper(text)
    structs = split_structs(strip_comments((args.root / UPW_H).read_text()))
    require(set(LINEAR.values()) | {'UPWINDALL'} <= set(structs), 'missing linear interpolation struct')
    for header in scalar.HEADERS:
        found = split_structs(strip_comments((args.root / header).read_text()))
        require(set(found) == scalar.EXPECTED_BY_HEADER[header], 'WENO struct set changed')
        structs.update(found)
    field = z3.Function('component_velocity', *([z3.IntSort()] * 4), z3.RealSort())
    blend_h, blend_v, flow = z3.Reals('horizontal_blend vertical_blend sample_flow')
    prove(sign(flow) * blend_h == z3.If(flow == 0, 0, z3.If(flow > 0, blend_h, -blend_h)),
          'arbitrary blend normalization prerequisite', [])
    horizontal = (args.horizontal,) if args.horizontal else tuple(SCHEMES)
    vertical = (args.vertical,) if args.vertical else tuple(SCHEMES)
    reconstructions = {(scheme, direction): Reconstruction(scheme, direction, structs, constants, enums,
                        field, blend_v if direction == 'Z' else blend_h)
                       for direction in 'XYZ' for scheme in (vertical if direction == 'Z' else horizontal)}
    wall = LinearZ('UPWINDALL', structs['UPWINDALL'], constants, enums)
    i, j, k, n = z3.Ints('sample_i sample_j sample_k sample_component')
    for scheme in LINEAR:
        prove(wall.formula(field, i, j, k, n, flow, blend_v, z3.IntVal(enums[scheme])) ==
              reference(scheme, field, i, j, k, n, flow, blend_v),
              f'wall/{scheme}: interpolation prerequisite', [])
    face_lemmas = {}
    for h, v, component in product(horizontal, vertical, components):
        hp = {direction: reconstructions[h, direction] for direction in 'XY'}
        vp = reconstructions[v, 'Z']
        model = TendencyModel(enums, constants, component, field,
                              lambda direction, *values: hp[direction].actual(*values))
        model.env['vert_adv_type'] = z3.IntVal(enums[v])
        model.env['dzInv'] = model.spacing(wrappers[component][0])
        velocity = 'uvw'['XYZ'.index(component)]
        model.vertical_functions[f'interp_{velocity}_v', 'Z'] = lambda *values: vp.actual(*values[:5])
        if component == 'Z':
            model.vertical_functions['interp_w_wall', 'Z'] = lambda x, y, z, q, f, selected: wall.formula(
                field, x, y, z, q, f, blend_v, selected)
        sequence, tail = bodies[component]
        final = model.execute(sequence, model.env, [])
        metric, spacing, assumptions = geometry(model, component)
        fluxes = {}
        complete = True
        # Prove all six final flux identities before composing with the
        # parsed tail. Reuse only identical normalized terms and assumptions;
        # this avoids repeating H-only/V-only lemmas across 144 mixed pairs.
        # Unknown/failed prerequisites never authorize a substitution.
        face_assumptions = ([model.env['lo_z_face'] < model.env['k'],
                             model.env['k'] < model.env['hi_z_face']] if component == 'Z' else [])
        for direction, high in product('XYZ', (False, True)):
            if direction == 'Z':
                flux = vertical_flux(model, component, high, vp)
            else:
                mass, point = mass_average(model, component, direction, high)
                flux = mass * hp[direction].wanted(*point, 0, mass, canonical=True)
            edge = 'hi' if high else 'lo'
            local = direction.lower() + 'flux_' + edge
            key = tuple(z3.simplify(term, som=True).sexpr()
                        for term in (final[local], flux, *face_assumptions))
            if key not in face_lemmas:
                before = len(failures)
                scheme = v if direction == 'Z' else h
                prove(final[local] == flux, f'{scheme}/{component}.{direction}/{edge}: reconstructed face prerequisite',
                      face_assumptions)
                face_lemmas[key] = len(failures) == before
            complete &= face_lemmas[key]
            # The consumer identity is universal in its six inputs. Proving
            # it for arbitrary face values, then substituting the identities
            # above, avoids expanding nonlinear kernels a second time. A sat
            # consumer result is an interface witness, not a field witness.
            fluxes[direction, high] = z3.Real('reconstructed_' + local)
            final[local] = fluxes[direction, high]
        if not complete:
            print(f'  skipping {h}/{v}/{component}: unproved reconstructed face')
            continue
        returned = model.finish(final, tail)
        actual = model.arithmetic(wrappers[component][1], {**model.env, 'helper_result': returned})
        expected = num(0)
        for direction, scale in [('X', model.env['dxInv'] * metric),
                                 ('Y', model.env['dyInv'] * metric), ('Z', spacing)]:
            expected -= (fluxes[direction, True] - fluxes[direction, False]) * scale
        prove(actual == expected, f'{h}/{v}/{component}: complete reconstructed momentum tendency', assumptions)
    total = len(failures) + len(scalar.failures)
    print(f'\n{total} check(s) failed')
    return 1 if total else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, z3.Z3Exception) as exc:
        print(f'FAIL momentum tendency source contract: {exc}')
        sys.exit(1)
