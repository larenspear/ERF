#!/usr/bin/env python3
"""Compose all seven WENO Z reconstructions with normal vertical momentum flux.

Discharge source-derived scalar candidate/weight obligations, then prove the
sample/call/flow assembly for any pure scalar kernel of those ordered samples.
This modular proof applies to the checked nonlinear source expression. Wall
reconstructions use explicit source-derived linear arithmetic and independent
rational weights. Both final fluxes and shared-face consistency are checked.

All integer lo < k < hi, all real sample values and flow signs (including zero)
are covered. WENO ignores the supplied blend fraction. Scalar proofs assume
eps>0, as in both source precision branches. Scalar prerequisites independently
specify smoothness, ideal reconstruction and nonlinear weight policies; they
prove actual nonlinear constant/affine exactness and all-data division definedness.
The interface proof does not
establish WENO convergence. Dispatch/caller bindings are checked separately; floating
point, horizontal fluxes, complete divergence and halo validity are excluded.

usage: check_momentum_weno_z_flux.py [ERF_ROOT]
exit: 0 success, 1 a check fails, 77 z3 unavailable (CTest skip)
"""
import sys
from pathlib import Path

from advection_source import checked_constants
from check_momentum_dispatch import enum_values, require
from check_momentum_wall_order import ENUM, MOM, Model, failures, prove, read_helper, z3
from check_upwindall_z_reach import UPW_H, split_structs, strip_comments
from linear_z_interpolation import LinearZ, reference
from weno_z_interpolation import WENO, WenoZ, scalar


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    enums = enum_values(strip_comments((root / ENUM).read_text()))
    constants = checked_constants(root, {'zero': '0', 'myhalf': '1/2', 'three': '3'})
    structs = {}
    for header in scalar.HEADERS:
        found = split_structs(strip_comments((root / header).read_text()))
        require(set(found) == scalar.EXPECTED_BY_HEADER[header], 'WENO struct set changed')
        structs.update(found)
    policies = {name: WenoZ(name, body) for name, body in structs.items()}
    wall_body = split_structs(strip_comments((root / UPW_H).read_text()))['UPWINDALL']
    wall = LinearZ('UPWINDALL', wall_body, constants, enums)
    high, low, pure_regions = read_helper(strip_comments((root / MOM).read_text()))
    field = z3.Function('velocity', *([z3.IntSort()] * 4), z3.RealSort())
    i, j, q, component = z3.Ints('sample_i sample_j sample_q component')
    flow, blend = z3.Reals('mass_flux blend')
    print('\nWENO Z assembly: arbitrary pure checked kernels, samples, and mass-flux sign')
    for scheme in ('Centered_2nd', 'Centered_4th'):
        actual = wall.formula(field, i, j, q, component, flow, blend, z3.IntVal(enums[scheme]))
        prove(actual == reference(scheme, field, i, j, q, component, flow, blend),
              f'wall/{scheme}: source reconstruction matches independent weights', [])
    for scheme, name in WENO.items():
        policy = policies[name]
        prove(policy.formula(field, i, j, q, component, flow) ==
              policy.reference(field, i, j, q, component, flow),
              f'{name}: source Z assembly matches oriented scalar kernels', [])

        model = Model(enums, constants)
        model.env['vert_adv_type'] = z3.IntVal(enums[scheme])
        model.arrays['w'] = lambda x, y, z: field(x, y, z, 0)
        model.interp = {
            'interp_w_v': lambda x, y, z, n, f, selected: policy.formula(field, x, y, z, n, f),
            'interp_w_wall': lambda x, y, z, n, f, selected: wall.formula(field, x, y, z, n, f, blend, selected),
        }
        for expression in pure_regions:
            model.expression(expression, model.env)
        env = model.execute([high], model.env, [])
        env = model.execute([low], env, [])
        i, j, k, lo, hi = [model.env[key] for key in ('i', 'j', 'k', 'lo_z_face', 'hi_z_face')]
        for edge, face in [('hi', k + 1), ('lo', k)]:
            avg = (model.arrays['rho_w'](i, j, face - 1) + model.arrays['rho_w'](i, j, face)) / 2
            distance = z3.If(face - lo < hi - face + 1, face - lo, hi - face + 1)
            wanted = avg * z3.If(distance == 1,
                                reference('Centered_2nd', field, i, j, face, 0, avg, blend),
                                z3.If(distance == 2,
                                      reference('Centered_4th', field, i, j, face, 0, avg, blend),
                                      policy.reference(field, i, j, face, 0, avg)))
            prove(env['zflux_' + edge] == wanted, f'{scheme}/{edge}: composed WENO flux matches reconstruction',
                  [lo < k, k < hi])
        neighbor = z3.substitute(env['zflux_lo'], (k, k + 1))
        prove(env['zflux_hi'] == neighbor, f'{scheme}: reconstructed shared vertical flux agrees',
              [lo < k, k + 1 < hi])
    total = len(failures) + len(scalar.failures)
    print(f'\n{total} check(s) failed')
    return 1 if total else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, NameError, KeyError, z3.Z3Exception) as exc:
        print(f'FAIL momentum WENO Z-flux source contract: {exc}')
        sys.exit(1)
