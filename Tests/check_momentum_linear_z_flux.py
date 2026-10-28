#!/usr/bin/env python3
"""Compose live linear Z reconstructions with the normal vertical momentum flux.

Check five standalone policies and UPWINDALL against independently derived
rational reconstruction weights, then substitute the source formulas into the
live Z-momentum helper. Prove both vertical fluxes and shared-face consistency
for all integer lo < k < hi, arbitrary data, and arbitrary real blend fraction.
Flow normalization includes positive, negative, and zero mass flux.

This is exact-real algebra, not a convergence claim on stretched meshes or a
floating-point/storage proof. WENO, horizontal flux arithmetic, and the complete
momentum divergence remain outside this check. Dispatch/wrapper object bindings
are checked separately; constructors and Z method/Evaluate bindings are checked
here. The caller's interior-domain range remains a precondition.

usage: check_momentum_linear_z_flux.py [ERF_ROOT]
exit: 0 success, 1 a check fails, 77 z3 unavailable (CTest skip)
"""
import sys
from pathlib import Path

from advection_source import checked_constants
from check_momentum_dispatch import enum_values, require
from check_momentum_wall_order import ENUM, MOM, Model, failures, prove, read_helper, z3
from check_upwindall_z_reach import UPW_H, split_structs, strip_comments
from linear_z_interpolation import LINEAR, LinearZ, reference, sign


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    enums = enum_values(strip_comments((root / ENUM).read_text()))
    constants = checked_constants(root, {'zero': '0', 'myhalf': '1/2', 'three': '3'})
    structs = split_structs(strip_comments((root / UPW_H).read_text()))
    require(set(LINEAR.values()) | {'UPWINDALL'} <= set(structs), 'missing linear reconstruction struct')
    policies = {name: LinearZ(name, structs[name], constants, enums) for name in (*LINEAR.values(), 'UPWINDALL')}
    high, low, pure_regions = read_helper(strip_comments((root / MOM).read_text()))
    field = z3.Function('velocity', *([z3.IntSort()] * 4), z3.RealSort())
    i, j, q, component = z3.Ints('sample_i sample_j sample_q component')
    flow, blend = z3.Reals('mass_flux blend')
    print('Linear Z reconstruction: arbitrary field, mass-flux sign, and real blend fraction')
    prove(sign(flow) * blend == z3.If(flow == 0, 0, z3.If(flow > 0, blend, -blend)),
          'source normalization yields the signed blend, including zero flow', [])
    for scheme, policy in LINEAR.items():
        wanted = reference(scheme, field, i, j, q, component, flow, blend)
        for name in (policy, 'UPWINDALL'):
            actual = policies[name].formula(field, i, j, q, component, flow, blend, z3.IntVal(enums[scheme]))
            prove(actual == wanted, f'{name}/{scheme}: source reconstruction matches independent weights', [])
    print('\nComposed vertical momentum fluxes: arbitrary integer lo < k < hi')
    for scheme, policy in LINEAR.items():
        model = Model(enums, constants)
        model.env['vert_adv_type'] = z3.IntVal(enums[scheme])
        model.arrays['w'] = lambda x, y, z: field(x, y, z, 0)
        model.interp = {
            'interp_w_v': lambda x, y, z, n, f, selected: policies[policy].formula(field, x, y, z, n, f, blend),
            'interp_w_wall': lambda x, y, z, n, f, selected: policies['UPWINDALL'].formula(field, x, y, z, n, f, blend, selected),
        }
        for expression in pure_regions:
            model.expression(expression, model.env)
        env = model.execute([high], model.env, [])
        env = model.execute([low], env, [])
        i, j, k, lo, hi = [model.env[name] for name in ('i', 'j', 'k', 'lo_z_face', 'hi_z_face')]
        for edge, face in [('hi', k + 1), ('lo', k)]:
            avg = (model.arrays['rho_w'](i, j, face - 1) + model.arrays['rho_w'](i, j, face)) / 2
            distance = z3.If(face - lo < hi - face + 1, face - lo, hi - face + 1)
            cap = scheme if scheme in ('Centered_2nd', 'Upwind_3rd') else 'Centered_4th'
            wanted = avg * z3.If(distance == 1,
                                reference('Centered_2nd', field, i, j, face, 0, avg, blend),
                                z3.If(distance == 2, reference(cap, field, i, j, face, 0, avg, blend),
                                      reference(scheme, field, i, j, face, 0, avg, blend)))
            prove(env['zflux_' + edge] == wanted, f'{scheme}/{edge}: composed source flux matches reconstruction',
                  [lo < k, k < hi])
        neighbor = z3.substitute(env['zflux_lo'], (k, k + 1))
        prove(env['zflux_hi'] == neighbor, f'{scheme}: reconstructed shared vertical flux agrees',
              [lo < k, k + 1 < hi])
    print(f'\n{len(failures)} check(s) failed')
    return 1 if failures else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, z3.Z3Exception) as exc:
        print(f'FAIL momentum linear Z-flux source contract: {exc}')
        sys.exit(1)
