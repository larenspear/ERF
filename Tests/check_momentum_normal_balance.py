#!/usr/bin/env python3
"""Prove volume-weighted shared-face cancellation for normal momentum.

Rerun the complete reconstruction/tendency checks for all 12x12 H/V policies
and three components. Translate actual parsed low fluxes to the neighbor and
prove they equal the final high fluxes. Decompose the parsed helper/wrapper RHS
as a homogeneous linear function of its six faces. Its two shared-face
coefficients must cancel under the independently specified volume weight
V = 1/(dxInv*dyInv*dzInv*metric). Prove the equivalent cross-multiplied identity
under positive geometry, then compose it with the proved shared-flux identity.

All coordinates, field samples, H/V blend fractions and flow signs are symbolic.
Both Z-momentum nodes must satisfy lo < k < hi. Translation preserves absolute
metric index zero and shifts dz samples/coordinate branches. Schemes and blends
are identical in the two cells. The weight is an algebraic specification, not a
parsed production volume routine. This is not a floating-point, halo, external
boundary, AMR/refluxing or whole-time-update conservation proof. WENO kernel and
consumer counterexamples remain abstract interface witnesses.

usage: check_momentum_normal_balance.py [ERF_ROOT] [--horizontal NAME]
       [--vertical NAME] [--component X|Y|Z]
exit: 0 success, 1 failed, 77 z3 unavailable (CTest skip)
"""
import sys

import check_momentum_tendency as tendency
from check_momentum_horizontal_flux import prove
from check_momentum_wall_order import failures, z3
from linear_z_interpolation import num


class Balance:
    def __init__(self):
        self.lemmas = {}
        self.cases = self.composed = 0

    def check(self, claim, label, assumptions, family):
        # Cache only exactly matching normalized expressions and assumptions.
        # A failed/unknown lemma remains a failure and cannot authorize a proof.
        key = (family, *(z3.simplify(term, som=True).sexpr() for term in (claim, *assumptions)))
        if key not in self.lemmas:
            before = len(failures)
            prove(claim, label, assumptions)
            self.lemmas[key] = len(failures) == before
        return self.lemmas[key]

    def __call__(self, model, component, horizontal, vertical, source_fluxes, faces,
                 actual_rhs, metric, spacing, assumptions):
        label = f'{horizontal}/{vertical}/{component}'
        names = tuple(faces)

        def evaluate(selected=None):
            return z3.substitute(actual_rhs, *[(faces[key], num(int(key == selected))) for key in names])

        constant = evaluate()
        coefficients = {key: z3.simplify(evaluate(key) - constant) for key in names}
        linear = constant + sum((coefficients[key] * faces[key] for key in names), num(0))
        homogeneous = self.check(constant == 0, label + ': zero faces give zero tendency', assumptions,
                                 (component, 'homogeneous'))
        linearity = self.check(actual_rhs == linear, label + ': tendency is linear in all six faces', assumptions,
                              (component, 'linear'))
        inverse_volume = model.env['dxInv'] * model.env['dyInv'] * spacing * metric
        for direction in 'XYZ':
            self.cases += 1
            coordinate = model.env['ijk'['XYZ'.index(direction)]]

            def neighbor(term):
                # Array samples are function applications of integer indices.
                # Simultaneous coordinate substitution shifts samples and branch
                # conditions once, retaining literals (notably metric k=0).
                return z3.substitute(term, (coordinate, coordinate + 1))

            interior = ([model.env['lo_z_face'] < model.env['k'], model.env['k'] < model.env['hi_z_face']]
                        if component == 'Z' else [])
            interior += [neighbor(term) for term in list(interior)]
            positive = [*assumptions, *[neighbor(term) for term in assumptions],
                        model.env['dxInv'] > 0, model.env['dyInv'] > 0]
            axis = direction.lower()
            shared = self.check(source_fluxes[axis + 'flux_hi'] == neighbor(source_fluxes[axis + 'flux_lo']),
                                label + f'.{direction}: parsed shared-face flux agrees', interior,
                                (component, direction, 'shared'))
            weight = self.check(z3.And(inverse_volume > 0, neighbor(inverse_volume) > 0),
                                label + f'.{direction}: both volume weights are positive', positive,
                                (component, direction, 'weight'))
            # Derive coefficients from the actual RHS. Deriving the weights
            # from its actual metric/spacing would hide common source mistakes.
            cancellation = (coefficients[direction, True] * neighbor(inverse_volume) +
                            neighbor(coefficients[direction, False]) * inverse_volume)
            balanced = self.check(cancellation == 0,
                                  label + f'.{direction}: weighted shared-face coefficients cancel', positive,
                                  (component, direction, 'weighted'))
            self.composed += homogeneous and linearity and shared and weight and balanced


def main():
    balance = Balance()
    result = tendency.main(balance, __doc__)
    print(f'{balance.composed}/{balance.cases} component/direction/policy cases establish weighted interior cancellation')
    return result


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, SyntaxError, z3.Z3Exception) as exc:
        print(f'FAIL normal momentum balance source contract: {exc}')
        sys.exit(1)
