"""Independent nonlinear policies for the uniform-grid WENO family checked here.

These are explicit algorithm specifications, not consequences of polynomial
accuracy and not coefficients extracted from ERF. All policies use exponent two.
For JS weights alpha_i is proportional to g_i/(eps+beta_i)^2; for Z/MZQ it is
proportional to g_i*(1+T/(eps+beta_i)^2), where T is squared global smoothness.

The Z3/5 policies use the difference of the outer beta indicators; Z7 uses the
alternating combination of the two outer and two inner indicators. MZQ3 uses
the sum of magnitudes of the full-stencil beta's differences from the two linear
betas, divided by 32, then squared. The sign of a stored tau is immaterial.
This specifies ERF's selected variants, not every algorithm called WENO-Z/MZQ.
The validated ideal weights are parameters (including MZQ's admissible family).
The reference is pure arithmetic and never reads source text or solver results.
"""


def tau_squared(name, beta, magnitude):
    if name in ('WENO_Z3', 'WENO_Z5'):
        difference = beta[0] - beta[-1]
        return difference * difference
    if name == 'WENO_Z7':
        difference = (beta[0] + beta[3]) - (beta[1] + beta[2])
        return difference * difference
    if name == 'WENO_MZQ3':
        difference = magnitude(beta[2] - beta[0]) + magnitude(beta[2] - beta[1])
        return difference * difference / 1024
    raise ValueError(f'{name}: no global smoothness policy')


def alphas(name, ideals, beta, eps, tau):
    if name in ('WENO3', 'WENO5', 'WENO7'):
        return [g / ((eps + b) * (eps + b)) for g, b in zip(ideals, beta)]
    if name in ('WENO_Z3', 'WENO_Z5', 'WENO_Z7', 'WENO_MZQ3'):
        return [g * (1 + (tau / (eps + b)) * (tau / (eps + b)))
                for g, b in zip(ideals, beta)]
    raise ValueError(f'{name}: no nonlinear weight policy')
