"""Shared-face identities for parsed centered momentum fluxes and RHS terms.

Translate the actual source model to the neighboring control volume, then check
high(I) == low(I+e_d). These identities do not use the reconstruction oracle.
For positive spacings and component metric M, the control-volume weight is
V = 1 / (dxInv * dyInv * dzInv * M). A shared face must contribute equal and
opposite amounts to V*RHS. We prove the equivalent cross-multiplied identity
to avoid introducing additional division-by-zero terms into the solver.

The top Z/Z face is an external boundary, excluded by k < hi_z_face. Z momentum
also requires k > 0, matching the source's bxz precondition. Both cells are
assumed to be in the component's valid domain; allocation and halo validity,
external boundary budgets, and the full time update are outside this proof.
"""
import re

from check_scalar_flux_divergence import number, z3


def translate(expression, direction, array_reads, to_z3):
    """Evaluate a parsed term at I+e_d, retaining absolute indices such as 0."""
    coordinate = 'ijk'['xyz'.index(direction)]

    def index(text):
        if re.fullmatch(r'-?\d+', text):
            return str(int(text))
        match = re.fullmatch(r'([ijk])([+-]\d+)?', text)
        if match is None:
            raise ValueError(f'unsupported array index in neighbor translation: {text}')
        base, offset = match[1], int(match[2] or 0)
        offset += int(base == coordinate)
        return base if offset == 0 else f'{base}{offset:+d}'

    present, pending = set(), [expression]
    while pending:
        term = pending.pop()
        if term.get_id() not in present:
            present.add(term.get_id())
            pending.extend(term.children())

    replacements = [(z3.Int(coordinate), z3.Int(coordinate) + 1)]
    # to_z3 may intern new neighboring samples; take a snapshot before iterating.
    for key, symbol in list(array_reads.items()):
        if symbol.get_id() not in present:
            continue
        call = re.fullmatch(r'(\w+)\(([^()]*)\)', key)
        bracket = re.fullmatch(r'(\w+)\[([^\[\]]+)\]', key)
        if call:
            shifted = f'{call[1]}({",".join(index(arg) for arg in call[2].split(","))})'
        elif bracket:
            shifted = f'{bracket[1]}[{index(bracket[2])}]'
        else:
            raise ValueError(f'unsupported array read in neighbor translation: {key}')
        replacements.append((symbol, to_z3(shifted, {})))
    # Simultaneous substitution matters: newly shifted samples must not shift twice.
    return z3.substitute(expression, *replacements)


def check_shared_faces(component, actual_fluxes, actual_rhs, env, expected_metric,
                       expected_dz, array_reads, to_z3, prove_equal):
    """Prove all three shared faces and their volume-weighted RHS cancellation."""
    inverse_volume = env['dxInv'] * env['dyInv'] * expected_dz * expected_metric
    flux_names = [f'{axis}flux_{edge}' for axis in 'xyz' for edge in ('hi', 'lo')]

    def contribution(selected):
        return z3.substitute(actual_rhs, *[
            (env[name], actual_fluxes[name] if name == selected else number(0))
            for name in flux_names])

    for direction in 'xyz':
        def neighbor(term):
            return translate(term, direction, array_reads, to_z3)

        high, low = f'{direction}flux_hi', f'{direction}flux_lo'
        interior = [z3.Int('k') > 0] if component == 'Z' else []
        if component == 'Z' and direction == 'z':
            interior.append(z3.Int('k') < z3.Int('hi_z_face'))
        prove_equal(actual_fluxes[high], neighbor(actual_fluxes[low]),
                    f'{component}.{direction}: shared interior face flux agrees',
                    assumptions=interior)

        # The independently specified geometry defines the volume weight. The RHS
        # uses its parsed metric/spacing, so a bad source coefficient cannot be
        # hidden by deriving both the weight and contribution from that mistake.
        positive_geometry = [env['dxInv'] > 0, env['dyInv'] > 0,
                             expected_dz > 0, neighbor(expected_dz) > 0,
                             expected_metric > 0, neighbor(expected_metric) > 0]
        left = contribution(high) * neighbor(inverse_volume)
        right = neighbor(contribution(low)) * inverse_volume
        prove_equal(left + right, number(0),
                    f'{component}.{direction}: shared interior face cancels in weighted RHS',
                    assumptions=interior + positive_geometry)
