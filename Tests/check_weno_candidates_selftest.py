#!/usr/bin/env python3
"""Inject WENO candidate and weight bugs and require the checker to catch them.

Mutants live in temporary copies of the two WENO headers; the repository files
are never changed.

usage: check_weno_candidates_selftest.py [WORK_DIR]
exit:  0 every outcome as expected, 1 otherwise, 77 z3 is not importable
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).resolve().parent / "check_weno_candidates.py"
HEADERS = (
    "Source/Utils/ERF_Interpolation_WENO.H",
    "Source/Utils/ERF_Interpolation_WENO_Z.H",
)

MUTANTS = {
    "baseline (no change)": None,
    "WENO3 candidate loses linear exactness": (
        HEADERS[0], "-sm1 + amrex::Real(3) * s", "-sm1 + amrex::Real(2) * s"),
    "WENO5 candidate coefficient typo": (
        HEADERS[0], "amrex::Real(2) * sm2 - amrex::Real(7.0) * sm1",
        "amrex::Real(3) * sm2 - amrex::Real(7.0) * sm1"),
    "WENO7 candidate coefficient typo": (
        HEADERS[0], "amrex::Real(23.0)/amrex::Real(12.0)",
        "amrex::Real(22.0)/amrex::Real(12.0)"),
    "WENO-MZQ3 third-order candidate typo": (
        HEADERS[1], "amrex::Real(5.0) * s + amrex::Real(2) * sp1) / amrex::Real(6.0)",
        "amrex::Real(5.0) * s + amrex::Real(3) * sp1) / amrex::Real(6.0)"),
    "negative WENO7 unnormalized weight": (
        HEADERS[0], "amrex::Real w4 = g4 / ( (eps + b4) * (eps + b4) );",
        "amrex::Real w4 = -g4 / ( (eps + b4) * (eps + b4) );"),
    "WENO5 normalizer drops a weight": (
        HEADERS[0], "amrex::Real wsum = w1 + w2 + w3;",
        "amrex::Real wsum = w1 + w2;"),
    "negative WENO-MZQ3 unnormalized weight": (
        HEADERS[1], "amrex::Real a3 = g3 * ( amrex::Real(1) +",
        "amrex::Real a3 = -g3 * ( amrex::Real(1) +"),
    'WENO3 return subtracts second candidate': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v1 - w2 * v2) / (amrex::Real(2) * wsum) );'),
    'WENO3 return duplicates second candidate': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v2 + w2 * v2) / (amrex::Real(2) * wsum) );'),
    'WENO3 return drops second candidate': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v1) / (amrex::Real(2) * wsum) );'),
    'WENO3 return divides by wrong normalizer': ('Source/Utils/ERF_Interpolation_WENO.H',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * wsum) );',
         'return ( (w1 * v1 + w2 * v2) / (amrex::Real(2) * (wsum + w1)) );'),
    'WENO-MZQ3 return reverses correction': ('Source/Utils/ERF_Interpolation_WENO_Z.H',
         '(v3 - g1 * v1 - g2 * v2)',
         '(v3 + g1 * v1 - g2 * v2)'),
    'WENO3 weight is updated after initialization': ('Source/Utils/ERF_Interpolation_WENO.H',
         'amrex::Real wsum = w1 + w2;',
         'w1 *= -1;\n        amrex::Real wsum = w1 + w2;'),
}


def cases(originals):
    """Also distinguish wrong-but-positive beta/weights from positivity failures."""
    for name, edit in MUTANTS.items():
        edits = {}
        if edit:
            header, old, new = edit
            if originals[header].count(old) != 1:
                raise AssertionError(f'{name}: ambiguous mutation anchor')
            edits[header] = originals[header].replace(old, new)
        yield name, edits, edit is not None, ()

    for header, text in originals.items():
        for struct in re.finditer(r'^struct (\w+)\s*\n\{(.*?)^\};', text, re.S | re.M):
            name, body = struct[1], struct[2]

            def case(description, changed, required=(), mutated=True):
                if changed == body:
                    raise AssertionError('unchanged fixture: ' + description)
                return name + ' ' + description, {
                    header: text[:struct.start(2)] + changed + text[struct.end(2):]}, mutated, required

            beta = list(re.finditer(r'amrex::Real\s+(b\d+)\s*=\s*([^;]+);', body))
            constants = list(re.finditer(r'static constexpr amrex::Real\s+(g\d+)\s*=\s*([^;]+);', body))
            returned = re.search(r'\breturn\s+([^;]+);', body)
            ideal = f'{name}: source ideal return matches full-stencil reconstruction'
            defined = f'{name}: constant source divisions are defined for eps>0'
            affine_defined = f'{name}: affine source divisions are defined for eps>0'
            exact = f'{name}: actual nonlinear return reproduces'
            positive_fields = ('ok   ' + defined, 'ok   ' + affine_defined,
                               f'ok   {exact} constant fields for eps>0 [unsat]',
                               f'ok   {exact} affine fields for eps>0 [unsat]')
            raw_prefix = 'a' if name == 'WENO_MZQ3' else 'w'
            weights = list(re.finditer(rf'amrex::Real\s+({raw_prefix}\d+)\s*=\s*([^;]+);', body))
            for match in weights:
                # On constant data beta=0. Positivity of an abstract alpha alone
                # is not a substitute for checking each actual source divisor.
                expr = re.sub(r'eps\s*\+\s*', '', match[2])
                changed = body[:match.start(2)] + expr + body[match.end(2):]
                yield case(match[1] + ' loses epsilon regularization', changed,
                           ('FAIL ' + defined, 'FAIL ' + affine_defined))

            # A canceled or unused singular expression used to escape every
            # algebraic identity. Preserve evidence that the old ideal operator
            # and normalized return identities still hold for these fixtures.
            for label, expr in (
                ('unused zero-numerator singularity', 'amrex::Real(0) / b1'),
                ('nested singularity', 'amrex::Real(1) / (amrex::Real(1) / b1)'),
                ('canceled singularity', 'eps / (b1-b2) - eps / (b1-b2)'),
            ):
                changed = body[:returned.start()] + f'amrex::Real unused = {expr};\n' + body[returned.start():]
                yield case(label, changed, ('FAIL ' + defined, 'FAIL ' + affine_defined,
                                           'ok   ' + ideal,
                                           f'ok   {name}: complete source return equals the normalized candidate reconstruction'))
            changed = body[:returned.start(1)] + '(' + returned[1] + ') + amrex::Real(0)*(amrex::Real(1)/(w1-w1))' + body[returned.end(1):]
            yield case('hidden singularity in actual return', changed,
                       ('FAIL ' + defined, 'FAIL ' + affine_defined, 'ok   ' + ideal,
                        f'ok   {name}: complete source return equals the normalized candidate reconstruction'))
            tau = re.search(r'amrex::Real\s+t5\s*=\s*([^;]+);', body)
            if tau:
                changed = body[:tau.start(1)] + '(' + tau[1] + ') + amrex::Real(0)*(amrex::Real(1)/(b1-b2))' + body[tau.end(1):]
                yield case('hidden singularity in global smoothness', changed,
                           ('FAIL ' + defined, 'FAIL ' + affine_defined, 'ok   ' + ideal))
            # A removable singularity at eps=1 must fail despite every source
            # precision branch using a much smaller positive eps: the contract
            # explicitly quantifies over all positive eps.
            match = weights[0]
            changed = body[:match.start(2)] + '(' + match[2] + ') * ((eps-amrex::Real(1))/(eps-amrex::Real(1)))' + body[match.end(2):]
            yield case('weight has a removable singularity at positive epsilon', changed,
                       ('FAIL ' + defined, 'FAIL ' + affine_defined))
            changed = body[:match.start(2)] + '(' + match[2] + ') * ((eps+b1)/(eps+b1))' + body[match.end(2):]
            yield case('weight has a defined canceling ratio', changed, positive_fields, mutated=False)
            # This used to be an accepted control: affine accuracy alone cannot
            # determine nonlinear weighting. The new independent policy rejects it.
            changed = body[:match.start(2)] + 'amrex::Real(2)*(' + match[2] + ')' + body[match.end(2):]
            policy = f'{name}: normalized nonlinear weights match independent policy'
            all_defined = f'{name}: all-data source divisions are defined for eps>0'
            preserved = positive_fields + ('ok   ' + all_defined,)
            yield case('alternative positive nonlinear weight', changed,
                       preserved + ('FAIL ' + policy,))
            for index, weight in enumerate(weights, 1):
                for label, expr in (
                    ('constant ideal alpha', f'g{index}'),
                    ('wrong beta in alpha', re.sub(rf'\bb{index}\b', f'b{index % len(weights) + 1}', weight[2])),
                ):
                    changed = body[:weight.start(2)] + expr + body[weight.end(2):]
                    yield case(weight[1] + ' ' + label, changed, preserved + ('FAIL ' + policy,))
            for label, transform in (
                ('wrong epsilon in every alpha', lambda expr: re.sub(r'\beps\b', '(amrex::Real(2)*eps)', expr)),
                ('first-power denominators in every alpha', lambda expr: re.sub(
                    r'\(\s*\(eps \+ (b\d+)\)\s*\*\s*\(eps \+ \1\)\s*\)', r'(eps + \1)', expr)),
                ('common constant alpha scaling', lambda expr: 'amrex::Real(2)*(' + expr + ')'),
                ('common positive data-dependent alpha scaling', lambda expr: '(eps+b1)*(' + expr + ')'),
            ):
                changed = body
                for weight in reversed(weights):
                    changed = changed[:weight.start(2)] + transform(weight[2]) + changed[weight.end(2):]
                fault = label.startswith(('wrong', 'first-power'))
                yield case(label, changed, preserved + (('FAIL ' if fault else 'ok   ') + policy,), fault)
            if tau:
                tau_policy = f'{name}: squared global smoothness matches independent policy'
                for label, expr in (
                    ('scaled global smoothness', 'amrex::Real(2)*(' + tau[1] + ')'),
                    ('zero global smoothness', 'amrex::Real(0)'),
                    ('wrong global smoothness combination', 'b1'),
                    ('opposite global smoothness sign', '-(' + tau[1] + ')'),
                ):
                    changed = body[:tau.start(1)] + expr + body[tau.end(1):]
                    fault = not label.startswith('opposite')
                    yield case(label, changed, preserved + ('ok   ' + policy,
                               ('FAIL ' if fault else 'ok   ') + tau_policy), fault)
                if name != 'WENO_MZQ3':
                    changed = body[:tau.start(1)] + tau[1].replace('std::abs', '') + body[tau.end(1):]
                    yield case('signed global difference before squaring', changed,
                               preserved + ('ok   ' + tau_policy,), False)
                else:
                    expr = 'std::abs(amrex::Real(2)*b3 - b1 - b2)/amrex::Real(32)'
                    changed = body[:tau.start(1)] + expr + body[tau.end(1):]
                    yield case('magnitude of sum replaces sum of magnitudes', changed,
                               preserved + ('FAIL ' + tau_policy,))
                expr = '(' + tau[1] + ') + amrex::Real(0)*(amrex::Real(1)/(amrex::Real(1)-(b1-b2)*(b1-b2)))'
                changed = body[:tau.start(1)] + expr + body[tau.end(1):]
                yield case('global smoothness singular only away from affine data', changed,
                           positive_fields + (f'FAIL {name}: global smoothness divisions are defined',
                                              'FAIL ' + all_defined))
            # Curvature vanishes on every affine field, while arbitrary cell
            # averages can make these denominators zero. Do not assume them away.
            curvature = '(sp1-amrex::Real(2)*s+sm1)'
            singular = f'amrex::Real(1)/(amrex::Real(1)-{curvature}*{curvature})'
            changed = body[:returned.start()] + f'amrex::Real unused = {singular};\n' + body[returned.start():]
            yield case('unused division singular only away from affine data', changed,
                       positive_fields + ('FAIL ' + all_defined, 'ok   ' + policy))
            beta_first = beta[0]
            expr = '(' + beta_first[2] + ') + amrex::Real(0)*(' + singular + ')'
            changed = body[:beta_first.start(2)] + expr + body[beta_first.end(2):]
            yield case('hidden beta division singular only away from affine data', changed,
                       positive_fields + ('FAIL ' + all_defined,
                                          f'ok   {name}.b1: smoothness indicator is non-negative'))
            expr = '(' + match[2] + ') + amrex::Real(0)*(amrex::Real(1)/(amrex::Real(1)-(b1-b2)*(b1-b2)))'
            changed = body[:match.start(2)] + expr + body[match.end(2):]
            yield case('hidden alpha division singular only away from affine data', changed,
                       positive_fields + ('FAIL ' + all_defined,
                                          f'FAIL {name}: nonlinear weight divisions are defined'))
            changed = body[:returned.start()] + 'amrex::Real unused = amrex::Real(1)/(eps+b1);\n' + body[returned.start():]
            yield case('unused but defined division', changed, positive_fields, mutated=False)

            candidate = re.search(r'amrex::Real\s+v1\s*=\s*([^;]+);', body)
            for label, perturbation, required in (
                ('candidate constant shift', 'amrex::Real(1)',
                 (f'FAIL {exact} constant fields for eps>0 [sat]',
                  f'FAIL {exact} affine fields for eps>0 [sat]')),
                ('candidate slope shift', '(s-sm1)',
                 (f'ok   {exact} constant fields for eps>0 [unsat]',
                  f'FAIL {exact} affine fields for eps>0 [sat]')),
            ):
                # For MZQ the low-order candidates cancel at ideal weights;
                # perturb its full-stencil candidate to change the affine result.
                selected = (re.search(r'amrex::Real\s+v3\s*=\s*([^;]+);', body)
                            if name == 'WENO_MZQ3' else candidate)
                changed = body[:selected.start(1)] + '(' + selected[1] + ') + ' + perturbation + body[selected.end(1):]
                yield case(label, changed, ('ok   ' + defined, 'ok   ' + affine_defined) + required)

            first = constants[0]
            changed = body[:first.start(2)] + '(' + first[2] + ') + amrex::Real(0)*(amrex::Real(1)/amrex::Real(0))' + body[first.end(2):]
            yield case('hidden zero divisor in member constant', changed,
                       ('zero divisor in member constant',))
            for index, match in enumerate(beta):
                label, expr = match[1], match[2]
                required = (f'FAIL {name}.{label}: equals derivative-integral smoothness indicator',
                            f'ok   {name}.{label}: smoothness indicator is non-negative',
                            'ok   ' + ideal)
                changed = body[:match.start(2)] + 'amrex::Real(2) * (' + expr + ')' + body[match.end(2):]
                yield case(label + ' scaled but still positive semidefinite', changed, required)
                other = beta[(index + 1) % len(beta)][2]
                changed = body[:match.start(2)] + other + body[match.end(2):]
                yield case(label + ' uses another positive indicator', changed, required)
            for match in constants:
                label, expr = match[1], match[2]
                changed = body[:match.start(2)] + 'amrex::Real(2) * (' + expr + ')' + body[match.end(2):]
                diagnostic = (f'{name}: corrected ideal weights sum to one' if name == 'WENO_MZQ3'
                              else f'{name}: normalized ideal weights match full-stencil reconstruction')
                yield case(label + ' wrong positive ideal weight', changed,
                           ('FAIL ' + diagnostic, 'FAIL ' + ideal,
                            f'ok   {name}.{label}: source ideal weight is positive',
                            f'ok   {name}: normalized nonlinear weights sum to one'))
            # The ideal operator must use the actual parsed return expression.
            changed = body[:returned.start(1)] + '(' + returned[1] + ') + amrex::Real(1)' + body[returned.end(1):]
            yield case('extra term in ideal return', changed, ('FAIL ' + ideal,))
            if name in ('WENO5', 'WENO_Z5'):
                changed = body.replace('amrex::Real(13.0)/amrex::Real(12.0)', 'amrex::Real(14.0)/amrex::Real(12.0)')
                yield case('wrong positive curvature coefficient', changed,
                           (f'FAIL {name}.b1: equals derivative-integral smoothness indicator',
                            f'ok   {name}.b1: smoothness indicator is non-negative'))

            first = constants[0]
            changed = body[:first.start(2)] + '1/3' + body[first.end(2):]
            yield case('integer division must not become real division', changed,
                       ('integer division is unsupported in WENO expressions',))
            if name == 'WENO3':
                changed = body[:first.start(2)] + 'amrex::Real(1/3)' + body[first.end(2):]
                yield case('cast after integer division', changed,
                           ('integer division is unsupported in WENO expressions',))
                for expr in ('1/amrex::Real(3)', 'amrex::Real(1)/3'):
                    changed = body[:first.start(2)] + expr + body[first.end(2):]
                    yield case('mixed real/integer division ' + expr, changed, mutated=False)
            changed = body[:first.end()] + '\n' + first[0] + body[first.end():]
            yield case('duplicate member constant', changed, ('duplicate or shadowing member constant',))
            first_beta = beta[0]
            for local, value in [('s', 'sm1'), ('g1', 'amrex::Real(1)')]:
                changed = body[:first_beta.start()] + f'amrex::Real {local} = {value};\n' + body[first_beta.start():]
                yield case('shadowed ' + local, changed, ('Evaluate() shadows an input/member',))

            # Common ideal-weight scaling cancels in ordinary WENO. In MZQ it
            # scales the corrected reconstruction, so only unit-sum choices pass.
            changed = body
            for match in reversed(constants):
                changed = changed[:match.start(2)] + 'amrex::Real(2) * (' + match[2] + ')' + changed[match.end(2):]
            yield case('common ideal-weight scaling', changed,
                       ('FAIL ' + ideal,) if name == 'WENO_MZQ3' else (), name == 'WENO_MZQ3')
            # Equivalent polynomial spelling, plus multiline constants that
            # were previously vulnerable to line-oriented source collection.
            match = beta[0]
            changed = body[:match.start(2)] + '((' + match[2] + ') + (s-sm1)*(s-sm1) - (s-sm1)*(s-sm1))' + body[match.end(2):]
            yield case('equivalent expanded smoothness', changed, mutated=False)
            changed = re.sub(r'(static constexpr amrex::Real\s+\w+\s*=)\s*([^;]+);',
                             lambda match: match[1] + '\n (' + match[2] + ')\n;', body)
            yield case('multiline member constants', changed, mutated=False)
            if name == 'WENO_MZQ3':
                changed = body
                for match, value in reversed(list(zip(constants, ('1/4', '1/2', '1/4')))):
                    numerator, denominator = value.split('/')
                    expr = f'amrex::Real({numerator})/amrex::Real({denominator})'
                    changed = changed[:match.start(2)] + expr + changed[match.end(2):]
                yield case('alternative positive unit-sum ideal weights', changed, mutated=False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix="erf-weno-candidate-mutants-"))
    originals = {header: (ERF / header).read_text() for header in HEADERS}
    print(f"mutants in {work}\n")
    unexpected = mutants = positives = 0
    for i, (name, edits, mutated, required) in enumerate(cases(originals)):
        root = work / f"m{i}"
        for header in HEADERS:
            target = root / header
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(header, originals[header]))
        result = subprocess.run([sys.executable, str(VERIFY), str(root)],
                                capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        if result.returncode == 77:
            print(result.stdout.strip() or "check_weno_candidates.py skipped (exit 77)")
            return 77
        caught, expected, failures = mutation_outcome(result, mutated)
        missing = [message for message in required if message not in result.stdout]
        if 'unknown' in result.stdout:
            missing.append('all solver obligations must be decided')
        expected = expected and not missing
        unexpected += not expected
        mutants += mutated
        positives += not mutated and name != 'baseline (no change)'
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        for failure in failures[:3] + ['missing diagnostic: ' + message for message in missing]:
            print(f"           {failure}")
    print(f"\n{mutants} mutations; {positives} equivalent/specification-preserving rewrites; {unexpected} unexpected outcome(s)")
    return 1 if unexpected else 0


if __name__ == "__main__":
    sys.exit(main())
