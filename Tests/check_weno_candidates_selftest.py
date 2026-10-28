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
