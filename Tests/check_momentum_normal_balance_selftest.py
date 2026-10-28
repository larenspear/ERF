#!/usr/bin/env python3
"""Distinguish reconstructed-flux accuracy from weighted interior cancellation.

Require targeted sat/unsat diagnostics for shared-face, coefficient, homogeneous
and linearity faults. Common numerical errors that preserve conservation must
still fail reconstruction/tendency prerequisites. Unknown/crashes never count.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_momentum_linear_z_flux_selftest import ENUM, MOM, UPW, once
from check_momentum_tendency_selftest import cases as tendency_cases
from check_momentum_weno_z_flux_selftest import HEADERS

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_normal_balance.py')


def cases(originals):
    yield 'baseline', {}, False, ('1296/1296 component/direction/policy cases establish weighted interior cancellation',), []
    source = originals[MOM]
    for component in 'XYZ':
        start = source.index(f'AdvectionSrcFor{component}Mom_N (')
        end = source.index('/**', start)
        body = source[start:end]
        prefix = f'Centered_2nd/Centered_2nd/{component}'
        options = ['--horizontal', 'Centered_2nd', '--vertical', 'Centered_2nd', '--component', component]

        def case(label, changed, required=(), mutated=True, full=False):
            value = changed if full else source[:start] + changed + source[end:]
            if value == source:
                raise AssertionError('unchanged mutation: ' + label)
            return component + ' ' + label, {MOM: value}, mutated, required, options

        metric = re.search(r'amrex::Real mfsq = ([^;]+);', body)
        for direction in 'XYZ':
            axis = direction.lower()
            shared = prefix + f'.{direction}: parsed shared-face flux agrees'
            balanced = prefix + f'.{direction}: weighted shared-face coefficients cancel'
            difference = f'({axis}flux_hi - {axis}flux_lo)'
            # Insert after every branch, so the test observes the final flux.
            inserted = body[:metric.start()] + f'{axis}flux_hi = 2 * {axis}flux_hi;\n' + body[metric.start():]
            yield case(direction + ' high-only scaling', inserted, (shared + ' [sat]', balanced + ' [unsat]'))
            inserted = body[:metric.start()] + f'{axis}flux_lo = 2 * {axis}flux_lo;\n' + body[metric.start():]
            yield case(direction + ' low-only scaling', inserted, (shared + ' [sat]', balanced + ' [unsat]'))
            inserted = body[:metric.start()] + f'{axis}flux_hi = 2 * {axis}flux_hi;\n{axis}flux_lo = 2 * {axis}flux_lo;\n' + body[metric.start():]
            yield case(direction + ' common scaling preserves cancellation', inserted,
                       (shared + ' [unsat]', balanced + ' [unsat]',
                        f'Centered_2nd/{component}.{direction}/hi: reconstructed face prerequisite [sat]',
                        '3/3 component/direction/policy cases establish weighted interior cancellation'))
            yield case(direction + ' high coefficient sign', once(body, difference,
                       f'(-{axis}flux_hi - {axis}flux_lo)'), (shared + ' [unsat]', balanced + ' [sat]'))
            yield case(direction + ' omitted low coefficient', once(body, difference,
                       f'{axis}flux_hi'), (shared + ' [unsat]', balanced + ' [sat]'))
            yield case(direction + ' common coefficient sign preserves cancellation', once(body, difference,
                       f'({axis}flux_lo - {axis}flux_hi)'),
                       (balanced + ' [unsat]', prefix + ': complete reconstructed momentum tendency [sat]',
                        '3/3 component/direction/policy cases establish weighted interior cancellation'))

        assignment = re.search(r'advectionSrc\s*=\s*([^;]+);', body)
        yield case('nonlinear face coupling', once(body, assignment[0],
                   'advectionSrc = (' + assignment[1] + ') + xflux_hi * yflux_lo;'),
                   (prefix + ': zero faces give zero tendency [unsat]',
                    prefix + ': tendency is linear in all six faces [sat]'))
        yield case('flux-independent tendency', once(body, 'return advectionSrc;', 'return advectionSrc + one;'),
                   (prefix + ': zero faces give zero tendency [sat]',
                    prefix + ': tendency is linear in all six faces [unsat]'))
        yield case('coordinate-dependent return scaling', once(body, 'return advectionSrc;', 'return (k + 1) * advectionSrc;'),
                   (prefix + '.X: weighted shared-face coefficients cancel [unsat]',
                    prefix + '.Z: weighted shared-face coefficients cancel [sat]'))
        yield case('source metric depends on vertical coordinate', once(body, metric[0], metric[0].replace('i,j,0', 'i,j,k')),
                   (prefix + '.Z: parsed shared-face flux agrees [unsat]',
                    prefix + '.X: weighted shared-face coefficients cancel [sat]'))
        yield case('uniform return sign preserves cancellation', once(body, 'return advectionSrc;', 'return -advectionSrc;'),
                   (prefix + ': complete reconstructed momentum tendency [sat]',
                    '3/3 component/direction/policy cases establish weighted interior cancellation'))

        velocity = 'uvw'['XYZ'.index(component)]
        wrapper = source.index('AdvectionSrcForMomWrapper_N (')
        rhs_start = source.index(f'rho_{velocity}_rhs(i, j, k)', wrapper)
        dz_start = source.rindex('auto dzInv = ', wrapper, rhs_start)
        dz_end = source.index(';', dz_start) + 1
        dz = source[dz_start:dz_end]
        changed = source[:dz_start] + dz.replace('dz_ptr[k]', 'dz_ptr[k+1]') + source[dz_end:]
        yield case('shifted wrapper dz sample', changed,
                   (prefix + '.Z: parsed shared-face flux agrees [unsat]',
                    prefix + '.Z: weighted shared-face coefficients cancel [sat]'), full=True)
        if component == 'Z':
            changed = source[:dz_start] + dz.replace('k == 0', 'k == 1') + source[dz_end:]
            yield case('shifted spacing branch', changed, (prefix + '.Z: weighted shared-face coefficients cancel [sat]',), full=True)
            changed = once(body, 'if (k == hi_z_face-1) {', 'if (k == hi_z_face-2) {')
            opts = ['--horizontal', 'Centered_2nd', '--vertical', 'Weno_5', '--component', 'Z']
            yield 'Z wrong upper wall band', {MOM: source[:start] + changed + source[end:]}, True, (
                'Centered_2nd/Weno_5/Z.Z: parsed shared-face flux agrees [sat]',), opts

        for direction in 'XYZ':
            axis = direction.lower()
            term = f'({axis}flux_hi - {axis}flux_lo) * d{axis}Inv'
            equivalent = f'({axis}flux_lo - {axis}flux_hi) * (-d{axis}Inv)'
            yield case(direction + ' equivalent orientation', once(body, term, equivalent), (), False)
        inserted = body[:metric.start()] + 'rho_u_avg_hi = zero;\n' + body[metric.start():]
        yield case('irrelevant late temporary write', inserted, (), False)

    # A translationally invariant interpolation error is still conservative.
    # The embedded reconstruction proof must reject it for every policy/axis.
    for label, edits, mutated, required, options in tendency_cases(originals):
        if label.endswith('shifted sample'):
            yield label + ' preserves balance', edits, mutated, required + (
                '3/3 component/direction/policy cases establish weighted interior cancellation',), options
        elif 'compensated helper and wrapper scaling' in label:
            yield label, edits, mutated, required, options


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-momentum-normal-balance-mutants-'))
    originals = {name: (ERF / name).read_text() for name in (MOM, ENUM, UPW, *HEADERS, CONST_H)}
    fixtures = list(cases(originals))
    unexpected = mutants = positives = 0
    for index, (name, edits, mutated, required, options) in enumerate(fixtures):
        root = work / f'{index:03d}'
        for path, original in originals.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, original))
        result = subprocess.run([sys.executable, str(VERIFY), str(root), *options], capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        if result.returncode == 77:
            print(result.stdout.strip())
            return 77
        caught, expected, failures = mutation_outcome(result, mutated)
        missing = [message for message in required if message not in result.stdout]
        if '[unknown]' in result.stdout:
            missing.append('solver must decide every obligation')
        expected = expected and not missing
        unexpected += not expected
        mutants += mutated
        positives += not mutated and name != 'baseline'
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        for failure in failures[:2] + ['missing diagnostic: ' + message for message in missing]:
            print('           ' + failure)
    print(f'\n{mutants} mutations; {positives} equivalent rewrites; {unexpected} unexpected outcome(s)')
    return 1 if unexpected else 0


if __name__ == '__main__':
    sys.exit(main())
