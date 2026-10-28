#!/usr/bin/env python3
"""Mutate complete momentum tendencies, requiring explicit proof/parser failures.

The baseline covers every pair/component. Faults use documented selectors to
target their scope. Numerical tail faults must fail the final tendency identity,
not just a disconnected flux assertion. Unknown and subprocess crashes do not
count as successful detection. Equivalent arithmetic must still pass.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_momentum_dispatch import SCHEMES
from check_momentum_linear_z_flux_selftest import ENUM, MOM, UPW, method, once, struct
from check_momentum_weno_z_flux_selftest import HEADERS

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_tendency.py')


def cases(originals):
    yield 'baseline', {}, False, (), []
    source = originals[MOM]
    for component, velocity in zip('XYZ', 'uvw'):
        start = source.index(f'AdvectionSrcFor{component}Mom_N (')
        end = source.index('/**', start)
        body = source[start:end]
        diagnostic = f'Centered_2nd/Centered_2nd/{component}: complete reconstructed momentum tendency [sat]'
        options = ['--horizontal', 'Centered_2nd', '--vertical', 'Centered_2nd', '--component', component]

        def case(label, changed, required=(diagnostic,), mutated=True, full_source=False):
            value = changed if full_source else source[:start] + changed + source[end:]
            if value == source:
                raise AssertionError('unchanged case: ' + label)
            return component + ' ' + label, {MOM: value}, mutated, required, options

        for axis in 'xyz':
            difference = f'({axis}flux_hi - {axis}flux_lo)'
            yield case(axis + ' reversed flux difference', once(body, difference,
                       f'({axis}flux_lo - {axis}flux_hi)'))
            yield case(axis + ' omitted low face', once(body, difference, f'{axis}flux_hi'))
            scale = f'* d{axis}Inv'
            yield case(axis + ' wrong spacing', once(body, scale, '* d' + ('y' if axis == 'x' else 'x') + 'Inv'))
        yield case('horizontal metric omitted', once(body, '(xflux_hi - xflux_lo) * dxInv * mfsq',
                   '(xflux_hi - xflux_lo) * dxInv'))
        yield case('vertical metric added', once(body, '(zflux_hi - zflux_lo) * dzInv',
                   '(zflux_hi - zflux_lo) * dzInv * mfsq'))
        metric = re.search(r'amrex::Real mfsq = ([^;]+);', body)
        yield case('metric doubled', once(body, metric[0], 'amrex::Real mfsq = 2 * (' + metric[1] + ');'))
        yield case('metric read shifted', once(body, metric[0], metric[0].replace('i,j,0', 'i+1,j,0')))
        yield case('helper return sign reversed', once(body, 'return advectionSrc;', 'return -advectionSrc;'))
        yield case('helper return offset', once(body, 'return advectionSrc;', 'return advectionSrc + one;'))
        yield case('helper returns interpolation temporary', once(body, 'return advectionSrc;', 'return interp_hi;'))
        yield case('late unsupported write', once(body, 'return advectionSrc;',
                   'advectionSrc *= -1;\n return advectionSrc;'), ())
        yield case('integer truncation must not become real division', once(body, 'return advectionSrc;',
                   'return advectionSrc + (k/2 + k/2 - k);'), ('integer division is unsupported',))
        # Make sure complete-helper execution, not separately collected faces,
        # supplies the reconstruction prerequisites used by the tail proof.
        insertion = metric.start()
        changed = body[:insertion] + 'xflux_hi = zflux_lo;\n' + body[insertion:]
        yield case('vertical result corrupts horizontal face', changed,
                   (f'Centered_2nd/{component}.X/hi: reconstructed face prerequisite [sat]',))

        assignment = re.search(r'advectionSrc\s*=\s*([^;]+);', body)
        x = '(xflux_hi - xflux_lo) * dxInv * mfsq'
        y = '(yflux_hi - yflux_lo) * dyInv * mfsq'
        z = '(zflux_hi - zflux_lo) * dzInv'
        yield case('reordered divergence', once(body, assignment[0], f'advectionSrc = {z} + {x} + {y};'), (), False)
        yield case('factored horizontal metric', once(body, assignment[0],
                   'advectionSrc = ((xflux_hi - xflux_lo) * dxInv + '
                   '(yflux_hi - yflux_lo) * dyInv) * mfsq + ' + z + ';'), (), False)
        yield case('equivalent helper return', once(body, 'return advectionSrc;',
                   'return advectionSrc + zero;'), (), False)

        wrapper_start = source.index('AdvectionSrcForMomWrapper_N (')
        lhs = f'rho_{velocity}_rhs(i, j, k) = '
        rhs_start = source.index(lhs, wrapper_start)
        rhs_end = source.index(';', rhs_start) + 1
        rhs = source[rhs_start:rhs_end]
        yield case('wrapper sign reversed', once(source, rhs, rhs.replace('= -', '= ')), full_source=True)
        yield case('wrapper wrong destination', once(source, rhs, rhs.replace(f'rho_{velocity}_rhs',
                   'rho_' + ('v' if velocity == 'u' else 'u') + '_rhs')), (), full_source=True)
        yield case('wrapper shifted destination', once(source, rhs, rhs.replace(lhs,
                   f'rho_{velocity}_rhs(i, j, k+1) = ')), (), full_source=True)
        yield case('wrapper wrong helper', once(source, rhs, rhs.replace(f'AdvectionSrcFor{component}Mom_N',
                   'AdvectionSrcFor' + ('Y' if component == 'X' else 'X') + 'Mom_N')), (), full_source=True)
        yield case('wrapper exchanged policy arguments', once(source, rhs,
                   rhs.replace(f'interp_{velocity}_h, interp_{velocity}_v',
                               f'interp_{velocity}_v, interp_{velocity}_h')), (), full_source=True)
        launch = f'amrex::ParallelFor(bx{component.lower()},'
        yield case('wrapper wrong launch box', once(source, launch, 'amrex::ParallelFor(bx' +
                   ('y' if component == 'X' else 'x') + ','), (), full_source=True)
        positive = lhs + '-(' + rhs[len(lhs) + 1:-1] + ');'
        yield case('parenthesized wrapper negation', once(source, rhs, positive), (), False, True)
        # Overall equivalence must be checked across the helper/consumer edge.
        changed = source[:start] + once(body, 'return advectionSrc;', 'return 2 * advectionSrc;') + source[end:]
        changed = once(changed, rhs, rhs[:-1] + ' / 2;')
        yield case('compensated helper and wrapper scaling', changed, (), False, True)

        dz_start = source.rindex('auto dzInv = ', wrapper_start, rhs_start)
        dz_end = source.index(';', dz_start) + 1
        dz = source[dz_start:dz_end]
        expression = dz[len('auto dzInv = '):-1]
        replace_dz = lambda changed: source[:dz_start] + changed + source[dz_end:]
        # Multiplication stays inside each ternary arm for Z.
        yield case('wrapper dz numerator doubled', replace_dz(dz.replace('one /', '(one + one) /')
                   .replace('one/', '(one + one)/').replace('two/', '(two + two)/')), full_source=True)
        yield case('wrapper wrong dz sample', replace_dz(dz.replace('dz_ptr[k]', 'dz_ptr[k+1]')), full_source=True)
        equivalent = expression.replace('one/', '(one + zero)/').replace('one /', '(one + zero) /')
        yield case('equivalent dz numerator', replace_dz('auto dzInv = ' + equivalent + ';'), (), False, True)
        if component == 'Z':
            yield case('wrong dz boundary condition', replace_dz(dz.replace('k == 0', 'k == 1')), full_source=True)
            yield case('wrong dz neighbor', replace_dz(dz.replace('k-1', 'k+1')), full_source=True)

    options = ['--horizontal', 'Upwind_3rd', '--vertical', 'Upwind_5th', '--component', 'X']
    for velocity in 'uvw':
        for direction in 'hv':
            constructor = f'interp_{velocity}_{direction}({velocity}, upw_frac_{direction})'
            swapped = constructor.replace('upw_frac_' + direction, 'upw_frac_' + ('v' if direction == 'h' else 'h'))
            yield 'wrong blend ' + constructor, {MOM: once(source, constructor, swapped)}, True, (), options
            changed = constructor.replace(f'({velocity},', '(' + ('v' if velocity == 'u' else 'u') + ',')
            yield 'wrong field ' + constructor, {MOM: once(source, constructor, changed)}, True, (), options
    yield 'wall uses horizontal blend', {MOM: once(source, 'interp_w_wall(w, upw_frac_v)',
          'interp_w_wall(w, upw_frac_h)')}, True, (), options
    yield 'wrong horizontal spacing alias', {MOM: once(source, 'dyInv = cellSizeInv[1]',
          'dyInv = cellSizeInv[0]')}, True, (), options

    for scheme, name in SCHEMES.items():
        header = HEADERS[int('_' in name)] if name.startswith('WENO') else UPW
        original = originals[header]
        start, end, body = struct(original, name)
        for axis, selector, component in [('X', '--horizontal', 'X'), ('Z', '--vertical', 'Y')]:
            interp = method(body, 'InterpolateIn' + axis)
            read = re.search(r'm_phi\(([^()]*)\)', interp)
            args = read[1].split(',')
            cross_axis = 1 if axis == 'X' else 0
            args[cross_axis] = 'ij'[cross_axis] + '+1'
            changed = once(interp, read[0], 'm_phi(' + ','.join(args) + ')')
            replacement = once(body, interp, changed)
            if replacement == body:
                raise AssertionError('unchanged interpolation mutant')
            options = [selector, scheme, '--vertical' if axis == 'X' else '--horizontal',
                       'Centered_2nd', '--component', component]
            diagnostic = f'{scheme}/{axis}: interpolation prerequisite [sat]'
            yield name + ' ' + axis + ' shifted sample', {header: original[:start] + replacement + original[end:]}, True, (diagnostic,), options


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-momentum-tendency-mutants-'))
    originals = {name: (ERF / name).read_text() for name in (MOM, ENUM, UPW, *HEADERS, CONST_H)}
    unexpected = mutants = positives = 0
    # Check all mutation anchors before spending time on the baseline proof.
    for index, (name, edits, mutated, required, options) in enumerate(list(cases(originals))):
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
