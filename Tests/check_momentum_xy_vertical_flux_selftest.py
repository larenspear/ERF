#!/usr/bin/env python3
"""Fault-inject vertical reconstruction and X/Y-momentum stagger bindings.

The unrestricted baseline covers every policy/component. Mutations normally
select the affected policy/component; flow-direction faults also require the
centered result to remain valid. WENO sat diagnostics are interface witnesses,
not concrete numerical WENO counterexamples. Unexpected crashes do not count.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_momentum_horizontal_flux_selftest import POLICIES
from check_momentum_linear_z_flux_selftest import ENUM, MOM, UPW, method, once, struct
from check_momentum_weno_z_flux_selftest import HEADERS

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_xy_vertical_flux.py')


def cases(originals):
    yield 'baseline', None, None, False, (), []
    for name, scheme in POLICIES:
        header = HEADERS[int('_' in name)] if name.startswith('WENO') else UPW
        source = originals[header]
        start, end, body = struct(source, name)
        interp = method(body, 'InterpolateInZ')
        diagnosis = (f'{scheme}/Z: source interpolation matches reconstruction [sat]',
                     f'{scheme}/X/hi: vertical flux matches reconstruction [sat]',
                     f'{scheme}/Y/lo: vertical flux matches reconstruction [sat]')

        def case(label, changed, required=diagnosis, mutated=True):
            if changed == body:
                raise AssertionError(f'unchanged case {name} {label}')
            return name + ' ' + label, header, source[:start] + changed + source[end:], mutated, required, ['--scheme', scheme]

        read = re.search(r'm_phi\(([^()]*)\)', interp)
        args = read[1].split(',')
        args[0] = 'i+1'
        changed = once(interp, read[0], 'm_phi(' + ','.join(args) + ')')
        yield case('cross-axis sample shift', once(body, interp, changed))
        call = re.search(r'Evaluate\(([^()]*)\)', interp)
        args = call[1].split(',')
        args[0] = args[1]
        changed = once(interp, call[0], 'Evaluate(' + ','.join(args) + ')')
        yield case('duplicated Evaluate sample', once(body, interp, changed))
        result = re.search(r'val_lo = ([^;]+);', interp)
        changed = once(interp, result[0], result[0] + '\n val_lo *= -1;')
        yield case('late interpolation write', once(body, interp, changed), ())
        signature = re.search(r'InterpolateInZ\s*\(([^()]*)\)', body)
        params = signature[1].split(',')
        params[0], params[2] = params[2], params[0]
        yield case('swapped Z coordinate parameters', once(body, signature[0],
                   signature[0].replace(signature[1], ','.join(params))), ())
        if name.startswith(('UPWIND', 'WENO')):
            yield case('reversed sign normalization', once(body, interp,
                       once(interp, 'upw_lo > 0', 'upw_lo < 0')), ())
        if not name.startswith('WENO'):
            coefficient = re.search(r'static constexpr amrex::Real g1\s*=\s*([^;]+);', body)
            changed = once(body, coefficient[0], coefficient[0].replace(coefficient[1],
                           '(' + coefficient[1] + ') + amrex::Real(1)'))
            yield case('wrong common coefficient', changed, diagnosis +
                       (f'{scheme}/X: shared vertical flux agrees [unsat]',
                        f'{scheme}/Y: shared vertical flux agrees [unsat]'))
        else:
            ebody = method(body, 'Evaluate')
            returned = re.search(r'return\s+([^;]+);', ebody)
            changed = once(ebody, returned[0], 'return -(' + returned[1] + ');')
            yield case('negated scalar return', once(body, ebody, changed),
                       (f'FAIL {name}: complete source return equals the normalized candidate reconstruction',
                        f'{scheme}/Z: source interpolation matches reconstruction [unsat]',
                        f'{scheme}/X/hi: vertical flux matches reconstruction [unsat]'))
        if name in ('UPWIND5', 'WENO7'):
            loads = list(re.finditer(r'amrex::Real\s+s\w*\s*=\s*m_phi\([^;]+;', interp))
            a, b = loads[:2]
            changed = interp[:a.start()] + b[0] + interp[a.end():b.start()] + a[0] + interp[b.end():]
            yield case('reordered independent loads', once(body, interp, changed), (), False)
        if name == 'CENTERED2':
            args = call[1].split(',')
            args[0], args[1] = args[1], args[0]
            changed = once(interp, call[0], 'Evaluate(' + ','.join(args) + ')')
            yield case('symmetric Evaluate argument swap', once(body, interp, changed), (), False)
        if name == 'WENO3':
            changed = once(interp, result[0], 'val_lo = (upw_lo * upw_lo) * (' + result[1] + ');')
            yield case('zero-flow value erased', once(body, interp, changed),
                       (f'{scheme}/Z: source interpolation matches reconstruction [sat]',
                        f'{scheme}/X/hi: vertical flux matches reconstruction [unsat]',
                        f'{scheme}/Y/lo: vertical flux matches reconstruction [unsat]'))

    source = originals[MOM]
    for component, velocity in zip('XY', 'uv'):
        start = source.index(f'AdvectionSrcFor{component}Mom_N (')
        end = source.index('/**', start)
        body = source[start:end]

        def case(label, changed, required=(), scheme='Centered_2nd', mutated=True):
            if changed == body:
                raise AssertionError(f'unchanged momentum case {label}')
            options = ['--component', component] + (['--scheme', scheme] if scheme else [])
            return component + ' ' + label, MOM, source[:start] + changed + source[end:], mutated, required, options

        for edge in ('hi', 'lo'):
            diagnosis = (f'Centered_2nd/{component}/{edge}: vertical flux matches reconstruction [sat]',
                         f'Centered_2nd/{component}: shared vertical flux agrees [sat]')
            average = re.search(rf'\brho_w_avg_{edge}\s*=\s*([^;]+);', body)
            changed = once(body, average[0], average[0].replace(average[1], '2 * (' + average[1] + ')'))
            yield case(edge + ' doubled mass average', changed, diagnosis)
            reads = list(re.finditer(r'rho_w\(([^()]*)\)', average[0]))
            if len(reads) != 2:
                raise AssertionError('expected two rho_w samples')
            c = 'XY'.index(component)
            other = 1 - c
            args = reads[1][1].split(',')
            args[c], args[other] = 'ij'[c], 'ij'[other] + '-1'
            changed_average = once(average[0], reads[1][0], 'rho_w(' + ','.join(args) + ')')
            yield case(edge + ' wrong stagger axis', once(body, average[0], changed_average), diagnosis)
            yield case(edge + ' wrong mass field', once(body, average[0],
                       average[0].replace('rho_w(', 'rho_u(')), diagnosis)
            changed_average = average[0].replace(average[1], '(' + average[1] + ') * mf_uy_inv(i,j,0)')
            yield case(edge + ' extraneous map factor', once(body, average[0], changed_average), diagnosis)

            call = re.search(rf'interp_{velocity}_v\.InterpolateInZ\(([^()]*)\);',
                             body[average.end():])
            if call is None:
                raise AssertionError('missing vertical call')
            args = call[1].split(',')
            for label, index, value in [('shifted X coordinate', 0, 'i+1'),
                                        ('shifted Z coordinate', 2, 'k+2'),
                                        ('wrong data component', 3, '1')]:
                changed_args = list(args)
                changed_args[index] = value
                changed = once(body, call[0], call[0].replace(call[1], ','.join(changed_args)))
                yield case(edge + ' ' + label, changed, diagnosis)
            other_edge = 'lo' if edge == 'hi' else 'hi'
            yield case(edge + ' wrong output temporary', once(body, call[0],
                       call[0].replace('interp_' + edge, 'interp_' + other_edge)), diagnosis)
            yield case(edge + ' omitted interpolation', once(body, call[0], ''), diagnosis)
            yield case(edge + ' wrong method direction', once(body, call[0],
                       call[0].replace('InterpolateInZ', 'InterpolateInX')), diagnosis)
            yield case(edge + ' wrong policy object', once(body, call[0],
                       call[0].replace(f'interp_{velocity}_v', f'interp_{velocity}_h')))
            yield case(edge + ' reversed flow', once(body, call[0], call[0].replace(',rho_w_avg_', ',-rho_w_avg_')),
                       (f'Upwind_3rd/{component}/{edge}: vertical flux matches reconstruction [sat]',
                        f'Centered_2nd/{component}/{edge}: vertical flux matches reconstruction [unsat]',
                        f'Upwind_3rd/{component}: shared vertical flux agrees [sat]'), scheme=None)
            product = f'zflux_{edge} = rho_w_avg_{edge} * interp_{edge};'
            yield case(edge + ' reversed flux sign', once(body, product, product.replace('= ', '= -')), diagnosis)

        changed = body
        for edge in ('hi', 'lo'):
            product = f'zflux_{edge} = rho_w_avg_{edge} * interp_{edge};'
            changed = once(changed, product, product.replace('= ', '= 2 * '))
        yield case('both fluxes doubled', changed,
                   (f'Centered_2nd/{component}/hi: vertical flux matches reconstruction [sat]',
                    f'Centered_2nd/{component}: shared vertical flux agrees [unsat]'))
        product = 'zflux_lo = rho_w_avg_lo * interp_lo;'
        yield case('low block corrupts high flux', once(body, product, product + '\n zflux_hi = 0;'),
                   (f'Centered_2nd/{component}/hi: vertical flux matches reconstruction [sat]',))
        yield case('late unsupported arithmetic', once(body, product, product + '\n zflux_hi *= -1;'))
        signature = re.search(rf'AdvectionSrcFor{component}Mom_N\s*\(([^()]*)\)', body)
        yield case('swapped helper mass-field parameters', once(body, signature[0], signature[0]
                   .replace('& rho_u,', '& TEMP,').replace('& rho_w,', '& rho_u,').replace('& TEMP,', '& rho_w,')))

        average = re.search(r'\brho_w_avg_hi\s*=\s*([^;]+);', body)
        reads = list(re.finditer(r'rho_w\([^()]*\)', average[0]))
        changed = once(average[0], reads[0][0], 'PLACEHOLDER')
        changed = once(changed, reads[1][0], reads[0][0]).replace('PLACEHOLDER', reads[1][0])
        yield case('reordered mass average', once(body, average[0], changed), mutated=False)
        changed = body
        for edge in ('hi', 'lo'):
            product = f'zflux_{edge} = rho_w_avg_{edge} * interp_{edge};'
            changed = once(changed, product, f'zflux_{edge} = interp_{edge} * rho_w_avg_{edge};')
        yield case('commuted flux products', changed, mutated=False)
        changed = once(body, average[0], average[0].replace('myhalf', '(myhalf * myhalf * 2)'))
        yield case('equivalent averaging coefficient', changed, mutated=False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-momentum-xy-vertical-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (UPW, *HEADERS, MOM, ENUM, CONST_H)}
    unexpected = mutants = positives = 0
    for index, (name, path, changed, mutated, required, options) in enumerate(cases(originals)):
        root = work / f'{index:03d}'
        for source_path, original in originals.items():
            target = root / source_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(changed if source_path == path else original)
        result = subprocess.run([sys.executable, str(VERIFY), str(root), *options], capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        if result.returncode == 77:
            print(result.stdout.strip())
            return 77
        caught, expected, failures = mutation_outcome(result, mutated)
        missing = [message for message in required if message not in result.stdout]
        expected = expected and not missing
        unexpected += not expected
        mutants += mutated
        positives += not mutated and name != 'baseline'
        print(f"[{'CAUGHT' if expected and caught else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        for failure in failures[:2] + ['missing diagnostic: ' + message for message in missing]:
            print('           ' + failure)
    print(f'\n{mutants} mutations; {positives} equivalent rewrites; {unexpected} unexpected outcome(s)')
    return 1 if unexpected else 0


if __name__ == '__main__':
    sys.exit(main())
