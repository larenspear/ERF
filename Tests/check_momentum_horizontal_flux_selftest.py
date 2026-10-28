#!/usr/bin/env python3
"""Mutate horizontal stencil bindings, stagger geometry and final flux writes.

Run the unrestricted baseline, then select the affected policy/component for
each fault. Require explicit sat/unsat diagnostics on numerical/interface cases;
unsupported source rejection counts, unexpected subprocess crashes do not.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_momentum_linear_z_flux_selftest import ENUM, MOM, UPW, method, once, struct
from check_momentum_weno_z_flux_selftest import HEADERS

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_horizontal_flux.py')
POLICIES = [('CENTERED2', 'Centered_2nd'), ('UPWIND3', 'Upwind_3rd'),
            ('CENTERED4', 'Centered_4th'), ('UPWIND5', 'Upwind_5th'), ('CENTERED6', 'Centered_6th'),
            ('WENO3', 'Weno_3'), ('WENO_Z3', 'Weno_3Z'), ('WENO_MZQ3', 'Weno_3MZQ'),
            ('WENO5', 'Weno_5'), ('WENO_Z5', 'Weno_5Z'), ('WENO7', 'Weno_7'), ('WENO_Z7', 'Weno_7Z')]


def cases(originals):
    yield 'baseline', None, None, False, (), []
    for name, scheme in POLICIES:
        header = HEADERS[int('_' in name)] if name.startswith('WENO') else UPW
        source = originals[header]
        start, end, body = struct(source, name)
        for direction in 'XY':
            interp = method(body, 'InterpolateIn' + direction)
            diagnosis = (f'{scheme}/{direction}: source interpolation matches reconstruction [sat]',)

            def case(label, changed, required=diagnosis, mutated=True):
                if changed == body:
                    raise AssertionError(f'unchanged case {name}/{direction} {label}')
                return (name + '/' + direction + ' ' + label, header, source[:start] + changed + source[end:],
                        mutated, required, ['--scheme', scheme])

            read = re.search(r'm_phi\(([^()]*)\)', interp)
            args = read[1].split(',')
            args[2] = 'k+1'
            changed = once(interp, read[0], 'm_phi(' + ','.join(args) + ')')
            yield case('sample shifted in Z', once(body, interp, changed))
            call = re.search(r'Evaluate\(([^()]*)\)', interp)
            args = call[1].split(',')
            args[0] = args[1]
            changed = once(interp, call[0], 'Evaluate(' + ','.join(args) + ')')
            yield case('duplicated Evaluate sample', once(body, interp, changed))
            result = re.search(r'val_lo = ([^;]+);', interp)
            changed = once(interp, result[0], 'val_lo = -(' + result[1] + ');')
            yield case('negated interpolation', once(body, interp, changed),
                       diagnosis if name.startswith('WENO') else ())
            changed = once(interp, result[0], result[0] + '\n val_lo *= -1;')
            yield case('unmodeled output write', once(body, interp, changed), ())
            signature = re.search(rf'InterpolateIn{direction}\s*\(([^()]*)\)', body)
            params = signature[1].split(',')
            params[0], params[1] = params[1], params[0]
            yield case('swapped coordinate parameters', once(body, signature[0],
                       signature[0].replace(signature[1], ','.join(params))), ())
            if name.startswith(('UPWIND', 'WENO')):
                changed = once(interp, 'upw_lo > 0', 'upw_lo < 0')
                yield case('reversed flow normalization', once(body, interp, changed), ())
            if name == 'CENTERED2':
                args = call[1].split(',')
                args[0], args[1] = args[1], args[0]
                changed = once(interp, call[0], 'Evaluate(' + ','.join(args) + ')')
                yield case('symmetric argument swap', once(body, interp, changed), (), False)
            if name in ('CENTERED6', 'WENO_Z7'):
                loads = list(re.finditer(r'amrex::Real\s+s\w*\s*=\s*m_phi\([^;]+;', interp))
                a, b = loads[:2]
                changed = interp[:a.start()] + b[0] + interp[a.end():b.start()] + a[0] + interp[b.end():]
                yield case('reordered independent loads', once(body, interp, changed), (), False)

        if not name.startswith('WENO'):
            coefficient = re.search(r'static constexpr amrex::Real g1\s*=\s*([^;]+);', body)
            changed = once(body, coefficient[0], coefficient[0].replace(coefficient[1],
                           '(' + coefficient[1] + ') + amrex::Real(1)'))
            yield (name + ' changed common coefficient', header, source[:start] + changed + source[end:], True,
                   (f'{scheme}/X: source interpolation matches reconstruction [sat]',
                    f'{scheme}/X.x/hi: horizontal flux matches reconstruction [sat]',
                    f'{scheme}/X.x: shared horizontal flux agrees [unsat]'), ['--scheme', scheme, '--component', 'X'])
        if name == 'WENO5':
            signature = re.search(r'Evaluate\s*\(([^()]*)\)\s*const', body)
            params = signature[1].split(',')
            params[0], params[1] = params[1], params[0]
            changed = once(body, signature[0], signature[0].replace(signature[1], ','.join(params)))
            for direction in 'XYZ':
                interp = method(body, 'InterpolateIn' + direction)
                permuted = interp
                for call in re.finditer(r'Evaluate\(([^()]*)\)', interp):
                    args = call[1].split(',')
                    args[0], args[1] = args[1], args[0]
                    permuted = once(permuted, call[0], 'Evaluate(' + ','.join(args) + ')')
                changed = once(changed, interp, permuted)
            yield 'WENO5 consistent parameter/call permutation', header, source[:start] + changed + source[end:], False, (), ['--scheme', scheme]
            ebody = method(body, 'Evaluate')
            candidate = re.search(r'amrex::Real v1\s*=\s*([^;]+);', ebody)
            changed = once(body, candidate[0], candidate[0].replace(candidate[1], '(' + candidate[1] + ') + s'))
            yield 'WENO5 wrong scalar candidate', header, source[:start] + changed + source[end:], True, (
                'FAIL WENO5.v1: constant-data candidate scale',), ['--scheme', scheme]

    source = originals[MOM]
    for component, velocity in zip('XYZ', 'uvw'):
        start = source.index(f'AdvectionSrcFor{component}Mom_N (')
        end = source.index('/**', start)
        body = source[start:end]

        def mom_case(label, changed, required=(), scheme='Centered_2nd', mutated=True):
            if changed == body:
                raise AssertionError(f'unchanged momentum case {label}')
            return (component + ' ' + label, MOM, source[:start] + changed + source[end:], mutated,
                    required, ['--scheme', scheme, '--component', component])

        for direction, mass_component in zip('XY', 'uv'):
            axis = direction.lower()
            for edge in ('hi', 'lo'):
                diagnosis = (f'Centered_2nd/{component}.{axis}/{edge}: horizontal flux matches reconstruction [sat]',
                             f'Centered_2nd/{component}.{axis}: shared horizontal flux agrees [sat]')
                average = re.search(rf'\brho_{mass_component}_avg_{edge}\s*=\s*([^;]+);', body)
                changed = once(body, average[0], average[0].replace(average[1], '2 * (' + average[1] + ')'))
                yield mom_case(axis + '/' + edge + ' doubled mass average', changed, diagnosis)
                metric = 'mf_uy_inv' if direction == 'X' else 'mf_vx_inv'
                wrong = 'mf_vx_inv' if direction == 'X' else 'mf_uy_inv'
                changed = once(body, average[0], average[0].replace(metric, wrong))
                yield mom_case(axis + '/' + edge + ' wrong map factor', changed, diagnosis)
                changed_average, count = re.subn(rf'({metric}\([^,]+,[^,]+,)\s*0\)', r'\g<1>k)', average[0])
                if not count:
                    raise AssertionError('missing absolute metric plane')
                yield mom_case(axis + '/' + edge + ' metric uses variable Z plane',
                               once(body, average[0], changed_average), diagnosis)

                call = re.search(rf'interp_{velocity}_h\.InterpolateIn{direction}\(([^;]+),interp_{edge},rho_{mass_component}_avg_{edge}\);', body)
                other_direction = 'Y' if direction == 'X' else 'X'
                yield mom_case(axis + '/' + edge + ' wrong interpolation direction',
                               once(body, call[0], call[0].replace('InterpolateIn' + direction,
                                                                 'InterpolateIn' + other_direction)), diagnosis)
                yield mom_case(axis + '/' + edge + ' wrong interpolation policy object',
                               once(body, call[0], call[0].replace(f'interp_{velocity}_h', f'interp_{velocity}_v')), diagnosis)
                yield mom_case(axis + '/' + edge + ' reversed flow argument',
                               once(body, call[0], call[0].replace(',rho_', ',-rho_')),
                               (f'Upwind_3rd/{component}.{axis}/{edge}: horizontal flux matches reconstruction [sat]',
                                f'Upwind_3rd/{component}.{axis}: shared horizontal flux agrees [sat]'), 'Upwind_3rd')
                product = f'{axis}flux_{edge} = rho_{mass_component}_avg_{edge} * interp_{edge};'
                yield mom_case(axis + '/' + edge + ' reversed flux sign',
                               once(body, product, product.replace('= ', '= -')), diagnosis)
            # A common scale error can preserve conservation while failing
            # reconstruction. Require both results, not merely any failure.
            changed = body
            for edge in ('hi', 'lo'):
                product = f'{axis}flux_{edge} = rho_{mass_component}_avg_{edge} * interp_{edge};'
                changed = once(changed, product, product.replace('= ', '= 2 * '))
            yield mom_case(axis + ' both fluxes doubled', changed, (
                f'Centered_2nd/{component}.{axis}/hi: horizontal flux matches reconstruction [sat]',
                f'Centered_2nd/{component}.{axis}: shared horizontal flux agrees [unsat]'))

        # A later vertical block must not corrupt an earlier horizontal result.
        marker = 'zflux_lo = rho_w_avg_lo * interp_lo;'
        yield mom_case('vertical block overwrites horizontal flux',
                       once(body, marker, marker + '\n xflux_hi = 0;'),
                       (f'Centered_2nd/{component}.x/hi: horizontal flux matches reconstruction [sat]',))
        signature = re.search(rf'AdvectionSrcFor{component}Mom_N\s*\(([^()]*)\)', body)
        changed = once(body, signature[0], signature[0].replace('int i, int j', 'int j, int i'))
        yield mom_case('swapped helper indices', changed)
        average = re.search(r'\brho_u_avg_hi\s*=\s*([^;]+);', body)
        changed = once(body, average[0], average[0].replace('myhalf', '(myhalf * myhalf * 2)'))
        yield mom_case('equivalent averaging coefficient', changed, mutated=False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-momentum-horizontal-mutants-'))
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
