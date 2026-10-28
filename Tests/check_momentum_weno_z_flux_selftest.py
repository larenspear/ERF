#!/usr/bin/env python3
"""Fault-inject WENO scalar, interpolation, and momentum composition bindings.

Required sat diagnostics refer to the arbitrary-kernel interface model. They
must not be interpreted as concrete WENO sample counterexamples. Scalar faults
must fail their scalar prerequisite even when interface identities still hold.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_momentum_linear_z_flux_selftest import ENUM, MOM, UPW, method, once, struct

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_weno_z_flux.py')
HEADERS = ('Source/Utils/ERF_Interpolation_WENO.H', 'Source/Utils/ERF_Interpolation_WENO_Z.H')
POLICIES = [('WENO3', 'Weno_3'), ('WENO5', 'Weno_5'), ('WENO7', 'Weno_7'),
            ('WENO_Z3', 'Weno_3Z'), ('WENO_MZQ3', 'Weno_3MZQ'),
            ('WENO_Z5', 'Weno_5Z'), ('WENO_Z7', 'Weno_7Z')]


def cases(originals):
    yield 'baseline', None, None, False, ()
    for name, scheme in POLICIES:
        header = HEADERS[int('_' in name)]
        source = originals[header]
        start, end, body = struct(source, name)
        zbody, ebody = method(body, 'InterpolateInZ'), method(body, 'Evaluate')
        diagnosis = (f'{name}: source Z assembly matches oriented scalar kernels [sat]',)

        def case(label, changed, required=diagnosis, mutated=True):
            if changed == body:
                raise AssertionError(f'unchanged case {name} {label}')
            return name + ' ' + label, header, source[:start] + changed + source[end:], mutated, required

        read = re.search(r'm_phi\(([^()]*)\)', zbody)
        indices = read[1].split(',')
        for label, slot, value in [('wrong vertical sample', 2, 'k+5'), ('cross-axis sample', 0, 'i+1'),
                                  ('wrong component', 3, 'qty_index+1')]:
            args = list(indices)
            args[slot] = value
            changed = once(zbody, read[0], 'm_phi(' + ','.join(args) + ')')
            yield case(label, once(body, zbody, changed), () if slot == 3 else diagnosis)

        calls = list(re.finditer(r'amrex::Real\s+(Fhi|Flo)\s*=\s*Evaluate\(([^()]*)\);', zbody))
        if len(calls) != 2:
            raise AssertionError('expected two Evaluate calls')
        for call in calls:
            args = call[2].split(',')
            args[0] = args[1]
            changed = once(zbody, call[0], call[0].replace(call[2], ','.join(args)))
            yield case(call[1] + ' duplicates sample', once(body, zbody, changed))

        swapped = zbody.replace('Real Fhi =', 'Real Ftemp =').replace('Real Flo =', 'Real Fhi =')
        swapped = swapped.replace('Real Ftemp =', 'Real Flo =')
        yield case('swapped call destinations', once(body, zbody, swapped))

        signature = re.search(r'Evaluate\s*\(([^()]*)\)\s*const', body)
        params = signature[1].split(',')
        params[0], params[1] = params[1], params[0]
        permuted = once(body, signature[0], signature[0].replace(signature[1], ','.join(params)))
        yield case('swapped Evaluate parameters', permuted)
        changed = permuted
        for call in calls:
            args = call[2].split(',')
            args[0], args[1] = args[1], args[0]
            if changed.count(call[0]) != 3:
                raise AssertionError('expected matching X/Y/Z calls')
            changed = changed.replace(call[0], call[0].replace(call[2], ','.join(args)))
        yield case('consistent parameter and call permutation', changed, (), False)

        normalization = re.search(r'if \(upw_lo != amrex::Real\(0\)\)[^;]+;', zbody)
        yield case('missing flow normalization', once(body, zbody, once(zbody, normalization[0], '')), ())
        yield case('reversed sign test', once(body, zbody, once(zbody, 'upw_lo > 0', 'upw_lo < 0')), ())
        # At +/-1 this is unchanged; at zero it loses the symmetric average.
        # Multiplication by zero flow hides this fault in the momentum flux.
        result = re.search(r'val_lo = ([^;]+);', zbody)
        changed = once(zbody, result[0], 'val_lo = (upw_lo * upw_lo) * (' + result[1] + ');')
        yield case('zero-flow reconstruction lost', once(body, zbody, changed), diagnosis +
                   (f'{scheme}/hi: composed WENO flux matches reconstruction [unsat]',
                    f'{scheme}: reconstructed shared vertical flux agrees [unsat]'))
        changed = once(zbody, result[0], result[0].replace(' + upw_lo', ' - upw_lo', 1))
        yield case('wrong positive-flow coefficient', once(body, zbody, changed))

        zsig = re.search(r'InterpolateInZ\s*\(([^()]*)\)', body)
        params = zsig[1].split(',')
        params[0], params[2] = params[2], params[0]
        yield case('swapped Z index parameters', once(body, zsig[0], zsig[0].replace(zsig[1], ','.join(params))), ())
        yield case('unbound constructor array', once(body, ': m_phi(phi) {}', ': m_phi() {}'), ())

        returned = re.search(r'return\s+([^;]+);', ebody)
        changed = once(ebody, returned[0], 'return -(' + returned[1] + ');')
        yield case('negated scalar reconstruction', once(body, ebody, changed),
                   (f'FAIL {name}: complete source return equals the normalized candidate reconstruction',
                    f'{name}: source Z assembly matches oriented scalar kernels [unsat]'))
        candidate = re.search(r'amrex::Real v1\s*=\s*([^;]+);', ebody)
        changed = once(ebody, candidate[0], candidate[0].replace(candidate[1], '(' + candidate[1] + ') + s'))
        yield case('wrong scalar candidate', once(body, ebody, changed),
                   (f'FAIL {name}.v1: constant-data candidate scale',))
        # A modular transport identity can hold for the wrong scalar formula.
        # Require the independent numerical prerequisites to reject it while
        # the unchanged source/transport bindings still prove their identities.
        beta = re.search(r'amrex::Real b1\s*=\s*([^;]+);', ebody)
        changed = once(ebody, beta[0], beta[0].replace(beta[1], 'amrex::Real(2) * (' + beta[1] + ')'))
        preserved = (f'{scheme}/hi: composed WENO flux matches reconstruction [unsat]',
                     f'{scheme}: reconstructed shared vertical flux agrees [unsat]')
        yield case('positive but wrong scalar smoothness', once(body, ebody, changed),
                   (f'FAIL {name}.b1: equals derivative-integral smoothness indicator',) + preserved)
        changed = once(ebody, returned[0],
                       'amrex::Real unused = amrex::Real(0)/b1;\n' + returned[0])
        yield case('undefined scalar evaluation despite unchanged return', once(body, ebody, changed),
                   (f'FAIL {name}: constant source divisions are defined for eps>0',
                    f'FAIL {name}: affine source divisions are defined for eps>0',
                    f'ok   {name}: source ideal return matches full-stencil reconstruction') + preserved)
        selected = (re.search(r'amrex::Real v3\s*=\s*([^;]+);', ebody)
                    if name == 'WENO_MZQ3' else candidate)
        changed = once(ebody, selected[0], selected[0].replace(selected[1], '(' + selected[1] + ') + (s-sm1)'))
        yield case('scalar candidate loses affine exactness', once(body, ebody, changed),
                   (f'ok   {name}: actual nonlinear return reproduces constant fields for eps>0 [unsat]',
                    f'FAIL {name}: actual nonlinear return reproduces affine fields for eps>0 [sat]') + preserved)
        ideal = re.search(r'static constexpr amrex::Real g1\s*=\s*([^;]+);', body)
        changed = once(body, ideal[0], ideal[0].replace(ideal[1], 'amrex::Real(2) * (' + ideal[1] + ')'))
        yield case('positive but wrong scalar ideal weight', changed,
                   (f'FAIL {name}: source ideal return matches full-stencil reconstruction',) + preserved)
        changed = once(ebody, returned[0], 'v1 *= -1; ' + returned[0])
        yield case('late scalar write', once(body, ebody, changed), ())
        changed = once(zbody, result[0], result[0] + '\n val_lo *= -1;')
        yield case('late interpolation write', once(body, zbody, changed), ())

        # Positives ensure statement order is tracked without demanding the
        # original spelling/order where independent operations commute.
        if name in ('WENO3', 'WENO_Z7'):
            changed = once(zbody, calls[0][0], 'CALL_PLACEHOLDER')
            changed = once(changed, calls[1][0], calls[0][0]).replace('CALL_PLACEHOLDER', calls[1][0])
            yield case('reordered independent calls', once(body, zbody, changed), (), False)
        if name == 'WENO3':
            changed = once(ebody, 'w1 * v1 + w2 * v2', 'w2 * v2 + w1 * v1')
            yield case('reordered scalar numerator', once(body, ebody, changed), (), False)

    mom = originals[MOM]
    start = mom.index('AdvectionSrcForZMom_N (')
    end = mom.index('/**', start)
    helper = mom[start:end]

    def momentum(label, old, new, required):
        return label, MOM, mom[:start] + once(helper, old, new) + mom[end:], True, required

    for edge, q in [('hi', 'k+1'), ('lo', 'k')]:
        call = f'interp_w_v.InterpolateInZ(i,j,{q},0,interp_{edge},rho_w_avg_{edge});'
        diagnosis = (f'Weno_3/{edge}: composed WENO flux matches reconstruction [sat]',
                     'Weno_3: reconstructed shared vertical flux agrees [sat]')
        yield momentum(edge + ' reversed flow', call, call.replace(',rho_w_avg_', ',-rho_w_avg_'), diagnosis)
        yield momentum(edge + ' shifted face', call, call.replace(f',j,{q},', ',j,k+2,'), diagnosis)
        yield momentum(edge + ' wrong component', call, call.replace(',0,', ',1,'), diagnosis)
        product = f'zflux_{edge} = rho_w_avg_{edge} * interp_{edge};'
        yield momentum(edge + ' negated flux', product, product.replace('= ', '= -'), diagnosis)
    product = 'zflux_lo = rho_w_avg_lo * interp_lo;'
    yield momentum('low block overwrites high flux', product, product + '\n zflux_hi = 0;',
                   ('Weno_3/hi: composed WENO flux matches reconstruction [sat]',))
    changed = once(helper, 'rho_w(i,j,k) + rho_w(i,j,k+1)', 'rho_w(i,j,k+1) + rho_w(i,j,k)')
    yield 'reordered momentum average', MOM, mom[:start] + changed + mom[end:], False, ()

    source = originals[UPW]
    start, end, body = struct(source, 'UPWINDALL')
    zbody = method(body, 'InterpolateInZ')
    changed = once(body, zbody, once(zbody, 'val_lo = myhalf *', 'val_lo = (myhalf + myhalf) *'))
    yield 'doubled nearest-wall reconstruction', UPW, source[:start] + changed + source[end:], True, (
        'wall/Centered_2nd: source reconstruction matches independent weights [sat]',
        'Weno_3/hi: composed WENO flux matches reconstruction [sat]',
        'Weno_3: reconstructed shared vertical flux agrees [unsat]')


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-momentum-weno-z-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (*HEADERS, UPW, MOM, ENUM, CONST_H)}
    unexpected = mutants = positives = 0
    for index, (name, path, changed, mutated, required) in enumerate(cases(originals)):
        root = work / f'{index:03d}'
        for source_path, original in originals.items():
            target = root / source_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(changed if source_path == path else original)
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        if result.returncode == 77:
            print(result.stdout.strip())
            return 77
        caught, expected, failures = mutation_outcome(result, mutated)
        missing = [message for message in required if message not in result.stdout]
        if 'unknown' in result.stdout:
            missing.append('all solver obligations must be decided')
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
