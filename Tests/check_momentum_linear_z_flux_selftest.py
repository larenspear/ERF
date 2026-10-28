#!/usr/bin/env python3
"""Mutate linear reconstruction and its momentum composition; require detection."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_upwindall_z_reach import brace_block

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_linear_z_flux.py')
UPW = 'Source/Utils/ERF_Interpolation_UPW.H'
MOM = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'
ENUM = 'Source/ERF_IndexDefines.H'
POLICIES = [('CENTERED2', 'Centered_2nd'), ('UPWIND3', 'Upwind_3rd'),
            ('CENTERED4', 'Centered_4th'), ('UPWIND5', 'Upwind_5th'), ('CENTERED6', 'Centered_6th')]


def once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f'expected one anchor {old!r}, got {text.count(old)}')
    return text.replace(old, new)


def struct(text, name):
    match = re.search(rf'^struct {name}\s*\n\{{', text, re.M)
    if match is None:
        raise AssertionError(f'missing struct {name}')
    body = brace_block(text, match.end() - 1)
    return match.end(), match.end() + len(body), body


def method(body, name):
    match = re.search(rf'\b{name}\s*\([^()]*\)\s*const\s*\{{', body)
    if match is None:
        raise AssertionError(f'missing method {name}')
    return brace_block(body, match.end() - 1)


def cases(originals):
    yield 'baseline', None, None, False, ()
    source = originals[UPW]
    for name, scheme in POLICIES:
        start, end, body = struct(source, name)
        zbody = method(body, 'InterpolateInZ')
        ebody = method(body, 'Evaluate')
        diagnosis = (f'{name}/{scheme}: source reconstruction matches independent weights [sat]',)

        def case(label, changed, required=diagnosis, mutated=True):
            if changed == body:
                raise AssertionError(f'unchanged case {label}')
            return name + ' ' + label, UPW, source[:start] + changed + source[end:], mutated, required

        coefficient = re.search(r'static constexpr amrex::Real g1\s*=\s*([^;]+);', body)
        if coefficient is None:
            raise AssertionError('missing leading coefficient')
        altered = coefficient[0].replace(coefficient[1], '(' + coefficient[1] + ' + amrex::Real(1))')
        yield case('wrong leading coefficient', once(body, coefficient[0], altered),
                   diagnosis + (f'{scheme}/hi: composed source flux matches reconstruction [sat]',
                                f'{scheme}: reconstructed shared vertical flux agrees [unsat]'))

        call = re.search(r'Evaluate\(([^()]*)\)', zbody)
        args = call[1].split(',')
        swapped = [args[1], args[0], *args[2:]]
        changed = once(zbody, call[0], 'Evaluate(' + ','.join(swapped) + ')')
        # CENTERED2 is symmetric in its two samples: these permutations are
        # equivalent programs, not faults. The asymmetric cases must fail.
        symmetric = name == 'CENTERED2'
        yield case('swapped Evaluate arguments', once(body, zbody, changed),
                   () if symmetric else diagnosis, not symmetric)
        signature = re.search(r'Evaluate\s*\(([^()]*)\)\s*const', body)
        params = signature[1].split(',')
        swapped_params = [params[1], params[0], *params[2:]]
        changed = once(body, signature[0], signature[0].replace(signature[1], ','.join(swapped_params)))
        yield case('swapped Evaluate parameters', changed, () if symmetric else diagnosis, not symmetric)
        if name == 'UPWIND3':
            changed = once(changed, zbody, once(zbody, call[0], 'Evaluate(' + ','.join(swapped) + ')'))
            yield case('consistent parameter/call permutation', changed, (), False)

        read = re.search(r'm_phi\(([^()]*)\)', zbody)
        indices = read[1].split(',')
        indices[2] = 'k+4'
        changed = once(zbody, read[0], 'm_phi(' + ','.join(indices) + ')')
        yield case('wrong sample index', once(body, zbody, changed))
        returned = re.search(r'return\s+([^;]+);', ebody)
        changed = once(ebody, returned[0], 'return -(' + returned[1] + ');')
        yield case('negated reconstruction return', once(body, ebody, changed))
        zsig = re.search(r'InterpolateInZ\s*\(([^()]*)\)', body)
        params = zsig[1].split(',')
        params[0], params[2] = params[2], params[0]
        changed = once(body, zsig[0], zsig[0].replace(zsig[1], ','.join(params)))
        yield case('swapped Z index parameters', changed, ())

        if name.startswith('UPWIND'):
            yield case('reversed upwind sign test', once(body, zbody, once(zbody, 'upw_lo > 0', 'upw_lo < 0')), ())
            yield case('omitted blend scaling', once(body, zbody, once(zbody, 'upw_lo *= m_upw_frac;', '')), ())
            changed = once(zbody, call[0], call[0].replace('upw_lo', 'sm1'))
            yield case('uses sample as upwind argument', once(body, zbody, changed))
            yield case('constructor resets blend', once(body, ', m_upw_frac(upw_frac)\n    {}',
                                                       ', m_upw_frac(upw_frac)\n    {m_upw_frac = 0;}'), ())
        if name == 'CENTERED2':
            duplicate = once(zbody, call[0], 'Evaluate(' + args[0] + ',' + args[0] + ')')
            yield case('duplicated Evaluate sample', once(body, zbody, duplicate))
            yield case('equivalent leading coefficient', once(body, 'g1=(myhalf)',
                                                              'g1=(amrex::Real(1)/amrex::Real(2))'), (), False)
        if name == 'CENTERED6':
            declarations = list(re.finditer(r'amrex::Real\s+s\w*\s*=\s*m_phi\([^;]+;', zbody))
            if len(declarations) != 6:
                raise AssertionError('expected six stencil declarations')
            first, second = declarations[:2]
            changed = zbody[:first.start()] + second[0] + zbody[first.end():second.start()] + first[0] + zbody[second.end():]
            yield case('reordered independent loads', once(body, zbody, changed), (), False)

    start, end, body = struct(source, 'UPWINDALL')
    zbody, ebody = method(body, 'InterpolateInZ'), method(body, 'Evaluate')

    def wall_case(label, changed, required=()):
        return 'UPWINDALL ' + label, UPW, source[:start] + changed + source[end:], True, required

    changed = once(zbody, 'val_lo = myhalf *', 'val_lo = (myhalf + myhalf) *')
    yield wall_case('doubled live early return', once(body, zbody, changed),
                    ('UPWINDALL/Centered_2nd: source reconstruction matches independent weights [sat]',))
    for _, scheme in POLICIES[1:]:
        branch = re.search(rf'if \(adv_type == AdvType::{scheme}\)\s*\{{\s*iv = ([^;]+);', ebody)
        if branch is None:
            raise AssertionError('missing wall Evaluate branch')
        changed = once(ebody, branch[0], branch[0].replace(branch[1], '-(' + branch[1] + ')'))
        yield wall_case(scheme + ' negated polynomial', once(body, ebody, changed),
                        (f'UPWINDALL/{scheme}: source reconstruction matches independent weights [sat]',))
    yield wall_case('extra write before return', once(body, 'return ( iv );', 'iv *= -1; return ( iv );'))
    yield wall_case('constructor resets blend', once(body, ', m_upw_frac(upw_frac)\n    {}',
                                                    ', m_upw_frac(upw_frac)\n    {m_upw_frac = 0;}'))
    yield wall_case('wrong Evaluate enum argument', once(body, zbody, once(zbody,
                    'Evaluate(sp2,sp1,s,sm1,sm2,sm3,upw_lo,adv_type)',
                    'Evaluate(sp2,sp1,s,sm1,sm2,sm3,upw_lo,upw_lo)')))
    yield wall_case('early-return destination changed', once(body, zbody, once(zbody,
                    'val_lo = myhalf *', 'upw_lo = myhalf *')))

    mom = originals[MOM]
    start = mom.index('AdvectionSrcForZMom_N (')
    end = mom.index('/**', start)
    helper = mom[start:end]

    def momentum(label, old, new, required):
        return label, MOM, mom[:start] + once(helper, old, new) + mom[end:], True, required

    for edge, q in [('hi', 'k+1'), ('lo', 'k')]:
        actual = f'interp_w_v.InterpolateInZ(i,j,{q},0,interp_{edge},rho_w_avg_{edge});'
        yield momentum(f'{edge} reversed mass-flux direction at call', actual,
                       actual.replace(f',rho_w_avg_{edge}', f',-rho_w_avg_{edge}'),
                       (f'Upwind_3rd/{edge}: composed source flux matches reconstruction [sat]',
                        f'Centered_2nd/{edge}: composed source flux matches reconstruction [unsat]',
                        'Upwind_3rd: reconstructed shared vertical flux agrees [sat]'))
        product = f'zflux_{edge} = rho_w_avg_{edge} * interp_{edge};'
        yield momentum(f'{edge} reversed flux sign', product, product.replace('= ', '= -'),
                       (f'Centered_2nd/{edge}: composed source flux matches reconstruction [sat]',))
    product = 'zflux_lo = rho_w_avg_lo * interp_lo;'
    yield momentum('low block corrupts high flux', product, product + '\n zflux_hi = 0;',
                   ('Centered_2nd/hi: composed source flux matches reconstruction [sat]',))
    changed = once(helper, 'rho_w(i,j,k) + rho_w(i,j,k+1)', 'rho_w(i,j,k+1) + rho_w(i,j,k)')
    yield 'equivalent momentum average', MOM, mom[:start] + changed + mom[end:], False, ()
    constants = originals[CONST_H]
    yield ('source three constant changed', CONST_H,
           once(constants, 'three        = amrex::Real(3.0)', 'three        = amrex::Real(4.0)'), True, ())


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix='erf-momentum-linear-z-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (UPW, MOM, ENUM, CONST_H)}
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
        expected = expected and not missing
        unexpected += not expected
        mutants += mutated
        positives += not mutated and name != 'baseline'
        print(f"[{'CAUGHT' if expected and caught else 'passes' if expected else 'UNEXPECTED'}] {name}")
        for failure in failures[:2] + ['missing diagnostic: ' + message for message in missing]:
            print('           ' + failure)
    print(f'\n{mutants} mutations; {positives} equivalent rewrites; {unexpected} unexpected outcome(s)')
    return 1 if unexpected else 0


if __name__ == '__main__':
    sys.exit(main())
