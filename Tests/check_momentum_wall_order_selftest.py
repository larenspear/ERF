#!/usr/bin/env python3
"""Mutate the wall selector, vertical flux wiring, and live reduced-stencil reads."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_wall_order.py')
MOM = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'
UPW = 'Source/Utils/ERF_Interpolation_UPW.H'
ENUM = 'Source/ERF_IndexDefines.H'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f'expected one mutation anchor: {old!r}, found {text.count(old)}')
    return text.replace(old, new)


def cases(originals):
    # (name, path, replacement source, reject?, required diagnostic fragments)
    yield 'baseline', None, None, False, ()
    original = originals[MOM]
    start = original.index('AdvectionSrcForZMom_N (')
    end = original.index('/**', start)
    helper = original[start:end]

    def change(name, old, new, required=()):
        changed = replace_once(helper, old, new)
        return name, MOM, original[:start] + changed + original[end:], True, required

    for edge, bound, sign, other, other_sign in [('hi', 'hi_z_face', '-', 'lo_z_face', '+'),
                                                ('lo', 'lo_z_face', '+', 'hi_z_face', '-')]:
        routing = (f'{edge}: nearest-wall policy, call bindings, and flux product [sat]',)
        near = f'k == {bound}{sign}1'
        second = f'k == {bound}{sign}2 || k == {other}{other_sign}1'
        # Limit each replacement to its own face block when the two blocks share
        # identical scheme conditions or UPWINDALL arguments.
        block_start = helper.index(f'if (k == {bound})')
        block_end = helper.index('if (k == lo_z_face)') if edge == 'hi' else helper.index('amrex::Real mfsq')
        block = helper[block_start:block_end]

        def local(name, old, new, diagnostics=routing):
            changed = replace_once(block, old, new)
            changed_helper = helper[:block_start] + changed + helper[block_end:]
            return name, MOM, original[:start] + changed_helper + original[end:], True, diagnostics

        yield local(f'{edge} direct branch moved inward', f'if (k == {bound})', f'if ({near})')
        yield local(f'{edge} nearest-wall test moved', f'if ({near})', f'if (k == {bound}{sign}2)')
        yield local(f'{edge} second-band same-wall distance changed', second,
                    second.replace(f'{bound}{sign}2', f'{bound}{sign}3'),
                    routing + ('adjacent interior cells compute the same shared vertical flux [sat]',))
        yield local(f'{edge} second-band opposite-wall distance changed', second,
                    second.replace(f'{other}{other_sign}1', f'{other}{other_sign}2'))
        yield local(f'{edge} second-band OR becomes AND', second, second.replace('||', '&&'))
        exempt = 'vert_adv_type != AdvType::Centered_2nd && vert_adv_type != AdvType::Upwind_3rd'
        yield local(f'{edge} preserve test uses OR', exempt, exempt.replace('&&', '||'))
        yield local(f'{edge} preserves wrong scheme', exempt, exempt.replace('Upwind_3rd', 'Weno_3'))
        yield local(f'{edge} nearest wall uses third order', 'rho_w_avg_' + edge + ',AdvType::Centered_2nd',
                    'rho_w_avg_' + edge + ',AdvType::Upwind_3rd')
        yield local(f'{edge} fourth-order cap raised to sixth', ',AdvType::Centered_4th', ',AdvType::Centered_6th',
                    routing + ('reduced wall stencils read only within [lo, hi] [sat]',))
        yield local(f'{edge} discards preserved scheme', ',vert_adv_type);', ',AdvType::Centered_4th);')

        calls = list(re.finditer(r'interp_w_(?:v|wall)\.InterpolateInZ\(([^()]+)\);', block))
        if len(calls) != 4:
            raise AssertionError(f'{edge}: expected four interpolation sites')
        for index, call in enumerate(calls):
            args = call[1].split(',')
            for slot, value, label in [(2, 'k' if edge == 'hi' else 'k+1', 'shifted face'),
                                        (4, 'interp_lo' if edge == 'hi' else 'interp_hi', 'wrong destination'),
                                        (5, 'rho_w_avg_lo' if edge == 'hi' else 'rho_w_avg_hi', 'wrong mass-flux argument')]:
                altered = list(args)
                altered[slot] = value
                yield local(f'{edge} site {index} {label}', call[0], call[0].replace(call[1], ','.join(altered)))
        call = calls[0][0]
        yield local(f'{edge} duplicated interpolation', call, call + '\n' + call,
                    (f'{edge}: exactly one interpolation call on every interior path [sat]',
                     f'{edge}: nearest-wall policy, call bindings, and flux product [unsat]',
                     'adjacent interior cells compute the same shared vertical flux [unsat]'))
        yield local(f'{edge} omitted interpolation', call, '',
                    (f'{edge}: exactly one interpolation call on every interior path [sat]',))
        yield local(f'{edge} off-axis interpolation', calls[-1][0], calls[-1][0].replace('InterpolateInZ', 'InterpolateInX'), ())
        yield local(f'{edge} wrong component', call, call.replace(',0,', ',1,'))
        yield local(f'{edge} off-axis cell', call, call.replace('(i,j,', '(i+1,j,'))
        avg = f'rho_w_avg_{edge} = myhalf'
        yield local(f'{edge} doubled mass average', avg, f'rho_w_avg_{edge} = (myhalf + myhalf)')
        product = f'zflux_{edge} = rho_w_avg_{edge} * interp_{edge};'
        yield local(f'{edge} reversed flux sign', product, product.replace('= ', '= -'))
        direct = re.search(rf'zflux_{edge}\s*=\s*rho_w\(i,j,k\)\s*\*\s*w\(i,j,k\);', block)
        if direct is None:
            raise AssertionError('missing direct boundary flux')
        yield local(f'{edge} direct flux samples wrong velocity', direct[0], direct[0].replace('w(i,j,k);', 'w(i,j,k-1);'),
                    (f'{edge}: external direct flux without interpolation [sat]',))

    product = 'zflux_lo = rho_w_avg_lo * interp_lo;'
    yield change('low block overwrites high flux', product, product + '\n zflux_hi = 0;',
                 ('hi: nearest-wall policy, call bindings, and flux product [sat]',))
    yield change('write after checked flux blocks', 'amrex::Real mfsq', 'zflux_hi = 0;\n amrex::Real mfsq')
    yield change('early return in high block', 'if (k == hi_z_face) {', 'if (k == hi_z_face) { return 0;')
    yield change('protected boundary assignment', product, product + '\n hi_z_face = 0;')
    yield change('swapped helper boundary parameters', 'const int lo_z_face, const int hi_z_face',
                 'const int hi_z_face, const int lo_z_face')

    upw = originals[UPW]
    start_upw = upw.index('struct UPWINDALL')
    prefix, wall = upw[:start_upw], upw[start_upw:]
    gate = 'adv_type == AdvType::Upwind_5th || adv_type == AdvType::Centered_6th'
    if wall.count(gate) != 2:
        raise AssertionError('expected two outer-load gates')
    changed = wall.replace(gate, gate + ' || adv_type == AdvType::Centered_4th', 1)
    yield ('wall cap reads outer positive cell', UPW, prefix + changed, True,
           ('reduced wall stencils read only within [lo, hi] [sat]',))
    old = 'const int& k,\n                    const int& qty_index,'
    new = 'const int& qty_index,\n                    const int& k,'
    yield 'wall callee swapped index/component parameters', UPW, prefix + replace_once(wall, old, new), True, ()
    constants = originals[CONST_H]
    yield ('source half constant changed', CONST_H,
           replace_once(constants, 'myhalf       = amrex::Real(0.5)', 'myhalf       = amrex::Real(0.25)'), True, ())

    # Equivalent rewrites must pass the semantic checks without editing the oracle.
    controls = [
        ('reversed band predicates', helper.replace('k == hi_z_face-2 || k == lo_z_face+1',
                                                    'k == lo_z_face+1 || k == hi_z_face-2')),
        ('equivalent comparison direction', helper.replace('k == lo_z_face+1', 'lo_z_face+1 == k')),
        ('reordered mass average', helper.replace('rho_w(i,j,k) + rho_w(i,j,k+1)',
                                                 'rho_w(i,j,k+1) + rho_w(i,j,k)')),
        ('reordered flux factors', helper.replace('rho_w_avg_lo * interp_lo', 'interp_lo * rho_w_avg_lo')),
    ]
    for name, changed in controls:
        if changed == helper:
            raise AssertionError(f'positive control did not change source: {name}')
        yield name, MOM, original[:start] + changed + original[end:], False, ()


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix='erf-momentum-wall-order-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (MOM, UPW, ENUM, CONST_H)}
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
        for failure in (failures[:2] + ['missing diagnostic: ' + message for message in missing]):
            print('           ' + failure)
    print(f'\n{mutants} mutations; {positives} equivalent rewrites; {unexpected} unexpected outcome(s)')
    return 1 if unexpected else 0


if __name__ == '__main__':
    sys.exit(main())
