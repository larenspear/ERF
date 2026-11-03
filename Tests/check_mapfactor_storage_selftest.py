#!/usr/bin/env python3
"""Mutation and positive controls for the finite map-factor storage contract.

Faults must produce explicit source-contract failures, never crashes. Controls
cover enum aliases/permutations, reordered allocations, literal preprocessor
branches and comments. The six-slot control is a supported storage configuration,
not a claim of whole-program equivalence or non-conformal mapping support.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_mapfactor_storage import ALLOCATOR, MAPFAC_H
from check_momentum_dispatch_selftest import replace_once

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_mapfactor_storage.py')
PASSED = 'Map-factor allocation slice contract passed'


def cases(originals):
    yield 'baseline', {}, False, PASSED
    original = originals[ALLOCATOR]

    def edit(name, before, after, diagnostic='', fault=True):
        return name, {ALLOCATOR: replace_once(original, before, after)}, fault, diagnostic if fault else PASSED

    projection = 'for (auto& b : bl2d) {\n        b.setRange(2,0);\n    }'
    for before, after, label in [
        ('ba.boxList()', 'other.boxList()', 'wrong projection source'),
        ('auto& b', 'auto b', 'projection by value'),
        ('setRange(2,0)', 'setRange(0,0)', 'compress X instead of Z'),
        ('setRange(2,0)', 'setRange(2,1)', 'project to k=1'),
        ('setRange(2,0)', 'setRange(2,0,2)', 'two Z cells'),
        ('setRange(2,0);', 'setRange(2,0); b.grow(1);', 'extra box growth'),
    ]:
        region = 'BoxList bl2d = ba.boxList();\n    ' + projection
        yield edit(label, region, replace_once(region, before, after), 'unsupported source')
    yield edit('missing projection', projection, '', 'unsupported source')
    for before, after in [
        ('ba2d[lev]  = BoxArray(std::move(bl2d));', 'ba2d[lev] = BoxArray(std::move(bl1d));'),
        ('ba2d[lev]  = BoxArray(std::move(bl2d));', 'ba2d[lev+1] = BoxArray(std::move(bl2d));'),
        ('define_column_kextent(lev, ba, dm);', 'define_column_kextent(lev, ba, dm); mapfac[lev].clear();'),
        ('mapfac[lev].resize(MapFacType::num);', 'mapfac[lev].resize(2);'),
        ('mapfac[lev].resize(MapFacType::num);', ''),
    ]:
        yield edit('changed setup: ' + after, before, after, 'unsupported source')

    lines = {}
    for grid in 'muv':
        line = re.search(r'^\s*mapfac\[lev\]\[MapFacType::' + grid + r'_x\][^\n]+;', original, re.M)[0]
        lines[grid] = line
        yield edit(grid + ' missing allocation', line, '', 'not every live map-factor slot is allocated')
        yield edit(grid + ' duplicate allocation', line, line + line, 'duplicate allocation')
        for axis, ghosts in [('x', '2,3,0'), ('y', '3,2,0'), ('z', '3,3,1')]:
            yield edit(grid + ' wrong ' + axis + ' ghosts', line, line.replace('IntVect(3,3,0)', 'IntVect(' + ghosts + ')'), 'ghost layout')
        for before, after, label, diagnostic in [
            (',dm,1,', ',dm,2,', 'component count', 'one component'),
            (',dm,1,', ',other_dm,1,', 'distribution mapping', 'unsupported source'),
            ('ba2d[lev]', 'ba[lev]', 'box source', 'unsupported source'),
            ('ba2d[lev]', 'ba2d[lev+1]', 'box level', 'unsupported source'),
            ('mapfac[lev]', 'mapfac[lev+1]', 'metric level', 'unsupported source'),
            ('std::make_unique<MultiFab>', 'std::make_unique<iMultiFab>', 'element type', 'unsupported source'),
        ]:
            yield edit(grid + ' wrong ' + label, line, line.replace(before, after), diagnostic)
        yield edit(grid + ' equivalent Y alias', line, line.replace('::' + grid + '_x]', '::' + grid + '_y]'), fault=False)

    for grid in 'uv':
        correct = 'IntVect(1,0,0)' if grid == 'u' else 'IntVect(0,1,0)'
        for wrong in ('IntVect(0,0,0)', 'IntVect(0,0,1)', 'IntVect(1,1,0)'):
            yield edit(grid + ' wrong staggering ' + wrong, lines[grid], lines[grid].replace(correct, wrong), 'wrong staggering')
    yield edit('mass becomes X nodal', lines['m'], lines['m'].replace('ba2d[lev]', 'convert(ba2d[lev],IntVect(1,0,0))'), 'wrong staggering')
    yield edit('swapped U/V allocation destinations', lines['u'] + '\n' + lines['v'],
               lines['u'].replace('::u_x]', '::v_x]') + '\n' + lines['v'].replace('::v_x]', '::u_x]'), 'wrong staggering')
    yield edit('reordered independent allocations', '\n'.join(lines[grid] for grid in 'muv'),
               '\n'.join(lines[grid] for grid in 'vmu'), fault=False)
    yield edit('explicit cell-centered mass conversion', lines['m'],
               lines['m'].replace('ba2d[lev]', 'convert(ba2d[lev],IntVect(0,0,0))'), fault=False)

    for before, after, label, diagnostic in [
        ('int i = 0; i < 3;', 'int i = 1; i < 3;', 'test skips first slot', 'test path must initialize'),
        ('int i = 0; i < 3;', 'int i = 0; i < 4;', 'test overruns container', 'range exceeds'),
        ('int i = 3; i < mapfac[lev].size();', 'int i = 2; i < mapfac[lev].size();', 'test overlapping ranges', 'duplicate map-factor initialization'),
        ('int i = 0; i < mapfac[lev].size();', 'int i = 1; i < mapfac[lev].size();', 'default skips first slot', 'default path must initialize'),
        ('mapfac[lev][i]->setVal(myhalf);', 'mapfac[lev][i]->setVal(one);', 'wrong test value', 'test path must initialize'),
        ('mapfac[lev][i]->setVal(one);', 'mapfac[lev][i]->setVal(myhalf);', 'wrong default value', 'default path must initialize'),
        ('mapfac[lev][i]->setVal(one);', 'mapfac[lev][i]->setVal(zero);', 'zero default value', 'unsupported map-factor initial value'),
        ('mapfac[lev][i]->setVal(one);', 'mapfac[lev][0]->setVal(one);', 'constant destination index', 'unsupported source'),
        ('mapfac[lev][i]->setVal(one);', 'mapfac[lev][i]->setVal(one); return;', 'extra control transfer', 'unsupported source'),
        ('if (solverChoice.test_mapfactor)', 'if (!solverChoice.test_mapfactor)', 'inverted test gate', 'unsupported source'),
    ]:
        yield edit(label, before, after, diagnostic)
    boundary = 'if (solverChoice.lsm_type == LandSurfaceType::SLM)'
    for statement in ('mapfac[lev].clear();', 'ba2d[lev].clear();', 'unknown(mapfac);', 'return;'):
        yield edit('intervening statement ' + statement, boundary, statement + '\n    ' + boundary, 'unmodeled statements')
    for directive in ('#if 0', '#if 1', '#ifdef SOMETHING'):
        changed = replace_once(original, 'BoxList bl2d =', directive + '\n    BoxList bl2d =')
        changed = replace_once(changed, boundary, '#endif\n    ' + boundary)
        yield 'preprocessor crossing slice start ' + directive, {ALLOCATOR: changed}, True, 'preprocessor region crosses slice boundary'
    changed = replace_once(original, 'BoxList bl2d =', 'if (false) { BoxList bl2d =')
    changed = replace_once(changed, boundary, '}\n    ' + boundary)
    yield 'runtime scope crossing slice start', {ALLOCATOR: changed}, True, 'caller slice ends inside'

    disabled = re.search(r'#if 0\n.*?#endif', original, re.S)[0]
    yield edit('removed disabled allocation code', disabled, '', fault=False)
    yield edit('enabled alias guards with isotropic enum', disabled, disabled.replace('#if 0', '#if 1'), fault=False)
    yield edit('disabled nonsense is inert', disabled, '#if 0\nnot valid C++;\n#endif', fault=False)
    for directive in ('#ifdef SOMETHING', '#if 2', '#if 0\n#if 0', '#else'):
        yield edit('unsupported directive ' + directive, disabled, disabled.replace('#if 0', directive), 'allocator preprocessor')

    enum = originals[MAPFAC_H]
    match = re.search(r'\benum MapFacType\s*\{[^{}]*\};', enum)

    def enum_edit(body):
        return enum[:match.start()] + 'enum MapFacType { ' + body + ' };' + enum[match.end():]

    independent = enum_edit('m_x, u_x, v_x, m_y, u_y, v_y, num')
    yield 'six-slot enum with disabled Y allocation', {MAPFAC_H: independent}, True, 'not every live map-factor slot is allocated'
    yield 'six-slot enum with enabled Y allocation', {
        MAPFAC_H: independent, ALLOCATOR: replace_once(original, disabled, disabled.replace('#if 0', '#if 1'))}, False, PASSED
    yield 'numeric slot permutation', {MAPFAC_H: enum_edit('m_x=2, u_x=0, v_x=1, num=3, m_y=m_x, u_y=u_x, v_y=v_x')}, False, PASSED
    yield 'named axis aliases', {MAPFAC_H: enum_edit('m_x, u_x, v_x, num, m_y=m_x, u_y=u_x, v_y=v_x')}, False, PASSED
    yield 'cross-stagger alias', {MAPFAC_H: enum_edit('m_x, u_x, v_x, num, m_y=u_x, u_y=u_x, v_y=v_x')}, True, 'alias different staggerings'
    yield 'four-slot enum with missing Y allocation', {MAPFAC_H: enum_edit('m_x, u_x, v_x, m_y, num, u_y=u_x, v_y=v_x')}, True, 'not every live map-factor slot is allocated'
    for name in ('one', 'myhalf', 'fourth'):
        constants = originals[CONST_H]
        changed, count = re.subn(r'(constexpr\s+amrex::Real\s+' + name + r'\s*=)[^;]+;', r'\g<1> -1;', constants)
        if count != 1:
            raise ValueError('constant mutation anchor failed: ' + name)
        yield 'nonpositive source constant ' + name, {CONST_H: changed}, True, 'source constant ' + name

    yield edit('allocator ba signature loses constness', 'const BoxArray& ba, const DistributionMapping& dm,',
               'BoxArray& ba, const DistributionMapping& dm,', 'allocator input signature changed')
    yield 'whitespace and mixed comments', {ALLOCATOR: original.replace(',', ',/* gap */ ').replace(';', '; //*** line comment\n')}, False, PASSED


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-mapfactor-storage-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (ALLOCATOR, MAPFAC_H, CONST_H)}
    fixtures = list(cases(originals))  # Check all mutation anchors before running.
    unexpected = faults = controls = 0
    for index, (name, edits, fault, diagnostic) in enumerate(fixtures):
        root = work / f'{index:03d}'
        for path, original in originals.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, original))
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        caught, expected, messages = mutation_outcome(result, fault)
        if diagnostic not in result.stdout:
            expected = False
            messages.append('missing diagnostic: ' + diagnostic)
        unexpected += not expected
        faults += fault
        controls += not fault and name != 'baseline'
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            for message in messages:
                print('           ' + message)
    print(f'\n{faults} mutations; {controls} equivalent/specification-preserving controls; {unexpected} unexpected outcome(s)')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
