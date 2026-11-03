#!/usr/bin/env python3
"""Fault injection for the finite spacing-consumer preservation contract."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_consumers import (FAST, FAST_H, FILES, FUNCTIONS, READS,
                                        SLOW_H, STRAIN, STRAIN_H, TAU, pattern)

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_consumers.py'
SUCCESS = '3 spacing consumer source contracts passed;'


def cases(originals):
    yield 'baseline', {}, False, SUCCESS

    def edit(name, path, before, after, diagnostic='FAIL source contract:', fault=True):
        assert originals[path].count(before) == 1, (name, originals[path].count(before))
        return name, {path: originals[path].replace(before, after)}, fault, diagnostic if fault else SUCCESS

    alias = 'auto dz_ptr = stretched_dz_d.data();'
    for path, header, name, const in FUNCTIONS:
        for label, statement in (
            ('vector clear', 'stretched_dz_d.clear();'),
            ('vector resize', 'stretched_dz_d.resize(0);'),
            ('vector swap', 'stretched_dz_d.swap(other);'),
            ('vector assignment', 'stretched_dz_d = other;'),
            ('vector escape', 'retain(stretched_dz_d);'),
            ('reference alias', 'auto& spacing = stretched_dz_d;'),
            ('second data pointer', 'auto extra = stretched_dz_d.data();'),
            ('pointer escape', 'retain(dz_ptr);'),
            ('pointer alias', 'auto extra = dz_ptr;'),
            ('pointer rebind', 'dz_ptr = nullptr;'),
            ('pointer increment', '++dz_ptr;'),
            ('element write', 'dz_ptr[0] = 1;'),
            ('element compound write', 'dz_ptr[0] *= 2;'),
            ('element increment', 'dz_ptr[0]++;'),
            ('element reference', 'auto& spacing = dz_ptr[0];'),
            ('element address', 'retain(&dz_ptr[0]);'),
            ('element mutable call', 'mutate(dz_ptr[0]);'),
            ('cast away const', 'const_cast<Real*>(dz_ptr)[0] = 1;'),
            ('explicit capture escape', 'retain([p = dz_ptr] { return p; });'),
            ('quoted brace hiding write', 'BL_PROFILE("}"); dz_ptr[0] = 1;'),
            ('quoted comment hiding write', 'BL_PROFILE("/*"); dz_ptr[0] = 1;'),
        ):
            yield edit(name+' '+label, path, alias, alias+' '+statement, 'unmodeled use of')
        for label, replacement in (
            ('wrong data source', 'auto dz_ptr = other.data();'),
            ('offset data source', 'auto dz_ptr = stretched_dz_d.data()+1;'),
            ('reference data binding', 'auto& dz_ptr = stretched_dz_d.data();'),
            ('nonstandalone data binding', 'if (false) '+alias),
        ):
            yield edit(name+' '+label, path, alias, replacement)
        yield edit(name+' body macro', path, alias, '#define dz_ptr bad_pointer\n'+alias, 'unsupported macro')
        yield edit(name+' body include', path, alias, alias+'\n#include "unmodeled.H"', 'unsupported consumer preprocessor')
        yield edit(name+' unsupported lexical splice', path, alias, alias+'\n// hidden \\\ndz_ptr[0] = 1;', 'unsupported C++ lexical form')
        # Check each individual read, including all repeated wall/interior sites.
        text = strip_comments(originals[path])
        index = 0
        for statement, count in READS[path]:
            matches = list(re.finditer(pattern(statement), text))
            assert len(matches) == count
            for match in matches:
                changed = match[0].replace('dz_ptr[', 'expose(dz_ptr)[', 1)
                assert changed != match[0]
                yield f'{name} read {index} pointer escape', {path: text[:match.start()]+changed+text[match.end():]}, True, 'scalar read inventory changed'
                index += 1
        # Change the capture/launch surrounding the first protected scalar read.
        anchor = re.search(pattern(READS[path][0][0]), text).start()
        start = text.rfind('[=]', 0, anchor)
        assert start >= 0
        for label, capture in [('reference kernel capture', '[&]'), ('reference pointer capture', '[=,&dz_ptr]')]:
            yield name+' '+label, {path: text[:start]+capture+text[start+3:]}, True, 'bare value-captured ParallelFor kernel'
        start = text.rfind('ParallelFor(', 0, anchor)
        assert start >= 0
        yield name+' unknown launch', {path: text[:start]+'deferFor('+text[start+len('ParallelFor('):]}, True, 'bare value-captured ParallelFor kernel'
        # Wrapping the complete body can retain implicit captures without a
        # second explicit use of the pointer. Such outer closures must fail.
        body_start = text.index('{', text.index(name+' '))
        body_end = len(text.rstrip())-1
        if path == TAU:
            body_end = text.rfind('}', body_start, text.index('void copy_surface_tau_for_implicit'))
        wrapped = text[:body_start+1]+'\nretain([&] {\n'+text[body_start+1:body_end]+'\n});\n'+text[body_end:]
        yield name+' deferred body', {path: wrapped}, True, 'host statement scope'
        wrapped = text[:body_start+1]+'\nauto later = [&] {\n'+text[body_start+1:body_end]+'\n};\n'+text[body_end:]
        yield name+' no-parameter outer closure', {path: wrapped}, True, 'nested in a closure'
        yield edit(name+' launcher shadow', path, alias, alias+' auto ParallelFor = other_launcher;', 'kernel launcher')
        wrapped = text[:body_start+1]+'\nif (flags[0]) {\n'+text[body_start+1:body_end]+'\n}\n'+text[body_end:]
        yield name+' indexed host condition', {path: wrapped}, False, SUCCESS

    for path in (TAU, FAST):
        text = originals[path]
        at = text.index('auto dz_ptr = stretched_dz_d.data();')
        after = text[:at]+'{ auto dz_ptr = stretched_dz_d.data(); }'+text[at+len(alias):]
        yield Path(path).name+' pointer out of scope', {path: after}, True, 'pointer does not dominate'

    for path, header, name, const in FUNCTIONS:
        text = originals[path]
        match = re.search(r'(?:amrex::)?Gpu::DeviceVector<(?:amrex::)?Real>& stretched_dz_d,', text)
        assert match
        diagnostic = 'parameter type stretched_dz_d' if const else 'type mismatch'
        yield edit(name+' definition copies vector', path, match[0], match[0].replace('&', ''), diagnostic)
        before = 'amrex::Gpu::DeviceVector<amrex::Real>& stretched_dz_d,'
        # The slow header has several spacing parameters; mutate only the tau declaration.
        at = originals[header].index(before, originals[header].index(name+' '))
        text = originals[header]
        yield name+' declaration copies vector', {header: text[:at]+before.replace('&', '')+text[at+len(before):]}, True, 'type mismatch'
        start = text.index('void '+name+' ')
        end = text.index(';', start)+1
        yield name+' disabled declaration', {header: text[:start]+'#if 0\n'+text[start:end]+'\n#endif\n'+text[end:]}, True, 'unsupported conditional scope'
    yield edit('wrong strain spacing', TAU, 'stretched_dz_d, dxInv,', 'other, dxInv,', 'tau to strain: argument binding')
    yield edit('strain forwarding nested expression', TAU, 'stretched_dz_d, dxInv,', 'mutate(stretched_dz_d), dxInv,', 'tau to strain: argument binding')
    yield edit('duplicate strain spacing argument', TAU, 'stretched_dz_d, dxInv,', 'stretched_dz_d, stretched_dz_d,', 'unmodeled use of')
    yield edit('wrong strain size', STRAIN, 'static_cast<int>(stretched_dz_d.size())', 'static_cast<int>(other.size())', 'spacing size')
    yield edit('mutable element alias in strain', STRAIN, 'Real dz0  = dz_ptr[k];\n            Real idz0', 'const Real& dz0 = dz_ptr[k];\n            Real idz0', 'scalar read inventory changed')
    yield edit('scalar token fusion', STRAIN, 'Real dz0  = dz_ptr[k];\n            Real idz0', 'Realdz0 = dz_ptr[k];\n            Real idz0', 'scalar read inventory changed')
    yield edit('conditional hidden pointer write', FAST, '#ifdef AMREX_USE_GPU', '#ifdef AMREX_USE_GPU\n dz_ptr[0] = 1;', 'conditional region')
    yield edit('conditional hidden vector clear', TAU, '#ifdef ERF_IMPLICIT_W', '#ifdef ERF_IMPLICIT_W\n stretched_dz_d.clear();', 'conditional region')
    for path in (SLOW_H, FAST_H, STRAIN_H):
        yield Path(path).name+' protected macro', {path: originals[path]+'\n#define stretched_dz_d other\n'}, True, 'unsupported macro'

    # These controls preserve the finite storage contract, not all numerics.
    yield 'comments and whitespace', {p: strip_comments(t).replace(',', ', /* gap */ ').replace(';', ';\n') for p, t in originals.items()}, False, SUCCESS
    for path, _, name, _ in FUNCTIONS:
        yield edit(name+' quoted brace control', path, alias, alias+' BL_PROFILE("} // /* dz_ptr[0] = 1;");', fault=False)
        yield edit(name+' unrelated local', path, alias, alias+' const int unrelated = 2;', fault=False)
        yield edit(name+' data whitespace', path, alias, 'auto dz_ptr = stretched_dz_d . data ( ) ;', fault=False)
    yield 'qualified kernels', {p: originals[p].replace('ParallelFor(', 'amrex::ParallelFor(') for p in (TAU, FAST, STRAIN)}, False, SUCCESS
    yield edit('unrelated fast arithmetic', FAST, 'Real h_zeta = dz_ptr[k] * dzi;\n                avg_xmom_arr',
               'Real h_zeta = dz_ptr[k] * dzi;\n                const Real unrelated = one + one;\n                avg_xmom_arr', fault=False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-consumers-mutants-'))
    originals = {path: (ROOT/path).read_text() for path in FILES}
    fixtures = list(cases(originals))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, edits, fault, required) in enumerate(fixtures):
        root = work/f'{index:03d}'
        for path, text in originals.items():
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, text))
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught, expected, _ = mutation_outcome(result, fault)
        expected = expected and required in result.stdout
        faults += fault
        controls += not fault and name != 'baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
            print('required: '+required)
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
