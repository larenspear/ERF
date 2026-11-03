#!/usr/bin/env python3
"""Mutation fixtures for diffusion spacing preservation and pointer forwarding."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import brace_block, no_literals, signature
from check_stretched_dz_diffusion import (BUILD, DIV, FILES, GRAD, GUARDS, HEADER,
    IMPL, MOM, RESULT, SCALAR, SETUP, STATE, WEIGHTS)
from momentum_geometry import call_arguments, unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_diffusion.py'
SUCCESS = '7 spacing diffusion source contracts passed;'


def cases(originals):
    yield 'baseline', {}, False, SUCCESS

    def edit(name, path, before, after, fault=True):
        assert originals[path].count(before) == 1, (name, originals[path].count(before))
        return name, {path: originals[path].replace(before, after)}, fault, 'FAIL source contract:' if fault else SUCCESS

    entries = ((STATE, 'DiffusionSrcForState_S'), (MOM, 'DiffusionSrcForMom'))
    functions = (*entries, (SCALAR, BUILD), (SCALAR, IMPL), (SCALAR, DIV))
    for path, name in functions:
        text = strip_comments(originals[path])
        _, body, match = signature(no_literals(text), name)
        begin, end = match.end(), match.end()+len(body)
        pointer = 'dz_ptr' if path in (STATE, MOM) else 'dz'
        additions = [
            ('pointer escape', f'retain({pointer});'),
            ('pointer alias', f'auto alias = {pointer};'),
            ('pointer reference', f'auto& alias = {pointer};'),
            ('element reference', f'const auto& alias = {pointer}[0];'),
            ('element address', f'retain(&{pointer}[0]);'),
            ('element write', f'const_cast<Real*>({pointer})[0] = 1;'),
            ('pointer increment', f'++{pointer};'),
            ('pointer rebind', f'{pointer} = other;'),
            ('explicit capture', f'retain([{pointer}] {{}});'),
            ('quoted brace hiding', f'BL_PROFILE("}}"); retain({pointer});'),
            ('quoted comment hiding', f'BL_PROFILE("/*"); retain({pointer});'),
            ('body include', '\n#include "unmodeled.H"\n'),
            ('conditional hidden write', f'\n#if 0\nretain({pointer});\n#endif\n'),
        ]
        if path in (STATE, MOM):
            additions += [(label, code) for label, code in (
                ('vector clear', 'stretched_dz_d.clear();'),
                ('vector resize', 'stretched_dz_d.resize(0);'),
                ('vector swap', 'stretched_dz_d.swap(other);'),
                ('vector assignment', 'stretched_dz_d = other;'),
                ('vector alias', 'auto& alias = stretched_dz_d;'),
                ('vector escape', 'retain(stretched_dz_d);'),
            )]
            at = text.index('auto dz_ptr = stretched_dz_d.data();', begin)+len('auto dz_ptr = stretched_dz_d.data();')
        else:
            at = begin
        for label, code in additions:
            yield name+' '+label, {path: text[:at]+code+text[at:]}, True, 'FAIL source contract:'
        for label, prefix, suffix, fault in (
            ('deferred body', 'retain([&] {', '});', True),
            ('named closure', 'auto deferred = [=] {', '};', True),
            ('generic closure', 'auto deferred = [&]<typename T>() {', '};', True),
            ('host block control', '{', '}', False),
            ('indexed host control', 'if (flags[0]) {', '}', False),
        ):
            yield name+' '+label, {path: text[:begin]+prefix+text[begin:end]+suffix+text[end:]}, fault, 'FAIL source contract:' if fault else SUCCESS
        yield name+' unrelated local', {path: text[:begin]+'const int unused = 1;'+text[begin:]}, False, SUCCESS
        yield name+' quoted control', {path: text[:begin]+f'BL_PROFILE("}} // /* {pointer}");'+text[begin:]}, False, SUCCESS
        prefix = text[match.start():match.end()]
        changed, count = re.subn(r'\b(?:bx|bxx)\b', 'ParallelFor', prefix)
        assert count == 1
        yield name+' launcher parameter shadow', {path: text[:match.start()]+changed+text[match.end():]}, True, 'FAIL source contract:'
        for target, ending in ((path, '{'), (HEADER, ';')) if path in (STATE, MOM) else ((path, '{'),):
            source = strip_comments(originals[target])
            _, _, definition = signature(no_literals(source), name, ending)
            fragment = source[definition.start():definition.end()]
            before = 'const Gpu::DeviceVector<Real>&' if path in (STATE, MOM) else 'const amrex::Real*'
            if target == HEADER:
                before = 'const amrex::Gpu::DeviceVector<amrex::Real>&'
            assert fragment.count(before) == 1
            for label, replacement in (
                ('mutable parameter', before.replace('const ', '')),
                ('reference parameter', before+'&'),
            ):
                changed = fragment.replace(before, replacement)
                yield name+' '+label+' '+Path(target).name, {target: source[:definition.start()]+changed+source[definition.end():]}, True, 'FAIL source contract:'

    for path, name in entries:
        for label, replacement in (
            ('wrong data origin', 'auto dz_ptr = other.data();'),
            ('offset data origin', 'auto dz_ptr = stretched_dz_d.data()+1;'),
            ('reference data origin', 'auto& dz_ptr = stretched_dz_d.data();'),
            ('pointer token fusion', 'auto other_dz_ptr = stretched_dz_d.data();'),
        ):
            yield edit(name+' '+label, path, 'auto dz_ptr = stretched_dz_d.data();', replacement)

    for path, name, callees in ((STATE, 'DiffusionSrcForState_S', (BUILD, DIV)),
                               (SCALAR, BUILD, (IMPL,)), (SCALAR, IMPL, (GRAD,))):
        text = strip_comments(originals[path])
        _, body, definition = signature(no_literals(text), name)
        origin = definition.end()
        for callee in callees:
            expression = r'(?<![\w:.>])'+callee+(r'\s*<\s*(?:true|false)\s*,\s*(?:true|false)\s*>' if callee == IMPL else '')
            calls = list(call_arguments(body, expression))
            for index, arguments in enumerate(calls):
                pointer = 'dz_ptr' if path == STATE else 'dz'
                slots = [(lo, hi) for lo, hi in arguments if body[lo:hi].strip() == pointer]
                assert len(slots) == 1
                lo, hi = (origin+v for v in slots[0])
                for label, replacement in (
                    ('wrong pointer', 'other'), ('offset pointer', pointer+'+1'),
                    ('mutating argument', 'mutate('+pointer+')'),
                ):
                    yield f'{name} to {callee} {index} '+label, {path: text[:lo]+replacement+text[hi:]}, True, 'FAIL source contract:'
                # Another argument must not receive an unchecked pointer alias.
                lo, hi = (origin+v for v in arguments[0])
                yield f'{name} extra alias to {callee} {index}', {path: text[:lo]+pointer+text[hi:]}, True, 'FAIL source contract:'
            yield name+' local shadow '+callee, {path: text[:origin]+f'auto {callee} = other;'+text[origin:]}, True, 'FAIL source contract:'
            # Matching declaration types would not catch a renamed parameter.
            prefix = text[definition.start():definition.end()]
            changed, count = re.subn(r'\bbx\b', callee, prefix)
            assert count == 1
            yield name+' parameter shadow '+callee, {path: text[:definition.start()]+changed+text[definition.end():]}, True, 'FAIL source contract:'

    for label, code in (
        ('vector alias', 'auto& alias = stretched_dz_d;'),
        ('vector mutation', 'stretched_dz_d.clear();'),
        ('closure', 'retain([&] { retain(stretched_dz_d); });'),
        ('callee shadow', 'auto '+BUILD+' = other;'),
        ('nested include', '#include "unmodeled.H"'),
        ('macro', '#define dz_ptr other'),
    ):
        yield 'setup '+label, {SETUP: originals[SETUP]+'\n'+code+'\n'}, True, 'FAIL source contract:'
    yield 'setup harmless local', {SETUP: originals[SETUP]+'\nconst int unused = 1;\n'}, False, SUCCESS

    # Each selected kernel capture is exercised independently, including all
    # three momentum components sharing a single ParallelFor call.
    for path, name in ((MOM, 'DiffusionSrcForMom'), (SCALAR, IMPL), (SCALAR, DIV)):
        text = strip_comments(originals[path])
        _, body, definition = signature(no_literals(text), name)
        captures = list(re.finditer(r'\[=\]\s*AMREX_GPU_DEVICE', body))
        selected = captures[3:6] if path == MOM else [captures[2]] if name == IMPL else captures
        assert len(selected) == (3 if path == MOM else 1)
        for index, match in enumerate(selected):
            at = definition.end()+match.start()
            for label, capture in (('reference capture', '[&]'), ('explicit reference capture', '[=,&'+('dz_ptr' if path == MOM else 'dz')+']')):
                yield f'{name} kernel {index} '+label, {path: text[:at]+capture+text[at+3:]}, True, 'FAIL source contract:'
            brace = text.index('{', at)
            end = brace+len(brace_block(text, brace))+2
            yield f'{name} kernel {index} retained closure', {path: text[:at]+'retain('+text[at:end]+')'+text[end:]}, True, 'FAIL source contract:'

    for name in (GRAD, WEIGHTS):
        text = strip_comments(originals[SCALAR])
        definition = unique(r'\b'+name+r'\s*\([^;{}]*?\)\s*noexcept\s*\{', text, name)
        at = definition.end()
        pointer = 'dz' if name == GRAD else 'dz_near'
        for label, code in (('escape', f'retain({pointer});'), ('defer', f'retain([&] {{ retain({pointer}); }});'),
                            ('include', '\n#include "unmodeled.H"\n')):
            yield name+' '+label, {SCALAR: text[:at]+code+text[at:]}, True, 'FAIL source contract:'
    for label, before, after in (
        ('weights reference', 'const amrex::Real dz_near, const amrex::Real dz_next', 'const amrex::Real& dz_near, const amrex::Real dz_next'),
        ('weights result pointer', 'amrex::Real c1, c2, c3;', 'amrex::Real *c1, c2, c3;'),
        ('weights extra field', 'amrex::Real c1, c2, c3;', 'amrex::Real c1, c2, c3; const amrex::Real* alias;'),
        ('weights copy hook', 'amrex::Real c1, c2, c3;', 'amrex::Real c1, c2, c3; StretchedDiffusionBoundaryWeights(const StretchedDiffusionBoundaryWeights& x) { retain(&x); }'),
        ('gradient element address', 'StretchedDiffusionDirichletWeights(dz[k], dz[k + 1])', 'StretchedDiffusionDirichletWeights(&dz[k], dz[k + 1])'),
        ('gradient reference', 'const amrex::Real inv = (k == klo)', 'const amrex::Real& inv = (k == klo)'),
        ('divergence reference', 'const amrex::Real dzk_inv =', 'const amrex::Real& dzk_inv ='),
        ('divergence token fusion', 'const amrex::Real dzk_inv =', 'const amrex::Realdzk_inv ='),
        ('gradient result reference', 'const Real grad = StretchedScalarGradient(', 'const Real& grad = StretchedScalarGradient('),
    ):
        yield edit(label, SCALAR, before, after)
    for name in (BUILD, IMPL, DIV, GRAD, WEIGHTS, RESULT):
        yield name+' extra declaration', {SCALAR: originals[SCALAR]+f'\nvoid {name}();\n'}, True, 'FAIL source contract:'
    for path in FILES:
        for label, extra in (('macro', '#define dz other'), ('line splice', '// hidden \\\nretain(dz);'), ('raw string', 'R"(dz)";')):
            yield Path(path).name+' '+label, {path: originals[path]+'\n'+extra+'\n'}, True, 'FAIL source contract:'
    for path in GUARDS:
        guard = GUARDS[path]
        yield edit(Path(path).name+' extra conditional', path, '#define '+guard, '#define '+guard+'\n#if 0')
    yield 'comments and whitespace', {p: strip_comments(t).replace(',', ', /* gap */ ').replace(';', ';\n') for p, t in originals.items()}, False, SUCCESS


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-diffusion-mutants-'))
    originals = {path: (ROOT/path).read_text() for path in FILES}
    fixtures = list(cases(originals))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, edits, fault, required) in enumerate(fixtures):
        root = work/f'{index:03d}'
        for path, source in originals.items():
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, source))
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
