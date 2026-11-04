#!/usr/bin/env python3
"""Mutation fixtures for implicit CPU/GPU spacing preservation contracts."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import signature, no_literals
from check_stretched_dz_consumers import pattern
from check_stretched_dz_implicit import FILES, HEADER, MACRO, MOM, READS, SETUP, SOURCE, STATE
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_implicit.py'
SUCCESS = '5 spacing implicit source contracts passed;'


def raw_function(text, name):
    # The raw body is deliberately not brace-parsed: CPU and GPU alternatives
    # have different brace counts. The known next declaration marks its end.
    definition = unique(r'\bvoid\s+'+name+r'\s*\([^{}]*?\)\s*\{', text, name)
    boundary = text.index('template <int stagdir>') if name == STATE else text.index('#define '+MACRO)
    end = text.rfind('}', definition.end(), boundary)
    assert end > definition.end()
    return definition, end


def cases(originals):
    yield 'baseline', {}, False, SUCCESS

    def fixture(name, path, text, fault=True):
        assert text != originals[path], name
        return name, {path: text}, fault, 'FAIL source contract:' if fault else SUCCESS

    source = strip_comments(originals[SOURCE])
    for name in (STATE, MOM):
        definition, end = raw_function(source, name)
        begin = definition.end()
        body = source[begin:end]
        alias = unique(pattern('auto dz_ptr = stretched_dz_d.data();'), body, name+' alias')
        at = begin+alias.end()
        for label, code in (
            ('vector clear', 'stretched_dz_d.clear();'),
            ('vector resize', 'stretched_dz_d.resize(0);'),
            ('vector swap', 'stretched_dz_d.swap(other);'),
            ('vector alias', 'auto& alias = stretched_dz_d;'),
            ('vector escape', 'retain(stretched_dz_d);'),
            ('pointer alias', 'auto alias = dz_ptr;'),
            ('pointer reference', 'auto& alias = dz_ptr;'),
            ('pointer escape', 'retain(dz_ptr);'),
            ('element reference', 'const auto& alias = dz_ptr[0];'),
            ('element address', 'retain(&dz_ptr[0]);'),
            ('pointer write', 'const_cast<Real*>(dz_ptr)[0] = 1;'),
            ('pointer rebind', 'dz_ptr = other;'),
            ('pointer increment', '++dz_ptr;'),
            ('explicit capture', 'retain([&dz_ptr] {});'),
            ('quoted brace hiding', 'BL_PROFILE("}"); retain(dz_ptr);'),
            ('quoted comment hiding', 'BL_PROFILE("/*"); retain(dz_ptr);'),
            ('unknown include', '\n#include "unmodeled.H"\n'),
            ('unsupported conditional', '\n#if 0\nretain(dz_ptr);\n#endif\n'),
            ('launcher shadow', 'auto ParallelFor = other;'),
        ):
            yield fixture(name+' '+label, SOURCE, source[:at]+code+source[at:])
        for gpu in (False, True):
            for label, code, fault in (
                ('hidden mutation', 'const_cast<Real*>(dz_ptr)[0] = 1;', True),
                ('hidden escape', 'retain(stretched_dz_d);', True),
                ('harmless local', 'const int unused = 1;', False),
            ):
                conditional = '\n#ifdef AMREX_USE_GPU\n'+(code if gpu else '')+'\n#else\n'+('' if gpu else code)+'\n#endif\n'
                yield fixture(name+f' {"GPU" if gpu else "CPU"} '+label, SOURCE, source[:at]+conditional+source[at:], fault)
        for label, prefix, suffix, fault in (
            ('deferred body', 'retain([&] {', '});', True),
            ('named body', 'auto deferred = [=] {', '};', True),
            ('generic body', 'auto deferred = [&]<typename T>() {', '};', True),
            ('host block', '{', '}', False),
            ('indexed host block', 'if (flags[0]) {', '}', False),
        ):
            yield fixture(name+' '+label, SOURCE, source[:begin]+prefix+body+suffix+source[end:], fault)
        yield fixture(name+' literal control', SOURCE, source[:at]+'BL_PROFILE("} // /* dz_ptr stretched_dz_d");'+source[at:], False)
        for shadow in ('dz_ptr', 'ParallelFor', 'dz_inv'):
            fragment = source[definition.start():definition.end()]
            changed, count = re.subn(r'\bbx\b', shadow, fragment)
            assert count == 1
            yield fixture(name+' parameter shadow '+shadow, SOURCE, source[:definition.start()]+changed+source[definition.end():])

        for target, ending in ((SOURCE, '{'), (HEADER, ';')):
            text = source if target == SOURCE else strip_comments(originals[HEADER])
            match = definition if target == SOURCE else signature(no_literals(text), name, ending)[2]
            fragment = text[match.start():match.end()]
            parameter = unique(r'const\s+(?:amrex::)?Gpu::DeviceVector<(?:amrex::)?Real>\s*&\s*stretched_dz_d', fragment, 'vector parameter')
            for label, replacement in (
                ('mutable parameter', parameter[0].replace('const ', '', 1)),
                ('copied parameter', parameter[0].replace('&', '')),
                ('fused const token', re.sub(r'const\s+(?:amrex::)?Gpu', 'constGpu', parameter[0])),
            ):
                changed = fragment[:parameter.start()]+replacement+fragment[parameter.end():]
                yield fixture(name+' '+label+' '+Path(target).name, target, text[:match.start()]+changed+text[match.end():])
        for label, replacement in (
            ('wrong origin', 'auto dz_ptr = other.data();'),
            ('offset origin', 'auto dz_ptr = stretched_dz_d.data()+1;'),
            ('reference origin', 'auto& dz_ptr = stretched_dz_d.data();'),
            ('sibling origin', '{'+alias[0]+'}'),
        ):
            lo, hi = begin+alias.start(), begin+alias.end()
            yield fixture(name+' '+label, SOURCE, source[:lo]+replacement+source[hi:])

        for index, statement in enumerate(READS):
            read = unique(pattern(statement), body, name+' read')
            lo, hi = begin+read.start(), begin+read.end()
            for label, replacement in (
                ('wrong pointer', read[0].replace('dz_ptr', 'other')),
                ('retained read', 'retain([=] {'+read[0]+'});'),
                ('nested expression', 'consume('+read[0]+');'),
            ):
                yield fixture(f'{name} read {index} '+label, SOURCE, source[:lo]+replacement+source[hi:])
        # A scalar declaration at column scope must not become a reference, or
        # be shadowed before the protected assignments in a nested block.
        locals_ = unique(pattern('Real dz_inv, dz_inv_lo, dz_inv_hi;'), body, 'inverse locals')
        lo, hi = begin+locals_.start(), begin+locals_.end()
        for label, replacement in (
            ('reference local', 'Real& dz_inv = other; Real dz_inv_lo, dz_inv_hi;'),
            ('fused local type', 'Realdz_inv, dz_inv_lo, dz_inv_hi;'),
            ('late locals', ''),
        ):
            changed = source[:lo]+replacement+source[hi:]
            if label == 'late locals':
                changed = changed[:end-(hi-lo)]+locals_[0]+changed[end-(hi-lo):]
            yield fixture(name+' '+label, SOURCE, changed)
        first = unique(pattern(READS[0]), body, 'bottom inverse')
        at = begin+first.start()
        for label, code in (('reference shadow', 'Real& dz_inv = other;'),
                            ('class shadow', 'Custom dz_inv;'), ('structured shadow', 'auto [dz_inv] = other;')):
            yield fixture(name+' '+label, SOURCE, source[:at]+code+source[at:])
        capture = unique(r'\[=\]\s*AMREX_GPU_DEVICE', body, 'column capture')
        at = begin+capture.start()
        for label, replacement in (('reference capture', '[&]'), ('explicit reference capture', '[=,&dz_ptr]')):
            yield fixture(name+' '+label, SOURCE, source[:at]+replacement+source[at+3:])
        cpu = unique(pattern('for(int j(jlo);j<=jhi;++j) { for(int i(ilo);i<=ihi;++i) {'), body, 'CPU column')
        at = begin+cpu.start()
        closing = unique(r'#else\s*\n\s*\}\s*\}\s*\n#endif', body, 'CPU closing branch')
        close_at = begin+body.index('#endif', closing.start())
        changed = source[:at]+'retain([&] { '+source[at:close_at]+'});\n'+source[close_at:]
        yield name+' deferred CPU column', {SOURCE: changed}, True, 'protected spacing use must be in host statement scope'
        # This changes only the CPU header; both configurations must be checked.
        yield fixture(name+' CPU loop binding', SOURCE, source[:at]+source[at:].replace('j(jlo)', 'j(other)', 1))

    for label, code in (
        ('vector escape', 'retain(stretched_dz_d);'), ('vector clear', 'stretched_dz_d.clear();'),
        ('pointer alias', 'auto alias = stretched_dz_d.data();'), ('include', '#include "other.H"'),
        ('macro', '#define dz_ptr other'), ('conditional', '#ifdef AMREX_USE_GPU\nretain(stretched_dz_d);\n#endif'),
    ):
        yield fixture('setup '+label, SETUP, originals[SETUP]+'\n'+code+'\n')
    yield fixture('setup local control', SETUP, originals[SETUP]+'\nconst int unused = 1;\n', False)

    macro_at = originals[SOURCE].index('#define '+MACRO)
    prefix, macro = originals[SOURCE][:macro_at], originals[SOURCE][macro_at:]
    for label, before, after in (
        ('wrong instantiation', MACRO+'(2)', MACRO+'(3)'),
        ('missing instantiation', MACRO+'(1)', ''),
        ('extra instantiation', MACRO+'(2)', MACRO+'(2)\n'+MACRO+'(3)'),
        ('object-like macro', MACRO+'(STAGDIR)', MACRO+' (STAGDIR)'),
        ('mutable vector type', 'const Gpu::DeviceVector<Real>&', 'Gpu::DeviceVector<Real>&'),
        ('fused const token', 'const Gpu::DeviceVector<Real>&', 'constGpu::DeviceVector<Real>&'),
        ('wrong direction parameter', '<STAGDIR>', '<0>'),
        ('extra body effect', 'const bool);', 'const bool); retain(stretched_dz_d);'),
        ('missing first continuation', ') \\\n', ')\n'),
        ('continued invocation', 'const bool);\n', 'const bool); \\\n'),
        ('comment eats body', ') \\\n', ') // hidden \\\n'),
    ):
        assert macro.count(before) == 1, label
        yield fixture('macro '+label, SOURCE, prefix+macro.replace(before, after))
    yield fixture('extra macro use', SOURCE, originals[SOURCE]+'\n'+MACRO+'(0)\n')
    yield fixture('macro comment control', SOURCE, prefix+macro.replace('const Box&,', 'const /* box */ Box&,'), False)
    yield fixture('macro trailing comment control', SOURCE, originals[SOURCE]+'\n// end of instantiations\n', False)
    for path in FILES:
        for label, code in (('macro', '#define stretched_dz_d other'), ('line splice', '// hidden \\\nretain(stretched_dz_d);'),
                            ('raw string', 'R"(stretched_dz_d)";'), ('digraph', '<% retain(stretched_dz_d); %>')):
            yield fixture(Path(path).name+' '+label, path, originals[path]+'\n'+code+'\n')
    for name in (STATE, MOM):
        yield fixture(name+' header overload', HEADER, originals[HEADER]+f'\nfloat {name}(float);\n')
    yield fixture('unknown preprocessor condition', SOURCE, originals[SOURCE].replace('AMREX_USE_GPU', 'OTHER_GPU'))
    yield fixture('header condition', HEADER, originals[HEADER].replace('#define ERF_DIFFUSION_H_', '#define ERF_DIFFUSION_H_\n#if 0', 1))
    yield fixture('default argument mutation', HEADER, originals[HEADER].replace('const bool use_mrf_countergradient = false);', 'const bool use_mrf_countergradient = mutate());'))
    yield 'comments and whitespace', {p: strip_comments(t).replace(',', ', /* gap */ ').replace(';', ';\n') for p, t in originals.items()}, False, SUCCESS
    yield 'CRLF control', {p: t.replace('\n', '\r\n') for p, t in originals.items()}, False, SUCCESS


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-implicit-mutants-'))
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
