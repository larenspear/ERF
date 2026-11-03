#!/usr/bin/env python3
"""Mutation fixtures for normal-advection spacing forwarding and kernel reads."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_advection import (CONSTANT, FILES, FUNCTIONS, GUARD, HEADER,
    NORMAL, READS, STRETCHED, VECTOR, VERT, WRAPPER)
from check_stretched_dz_bindings import brace_block, no_literals, signature
from check_stretched_dz_consumers import pattern
from momentum_geometry import call_arguments

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_advection.py'
SUCCESS = '4 spacing advection source contracts passed;'


def cases(originals):
    yield 'baseline', {}, False, SUCCESS

    def fixture(name, path, changed, fault=True):
        assert changed != originals[path], name
        return name, {path: changed}, fault, 'FAIL source contract:' if fault else SUCCESS

    for path, name in FUNCTIONS:
        text = strip_comments(originals[path])
        _, body, match = signature(no_literals(text), name)
        begin, end = match.end(), match.end()+len(body)
        for label, code in (
            ('clear', 'stretched_dz_d.clear();'),
            ('resize', 'stretched_dz_d.resize(0);'),
            ('swap', 'stretched_dz_d.swap(other);'),
            ('assignment', 'stretched_dz_d = other;'),
            ('reference alias', 'auto& alias = stretched_dz_d;'),
            ('pointer alias', 'auto alias = stretched_dz_d.data();'),
            ('element write', 'stretched_dz_d[0] = 1;'),
            ('element reference', 'const auto& alias = stretched_dz_d[0];'),
            ('escape', 'retain(stretched_dz_d);'),
            ('explicit capture', 'retain([&stretched_dz_d] {});'),
            ('quoted brace hiding', 'BL_PROFILE("}"); retain(stretched_dz_d);'),
            ('quoted comment hiding', 'BL_PROFILE("/*"); retain(stretched_dz_d);'),
            ('body include', '\n#include "unmodeled.H"\n'),
            ('conditional hidden write', '\n#if 0\nstretched_dz_d.clear();\n#endif\n'),
        ):
            yield fixture(name+' '+label, path, text[:begin]+code+text[begin:])
        for label, prefix, suffix, fault in (
            ('deferred body', 'retain([&] {', '});', True),
            ('named closure', 'auto deferred = [=] {', '};', True),
            ('generic closure', 'auto deferred = [&]<typename T>() {', '};', True),
            ('host block control', '{', '}', False),
            ('indexed host control', 'if (flags[0]) {', '}', False),
        ):
            yield fixture(name+' '+label, path, text[:begin]+prefix+text[begin:end]+suffix+text[end:], fault)
        yield fixture(name+' unrelated local', path, text[:begin]+'const int unused = 1;'+text[begin:], False)
        yield fixture(name+' literal control', path, text[:begin]+'BL_PROFILE("} // /* stretched_dz_d dz_ptr");'+text[begin:], False)
        for shadow in (VERT, WRAPPER, 'ParallelFor', 'dz_ptr'):
            prefix = text[match.start():match.end()]
            changed, count = re.subn(r'\bbxx\b', shadow, prefix)
            assert count == 1
            yield fixture(name+' parameter shadow '+shadow, path, text[:match.start()]+changed+text[match.end():])

        for target, ending in ((path, '{'), (HEADER, ';')) if path != NORMAL else ((path, '{'),):
            source = strip_comments(originals[target])
            _, _, declaration = signature(no_literals(source), name, ending)
            fragment = source[declaration.start():declaration.end()]
            found = re.search(r'const\s+(?:amrex::)?Gpu::DeviceVector\s*<\s*(?:amrex::)?Real\s*>\s*&\s*stretched_dz_d', fragment)
            assert found
            for label, replacement in (
                ('mutable parameter', found[0].replace('const ', '', 1)),
                ('copied parameter', found[0].replace('&', '')),
                ('pointer parameter', found[0].replace('&', '*')),
                ('fused const token', re.sub(r'const\s+(?:amrex::)?Gpu', 'constGpu', found[0])),
            ):
                changed = fragment[:found.start()]+replacement+fragment[found.end():]
                yield fixture(name+' '+label+' '+Path(target).name, target,
                              source[:declaration.start()]+changed+source[declaration.end():])

        if name != WRAPPER:
            callee = WRAPPER if name == VERT else VERT
            calls = list(call_arguments(body, r'(?<![\w:.>])'+callee+r'\s*<[^<>]+>'))
            assert len(calls) == 12
            for index, arguments in enumerate(calls):
                slots = [(lo, hi) for lo, hi in arguments if body[lo:hi].strip() == VECTOR]
                assert len(slots) == 1
                lo, hi = (begin+v for v in slots[0])
                for label, replacement in (('wrong vector', 'other'), ('raw pointer', 'stretched_dz_d.data()'),
                                           ('mutating expression', 'mutate(stretched_dz_d)')):
                    yield fixture(f'{name} forwarding {index} '+label, path, text[:lo]+replacement+text[hi:])
                lo, hi = (begin+v for v in arguments[0])
                yield fixture(f'{name} forwarding {index} extra alias', path, text[:lo]+VECTOR+text[hi:])
            yield fixture(name+' local callee shadow', path, text[:begin]+f'auto {callee} = other;'+text[begin:])
            first = re.search(r'\b'+callee+r'\s*<[^<>]+>', body)
            lo, hi = begin+first.start(), begin+first.end()
            yield fixture(name+' wrong template', path, text[:lo]+callee+'<Other>'+text[hi:])

        if path == STRETCHED or name == WRAPPER:
            alias = 'auto dz_ptr = stretched_dz_d.data();'
            at = text.index(alias, begin)
            after = at+len(alias)
            for label, code in (
                ('pointer escape', 'retain(dz_ptr);'),
                ('pointer alias', 'auto alias = dz_ptr;'),
                ('pointer reference', 'auto& alias = dz_ptr;'),
                ('element address', 'retain(&dz_ptr[0]);'),
                ('element reference', 'const auto& alias = dz_ptr[0];'),
                ('pointer write', 'const_cast<Real*>(dz_ptr)[0] = 1;'),
                ('pointer rebind', 'dz_ptr = other;'),
                ('pointer increment', '++dz_ptr;'),
            ):
                yield fixture(name+' '+label, path, text[:after]+code+text[after:])
            for label, replacement in (
                ('wrong origin', 'auto dz_ptr = other.data();'),
                ('offset origin', 'auto dz_ptr = stretched_dz_d.data()+1;'),
                ('reference origin', 'auto& dz_ptr = stretched_dz_d.data();'),
                ('sibling origin', '{'+alias+'}'),
            ):
                yield fixture(name+' '+label, path, text[:at]+replacement+text[after:])
            # Each of the six scalar reads must remain a value declaration in
            # the immediate inline kernel, not a nested/retained evaluation.
            typ = 'Real' if path == STRETCHED else 'auto'
            reads = [read for statement, _ in READS for read in re.finditer(pattern(typ+' '+statement), body)]
            assert len(reads) == 3
            for index, read in enumerate(reads):
                lo, hi = begin+read.start(), begin+read.end()
                for label, replacement in (
                    ('reference read', read[0].replace(typ, typ+'&', 1)),
                    ('deferred read', 'retain([=] {'+read[0]+'});'),
                    ('call expression read', 'consume('+read[0]+');'),
                ):
                    yield fixture(f'{name} read {index} '+label, path, text[:lo]+replacement+text[hi:])
            captures = list(re.finditer(r'\[=\]\s*AMREX_GPU_DEVICE\s*\(int i, int j, int k\)', body))
            assert len(captures) == 3
            for index, capture in enumerate(captures):
                at = begin+capture.start()
                for label, replacement in (('reference capture', '[&]'), ('explicit reference capture', '[=,&dz_ptr]')):
                    yield fixture(f'{name} kernel {index} '+label, path, text[:at]+replacement+text[at+3:])
                brace = text.index('{', at)
                close = brace+len(brace_block(text, brace))+2
                yield fixture(f'{name} kernel {index} retained closure', path, text[:at]+'retain('+text[at:close]+')'+text[close:])
            yield fixture(name+' local launcher shadow', path, text[:begin]+'auto ParallelFor = other;'+text[begin:])

    for path in FILES:
        for label, extra in (('macro', '#define stretched_dz_d other'), ('line splice', '// hidden \\\nstretched_dz_d.clear();'),
                             ('raw string', 'R"(stretched_dz_d)";'), ('digraph', '<% stretched_dz_d.clear(); %>')):
            yield fixture(Path(path).name+' '+label, path, originals[path]+'\n'+extra+'\n')
    for path, name in ((CONSTANT, 'AdvectionSrcForMom_ConstantDz'), (STRETCHED, 'AdvectionSrcForMom_StretchedDz')):
        yield fixture(name+' extra header overload', HEADER, originals[HEADER]+f'\nfloat {name}(float);\n')
        yield fixture(name+' suffix write', path, originals[path]+'\nstretched_dz_d.clear();\n')
    yield fixture('guarded declaration disabled', HEADER, originals[HEADER].replace('#define '+GUARD, '#define '+GUARD+'\n#if 0', 1))
    for label, extra in (('conditional', '#if 0'), ('unexpected include', '#include "other.H"'),
                         ('overload', 'float '+WRAPPER+'(float);'), ('specialization', 'template<> void '+VERT+'<Other>() {}'),
                         ('global spacing alias', 'auto& alias = stretched_dz_d;')):
        yield fixture('normal header '+label, NORMAL, originals[NORMAL]+'\n'+extra+'\n')
    yield fixture('normal header unrelated helper', NORMAL, originals[NORMAL]+'\ninline int unrelated(int value) { return value; }\n', False)
    yield 'comments and whitespace', {p: strip_comments(t).replace(',', ', /* gap */ ').replace(';', ';\n') for p, t in originals.items()}, False, SUCCESS


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-advection-mutants-'))
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
