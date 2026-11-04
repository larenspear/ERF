#!/usr/bin/env python3
"""Fault injection for the selected spacing-reader MFIter enclosure contract."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, signature
from check_stretched_dz_consumers import pattern
from check_stretched_dz_mfiter import (CONTROLS, DYCORE, FAST, FILES, FUNCTIONS,
                                     LOOPS, STRAIN, TAU, TILE, header)
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_mfiter.py'
SUCCESS = '4 MFIter enclosure source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path: strip_comments(text) for path, text in original.items()}

    def fixture(name, path, text, fault=True):
        assert text != texts[path], name
        return name, {**texts, path: text}, fault

    def replace(name, path, before, after, fault=True):
        m = unique(pattern(before), texts[path], name)
        return fixture(name, path, texts[path][:m.start()]+after+texts[path][m.end():], fault)

    yield replace('TileNoZ changes overload', TILE, 'amrex::IntVect TileNoZ', 'amrex::MFItInfo TileNoZ')
    yield replace('TileNoZ disables sync', TILE, 'return amrex::IntVect::TheZeroVector();',
                  'Gpu::setNoSyncRegion(true); return amrex::IntVect::TheZeroVector();')
    yield replace('TileNoZ unknown conditional', TILE, '#include <AMReX.H>',
                  '#include <AMReX.H>\n#ifdef UNKNOWN\n#endif')
    yield replace('TileNoZ hidden call', TILE, 'return amrex::IntVect::TheZeroVector();',
                  'other(); return amrex::IntVect::TheZeroVector();')

    bodies = {}
    for path, name in {**FUNCTIONS, STRAIN: 'ComputeStrain_S'}.items():
        _, body, definition = signature(texts[path], name)
        bodies[path] = body, definition.end()

    def body_fixture(name, path, changed, fault=True):
        body, offset = bodies[path]
        return fixture(name, path, texts[path][:offset]+changed+texts[path][offset+len(body):], fault)

    for path in FUNCTIONS:
        body, _ = bodies[path]
        label = FUNCTIONS[path]
        specs = LOOPS[path][-1:] if path == DYCORE else LOOPS[path]
        for container, tiling, _ in specs:
            matches = list(re.finditer(pattern(header(container, tiling)), body))
            for index, m in enumerate(matches):
                scope_end = m.end()+len(brace_block(body, m.end()-1))+1
                loop = body[m.start():scope_end]
                tag = label+' '+tiling+f' loop {index}'
                for name, changed in (
                    ('disabled sync', m[0].replace(tiling, 'MFItInfo().DisableDeviceSync()')),
                    ('disabled post sync', m[0].replace(tiling, 'MFItInfo().DisableDeviceSyncPost()')),
                    ('wrong iterator type', m[0].replace('MFIter', 'OtherIter')),
                    ('missing increment', m[0].replace('++mfi', '')),
                    ('early finalize condition', m[0].replace('mfi.isValid()', '(mfi.Finalize(), mfi.isValid())')),
                ):
                    yield body_fixture(tag+' '+name, path, body[:m.start()]+changed+body[m.end():])
                for name, changed in (
                    ('deferred loop', 'auto later = [&] { '+loop+' };'),
                    ('iterator alias', loop[:m.end()-m.start()]+'auto& alias = mfi;'+loop[m.end()-m.start():]),
                    ('iterator address escape', loop[:m.end()-m.start()]+'retain(&mfi);'+loop[m.end()-m.start():]),
                ):
                    yield body_fixture(tag+' '+name, path, body[:m.start()]+changed+body[scope_end:])
        for name, code in (
            ('NoSyncRegion', 'Gpu::NoSyncRegion region;'),
            ('SyncAtExitOnly', 'Gpu::SyncAtExitOnly region;'),
            ('stream index change', 'Gpu::Device::setStreamIndex(1);'),
            ('external stream', 'Gpu::ExternalGpuStreamRegion region(other);'),
            ('graph capture', 'Gpu::GraphSafeGuard region(true);'),
            ('hidden GPU override', '\n#ifdef AMREX_USE_GPU\nGpu::setNoSyncRegion(true);\n#endif\n'),
            ('hidden CPU override', '\n#ifndef AMREX_USE_GPU\nGpu::setNoSyncRegion(true);\n#endif\n'),
            ('runtime include', '\n#include "unmodeled.H"\n'),
            ('unknown conditional', '\n#ifdef OTHER\n#endif\n'),
            ('goto', 'goto outside;'),
        ):
            yield body_fixture(label+' '+name, path, code+body)

    for path in (TAU, DYCORE):
        body, _ = bodies[path]
        call = unique(pattern('ComputeStrain_S('), body, 'strain call')
        # The call arguments contain no statement semicolon.
        end = body.index(';', call.end())+1
        statement = body[call.start():end]
        yield body_fixture(FUNCTIONS[path]+' deferred strain', path,
                           body[:call.start()]+'auto later = [&] { '+statement+' };'+body[end:])
        yield body_fixture(FUNCTIONS[path]+' strain after iterator destruction', path,
                           body[:call.start()]+body[end:]+statement)
    body, _ = bodies[STRAIN]
    for name, code in (
        ('strain NoSyncRegion', 'Gpu::NoSyncRegion region;'),
        ('strain stream change', 'Gpu::Device::setStreamIndex(1);'),
        ('strain nested iterator', 'MFIter mfi(other);'),
        ('strain conditional', '\n#ifdef OTHER\n#endif\n'),
        ('strain runtime include', '\n#include "unmodeled.H"\n'),
    ):
        yield body_fixture(name, STRAIN, code+body)
    for path in (TAU, FAST):
        body, _ = bodies[path]
        yield body_fixture(FUNCTIONS[path]+' escaped pointer', path, body+'retain(dz_ptr);')
    for name, code in (
        ('macro override', '#define MFIter OtherIter'),
        ('macro undef', '#undef AMREX_USE_GPU'),
        ('line splice', '// hidden \\\n#define MFIter OtherIter'),
        ('raw string', 'R"(hidden)";'),
        ('digraph', '<% unknown(); %>'),
        ('quoted escape', 'Print() << "} // /*"; Gpu::setNoSyncRegion(true);'),
    ):
        body, _ = bodies[FAST]
        yield body_fixture(name, FAST, body+'\n'+code+'\n')

    yield 'comments removed', texts, False
    yield 'CRLF', {path: text.replace('\n', '\r\n') for path, text in original.items()}, False
    def whitespace(text):
        parts = re.split('('+STRING+')', text)
        return ''.join(part if i % 2 else part.replace(',', ', /* gap */ ').replace(';', ';\n')
                       for i, part in enumerate(parts))
    yield 'token whitespace', {path: whitespace(text) for path, text in texts.items()}, False
    for path in FUNCTIONS:
        body, _ = bodies[path]
        for name, code in (
            ('local', 'const int unused = 1;'),
            ('quoted control names', 'Print() << "} /* // '+ ' '.join(CONTROLS)+'";'),
        ):
            yield body_fixture(FUNCTIONS[path]+' '+name, path, code+body, False)
    body, _ = bodies[FAST]
    yield body_fixture('explicit additional synchronization', FAST, 'Gpu::streamSynchronize();'+body, False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-mfiter-mutants-'))
    fixtures = list(cases({path: (ROOT/path).read_text() for path in FILES}))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, sources, fault) in enumerate(fixtures):
        root = work/f'{index:03d}'
        for path, source in sources.items():
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught, expected, _ = mutation_outcome(result, fault)
        required = 'FAIL source contract:' if fault else SUCCESS
        expected = expected and required in result.stdout
        faults += fault
        controls += not fault and name != 'baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
