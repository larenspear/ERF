#!/usr/bin/env python3
"""Fault injection for slow-RHS and implicit MFIter completion premises."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, signature
from check_stretched_dz_consumers import pattern
from check_stretched_dz_solver_mfiter import (CALLEES, CALLS, CONTROLS, FILES,
    FUNCTIONS, IMPLICIT, IPRE, IPOST, LOOPS, MOM, POST, PRE, header)
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_solver_mfiter.py'
SUCCESS = '5 solver MFIter source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path: strip_comments(text) for path, text in original.items()}

    def fixture(name, path, text, fault=True):
        assert text != texts[path], name
        return name, {**texts, path: text}, fault

    def replace(name, path, before, after, fault=True):
        m = unique(pattern(before), texts[path], name)
        return fixture(name, path, texts[path][:m.start()]+after+texts[path][m.end():], fault)

    bodies = {}
    for path in LOOPS:
        if path in FUNCTIONS:
            _, body, definition = signature(texts[path], FUNCTIONS[path])
            bodies[path] = body, definition.end()
        else:
            bodies[path] = texts[path], 0

    def body_fixture(name, path, changed, fault=True):
        body, offset = bodies[path]
        return fixture(name, path, texts[path][:offset]+changed+texts[path][offset+len(body):], fault)

    for path, (container, tiling, count) in LOOPS.items():
        body, _ = bodies[path]
        tag = Path(path).stem
        matches = list(re.finditer(pattern(header(container, tiling)), body))
        assert len(matches) == count
        for index, m in enumerate(matches):
            end = m.end()+len(brace_block(body, m.end()-1))+1
            loop = body[m.start():end]
            label = tag+f' loop {index}'
            for name, changed in (
                ('wrong iterator type', m[0].replace('MFIter', 'OtherIter')),
                ('wrong constructor', m[0].replace(tiling, 'other_info')),
                ('missing increment', m[0].replace('++mfi', '')),
                ('changed condition', m[0].replace('mfi.isValid()', 'other_valid()')),
                ('disable post sync', m[0].replace(tiling, 'MFItInfo().DisableDeviceSyncPost()')),
            ):
                yield body_fixture(label+' '+name, path, body[:m.start()]+changed+body[m.end():])
            for name, code in (
                ('early finalize', 'mfi.Finalize();'),
                ('iterator alias', 'auto& saved = mfi;'),
                ('iterator escape', 'retain(&mfi);'),
                ('iterator move', 'auto moved = std::move(mfi);'),
            ):
                yield body_fixture(label+' '+name, path, body[:m.end()]+code+body[m.end():])
            yield body_fixture(label+' deferred loop', path,
                               body[:m.start()]+'auto later = [&] { '+loop+' };'+body[end:])

        specs = CALLS[path]+((('ImplicitDiffForMomLU_S<2>', 1),) if path == IPRE else ())
        for callee, count in specs:
            matches = list(re.finditer(pattern(callee+'('), body))
            assert len(matches) == count
            for index, m in enumerate(matches):
                end = body.index(';', m.end())+1
                statement = body[m.start():end]
                label = tag+' '+callee+f' call {index}'
                for name, changed in (
                    ('deferred call', body[:m.start()]+'auto later = [&] { '+statement+' };'+body[end:]),
                    ('after iterator destruction', body[:m.start()]+body[end:]+statement),
                    ('missing call', body[:m.start()]+body[end:]),
                    ('callee escape', body[:m.start()]+'retain('+callee.split('<')[0]+');'+body[m.start():]),
                ):
                    yield body_fixture(label+' '+name, path, changed)

        for name, code in (
            ('NoSyncRegion', 'Gpu::NoSyncRegion region;'),
            ('stream change', 'Gpu::Device::setStreamIndex(3);'),
            ('graph capture', 'Gpu::GraphSafeGuard region(true);'),
            ('runtime include', '\n#include "unmodeled.H"\n'),
            ('unknown conditional', '\n#ifdef OTHER\n#endif\n'),
            ('extra iterator', 'MFIter extra(other);'),
            ('goto', 'goto outside;'),
            ('iterator use after exit', ''),
        ):
            changed = body+'retain(mfi);' if not code else code+body
            yield body_fixture(tag+' '+name, path, changed)

    for path in CALLEES:
        for name, code in (
            ('NoSyncRegion', 'Gpu::NoSyncRegion region;'),
            ('stream change', 'Gpu::Device::setStreamIndex(3);'),
            ('extra iterator', 'MFIter extra(other);'),
        ):
            # Prefix keeps the exact implicit instantiation macro at file end.
            yield fixture(Path(path).stem+' '+name, path, code+'\n'+texts[path])
    yield replace('router mutable iterator', MOM, 'const MFIter& mfi,', 'MFIter& mfi,')
    yield replace('router iterator alias', MOM, 'AdvectionSrcForMom_EB(mfi,',
                  'auto& saved = mfi; AdvectionSrcForMom_EB(mfi,')
    yield replace('router iterator escape', MOM, 'AdvectionSrcForMom_EB(mfi,',
                  'retain(&mfi); AdvectionSrcForMom_EB(mfi,')
    router = texts[MOM]
    m = unique(pattern('AdvectionSrcForMom_EB(mfi,'), router, 'router forwarding')
    end = router.index(';', m.end())+1
    yield fixture('router forwarding inside closure', MOM,
                  router[:m.start()]+'auto later = [&] { '+router[m.start():end]+' };'+router[end:])

    for path, macro in ((PRE, 'ERF_USE_EAMXX_SHOC'), (POST, 'ERF_USE_EAMXX_SHOC'),
                        (IPRE, 'ERF_IMPLICIT_W'), (IPOST, 'ERF_IMPLICIT_W')):
        body, _ = bodies[path]
        for name, code in (
            ('hidden sync override', 'Gpu::setNoSyncRegion(true);'),
            ('hidden iterator alias', 'auto& saved = mfi;'),
            ('hidden stream override', 'Gpu::Device::setStreamIndex(1);'),
        ):
            yield body_fixture(Path(path).stem+' '+name, path,
                '\n#ifdef '+macro+'\n'+code+'\n#endif\n'+body)
        yield body_fixture(Path(path).stem+' unsupported else', path,
            '\n#ifdef '+macro+'\n#else\n#endif\n'+body)

    body, _ = bodies[PRE]
    for name, code in (
        ('macro override', '#define MFIter OtherIter'),
        ('macro undef', '#undef AMREX_USE_GPU'),
        ('line splice', '// hidden \\\n#define MFIter OtherIter'),
        ('raw string', 'R"(hidden)";'),
        ('digraph', '<% unknown(); %>'),
        ('quoted escape', 'Print() << "} // /*"; Gpu::setNoSyncRegion(true);'),
    ):
        yield body_fixture(name, PRE, body+'\n'+code+'\n')
    yield fixture('OpenMP without GPU guard', PRE,
                  texts[PRE].replace('omp parallel if (Gpu::notInLaunchRegion())', 'omp parallel', 1))
    body, _ = bodies[PRE]
    tau = unique(pattern('erf_make_tau_terms('), body, 'tau')
    end = body.index(';', tau.end())+1
    statement = body[tau.start():end]
    yield body_fixture('deferred tau call', PRE,
                       body[:tau.start()]+'auto later = [&] { '+statement+' };'+body[end:])

    yield 'comments removed', texts, False
    yield 'CRLF', {path: text.replace('\n', '\r\n') for path, text in original.items()}, False
    def whitespace(text):
        parts = re.split('('+STRING+')', text)
        return ''.join(part if i % 2 else part.replace(',', ', /* gap */ ').replace(';', ';\n')
                       for i, part in enumerate(parts))
    # The implicit instantiation macro is a separately fixed logical-line contract.
    yield 'token whitespace', {path: text if path == IMPLICIT else whitespace(text)
                              for path, text in texts.items()}, False
    for path in LOOPS:
        body, _ = bodies[path]
        for name, code in (
            ('local', 'const int unused = 1;'),
            ('quoted control names', 'Print() << "} /* // '+' '.join(CONTROLS)+'";'),
            ('additional synchronization', 'Gpu::streamSynchronize();'),
        ):
            yield body_fixture(Path(path).stem+' '+name, path, code+body, False)
    body, _ = bodies[POST]
    m = unique(pattern('DiffusionSrcForState_S('), body, 'post diffusion')
    end = body.index(';', m.end())+1
    yield body_fixture('extra synchronous block', POST,
                       body[:m.start()]+'{ '+body[m.start():end]+' }'+body[end:], False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-solver-mfiter-mutants-'))
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
        expected = expected and ('FAIL source contract:' if fault else SUCCESS) in result.stdout
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
