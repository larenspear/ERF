#!/usr/bin/env python3
"""Source fault injection for temporary descriptor tables; fixtures are not compiled."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_async_vectors import (CALLBACK, COUNT, ENUM, FILES, FORWARD,
    INDEX, NAMES, OWNERS, SETUP, kernel_spans)
from check_stretched_dz_bindings import STRING, brace_block
from check_stretched_dz_consumers import pattern
from check_stretched_dz_no_substep_completion import HEADER, LOOPS
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_async_vectors.py'
SUCCESS = '3 AsyncVector source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path: strip_comments(text) for path, text in original.items()}

    def fixture(name, path, changed, fault=True):
        assert changed != texts[path], name
        return name, {**texts, path: changed}, fault

    def replace(name, path, before, after, fault=True):
        m = unique(pattern(before), texts[path], name)
        return fixture(name, path, texts[path][:m.start()]+after+texts[path][m.end():], fault)

    header = unique(pattern(HEADER), texts[CALLBACK], 'callback header')
    body = brace_block(texts[CALLBACK], header.end()-1)

    def changed_body(name, changed, fault=True):
        return fixture(name, CALLBACK, texts[CALLBACK][:header.end()]+changed+
                       texts[CALLBACK][header.end()+len(body):], fault)

    for tag, before, after in (
        ('nonzero first slot', 'cons = 0', 'cons = 1'),
        ('negative slot', 'cons = 0', 'cons = -1'),
        ('duplicate slot', 'xmom,', 'xmom = 0,'),
        ('large last slot', 'zmom,', 'zmom = 7,'),
        ('small count', 'NumTypes', 'NumTypes = 3'),
        ('large count', 'NumTypes', 'NumTypes = 5'),
        ('extra slot', 'NumTypes', 'extra, NumTypes'),
        ('renamed namespace', 'IntVars', 'OtherVars'),
    ):
        yield replace('enum '+tag, INDEX, ENUM, ENUM.replace(before, after))
    for tag, wrapper in (
        ('conditional enum', '#ifdef OTHER\n'+ENUM+'\n#endif'),
        ('nested enum', 'namespace Other { '+ENUM+' }'),
        ('duplicate enum', ENUM+ENUM),
    ):
        yield replace(tag, INDEX, ENUM, wrapper)
    for tag, after in (
        ('wrong count', 'int n_data = 3;'),
        ('count reference', 'int& n_data = IntVars::NumTypes;'),
        ('duplicate count', COUNT+COUNT),
        ('deferred count', 'auto later = [&] { '+COUNT+' };'),
        ('conditional count', 'if (other) { '+COUNT+' }'),
        ('unbraced count', 'if (other) '+COUNT),
    ):
        yield replace(tag, CALLBACK, COUNT, after)
    yield changed_body('count mutation', body+'n_data = 3;')
    yield changed_body('count escape', body+'retain(n_data);')

    setups = list(re.finditer(pattern(SETUP), body))
    assert len(setups) == 2
    for index, setup in enumerate(setups):
        def setup_fixture(tag, changed):
            return changed_body(f'loop {index} '+tag,
                                body[:setup.start()]+changed+body[setup.end():])

        def setup_replace(tag, before, after):
            m = unique(pattern(before), setup[0], tag)
            return setup_fixture(tag, setup[0][:m.start()]+after+setup[0][m.end():])

        for name, state in (('sold', 'S_old'), ('ssum', 'S_sum'), ('fslow', 'F_slow')):
            yield setup_replace(name+' short host table', name+'_h(n_data)', name+'_h(n_data-1)')
            yield setup_replace(name+' wrong population source',
                name+'_h[i] = '+state+'[i].array(mfi);', name+'_h[i] = Other[i].array(mfi);')
            yield setup_replace(name+' wrong population slot', name+'_h[i]', name+'_h[0]')
            yield setup_replace(name+' wrong input slot', state+'[i]', state+'[i+1]')
            yield setup_replace(name+' wrong iterator', state+'[i].array(mfi)', state+'[i].array(other)')
            yield setup_replace(name+' synchronous owner',
                'Gpu::AsyncVector<Array4<Real> > '+name+'_d(n_data);',
                'Gpu::DeviceVector<Array4<Real> > '+name+'_d(n_data);')
            yield setup_replace(name+' short device owner', name+'_d(n_data)', name+'_d(n_data-1)')
            copy = 'Gpu::copy(Gpu::hostToDevice, '+name+'_h.begin(), '+name+'_h.end(), '+name+'_d.begin());'
            for tag, changed in (
                ('async copy', copy.replace('Gpu::copy(', 'Gpu::copyAsync(')),
                ('reversed copy direction', copy.replace('hostToDevice', 'deviceToHost')),
                ('wrong copy source', copy.replace(name+'_h', 'other_h')),
                ('wrong copy destination', copy.replace(name+'_d', 'other_d')),
                ('short copy range', copy.replace('_h.end()', '_h.end()-1')),
                ('empty copy range', copy.replace('_h.end()', '_h.begin()')),
                ('missing copy', ''),
                ('duplicate copy', copy+copy),
            ):
                yield setup_replace(name+' '+tag, copy, changed)
            binding = 'Array4<Real>* '+name+' = '+name+'_d.dataPtr();'
            yield setup_replace(name+' wrong owner binding', binding, binding.replace(name+'_d', 'other_d'))
            yield setup_replace(name+' bind host table', binding, binding.replace(name+'_d', name+'_h'))
            # Move a complete binding before its copy, keeping both exact statements.
            cm = unique(pattern(copy), setup[0], 'copy')
            bm = unique(pattern(binding), setup[0], 'binding')
            yield setup_fixture(name+' pointer bound before copy',
                setup[0][:cm.start()]+bm[0]+setup[0][cm.start():bm.start()]+setup[0][bm.end():])
            for tag, code in (
                ('pointer alias', 'auto* alias = '+name+';'),
                ('pointer escape', 'retain('+name+');'),
                ('pointer reassignment', name+' = nullptr;'),
                ('pointer advance', '++'+name+';'),
                ('host descriptor access', 'auto value = '+name+'[IntVars::cons](0,0,0);'),
                ('descriptor overwrite', name+'[IntVars::cons] = other;'),
            ):
                yield setup_fixture(name+' '+tag, setup[0]+code)
        for name in OWNERS:
            for tag, code in (
                ('alias', 'auto& alias = '+name+';'),
                ('move', 'auto moved = std::move('+name+');'),
                ('resize', name+'.resize(0);'),
                ('escape', 'retain('+name+');'),
            ):
                yield setup_fixture(name+' '+tag, setup[0]+code)
        for tag, before, after in (
            ('population starts late', 'int i = 0', 'int i = 1'),
            ('population overruns', 'i < n_data', 'i <= n_data'),
            ('population truncated', 'i < n_data', 'i < n_data-1'),
            ('population skips slots', '++i', 'i += 2'),
        ):
            yield setup_replace(tag, before, after)
        for tag, changed in (
            ('shortened owner scope', '{ '+setup[0]+' }'),
            ('conditional setup', 'if (other) { '+setup[0]+' }'),
            ('unbraced conditional setup', 'if (other) '+setup[0]),
            ('deferred setup', 'auto later = [&] { '+setup[0]+' };'),
            ('missing setup', ''),
            ('duplicate setup', setup[0]*2),
        ):
            yield setup_fixture(tag, changed)
        loop = unique(pattern(LOOPS[index]), body, 'loop')
        end = loop.end()+len(brace_block(body, loop.end()-1))
        yield changed_body(f'loop {index} setup after kernels',
            body[:setup.start()]+body[setup.end():end]+setup[0]+body[end:])
        yield setup_fixture('explicit owner destructor', setup[0]+'sold_d.~PODVector();')
        yield setup_fixture('stream switch before destruction', setup[0]+'Gpu::Device::setStreamIndex(1);')

    kernels = list(kernel_spans(body))
    assert len(kernels) == 15
    for index, (_, lo, hi) in enumerate(kernels):
        kernel = body[lo:hi]
        m = re.search(r'\b(?:ssum|sold|fslow)\s*\[\s*IntVars::\w+\s*\]', kernel)
        assert m
        for tag, changed in (
            ('descriptor index out of bounds', m[0].split('[')[0]+'[IntVars::NumTypes]'),
            ('descriptor index negative', m[0].split('[')[0]+'[-1]'),
            ('dynamic descriptor index', m[0].split('[')[0]+'[i]'),
            ('slot arithmetic', m[0][:-1]+'+4]'),
        ):
            yield changed_body(f'kernel {index} '+tag,
                body[:lo+m.start()]+changed+body[lo+m.end():])
        for tag, changed in (
            ('deferred pointer access', 'auto later = [=] { '+kernel+' };'),
            ('owner captured', 'auto value = sold_d.dataPtr();'+kernel),
            ('bare pointer escape', 'retain(sold);'+kernel),
            ('bare descriptor escape', 'retain(sold[IntVars::cons]);'+kernel),
            ('descriptor address escape', 'retain(&sold[IntVars::cons]);'+kernel),
            ('descriptor alias', 'auto& alias = sold[IntVars::cons];'+kernel),
        ):
            yield changed_body(f'kernel {index} '+tag, body[:lo]+changed+body[hi:])
    for name in NAMES:
        yield changed_body(name+' use after both iterations', body+'retain('+name+');')
    for tag, changed in (
        ('unknown descriptor callee', FORWARD.replace('WFromOmega', 'retain')),
        ('address passed to approved callee', FORWARD.replace('ssum[IntVars::xmom]', '&ssum[IntVars::xmom]')),
        ('wrong forwarded descriptor', FORWARD.replace('ssum[IntVars::xmom]', 'sold[IntVars::xmom]')),
        ('reversed forwarded descriptors', FORWARD.replace('ssum[IntVars::xmom]', 'ssum[IntVars::zmom]')),
        ('duplicate descriptor call', '('+FORWARD+','+FORWARD+')'),
    ):
        yield replace(tag, CALLBACK, FORWARD, changed)
    yield changed_body('approved callee alias', body+'retain(WFromOmega);')
    for path in FILES:
        # Explicit macro/stream changes in every scanned source are rejected.
        yield fixture(Path(path).stem+' macro rewrite', path, '#define AsyncVector DeviceVector\n'+texts[path])
        if path != INDEX:
            yield fixture(Path(path).stem+' stream override', path, texts[path]+'\nGpu::setNoSyncRegion(true);')
    for name in ('IntVars', 'NumTypes', 'enum', 'n_data', 'sold', 'copy', 'dataPtr',
                 'for', 'i', 'S_old', 'AMREX_GPU_DEVICE', 'AMREX_USE_GPU',
                 'MFIter', 'setStreamIndex'):
        yield fixture('index macro '+name, INDEX, texts[INDEX]+'\n#define '+name+' other\n')
    yield fixture('index guard undef', INDEX, texts[INDEX]+'\n#undef ERF_INDEX_DEFINES_H_\n')
    for path in (CALLBACK, INDEX):
        for tag, code in (
            ('line splice', '// hidden \\\n#define sold other'),
            ('raw string', 'R"(hidden)";'),
            ('digraph', '<% unknown(); %>'),
        ):
            yield fixture(Path(path).stem+' '+tag, path, texts[path]+'\n'+code+'\n')

    yield 'comments removed', texts, False
    yield 'CRLF', {path: text.replace('\n', '\r\n') for path, text in original.items()}, False
    def whitespace(text):
        parts = re.split('('+STRING+')', text)
        return ''.join(part if i%2 else part.replace(',', ', /* gap */ ').replace(';', ';\n')
                       for i, part in enumerate(parts))
    yield 'token whitespace', {path: whitespace(text) for path, text in texts.items()}, False
    for tag, code in (
        ('local', 'const int unrelated = 1;'),
        ('quoted names', 'Print() << "} /* // sold sold_d n_data AsyncVector WFromOmega";'),
        ('additional synchronization', 'Gpu::streamSynchronize();'),
    ):
        yield changed_body(tag, code+body, False)
    for index, setup in enumerate(setups):
        for tag, code in (
            ('additional synchronization', 'Gpu::streamSynchronize();'),
            ('quoted owner destruction', 'Print() << "sold_d.clear(); }";'),
            ('harmless local', 'const int unrelated = 1;'),
        ):
            yield changed_body(f'loop {index} '+tag, body[:setup.end()]+code+body[setup.end():], False)
    yield fixture('unrelated index macro', INDEX, texts[INDEX]+'\n#define UNRELATED_NUMBER 17\n', False)
    _, lo, hi = kernels[-1]
    yield changed_body('kernel quoted names', body[:lo]+'printf("sold_d.resize(0); }");'+body[lo:], False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-async-vector-mutants-'))
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
