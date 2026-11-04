#!/usr/bin/env python3
"""Fault injection for the slow-RHS callback completion source contracts."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, signature
from check_stretched_dz_callback_completion import (CALLBACKS, CONTROLS, CPRE,
    CPOST, DYCORE, FILES, FRAGMENTS, HEADERS, IPRE, IPOST, LIMIT, MACROS, MRI,
    REGISTRATION, SLOW_H)
from check_stretched_dz_consumers import pattern
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_callback_completion.py'
SUCCESS = '4 callback completion source contracts passed;'


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
    for path in CALLBACKS:
        m = unique(pattern(HEADERS[path]), texts[path], 'callback header')
        bodies[path] = brace_block(texts[path], m.end()-1), m.end()
    _, body, definition = signature(texts[DYCORE], 'ERF::advance_dycore')
    bodies[DYCORE] = body, definition.end()
    advance = unique(pattern('double advance(T& S_old, T& S_new, double time, const double time_step) {'),
                     texts[MRI], 'MRI advance')
    bodies[MRI] = brace_block(texts[MRI], advance.end()-1), advance.end()

    def body_fixture(name, path, changed, fault=True):
        body, offset = bodies[path]
        return fixture(name, path, texts[path][:offset]+changed+texts[path][offset+len(body):], fault)

    for path, (kind, _, implicit) in CALLBACKS.items():
        body, _ = bodies[path]
        label = kind+' callback'
        yield replace(label+' value capture', path, HEADERS[path], HEADERS[path].replace('[&', '[='))
        yield replace(label+' wrong parameter type', path, HEADERS[path],
                      HEADERS[path].replace('const int nrk', 'const int& nrk'))
        name = 'erf_slow_rhs_'+kind
        m = unique(pattern(name+'('), body, 'RHS')
        end = body.index(';', m.end())+1
        call = body[m.start():end]
        for tag, changed in (
            ('deferred RHS', 'auto later = [&] { '+call+' };'),
            ('conditional RHS', 'if (other) { '+call+' }'),
            ('missing RHS', ''),
            ('duplicate RHS', call+call),
            ('wrong spacing level', call.replace('stretched_dz_d[level]', 'stretched_dz_d[level+1]')),
            ('aliased spacing', call.replace('stretched_dz_d[level]', 'other_spacing')),
            ('wrong geometry', call.replace('fine_geom', 'other_geom')),
        ):
            yield body_fixture(label+' '+tag, path, body[:m.start()]+changed+body[end:])
        include = '#include "'+Path(implicit).name+'"'
        im = unique(re.escape(include), body, 'implicit include')
        for tag, changed in (
            ('missing implicit include', ''),
            ('duplicate implicit include', include+'\n'+include),
            ('deferred implicit include', 'auto later = [&] {\n'+include+'\n};'),
            ('conditional implicit include', 'if (other) {\n'+include+'\n}'),
            ('unknown implicit include', '#include "unmodeled.H"'),
        ):
            yield body_fixture(label+' '+tag, path, body[:im.start()]+changed+body[im.end():])
        yield body_fixture(label+' RHS after implicit include', path,
                           body[:m.start()]+body[end:im.end()]+'\n'+call+body[im.end():])
        yield body_fixture(label+' extra device spacing alias', path, body+'\nauto& alias = stretched_dz_d[level];')
        yield body_fixture(label+' callback escape', path, body+'\nretain(slow_rhs_fun_'+kind+');')
        for macro in MACROS:
            yield body_fixture(label+' '+macro+' hidden deferred include', path,
                body[:im.start()]+'\n#ifdef '+macro+'\nauto later = [&] {\n#endif\n'+include+
                '\n#ifdef '+macro+'\n};\n#endif\n'+body[im.end():])
            yield body_fixture(label+' '+macro+' hidden override', path,
                               '\n#ifdef '+macro+'\nGpu::setNoSyncRegion(true);\n#endif\n'+body)
        for tag, code in (
            ('unknown conditional', '#ifdef OTHER\n#endif'),
            ('unsupported else', '#ifdef ERF_USE_NETCDF\n#else\n#endif'),
            ('runtime include', '#include "unmodeled.H"'),
            ('goto', 'goto outside;'),
        ):
            yield body_fixture(label+' '+tag, path, '\n'+code+'\n'+body)

    yield replace('pre implicit gate changed', CPRE, 'solverChoice.implicit_before_substep', 'other_gate')
    yield replace('post limiter deferred', CPOST, '#include "ERF_ScalarLimitPost.H"',
                  'auto later = [&] {\n#include "ERF_ScalarLimitPost.H"\n};')
    yield replace('post limiter before implicit', CPOST, '#include "ERF_ImplicitPost.H"',
                  '#include "ERF_ScalarLimitPost.H"\n#include "ERF_ImplicitPost.H"')

    body, _ = bodies[MRI]
    for kind, count in (('pre', 3), ('post', 4)):
        name = 'slow_rhs_'+kind
        typ = 'std::function<void('+', '.join(['T&']*count+['const double']*3+['const int'])+')>'
        yield replace(name+' stored by reference', MRI, typ+' '+name+';', typ+'& '+name+';')
        yield replace(name+' setter retains reference', MRI, name+' = F;', name+' = std::ref(F);')
        yield replace(name+' setter deferred', MRI, name+' = F;', 'auto later = [&] { '+name+' = F; };')
        statement = name+'(*F_slow, S_old, S_new, '+('*S_sum, ' if kind=='post' else '')+'time, old_time_stage, time_stage, nrk);'
        matches = list(re.finditer(pattern(statement), body))
        assert len(matches)==2
        for index, m in enumerate(matches):
            for tag, changed in (
                ('deferred call', 'auto later = [&] { '+m[0]+' };'),
                ('conditional call', 'if (other) { '+m[0]+' }'),
                ('missing call', ''),
                ('duplicate call', m[0]+m[0]),
                ('wrong stage', m[0].replace('nrk', 'nrk+1')),
                ('outside stage loop', ''),
            ):
                modified = body[:m.start()]+changed+body[m.end():]
                if tag=='outside stage loop':
                    modified += m[0]
                yield body_fixture(name+f' site {index} '+tag, MRI, modified)
        yield body_fixture(name+' field escape', MRI, body+'retain('+name+');')
    pre = list(re.finditer(pattern('slow_rhs_pre('), body))
    post = list(re.finditer(pattern('slow_rhs_post('), body))
    for index, (a,b) in enumerate(zip(pre,post)):
        ae,be = body.index(';',a.end())+1,body.index(';',b.end())+1
        yield body_fixture(f'MRI stage {index} post before pre', MRI,
                           body[:a.start()]+body[b.start():be]+body[ae:b.start()]+body[a.start():ae]+body[be:])
    yield body_fixture('MRI additional iterator', MRI, 'MFIter outer(other);'+body)
    yield body_fixture('MRI runtime include', MRI, '\n#include "unmodeled.H"\n'+body)

    body, _ = bodies[DYCORE]
    m = unique(pattern(REGISTRATION), body, 'registration sequence')
    for tag, changed in (
        ('deferred advance sequence', 'auto later = [&] {\n'+m[0]+'\n};'),
        ('conditional advance sequence', 'if (other) {\n'+m[0]+'\n}'),
        ('wrong integrator level', m[0].replace('mri_integrator_mem[level]', 'mri_integrator_mem[level+1]')),
        ('wrong post callback', m[0].replace('set_slow_rhs_post(slow_rhs_fun_post)', 'set_slow_rhs_post(other_callback)')),
        ('missing post callback include', m[0].replace('#include "ERF_TI_slow_rhs_post.H"', '')),
    ):
        yield body_fixture(tag, DYCORE, body[:m.start()]+changed+body[m.end():])
    for name, code in (
        ('dycore callback escape', 'retain(slow_rhs_fun_post);'),
        ('dycore extra invocation', 'mri_integrator.advance(state_old,state_new,old_time,dt_advance);'),
    ):
        yield body_fixture(name, DYCORE, body+code)

    for path in FILES:
        for name, code in (
            ('NoSyncRegion', 'Gpu::NoSyncRegion region;'),
            ('stream override', 'Gpu::Device::setStreamIndex(2);'),
        ):
            yield fixture(Path(path).stem+' '+name, path, code+'\n'+texts[path])
    for path in FRAGMENTS:
        yield fixture(Path(path).stem+' unknown runtime include', path, texts[path]+'\n#include "unmodeled.H"\n')
    yield fixture('auxiliary callback shadow', FRAGMENTS[0], texts[FRAGMENTS[0]]+'\nauto slow_rhs_fun_post = other;')
    for name, code in (
        ('macro override', '#define slow_rhs_post other'),
        ('macro undef', '#undef AMREX_USE_GPU'),
        ('line splice', '// hidden \\\n#define slow_rhs_post other'),
        ('raw string', 'R"(hidden)";'),
        ('digraph', '<% unknown(); %>'),
        ('quoted escape', 'Print() << "} // /*"; Gpu::setNoSyncRegion(true);'),
    ):
        yield body_fixture(name, CPRE, '\n'+code+'\n'+bodies[CPRE][0])
    yield fixture('OpenMP without GPU guard', CPRE,
                  texts[CPRE].replace('omp parallel if (amrex::Gpu::notInLaunchRegion())', 'omp parallel'))

    yield 'comments removed', texts, False
    yield 'CRLF', {path: text.replace('\n','\r\n') for path,text in original.items()}, False
    def whitespace(text):
        parts = re.split('('+STRING+')',text)
        return ''.join(part if i%2 else part.replace(',', ', /* gap */ ').replace(';',';\n')
                       for i,part in enumerate(parts))
    yield 'token whitespace', {path: whitespace(text) for path,text in texts.items()}, False
    for path in bodies:
        for name, code in (
            ('local', 'const int unused = 1;'),
            ('quoted controls', 'Print() << "} /* // '+' '.join(CONTROLS)+'";'),
            ('additional synchronization', 'Gpu::streamSynchronize();'),
        ):
            yield body_fixture(Path(path).stem+' '+name, path, code+bodies[path][0], False)
    for path in CALLBACKS:
        yield body_fixture(Path(path).stem+' harmless supported conditional', path,
                           '\n#ifdef ERF_USE_NETCDF\nconst int unused = 1;\n#endif\n'+bodies[path][0], False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-callback-completion-mutants-'))
    fixtures = list(cases({path: (ROOT/path).read_text() for path in FILES}))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index,(name,sources,fault) in enumerate(fixtures):
        root = work/f'{index:03d}'
        for path,source in sources.items():
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(source)
        result = subprocess.run([sys.executable,str(VERIFY),str(root)],capture_output=True,text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught,expected,_ = mutation_outcome(result,fault)
        expected = expected and ('FAIL source contract:' if fault else SUCCESS) in result.stdout
        faults += fault
        controls += not fault and name!='baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}",flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
