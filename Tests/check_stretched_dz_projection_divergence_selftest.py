#!/usr/bin/env python3
"""Source fault injection for the projection divergence reader; no C++ compilation."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, no_literals, signature
from check_stretched_dz_consumers import pattern
from check_stretched_dz_projection_divergence import (ALIAS, CALLEE, CLASS, CONTROLS,
    DEFINITION, DISPATCH, FILES, GEOMETRY, LEVELCALL, LOOP, PERIODICITY,
    PROJECTION, READ, SOURCE, STRETCHED, SUBCALL)
from momentum_geometry import call_arguments, unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_projection_divergence.py'
SUCCESS = '4 projection divergence source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path: strip_comments(text) for path, text in original.items()}

    def fixture(name, path, changed, fault=True):
        assert changed != texts[path], name
        return name, {**texts, path: changed}, fault

    def replace(name, path, before, after, fault=True):
        m = unique(pattern(before), texts[path], name)
        return fixture(name, path, texts[path][:m.start()]+after+texts[path][m.end():], fault)

    bodies = {}
    for path, name in ((SOURCE,'ERF::compute_divergence'), (PROJECTION,'ERF::project_momenta')):
        _, body, definition = signature(texts[path], name)
        bodies[path] = (body, definition.end())

    def body_fixture(name, path, changed, fault=True):
        body, offset = bodies[path]
        return fixture(name, path, texts[path][:offset]+changed+texts[path][offset+len(body):], fault)

    for tag, before, after in (
        ('level reference', 'int lev', 'int& lev'),
        ('RHS copied', 'MultiFab& rhs', 'MultiFab rhs'),
        ('momentum pointers referenced', 'rho0_u_const', '&rho0_u_const'),
        ('mutable map factor', 'MultiFab const& mf_mx', 'MultiFab& mf_mx'),
        ('Geometry copied', 'Geometry const& geom_at_lev', 'Geometry geom_at_lev'),
        ('extra parameter', 'int lev', 'int other, int lev'),
    ):
        yield replace('definition '+tag, SOURCE, DEFINITION, DEFINITION.replace(before, after))
    _, _, declaration = signature(texts[CLASS], 'compute_divergence', ';')
    for tag, changed in (
        ('level reference', declaration[0].replace('int lev','int& lev')),
        ('RHS copied', declaration[0].replace('MultiFab& rhs','MultiFab rhs')),
        ('extra parameter', declaration[0].replace('int lev','int other, int lev')),
        ('renamed Geometry', declaration[0].replace('geom_at_lev','other_geom')),
        ('conditional', '#ifdef OTHER\n'+declaration[0]+'\n#endif'),
        ('nested class', 'struct Inner { '+declaration[0]+' };'),
        ('duplicate', declaration[0]*2),
        ('outside ERF', ''),
    ):
        changed_text = texts[CLASS][:declaration.start()]+changed+texts[CLASS][declaration.end():]
        if tag == 'outside ERF':
            changed_text += declaration[0]
        yield fixture('declaration '+tag, CLASS, changed_text)
    yield fixture('definition nested in namespace', SOURCE,
                  'namespace Other {\n'+texts[SOURCE]+'\n}')
    yield fixture('extra definition suffix', SOURCE, texts[SOURCE]+'\nretain(stretched_dz_d);')

    for tag, before, after in (
        ('EB gate reversed', 'terrain_type == TerrainType::EB', 'terrain_type != TerrainType::EB'),
        ('constant gate reversed', 'mesh_type == MeshType::ConstantDz', 'mesh_type != MeshType::ConstantDz'),
        ('EB centroid flag', 'already_on_centroids = true', 'already_on_centroids = false'),
        ('EB wrong RHS', 'EB_computeDivergence(rhs,', 'EB_computeDivergence(other,'),
        ('constant wrong Geometry', 'computeDivergence(rhs, rho0_u_const, geom_at_lev)',
         'computeDivergence(rhs, rho0_u_const, other_geom)'),
    ):
        yield replace('dispatch '+tag, SOURCE, DISPATCH, DISPATCH.replace(before, after))
    for tag, changed in (
        ('wrong branch', STRETCHED.replace('StretchedDz','VariableDz')),
        ('reversed branch', STRETCHED.replace('==','!=')),
    ):
        yield replace(tag, SOURCE, STRETCHED, changed)

    body, _ = bodies[SOURCE]
    loop = unique(pattern(LOOP), body, 'loop')
    loop_end = loop.end()+len(brace_block(body, loop.end()-1))+1
    for tag, changed in (
        ('wrong iterator data', LOOP.replace('mfi(rhs,','mfi(other,')),
        ('no-sync iterator', LOOP.replace('TilingIfNotGPU()', 'MFItInfo().DisableDeviceSync()')),
        ('heap iterator', LOOP.replace('MFIter mfi', 'auto* mfi = new MFIter')),
        ('wrong increment', LOOP.replace('++mfi', 'mfi++')),
        ('extra iterator parameter', LOOP.replace('TilingIfNotGPU()', 'TilingIfNotGPU(), other')),
    ):
        yield replace(tag, SOURCE, LOOP, changed)
    for tag, changed in (
        ('deferred iterator', 'auto later = [&] { '+body[loop.start():loop_end]+' };'),
        ('conditional iterator', 'if (other) { '+body[loop.start():loop_end]+' }'),
        ('unbraced conditional iterator', 'if (other) '+body[loop.start():loop_end]),
        ('duplicated iterator', body[loop.start():loop_end]*2),
    ):
        yield body_fixture(tag, SOURCE, body[:loop.start()]+changed+body[loop_end:])
    for tag, code in (
        ('iterator alias', 'auto& alias = mfi;'),
        ('iterator escape', 'retain(mfi);'),
        ('early Finalize', 'mfi.Finalize();'),
        ('extra increment', '++mfi;'),
    ):
        yield body_fixture(tag, SOURCE, body[:loop.end()]+code+body[loop.end():])
    for tag, changed in (
        ('host spacing source', ALIAS.replace('stretched_dz_d[', 'stretched_dz_h[')),
        ('wrong level', ALIAS.replace('[lev]', '[lev+1]')),
        ('shifted pointer', ALIAS.replace('.data()', '.data()+1')),
        ('pointer reference', ALIAS.replace('Real* ', 'Real*& ')),
        ('deferred pointer binding', 'auto later = [&] { '+ALIAS+' };'),
        ('conditional pointer binding', 'if (other) { '+ALIAS+' }'),
        ('duplicate pointer binding', ALIAS+ALIAS),
    ):
        yield replace(tag, SOURCE, ALIAS, changed)
    for tag, changed in (
        ('wrong cell index', READ.replace('[k]','[k+1]')),
        ('negative index', READ.replace('[k]','[-1]')),
        ('different inverse', READ.replace('one/','two/')),
        ('mutable scalar alias', 'Real& inv_dz = stretched_dz_d_ptr[k];'),
        ('spacing write', 'stretched_dz_d_ptr[k] = one;'),
        ('missing read', ''),
        ('duplicated read', READ+READ),
    ):
        yield replace(tag, SOURCE, READ, changed)
    alias = unique(pattern(ALIAS), body, 'alias')
    read = unique(pattern(READ), body, 'read')
    yield body_fixture('scalar read on host', SOURCE,
        body[:alias.end()]+READ+body[alias.end():read.start()]+body[read.end():])
    for tag, code in (
        ('pointer alias', 'auto* other = stretched_dz_d_ptr;'),
        ('pointer escape', 'retain(stretched_dz_d_ptr);'),
        ('pointer rebind', 'stretched_dz_d_ptr = nullptr;'),
        ('member resize', 'stretched_dz_d[lev].resize(0);'),
        ('member alias', 'auto& other = stretched_dz_d;'),
        ('this escape', 'retain(this);'),
        ('level reassignment', 'lev = 0;'),
        ('quoted then real escape', 'Print() << "} // /*"; retain(stretched_dz_d_ptr);'),
    ):
        yield body_fixture(tag, SOURCE, body[:alias.end()]+code+body[alias.end():])

    matches = list(re.finditer(CALLEE+r'\s*\(', no_literals(body)))
    arguments = list(call_arguments(no_literals(body), CALLEE))
    assert len(matches) == 2
    for index, (launch, args) in enumerate(zip(matches,arguments)):
        end = body.index(';',args[-1][1])+1
        statement = body[launch.start():end]
        for tag, changed in (
            ('deferred launch', 'auto later = [&] { '+statement+' };'),
            ('missing launch', ''),
            ('duplicate launch', statement*2),
            ('outside iterator', ''),
            ('reference capture', statement.replace('[=]', '[&]',1)),
            ('host lambda', statement.replace('AMREX_GPU_DEVICE','',1)),
            ('wrong box', statement.replace('(bx,','(other_box,',1)),
            ('wrapper around kernel', statement[:statement.index('[=]')]+'wrap('+statement[statement.index('[=]'):].replace('});','}));')),
        ):
            changed_body = body[:launch.start()]+changed+body[end:]
            if tag == 'outside iterator':
                changed_body += statement
            yield body_fixture(f'launch {index} '+tag, SOURCE, changed_body)
    yield replace('deferred spacing read', SOURCE, READ, 'auto later = [=] { '+READ+' };')
    yield replace('duplicate nested spacing read', SOURCE, READ, READ+' auto later = [=] { '+READ+' };')
    yield body_fixture('extra launcher alias', SOURCE, body+'retain(ParallelFor);')
    yield body_fixture('extra delegated call', SOURCE, body+'computeDivergence(rhs, rho0_u_const, geom_at_lev);')

    body, _ = bodies[PROJECTION]
    call_matches = sorted([*re.finditer(pattern(SUBCALL), body), *re.finditer(pattern(LEVELCALL), body)], key=lambda m:m.start())
    assert len(call_matches) == 3
    for index, call in enumerate(call_matches):
        args = list(call_arguments(call[0], r'\bcompute_divergence'))[0]
        for argument in range(3, 10):
            lo, hi = args[argument]
            changed = call[0][:lo]+'other_field'+call[0][hi:]
            yield body_fixture(f'projection site {index} wrong field {argument}', PROJECTION,
                               body[:call.start()]+changed+body[call.end():])
        for tag, changed in (
            ('wrong level', call[0].replace('(lev,', '(lev+1,',1)),
            ('wrong RHS', re.sub(r',\s*(?:rhs_sub\[0\]|rhs_lev)\s*,', ', other_rhs,',call[0],count=1)),
            ('wrong momenta', call[0].replace('rho0_u_const','other_mom')),
            ('wrong Geometry', call[0].replace('geom_tmp[0]','geom[lev+1]')),
            ('missing call', ''),
            ('duplicate call', call[0]*2),
            ('deferred call', 'auto later = [&] { '+call[0]+' };'),
            ('conditional call', 'if (other) { '+call[0]+' }'),
            ('unbraced conditional call', 'if (other) '+call[0]),
            ('nested MFIter call', 'for (MFIter mfi(other); mfi.isValid(); ++mfi) { '+call[0]+' }'),
            ('FFT-only call', '\n#ifdef ERF_USE_FFT\n'+call[0]+'\n#endif\n'),
            ('no-FFT-only call', '\n#ifndef ERF_USE_FFT\n'+call[0]+'\n#endif\n'),
        ):
            yield body_fixture(f'projection site {index} '+tag, PROJECTION,
                               body[:call.start()]+changed+body[call.end():])
    for index, statement in enumerate(PERIODICITY):
        yield replace(f'periodicity {index} escape', PROJECTION, statement,
                      'retain(geom_tmp[0].periodicity());')
        yield replace(f'periodicity {index} deferred use', PROJECTION, statement,
                      'auto later = [&] { '+statement+' };')
    for tag, changed in (
        ('wrong Geometry level', GEOMETRY.replace('geom[lev]','geom[lev+1]')),
        ('Geometry reference', GEOMETRY.replace('Vector<Geometry> geom_tmp','Vector<Geometry>& geom_tmp')),
        ('conditional Geometry', 'if (other) { '+GEOMETRY+' }'),
        ('deferred Geometry', 'auto later = [&] { '+GEOMETRY+' };'),
        ('duplicate Geometry', GEOMETRY+GEOMETRY),
    ):
        yield replace(tag, PROJECTION, GEOMETRY, changed)
    for tag, code in (
        ('Geometry alias', 'auto& other = geom_tmp;'),
        ('Geometry mutation', 'geom_tmp[0] = other;'),
        ('Geometry resize', 'geom_tmp.resize(2);'),
        ('extra divergence call', SUBCALL),
        ('divergence callee escape', 'retain(compute_divergence);'),
        ('explicit member spacing mutation', 'stretched_dz_d[lev].clear();'),
        ('hidden debug reader', '\n#if 0\n'+SUBCALL+'\n#endif\n'),
    ):
        yield body_fixture(tag, PROJECTION, body+code)
    geometry = unique(pattern(GEOMETRY), body, 'geometry')
    yield body_fixture('Geometry after first call', PROJECTION,
        body[:geometry.start()]+body[geometry.end():call_matches[0].end()]+GEOMETRY+body[call_matches[0].end():])
    yield replace('projection level reference', PROJECTION, 'void ERF::project_momenta(int lev,',
                  'void ERF::project_momenta(int& lev,')
    for tag, code in (
        ('unsupported macro conditional', '#ifdef OTHER\n#endif'),
        ('runtime include', '#include "runtime.H"'),
        ('debug block enabled', '#if 1\n#endif'),
        ('malformed conditional', '#else\n#endif'),
    ):
        yield body_fixture(tag, PROJECTION, '\n'+code+'\n'+body)
    for path in FILES:
        for tag, code in (
            ('stream override', 'Gpu::Device::setStreamIndex(1);'),
            ('NoSyncRegion', 'Gpu::NoSyncRegion region;'),
            ('macro override', '#define stretched_dz_d other'),
            ('raw string', 'R"(hidden)";'),
            ('line splice', '// hidden \\\n#define lev other'),
            ('nonlocal transfer', 'goto outside;'),
        ):
            yield fixture(Path(path).stem+' '+tag,path,texts[path]+'\n'+code+'\n')
    yield body_fixture('divergence runtime include', SOURCE, bodies[SOURCE][0]+'\n#include "runtime.H"\n')

    yield 'comments removed', texts, False
    yield 'CRLF', {path:text.replace('\n','\r\n') for path,text in original.items()},False
    def whitespace(text):
        parts=re.split('('+STRING+')',text)
        return ''.join(part if i%2 else part.replace(',',', /* gap */ ').replace(';',';\n')
                       for i,part in enumerate(parts))
    yield 'token whitespace',{path:whitespace(text) for path,text in texts.items()},False
    for path,(body,_) in bodies.items():
        for tag,code in (
            ('harmless local','const int unused = 1;'),
            ('quoted controls','Print() << "} // /* '+' '.join(CONTROLS)+' stretched_dz_d_ptr compute_divergence";'),
            ('additional synchronization','Gpu::streamSynchronize();'),
        ):
            yield body_fixture(Path(path).stem+' '+tag,path,code+body,False)
    body,_=bodies[SOURCE]
    yield body_fixture('loop additional synchronization',SOURCE,body[:loop.end()]+'Gpu::streamSynchronize();'+body[loop.end():],False)
    yield replace('read comments',SOURCE,READ,'Real inv_dz = one /* scalar */ / stretched_dz_d_ptr[k];',False)
    yield replace('optional kernel noexcept',SOURCE,'ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept { '+READ,
                  'ParallelFor(bx, [=] AMREX_GPU_DEVICE (int i, int j, int k) { '+READ,False)


def main():
    work=Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-projection-divergence-mutants-'))
    fixtures=list(cases({path:(ROOT/path).read_text() for path in FILES}))
    faults=controls=unexpected=0
    print('Fixture directory: '+str(work),flush=True)
    for index,(name,sources,fault) in enumerate(fixtures):
        root=work/f'{index:03d}'
        for path,source in sources.items():
            target=root/path
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text(source)
        result=subprocess.run([sys.executable,str(VERIFY),str(root)],capture_output=True,text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught,expected,_=mutation_outcome(result,fault)
        expected=expected and ('FAIL source contract:' if fault else SUCCESS) in result.stdout
        faults+=fault
        controls+=not fault and name!='baseline'
        unexpected+=not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}",flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__=='__main__':
    sys.exit(main())
