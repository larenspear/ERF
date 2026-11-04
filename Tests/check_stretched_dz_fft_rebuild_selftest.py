#!/usr/bin/env python3
"""Source fault injection for FFT rebuild order; no C++ compilation."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, signature
from check_stretched_dz_fft_rebuild import (ARRAYS, BUILD, CALLS, CLASS, FILES,
    GATE, INIT, LEVELS, LOOP, PARAMS, REBUILD, RESTART, SETTERS, START, SUBDOMAINS, pattern)
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_fft_rebuild.py'
SUCCESS = '3 FFT rebuild source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path:strip_comments(text) for path,text in original.items()}
    bodies = {}
    for path,names in ((ARRAYS,('init_stuff',)), (LEVELS,tuple(CALLS)),
                       (INIT,('InitData','InitData_pre','InitData_post','restart'))):
        for name in names:
            _,body,definition = signature(texts[path],'ERF::'+name)
            bodies[name] = (path,body,definition)

    def fixture(name,path,changed,fault=True):
        assert changed != texts[path], name
        return name,{**texts,path:changed},fault

    def body_fixture(name,owner,changed,fault=True):
        path,body,definition = bodies[owner]
        return fixture(name,path,texts[path][:definition.end()]+changed+
                       texts[path][definition.end()+len(body):],fault)

    def replace(name,owner,before,after,fault=True):
        _,body,_ = bodies[owner]
        match=unique(pattern(before),body,name)
        return body_fixture(name,owner,body[:match.start()]+after+body[match.end():],fault)

    for owner in bodies:
        path,body,definition = bodies[owner]
        for name,before,after in (
            ('reference parameter','int lev','int& lev'),
            ('renamed parameter','int lev','int other'),
            ('merged parameter','int lev','intlev'),
        ) if 'int lev' in definition[1] else (('extra parameter','()', '(int unused)'),):
            old=definition[0]
            changed=old.replace(before,after) if before != '()' else re.sub(r'\(\s*\)',after,old)
            assert changed!=old
            yield fixture(owner+' '+name,path,texts[path][:definition.start()]+changed+texts[path][definition.end():])
        end=definition.end()+len(body)+1
        for name,opening,closing in (
            ('conditional definition','#ifdef ERF_USE_NETCDF\n','\n#endif\n'),
            ('nested definition','namespace Other {\n','\n}\n'),
        ):
            yield fixture(owner+' '+name,path,texts[path][:definition.start()]+opening+
                          texts[path][definition.start():end]+closing+texts[path][end:])
        for name,code in (
            ('early return','return;'),('early conditional return','if (stop) { return; }'),
            ('goto','goto outside;'),('throw','throw failure;'),
            ('runtime include','\n#include "runtime.H"\n'),
            ('unknown feature','\n#ifdef OTHER\nnoop();\n#endif\n'),
        ):
            yield body_fixture(owner+' '+name,owner,code+body)

    _,_,decl=signature(texts[CLASS],'init_stuff',';')
    for name,change in (
        ('level reference',decl[0].replace('int lev','int& lev')),
        ('merged type',decl[0].replace('int lev','intlev')),
        ('extra parameter',decl[0].replace('int lev','int lev, int extra')),
        ('duplicate',decl[0]*2),('nested','struct Inner { '+decl[0]+' };'),
        ('conditional','#ifdef ERF_USE_FFT\n'+decl[0]+'\n#endif'),
    ):
        yield fixture('declaration '+name,CLASS,texts[CLASS][:decl.start()]+change+texts[CLASS][decl.end():])

    for owner,block in (('init_stuff',BUILD), ('InitData_post',REBUILD)):
        for name,before,after in (
            ('missing build','build_fft_solvers(lev);',''),
            ('wrong level','build_fft_solvers(lev);','build_fft_solvers(lev+1);'),
            ('extra argument','build_fft_solvers(lev);','build_fft_solvers(lev, 1);'),
            ('deferred build','build_fft_solvers(lev);','auto later = [&] { build_fft_solvers(lev); };'),
            ('conditional build','build_fft_solvers(lev);','if (ready) build_fft_solvers(lev);'),
            ('duplicate build','build_fft_solvers(lev);','build_fft_solvers(lev); build_fft_solvers(lev);'),
            ('negative FFT feature','#ifdef ERF_USE_FFT','#ifndef ERF_USE_FFT'),
            ('wrong FFT feature','#ifdef ERF_USE_FFT','#ifdef ERF_USE_NETCDF'),
            ('inactive FFT feature','#ifdef ERF_USE_FFT','#if 0'),
            ('anelastic level','anelastic[lev]','anelastic[lev+1]'),
            ('anelastic sense','anelastic[lev] == 1','anelastic[lev] != 1'),
            ('projection level','project_initial_velocity[lev]','project_initial_velocity[lev+1]'),
            ('projection sense','project_initial_velocity[lev] == 1','project_initial_velocity[lev] == 0'),
            ('selector union','== 1) ||','== 1) &&'),
            ('mesh conjunction',')) &&',')) ||'),
            ('mesh kind','StretchedDz','VariableDz'),
        ):
            changed=block.replace(before,after)
            assert changed!=block,name
            yield replace(owner+' '+name,owner,block,changed)
        for name,opening,closing in (
            ('deferred whole build','auto later = [&] {','};'),
            ('runtime gate','if (ready) {','}'),
            ('NETCDF gate','\n#ifdef ERF_USE_NETCDF\n','\n#endif\n'),
        ):
            yield replace(owner+' '+name,owner,block,opening+'\n'+block+'\n'+closing)

    for owner,call in CALLS.items():
        setup='subdomains[lev].clear();' if owner=='RemakeLevel' else 'subdomains.resize(lev+1);'
        for index,code in enumerate((setup,SUBDOMAINS,*SETTERS,call)):
            for name,change in (
                ('missing',''),('duplicate',code*2),('wrong level',code.replace('lev','lev+1')),
                ('conditional','if (ready) '+code),('deferred','auto later = [&] { '+code+' };'),
                ('feature','\n#ifdef ERF_USE_FFT\n'+code+'\n#endif\n'),
            ):
                yield replace(owner+' site '+str(index)+' '+name,owner,code,change)
        _,body,_=bodies[owner]
        for index,code in enumerate((setup,SUBDOMAINS,*SETTERS)):
            match=unique(pattern(code),body,'move setup')
            moved=body[:match.start()]+body[match.end():]+code
            yield body_fixture(owner+' setup after build '+str(index),owner,moved)
        yield replace(owner+' input boxes',owner,SUBDOMAINS,SUBDOMAINS.replace('ba.simplified_list()','other.simplified_list()'))
        yield replace(owner+' init boxes',owner,call,call.replace('lev, ba, dm','lev, other, dm'))
        yield replace(owner+' init mapping',owner,call,call.replace('lev, ba, dm','lev, ba, other'))
        for name,code in (
            ('extra subdomain use','retain(subdomains);'),('extra init call',call),
            ('hidden init call','\n#if 0\n'+call+'\n#endif\n'),
            ('builder bypass','build_fft_solvers(lev);'),
        ):
            yield body_fixture(owner+' '+name,owner,body+code)
    yield replace('scratch alias level','MakeNewLevelFromScratch','auto& lev_new = vars_new[lev];',
                  'auto& lev_new = vars_new[lev+1];')
    yield replace('scratch alias value','MakeNewLevelFromScratch','auto& lev_old = vars_old[lev];',
                  'auto lev_old = vars_old[lev];')
    yield replace('coarse size source','MakeNewLevelFromCoarse','subdomains.resize(lev+1);','subdomains.resize(lev);')
    yield replace('remake clear all','RemakeLevel','subdomains[lev].clear();','subdomains.clear();')

    for name,before,after in (
        ('origin','int lev = 0','int lev = 1'),('upper bound','lev <= finest_level','lev < finest_level'),
        ('increment','lev++','lev--'),('split increment','lev++','+ +lev'),
        ('level count','finest_level','max_level'),('missing loop',LOOP,''),
        ('early continue',GATE,'continue; '+GATE),('early break',GATE,'break; '+GATE),
    ):
        yield replace('restart rebuild '+name,'InitData_post',REBUILD,REBUILD.replace(before,after))
    yield replace('reversed pre post','InitData','InitData_pre(); InitData_post();','InitData_post(); InitData_pre();')
    for name,before,after in (
        ('scratch selector','restart_chkfile.empty()','!restart_chkfile.empty()'),
        ('scratch time','InitFromScratch(zero)','InitFromScratch(one)'),
        ('scratch deferred','InitFromScratch(zero);','auto later = [&] { InitFromScratch(zero); };'),
        ('restart BC missing','init_bcs();',''),
    ):
        yield replace(name,'InitData_pre',START,START.replace(before,after))
    for name,change in (
        ('missing',''),('wrong sense','if (restart_chkfile.empty()) { restart(); }'),
        ('deferred','if (!restart_chkfile.empty()) { auto later = [&] { restart(); }; }'),
    ):
        yield replace('restore '+name,'InitData_post',RESTART,change)
    _,post,_=bodies['InitData_post']
    restore=unique(pattern(RESTART),post,'restore')
    yield body_fixture('restore after rebuild','InitData_post',post[:restore.start()]+post[restore.end():]+RESTART)
    rebuild=unique(pattern(REBUILD),post,'rebuild')
    yield body_fixture('rebuild before restore','InitData_post',REBUILD+post[:rebuild.start()]+post[rebuild.end():])
    opening=post.rfind('if (!restart_chkfile.empty())',0,rebuild.start())
    assert opening>restore.end()
    yield body_fixture('wrong rebuild restart gate','InitData_post',post[:opening]+post[opening:].replace('!restart_chkfile.empty()','restart_chkfile.empty()',1))
    yield replace('rebuild suffix mutation','InitData_post',REBUILD,REBUILD+'\nm_2D_poisson.clear();')
    for name,change in (
        ('missing',''),('conditional','if (ready) ReadCheckpointFile();'),
        ('deferred','auto later = [&] { ReadCheckpointFile(); };'),
        ('extra argument','ReadCheckpointFile(1);'),('duplicate','ReadCheckpointFile(); ReadCheckpointFile();'),
    ):
        yield replace('checkpoint '+name,'restart','ReadCheckpointFile();',change)

    for owner in ('init_stuff','InitData_pre','InitData_post','restart'):
        _,body,_=bodies[owner]
        for name,code in (
            ('build escape','retain(build_fft_solvers);'),
            ('hidden build','\n#if 0\nbuild_fft_solvers(0);\n#endif\n'),
        ):
            yield body_fixture(owner+' '+name,owner,body+code)
    for owner,code in (('MakeNewLevelFromScratch',SUBDOMAINS),
                        ('init_stuff',BUILD),('restart','ReadCheckpointFile();')):
        for name,prefix in (
            ('dangling feature if','\n#ifdef ERF_USE_PARTICLES\nif (ready)\n#else\nnoop();\n#endif\n'),
            ('dangling feature else','\n#ifdef ERF_USE_PARTICLES\nnoop();\n#else\nif (ready)\n#endif\n'),
            ('cross-feature scope','\n#ifdef ERF_USE_PARTICLES\nif (ready) {\n#endif\n'),
        ):
            suffix='\n#ifdef ERF_USE_PARTICLES\n}\n#endif\n' if name=='cross-feature scope' else ''
            yield replace(owner+' '+name,owner,code,prefix+code+suffix)
    for name,prefix,suffix in (
        ('NETCDF-only scope','\n#ifdef ERF_USE_NETCDF\nif (ready) {\n#endif\n','\n#ifdef ERF_USE_NETCDF\n}\n#endif\n'),
        ('disabled scope','\n#if 0\nif (ready) {\n#endif\n','\n#if 0\n}\n#endif\n'),
        ('feature expression','\n#if USE_FC_FACTORY\nif (ready)\n#endif\n',''),
        ('unknown pragma','\n#pragma unexpected\n',''),
        ('unconditional omp','\n#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())\n',''),
    ):
        yield replace(name,'restart','ReadCheckpointFile();',prefix+'ReadCheckpointFile();'+suffix)
    for path in FILES:
        for name,code in (
            ('macro','#define lev other'),('raw string','R"(hidden)";'),
            ('line splice','// hidden \\\n#define lev other'),
        ):
            yield fixture(Path(path).name+' '+name,path,texts[path]+'\n'+code+'\n')
    for path in (ARRAYS,LEVELS,INIT):
        yield fixture(Path(path).name+' outside build',path,texts[path]+'\nvoid unexpected() { build_fft_solvers(0); }')

    yield 'comments removed',texts,False
    yield 'CRLF',{path:text.replace('\n','\r\n') for path,text in original.items()},False
    def whitespace(text):
        return ''.join(part if i%2 else part.replace(',',', /* gap */ ').replace(';',';\n')
                       for i,part in enumerate(re.split('('+STRING+')',text)))
    yield 'token whitespace',{path:whitespace(text) for path,text in texts.items()},False
    for owner in ('init_stuff','MakeNewLevelFromScratch','MakeNewLevelFromCoarse','RemakeLevel','InitData_pre','InitData_post','restart'):
        _,body,_=bodies[owner]
        yield body_fixture(owner+' harmless local',owner,'const int unused = 1;'+body,False)
    for name,code in (
        ('quoted braces and controls','Print() << "} // /* return build_fft_solvers subdomains";'),
        ('local lambda return','auto unused = [] (int x) { return x; };'),
        ('local loop continue','for (int n=0; n<3; ++n) { if (n==1) continue; }'),
        ('extra synchronization','Gpu::streamSynchronize();'),
        ('unrelated supported feature','\n#ifdef ERF_USE_PARTICLES\nnoop();\n#else\nother();\n#endif\n'),
        ('disabled unrelated debug','\n#if 0\nunused_debug();\n#endif\n'),
    ):
        _,body,_=bodies['InitData_post']
        yield body_fixture(name,'InitData_post',code+body,False)


def main():
    work=Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-fft-rebuild-mutants-'))
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
        if not expected: print(result.stdout[-1600:]+result.stderr[-800:])
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__=='__main__':
    sys.exit(main())
