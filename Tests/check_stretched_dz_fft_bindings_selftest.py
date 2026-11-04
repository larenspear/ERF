#!/usr/bin/env python3
"""Source fault injection for FFT construction/callers; no C++ compilation."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, signature
from check_stretched_dz_fft_bindings import (BUILD, BUILD_PLAN, CALL, CLASS,
    CONTROLS, DISPATCH, FILES, GEOMETRY, LOOP, MESH, NON_EB, PLAN, PROJECTION,
    RESET, SLOTS, SOURCE, VARIABLE, pattern)
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_fft_bindings.py'
SUCCESS = '4 FFT construction/caller source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path:strip_comments(text) for path,text in original.items()}
    bodies = {}
    for path,name in ((SOURCE,'ERF::build_fft_solvers'),(PROJECTION,'ERF::project_momenta')):
        _,body,definition = signature(texts[path],name)
        bodies[path] = (body,definition.end())

    def fixture(name,path,changed,fault=True):
        assert changed != texts[path], name
        return name,{**texts,path:changed},fault

    def replace(name,path,before,after,fault=True):
        match=unique(pattern(before),texts[path],name)
        return fixture(name,path,texts[path][:match.start()]+after+texts[path][match.end():],fault)

    def body_fixture(name,path,changed,fault=True):
        body,offset=bodies[path]
        return fixture(name,path,texts[path][:offset]+changed+texts[path][offset+len(body):],fault)

    def alter(name,path,block,before,after):
        changed=block.replace(before,after)
        assert changed!=block,name
        return replace(name,path,block,changed)

    for before,after in (('int lev','int& lev'),('int lev','intlev'),('int lev','int other'),
                         ('int lev','int lev, int other')):
        yield alter('builder signature '+after,SOURCE,BUILD,before,after)
    _,_,decl=signature(texts[CLASS],'build_fft_solvers',';')
    for name,change in (
        ('level reference',decl[0].replace('int lev','int& lev')),
        ('duplicate',decl[0]*2),('nested','struct Inner { '+decl[0]+' };'),
        ('conditional','#ifdef OTHER\n'+decl[0]+'\n#endif'),('outside class',''),
    ):
        result=texts[CLASS][:decl.start()]+change+texts[CLASS][decl.end():]
        if name=='outside class': result+='\n'+decl[0]
        yield fixture('builder declaration '+name,CLASS,result)
    yield fixture('negative class FFT guard',CLASS,texts[CLASS].replace('#ifdef ERF_USE_FFT','#ifndef ERF_USE_FFT'))
    yield fixture('negative source FFT guard',SOURCE,texts[SOURCE].replace('#ifdef ERF_USE_FFT','#ifndef ERF_USE_FFT'))
    yield fixture('source namespace',SOURCE,'namespace Other {\n'+texts[SOURCE]+'\n}')
    yield fixture('source suffix',SOURCE,texts[SOURCE]+'\nretain(m_2D_poisson);')
    for member in ('m_3D_poisson','m_2D_poisson'):
        for before,after in (
            (member+'.size() > lev',member+'.size() >= lev'),
            (member+'[lev].size()',member+'[lev+1].size()'),
            (member+'[lev][n].reset()',member+'[lev+1][n].reset()'),
            (member+'[lev][n].reset()',member+'[lev][n+1].reset()'),
            (member+'[lev][n].reset()','retain('+member+'[lev][n])'),
        ):
            yield alter('reset '+before,SOURCE,RESET,before,after)
    for before,after in (('ConstantDz','VariableDz'),('StretchedDz','VariableDz'),('||','&&')):
        yield alter('builder mesh '+before,SOURCE,MESH,before,after)
    for path,block in ((SOURCE,BUILD_PLAN),(PROJECTION,PLAN)):
        for name,before,after in (
            ('initial flag','will_solve_with_mlmg = '+('(solverChoice.mesh_type == MeshType::ConstantDz)' if path==SOURCE else 'false'),
             'will_solve_with_mlmg = true'),
            ('all boxes flag','all_boxes_ok = true','all_boxes_ok = false'),
            ('loop origin','int isub = 0','int isub = 1'),
            ('loop bound','isub < subdomains[lev].size()','isub <= subdomains[lev].size()'),
            ('loop level','subdomains[lev].size()','subdomains[lev+1].size()'),
            ('loop increment','++isub','--isub'),
            ('split increment','++isub','+ +isub'),
            ('region level','subdomains[lev][isub]','subdomains[lev+1][isub]'),
            ('region slot','subdomains[lev][isub]','subdomains[lev][isub+1]'),
            ('rectangle comparison','my_region.numPts() ==','my_region.numPts() !='),
            ('rejected rectangle','if (!boxes_make_rectangle)','if (boxes_make_rectangle)'),
            ('all-box reduction','all_boxes_ok = false','all_boxes_ok = true'),
            ('accept gate','if (all_boxes_ok)','if (!all_boxes_ok)'),
            ('planner assignment','will_solve_with_mlmg = false; }','will_solve_with_mlmg = true; }'),
            ('merged declaration','bool all_boxes_ok','boolall_boxes_ok'),
            ('planner alias','bool all_boxes_ok','bool& all_boxes_ok'),
        ):
            yield alter(Path(path).stem+' '+name,path,block,before,after)
    for name,before,after in (
        ('use FFT reversed','if (use_fft)','if (!use_fft)'),
        ('non-FFT fallback','defaulting to MLMG','ignoring MLMG'),
        ('constant flag','will_solve_with_mlmg = true','will_solve_with_mlmg = false'),
    ):
        yield alter(name,PROJECTION,PLAN,before,after)

    for name,before,after in (
        ('builder region level','subdomains[lev][isub]','subdomains[lev+1][isub]'),
        ('builder region slot','subdomains[lev][isub]','subdomains[lev][isub+1]'),
        ('builder region lifetime','Box subdomain','static Box subdomain'),
        ('domain level','Geom(lev)','Geom(lev+1)'),
        ('cell size level','geom[lev].CellSizeArray()','geom[lev+1].CellSizeArray()'),
        ('BC level','get_fft_bc(geom[lev]','get_fft_bc(geom[lev+1]'),
        ('BC region','domain_bc_type,subdomain,','domain_bc_type,other,'),
        ('BC real flag','solverChoice.use_real_bcs','false'),
        ('BC buffer','domain_bc_type,','other_bcs,'),
        ('Geometry reference','Geometry my_geom','Geometry& my_geom'),
        ('Geometry lifetime','Geometry my_geom','static Geometry my_geom'),
        ('vertical periodicity','is_per[2] = 0','is_per[2] = 1'),
        ('periodic x extent','sub_hi.x == dom_hi.x','sub_hi.x != dom_hi.x'),
        ('periodic y extent','sub_lo.y == dom_lo.y','sub_lo.y != dom_lo.y'),
        ('periodic axis','isPeriodic(0)','isPeriodic(2)'),
        ('coordinate system','coord_sys = 0','coord_sys = 1'),
        ('full domain gate','subdomain == Geom','subdomain != Geom'),
        ('full domain argument','define(Geom(lev).Domain(),','define(other,'),
        ('full physical extent','Geom(lev).ProbDomain()','other_box'),
        ('subdomain argument','define(subdomain,','define(other,'),
        ('subdomain physical extent','subdomain, rb,','subdomain, other_box,'),
        ('periodicity argument','coord_sys, is_per','coord_sys, other_periodicity'),
        ('rectangularity gate','!boxes_make_rectangle &&','boxes_make_rectangle &&'),
        ('fallback gate','&& !will_solve_with_mlmg','&& will_solve_with_mlmg'),
        ('rectangle rejection','amrex::Abort','amrex::Warning'),
        ('Geometry escape','Geometry my_geom;','Geometry my_geom; retain(my_geom);'),
        ('BC escape','Geometry my_geom;','retain(bc_fft); Geometry my_geom;'),
    ):
        yield alter(name,SOURCE,GEOMETRY,before,after)
    for axis,index in zip('xyz',range(3)):
        for name,before,after in (
            ('lower',f'sub_lo.{axis}*dx[{index}]',f'(sub_lo.{axis}+1)*dx[{index}]'),
            ('upper',f'(sub_hi.{axis}+1)*dx[{index}]',f'sub_hi.{axis}*dx[{index}]'),
            ('spacing',f'dx[{index}]',f'dx[{(index+1)%3}]'),
        ):
            yield alter('RealBox '+axis+' '+name,SOURCE,GEOMETRY,before,after)
    for member in ('m_3D_poisson','m_2D_poisson'):
        for name,before,after in (
            ('level bound',member+'.size() <= lev',member+'.size() < lev'),
            ('level capacity',member+'.resize(lev+1)',member+'.resize(lev)'),
            ('slot bound',member+'[lev].size() <= isub',member+'[lev].size() < isub'),
            ('slot capacity',member+'[lev].resize(isub+1)',member+'[lev].resize(isub)'),
            ('assignment level',member+'[lev][isub] =',member+'[lev+1][isub] ='),
            ('assignment slot',member+'[lev][isub] =',member+'[lev][isub+1] ='),
            ('discarded construction',member+'[lev][isub] =','auto discarded ='),
        ):
            yield alter(member+' '+name,SOURCE,SLOTS,before,after)
    for name,before,after in (
        ('constructor Geometry','(my_geom,bc_fft)','(other_geom,bc_fft)'),
        ('constructor BC','(my_geom,bc_fft)','(my_geom,other_bc)'),
        ('hybrid type','PoissonHybrid<MultiFab>','Poisson<MultiFab>'),
        ('constant gate','ConstantDz && !will_solve_with_mlmg','ConstantDz && will_solve_with_mlmg'),
        ('stretched gate','MeshType::StretchedDz','MeshType::VariableDz'),
    ):
        yield alter(name,SOURCE,SLOTS,before,after)
    for path,statement in ((SOURCE,RESET),(SOURCE,BUILD_PLAN),(SOURCE,GEOMETRY),(SOURCE,SLOTS),
                           (PROJECTION,PLAN),(PROJECTION,DISPATCH)):
        for name,change in (
            ('deferred','auto later = [&] { '+statement+' };'),
            ('conditional','if (ready) { '+statement+' }'),
            ('unbraced','if (ready) '+statement),
            ('omitted',''),
        ):
            yield replace(Path(path).stem+' '+statement.split()[0]+' '+name,path,statement,change)
    build,_=bodies[SOURCE]
    yield body_fixture('builder extra mutation',SOURCE,build+'m_2D_poisson[lev].clear();')
    yield body_fixture('builder early return',SOURCE,'return;'+build)
    yield body_fixture('builder runtime include',SOURCE,'\n#include "runtime.H"\n'+build)
    yield body_fixture('builder inserted reconfiguration',SOURCE,build+'reconfigure();')
    yield body_fixture('builder split construction-loop increment',SOURCE,build.replace('++isub','+ +isub'))

    project,_=bodies[PROJECTION]
    calls=list(re.finditer(pattern(CALL),project))
    assert len(calls)==2
    for i,call in enumerate(calls):
        for before,after in (('lev,','lev+1,'),('isub,','isub+1,'),('my_region','other_region'),
                             ('rhs_sub[0]','rhs_sub[1]'),('phi_sub[0]','phi_sub[1]'),
                             ('fluxes_sub[0]','fluxes_sub[1]')):
            yield body_fixture(f'call {i} '+before,PROJECTION,
                               project[:call.start()]+CALL.replace(before,after)+project[call.end():])
        for name,change in (('deferred','auto later = [&] { '+CALL+' };'),
                            ('conditional','if (ready) { '+CALL+' }'),
                            ('extra argument',CALL.replace('fluxes_sub[0])','fluxes_sub[0], other)'))):
            yield body_fixture(f'call {i} '+name,PROJECTION,
                               project[:call.start()]+change+project[call.end():])
    for name,before,after in (
        ('constant dispatch','MeshType::ConstantDz','MeshType::VariableDz'),
        ('stretched dispatch','MeshType::StretchedDz','MeshType::VariableDz'),
        ('MLMG dispatch','if (will_solve_with_mlmg)','if (!will_solve_with_mlmg)'),
        ('region level','subdomains[lev][isub]','subdomains[lev+1][isub]'),
        ('region slot','subdomains[lev][isub]','subdomains[lev][isub+1]'),
        ('rectangle bypass','if (!boxes_make_rectangle)','if (boxes_make_rectangle)'),
        ('rectangle comparison','my_region.numPts() ==','my_region.numPts() !='),
        ('abort downgraded','amrex::Abort','amrex::Warning'),
        ('region escape','Box my_region','static Box my_region'),
        ('disabled FFT guard','#ifdef ERF_USE_FFT','#ifndef ERF_USE_FFT'),
    ):
        yield alter(name,PROJECTION,DISPATCH,before,after)
    yield replace('reversed EB gate',PROJECTION,NON_EB,NON_EB.replace('!=','=='))
    gate=unique(pattern(NON_EB),project,'non EB')
    end=gate.end()+len(brace_block(project,gate.end()-1))+1
    yield body_fixture('deferred entire dispatch',PROJECTION,project[:gate.start()]+
                       'auto later = [&] { '+project[gate.start():end]+' };'+project[end:])
    yield body_fixture('projection split subdomain-loop increment',PROJECTION,project.replace('++isub','+ +isub'))
    for name,code in (
        ('planner mutation','will_solve_with_mlmg = false;'),('planner alias','auto& alias = will_solve_with_mlmg;'),
        ('extra call',CALL),('hidden disabled call','\n#if 0\n'+CALL+'\n#endif\n'),
        ('callee escape','retain(solve_with_fft);'),('solver reset','m_2D_poisson[lev].clear();'),
        ('builder call','build_fft_solvers(lev);'),('spacing escape','retain(stretched_dz_h);'),
        ('runtime include','\n#include "runtime.H"\n'),('unknown conditional','\n#ifdef OTHER\n#endif\n'),
        ('nonlocal transfer','goto outside;'),
    ):
        yield body_fixture(name,PROJECTION,project+code)
    yield body_fixture('region escape after dispatch',PROJECTION,project[:end-1]+'retain(my_region);'+project[end-1:])
    yield alter('variable region mutation',PROJECTION,VARIABLE,'solve_with_gmres(lev, my_region,','solve_with_gmres(lev, ++my_region,')
    plan=unique(pattern(PLAN),project,'plan')
    yield body_fixture('planning after calls',PROJECTION,project[:plan.start()]+project[plan.end():]+PLAN)
    for path in FILES:
        for name,code in (
            ('stream control','Gpu::Device::setStreamIndex(1);'),('sync control','Gpu::NoSyncRegion region;'),
            ('macro','#define lev other'),('raw string','R"(hidden)";'),
            ('line splice','// hidden \\\n#define lev other'),
        ):
            yield fixture(Path(path).stem+' '+name,path,texts[path]+'\n'+code+'\n')

    yield 'comments removed',texts,False
    yield 'CRLF',{path:text.replace('\n','\r\n') for path,text in original.items()},False
    def whitespace(text):
        return ''.join(part if i%2 else part.replace(',',', /* gap */ ').replace(';',';\n')
                       for i,part in enumerate(re.split('('+STRING+')',text)))
    yield 'token whitespace',{path:whitespace(text) for path,text in texts.items()},False
    for name,code in (
        ('harmless projection local','const int unused = 1;'),
        ('quoted controls','Print() << "} // /* '+' '.join(CONTROLS)+' solve_with_fft will_solve_with_mlmg";'),
        ('extra synchronization','Gpu::streamSynchronize();'),
        ('disabled unrelated debug','\n#if 0\nunused_debug();\n#endif\n'),
    ):
        yield body_fixture(name,PROJECTION,code+project,False)
    yield fixture('declaration parameter rename',CLASS,texts[CLASS][:decl.start()]+decl[0].replace('lev','level')+texts[CLASS][decl.end():],False)
    yield replace('Geometry comment',SOURCE,'Geometry my_geom;','Geometry /* local */ my_geom;',False)
    yield replace('call comment',PROJECTION,DISPATCH,DISPATCH.replace(CALL,'/* direct */ '+CALL),False)


def main():
    work=Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-fft-bindings-mutants-'))
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
