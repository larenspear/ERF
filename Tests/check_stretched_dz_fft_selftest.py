#!/usr/bin/env python3
"""Source fault injection for FFT spacing; fixtures are not compiled C++."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, no_literals, signature
from check_stretched_dz_consumers import pattern
from check_stretched_dz_fft import (ALIAS, BC, BOUNDARY, CALLEE, CLASS, CONSTANT,
    CONTROLS, DEFINITION, FILES, GEOMETRY, LOCALS, LOOP, MEMBERS, READ, SOURCE,
    STRETCHED, SUBCOLUMN, UNIFORM, VERTICAL)
from momentum_geometry import call_arguments, unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_fft.py'
SUCCESS = '4 FFT spacing source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path:strip_comments(text) for path,text in original.items()}
    _, body, definition = signature(texts[SOURCE], 'ERF::solve_with_fft')

    def fixture(name, path, changed, fault=True):
        assert changed != texts[path], name
        return name, {**texts,path:changed}, fault

    def replace(name, path, before, after, fault=True):
        m = unique(pattern(before), texts[path], name)
        return fixture(name,path,texts[path][:m.start()]+after+texts[path][m.end():],fault)

    def body_fixture(name, changed, fault=True):
        return fixture(name,SOURCE,texts[SOURCE][:definition.end()]+changed+
                       texts[SOURCE][definition.end()+len(body):],fault)

    def alter(name, before, after):
        m = unique(pattern(before), body, name)
        return body_fixture(name,body[:m.start()]+after+body[m.end():])

    for before,after in (('int lev','int& lev'),('int isub','int& isub'),
                         ('const Box& subdomain','Box& subdomain'),
                         ('MultiFab& rhs','MultiFab rhs'),('MultiFab& phi','MultiFab phi'),
                         ('int lev','int other, int lev'),('lev','level')):
        yield replace('definition '+before+' -> '+after,SOURCE,DEFINITION,DEFINITION.replace(before,after))
    _,_,declaration = signature(texts[CLASS],'solve_with_fft',';')
    for name,change in (
        ('reference level',declaration[0].replace('int lev','int& lev')),
        ('conditional','#ifdef OTHER\n'+declaration[0]+'\n#endif'),
        ('nested','struct Inner { '+declaration[0]+' };'),
        ('duplicate',declaration[0]*2),
        ('outside class',''),
    ):
        result=texts[CLASS][:declaration.start()]+change+texts[CLASS][declaration.end():]
        if name=='outside class': result+='\n'+declaration[0]
        yield fixture('declaration '+name,CLASS,result)
    for member in MEMBERS:
        for name,change in (
            ('wrong solver',member.replace('Poisson','OtherPoisson')),
            ('pointer type',member.replace('unique_ptr','shared_ptr')),
            ('nested','struct Inner { '+member+' };'),
            ('conditional','#ifdef OTHER\n'+member+'\n#endif'),
            ('duplicate',member*2),
        ):
            yield replace(member.split()[-1]+' '+name,CLASS,member,change)
    yield fixture('negative class FFT guard',CLASS,texts[CLASS].replace('#ifdef ERF_USE_FFT','#ifndef ERF_USE_FFT'))
    yield fixture('negative source FFT guard',SOURCE,texts[SOURCE].replace('#ifdef ERF_USE_FFT','#ifndef ERF_USE_FFT'))
    yield fixture('source namespace',SOURCE,'namespace Other {\n'+texts[SOURCE]+'\n}')
    yield fixture('source suffix',SOURCE,texts[SOURCE]+'\nretain(stretched_dz_d);')

    for name,before,after in (
        ('host lifetime','Vector<Real> stretched_dz_h_sub;','static Vector<Real> stretched_dz_h_sub;'),
        ('device lifetime','Gpu::DeviceVector<Real> stretched_dz_d_sub;','static Gpu::DeviceVector<Real> stretched_dz_d_sub;'),
        ('device type','Gpu::DeviceVector<Real>','Gpu::AsyncVector<Real>'),
        ('host type','Vector<Real>','Gpu::DeviceVector<Real>'),
        ('wrong lower bound','lbound(subdomain)','lbound(other)'),
        ('wrong upper bound','ubound(subdomain)','ubound(other)'),
    ):
        yield alter(name,LOCALS,LOCALS.replace(before,after))
    for name,before,after in (
        ('subcolumn length','subdomain.length(2)','subdomain.length(1)'),
        ('loop lower','int k = sub_lo.z','int k = sub_lo.z+1'),
        ('loop upper','k <= sub_hi.z','k < sub_hi.z'),
        ('loop stride','k++','k+=2'),
        ('output index','k-sub_lo.z','k'),
        ('input level','stretched_dz_h[lev][k]','stretched_dz_h[lev+1][k]'),
        ('input index','stretched_dz_h[lev][k]','stretched_dz_h[lev][k-sub_lo.z]'),
        ('input alias','stretched_dz_h[lev][k]','retain(stretched_dz_h[lev][k])'),
        ('device size','resize(stretched_dz_h_sub.size())','resize(stretched_dz_h_sub.size()-1)'),
        ('async copy','Gpu::copy(','Gpu::copyAsync('),
        ('copy direction','Gpu::hostToDevice','Gpu::deviceToHost'),
        ('copy start','stretched_dz_h_sub.begin()','stretched_dz_h_sub.begin()+1'),
        ('copy end','stretched_dz_h_sub.end()','stretched_dz_h_sub.end()-1'),
        ('copy destination','stretched_dz_d_sub.begin()','stretched_dz_d_sub.begin()+1'),
        ('solver level','m_2D_poisson[lev]','m_2D_poisson[lev+1]'),
        ('solver subdomain','[isub]->solve','[isub+1]->solve'),
        ('solver solution','solve(phi, rhs,','solve(other, rhs,'),
        ('solver RHS','solve(phi, rhs,','solve(phi, other,'),
        ('solver overload','solve(phi, rhs, stretched_dz_d_sub)','solve(phi, rhs, stretched_dz_h_sub)'),
        ('solver spacing','solve(phi, rhs, stretched_dz_d_sub)','solve(phi, rhs, stretched_dz_d[lev])'),
        ('device destroyed before solve','m_2D_poisson','stretched_dz_d_sub.clear(); m_2D_poisson'),
        ('copy omitted','Gpu::copy(Gpu::hostToDevice, stretched_dz_h_sub.begin(),\n        stretched_dz_h_sub.end(), stretched_dz_d_sub.begin());',''),
    ):
        changed=SUBCOLUMN.replace(before,after)
        assert changed!=SUBCOLUMN,name
        yield alter(name,SUBCOLUMN,changed)
    for statement in (LOCALS,SUBCOLUMN,UNIFORM,BC,GEOMETRY):
        label=statement.split()[0]
        for name,changed in (
            ('deferred','auto later = [&] { '+statement+' };'),
            ('unbraced','if (ready) '+statement),
            ('nested','if (ready) { '+statement+' }'),
        ):
            yield alter(label+' '+name,statement,changed)
    yield alter('reversed solver dispatch',CONSTANT,CONSTANT.replace('==','!='))
    yield alter('reversed flux dispatch',STRETCHED,STRETCHED.replace('!=','=='))
    gate=unique(pattern(CONSTANT),body,'dispatch')
    end=gate.end()+len(brace_block(body,gate.end()-1))+1
    attached=re.match(r'\s*else\s*\{',body[end:])
    assert attached
    yield body_fixture('detached solver else',body[:end]+body[end:].replace('else','if (ready)',1))
    for statement,name in ((LOCALS,'locals after solve'),(BC,'BC before solve'),(GEOMETRY,'geometry before solve')):
        m=unique(pattern(statement),body,name)
        without=body[:m.start()]+body[m.end():]
        yield body_fixture(name,without+statement if statement==LOCALS else statement+without)

    loop=unique(pattern(LOOP),body,'loop')
    loop_end=loop.end()+len(brace_block(body,loop.end()-1))+1
    for name,change in (
        ('reference','for (MFIter& mfi = other; mfi.isValid(); ++mfi) {'),
        ('sync disabled',LOOP.replace('TilingIfNotGPU()','MFItInfo{}.DisableDeviceSync()')),
        ('wrong container',LOOP.replace('phi','rhs')),
        ('wrong advance',LOOP.replace('++mfi','advance()')),
        ('missing advance',LOOP.replace('++mfi','')),
        ('deferred','auto later = [&] { '+LOOP),
    ):
        changed=body[:loop.start()]+change+body[loop.end():]
        if name=='deferred':
            changed=body[:loop.start()]+'auto later = [&] { '+body[loop.start():loop_end]+' };'+body[loop_end:]
        yield body_fixture('iterator '+name,changed)
    for code in ('mfi.Finalize();','auto& escaped = mfi;','retain(&mfi);','mfi = other;',
                 'Gpu::Device::setStreamIndex(1);','Gpu::NoSyncRegion region;'):
        yield body_fixture('iterator '+code,body[:loop.end()]+code+body[loop.end():])
    arguments=list(call_arguments(no_literals(body),CALLEE))
    launches=list(re.finditer(CALLEE+r'\s*\(',no_literals(body)))
    for index,(launch,args) in enumerate(zip(launches,arguments)):
        lo,hi=args[1]
        kernel=body[lo:hi]
        for name,change in (
            ('reference capture',kernel.replace('[=]','[&]')),
            ('initial capture',kernel.replace('[=]','[alias = stretched_dz_d_ptr]')),
            ('host lambda',kernel.replace('AMREX_GPU_DEVICE','')),
            ('wrapped lambda','retain('+kernel+')'),
        ):
            yield body_fixture(f'kernel {index} '+name,body[:lo]+change+body[hi:])
        end=body.index(';',hi)+1
        statement=body[launch.start():end]
        for name,change in (
            ('deferred','auto later = [&] { '+statement+' };'),
            ('unbraced','if (ready) '+statement),
            ('wrong box',statement.replace(body[args[0][0]:args[0][1]],'other_box',1)),
        ):
            yield body_fixture(f'launch {index} '+name,body[:launch.start()]+change+body[end:])
    # Move a complete reader rather than merely changing its spelling.
    alias=unique(pattern(ALIAS),body,'alias')
    third_end=body.index(';',arguments[2][1][1])+1
    yield body_fixture('pointer declaration after reader',body[:alias.start()]+body[alias.end():third_end]+
                       ALIAS+body[third_end:])
    yield body_fixture('vertical reader outside iterator',body[:launches[2].start()]+
                       body[third_end:loop_end]+body[launches[2].start():third_end]+body[loop_end:])
    for name,change in (('pointer host read',READ),('pointer write','stretched_dz_d_ptr[0] = 1;'),
                        ('pointer reassignment','stretched_dz_d_ptr = other;'),
                        ('pointer reference','auto& alias = stretched_dz_d_ptr;')):
        yield body_fixture(name,body[:alias.end()]+change+body[alias.end():])
    zgate=unique(pattern(STRETCHED),body,'z gate')
    zend=zgate.end()+len(brace_block(body,zgate.end()-1))+1
    yield body_fixture('detached constant flux else',body[:zend]+body[zend:].replace('else','if (ready)',1))
    yield body_fixture('stretched gate deferred',body[:zgate.start()]+'auto later = [&] { '+
                       body[zgate.start():loop_end-1]+' };'+body[loop_end-1:])
    for index,axis in enumerate('xyz'):
        for before,after in ((f'mfi.nodaltilebox({index})','mfi.nodaltilebox(3)'),
                             (f'fluxes[{index}].array(mfi)',f'fluxes[{(index+1)%3}].array(mfi)')):
            yield alter(axis+' field '+before,before,after)
    for name,before,after in (
        ('pointer level','stretched_dz_d[lev]','stretched_dz_d[lev+1]'),
        ('pointer local subcolumn','stretched_dz_d[lev]','stretched_dz_d_sub'),
        ('pointer offset','.data()','.data()+1'),
        ('pointer lifetime','Real*','static Real*'),
    ):
        yield alter(name,ALIAS,ALIAS.replace(before,after))
    for name,before,after in (
        ('read reference','Real dz','Real& dz'),
        ('read index','[k-1]','[k+1]'),
        ('read escape','stretched_dz_d_ptr[k]','retain(stretched_dz_d_ptr[k])'),
        ('read weight','myhalf','one'),
        ('lower boundary','k == sub_lo.z','k == sub_lo.z+1'),
        ('upper boundary','k == sub_hi.z+1','k == sub_hi.z'),
        ('top boundary','!top_is_dirichlet','top_is_dirichlet'),
        ('read deferred',READ,'auto later = [=] { '+READ+' };'),
        ('read host call',READ,'on_host([=] { '+READ+' });'),
        ('extra pointer alias',READ,READ+'auto alias = stretched_dz_d_ptr;'),
        ('scalar mutated',READ,READ+'dz = other;'),
    ):
        yield alter(name,VERTICAL,VERTICAL.replace(before,after))
    for before,after in (('geom[lev]','geom[lev+1]'),('sub_hi.z == dom_hi.z','sub_hi.z != dom_hi.z'),
                         ('"Outflow"','"Inflow"'),('"Open"','"Closed"')):
        yield alter('geometry '+before,GEOMETRY,GEOMETRY.replace(before,after))
    for name,code in (
        ('host vector clear','stretched_dz_h_sub.clear();'),
        ('device vector clear','stretched_dz_d_sub.clear();'),
        ('host vector alias','auto& alias = stretched_dz_h_sub;'),
        ('device vector alias','auto& alias = stretched_dz_d_sub;'),
        ('pointer escape','retain(stretched_dz_d_ptr);'),
        ('member resize','stretched_dz_d[lev].resize(1);'),
        ('host member mutation','stretched_dz_h[lev][0] = 0;'),
        ('level mutation','++lev;'),('subdomain index mutation','++isub;'),
        ('subdomain alias','auto& alias = subdomain;'),('subdomain bound mutation','sub_lo.z = 1;'),
        ('boundary mutation','top_is_dirichlet = false;'),('solver escape','retain(m_2D_poisson);'),
        ('self escape','retain(this);'),('solution escape','retain(phi);'),('RHS escape','retain(rhs);'),
        ('flux escape','retain(fluxes);'),('mesh mutation','solverChoice.mesh_type = MeshType::ConstantDz;'),
        ('box escape','retain(zbx);'),('runtime include','#include "runtime.H"'),
        ('unknown conditional','#ifdef OTHER\n#endif'),('early return','return;'),
    ):
        yield body_fixture(name,body+'\n'+code+'\n')
    for path in FILES:
        for name,code in (
            ('stream control','Gpu::Device::setStreamIndex(1);'),
            ('sync control','Gpu::NoSyncRegion region;'),
            ('macro','#define lev other'),('raw string','R"(hidden)";'),
            ('line splice','// hidden \\\n#define lev other'),
        ):
            yield fixture(Path(path).stem+' '+name,path,texts[path]+'\n'+code+'\n')
    yield body_fixture('unsafe OpenMP',body.replace('Gpu::notInLaunchRegion()','true'))
    yield body_fixture('unguarded OpenMP',body.replace('#ifdef _OPENMP','').replace('#endif',''))

    yield 'comments removed',texts,False
    yield 'CRLF',{path:text.replace('\n','\r\n') for path,text in original.items()},False
    def whitespace(text):
        return ''.join(part if i%2 else part.replace(',',', /* gap */ ').replace(';',';\n')
                       for i,part in enumerate(re.split('('+STRING+')',text)))
    yield 'token whitespace',{path:whitespace(text) for path,text in texts.items()},False
    for name,code in (
        ('harmless local','const int unused = 1;'),
        ('quoted controls','Print() << "} // /* '+' '.join(CONTROLS)+' stretched_dz_d_ptr solve_with_fft";'),
        ('additional synchronization','Gpu::streamSynchronize();'),
    ):
        yield body_fixture(name,code+body,False)
    yield body_fixture('loop additional synchronization',body[:loop.end()]+'Gpu::streamSynchronize();'+body[loop.end():],False)
    yield body_fixture('optional kernel noexcept',body.replace('noexcept',''),False)
    yield body_fixture('qualified launch',body.replace('ParallelFor(','amrex::ParallelFor('),False)
    yield body_fixture('OpenMP removed',re.sub(r'#ifdef _OPENMP\s*#pragma[^\n]*\s*#endif','',body),False)
    yield replace('read comments',SOURCE,READ,READ.replace('myhalf *','myhalf /* scalar */ *'),False)
    yield fixture('declaration parameter rename',CLASS,texts[CLASS][:declaration.start()]+
                  declaration[0].replace('MultiFab& p','MultiFab& solution')+texts[CLASS][declaration.end():],False)


def main():
    work=Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-fft-spacing-mutants-'))
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
