#!/usr/bin/env python3
"""Source fault injection and equivalent controls for FFT face classification."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block
from check_stretched_dz_fft_bc import CALL, CLASS, FILES, LOOP, SIGNATURE, SOURCE, TYPE, UTILS, pattern
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_fft_bc.py'
SUCCESS = '3 FFT boundary source/model contracts passed;'


def cases(original):
    yield 'baseline',original,False
    texts={path:strip_comments(text) for path,text in original.items()}
    definition=unique(pattern(SIGNATURE),texts[UTILS],'helper')
    body=brace_block(texts[UTILS],definition.end()-1)
    end=definition.end()+len(body)+1

    def fixture(name,path,changed,fault=True):
        assert changed!=texts[path],name
        return name,{**texts,path:changed},fault

    def replace(name,path,before,after,fault=True):
        match=unique(pattern(before),texts[path],name)
        return fixture(name,path,texts[path][:match.start()]+after+texts[path][match.end():],fault)

    def changed_body(name,changed,fault=True):
        return fixture(name,UTILS,texts[UTILS][:definition.end()]+changed+texts[UTILS][end-1:],fault)

    def region(name,before,after,fault=True):
        match=unique(pattern(before),body,name)
        return changed_body(name,body[:match.start()]+after+body[match.end():],fault)

    for name,before,after in (
        ('result reference',TYPE+' get_fft_bc',TYPE+'& get_fft_bc'),
        ('geometry mutable','Geometry const&','Geometry&'),
        ('box value','Box const&','Box'),('real flag reference','bool use_real_bcs','bool& use_real_bcs'),
        ('extra parameter','bool use_real_bcs','bool use_real_bcs, int extra'),
        ('merged type','bool use_real_bcs','booluse_real_bcs'),
        ('wrong dimension','2*AMREX_SPACEDIM','AMREX_SPACEDIM'),
        ('missing noexcept',' noexcept',''),
    ):
        yield replace(name,UTILS,SIGNATURE,SIGNATURE.replace(before,after))
    for name,opening,closing in (
        ('namespace','namespace Other {','}'),
        ('runtime closure','auto later = [] {','};'),
        ('extra feature','#ifdef OTHER\n','\n#endif'),
        ('disabled definition','#if 0\n','\n#endif'),
    ):
        yield fixture(name,UTILS,texts[UTILS][:definition.start()]+opening+
                      texts[UTILS][definition.start():end]+closing+texts[UTILS][end:])
    for name,before,after in (
        ('negative FFT guard','#ifdef ERF_USE_FFT','#ifndef ERF_USE_FFT'),
        ('header guard','#ifndef ERF_SOLVER_UTILS_H_','#ifdef ERF_SOLVER_UTILS_H_'),
    ):
        yield replace(name,UTILS,before,after)
    yield fixture('extra helper declaration',UTILS,texts[UTILS]+'\nvoid get_fft_bc();')
    for name,before,after in (
        ('loop origin','dir = 0','dir = 1'),('loop bound','dir <= 2','dir < 2'),
        ('loop extent','dir <= 2','dir <= 3'),('loop decrement','dir++','dir--'),
        ('loop split increment','dir++','+ +dir'),('loop stride','dir++','dir+=2'),
    ):
        yield region(name,LOOP,LOOP.replace(before,after))
    for name,before,after in (
        ('static result',TYPE+' r;', 'static '+TYPE+' r;'),
        ('result alias',TYPE+' r;',TYPE+'& r;'),
        ('missing result','return r;','return other;'),
        ('wrong output axis','r[dir] = std::make_pair(first,second);','r[0] = std::make_pair(first,second);'),
        ('swapped output faces','r[dir] = std::make_pair(first,second);','r[dir] = std::make_pair(second,first);'),
        ('duplicated low face','r[dir] = std::make_pair(first,second);','r[dir] = std::make_pair(first,first);'),
        ('duplicated high face','r[dir] = std::make_pair(first,second);','r[dir] = std::make_pair(second,second);'),
    ):
        yield region(name,before,after)

    for side,slot,orientation,endpoint,other in (
            ('lo','first','low','smallEnd','bigEnd'), ('hi','second','high','bigEnd','smallEnd')):
        read='auto bc_type_'+side+' = l_domain_bc_type[Orientation(dir,Orientation::'+orientation+')];'
        for name,before,after in (
            ('wrong orientation',orientation,'high' if side=='lo' else 'low'),
            ('wrong axis','Orientation(dir','Orientation(0'),
            ('read alias','auto bc_type_','auto& bc_type_'),
            ('wrong names','l_domain_bc_type','other_names'),
        ):
            yield region(side+' '+name,read,read.replace(before,after))
        touch='lev_geom.Domain().'+endpoint+'(dir) == bounding_box.'+endpoint+'(dir)'
        periodic='lev_geom.isPeriodic(dir) && ('+touch+')'
        odd='('+touch+') && (bc_type_'+side+' == "Outflow" || bc_type_'+side+' == "Open") && (dir == 2 || !use_real_bcs)'
        for name,before,after in (
            ('periodic disabled',periodic,'false'),
            ('periodic unconditional',periodic,'true'),
            ('periodic inverted','lev_geom.isPeriodic(dir)','!lev_geom.isPeriodic(dir)'),
            ('periodic union','&&','||'),
            ('periodic wrong end',endpoint,other),
            ('periodic wrong contact','==','!='),
            ('periodic without contact',periodic,'lev_geom.isPeriodic(dir)'),
            ('periodic wrong axis','isPeriodic(dir)','isPeriodic(0)'),
        ):
            yield region(side+' '+name,periodic,periodic.replace(before,after))
        for name,before,after in (
            ('odd disabled',odd,'false'),('odd unconditional',odd,'true'),
            ('odd wrong end',endpoint,other),('odd contact union',') && (bc_type_',') || (bc_type_'),
            ('odd contact sense',touch,touch.replace('==','!=')),
            ('odd boundary intersection','|| bc_type_','&& bc_type_'),
            ('odd Outflow typo','"Outflow"','"OutFlow"'),('odd Open typo','"Open"','"Closed"'),
            ('odd names swapped','bc_type_'+side,'bc_type_'+('hi' if side=='lo' else 'lo')),
            ('odd Outflow negated','== "Outflow"','!= "Outflow"'),
            ('odd Open negated','== "Open"','!= "Open"'),
            ('odd wrong vertical','dir == 2','dir == 1'),
            ('odd nonvertical','dir == 2','dir != 2'),
            ('odd vertical conjunction','2 ||','2 &&'),
            ('odd real sense','!use_real_bcs','use_real_bcs'),
            ('odd ignores real','!use_real_bcs','true'),
            ('integer negation precedence','dir == 2','!dir != 2'),
            ('coordinate negation precedence',touch,'!'+touch.replace('==','!=')),
        ):
            yield region(side+' '+name,odd,odd.replace(before,after))
        for selected in ('periodic','odd','even'):
            before=slot+' = FFT::Boundary::'+selected+';'
            for alternative in ('periodic','odd','even'):
                if selected!=alternative:
                    yield region(side+' '+selected+' becomes '+alternative,before,before.replace(selected,alternative))
        for name,code in (
            ('side effect','use_real_bcs = true'),('unknown call','external()'),
            ('unknown variable','other_periodic'),('comma expression','(mutate(), true)'),
            ('unsupported operator','lev_geom.isPeriodic(dir) & true'),
        ):
            yield region(side+' '+name,periodic,code)
        # Equivalent source conditions must pass semantic comparison.
        for name,old,new in (
            ('periodic commuted',periodic,'('+touch+') && lev_geom.isPeriodic(dir)'),
            ('periodic double negation',periodic,'!!('+periodic+')'),
            ('odd flag commuted',odd,odd.replace('dir == 2 || !use_real_bcs','!use_real_bcs || dir == 2')),
            ('parenthesized integer comparison',odd,odd.replace('dir == 2','!(dir != 2)')),
            ('parenthesized coordinate comparison',periodic,periodic.replace(touch,'!('+touch.replace('==','!=')+')')),
            ('odd tautology',odd,'('+odd+') && (bc_type_'+side+' != "SlipWall" || bc_type_'+side+' == "SlipWall")'),
        ):
            yield region(side+' '+name,old,new,False)

    for name,code in (
        ('early return','return r;'),('extra output write','r[0] = r[1];'),
        ('escaped result','retain(r);'),('deferred mutation','auto later = [&] { first = FFT::Boundary::even; };'),
        ('runtime include','\n#include "other.H"\n'),('hidden mutation','\n#if 0\nmutate(r);\n#endif\n'),
        ('extra synchronization','Gpu::streamSynchronize();'),
    ):
        yield changed_body(name,code+body)
    for name,before,after in (
        ('caller level','geom[lev]','geom[lev+1]'),('caller geometry','geom[lev]','my_geom'),
        ('caller BC names','domain_bc_type','other_names'),('caller box','subdomain','other_box'),
        ('caller real flag','solverChoice.use_real_bcs','false'),
        ('caller alias','auto bc_fft','auto& bc_fft'),
        ('caller mutation','subdomain','++subdomain'),
    ):
        yield replace(name,SOURCE,CALL,CALL.replace(before,after))
    yield replace('deferred caller',SOURCE,CALL,'auto later = [&] { '+CALL+' };')
    yield fixture('extra helper call',SOURCE,texts[SOURCE]+'\nauto extra = get_fft_bc(other);')
    yield replace('ERF member shadows helper',CLASS,'void build_fft_solvers(int lev);',
                  'void get_fft_bc(); void build_fft_solvers(int lev);')
    for member,kind in (('m_3D_poisson','Poisson'),('m_2D_poisson','PoissonHybrid')):
        before=member+'[lev][isub] = std::make_unique<FFT::'+kind+'<MultiFab>>(my_geom,bc_fft);'
        yield replace(member+' boundary argument',SOURCE,before,before.replace('bc_fft','other_bc'))
    for path in FILES:
        for name,code in (
            ('macro','#define use_real_bcs false'),('raw string','R"(hidden)";'),
            ('line splice','// hidden \\\n#define first second'),
        ):
            yield fixture(Path(path).name+' '+name,path,texts[path]+'\n'+code+'\n')
    for path,guard in ((UTILS,'ERF_SOLVER_UTILS_H_'),(CLASS,'ERF_H_')):
        yield fixture(Path(path).name+' guard predefinition',path,'#define '+guard+'\n'+texts[path])
        yield fixture(Path(path).name+' duplicate guard macro',path,texts[path]+'\n#define '+guard+'\n')
    yield 'comments removed',texts,False
    yield 'CRLF',{path:text.replace('\n','\r\n') for path,text in original.items()},False
    def whitespace(text):
        return ''.join(part if i%2 else part.replace(',',', /* gap */ ').replace(';',';\n')
                       for i,part in enumerate(re.split('('+STRING+')',text)))
    yield 'token whitespace',{path:whitespace(text) for path,text in texts.items()},False
    yield fixture('quoted unrelated controls',UTILS,texts[UTILS]+'\nconstexpr char diagnostic[] = "} // get_fft_bc return r;";',False)
    yield replace('call comment',SOURCE,CALL,CALL.replace('subdomain,','subdomain /* solve box */, '),False)


def main():
    work=Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-fft-boundary-mutants-'))
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
