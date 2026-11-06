#!/usr/bin/env python3
"""Bind particle precipitation checkpoint fields to the tested per-level I/O.

This bounded source contract checks the current simple dispatch, slot mapping,
and checkpoint ordering. It is not a general C++ control-flow proof. Real AMReX
VisMF counterexamples and round trips are recorded in the verification evidence.
"""
import argparse
from pathlib import Path
import re
from check_noahmp_precipitation_order import TOKENS, body, compact, uncomment

MODEL='Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoist.H'
WRAPPER='Source/Microphysics/ERF_LagrangianMicrophysics.H'
CHECKPOINT='Source/IO/ERF_Checkpoint.cpp'
INIT='Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistInit.cpp'
UTILS='Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistUtils.cpp'
FILES=[MODEL,WRAPPER,CHECKPOINT,INIT,UTILS]

def level_block(text):
    """Find the complete all-level loop whose direct statements include qmoist."""
    text=uncomment(text)
    pattern=r'for\s*\(\s*int\s+lev\s*=\s*0\s*;\s*lev\s*<=\s*finest_level\s*;\s*\+\+lev\s*\)'
    blocks=[body(text[m.start():],r'\A'+pattern) for m in re.finditer(pattern,text)]
    matches=[b for b in blocks if 'Get_Qmoist_Restart_Vars' in b]
    assert len(matches)==1,'checkpoint qmoist I/O must run on every level'
    result=matches[0]
    offset=result.index('std::vector<int> qmoist_indices;')
    depth=sum(1 if m[0]=='{' else -1 if m[0]=='}' else 0
              for m in TOKENS.finditer(result[:offset]))
    assert depth==0,'qmoist restart dispatch must not have an extra conditional gate'
    return compact(result)

def verify(sources):
    fields=compact(body(sources[MODEL],r'void\s+Qmoist_Restart_Vars\s*\('))
    assert fields==('a_idx={8};a_names={"RainAccum"};if(m_with_ice){'
                    'a_idx.push_back(10);a_names.push_back("SnowAccum");}'), \
        'review active precipitation restart slots and payload names'
    wrapper=compact(body(sources[WRAPPER],r'void\s+Get_Qmoist_Restart_Vars\s*\('))
    assert wrapper=='m_moist_model->Qmoist_Restart_Vars(a_sc,a_idx,a_names);', \
        'single particle model must forward restart fields for every level'
    initial=compact(body(sources[INIT],r'SuperDropletsMoist::Init\s*\('))
    mapping=('m_mic_var_map={MicVar_SD::q_t,MicVar_SD::q_v,MicVar_SD::q_c,'
             'MicVar_SD::dqcdt,MicVar_SD::q_r,MicVar_SD::q_i,MicVar_SD::q_g,'
             'MicVar_SD::rh_w,MicVar_SD::rain_accum,MicVar_SD::graup_accum,MicVar_SD::snow_accum};')
    assert mapping in initial,'qmoist/native slot mapping changed'
    pointer=compact(body(sources[MODEL],r'Qmoist_Ptr\s*\('))
    assert 'constintlev=m_current_lev;' in pointer
    assert 'returnm_mic_fab_vars[lev][m_mic_var_map[a_idx]].get();' in pointer
    common=('std::vector<int>qmoist_indices;std::vector<std::string>qmoist_names;'
            'micro->Get_Qmoist_Restart_Vars(lev,solverChoice,qmoist_indices,qmoist_names);'
            'intqmoist_nvar=static_cast<int>(qmoist_indices.size());'
            'for(intvar=0;var<qmoist_nvar;var++){constintncomp=1;'
            'IntVectng_moist=qmoist[lev][qmoist_indices[var]]->nGrowVect();')
    write=body(sources[CHECKPOINT],r'ERF::WriteCheckpointFile\s*\(')
    writer=level_block(write)
    assert common+('MultiFabmoist_vars(grids[lev],dmap[lev],ncomp,ng_moist);'
        'MultiFab::Copy(moist_vars,*(qmoist[lev][qmoist_indices[var]]),0,0,ncomp,ng_moist);'
        'VisMF::Write(moist_vars,amrex::MultiFabFileFullPrefix(lev,checkpointname,"Level_",qmoist_names[var]));}') in writer
    read=body(sources[CHECKPOINT],r'ERF::ReadCheckpointFile\s*\(')
    reader=level_block(read)
    assert common+('conststd::stringmoist_file=amrex::MultiFabFileFullPrefix(lev,restart_chkfile,"Level_",qmoist_names[var]);'
        'if(solverChoice.moisture_type==MoistureType::SuperDroplets&&!amrex::FileExists(moist_file+"_H")){'
        'amrex::Abort("SuperDropletscheckpointismissingcumulativeprecipitationfield\'"+'
        'moist_file+"\'.Oldercheckpointsdidnotsavethishistory;"'
        '"theparticlestatecannotrecoveritforexactcontinuation.");}'
        'read_checkpoint_mf(*(qmoist[lev][qmoist_indices[var]]),moist_file,lev,0,0,ncomp,ng_moist,true);}') in reader, \
        'required active payloads must use the remapping reader with ghosts'
    read=compact(read)
    assert read.index('MakeNewLevelFromScratch(')<read.index(common)<read.index('"SlmPrecip"'), \
        'allocation, native restore, and SLM baseline ordering changed'
    update=compact(body(sources[UTILS],r'SuperDropletsMoist::Update_Micro_Vars\s*\('))
    assert update=='BL_PROFILE("SuperDropletsMoist::Update_Micro_Vars()");Copy_State_to_Micro(a_cons_vars);'
    copy=body(sources[UTILS],r'SuperDropletsMoist::Copy_State_to_Micro\s*\(')
    finish=body(sources[INIT],r'SuperDropletsMoist::FinishInit\s*\(')
    for name in ['rain_accum','snow_accum','rainAccumulation','snowAccumulation']:
        assert name not in copy+finish,'review post-restart accumulator changes'

def self_test(sources):
    faults=[
        (MODEL,'a_idx = {8};','a_idx = {9};'),
        (MODEL,'a_idx.push_back(10);','a_idx.push_back(9);'),
        (MODEL,'a_names = {"RainAccum"};','a_names = {"SnowAccum"};'),
        (MODEL,'if (m_with_ice) {\n                a_idx.push_back(10);',
               'if (!m_with_ice) {\n                a_idx.push_back(10);'),
        (MODEL,'const int lev = m_current_lev;','const int lev = 0;'),
        (WRAPPER,'m_moist_model->Qmoist_Restart_Vars(a_sc, a_idx, a_names);',
                 'if (a_lev == 0) m_moist_model->Qmoist_Restart_Vars(a_sc, a_idx, a_names);'),
        (INIT,'MicVar_SD::rain_accum,','MicVar_SD::snow_accum,'),
        (CHECKPOINT,'micro->Get_Qmoist_Restart_Vars(lev, solverChoice,','micro->Get_Qmoist_Restart_Vars(0, solverChoice,'),
        (CHECKPOINT,'qmoist[lev][qmoist_indices[var]]','qmoist[0][qmoist_indices[var]]'),
        (CHECKPOINT,'moist_file + "_H"','moist_file + "_missing"'),
        (CHECKPOINT,'!amrex::FileExists(moist_file','amrex::FileExists(moist_file'),
        (CHECKPOINT,'lev, 0, 0, ncomp, ng_moist, /*src_ghosts=*/true);',
                    'lev, 0, 0, ncomp, ng_moist, /*src_ghosts=*/false);'),
    ]
    for file,old,new in faults:
        assert old in sources[file],old
        changed=dict(sources); changed[file]=changed[file].replace(old,new)
        try: verify(changed)
        except (AssertionError,ValueError): continue
        raise AssertionError('accepted source fault: '+old)
    verify({name:'// harmless { } comment\n'+text for name,text in sources.items()})
    return len(faults)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--overlay',type=Path,help='isolated candidate files over source root')
    args=parser.parse_args()
    sources={name:((args.overlay/name) if args.overlay and (args.overlay/name).exists()
                   else args.source_root/name).read_text() for name in FILES}
    verify(sources)
    print(f'Particle precipitation restart source contract passed; {self_test(sources)} faults rejected')

if __name__=='__main__': main()
