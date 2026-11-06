#!/usr/bin/env python3
"""Bind named ground-mass storage to lifecycle and checkpoint operations.

This is a bounded textual source contract, not a C++ control-flow proof.
Compiled AMReX tests establish the separate remapping and I/O observations.
"""
import argparse
from pathlib import Path
from check_noahmp_precipitation_order import body,compact

MODEL='Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoist.H'
WRAPPER='Source/Microphysics/ERF_LagrangianMicrophysics.H'
BASE='Source/Microphysics/ERF_Microphysics.H'
NULL='Source/Microphysics/Null/ERF_NullMoistLagrangian.H'
LEVELS='Source/ERF_MakeNewLevel.cpp'
IO='Source/IO/ERF_Checkpoint.cpp'
REMAP='Source/Microphysics/ERF_GroundMassRemap.H'
SCHEMA='Source/Microphysics/ERF_GroundMassCheckpoint.H'
INIT='Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistInit.cpp'
FILES=[MODEL,WRAPPER,BASE,NULL,LEVELS,IO,REMAP,SCHEMA,INIT]

def verify(s):
    getter=compact(body(s[MODEL],r'GroundMassAccumulationFields\s+Get_Ground_Mass_Accumulation_Ptrs\s*\('))
    assert ('for(intis=m_istart_sp;is<m_num_species;++is){fields.push_back({"GroundSpecies_"+'
            'amrex::getEnumNameString(m_species[is]),m_mic_fab_vars[lev][s_accum_idx(is,m_istart_sp)].get()});}') in getter
    assert ('for(intia=0;ia<m_num_aerosols;++ia){fields.push_back({"GroundAerosol_"+std::to_string(ia)+"_"+'
            'amrex::getEnumNameString(m_aerosols[ia]),m_mic_fab_vars[lev][a_accum_idx(m_num_nonmoist_sp,ia)].get()});}') in getter
    assert compact(body(s[WRAPPER],r'GroundMassAccumulationFields\s+Get_Ground_Mass_Accumulation_Ptrs\s*\('))=='returnm_moist_model->Get_Ground_Mass_Accumulation_Ptrs(lev);'
    for file in [BASE,NULL]:
        assert compact(body(s[file],r'GroundMassAccumulationFields\s+Get_Ground_Mass_Accumulation_Ptrs\s*\('))=='return{};'
    initial=compact(body(s[INIT],r'SuperDropletsMoist::Init\s*\('))
    assert ('m_mic_var_map={MicVar_SD::q_t,MicVar_SD::q_v,MicVar_SD::q_c,MicVar_SD::dqcdt,'
            'MicVar_SD::q_r,MicVar_SD::q_i,MicVar_SD::q_g,MicVar_SD::rh_w,MicVar_SD::rain_accum,'
            'MicVar_SD::graup_accum,MicVar_SD::snow_accum};') in initial,'qmoist mapping must remain unchanged'
    new=compact(body(s[LEVELS],r'ERF::MakeNewLevelFromCoarse\s*\('))
    new_call=('erf_ground_mass::restore(micro->Get_Ground_Mass_Accumulation_Ptrs(lev),{},geom[lev],'
              'micro->Get_Ground_Mass_Accumulation_Ptrs(lev-1),&geom[lev-1],refRatio(lev-1));')
    assert new_call in new and new.index('micro->Init(')<new.index(new_call)
    remake=compact(body(s[LEVELS],r'ERF::RemakeLevel\s*\('))
    capture='autoold_ground_mass=erf_ground_mass::capture(micro->Get_Ground_Mass_Accumulation_Ptrs(lev),geom[lev]);'
    restore=('erf_ground_mass::restore(micro->Get_Ground_Mass_Accumulation_Ptrs(lev),old_ground_mass,geom[lev],'
             'lev>0?micro->Get_Ground_Mass_Accumulation_Ptrs(lev-1):GroundMassAccumulationFields{},'
             'lev>0?&geom[lev-1]:nullptr,lev>0?refRatio(lev-1):IntVect(1));')
    assert capture in remake and restore in remake
    assert remake.index(capture)<remake.index('micro->Define(')<remake.index('micro->Init(')<remake.index(restore)
    for method in [new,remake]:assert 'if(false)' not in method and '#if0' not in method
    remap=compact(body(s[REMAP],r'inline\s+void\s+restore\s*\([\s\S]*?amrex::IntVect\(1\)\)'))
    assert 'constIntVectratio(refinement[0],refinement[1],1);' in remap
    assert 'constRealchildren=Real(rx)*Real(ry);' in remap
    assert 'to(i,j,0)=from(amrex::coarsen(i,rx),amrex::coarsen(j,ry),0)/children;' in remap
    assert remap.index('/children;')<remap.index('slice.values->ParallelCopy(*retained[n].surface,')<remap.index('to(i,j,bottom)=from(i,j,0);')
    assert 'coarse_fields[n].name==fields[n].name' in remap and 'retained[n].name==fields[n].name' in remap
    for name in ['WriteCheckpointFile','ReadCheckpointFile']:
        method=compact(body(s[IO],r'ERF::'+name+r'\s*\('))
        assert 'for(intlev=0;lev<=finest_level;++lev)' in method
        assert 'constautoground_mass=micro->Get_Ground_Mass_Accumulation_Ptrs(lev);' in method
        assert 'for(constauto&field:ground_mass){' in method
        if name.startswith('Write'):
            assert 'erf_ground_mass::write_manifest(MultiFabFileFullPrefix(lev,checkpointname,"Level_","GroundMassFields"),ground_mass);' in method
            assert 'VisMF::Write(*field.mass,MultiFabFileFullPrefix(lev,checkpointname,"Level_",field.name));' in method
        else:
            assert 'constautomanifest_path=MultiFabFileFullPrefix(lev,restart_chkfile,"Level_","GroundMassFields");' in method
            assert 'if(ground_mass.empty()&&!FileExists(manifest_path)){' in method
            assert ('verified_legacy_empty=erf_ground_mass::legacy_empty_is_compatible('
                    'restart_chkfile+"/"+particle_model.getName()+"/Header",particles->varNames());') in method
            assert 'erf_ground_mass::validate_manifest(manifest_path,ground_mass,verified_legacy_empty);' in method
            assert 'if(!FileExists(path+"_H")){Abort(' in method
            assert 'read_checkpoint_mf(*field.mass,path,lev,0,0,1,field.mass->nGrowVect(),true);' in method
            assert method.index('MakeNewLevelFromScratch(')<method.index('erf_ground_mass::validate_manifest(')<method.index('read_checkpoint_mf(*field.mass,')
    schema=compact(s[SCHEMA])
    assert 'std::string(data.dataPtr(),data.size()-1)!=manifest(fields)' in schema
    assert 'AMREX_ALWAYS_ASSERT(names.insert(field.name).second);' in schema
    assert "(c>='A'&&c<='Z')||(c>='a'&&c<='z')||(c>='0'&&c<='9')||c=='_'" in schema
    assert 'if(!amrex::FileExists(path)){if(fields.empty()&&verified_legacy_empty)return;amrex::Abort(' in schema
    assert 'if(fields.empty()&&verified_legacy_empty)return;' in schema
    assert 'name!=expected' in schema and 'total_particles==num_particles' in schema

def self_test(s):
    faults=[
        (MODEL,'s_accum_idx(is,m_istart_sp)','s_qv_idx(is,m_istart_sp)'),
        (MODEL,'a_accum_idx(m_num_nonmoist_sp,ia)','a_accum_idx(m_num_nonmoist_sp,0)'),
        (MODEL,'std::to_string(ia)','std::to_string(0)'),
        (MODEL,'"GroundSpecies_"','"GroundAerosol_"'),
        (WRAPPER,'Get_Ground_Mass_Accumulation_Ptrs(lev);','Get_Ground_Mass_Accumulation_Ptrs(0);'),
        (LEVELS,'Get_Ground_Mass_Accumulation_Ptrs(lev-1)','Get_Ground_Mass_Accumulation_Ptrs(lev)'),
        (LEVELS,'old_ground_mass, geom[lev]','{}, geom[lev]'),
        (LEVELS,'auto old_ground_mass = erf_ground_mass::capture(','auto old_ground_mass = other_capture('),
        (REMAP,'Real(rx)*Real(ry)','Real(rx)*Real(ry)*Real(refinement[2])'),
        (REMAP,'/children;',';'),
        (REMAP,'*retained[n].surface','*retained[0].surface'),
        (IO,'VisMF::Write(*field.mass,','VisMF::Write(*ground_mass[0].mass,'),
        (IO,'read_checkpoint_mf(*field.mass, path, lev','read_checkpoint_mf(*field.mass, path, 0'),
        (IO,'erf_ground_mass::validate_manifest(','erf_ground_mass::other_validate('),
        (IO,'!FileExists(path + "_H")','FileExists(path + "_H")'),
        (SCHEMA,'std::string(data.dataPtr(), data.size()-1)','std::string(data.dataPtr())'),
        (SCHEMA,'names.insert(field.name).second','true'),
        (IO,'ground_mass.empty() && !FileExists(manifest_path)','!FileExists(manifest_path)'),
        (IO,'particles->varNames()','amrex::Vector<std::string>{}'),
        (IO,'ground_mass, verified_legacy_empty);','ground_mass, true);'),
        (SCHEMA,'fields.empty() && verified_legacy_empty','verified_legacy_empty'),
        (SCHEMA,'name != expected','false'),
        (SCHEMA,'total_particles == num_particles','true'),
    ]
    for file,old,new in faults:
        assert old in s[file],old
        modified=dict(s);modified[file]=s[file].replace(old,new)
        try:verify(modified)
        except (AssertionError,ValueError):continue
        raise AssertionError('accepted source fault: '+old)
    verify({file:'// harmless { } comment\n'+text for file,text in s.items()})
    return len(faults)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--overlay',type=Path)
    args=parser.parse_args()
    sources={name:((args.overlay/name) if args.overlay and (args.overlay/name).exists()
                   else args.source_root/name).read_text() for name in FILES}
    verify(sources)
    print(f'Ground mass lifecycle source contract passed; {self_test(sources)} faults rejected')

if __name__=='__main__':main()
