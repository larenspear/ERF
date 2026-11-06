#!/usr/bin/env python3
"""Bind the fine-level particle rain repair to the tested lifecycle ordering.

This bounded source contract checks the current simple dispatch and allocation
layout. It is not a general C++ control-flow or preprocessing proof. Numerical
remapping is covered by the compiled SurfacePrecipitationRemap tests.
"""
import argparse
from pathlib import Path
from check_noahmp_precipitation_order import body, compact

LEVELS="Source/ERF_MakeNewLevel.cpp"
WRAPPER="Source/Microphysics/ERF_LagrangianMicrophysics.H"
MODEL="Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoist.H"
INIT="Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistInit.cpp"
FILES=[LEVELS,WRAPPER,MODEL,INIT]

def verify(sources):
    policy=compact(body(sources[LEVELS],r"bool\s+preserve_surface_precipitation\s*\("))
    eulerian=["Kessler","Kessler_NoRain","SAM","SAM_NoIce","SAM_NoPrecip_NoIce",
              "Morrison","Morrison_NoIce","WSM6","WDM6"]
    expected="return"+"||".join("moisture==MoistureType::"+name for name in eulerian)
    expected+="||(moisture==MoistureType::SuperDroplets&&lev>0);"
    assert policy==expected,"review scheme opt-in and particle base-level exclusion"
    for name in ["MakeNewLevelFromCoarse","RemakeLevel"]:
        method=compact(body(sources[LEVELS],r"ERF::"+name+r"\s*\("))
        gate="constboolpreserve_surface_accumulation=preserve_surface_precipitation(solverChoice.moisture_type,lev);"
        assert method.count(gate)==1,"per-level opt-in changed"
        assert method.count("if(preserve_surface_accumulation){") == (1 if name=="MakeNewLevelFromCoarse" else 2),"native restoration gate changed"
        assert "if(false)" not in method and "#if0" not in method,"disabled lifecycle path"
        initialize=method.index("micro->Init(lev,")
        refresh=method.index("qmoist[lev][mvar]=micro->Get_Qmoist_Ptr(lev,mvar);")
        restore=method.index("erf_precipitation::restore_native_surface_accumulation(")
        seed=method.index("erf_precipitation::seed_slm_baseline_from_surface(")
        interpolate=method.index("Interp2DArrays(")
        assert initialize<refresh<restore<interpolate<seed,"native and baseline restoration order changed"
        assert "if(precip[lev]&&preserve_surface_accumulation){erf_precipitation::seed_slm_baseline_from_surface(*precip[lev],micro->Get_Surface_Precip_Accumulation_Ptrs(lev),geom[lev]);}" in method
        assert "if(erf_precipitation::is_native_surface_accumulator(qmoist[lev][mvar],sources)){" in method
        if name=="MakeNewLevelFromCoarse":
            assert "*qmoist[lev][mvar],nullptr,geom[lev],coarse,&geom[lev-1],refRatio(lev-1)" in method
        else:
            capture=method.index("erf_precipitation::capture_surface_accumulation(")
            define=method.index("micro->Define(lev,solverChoice);")
            assert capture<define<initialize,"native snapshot must precede destructive initialization"
            assert "*qmoist[lev][mvar],old_surface_accumulation[mvar].get(),geom[lev]," in method
            assert seed<method.index("erf_precipitation::restore_accumulation(*precip[lev],*old_precip,"),"retained baseline must overlay new-cell seed"
    dispatch=compact(body(sources[WRAPPER],r"void\s+Init\s*\("))
    assert dispatch==("if(lev==0){m_moist_model->Init(cons_in,grids,geom,dt_advance,z_phys_nd,detJ_cc);}"
                      "else{m_moist_model->InitLevel(lev,cons_in);}"),"particle fine-level lifecycle changed"
    getter=compact(body(sources[MODEL],r"Get_Surface_Precip_Accumulation_Ptrs\s*\("))
    assert "sources.rain={m_mic_fab_vars[a_lev][MicVar_SD::rain_accum].get(),amrex::Real(1.0)};" in getter
    initial=compact(body(sources[INIT],r"SuperDropletsMoist::Init\s*\("))
    assert "m_mic_var_map={MicVar_SD::q_t,MicVar_SD::q_v,MicVar_SD::q_c,MicVar_SD::dqcdt,MicVar_SD::q_r,MicVar_SD::q_i,MicVar_SD::q_g,MicVar_SD::rh_w,MicVar_SD::rain_accum,MicVar_SD::graup_accum,MicVar_SD::snow_accum};" in initial,"stable native qmoist slot mapping changed"

def self_test(sources):
    faults=[
        (LEVELS,"moisture == MoistureType::SuperDroplets && lev > 0","moisture == MoistureType::SuperDroplets && lev >= 0"),
        (LEVELS,"moisture == MoistureType::SuperDroplets && lev > 0","moisture == MoistureType::None && lev > 0"),
        (LEVELS,"preserve_surface_precipitation(solverChoice.moisture_type, lev);","preserve_surface_precipitation(solverChoice.moisture_type, 0);"),
        (LEVELS,"if (preserve_surface_accumulation)","if (!preserve_surface_accumulation)"),
        (LEVELS,"old_surface_accumulation[mvar].get(), geom[lev]","nullptr, geom[lev]"),
        (LEVELS,"micro->Get_Qmoist_Ptr(lev,mvar)","micro->Get_Qmoist_Ptr(0,mvar)"),
        (WRAPPER,"m_moist_model->InitLevel(lev, cons_in)","m_moist_model->InitLevel(0, cons_in)"),
        (MODEL,"sources.rain = { m_mic_fab_vars[a_lev][MicVar_SD::rain_accum].get(),","sources.rain = { m_mic_fab_vars[a_lev][MicVar_SD::snow_accum].get(),"),
    ]
    count=0
    for file,old,new in faults:
        assert old in sources[file],old
        changed=dict(sources); changed[file]=changed[file].replace(old,new)
        try: verify(changed)
        except (AssertionError,ValueError): count+=1; continue
        raise AssertionError("accepted source fault: "+old)
    # A call with the right name in the wrong phase must also fail.
    marker="ERF::RemakeLevel"
    for call,anchor in [("erf_precipitation::capture_surface_accumulation(","micro->Init("),
                        ("erf_precipitation::seed_slm_baseline_from_surface(","Interp2DArrays(")]:
        text=sources[LEVELS]; start=text.index(marker); pos=text.index(call,start)
        end=text.index(";",pos)+1; moved=text[pos:end]
        text=text[:pos]+text[end:]
        insert=text.index(anchor,start)
        if "capture" in call: insert=text.index(";",insert)+1
        text=text[:insert]+moved+text[insert:]
        changed=dict(sources); changed[LEVELS]=text
        try: verify(changed)
        except (AssertionError,ValueError): count+=1; continue
        raise AssertionError("accepted reordered source fault: "+call)
    control={key:"// harmless { } comment\n"+text for key,text in sources.items()}
    verify(control)
    return count

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--levels-file",type=Path,help="isolated candidate for the ERF level methods")
    args=parser.parse_args()
    sources={name:(args.levels_file if name==LEVELS and args.levels_file else args.source_root/name).read_text() for name in FILES}
    verify(sources)
    count=self_test(sources)
    print(f"Particle precipitation lifecycle source contract passed; {count} faults rejected")

if __name__=="__main__": main()
