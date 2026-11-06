#!/usr/bin/env python3
"""Check SLM/native restriction wiring and reject representative source faults.

This is a bounded textual contract. Numerical tests and full runtime comparisons
separately check pending intervals, units, history retention, and land behavior.
"""
import argparse
from pathlib import Path
from check_noahmp_precipitation_order import body, compact

DRIVER="Source/ERF.cpp"
STEP="Source/TimeIntegration/ERF_TimeStep.cpp"
PLOT="Source/IO/ERF_Plotfile.cpp"
HELPER="Source/Microphysics/ERF_PrecipitationHistory.H"

def verify(sources):
    wrapper=compact(body(sources[DRIVER],r"ERF::AverageDownMicrophysics\s*\("))
    assert "constboolhas_land=solverChoice.lsm_type==LandSurfaceType::SLM;" in wrapper
    assert "constintfirst=surface_only?coarsest_lev:0;" in wrapper
    capture="before[clev]=erf_precipitation::capture_slm_precipitation(micro->Get_Surface_Precip_Accumulation_Ptrs(clev),geom[clev]);"
    native="micro->AverageDownMicroVars(finest_lev);"
    rebase="erf_precipitation::rebase_slm_precipitation(*precip[clev],before[clev],after,geom[clev]);"
    assert wrapper.index(capture)<wrapper.index(native)<wrapper.index(rebase)
    assert wrapper.count("for(intclev=first;clev<finest_lev;++clev)")==2
    narrow=compact(body(sources[DRIVER],r"if\s*\(surface_only\)"))
    assert narrow.startswith("for(intflev=finest_lev;flev>first;--flev){")
    assert "is_native_surface_accumulator(qmoist[flev][mvar],fine)" in narrow
    assert "is_native_surface_accumulator(qmoist[flev-1][mvar],coarse)" in narrow
    assert "average_down_surface_accumulation(*qmoist[flev][mvar],*qmoist[flev-1][mvar],geom[flev],geom[flev-1],refRatio(flev-1));" in narrow
    assert "AverageDownMicroVars" not in narrow and "AverageDownMoistStateTo" not in narrow
    assert "if(solverChoice.lsm_type==LandSurfaceType::SLM&&precip[clev]){"+rebase+"}" in wrapper
    assert "Rebase_Precipitation" not in wrapper
    assert sum(compact(text).count("micro->AverageDownMicroVars(") for text in sources.values())==1

    timestep=compact(body(sources[STEP],r"ERF::timeStep\s*\("))
    gate="if(solverChoice.lsm_type==LandSurfaceType::None||solverChoice.lsm_type==LandSurfaceType::SLM){AverageDownMicrophysics(finest_level,true,lev);}"
    assert gate in timestep
    assert timestep.index(gate)<timestep.index("regrid(lev,static_cast<Real>(time));")
    assert timestep.count("AverageDownMicrophysics(")==2
    init=compact(body(sources[DRIVER],r"ERF::InitData_post\s*\("))
    assert ("if(restart_chkfile.empty()&&solverChoice.coupling_type==CouplingType::TwoWay&&"
            "Microphysics::modelType(solverChoice.moisture_type)==MoistureModelType::Lagrangian&&"
            "finest_level>=1){AverageDownMicrophysics(finest_level);") in init
    plot=compact(body(sources[PLOT],r"ERF::BuildPlot3DScratch\s*\("))
    assert ("if(sync_solution&&Microphysics::modelType(solverChoice.moisture_type)==MoistureModelType::Lagrangian&&"
            "solverChoice.coupling_type==CouplingType::TwoWay&&finest_level>=1){AverageDownMicrophysics(finest_level);") in plot
    capture_body=compact(body(sources[HELPER],r"capture_slm_precipitation\s*\("))
    assert "surface_precip_has_total_source(sources)?sources.total:sources.rain" in capture_body
    assert "saved.values=capture_surface_accumulation(*selected.accum,geometry);" in capture_body
    assert "saved.factor=selected.native_to_kg_m2;" in capture_body
    numeric=compact(body(sources[HELPER],r"inline\s+void\s+rebase_precipitation_snapshot\s*\("))
    assert "old_values.ParallelCopy(before,0,0,1,0,0);" in numeric
    assert "new_values.ParallelCopy(after,0,0,1,0,0);" in numeric
    assert ("if(prior(i,j,k)!=current(i,j,k)){constRealpending=prior(i,j,k)*factor-consumed(i,j,k);"
            "consumed(i,j,k)=current(i,j,k)*factor-pending;}") in numeric
    slm=compact(body(sources[HELPER],r"rebase_slm_precipitation\s*\("))
    assert "surface_precip_has_total_source(after)?after.total:after.rain" in slm
    assert "AMREX_ALWAYS_ASSERT(before.factor==new_source.native_to_kg_m2);" in slm
    assert "rebase_precipitation_snapshot(baseline,*before.values,*new_surface,before.factor);" in slm

def self_test(sources):
    faults=[
        (DRIVER,"lsm_type == LandSurfaceType::SLM;","lsm_type != LandSurfaceType::None;"),
        (DRIVER,"surface_only ? coarsest_lev : 0","surface_only ? 0 : coarsest_lev"),
        (DRIVER,"clev < finest_lev; ++clev","clev < finest_lev-1; ++clev"),
        (DRIVER,"flev = finest_lev; flev > first; --flev","flev = first+1; flev <= finest_lev; ++flev"),
        (DRIVER,"is_native_surface_accumulator(qmoist[flev][mvar], fine)","is_native_surface_accumulator(qmoist[flev][0], fine)"),
        (DRIVER,"qmoist[flev-1][mvar], coarse","qmoist[flev-1][0], coarse"),
        (DRIVER,"*qmoist[flev][mvar], *qmoist[flev-1][mvar]","*qmoist[flev-1][mvar], *qmoist[flev][mvar]"),
        (DRIVER,"refRatio(flev-1)","refRatio(0)"),
        (DRIVER,"if (surface_only) {","if (surface_only) { micro->AverageDownMicroVars(finest_lev);"),
        (DRIVER,"*precip[clev], before[clev], after, geom[clev]","*precip[clev], before[0], after, geom[clev]"),
        (DRIVER,"micro->AverageDownMicroVars(finest_lev);","micro->AverageDownMicroVars(0);"),
        (DRIVER,"if (restart_chkfile.empty() &&\n            solverChoice.coupling_type == CouplingType::TwoWay", "if (true &&\n            solverChoice.coupling_type == CouplingType::TwoWay"),
        (STEP,"LandSurfaceType::None ||","LandSurfaceType::NOAHMP ||"),
        (STEP,"AverageDownMicrophysics(finest_level, true, lev);","AverageDownMicrophysics(finest_level);"),
        (PLOT,"if (sync_solution","if (true"),
        (PLOT,"AverageDownMicrophysics(finest_level);","micro->AverageDownMicroVars(finest_level);"),
        (HELPER,"sources.total : sources.rain","sources.rain : sources.total"),
        (HELPER,"saved.factor = selected.native_to_kg_m2;","saved.factor = 1;"),
        (HELPER,"capture_surface_accumulation(*selected.accum, geometry)","capture_surface_accumulation(*sources.snow.accum, geometry)"),
        (HELPER,"ParallelCopy(before, 0, 0, 1, 0, 0)","ParallelCopy(after, 0, 0, 1, 0, 0)"),
        (HELPER,"prior(i,j,k) != current(i,j,k)","true"),
        (HELPER,"prior(i,j,k)*factor - consumed(i,j,k)","prior(i,j,k) - consumed(i,j,k)"),
        (HELPER,"current(i,j,k)*factor - pending","current(i,j,k)*factor"),
        (HELPER,"after.total : after.rain","after.rain : after.snow"),
        (HELPER,"before.factor == new_source.native_to_kg_m2","true"),
        (HELPER,"*new_surface, before.factor","*new_surface, 1"),
    ]
    for file,old,new in faults:
        assert old in sources[file],old
        changed=dict(sources);changed[file]=changed[file].replace(old,new,1)
        try: verify(changed)
        except (AssertionError,ValueError): continue
        raise AssertionError("accepted source fault: "+old)
    verify({file:"// harmless { }\n"+text for file,text in sources.items()})
    return len(faults)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overlay",type=Path)
    args=parser.parse_args()
    sources={file:((args.overlay / file) if args.overlay and (args.overlay / file).exists() else args.source_root / file).read_text() for file in (DRIVER,STEP,PLOT,HELPER)}
    verify(sources)
    print(f"Precipitation history restriction contract passed; {self_test(sources)} faults rejected")

if __name__=="__main__": main()
