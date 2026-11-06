#!/usr/bin/env python3
"""Bind particle ice-depth accumulation to water-equivalent typed consumers.

Bounded source contract for the current simple producer/getter/consumer and
AMR dispatches; not a general C++ control-flow proof. Exact source-linked CPU
AMReX accumulation, Noah staging, output, and AMR probes supply numerical evidence.
"""
import argparse
from pathlib import Path
from check_noahmp_precipitation_order import body, compact
from check_particle_precipitation_lifecycle import verify as verify_lifecycle

MODEL='Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoist.H'
UTILS='Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistUtils.cpp'
COLLECT='Source/LandSurfaceModel/Noah-MP/ERF_NOAHMP_Precip.cpp'
STAGE='Source/LandSurfaceModel/Noah-MP/ERF_NOAHMP_Advance.cpp'
SLM='Source/TimeIntegration/ERF_AdvanceLSM.cpp'
REMAP='Source/Microphysics/ERF_SurfacePrecipitationRemap.H'
OUTPUT='Source/IO/ERF_Plotfile2DPrecip.cpp'
FILES=[MODEL,UTILS,COLLECT,STAGE,SLM,REMAP,OUTPUT,
       'Source/ERF_MakeNewLevel.cpp','Source/Microphysics/ERF_LagrangianMicrophysics.H',
       'Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistInit.cpp']

def verify(sources):
    getter=compact(body(sources[MODEL],r'Get_Surface_Precip_Accumulation_Ptrs\s*\('))
    assert getter==(
        'SurfacePrecipAccumulationSourcessources;'
        'if(a_lev>=0&&a_lev<static_cast<int>(m_mic_fab_vars.size())){'
        'sources.rain={m_mic_fab_vars[a_lev][MicVar_SD::rain_accum].get(),amrex::Real(1.0)};'
        'if(m_with_ice){constauto&ice_material=m_super_droplets->getSpeciesMaterial(Species::Name::ice);'
        'sources.snow={m_mic_fab_vars[a_lev][MicVar_SD::snow_accum].get(),ice_material.m_density/amrex::Real(1000.0)};}}'
        'returnsources;'), 'review typed per-level ice field and native-to-water-equivalent units'
    accumulation=compact(body(sources[UTILS],r'SuperDropletsMoist::snowAccumulation\s*\('))
    assert 'if(m_with_ice){' in accumulation
    for required in [
        'constintlev=m_current_lev;',
        'auto&vapour_mat=m_super_droplets->getSpeciesMaterial(Species::Name::ice);',
        'automat_density=vapour_mat.m_density;',
        'm_super_droplets->speciesMassFlux(mf_zflux,a_z_phys_nd,lev,m_idx_i,2);',
        'constArray4<Real>&snow_accum_arr=m_mic_fab_vars[lev][MicVar_SD::snow_accum]->array(mfi);',
        'if(k==k_lo){autosnow_accum=std::max(Real(0),-zflux_arr(i,j,k)*dt/mat_density);'
        'snow_accum_arr(i,j,k)+=(snow_accum*Real(1000));}']:
        assert required in accumulation,'native ice-depth producer changed'
    update=compact(body(sources[UTILS],r'SuperDropletsMoist::Update_State_Vars\s*\('))
    assert 'rainAccumulation(a_z_phys_nd);snowAccumulation(a_z_phys_nd);' in update
    collect=compact(body(sources[COLLECT],r'NOAHMP::collect_precip_sources\s*\('))
    assert 'constSurfacePrecipAccumulationSourceslot_src[NoahmpPrecipSlot::NumSlots]={precip_sources.total,precip_sources.rain,precip_sources.snow,precip_sources.graupel,precip_sources.hail};' in collect
    assert 'p.factor[s]=slot_src[s].native_to_kg_m2;' in collect
    stage=compact(body(sources[STAGE],r'NOAHMP::stage_forcing\s*\('))
    assert 'accum_fac[s]=precip.factor[s];' in stage
    assert '(accum_now[s](i,j,kklo)-accum_prv[s](i,j,kklo))*accum_fac[s]' in stage
    assert 'dsnow=dd[NoahmpPrecipSlot::snow];' in stage
    assert 'drain=slot_present[NoahmpPrecipSlot::total]?dd[NoahmpPrecipSlot::total]:(dd[NoahmpPrecipSlot::rain]+dfroz);' in stage
    assert 'noahmpio->MP_SNOW(i,j)=dsnow_h;' in stage
    slm=compact(body(sources[SLM],r'void\s+update_slm_precipitation\s*\('))
    assert 'if(rain_arr){accumulation=rain_arr(i,j,k)*rain_factor;}' in slm
    assert 'sources.snow' not in slm,'SLM must retain its liquid-rain contract'
    selected=compact(body(sources[REMAP],r'bool\s+is_native_surface_accumulator\s*\('))
    assert 'field==sources.snow.accum' in selected,'AMR must select the exported frozen history'
    output=compact(body(sources[OUTPUT],r'fill_precipitation_accumulations\s*\('))
    assert 'constamrex::Realsnow_factor=sources.snow.native_to_kg_m2;' in output
    assert 'has_snow_source?snow_arr(i,j,k,0)*snow_factor:amrex::Real(0.0);' in output
    assert 'constamrex::Realfrozen=snow+graupel+hail;' in output
    verify_lifecycle(sources)

def self_test(sources):
    faults=[
        (MODEL,'ice_material.m_density / amrex::Real(1000.0)','amrex::Real(1.0)'),
        (MODEL,'ice_material.m_density / amrex::Real(1000.0)','amrex::Real(1000.0) / ice_material.m_density'),
        (MODEL,'getSpeciesMaterial(Species::Name::ice)','getSpeciesMaterial(Species::Name::H2O)'),
        (MODEL,'if (m_with_ice) {\n                    // snow_accum','if (!m_with_ice) {\n                    // snow_accum'),
        (MODEL,'sources.snow = { m_mic_fab_vars[a_lev][MicVar_SD::snow_accum].get(),',
               'sources.snow = { m_mic_fab_vars[0][MicVar_SD::snow_accum].get(),'),
        (MODEL,'sources.snow = {','sources.total = {'),
        (UTILS,'-zflux_arr(i,j,k)*dt/mat_density','-zflux_arr(i,j,k)*dt/Real(1000)'),
        (UTILS,'snow_accum * Real(1000)','snow_accum * Real(1)'),
        (COLLECT,'p.factor[s] = slot_src[s].native_to_kg_m2;','p.factor[s] = 1;'),
        (STAGE,'dsnow  = dd[NoahmpPrecipSlot::snow];','dsnow  = dd[NoahmpPrecipSlot::rain];'),
        (REMAP,'field == sources.snow.accum','field == sources.rain.accum'),
        (OUTPUT,'snow_factor = sources.snow.native_to_kg_m2','snow_factor = 1'),
    ]
    for file,old,new in faults:
        assert old in sources[file],old
        changed=dict(sources);changed[file]=changed[file].replace(old,new)
        try: verify(changed)
        except (AssertionError,ValueError): continue
        raise AssertionError('accepted source fault: '+old)
    verify({name:'// harmless { } comment\n'+text for name,text in sources.items()})
    return len(faults)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--overlay',type=Path)
    args=parser.parse_args()
    sources={name:((args.overlay/name) if args.overlay and (args.overlay/name).exists()
                  else args.source_root/name).read_text() for name in FILES}
    verify(sources)
    print(f'Particle frozen precipitation source contract passed; {self_test(sources)} faults rejected')

if __name__=='__main__': main()
