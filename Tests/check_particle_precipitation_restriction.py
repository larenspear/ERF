#!/usr/bin/env python3
"""Bind particle precipitation slots to the compiled surface restriction helper.

This bounded source check is not a general C++ control-flow proof. The helper's
surface coverage, arithmetic, and vertical independence have compiled tests.
"""
import argparse
from pathlib import Path
from check_noahmp_precipitation_order import body, compact

UTILS="Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistUtils.cpp"

def verify(source):
    average=compact(body(source,r"SuperDropletsMoist::AverageDownMicroVars\s*\("))
    selection=("constboolsurface_depth=v==MicVar_SD::rain_accum||"
               "v==MicVar_SD::snow_accum||v==MicVar_SD::graup_accum;")
    dispatch=("if(surface_depth){AMREX_ALWAYS_ASSERT(m_super_droplets!=nullptr);"
              "erf_precipitation::average_down_surface_accumulation("
              "*m_mic_fab_vars[lev][v],*m_mic_fab_vars[lev-1][v],"
              "m_super_droplets->Geom(lev),m_super_droplets->Geom(lev-1),rr);"
              "}else{average_down(*m_mic_fab_vars[lev][v],*m_mic_fab_vars[lev-1][v],"
              "0,m_mic_fab_vars[lev][v]->nComp(),rr);}")
    assert selection in average and dispatch in average,"surface slots must use horizontal restriction; volume slots retain 3D averaging"
    mass_selection=("boolsurface_mass=false;"
                    "for(intis=m_istart_sp;is<m_num_species;++is){"
                    "surface_mass=surface_mass||v==s_accum_idx(is,m_istart_sp);}"
                    "for(intia=0;ia<m_num_aerosols;++ia){"
                    "surface_mass=surface_mass||v==a_accum_idx(m_num_nonmoist_sp,ia);}")
    mass_dispatch=("if(surface_mass){AMREX_ALWAYS_ASSERT(m_super_droplets!=nullptr);"
                   "erf_precipitation::sum_down_surface_mass("
                   "*m_mic_fab_vars[lev][v],*m_mic_fab_vars[lev-1][v],"
                   "m_super_droplets->Geom(lev),m_super_droplets->Geom(lev-1),rr);"
                   "}else")
    assert selection+mass_selection+mass_dispatch+dispatch in average, "ground mass slots must sum horizontal children"
    assert "if(false)" not in average and "#if0" not in average
    assert "for(intlev=finest_level;lev>=1;--lev)" in average
    # The production accumulation methods establish the surface-depth layout.
    for name in ["rainAccumulation","snowAccumulation"]:
        method=compact(body(source,r"SuperDropletsMoist::"+name+r"\s*\("))
        assert "intk_lo=domain.smallEnd(2);" in method
        assert "if(k==k_lo){" in method
        assert "*dt/mat_density" in method,"review accumulated quantity units"

    # Mass flux times dt and horizontal cell area gives kg per ground cell.
    for name, slot in [("speciesAccumulation","s_accum_idx(is,m_istart_sp)"),
                       ("aerosolAccumulation","a_accum_idx(m_num_nonmoist_sp,ia)")]:
        method=compact(body(source,r"SuperDropletsMoist::"+name+r"\s*\("))
        assert "intk_lo=domain.smallEnd(2);" in method
        assert "if(k==k_lo){autoaccum=std::max(Real(0),-zflux_arr(i,j,k)*dt*dx[0]*dx[1]);accum_arr(i,j,k)+=accum;}" in method
        assert "m_mic_fab_vars[lev]["+slot+"]->array(mfi)" in method


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--utils-file",type=Path,help="isolated candidate override")
    args=parser.parse_args()
    source=(args.utils_file or args.source_root/UTILS).read_text()
    verify(source)
    faults=[("v == MicVar_SD::rain_accum","v == MicVar_SD::rho"),
            ("v == MicVar_SD::snow_accum","v == MicVar_SD::q_s"),
            ("if (surface_depth)","if (!surface_depth)"),
            ("m_super_droplets->Geom(lev-1), rr","m_super_droplets->Geom(lev), rr"),
            ("lev >= 1; --lev","lev > 1; --lev"),
            ("if (k == k_lo)","if (k >= k_lo)"),
            ("v == s_accum_idx(is, m_istart_sp)","v == s_qv_idx(is, m_istart_sp)"),
            ("v == a_accum_idx(m_num_nonmoist_sp, ia)","v == a_accum_idx(m_num_nonmoist_sp, 0)"),
            ("is < m_num_species; ++is","is < m_num_species-1; ++is"),
            ("ia < m_num_aerosols; ++ia","ia < m_num_aerosols-1; ++ia"),
            ("if (surface_mass)","if (!surface_mass)"),
            ("erf_precipitation::sum_down_surface_mass(","erf_precipitation::average_down_surface_accumulation("),
            ("*dt*dx[0]*dx[1]","*dt") ]
    for old,new in faults:
        assert old in source,old
        try: verify(source.replace(old,new))
        except (AssertionError,ValueError): continue
        raise AssertionError("accepted source fault: "+old)
    verify("// harmless { }\n"+source)
    print(f"Particle precipitation restriction source contract passed; {len(faults)} faults rejected")

if __name__=="__main__": main()
