#!/usr/bin/env python3
"""Bind the reviewed particle restart initialization dispatch to source.

This is a bounded source contract, not a general C++ control-flow proof.
Full-particle tests separately compare checkpoint state and particle attributes.
"""
import argparse
from pathlib import Path
import re
from check_noahmp_precipitation_order import body, compact

DRIVER = "Source/ERF.cpp"
UTILS = "Source/Microphysics/SuperDropletsMoist/ERF_SuperDropletsMoistUtils.cpp"
WRAPPER = "Source/Microphysics/ERF_LagrangianMicrophysics.H"
FILES = (DRIVER, UTILS, WRAPPER)
LOOP = """for (int lev = 0; lev <= finest_level; ++lev) {
    const amrex::MultiFab* base = solverChoice.anelastic[lev] ? &base_state[lev] : nullptr;
    micro->Update_Micro_Vars_Lev(lev, vars_new[lev][Vars::cons], base);
    if (restart_chkfile.empty() ||
        Microphysics::modelType(solverChoice.moisture_type) != MoistureModelType::Lagrangian) {
        micro->FinishInit(lev, vars_new[lev][Vars::cons], z_phys_nd);
    }
}"""


def verify(sources):
    driver = compact(body(sources[DRIVER], r"ERF::InitData_post\s*\("))
    assert compact(LOOP) in driver, "all levels must refresh state; fresh initialization must be gated"
    assert driver.count("micro->FinishInit(") == 1, "unexpected additional initializer"
    assert driver.count("micro->Update_Micro_Vars_Lev(") == 1, "unexpected additional state refresh"
    assert "if(restart_chkfile.empty()&&solverChoice.coupling_type==CouplingType::TwoWay&&" in driver, "restart must not acquire fresh-start restriction"
    wrapper = compact(body(sources[WRAPPER], r"void\s+Update_Micro_Vars_Lev\s*\("))
    assert wrapper == "m_moist_model->SetCurrentLevel(lev);m_moist_model->Update_Micro_Vars(cons_in,base_state);"
    update = compact(body(sources[UTILS], r"SuperDropletsMoist::Update_Micro_Vars\s*\("))
    assert update == 'BL_PROFILE("SuperDropletsMoist::Update_Micro_Vars()");Copy_State_to_Micro(a_cons_vars);'
    copy = compact(body(sources[UTILS], r"SuperDropletsMoist::Copy_State_to_Micro\s*\("))
    for name, comp in (("q_v", 1), ("q_c", 2), ("q_i", 3), ("q_r", 4), ("q_s", 5), ("q_g", 6)):
        assert f"{name}_arr(i,j,k)=states_arr(i,j,k,RhoQ{comp}_comp)*inv_rho;" in copy
    assert "autoqc_comp=q_qc_idx(is,m_istart_sp);" in copy
    assert "q_c_arr(i,j,k)=states_arr(i,j,k,qc_comp)/states_arr(i,j,k,Rho_comp);" in copy
    assert "q_t_arr(i,j,k)=q_v_arr(i,j,k)+q_c_arr(i,j,k);" in copy
    assert copy.index("q_c_arr(i,j,k)=states_arr(i,j,k,qc_comp)") < copy.index("q_t_arr(i,j,k)=q_v_arr(i,j,k)+q_c_arr(i,j,k);")
    for name in ("DensityScaling(", "SetAttributes(", "phaseChange("):
        assert name not in copy, "state restoration must not initialize particles"


def mutations(sources):
    faults = [
        (DRIVER, "restart_chkfile.empty() ||", "!restart_chkfile.empty() ||"),
        (DRIVER, "restart_chkfile.empty() ||", "restart_chkfile.empty() &&"),
        (DRIVER, "!= MoistureModelType::Lagrangian", "== MoistureModelType::Lagrangian"),
        (DRIVER, "micro->FinishInit(lev, vars_new[lev][Vars::cons], z_phys_nd);", "micro->FinishInit(0, vars_new[0][Vars::cons], z_phys_nd);"),
        (DRIVER, "micro->Update_Micro_Vars_Lev(lev, vars_new[lev][Vars::cons], base);", "micro->Update_Micro_Vars_Lev(0, vars_new[0][Vars::cons], base);"),
        (DRIVER, "micro->Update_Micro_Vars_Lev(lev, vars_new[lev][Vars::cons], base);", "if (restart_chkfile.empty()) micro->Update_Micro_Vars_Lev(lev, vars_new[lev][Vars::cons], base);"),
        (DRIVER, "if (restart_chkfile.empty() &&", "if (true &&"),
        (UTILS, "auto qc_comp = q_qc_idx(is,m_istart_sp);", "auto qc_comp = q_qv_idx(is,m_istart_sp);"),
        (UTILS, "q_c_arr(i,j,k) = states_arr(i,j,k,qc_comp) / states_arr(i,j,k,Rho_comp);", "q_c_arr(i,j,k) = 0;"),
        (UTILS, "q_t_arr(i,j,k) = q_v_arr(i,j,k) + q_c_arr(i,j,k);", "q_t_arr(i,j,k) = q_v_arr(i,j,k);"),
        (UTILS, "q_i_arr(i,j,k) = states_arr(i,j,k,RhoQ3_comp) * inv_rho;", "q_i_arr(i,j,k) = 0;"),
        (WRAPPER, "m_moist_model->SetCurrentLevel(lev);", "m_moist_model->SetCurrentLevel(0);"),
    ]
    for file, old, new in faults:
        assert old in sources[file], old
        changed = dict(sources)
        changed[file] = changed[file].replace(old, new)
        try:
            verify(changed)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("accepted source mutation: " + old)
    verify({file: "// harmless comment { }\n" + text for file, text in sources.items()})
    return len(faults)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    sources = {file: ((args.overlay / file) if args.overlay and (args.overlay / file).exists() else args.source_root / file).read_text() for file in FILES}
    verify(sources)
    print(f"Particle restart initialization contract passed; {mutations(sources)} mutations rejected")


if __name__ == "__main__":
    main()
