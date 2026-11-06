#!/usr/bin/env python3
"""Bind OneWay ground-only transfer to the known pre-regrid call site.

This bounded textual contract checks source wiring and mutation sensitivity.
It does not prove C++ control flow or replace compiled conservation tests.
"""
import argparse
from pathlib import Path
from check_noahmp_precipitation_order import body, compact

TIMESTEP = "Source/TimeIntegration/ERF_TimeStep.cpp"
DRIVER = "Source/ERF.cpp"
HOOK = """if (solverChoice.coupling_type == CouplingType::OneWay &&
    Microphysics::modelType(solverChoice.moisture_type) == MoistureModelType::Lagrangian) {
    for (int flev = finest_level; flev > lev; --flev) {
        const auto fine = micro->Get_Ground_Mass_Accumulation_Ptrs(flev);
        const auto coarse = micro->Get_Ground_Mass_Accumulation_Ptrs(flev-1);
        AMREX_ALWAYS_ASSERT(fine.size() == coarse.size());
        for (int n = 0; n < fine.size(); ++n) {
            AMREX_ALWAYS_ASSERT(fine[n].name == coarse[n].name);
            AMREX_ALWAYS_ASSERT(fine[n].mass && coarse[n].mass);
            erf_precipitation::sum_down_surface_mass(
                *fine[n].mass, *coarse[n].mass,
                geom[flev], geom[flev-1], refRatio(flev-1));
        }
    }
}"""
TWOWAY = """if (solverChoice.coupling_type == CouplingType::TwoWay &&
    solverChoice.moisture_type != MoistureType::None &&
    Microphysics::modelType(solverChoice.moisture_type) == MoistureModelType::Lagrangian &&
    finest_level >= 1) {
    micro->AverageDownMicroVars(finest_level);
    for (int flev = finest_level-1; flev >= lev; --flev) {
        AverageDownMoistStateTo(flev);
    }
}"""


def verify(sources):
    timestep = compact(body(sources[TIMESTEP], r"ERF::timeStep\s*\("))
    cadence = compact(body(sources[TIMESTEP],
        r"if\s*\(\s*\(istep\[lev\]\s*%\s*regrid_int\s*==\s*0\)\s*&&\s*\(istep\[lev\]\s*>\s*last_regrid_step\[lev\]\)\s*\)"))
    expected = compact("int old_finest = finest_level;" + HOOK + TWOWAY)
    assert cadence.startswith(expected), "transfer must precede regrid without widening atmospheric feedback"
    assert cadence.index(compact(HOOK)) < cadence.index("regrid(lev,static_cast<Real>(time));")
    assert timestep.count("sum_down_surface_mass(") == 1
    assert timestep.count("micro->AverageDownMicroVars(") == 1
    assert timestep.count("Get_Ground_Mass_Accumulation_Ptrs(") == 2
    assert "regrid(" not in compact(body(sources[DRIVER], r"ERF::restart\s*\(")), "new restart topology route needs lifecycle review"
    init = compact(body(sources[DRIVER], r"ERF::InitData_post\s*\("))
    assert "Get_Ground_Mass_Accumulation_Ptrs(" not in init
    assert ("if(restart_chkfile.empty()&&solverChoice.coupling_type==CouplingType::TwoWay&&"
            "Microphysics::modelType(solverChoice.moisture_type)==MoistureModelType::Lagrangian&&"
            "finest_level>=1){micro->AverageDownMicroVars(finest_level);") in init


def self_test(sources):
    faults = [
        (TIMESTEP, "coupling_type == CouplingType::OneWay", "coupling_type == CouplingType::TwoWay"),
        (TIMESTEP, "MoistureModelType::Lagrangian", "MoistureModelType::Eulerian"),
        (TIMESTEP, "flev = finest_level; flev > lev; --flev", "flev = lev+1; flev <= finest_level; ++flev"),
        (TIMESTEP, "flev > lev; --flev", "flev > 1; --flev"),
        (TIMESTEP, "Get_Ground_Mass_Accumulation_Ptrs(flev-1)", "Get_Ground_Mass_Accumulation_Ptrs(0)"),
        (TIMESTEP, "Get_Ground_Mass_Accumulation_Ptrs(flev)", "Get_Ground_Mass_Accumulation_Ptrs(0)"),
        (TIMESTEP, "fine.size() == coarse.size()", "true"),
        (TIMESTEP, "n < fine.size()", "n < 1"),
        (TIMESTEP, "fine[n].name == coarse[n].name", "true"),
        (TIMESTEP, "fine[n].mass && coarse[n].mass", "true"),
        (TIMESTEP, "erf_precipitation::sum_down_surface_mass(", "erf_precipitation::average_down_surface_accumulation("),
        (TIMESTEP, "*fine[n].mass, *coarse[n].mass", "*coarse[n].mass, *fine[n].mass"),
        (TIMESTEP, "geom[flev], geom[flev-1]", "geom[flev-1], geom[flev]"),
        (TIMESTEP, "refRatio(flev-1)", "refRatio(0)"),
        (TIMESTEP, "int old_finest = finest_level;", "int old_finest = finest_level; if (false)"),
        (TIMESTEP, "int old_finest = finest_level;", "int old_finest = finest_level;\n#if 0"),
        (TIMESTEP, "int old_finest = finest_level;", "int old_finest = finest_level; regrid(lev, time);"),
        (TIMESTEP, "micro->AverageDownMicroVars(finest_level);", "micro->AverageDownMicroVars(0);"),
        (DRIVER, "ReadCheckpointFile();", "ReadCheckpointFile(); regrid(0, t_new[0]);"),
    ]
    for file, old, new in faults:
        assert old in sources[file], old
        changed = dict(sources)
        changed[file] = changed[file].replace(old, new, 1)
        try:
            verify(changed)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("accepted source fault: " + old)
    verify({file: "// harmless { } comment\n" + text for file, text in sources.items()})
    return len(faults)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    sources = {name: ((args.overlay / name) if args.overlay and (args.overlay / name).exists()
                      else args.source_root / name).read_text() for name in (TIMESTEP, DRIVER)}
    verify(sources)
    print(f"OneWay ground mass source contract passed; {self_test(sources)} faults rejected")


if __name__ == "__main__":
    main()
