#!/usr/bin/env python3
"""Bind NoahMP lifecycle policy tests to guards before destructive operations.

This bounded source check does not prove arbitrary C++ control flow. It uses
the same comment/literal-aware function extraction as the precipitation check.
"""
import argparse
from pathlib import Path
from check_noahmp_precipitation_order import body, compact

LEVELS = "Source/ERF_MakeNewLevel.cpp"
INIT = "Source/LandSurfaceModel/Noah-MP/ERF_NOAHMP_Init.cpp"
LAND = "Source/LandSurfaceModel/ERF_LandSurface.H"

def verify(sources):
    create = compact(body(sources[LEVELS],r"ERF::MakeNewLevelFromCoarse\s*\("))
    dispatch = ("if(solverChoice.lsm_type==LandSurfaceType::NOAHMP){"
                "lsm.Define(lev,solverChoice);"
                "if(!noahmp_new_level_is_supported("
                "lsm.Will_Run_Own_Land_Driver(lev,nc_init_file),time)){Abort(")
    position = create.index(dispatch)
    # No prior state writes are allowed in this prefix. The two feature
    # assertions and the level assertion precede the dispatch.
    prefix = create[:position]
    assert prefix.count("AMREX_ALWAYS_ASSERT_WITH_MESSAGE(") == 2
    assert prefix.endswith("AMREX_ALWAYS_ASSERT(lev>0);")
    assert prefix.count(";") == 3, "creation guard is no longer before mutations"
    assert position < create.index("subdomains.resize(") < create.index("make_lsm_at_level(")

    clear = compact(body(sources[LEVELS],r"ERF::ClearLevel\s*\("))
    assert clear.startswith("if(solverChoice.lsm_type==LandSurfaceType::NOAHMP&&"
                            "!noahmp_level_rebuild_or_removal_is_supported("
                            "lsm.Runs_Own_Land_Driver(lev))){Abort("), "removal must reject before mutation"
    remake = compact(body(sources[LEVELS],r"ERF::RemakeLevel\s*\("))
    assert "!noahmp_level_rebuild_or_removal_is_supported(lsm.Runs_Own_Land_Driver(lev))" in remake

    initialization = compact(body(sources[INIT],r"NOAHMP::Init\s*\("))
    assert "m_has_nc_file=Will_Run_Own_Land_Driver(lev,nc_init_file);" in initialization
    selection = compact(body(sources[INIT],r"NOAHMP::Will_Run_Own_Land_Driver\s*\("))
    assert selection == ("constboolhas_init_file=!nc_init_file[lev].empty();"
                         "constboolnamelist_names_setup=(lev>0&&!has_init_file&&m_idealized_init)"
                         "?namelist_names_setup_file(lev):false;"
                         "returnnoahmp_level_runs_driver(lev,has_init_file,m_idealized_init,namelist_names_setup);")
    forwarding = compact(body(sources[LAND],r"bool\s+Will_Run_Own_Land_Driver\s*\("))
    assert forwarding == "returnm_lsm_model[lev]->Will_Run_Own_Land_Driver(lev,nc_init_file);"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sources = {name:(args.source_root/name).read_text() for name in [LEVELS,INIT,LAND]}
    verify(sources)
    faults = [
        (LEVELS,"lsm.Define(lev, solverChoice);","lsm.Define(0, solverChoice);"),
        (LEVELS,"lsm.Will_Run_Own_Land_Driver(lev, nc_init_file), time)","lsm.Will_Run_Own_Land_Driver(lev, nc_init_file), 0)"),
        (LEVELS,"ERF::ClearLevel (int lev)\n{","ERF::ClearLevel (int lev)\n{ vars_new[lev][0].clear();"),
        (INIT,"m_has_nc_file = Will_Run_Own_Land_Driver(lev, nc_init_file);","m_has_nc_file = false;"),
        (LAND,"m_lsm_model[lev]->Will_Run_Own_Land_Driver(lev, nc_init_file)","m_lsm_model[0]->Will_Run_Own_Land_Driver(0, nc_init_file)"),
    ]
    for name,old,new in faults:
        assert old in sources[name],old
        changed = dict(sources)
        changed[name] = changed[name].replace(old,new,1)
        try:
            verify(changed)
        except (AssertionError,ValueError):
            continue
        raise AssertionError("lifecycle source fault escaped: " + old)
    print("NoahMP lifecycle source contract passed; 5 faults rejected")

if __name__ == "__main__":
    main()
