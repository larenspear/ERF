#!/usr/bin/env python3
"""Bind the tested NoahMP preparation helper to known production call sites.

This is a bounded source contract, not a C++ control-flow or preprocessing proof.
The helper's numerical/I/O behavior is tested separately. The checks deliberately
require a simple first-statement dispatch; refactoring that dispatch requires
reviewing this contract. Self-tests reject misplaced, disabled, or misrouted hooks.
"""
import argparse
from pathlib import Path
import re

ADVANCE = "Source/TimeIntegration/ERF_Advance.cpp"
TIMESTEP = "Source/TimeIntegration/ERF_TimeStep.cpp"
SLOW = "Source/TimeIntegration/ERF_TI_slow_rhs_post.H"
LAND = "Source/LandSurfaceModel/ERF_LandSurface.H"
NOAH = "Source/LandSurfaceModel/Noah-MP/ERF_NOAHMP.H"
NULL = "Source/LandSurfaceModel/Null/ERF_NullSurf.H"
FILES = [ADVANCE, TIMESTEP, SLOW, LAND, NOAH, NULL]

# Match quoted literals first so comment-looking text inside a literal is kept.
TOKENS = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*[\s\S]*?\*/|[{}]')

def uncomment(text):
    return TOKENS.sub(lambda match: " " if match[0].startswith(("//", "/*")) else match[0], text)

def body(text, marker):
    text = uncomment(text)
    searchable = TOKENS.sub(lambda match: " " * len(match[0])
                            if match[0].startswith(('"', "'")) else match[0], text)
    matches = list(re.finditer(marker, searchable))
    if len(matches) != 1:
        raise AssertionError(f"expected one function matching {marker}, got {len(matches)}")
    start = searchable.index("{", matches[0].end())
    depth = 0
    for match in TOKENS.finditer(text,start):
        if match[0] == "{":
            depth += 1
        elif match[0] == "}":
            depth -= 1
            if depth == 0:
                return text[start+1:match.start()]
    raise AssertionError(f"unclosed function {marker}")

def compact(text):
    return re.sub(r"\s+", "", uncomment(text))

HOOK = """if (solverChoice.lsm_type == LandSurfaceType::NOAHMP) {
    lsm.Prepare_Precipitation(lev,
        micro ? micro->Get_Surface_Precip_Accumulation_Ptrs(lev)
              : SurfacePrecipAccumulationSources{});
}"""

def verify(sources):
    advance = compact(body(sources[ADVANCE],r"ERF::Advance\s*\("))
    expected_prefix = compact('BL_PROFILE("ERF::Advance()");' + HOOK)
    assert advance.startswith(expected_prefix), "preparation must be the first dispatch, with typed per-level sources"
    assert advance.count("lsm.Prepare_Precipitation(") == 1, "exactly one pre-step dispatch required"
    assert advance.index("advance_dycore(") < advance.index("advance_microphysics(") < advance.index("advance_lsm("), "review known atmospheric/land call order"

    land = compact(body(sources[LAND],r"void\s+Prepare_Precipitation\s*\("))
    assert land == "m_lsm_model[lev]->Prepare_Precipitation(lev,precip_sources);", "per-level forwarding changed"
    noah = compact(body(sources[NOAH],r"void\s+Prepare_Precipitation\s*\("))
    assert noah == "if(m_has_nc_file){prepare_precip_snapshots(lev,collect_precip_sources(precip_sources));}", "driver preparation must only seed missing snapshots"
    null = compact(body(sources[NULL],r"void\s+Prepare_Precipitation\s*\("))
    assert null == "", "other surface models require a no-op default"

    timestep = compact(body(sources[TIMESTEP],r"ERF::timeStep\s*\("))
    assert "Advance(lev,time,dt[lev],istep[lev],nsubsteps[lev]);" in timestep, "per-level Advance call changed"
    assert "timeStep(lev+1,strt_time_for_fine,i);" in timestep, "fine substep recursion changed"
    assert "advance_microphysics(level,S_new[0],slow_dt,123456789,old_step_time);" in compact(sources[SLOW]), "tight microphysics call changed"

def self_test(sources):
    failures = [
        ("missing hook", ADVANCE, "lsm.Prepare_Precipitation(", "lsm.Other_Preparation("),
        ("early return", ADVANCE, 'BL_PROFILE("ERF::Advance()");', 'BL_PROFILE("ERF::Advance()"); if (lev > 0) return;'),
        ("conditional hook", ADVANCE, "if (solverChoice.lsm_type == LandSurfaceType::NOAHMP)", "if (false) if (solverChoice.lsm_type == LandSurfaceType::NOAHMP)"),
        ("wrong source level", ADVANCE, "Get_Surface_Precip_Accumulation_Ptrs(lev)", "Get_Surface_Precip_Accumulation_Ptrs(0)"),
        ("wrong forwarded level", LAND, "m_lsm_model[lev]->Prepare_Precipitation(lev, precip_sources);", "m_lsm_model[0]->Prepare_Precipitation(0, precip_sources);"),
        ("wrong driver gate", NOAH, "if (m_has_nc_file)", "if (!m_has_nc_file)"),
        ("retire instead of prepare", NOAH, "prepare_precip_snapshots(lev, collect_precip_sources(precip_sources));", "advance_precip_snapshots(lev, collect_precip_sources(precip_sources));"),
        ("lost recursion", TIMESTEP, "timeStep(lev+1, strt_time_for_fine, i);", "otherStep(lev+1, strt_time_for_fine, i);"),
        ("wrong Advance level", TIMESTEP, "Advance(lev, time, dt[lev], istep[lev], nsubsteps[lev]);", "Advance(0, time, dt[lev], istep[lev], nsubsteps[lev]);"),
    ]
    for name, file, old, new in failures:
        assert old in sources[file], f"self-test anchor disappeared: {name}"
        changed = dict(sources)
        changed[file] = changed[file].replace(old,new,1)
        try:
            verify(changed)
        except (AssertionError,ValueError):
            continue
        raise AssertionError(f"self-test accepted fault: {name}")
    # Move the intact, well-formed hook after the actual dycore statement.
    changed = dict(sources)
    text = sources[ADVANCE]
    first = text.index("    if (solverChoice.lsm_type == LandSurfaceType::NOAHMP)")
    last = text.index("\n    }",first) + len("\n    }")
    hook = text[first:last]
    text = text[:first] + text[last:]
    after_dycore = text.index(";",text.index("advance_dycore(")) + 1
    changed[ADVANCE] = text[:after_dycore] + "\n" + hook + text[after_dycore:]
    try:
        verify(changed)
    except (AssertionError,ValueError):
        pass
    else:
        raise AssertionError("self-test accepted hook after dycore")
    # A complete preprocessor block must not hide the required invocation.
    changed[ADVANCE] = sources[ADVANCE].replace(hook,"#if 0\n" + hook + "\n#endif",1)
    try:
        verify(changed)
    except (AssertionError,ValueError):
        pass
    else:
        raise AssertionError("self-test accepted disabled hook")
    # Benign formatting and comments must not look like calls or alter braces.
    control = dict(sources)
    control[ADVANCE] = control[ADVANCE].replace("lsm.Prepare_Precipitation(","lsm.Prepare_Precipitation /* comment { } */ (",1)
    control = {name: "// lsm.Prepare_Precipitation({ wrong });\n" + text for name,text in control.items()}
    verify(control)
    return len(failures) + 2

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overlay",type=Path,help="optional isolated candidate tree overriding existing paths")
    args = parser.parse_args()
    sources = {}
    for name in FILES:
        candidate = args.overlay / name if args.overlay else None
        sources[name] = (candidate if candidate and candidate.is_file() else args.source_root / name).read_text()
    verify(sources)
    rejected = self_test(sources)
    print(f"NoahMP preparation source contract passed; {rejected} faults rejected, 1 formatting control passed")

if __name__ == "__main__":
    main()
