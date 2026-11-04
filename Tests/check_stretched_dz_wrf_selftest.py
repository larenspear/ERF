#!/usr/bin/env python3
"""Fault injection for WRF nominal-grid refresh ordering and branch contracts."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import brace_block, no_literals, signature, scopes
from check_stretched_dz_consumers import pattern
from check_stretched_dz_wrf import PROTECTED, REFRESH, TERRAIN, WRF
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_wrf.py'
SUCCESS = '4 WRF spacing source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    text = strip_comments(original)
    _, body, definition = signature(text, 'ERF::init_from_wrfinput')
    offset = definition.end()
    refresh = unique(pattern(REFRESH), body, 'refresh')
    build = unique(pattern(TERRAIN), body, 'terrain')
    stack = scopes(body, refresh.start())
    start = stack[1][1]+1
    grid = body[start:build.end()]

    def fixture(name, new_body, fault=True):
        assert new_body != body, name
        return name, text[:offset]+new_body+text[offset+len(body):], fault

    def replace(name, before, after, target=grid, fault=True):
        search = pattern(before).removeprefix(r'(?<![\w:.>])')
        match = unique(search, target, name)
        changed = target[:match.start()]+after+target[match.end():]
        yield fixture(name, body[:start]+changed+body[build.end():] if target is grid else changed, fault)

    for name, before, after in (
        ('missing refresh', REFRESH, ''),
        ('duplicate refresh', REFRESH, REFRESH+REFRESH),
        ('wrong refresh level', REFRESH, REFRESH.replace('(lev,', '(0,')),
        ('wrong refresh nodes', REFRESH, REFRESH.replace('zlevels_stag', 'other')),
        ('wrong refresh host', REFRESH, REFRESH.replace('stretched_dz_h', 'other')),
        ('wrong refresh device', REFRESH, REFRESH.replace('stretched_dz_d', 'other')),
        ('conditional refresh', REFRESH, 'if (read_atmos_state) { '+REFRESH+' }'),
        ('deferred refresh', REFRESH, 'auto later = [&] { '+REFRESH+' };'),
        ('unbraced conditional refresh', REFRESH, 'if (use_file_profile) '+REFRESH),
        ('wrong terrain level', TERRAIN, TERRAIN.replace('(lev,', '(0,')),
        ('wrong terrain nodes', TERRAIN, TERRAIN.replace('zlevels_stag[lev]', 'other')),
        ('missing terrain call', TERRAIN, ''),
        ('duplicate terrain call', TERRAIN, TERRAIN+TERRAIN),
        ('deferred terrain call', TERRAIN, 'auto later = [&] { '+TERRAIN+' };'),
        ('conditional terrain call', TERRAIN, 'if (read_atmos_state) { '+TERRAIN+' }'),
        ('wrong float tolerance', 'Real(1.e-4)', 'Real(1.e-8)'),
        ('wrong double tolerance', 'Real(1.e-8)', 'Real(1.e-4)'),
        ('wrong precision guard', '#ifdef AMREX_USE_FLOAT', '#ifdef AMREX_USE_GPU'),
        ('wrong layer count', 'static_cast<int>(zlevels_stag[lev].size()) - 1', 'static_cast<int>(zlevels_stag[lev].size())'),
        ('wrong layer-count level', 'static_cast<int>(zlevels_stag[lev].size()) - 1', 'static_cast<int>(zlevels_stag[0].size()) - 1'),
        ('profile reference', 'bool use_file_profile =', 'bool& use_file_profile ='),
        ('wrong profile setting', 'solverChoice.wrfinput_zlevels_from_file', 'solverChoice.avg_grid_faces_to_nodes'),
        ('wrong mean-level binding', 'geom[lev], ncol_min', 'geom[0], ncol_min'),
        ('inverted profile size guard', 'static_cast<int>(dz_prof.size()) == nlay', 'static_cast<int>(dz_prof.size()) != nlay'),
        ('inverted missing-column fallback', 'ncol_min == 0', 'ncol_min != 0'),
        ('missing-column profile retained', 'use_file_profile = false;', 'use_file_profile = true;'),
        ('wrong fallback threshold', 'ncol_min < ncol_domain', 'ncol_min > ncol_domain'),
        ('inverted positive-profile assertion', 'dz_prof[k] > zero', 'dz_prof[k] < zero'),
        ('profile accumulation changed', 'dz_tot += dz_prof[k];', 'dz_tot -= dz_prof[k];'),
        ('wrong fraction storage', 'Vector<Real> sfrac(nlay, zero);', 'Vector<Real> sfrac(nlay-1, zero);'),
        ('wrong weight fallback', 'ramp_closes = false;', 'ramp_closes = true;'),
        ('wrong scale selector', 'ramp_closes ? one : z_top/dz_tot', 'ramp_closes ? z_top/dz_tot : one'),
        ('wrong initial node', 'zlevels_stag[lev][0] = zero;', 'zlevels_stag[lev][0] = one;'),
        ('wrong initial-node level', 'zlevels_stag[lev][0] = zero;', 'zlevels_stag[0][0] = zero;'),
        ('wrong node loop bound', 'for (int k(1); k<=nlay; ++k)', 'for (int k(1); k<nlay; ++k)'),
        ('wrong node loop start', 'for (int k(1); k<=nlay; ++k)', 'for (int k(0); k<=nlay; ++k)'),
        ('wrong profile write level', 'zlevels_stag[lev][k] =', 'zlevels_stag[0][k] ='),
        ('wrong profile increment', '+ w*dz_prof[k-1]', '- w*dz_prof[k-1]'),
        ('wrong top node', 'zlevels_stag[lev][nlay] = z_top;', 'zlevels_stag[lev][nlay-1] = z_top;'),
        ('wrong top level', 'zlevels_stag[lev][nlay] = z_top;', 'zlevels_stag[0][nlay] = z_top;'),
        ('wrong geometric nodes', 'build_geometric_zlevels(zlevels_stag[lev],', 'build_geometric_zlevels(zlevels_stag[0],'),
        ('wrong geometric depth', 'dz0_max, z_top, tol', 'dz0_max, one, tol'),
        ('wrong geometric initial thickness', 'dz0_max, z_top, tol', 'one, z_top, tol'),
        ('inverted convergence assertion', '(gs.converged,', '(!gs.converged,'),
        ('wrong top assertion', 'std::fabs(zlevels_stag[lev].back() - z_top) <= tol', 'std::fabs(zlevels_stag[lev].back() - z_top) >= tol'),
    ):
        yield from replace(name, before, after)

    # Move calls intact across branch and mutation boundaries.
    without = body[:refresh.start()]+body[refresh.end():]
    for name, at in (
        ('refresh before grid', start),
        ('refresh only in profile', unique(pattern('zlevels_stag[lev][nlay] = z_top;'), body, 'top').end()),
        ('refresh only in geometric', unique(pattern('const Real dz0_max ='), body, 'geometric').start()),
    ):
        yield fixture(name, without[:at]+REFRESH+without[at:])
    yield fixture('refresh after terrain', body[:refresh.start()]+body[refresh.end():build.end()]+REFRESH+body[build.end():])
    for name, before, after in (
        ('inverted terrain gate', 'if (compute_terrain_here) {', 'if (!compute_terrain_here) {'),
        ('atmosphere-only terrain gate', 'if (compute_terrain_here) {', 'if (compute_terrain_here && read_atmos_state) {'),
        ('inverted fitted-coordinate gate', 'if (!solverChoice.avg_grid_faces_to_nodes) {', 'if (solverChoice.avg_grid_faces_to_nodes) {'),
        ('disabled terrain flag', 'bool compute_terrain_here = true;', 'bool compute_terrain_here = false;'),
        ('flag reference', 'bool compute_terrain_here = true;', 'bool& compute_terrain_here = other;'),
        ('inverted surface exit', 'if (!read_atmos_state) {', 'if (read_atmos_state) {'),
    ):
        yield from replace(name, before, after, body)
    leave = unique(pattern('if (!read_atmos_state) {'), body, 'surface exit')
    leave_end = leave.end()+len(brace_block(body, leave.end()-1))+1
    exit_block = body[leave.start():leave_end]
    yield fixture('surface-only exit before terrain', body[:start]+exit_block+body[start:leave.start()]+body[leave_end:])
    for name, before, after in (
        ('removed PH request', 'NC_names.push_back("PH");', ''),
        ('removed PHB request', 'NC_names.push_back("PHB");', ''),
        ('conditional PH request', 'NC_names.push_back("PH");', 'if (read_atmos_state) { NC_names.push_back("PH"); }'),
        ('conditional PHB request', 'NC_names.push_back("PHB");', 'if (read_atmos_state) { NC_names.push_back("PHB"); }'),
    ):
        yield from replace(name, before, after, body)
    for name in ('PH', 'PHB'):
        branch = unique(pattern('if (var_name == "'+name+'") {'), body, name+' branch')
        contents = brace_block(body, branch.end()-1)
        failure = unique(pattern('compute_terrain_here = false;'), contents, name+' failure')
        for label, value in (
            ('keep true', 'compute_terrain_here = true;'), ('missing reset', ''),
            ('conditional reset', 'if (read_atmos_state) { compute_terrain_here = false; }'),
            ('deferred reset', 'auto later = [&] { compute_terrain_here = false; };'),
        ):
            at, end = branch.end()+failure.start(), branch.end()+failure.end()
            yield fixture(name+' '+label, body[:at]+value+body[end:])
        success = unique(pattern('if (success) {'), contents, 'success gate')
        at, end = branch.end()+success.start(), branch.end()+success.end()
        yield fixture(name+' inverted success', body[:at]+'if (!success) {'+body[end:])
    # Guard the whole PH/PHB chain, leaving the following PB else-if valid.
    ph = unique(pattern('if (var_name == "PH") {'), body, 'PH')
    phb = unique(pattern('if (var_name == "PHB") {'), body, 'PHB')
    end = phb.end()+len(brace_block(body, phb.end()-1))+1
    # This mutant intentionally drops the later else keyword so the selected
    # CPU and GPU branches both remain syntactically connected statements.
    suffix = body[end:]
    suffix = re.sub(r'^\s*else\s+', '\n', suffix, count=1)
    yield fixture('feature-guarded PH handlers', body[:ph.start()]+'\n#ifdef AMREX_USE_GPU\n'+body[ph.start():end]+'\n#endif\n'+suffix)

    for name, code in (
        ('nominal escape', 'retain(zlevels_stag);'),
        ('device escape', 'retain(stretched_dz_d);'),
        ('host resize', 'stretched_dz_h.clear();'),
        ('flag alias', 'auto& alias = compute_terrain_here;'),
        ('flag reset', 'compute_terrain_here = true;'),
        ('callee shadow', 'auto update_stretched_dz = other;'),
        ('geometric shadow', 'auto build_geometric_zlevels = other;'),
        ('terrain shadow', 'auto make_terrain_fitted_coords = other;'),
        ('early return', 'return;'),
        ('goto', 'goto later;'),
        ('runtime include', '\n#include "unmodeled.H"\n'),
        ('literal hides write', 'Print() << "} // /*"; stretched_dz_d.clear();'),
    ):
        for site in (0, build.end(), len(body)):
            yield fixture(name+f' at {site}', body[:site]+code+body[site:])
    for guard in ('AMREX_USE_GPU', '_OPENMP', 'AMREX_DEBUG', 'AMREX_USE_FLOAT'):
        code = '\n#ifdef '+guard+'\nstretched_dz_d.clear();\n#endif\n'
        yield fixture('hidden mutation '+guard, body[:build.end()]+code+body[build.end():])
    for name, code in (
        ('macro', '#define update_stretched_dz ignore'),
        ('unknown conditional', '#ifdef OTHER\n#endif'),
        ('line splice', '// hidden \\\n#define update_stretched_dz ignore'),
        ('raw string', 'R"(stretched_dz_d)";'),
        ('digraph', '<% stretched_dz_d.clear(); %>'),
    ):
        yield name, text+'\n'+code+'\n', True
    for label, code in (
        ('diagnostic side effect', 'Print() << mutate(zlevels_stag);'),
        ('extra nominal mutation', 'zlevels_stag[lev][0] = one;'),
        ('spacing alias', 'auto& alias = stretched_dz_d;'),
        ('profile alias', 'auto& alias = dz_prof;'),
    ):
        yield fixture(label, body[:refresh.start()]+code+body[refresh.start():])

    yield 'comments removed', text, False
    yield 'CRLF', original.replace('\n', '\r\n'), False
    yield 'token whitespace', text.replace(',', ', /* gap */ ').replace(';', ';\n'), False
    for site in (0, build.end(), len(body)):
        yield fixture(f'unrelated local at {site}', body[:site]+'const int unused = 1;'+body[site:], False)
    yield fixture('quoted protected names', body[:build.end()]+'Print() << "} /* // '+ ' '.join(PROTECTED)+'";'+body[build.end():], False)
    # Logging can change without changing the checked control/data flow.
    warning = unique(r'Print\(\) << "WARNING: z_top', body, 'warning')
    end = no_literals(body).index(';', warning.start())+1
    yield fixture('warning wording', body[:warning.start()]+'Print() << "Warning; } /* // " << z_top;'+body[end:], False)
    yield fixture('warning removed', body[:warning.start()]+body[end:], False)
    for name in ('PH', 'PHB'):
        branch = unique(pattern('if (var_name == "'+name+'") {'), body, name)
        contents = brace_block(body, branch.end()-1)
        failure = unique(pattern('compute_terrain_here = false;'), contents, name)
        at = branch.end()+failure.start()
        yield fixture(name+' diagnostic control', body[:at]+'Print() << "unavailable; } // /*";'+body[at:], False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-wrf-mutants-'))
    fixtures = list(cases((ROOT/WRF).read_text()))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, source, fault) in enumerate(fixtures):
        root = work/f'{index:03d}'
        target = root/WRF
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught, expected, _ = mutation_outcome(result, fault)
        required = 'FAIL source contract:' if fault else SUCCESS
        expected = expected and required in result.stdout
        faults += fault
        controls += not fault and name != 'baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
