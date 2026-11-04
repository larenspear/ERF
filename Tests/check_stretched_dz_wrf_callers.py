#!/usr/bin/env python3
"""Connect the WRF spacing contract to its declaration and selected callers.

Check the true default argument and matching definition, init_only's ordinary
WRF branch, and MakeNewLevelFromCoarse's surface-only selection, false/default
calls, subsequent terrain updates and surface-only base-state/interpolation.
Audit all explicit mode, PSFC and WRF-callee uses in the two caller bodies, and
reject explicit nominal/spacing uses there. Both NETCDF and factory alternatives
are checked; disabled NETCDF must leave no WRF call or surface-only variable.

Four finite source-contract groups; no numerical or GPU-lifetime theorem. Assume
valid level/field/storage objects, ordinary C++/AMReX semantics and intended
type/callee resolution and the separate WRF initializer/refresh contracts.
Selectors and level/time must remain stable in unparsed caller code; other calls
must preserve spacing through implicit this, aliases and globals. Prefix/suffix
physics, allocation, I/O, the code that enters
these callers, and numerical bodies of the subsequent terrain/base-state calls
are outside the contract. No external macro rewriting, nonlocal transfer,
concurrency or undefined behavior is modeled. GPU completion, physical grid
validity, positivity and projection/EB preservation remain unproved.

usage: check_stretched_dz_wrf_callers.py [ERF_ROOT]
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (brace_block, direct, no_literals,
                                       reject_uses, scopes, signature)
from check_stretched_dz_consumers import pattern, standalone
from check_stretched_dz_wrf import WRF, prints, take
from momentum_geometry import unique

HEADER = 'Source/ERF.H'
INIT = 'Source/ERF.cpp'
COARSE = 'Source/ERF_MakeNewLevel.cpp'
FILES = (HEADER, INIT, COARSE, WRF)
NAME = 'init_from_wrfinput'
CALL = 'init_from_wrfinput(lev, *mf_PSFC[lev]);'
SURFACE_CALL = 'init_from_wrfinput(lev, *mf_PSFC[lev], false);'
SELECT = '''use_surface_only = solverChoice.interp_atmos_from_coarse && (lev > 0) &&
    (solverChoice.init_type == InitType::WRFInput);'''
GATE = '''if (((solverChoice.init_type == InitType::WRFInput) ||
    (solverChoice.init_type == InitType::Metgrid)) && !nc_init_file[lev].empty()) {'''
STORAGE = ('zlevels_stag', 'stretched_dz_h', 'stretched_dz_d')
PROTECTED = (NAME, 'mf_PSFC', 'use_surface_only', *STORAGE)


def pp_stack(text):
    """Track surrounding directives; unrelated conditions are not interpreted."""
    stack = []
    for m in re.finditer(r'(?m)^[ \t]*#\s*(\w+)\b([^\n]*)', no_literals(text)):
        kind, arg = m[1], m[2].strip()
        if kind in ('if', 'ifdef', 'ifndef'):
            stack.append((kind, arg))
        elif kind in ('else', 'elif'):
            require(stack and stack[-1][0] != 'else', 'invalid caller preprocessor branch')
            stack[-1] = (kind, arg)
        elif kind == 'endif':
            require(stack and not arg, 'unmatched caller endif')
            stack.pop()
    return stack


def configured(body, netcdf, factory):
    output, stack = [], []
    for line in body.splitlines(keepends=True):
        if '#' not in no_literals(line):
            if all(active for active, _ in stack):
                output.append(line)
            continue
        match = re.fullmatch(r'\s*#\s*(\w+)\s*([^\n]*?)\s*', line)
        require(match, 'unsupported caller directive syntax')
        kind, arg = match[1], match[2].strip()
        if kind == 'ifdef' and arg == 'ERF_USE_NETCDF':
            stack.append((netcdf, False))
        elif kind == 'if' and arg == 'USE_FC_FACTORY':
            stack.append((factory, False))
        elif kind == 'else':
            require(stack and not arg and not stack[-1][1], 'invalid caller else')
            stack[-1] = (not stack[-1][0], True)
        elif kind == 'endif':
            require(stack and not arg, 'invalid caller endif')
            stack.pop()
        else:
            raise ValueError('unsupported caller directive')
    require(not stack, 'unclosed caller conditional')
    return ''.join(output)


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported caller lexical form')
        text = strip_comments(raw)
        for m in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(path == HEADER and m[1] == 'define' and m[2].strip() == 'ERF_H_',
                    path+': unsupported caller macro')
        require(not pp_stack(text), path+': unclosed preprocessor scope')
        texts[path] = text
    return texts


def interface(texts):
    text = texts[HEADER]
    owner = unique(pattern('class ERF : public amrex::AmrCore {'), text, 'ERF class')
    _, _, decl = signature(no_literals(text), NAME, ';')
    require(re.fullmatch(pattern('int lev, amrex::MultiFab& mf_PSFC, bool read_atmos_state = true'),
                         decl[1].strip()), 'WRF declaration/default argument changed')
    require(scopes(text, decl.start()) == [('{', owner.end()-1)], 'WRF declaration must be an ERF member')
    require(pp_stack(text[:decl.start()]) == [('ifndef', 'ERF_H_'), ('ifdef', 'ERF_USE_NETCDF')],
            'WRF declaration requires only header and NETCDF guards')
    reject_uses(text, [NAME], [decl.span()], 'WRF declaration inventory')
    _, _, definition = signature(no_literals(texts[WRF]), 'ERF::'+NAME)
    require(re.fullmatch(pattern('int lev, MultiFab& mf_PSFC_lev, bool read_atmos_state'),
                         definition[1].strip()), 'WRF definition/signature mismatch')
    require(not scopes(texts[WRF], definition.start()), 'WRF definition must be at global scope')
    require(pp_stack(texts[WRF][:definition.start()]) == [('ifdef', 'ERF_USE_NETCDF')],
            'WRF definition requires only its NETCDF guard')
    reject_uses(texts[WRF], [NAME], [definition.span()], 'WRF definition inventory')
    print('  ok   ERF declaration and definition agree; omitted mode defaults to true')


def function(text, name, parameters):
    _, body, definition = signature(text, name)
    require(re.fullmatch(pattern(parameters), definition[1].strip()), name+': caller signature changed')
    require(not scopes(text, definition.start()) and not pp_stack(text[:definition.start()]),
            name+': caller definition must be unconditional/global')
    require(not re.search(r'\b(?:return|goto|try|catch|co_await|co_yield|co_return)\b', no_literals(body)),
            name+': unsupported caller control transfer')
    reject_uses(text, [NAME], [(definition.start(), definition.end()+len(body)+1)], name+' file call inventory')
    return body


def ordinary(body):
    call = unique(pattern(CALL), body, 'ordinary WRF call')
    stack = scopes(body, call.start())
    require(len(stack) == 1 and stack[0][0] == '{', 'ordinary WRF call must have one branch scope')
    gate = unique(pattern('else if (solverChoice.init_type == InitType::WRFInput && !nc_init_file[lev].empty()) {')+r'\s*$',
                  body[:stack[0][1]+1], 'ordinary WRF gate')
    previous = unique(pattern('if (solverChoice.init_type == InitType::Input_Sounding) {'), body, 'sounding branch')
    direct(body, previous.start(), 'initialization dispatch')
    previous_end = previous.end()+len(brace_block(body, previous.end()-1))+1
    require(not body[previous_end:gate.start()].strip(), 'ordinary WRF branch must follow sounding')
    contents = brace_block(body, gate.end()-1)
    source = Source(contents)
    take(source, CALL)
    take(source, '''make_physbcs(lev);
        (*physbcs_base[lev])(base_state[lev],0,base_state[lev].nComp(),base_state[lev].nGrowVect());''')
    source.done()
    reject_uses(body, PROTECTED, [call.span()], 'ordinary caller inventory')


def coarse(body):
    init = unique(pattern('[[maybe_unused]] bool use_surface_only = false;'), body, 'surface-only initialization')
    direct(body, init.start(), 'surface-only initialization')
    gate = unique(pattern(GATE), body, 'coarse file gate')
    direct(body, gate.start(), 'coarse file gate')
    selection = unique(pattern(SELECT), body, 'surface-only selection')
    require(scopes(body, selection.start()) == [('{', gate.end()-1)], 'selection must be directly in file gate')
    standalone(body, selection.start())
    require(init.end() < gate.start() < selection.start(), 'mode initialization/selection order')
    source = Source(body[selection.start():])
    take(source, SELECT)
    take(source, '''if (static_cast<int>(rad_interp_from_coarse_pending.size()) <= lev) {
            rad_interp_from_coarse_pending.resize(lev+1, 0);
        }
        rad_interp_from_coarse_pending[lev] = use_surface_only ? 1 : 0;
        if (use_surface_only && solverChoice.lsm_type == LandSurfaceType::SLM) {
            make_lsm_at_level(lev, false, lev-1, false, true);
            lsm_initialized = true;
        }
        if (solverChoice.init_type == InitType::Metgrid) {
            init_from_metgrid(lev);
        } else if (solverChoice.init_type == InitType::WRFInput) {
            if (use_surface_only) {''')
    prints(source, ('lev', 'lev-1'))
    take(source, SURFACE_CALL)
    take(source, '} else {')
    take(source, CALL)
    take(source, '''} }
        init_zphys(lev, time);
        update_terrain_arrays(lev);''')
    # Consume the intervening metric/boundary operations too: no inserted
    # mutation or conditional may separate terrain and surface completion.
    take(source, '''if ((SolverChoice::mesh_type != MeshType::ConstantDz) &&
        (solverChoice.coupling_type == CouplingType::TwoWay)) {
        for (int crse_lev = lev-1; crse_lev >= 0; crse_lev--) {
            average_down(*detJ_cc[crse_lev+1], *detJ_cc[crse_lev], 0, 1, refRatio(crse_lev));
            detJ_cc[crse_lev]->FillBoundary(geom[crse_lev].periodicity());
        }
    }
    make_physbcs(lev);
    dz_min[lev] = (*detJ_cc[lev]).min(0) * geom[lev].CellSize(2);''')
    take(source, '''if (use_surface_only) {
        rebuild_base_state_from_wrfinput(lev, base_state[lev]);
        (*physbcs_base[lev])(base_state[lev],0,base_state[lev].nComp(),base_state[lev].nGrowVect());
        FillCoarsePatch(lev, time);
    }''')
    end = selection.start()+source.pos
    require(not body[end:gate.end()+len(brace_block(body, gate.end()-1))].strip(),
            'unexpected file-branch suffix')
    allowed = [init.span(), (selection.start(), end)]
    later = unique(pattern('initialize_now = use_surface_only && solverChoice.lsm_type == LandSurfaceType::NOAHMP;'),
                   body, 'NoahMP mode read')
    standalone(body, later.start())
    require(end < later.start(), 'NoahMP read precedes surface completion')
    allowed.append(later.span())
    reject_uses(body, PROTECTED, allowed, 'coarse caller inventory')


def check(texts):
    interface(texts)
    init = function(texts[INIT], 'ERF::init_only', 'int lev, double elapsed_time')
    level = function(texts[COARSE], 'ERF::MakeNewLevelFromCoarse',
                     'int lev, Real time, const BoxArray& ba, const DistributionMapping& dm')
    for factory in (False, True):
        for netcdf in (False, True):
            ordinary_body = configured(init, netcdf, factory)
            coarse_body = configured(level, netcdf, factory)
            if netcdf:
                ordinary(ordinary_body)
                coarse(coarse_body)
            else:
                reject_uses(ordinary_body, PROTECTED, [], 'NETCDF-disabled ordinary caller')
                reject_uses(coarse_body, PROTECTED, [], 'NETCDF-disabled coarse caller')
    print('  ok   ordinary WRF branch passes the same-level PSFC field and default true mode')
    print('  ok   coarse selector, false/default calls, terrain and surface completion order')
    print('  ok   caller inventories hold with NETCDF/factory off and on')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    try:
        check(load(root))
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 WRF caller source contracts passed; GPU completion and whole-program lifetime remain unproved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
