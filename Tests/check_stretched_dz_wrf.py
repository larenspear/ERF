#!/usr/bin/env python3
"""Guard WRF nominal-grid construction, spacing refresh and terrain-call order.

Four finite source contracts check the terrain/face-averaging gates and the
surface-only return, the complete grid-construction-to-terrain slice (including
both precision tolerances), the PH/PHB requests and failure-gate assignments,
and all explicit spacing/nominal-grid uses in init_from_wrfinput. Both profile
and geometric branches must join before the same-level refresh and terrain call.
This is a source-shape contract, not an arithmetic or GPU-lifetime theorem.

Assume valid initialized storage, level and field indices, successful allocations,
ordinary C++/AMReX semantics and intended type/callee resolution. I/O dispatch
must reach the PH/PHB handlers with truthful var_name/success values; its loop
bounds, field-list preservation, data loading and all other calls are external
contracts. Those calls preserve nominal/spacing storage through implicit this,
aliases and globals. The separate refresh arithmetic/copy contract applies;
terrain construction preserves nominal nodes. No external macro rewriting,
nonlocal transfer, concurrency or undefined behavior is modeled. WRF callers,
GPU completion, positivity, profile accuracy, solver convergence, physical mesh
validity, projection/EB preservation and setup-array lifetime remain unproved.
Only the Python standard library is needed.

usage: check_stretched_dz_wrf.py [ERF_ROOT]
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
from momentum_geometry import unique

WRF = 'Source/Initialization/ERF_InitFromWRFInput.cpp'
FILES = (WRF,)
REFRESH = 'update_stretched_dz(lev, zlevels_stag, stretched_dz_h, stretched_dz_d);'
TERRAIN = '''make_terrain_fitted_coords(lev, geom[lev], *z_phys_nd[lev], zlevels_stag[lev],
    phys_bc_type, fine_terrain,
    (fine_terrain == FineTerrain::Transform) ? &z_phys_interp : nullptr);'''
PROTECTED = ('zlevels_stag', 'stretched_dz_h', 'stretched_dz_d', 'update_stretched_dz',
             'make_terrain_fitted_coords', 'build_geometric_zlevels', 'nlay',
             'use_file_profile', 'dz_prof')
STRING = r'"(?:\\.|[^"\\])*"'


def take(source, statement):
    return source.take(pattern(statement))


def prints(source, operands):
    operand = r'(?:'+STRING+r'\s*)+|'+'|'.join('(?:'+pattern(v)+')' for v in operands)
    while re.match(r'\s*(?:amrex\s*::\s*)?Print\s*\(', source.text[source.pos:]):
        source.take(r'(?:amrex\s*::\s*)?Print\s*\(\s*\)\s*(?:<<\s*(?:'+operand+r')\s*)+;')


def assertion(source, expression):
    take(source, 'AMREX_ALWAYS_ASSERT_WITH_MESSAGE('+expression+',')
    source.take(r'(?:'+STRING+r'\s*)+')
    take(source, ');')


def grid_contract(text):
    source = Source(text)
    # Read both compile-time alternatives; no branch may hide extra statements.
    take(source, '''#ifdef AMREX_USE_FLOAT
        const Real tol = Real(1.e-4);
        #else
        const Real tol = Real(1.e-8);
        #endif
        const int nlay = static_cast<int>(zlevels_stag[lev].size()) - 1;
        bool use_file_profile = solverChoice.wrfinput_zlevels_from_file;
        Vector<Real> dz_prof;
        if (use_file_profile) {
            Long ncol_min = 0;
            dz_prof = wrf_mean_layer_thickness(mf_PH, *mf_PHB, geom[lev], ncol_min);''')
    assertion(source, 'static_cast<int>(dz_prof.size()) == nlay')
    take(source, '''const Long ncol_domain = static_cast<Long>(geom[lev].Domain().length(0)) *
        static_cast<Long>(geom[lev].Domain().length(1));
        if (ncol_min == 0) {''')
    operands = ('lev', 'ncol_min', 'ncol_domain', 'z_top', 'dz_tot', 'built_dz0',
                'built_dzt', 'dz_prof[0]', 'dz_prof[nlay-1]', '(z_top-dz_tot)',
                'Real(100)*a_ramp*sfrac[nlay-1]', 'r_min', 'r_max', 'gs.dz0', 'gs.ratio')
    prints(source, operands)
    take(source, 'use_file_profile = false; } else if (ncol_min < ncol_domain) {')
    prints(source, operands)
    take(source, '''} }
        if (use_file_profile) {
            Real dz_tot = zero;
            for (int k(0); k<nlay; ++k) {''')
    assertion(source, 'dz_prof[k] > zero')
    take(source, '''dz_tot += dz_prof[k];
            }
            Vector<Real> sfrac(nlay, zero);
            {
                Real c = zero;
                for (int k(0); k<nlay; ++k) { sfrac[k] = c/dz_tot; c += dz_prof[k]; }
            }
            Real wsum = zero;
            for (int k(0); k<nlay; ++k) { wsum += sfrac[k]*dz_prof[k]; }
            Real a_ramp = (wsum > zero) ? (z_top - dz_tot)/wsum : zero;
            bool ramp_closes = true;
            if (one + a_ramp*sfrac[nlay-1] < Real(0.1)) {''')
    prints(source, operands)
    take(source, '''a_ramp = zero;
                ramp_closes = false;
            }
            const Real zscale = ramp_closes ? one : z_top/dz_tot;
            zlevels_stag[lev][0] = zero;
            for (int k(1); k<=nlay; ++k) {
                const Real w = zscale * (one + a_ramp*sfrac[k-1]);
                zlevels_stag[lev][k] = zlevels_stag[lev][k-1] + w*dz_prof[k-1];
            }
            zlevels_stag[lev][nlay] = z_top;
            Real r_min = one;
            Real r_max = one;
            for (int k(1); k<nlay-1; ++k) {
                const Real r = (zlevels_stag[lev][k+2] - zlevels_stag[lev][k+1])
                             / (zlevels_stag[lev][k+1] - zlevels_stag[lev][k]);
                if (k == 1) {
                    r_min = r; r_max = r;
                } else {
                    r_min = std::min(r_min,r); r_max = std::max(r_max,r);
                }
            }
            const Real built_dz0 = zlevels_stag[lev][1] - zlevels_stag[lev][0];
            const Real built_dzt = zlevels_stag[lev][nlay] - zlevels_stag[lev][nlay-1];''')
    prints(source, operands)
    take(source, 'if (r_min < Real(0.5) || r_max > Real(2.0)) {')
    prints(source, operands)
    take(source, '''}
        } else {
            const Real dz0_max = wrf_max_first_layer_thickness(mf_PH, *mf_PHB,
                boxes_at_level[lev][0].smallEnd(2));
            const erf_grid_utils::GeometricStretch gs =
                erf_grid_utils::build_geometric_zlevels(zlevels_stag[lev], dz0_max, z_top, tol);''')
    assertion(source, 'gs.converged')
    prints(source, operands)
    take(source, '}')
    assertion(source, 'std::fabs(zlevels_stag[lev].back() - z_top) <= tol')
    take(source, REFRESH)
    take(source, TERRAIN)
    source.done()


def directives(text):
    """Audit directive syntax/balance without choosing one feature configuration."""
    stack = []
    for match in re.finditer(r'(?m)^[ \t]*#\s*(\w+)\b([^\n]*)', text):
        kind, arg = match[1], match[2].strip()
        if kind == 'ifdef':
            require(arg in ('ERF_USE_NETCDF', 'AMREX_USE_GPU', 'AMREX_USE_FLOAT', '_OPENMP', 'AMREX_DEBUG'),
                    'unsupported WRF conditional')
            stack.append((arg, False))
        elif kind == 'else':
            require(stack and not arg and not stack[-1][1], 'invalid WRF else')
            stack[-1] = (stack[-1][0], True)
        elif kind == 'endif':
            require(stack and not arg, 'invalid WRF endif')
            stack.pop()
        elif kind == 'include':
            require(re.fullmatch(r'[<"][\w./-]+[>"]', arg), 'unsupported WRF include')
        elif kind == 'pragma':
            require(arg == 'omp parallel if (amrex::Gpu::notInLaunchRegion())' and
                    stack and stack[-1] == ('_OPENMP', False), 'unsupported WRF pragma')
        else:
            raise ValueError('unsupported WRF directive')
    return stack


def load(root):
    raw = (root/WRF).read_text()
    require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
            'unsupported WRF lexical form')
    text = strip_comments(raw)
    require(not directives(text), 'unclosed WRF directive')
    _, body, definition = signature(text, 'ERF::init_from_wrfinput')
    require(re.fullmatch(pattern('int lev, MultiFab& mf_PSFC_lev, bool read_atmos_state'), definition[1].strip()),
            'WRF initializer signature changed')
    require(not scopes(text, definition.start()), 'WRF definition is nested')
    require(directives(text[:definition.start()]) == [('ERF_USE_NETCDF', False)],
            'WRF definition must have only its NETCDF guard')
    require(not re.search(r'(?m)^\s*#\s*include\b', no_literals(body)), 'unsupported WRF runtime include')
    require(not re.search(r'\b(?:goto|try|catch|co_await|co_yield|co_return)\b', no_literals(body)),
            'unsupported WRF control transfer')
    return body


def header(body, opening, statement):
    return unique(pattern(statement)+r'\s*$', body[:opening+1], 'WRF scope '+statement)


def availability(body, terrain):
    setup = unique(pattern('bool compute_terrain_here = true;'), body, 'terrain flag initialization')
    direct(body, setup.start(), 'terrain flag initialization')
    allowed = [setup.span(), terrain.span()]
    for name in ('PH', 'PHB'):
        request = unique(pattern('NC_names.push_back("'+name+'");'), body, name+' request')
        direct(body, request.start(), name+' request')
        require(request.end() < setup.start(), name+' requested after terrain flag initialization')
        branch = unique(pattern('if (var_name == "'+name+'") {'), body, name+' handler')
        require(not directives(body[:branch.start()]), 'PH/PHB handler must not have a feature guard')
        stack = scopes(body, branch.start())
        require(len(stack) == 2 and all(kind == '{' for kind, _ in stack), 'PH/PHB handler loop scope')
        outer = header(body, stack[0][1], 'for (int idx = 0; idx < num_boxes_at_level[lev]; idx++) {')
        direct(body, outer.start(), 'WRF file loop')
        inner = header(body, stack[1][1], 'for (int ivar = 0; ivar < nvar; ++ivar) {')
        standalone(body, inner.start())
        require(setup.end() < outer.start() < branch.start() < terrain.start(), 'terrain gate order')
        if name == 'PH':
            standalone(body, branch.start())
        else:
            require(re.search(r'\}\s*else\s*$', body[:branch.start()]), 'PHB must follow the PH handler')
            # Ensure that the preceding sibling is precisely PH, not another gate.
            require(body[ph_end:branch.start()].strip() == 'else', 'PHB must be the PH else-if')
        contents = brace_block(body, branch.end()-1)
        source = Source(contents)
        success = take(source, 'if (success) {')
        unmodeled = brace_block(contents, success.end()-1)
        source.pos += len(unmodeled)
        take(source, '} else {')
        prints(source, ('var_name', 'std::endl'))
        failure = take(source, 'compute_terrain_here = false;')
        allowed.append((branch.end()+failure.start(), branch.end()+failure.end()))
        take(source, '}')
        source.done()
        ph_end = branch.end()+len(contents)+1
    remap = unique(pattern('if (compute_terrain_here && read_atmos_state) {'), body, 'atmosphere remap gate')
    direct(body, remap.start(), 'atmosphere remap gate')
    require(terrain.end() < remap.start(), 'terrain gate must precede atmosphere remap')
    allowed.append(remap.span())
    reject_uses(body, ['compute_terrain_here'], allowed, 'terrain availability gate')


def check(body):
    refresh = unique(pattern(REFRESH), body, 'WRF refresh')
    stack = scopes(body, refresh.start())
    require(len(stack) == 2 and all(kind == '{' for kind, _ in stack), 'refresh needs exactly two terrain gates')
    terrain = header(body, stack[0][1], 'if (compute_terrain_here) {')
    direct(body, terrain.start(), 'terrain gate')
    fitted = header(body, stack[1][1], 'if (!solverChoice.avg_grid_faces_to_nodes) {')
    standalone(body, fitted.start())
    direct(body[terrain.end():], fitted.start()-terrain.end(), 'fitted-coordinate gate')
    build = unique(pattern(TERRAIN), body, 'terrain construction')
    require(scopes(body, build.start()) == stack, 'refresh/terrain scopes differ')
    require(refresh.end() < build.start(), 'terrain construction precedes refresh')
    end = terrain.end()+len(brace_block(body, terrain.end()-1))+1
    leave = unique(pattern('if (!read_atmos_state) {'), body, 'surface-only exit')
    direct(body, leave.start(), 'surface-only exit')
    source = Source(brace_block(body, leave.end()-1))
    prints(source, ('lev',))
    ret = take(source, 'return;')
    source.done()
    require(end < leave.start(), 'surface-only return must follow terrain block')
    returns = [(leave.end()+ret.start(), leave.end()+ret.end())]
    # These returns belong to existing reduction/counting closures, not the
    # initializer. Their bodies are external to the spacing-order contract.
    for declaration in ('[=] AMREX_GPU_DEVICE (int i, int j, int k) -> ReduceTuple {',
                        'auto count_flag = [&] (Real target) -> Real {'):
        closure = unique(pattern(declaration), body, 'return-containing closure')
        returns.append((closure.start(), closure.end()+len(brace_block(body, closure.end()-1))+1))
    reject_uses(body, ['return'], returns, 'WRF return inventory')
    print('  ok   terrain gates and surface-only return preserve refresh placement')

    grid_contract(body[fitted.end():build.end()])
    print('  ok   complete profile/geometric branches join before refresh and terrain construction')
    availability(body, terrain)
    print('  ok   PH/PHB requests and failure assignments guard terrain availability')
    reject_uses(body, PROTECTED, [(fitted.end(), build.end())], 'WRF nominal/spacing inventory')
    print('  ok   no other explicit nominal/spacing uses in the initializer')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    try:
        check(load(root))
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 WRF spacing source contracts passed; caller ordering and GPU lifetime remain unproved.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
