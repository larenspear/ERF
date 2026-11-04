#!/usr/bin/env python3
"""Guard the projection divergence reader's spacing binding and completion.

Four finite source contracts cover the class/definition interface and dispatch,
one member-spacing pointer and scalar read, an automatic MFIter enclosing two
bare value-captured kernels, and three direct projection call sites in both FFT
configurations. The checked calls retain their arguments and Geometry binding.
The plotfile call is outside this caller contract.

Completion of the stretched-grid read before compute_divergence returns relies
on the previously inspected pinned AMReX MFIter semantics, enabled GPU sync,
GPU launch mode, stable streams, ordinary destruction, no graph capture or
concurrent host launches, and intended C++/AMReX resolution. Assume a live ERF,
valid initialized level spacing, valid fields/indices, disjoint outputs and
preservation of spacing, level, Geometry, iterator and stream state by other
code/callees/aliases/globals. Headers must not rewrite macros or execute runtime
statements. Undefined behavior and nonlocal transfer are excluded.

This does not prove the divergence formula, subdomain alias correctness,
positivity/bounds, delegated EB/constant-grid divergence, FFT/MLMG completion,
projection temporary ownership, plotfile callers or whole-program lifetime.
Standard-library Python only; no new arithmetic solver obligations.
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require
from check_stretched_dz_bindings import (CLASS, brace_block, direct, no_literals,
    reject_uses, scope_header, scopes, signature)
from check_stretched_dz_consumers import (declaration_scope, host_scope,
    kernel_regions, pattern, read_context, standalone)
from check_stretched_dz_mfiter import CONTROLS, enclosing, header
from momentum_geometry import call_arguments, literal, unique

SOURCE = 'Source/LinearSolvers/ERF_ComputeDivergence.cpp'
PROJECTION = 'Source/LinearSolvers/ERF_PoissonSolve.cpp'
FILES = (SOURCE, CLASS, PROJECTION)
PARAMETERS = '''int lev, MultiFab& rhs, Array<MultiFab const*,AMREX_SPACEDIM> rho0_u_const,
    MultiFab const& mf_mx, MultiFab const& mf_my, MultiFab const& mf_vx,
    MultiFab const& mf_uy, MultiFab const& ax_in, MultiFab const& ay_in,
    MultiFab const& dJ_in, Geometry const& geom_at_lev'''
DEFINITION = 'void ERF::compute_divergence('+PARAMETERS+') {'
DISPATCH = '''if (solverChoice.terrain_type == TerrainType::EB) {
    bool already_on_centroids = true;
    EB_computeDivergence(rhs, rho0_u_const, geom_at_lev, already_on_centroids);
} else if (SolverChoice::mesh_type == MeshType::ConstantDz) {
    computeDivergence(rhs, rho0_u_const, geom_at_lev);
} else {'''
STRETCHED = 'if (SolverChoice::mesh_type == MeshType::StretchedDz)'
LOOP = header('rhs', 'TilingIfNotGPU()')
POINTER = 'stretched_dz_d_ptr'
ALIAS = 'Real* stretched_dz_d_ptr = stretched_dz_d[lev].data();'
READ = 'Real inv_dz = one/stretched_dz_d_ptr[k];'
GEOMETRY = 'Vector<Geometry> geom_tmp; geom_tmp.push_back(geom[lev]);'
PERIODICITY = tuple('gradp[lev][GpVars::'+axis+'].FillBoundary(geom_tmp[0].periodicity());'
                    for axis in ('gpx', 'gpy', 'gpz'))
SUBCALL = '''compute_divergence(lev, rhs_sub[0], rho0_u_const, mfmx_sub, mfmy_sub,
    mfvx_sub, mfuy_sub, ax_sub, ay_sub, dJ_sub, geom_tmp[0]);'''
LEVELCALL = '''compute_divergence(lev, rhs_lev, rho0_u_const, *mapfac[lev][MapFacType::m_x],
    *mapfac[lev][MapFacType::m_y], *mapfac[lev][MapFacType::v_x],
    *mapfac[lev][MapFacType::u_y], *ax[lev], *ay[lev], *detJ_cc[lev], geom_tmp[0]);'''
ISUB = 'for(intisub=0;isub<subdomains[lev].size();++isub)'
CALL_SCOPES = ((ISUB,), (ISUB, 'if(lev==0&&solverChoice.use_real_bcs)'), ('if(mg_verbose>0)',))
CALLEE = r'(?<![\w:.>])(?:amrex::)?ParallelFor'


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        text = strip_comments(raw)
        for macro in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(path == CLASS and macro[1] == 'define' and macro[2].strip() == 'ERF_H_',
                    path+': unsupported macro')
        reject_uses(text, CONTROLS, [], path+' stream/sync controls')
        require(not re.search(r'\b(?:goto|try|catch|co_await|co_yield|co_return)\b', no_literals(text)),
                path+': unsupported control transfer')
        texts[path] = text
    return texts


def interface(texts):
    text = texts[SOURCE]
    source = Source(text)
    literal(source, '#include "ERF.H" #include "ERF_Utils.H" using namespace amrex; '+DEFINITION)
    _, body, definition = signature(text, 'ERF::compute_divergence')
    require(source.pos == definition.end(), 'unsupported divergence definition prefix')
    require(not text[definition.end()+len(body)+1:].strip(), 'unsupported divergence definition suffix')
    require('#' not in no_literals(body), 'unsupported divergence directive')
    expected, _, _ = signature(DEFINITION+'}', 'ERF::compute_divergence')
    declared, _, declaration = signature(texts[CLASS], 'compute_divergence', ';')
    require(declared == expected, 'divergence declaration/definition mismatch')
    cls = unique(pattern('class ERF : public amrex::AmrCore {'), texts[CLASS], 'ERF class')
    declaration_scope(texts[CLASS], cls.start(), 'ERF_H_')
    class_body = brace_block(texts[CLASS], cls.end()-1)
    require(cls.end() <= declaration.start() < declaration.end() <= cls.end()+len(class_body),
            'divergence declaration must remain in ERF')
    direct(class_body, declaration.start()-cls.end(), 'divergence declaration')
    reject_uses(texts[CLASS], ['compute_divergence'], [declaration.span()], 'divergence declarations')
    dispatch = unique(pattern(DISPATCH), body, 'divergence dispatch')
    direct(body, dispatch.start(), 'divergence dispatch')
    reject_uses(body, ['already_on_centroids', 'EB_computeDivergence', 'computeDivergence'],
                [dispatch.span()], 'delegated divergence calls')
    print('  ok   matching member interface and EB/constant/general dispatch')
    return body, dispatch.end()-1


def reader(body, outer):
    loop = unique(pattern(LOOP), no_literals(body), 'divergence automatic iterator')
    standalone(body, loop.start())
    host_scope(body, loop.start())
    require(scopes(body, loop.start()) == [('{', outer)], 'divergence loop must remain in general-grid branch')
    end = loop.end()+len(brace_block(body, loop.end()-1))
    found = [(loop, end)]
    reject_uses(body, ['MFIter'], [loop.span()], 'divergence iterator inventory')
    allowed = [loop.span()]
    for expression in (r'\b(mfi)\s*\.\s*tilebox\s*\(',
                       r'\b(?:array|const_array)\s*\(\s*(mfi)\s*\)'):
        for use in re.finditer(expression, no_literals(body)):
            enclosing(body, use.start(), found)
            allowed.append(use.span(1))
    reject_uses(body, ['mfi'], allowed, 'divergence iterator uses')
    matches = list(re.finditer(CALLEE+r'\s*\(', no_literals(body)))
    arguments = list(call_arguments(no_literals(body), CALLEE))
    regions = kernel_regions(no_literals(body))
    require(len(matches) == len(arguments) == len(regions) == 2, 'divergence launch inventory changed')
    for index, (match, args, region) in enumerate(zip(matches, arguments, regions)):
        standalone(body, match.start())
        enclosing(body, match.start(), found)
        require(len(args) == 2 and compact(body[args[0][0]:args[0][1]]) == 'bx' and
                region[3] == match.start(), 'divergence launch arguments changed')
        require(re.match(r'\s*\)\s*;', body[args[-1][1]:]), 'divergence launch must end as a statement')
        stack = scopes(body, match.start())
        require(len(stack) == 3 and scope_header(body, stack[-1][1]) ==
                (compact(STRETCHED) if index == 0 else 'else'), 'divergence kernel branch changed')
    reject_uses(body, ['ParallelFor'], [m.span() for m in matches], 'divergence launcher uses')
    # Keep both branches attached to the checked selector and its original loop.
    gate = unique(pattern(STRETCHED+' {'), body, 'stretched-grid branch')
    require(scopes(body, gate.start()) == [('{', outer), ('{', loop.end()-1)],
            'stretched selector must remain directly in the iterator')
    then_end = gate.end()+len(brace_block(body, gate.end()-1))+1
    tail = Source(body[then_end:])
    literal(tail, 'else {')
    require(scopes(body, matches[1].start())[-1] == ('{', then_end+tail.pos-1),
            'general-terrain kernel must remain in the attached else')
    print('  ok   one automatic iterator encloses both direct value-captured kernels')
    return found, regions, gate


def spacing(body, found, regions, gate):
    alias = unique(pattern(ALIAS), no_literals(body), 'same-level member spacing pointer')
    standalone(body, alias.start())
    require(scopes(body, alias.start()) == scopes(body, gate.start())+[('{', gate.end()-1)],
            'spacing pointer must be local to the stretched branch')
    read = unique(pattern(READ), no_literals(body), 'scalar spacing read')
    read_context(body, read.start(), alias, regions)
    require(regions[0][0] < read.start() < regions[0][1], 'spacing read must stay in stretched kernel')
    enclosing(body, regions[0][3], found)
    reject_uses(body, ['stretched_dz_h', 'stretched_dz_d', POINTER, 'lev', 'this'],
                [alias.span(), read.span()], 'divergence spacing uses')
    print('  ok   same-level device pointer read only as a scalar inside the stretched kernel')


def configured(body, fft):
    """Select known FFT branches and the disabled debug block, rejecting others."""
    output, stack = [], []
    for line in body.splitlines(keepends=True):
        if '#' not in no_literals(line):
            if all(active for active, _ in stack):
                output.append(line)
            continue
        m = re.fullmatch(r'\s*#\s*(\w+)\s*([^\n]*?)\s*', line)
        require(m, 'unsupported projection directive syntax')
        kind, arg = m[1], m[2].strip()
        if kind in ('ifdef', 'ifndef') and arg == 'ERF_USE_FFT':
            stack.append((fft if kind == 'ifdef' else not fft, False))
        elif kind == 'if' and arg == '0':
            stack.append((False, False))
        elif kind == 'else':
            require(stack and not arg and not stack[-1][1], 'unsupported projection else')
            stack[-1] = (not stack[-1][0], True)
        else:
            require(kind == 'endif' and stack and not arg, 'unsupported projection directive')
            stack.pop()
    require(not stack, 'unclosed projection conditional')
    return ''.join(output)


def callers(text):
    parameters, raw, definition = signature(text, 'ERF::project_momenta')
    require(parameters == [('lev','int'), ('l_time','double'), ('l_dt_d','double'),
                           ('mom_mf','Vector<MultiFab>&')], 'projection parameters changed')
    direct(text, definition.start(), 'projection definition')
    reject_uses(text, ['compute_divergence'], [(definition.end(),definition.end()+len(raw))],
                'projection file divergence uses')
    reject_uses(raw, ['stretched_dz_h', 'stretched_dz_d', POINTER], [], 'projection explicit spacing uses')
    # No call hidden in disabled debug code or an extra FFT branch is allowed.
    require(len(list(re.finditer(r'\bcompute_divergence\b', no_literals(raw)))) == 3,
            'raw projection divergence inventory changed')
    for fft in (False, True):
        body = configured(raw, fft)
        sub = unique(pattern(SUBCALL), no_literals(body), 'subdomain divergence call')
        level = list(re.finditer(pattern(LEVELCALL), no_literals(body)))
        require(len(level) == 2, 'level divergence call inventory changed')
        calls = [sub, *level]
        require([m.start() for m in calls] == sorted(m.start() for m in calls), 'projection call order changed')
        reject_uses(body, ['compute_divergence'], [m.span() for m in calls], 'projection divergence uses')
        subdomain = None
        for index, (call, expected) in enumerate(zip(calls, CALL_SCOPES)):
            standalone(body, call.start())
            host_scope(body, call.start())
            stack = scopes(body, call.start())
            require(tuple(scope_header(body, pos) for _, pos in stack) == expected,
                    'projection divergence call enclosure changed')
            if index == 0:
                subdomain = stack[0][1]
            elif index == 1:
                require(stack[0][1] == subdomain, 'correction call moved to another subdomain loop')
            else:
                require(subdomain+1+len(brace_block(body, subdomain)) < call.start(),
                        'diagnostic call precedes subdomain loop completion')
        geometry = unique(pattern(GEOMETRY), body, 'projection Geometry copy')
        direct(body, geometry.start(), 'projection Geometry')
        require(geometry.end() < sub.start(), 'projection Geometry copy must precede divergence')
        periodicity = [unique(pattern(statement), no_literals(body), 'projection periodicity call')
                       for statement in PERIODICITY]
        for call in periodicity:
            direct(body, call.start(), 'projection periodicity call')
        reject_uses(body, ['geom_tmp'], [geometry.span(), *(m.span() for m in calls),
                    *(m.span() for m in periodicity)], 'projection Geometry uses')
    print('  ok   three direct projection calls retain arguments and Geometry in both FFT configurations')


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        body, outer = interface(texts)
        found, regions, gate = reader(body, outer)
        spacing(body, found, regions, gate)
        callers(texts[PROJECTION])
    except (OSError, ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 projection divergence source contracts passed; GPU completion remains conditional on AMReX and caller state.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
