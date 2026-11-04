#!/usr/bin/env python3
"""Guard MFIter completion premises for slow-RHS and implicit spacing readers.

Five finite source contracts check six automatic iterator headers, their uses,
and enclosure of diffusion, normal momentum advection and TKE calls. Both SHOC
and ERF_IMPLICIT_W configurations are checked. The selected callee source files
must contain no enumerated stream/sync override or additional MFIter. The
advection router may only forward its const iterator to the existing EB call.

Completion is conditional on the separately audited pinned AMReX semantics,
GPU launch mode, enabled synchronization, stable streams, normal destruction,
no graph capture/concurrent host launches, and the existing TileNoZ, spacing
binding/preservation and kernel-launch contracts. Other callees, including EB,
SHOC and flux registers, must preserve iterator/stream/sync/spacing state and
must not defer host launches beyond the loop. Assume intended C++/AMReX/type
resolution, valid storage/indices, disjoint outputs, no undefined behavior,
nonlocal transfer or external macro rewriting. External headers may not execute
runtime statements. Implicit fragments are checked relative to their own entry;
their enclosing callback/include state remains an external obligation.

This adds no arithmetic, GPU execution or whole-program lifetime theorem.
Regrid/restart/destruction, projection/EB and setup-array lifetimes remain open.
Only the Python standard library is required.
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_advection import NORMAL
from check_stretched_dz_bindings import (brace_block, no_literals, reject_uses,
                                       scopes, signature)
from check_stretched_dz_consumers import (host_scope, no_outer_closures, pattern,
                                         preprocess, standalone)
from check_stretched_dz_diffusion import SCALAR, SETUP as SCALAR_SETUP
from check_stretched_dz_implicit import (SOURCE as IMPLICIT, SETUP as IMPLICIT_SETUP,
                                       instantiations)
from check_stretched_dz_mfiter import CONTROLS, enclosing, header
from check_stretched_dz_rhs import (CONSTANT, DIFF_MOM, MOM, PBL, POST, PRE,
                                  STATE, STRETCHED, TKE)
from momentum_geometry import call_arguments, unique

IPRE = 'Source/TimeIntegration/ERF_ImplicitPre.H'
IPOST = 'Source/TimeIntegration/ERF_ImplicitPost.H'
FUNCTIONS = {PRE: 'erf_slow_rhs_pre', POST: 'erf_slow_rhs_post'}
LOOPS = {PRE: ('S_data[IntVars::cons]', 'TileNoZ()', 2),
         POST: ('S_data[IntVars::cons]', 'false', 1),
         IPRE: ('S_old[IntVars::cons]', 'TileNoZ()', 1),
         IPOST: ('S_new[IntVars::cons]', 'TileNoZ()', 2)}
CALLEES = (MOM, STATE, DIFF_MOM, SCALAR, SCALAR_SETUP, CONSTANT, STRETCHED,
           NORMAL, PBL, TKE, IMPLICIT, IMPLICIT_SETUP)
FILES = (*LOOPS, *CALLEES)
GUARDS = {SCALAR: 'ERF_SCALAR_DIFFUSION_H_', PBL: 'ERF_PBLMODELS_H_',
          TKE: 'ERF_TURB_KE_SOURCES_H_'}
CALLS = {PRE: (('DiffusionSrcForState_S', 1), ('AdvectionSrcForMom', 1),
               ('DiffusionSrcForMom', 1)),
         POST: (('DiffusionSrcForState_S', 1), ('AddTurbKESources', 3)),
         IPRE: (('ImplicitDiffForStateLU_S', 1), ('ImplicitDiffForMomLU_S<0>', 1),
                ('ImplicitDiffForMomLU_S<1>', 1)),
         IPOST: (('ImplicitDiffForStateLU_S', 2),)}
CALLS_NAMES = tuple(dict.fromkeys(name.split('<')[0] for specs in CALLS.values() for name, _ in specs))


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        if path == IMPLICIT:
            raw = instantiations(raw)
        require(not re.search(r'\\\r?\n', raw), path+': unsupported line continuation')
        text = strip_comments(raw)
        for m in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(m[1] == 'define' and m[2].strip() == GUARDS.get(path),
                    path+': unsupported macro')
        reject_uses(text, CONTROLS, [], path+' stream/sync controls')
        texts[path] = text
    return texts


def select(body, macro, enabled):
    """Evaluate the supported ifdef blocks; leave OpenMP for its exact audit."""
    # Audit all configurations before removing code, including disabled code.
    omp = r'#ifdef\s+_OPENMP\s*\n\s*#pragma\s+omp\s+parallel\s+if\s*\(\s*Gpu::notInLaunchRegion\s*\(\s*\)\s*\)\s*\n\s*#endif'
    body = re.sub(omp, '', body)
    result, stack = [], []
    for line in body.splitlines(keepends=True):
        m = re.match(r'\s*#\s*(\w+)\b(.*)', line)
        if not m:
            if all(stack):
                result.append(line)
            continue
        kind, argument = m[1], m[2].strip()
        if kind == 'ifdef':
            require(argument == macro, 'unsupported solver conditional')
            stack.append(enabled)
        else:
            require(kind == 'endif' and stack and not argument, 'unsupported solver directive')
            stack.pop()
    require(not stack, 'unclosed solver conditional')
    return preprocess(''.join(result))


def caller_body(text, path, enabled):
    if path in FUNCTIONS:
        _, body, definition = signature(text, FUNCTIONS[path])
        prefix = re.sub(r'(?m)^\s*#include\s*[<"][\w./]+[>"]\s*$', '', text[:definition.start()])
        prefix = re.sub(r'\busing\s+namespace\s+amrex\s*;', '', prefix)
        require(not prefix.strip(), 'unsupported solver definition prefix')
        require(not text[definition.end()+len(body)+1:].strip(), 'unsupported solver definition suffix')
        # Parameter directives are handled by the existing RHS type contract.
        reject_uses(definition[1], ['MFIter', 'mfi', *CALLS_NAMES], [], 'solver parameter shadows')
    else:
        body = text
    require(not re.search(r'\b(?:goto|try|catch|co_await|co_yield|co_return)\b', no_literals(body)),
            'unsupported solver control transfer')
    return select(body, 'ERF_USE_EAMXX_SHOC' if path in FUNCTIONS else 'ERF_IMPLICIT_W', enabled)


def loops(body, path):
    container, tiling, count = LOOPS[path]
    matches = list(re.finditer(pattern(header(container, tiling)), body))
    require(len(matches) == count, 'solver MFIter constructor/iteration inventory changed')
    found = []
    for m in matches:
        standalone(body, m.start())
        host_scope(body, m.start())
        found.append((m, m.end()+len(brace_block(body, m.end()-1))))
    reject_uses(body, ['MFIter'], [m.span() for m, _ in found], 'solver iterator declarations')
    allowed = [m.span() for m, _ in found]
    # Unlike the earlier reader loops, these callers also use a component
    # offset and pass the iterator to advection, EB diffusion and SHOC.
    uses = (
        r'\b(mfi)\s*\.\s*(?:validbox|tilebox|growntilebox|nodaltilebox|grownnodaltilebox)\s*\(',
        r'\b(?:array|const_array)\s*\(\s*(mfi)\s*(?:,\s*flux_comp\s*)?\)',
        r'\b(?:AdvectionSrcForMom|DiffusionSrcForMom_EB|add_slow_tend|CrseAdd|FineAdd)\s*\(\s*(mfi)\s*,',
        r'\bgetMultiEBCellFlagFab\s*\(\s*\)\s*\[\s*(mfi)\s*\]',
        r'\bavg_[xyz]mom\s*\[\s*(mfi)\s*\]',
    )
    for expression in uses:
        for m in re.finditer(expression, body):
            stack = scopes(body, m.start())
            no_outer_closures(body, stack)
            require(sum(('{', loop.end()-1) in stack for loop, _ in found) == 1,
                    'iterator use must remain inside its automatic loop')
            allowed.append(m.span(1))
    reject_uses(body, ['mfi'], allowed, 'solver iterator uses')
    return found


def calls(body, path, found, enabled):
    specs = CALLS[path]
    if path == IPRE and enabled:
        specs += (('ImplicitDiffForMomLU_S<2>', 1),)
    allowed = []
    for name, count in specs:
        matches = list(re.finditer(pattern(name+'('), body))
        arguments = list(call_arguments(body, pattern(name)))
        require(len(matches) == len(arguments) == count, name+': solver call inventory changed')
        for m, args in zip(matches, arguments):
            require(args and args[0][0] == m.end(), 'solver call argument boundary')
            require(re.match(r'\s*\)\s*;', body[args[-1][1]:]), 'solver call must end as a statement')
            standalone(body, m.start())
            enclosing(body, m.start(), found)
            allowed.append(m.span())
    reject_uses(body, CALLS_NAMES, allowed, 'solver callee uses')
    if path == PRE:
        # Tau has its own iterator completion contract, outside these loops.
        tau = unique(pattern('erf_make_tau_terms('), body, 'standalone tau call')
        standalone(body, tau.start())
        host_scope(body, tau.start())
        require(not any(m.end() <= tau.start() < end for m, end in found), 'tau call nested in solver iterator')
        reject_uses(body, ['erf_make_tau_terms'], [tau.span()], 'tau uses')


def check(texts):
    for path in CALLEES:
        text, allowed = texts[path], []
        if path == MOM:
            _, body, definition = signature(text, 'AdvectionSrcForMom')
            parameter = unique(r'^\s*const\s+MFIter\s*&\s*mfi\s*,', definition[1], 'router const iterator parameter')
            allowed.append((definition.start(1)+parameter.start(), definition.start(1)+parameter.end()))
            forwarding = unique(pattern('AdvectionSrcForMom_EB(mfi,'), body, 'router EB iterator forwarding')
            standalone(body, forwarding.start())
            host_scope(body, forwarding.start())
            allowed.append((definition.end()+forwarding.start(), definition.end()+forwarding.end()))
        reject_uses(text, ['MFIter', 'mfi'], allowed, path+' callee iterator inventory')
    print('  ok   12 selected callee files: no extra iterator or explicit stream/sync override')
    for path in LOOPS:
        for enabled in (False, True):
            body = caller_body(texts[path], path, enabled)
            found = loops(body, path)
            calls(body, path, found, enabled)
        print('  ok   '+Path(path).name+': automatic iterator enclosure in both configurations')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    try:
        check(load(root))
    except (OSError, ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('5 solver MFIter source contracts passed; GPU completion and entry state remain conditional.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
