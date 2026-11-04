#!/usr/bin/env python3
"""Guard no-substep callback invocation, update enclosure and projection order.

Four finite source contracts cover dycore registration, MRI value storage and
two synchronous stage calls, two automatic MFIters enclosing eight ParallelFor
calls with fifteen value-captured kernels, and boundary/projection call order.
The callback has only guarded OpenMP, inactive in the assumed GPU launch mode.
No explicit spacing-vector use is admitted in this callback.

Completion of the enclosed kernels before boundary/projection calls and callback
return is conditional on the inspected pinned AMReX semantics, enabled GPU sync,
stable streams, ordinary destruction, no graph capture/concurrent host launches,
valid objects/indices and disjoint outputs. Assume prior ownership, spacing,
kernel and slow/acoustic callback contracts and intended C++/AMReX/std::function
resolution. Other code/callees must preserve iterator, stream/sync and captured
state through aliases/globals and cannot defer host launches. Temporary arrays
must remain valid for their GPU uses; that lifetime is not proved here. External
headers cannot rewrite macros or execute runtime statements. Undefined behavior,
nonlocal transfer and stored callback invocation outside dycore are excluded.

This is not a numerical, projection-solver, GPU execution or whole-program
lifetime theorem. Projection, boundary-condition and EB redistribution internals,
temporary AsyncVector lifetimes, initial GPU state, other readers and complete
regrid/restart/destruction remain obligations. Standard-library Python only.
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require
from check_stretched_dz_bindings import (DYCORE, FRAGMENTS, MRI, brace_block,
    direct, no_literals, reject_uses, scope_header, scopes, signature)
from check_stretched_dz_callback_completion import (INCLUDES, REGISTRATION,
    configured, dycore, include_inventory)
from check_stretched_dz_consumers import (declaration_scope, host_scope,
                                        pattern, standalone)
from check_stretched_dz_mfiter import CONTROLS, enclosing, header
from momentum_geometry import call_arguments, literal, unique

CALLBACK = 'Source/TimeIntegration/ERF_TI_no_substep_fun.H'
FILES = (DYCORE, MRI, *FRAGMENTS)
HEADER = '''auto no_substep_fun = [&,bogus_large_value_d=bogus_large_value](
    Vector<MultiFab>& S_sum, Vector<MultiFab>& S_old, Vector<MultiFab>& F_slow,
    const double time_for_fp, const double slow_dt_d, const int nrk) {'''
TYPE = 'std::function<void(T&, T&, T&, const double, const double, int)>'
INVOKE = 'no_substep(*S_sum, S_old, *F_slow, time + nsubsteps*dtau, nsubsteps*dtau, nrk);'
TILING = '''const bool do_tiling = (solverChoice.terrain_type == TerrainType::MovingFittedMesh)
    ? false : TilingIfNotGPU();'''
LOOPS = (header('S_sum[IntVars::cons]', 'do_tiling'),
         header('S_sum[IntVars::cons]', 'TilingIfNotGPU()'))
BCS = '''apply_bcs(S_sum, time_for_fp, S_sum[IntVars::cons].nGrow(), S_sum[IntVars::xmom].nGrow(),
    fast_only=true, vel_and_mom_synced=false);'''
PROJECTION = '''if (solverChoice.anelastic[level]) {
    bool have_tb = (thin_xforce[0] || thin_yforce[0] || thin_zforce[0]);
    if (have_tb) { project_velocity_tb(level, slow_dt, S_sum); }
    else { project_momenta(level, time_for_fp, slow_dt, S_sum); }
}'''
# Exact launch arguments and lambda arities, in source order. Bodies remain
# uninterpreted; each complete launch must stay inside its original iterator.
CELL = (('bx', 'ncomp_fast[IntVars::cons]'), (4,))
FACES = (('tbx', 'tby', 'tbz'), (3, 3, 3))
LAUNCHES = (CELL, (('tbx', 'tby'), (3, 3)), (('tbz',), (3,)),
            CELL, FACES, FACES, CELL, FACES)


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        text = strip_comments(raw)
        for m in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(path == MRI and m[1] == 'define' and m[2].strip() == 'ERF_MRI_H',
                    path+': unsupported macro')
        reject_uses(text, CONTROLS, [], path+' stream/sync controls')
        require(not re.search(r'\b(?:goto|try|catch|co_await|co_yield|co_return)\b', no_literals(text)),
                path+': unsupported control transfer')
        if path in FRAGMENTS:
            include_inventory(text, INCLUDES[path])
            if path != CALLBACK:
                reject_uses(text, ['no_substep_fun', 'set_no_substep'], [], 'auxiliary callback uses')
        texts[path] = text
    return texts


def registration(text):
    dycore(text)
    _, body, _ = signature(text, 'ERF::advance_dycore')
    sequence = unique(pattern(REGISTRATION), body, 'no-substep registration sequence')
    reject_uses(body, ['no_substep_fun', 'set_no_substep'], [sequence.span()], 'dycore no-substep uses')


def else_of(text, opening, expected):
    """An else scope must immediately follow the expected if's closing brace."""
    clause = unique(r'\belse\s*$', no_literals(text[:opening]), 'else clause')
    prefix = no_literals(text[:clause.start()]).rstrip()
    require(prefix.endswith('}'), 'else must follow a complete braced if')
    stack = scopes(text, len(prefix)-1)
    require(stack and stack[-1][0] == '{' and
            scope_header(text, stack[-1][1]) == expected, 'else attached to wrong if')


def mri(text):
    reject_uses(text, ['MFIter', 'mfi'], [], 'MRI outer iterator inventory')
    cls = unique(pattern('template<class T> class MRISplitIntegrator {'), text, 'MRI class')
    declaration_scope(text, cls.start(), 'ERF_MRI_H')
    body = brace_block(text, cls.end()-1)
    require('#' not in no_literals(body), 'unsupported MRI class directive')
    member = unique(pattern(TYPE+' no_substep;'), body, 'no-substep value member')
    direct(body, member.start(), 'no-substep member')
    setter = unique(pattern('void set_no_substep('), body, 'no-substep setter')
    direct(body, setter.start(), 'no-substep setter')
    source = Source(body[setter.start():])
    literal(source, 'void set_no_substep('+TYPE+' F) { no_substep = F; }')
    advance = unique(pattern('double advance(T& S_old, T& S_new, double time, const double time_step) {'),
                     body, 'MRI advance')
    direct(body, advance.start(), 'MRI advance')
    stage = brace_block(body, advance.end()-1)
    invokes = list(re.finditer(pattern(INVOKE), stage))
    pre = list(re.finditer(pattern('slow_rhs_pre('), stage))
    post = list(re.finditer(pattern('slow_rhs_post('), stage))
    require(len(invokes) == len(pre) == len(post) == 2, 'MRI no-substep stage inventory changed')
    expected = (('if(!anelastic)', 'for(intnrk=0;nrk<3;nrk++)', 'else'),
                ('else', 'for(intnrk=0;nrk<2;nrk++)'))
    for index, invoke in enumerate(invokes):
        standalone(stage, invoke.start())
        host_scope(stage, invoke.start())
        stack = scopes(stage, invoke.start())
        require(tuple(scope_header(stage, pos) for _, pos in stack) == expected[index],
                'no-substep invocation scope changed')
        else_of(stage, stack[-1 if index == 0 else 0][1],
                'if(!no_substepping)' if index == 0 else 'if(!anelastic)')
        require(pre[index].start() < invoke.start() < post[index].start(),
                'no-substep call must lie between its slow callbacks')
    reject_uses(body, ['no_substep', 'set_no_substep', 'this'],
        [member.span(), (setter.start(), setter.start()+source.pos),
         *((advance.end()+m.start(), advance.end()+m.end()) for m in invokes)], 'MRI no-substep uses')
    print('  ok   value-stored no-substep callback invoked directly in both MRI stage paths')


def callback_body(text):
    source = Source(text)
    literal(source, HEADER)
    body = brace_block(text, source.pos-1)
    tail = Source(text[source.pos+len(body):])
    literal(tail, '};')
    tail.done()
    reject_uses(body, ['no_substep_fun', 'set_no_substep', 'this', 'stretched_dz_h', 'stretched_dz_d'],
                [], 'no-substep callback uses')
    # Only OpenMP guarded by notInLaunchRegion is admitted; no other directives.
    body = configured(body, {})
    require('#' not in no_literals(body), 'unsupported no-substep directive')
    return body


def loops(body):
    found = []
    for index, statement in enumerate(LOOPS):
        loop = unique(pattern(statement), body, 'no-substep automatic iterator')
        standalone(body, loop.start())
        host_scope(body, loop.start())
        stack = scopes(body, loop.start())
        require(len(stack) == index+1, 'no-substep loop enclosure changed')
        outer = stack[0][1]
        direct(body, outer, 'update block')
        standalone(body, outer)
        if index == 1:
            require(outer == scopes(body, found[0][0].start())[0][1] and
                    scope_header(body, stack[1][1]) == 'if(solverChoice.terrain_type==TerrainType::EB)',
                    'second update loop must remain in the shared update block under the EB gate')
        found.append((loop, loop.end()+len(brace_block(body, loop.end()-1))))
    require(found[0][1] < found[1][0].start(), 'update loops must remain ordered and disjoint')
    tiling = unique(pattern(TILING), body, 'moving-terrain tiling selection')
    require(scopes(body, tiling.start()) == scopes(body, found[0][0].start()) and
            tiling.end() <= found[0][0].start(), 'tiling selection must precede first loop')
    standalone(body, tiling.start())
    reject_uses(body, ['do_tiling'], [tiling.span(), found[0][0].span()], 'tiling selector uses')
    reject_uses(body, ['MFIter'], [m.span() for m, _ in found], 'iterator declarations')
    allowed = [m.span() for m, _ in found]
    for expression in (r'\b(mfi)\s*\.\s*(?:tilebox|nodaltilebox)\s*\(',
                       r'\b(?:array|const_array)\s*\(\s*(mfi)\s*\)'):
        for use in re.finditer(expression, no_literals(body)):
            enclosing(body, use.start(), found)
            allowed.append(use.span(1))
    reject_uses(body, ['mfi'], allowed, 'iterator uses')
    return found, outer+1+len(brace_block(body, outer))


def launches(body, found):
    masked = no_literals(body)
    callee = r'(?<![\w:.>])(?:amrex::)?ParallelFor'
    matches = list(re.finditer(callee+r'\s*\(', masked))
    arguments = list(call_arguments(masked, callee))
    require(len(matches) == len(arguments) == len(LAUNCHES), 'update launch inventory changed')
    for index, (match, args, (prefix, arities)) in enumerate(zip(matches, arguments, LAUNCHES)):
        standalone(body, match.start())
        # Six calls in the main update loop, two in the EB update loop.
        enclosing(body, match.start(), [found[0 if index < 6 else 1]])
        require(len(args) == len(prefix)+len(arities), 'launch argument count changed')
        require(tuple(compact(body[lo:hi]) for lo, hi in args[:len(prefix)]) == prefix,
                'launch box/component arguments changed')
        require(re.match(r'\s*\)\s*;', body[args[-1][1]:]), 'launch must end as a statement')
        for (lo, hi), arity in zip(args[len(prefix):], arities):
            kernel = body[lo:hi]
            signature = '[=] AMREX_GPU_DEVICE (int i, int j, int k'+(', int nn' if arity == 4 else '')+')'
            start = re.match(r'\s*'+pattern(signature)+r'\s*(?:noexcept\s*)?\{', kernel)
            require(start, 'launch requires a bare value-captured device kernel')
            end = start.end()+len(brace_block(kernel, start.end()-1))
            require(kernel[end:].strip() == '}', 'kernel must fill its launch argument')
    reject_uses(body, ['ParallelFor'], [m.span() for m in matches], 'launch aliases')
    print('  ok   two automatic update iterators enclose eight launches and fifteen value-captured kernels')


def completion_order(body, after):
    bcs = unique(pattern(BCS), body, 'boundary call')
    direct(body, bcs.start(), 'boundary call')
    require(after < bcs.start(), 'boundary call must follow destruction of both update iterators')
    source = Source(body[bcs.start():])
    literal(source, BCS+PROJECTION)
    reject_uses(body, ['apply_bcs', 'project_velocity_tb', 'project_momenta', 'have_tb'],
                [(bcs.start(), bcs.start()+source.pos)], 'boundary/projection uses')
    interval = unique(pattern('amrex::Real slow_dt = static_cast<Real>(slow_dt_d);'), body, 'stage interval')
    direct(body, interval.start(), 'stage interval')
    require(interval.end() < after, 'stage interval must precede the update block end')
    print('  ok   boundary update and gated projection follow the completed update loops')


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        registration(texts[DYCORE])
        mri(texts[MRI])
        body = callback_body(texts[CALLBACK])
        found, after = loops(body)
        launches(body, found)
        completion_order(body, after)
    except (OSError, ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 no-substep completion source contracts passed; projection and temporary lifetimes remain conditional.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
