#!/usr/bin/env python3
"""Connect selected spacing-reader completion to slow-RHS callback return.

Four finite source contracts guard dycore's definition/registration/advance
sequence, both MRI callback members/setters and four synchronous stage calls,
and both slow-RHS callbacks' calls and implicit-include order. NETCDF, SHOC and
implicit-W alternatives are checked. Enumerated stream/sync overrides and
unsupported macros/includes fail closed in the dycore, MRI and eight statement
fragments. The separate acoustic-substep implicit include is scanned, but its
enclosure is not covered by the slow-RHS callback contracts.

Assume the prior ownership, spacing-preservation, kernel, TileNoZ and MFIter
contracts and the inspected pinned AMReX semantics. Completion of the selected
readers before callback return requires GPU launch mode, enabled synchronization,
stable streams, normal destruction, no graph capture/concurrent host launches,
valid objects/indices, intended C++/AMReX/std::function/type resolution, disjoint
outputs, and no undefined behavior/nonlocal transfer. Other calls must preserve
stream/sync/spacing and captured state through aliases/globals and must not
defer host launches. External headers cannot rewrite macros or execute runtime
statements. The callbacks remain stored afterward; invocation outside this
same-level definition-to-advance path is excluded.

This does not prove initial GPU state, completion of every callback operation,
or whole-program storage lifetime. Acoustic/no-substep paths, other readers,
projection/EB, setup arrays and regrid/restart/destruction remain obligations.
No arithmetic theorem or GPU execution is added; standard-library Python only.
"""
import itertools
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (DYCORE, FRAGMENTS, MRI, binding,
    brace_block, call, direct, no_literals, reject_uses, scope_header, scopes,
    shoc, signature)
from check_stretched_dz_consumers import (SLOW_H, declaration_scope, host_scope,
                                        pattern, standalone)
from check_stretched_dz_mfiter import CONTROLS
from check_stretched_dz_rhs import PRE, POST
from check_stretched_dz_solver_mfiter import IPRE, IPOST
from momentum_geometry import literal, unique

CPRE = 'Source/TimeIntegration/ERF_TI_slow_rhs_pre.H'
CPOST = 'Source/TimeIntegration/ERF_TI_slow_rhs_post.H'
LIMIT = 'Source/TimeIntegration/ERF_ScalarLimitPost.H'
FILES = (DYCORE, MRI, *FRAGMENTS, PRE, POST, SLOW_H)
GUARDS = {MRI: 'ERF_MRI_H', SLOW_H: 'ERF_SLOW_INTEGRATION_H_'}
MACROS = ('ERF_USE_NETCDF', 'ERF_USE_EAMXX_SHOC', 'ERF_IMPLICIT_W')
CALLBACKS = {CPRE: ('pre', PRE, IPRE), CPOST: ('post', POST, IPOST)}
HEADERS = {
    CPRE: '''auto slow_rhs_fun_pre = [&,zero_d=zero,one_d=one](
        Vector<MultiFab>& S_rhs, Vector<MultiFab>& S_old, Vector<MultiFab>& S_data,
        const double old_step_time, const double old_stage_time,
        const double new_stage_time, const int nrk) {''',
    CPOST: '''auto slow_rhs_fun_post = [&,bogus_large_value_d=bogus_large_value](
        Vector<MultiFab>& S_rhs, Vector<MultiFab>& S_old, Vector<MultiFab>& S_new,
        Vector<MultiFab>& S_data, const double old_step_time,
        const double old_stage_time, const double new_stage_time, const int nrk) {''',
}
REGISTRATION = '''
    #include "ERF_TI_no_substep_fun.H"
    #include "ERF_TI_substep_fun.H"
    #include "ERF_TI_slow_rhs_pre.H"
    #include "ERF_TI_slow_rhs_post.H"
    MRISplitIntegrator<Vector<MultiFab> >& mri_integrator = *mri_integrator_mem[level];
    mri_integrator.set_slow_rhs_pre(slow_rhs_fun_pre);
    mri_integrator.set_slow_rhs_post(slow_rhs_fun_post);
    mri_integrator.set_acoustic_substepping(acoustic_substepping_fun);
    mri_integrator.set_slow_fast_timestep_ratio(fixed_mri_dt_ratio > 0 ? fixed_mri_dt_ratio : dt_mri_ratio[level]);
    mri_integrator.set_no_substep(no_substep_fun);
    mri_integrator.advance(state_old, state_new, old_time, dt_advance);'''
INCLUDES = {path: () for path in FRAGMENTS}
INCLUDES.update({CPRE: ('ERF_SrcHeaders.H', 'ERF_ImplicitPre.H'),
                 CPOST: ('ERF_SrcHeaders.H', 'ERF_ImplicitPost.H', 'ERF_ScalarLimitPost.H'),
                 FRAGMENTS[2]: ('ERF_SrcHeaders.H', 'ERF_ImplicitPre.H')})
PRE_GATE = 'if((solverChoice.vert_implicit_fac[level][nrk]>zero_d)&&solverChoice.implicit_before_substep)'
INCLUDE = r'(?m)^[ \t]*#\s*include\s+"([\w.]+)"[ \t]*$'


def include_inventory(text, expected):
    matches = list(re.finditer(INCLUDE, text))
    require([m[1] for m in matches] == list(expected), 'statement include inventory/order changed')
    reject_uses(no_literals(text), ['include'], [m.span() for m in matches], 'statement includes')
    return matches


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        text = strip_comments(raw)
        for m in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(m[1] == 'define' and m[2].strip() == GUARDS.get(path), path+': unsupported macro')
        reject_uses(text, CONTROLS, [], path+' stream/sync controls')
        require(not re.search(r'\b(?:goto|try|catch|co_await|co_yield|co_return)\b', no_literals(text)),
                path+': unsupported control transfer')
        if path in INCLUDES:
            include_inventory(text, INCLUDES[path])
            if path not in CALLBACKS:
                reject_uses(text, ['slow_rhs_fun_pre', 'slow_rhs_fun_post',
                    'set_slow_rhs_pre', 'set_slow_rhs_post'], [], 'auxiliary fragment callback uses')
        texts[path] = text
    return texts


def configured(body, options):
    """Evaluate only the supported callback conditionals and guarded OpenMP."""
    omp = r'#ifdef\s+_OPENMP\s*\n\s*#pragma\s+omp\s+parallel\s+if\s*\(\s*(?:amrex::)?Gpu::notInLaunchRegion\s*\(\s*\)\s*\)\s*\n\s*#endif'
    body = re.sub(omp, '', body)
    output, stack = [], []
    for line in body.splitlines(keepends=True):
        if '#' not in no_literals(line):
            if all(stack):
                output.append(line)
            continue
        m = re.fullmatch(r'\s*#\s*(\w+)\s*([^\n]*?)\s*', line)
        require(m, 'unsupported callback directive syntax')
        kind, arg = m[1], m[2].strip()
        if kind == 'ifdef' and arg in options:
            stack.append(options[arg])
        elif kind == 'if' and re.fullmatch(pattern('defined(ERF_USE_NETCDF)'), arg):
            stack.append(options['ERF_USE_NETCDF'])
        elif kind == 'endif':
            require(stack and not arg, 'unmatched callback endif')
            stack.pop()
        else:
            require(kind == 'include' and re.fullmatch(r'"[\w.]+"', arg), 'unsupported callback directive')
            if all(stack):
                output.append(line)
    require(not stack, 'unclosed callback conditional')
    return ''.join(output)


def callback_body(text, path):
    source = Source(text)
    literal(source, '#include "ERF_SrcHeaders.H" '+HEADERS[path])
    body = brace_block(text, source.pos-1)
    tail = Source(text[source.pos+len(body):])
    literal(tail, '};')
    tail.done()
    reject_uses(body, ['slow_rhs_fun_pre', 'slow_rhs_fun_post', 'this'], [], 'callback captures/aliases')
    return body


def callback(texts, path):
    kind, rhs, implicit = CALLBACKS[path]
    raw_body = callback_body(texts[path], path)
    for flags in itertools.product((False, True), repeat=len(MACROS)):
        options = dict(zip(MACROS, flags))
        body = configured(raw_body, options)
        expected = (Path(implicit).name,) + ((Path(LIMIT).name,) if path == CPOST else ())
        includes = include_inventory(body, expected)
        declared, _, decl = signature(shoc(texts[SLOW_H], options['ERF_USE_EAMXX_SHOC']), 'erf_slow_rhs_'+kind, ';')
        parameters, _, _ = signature(shoc(texts[rhs], options['ERF_USE_EAMXX_SHOC']), 'erf_slow_rhs_'+kind)
        require([typ for _, typ in declared] == [typ for _, typ in parameters], 'RHS declaration/definition mismatch')
        declaration_scope(shoc(texts[SLOW_H], options['ERF_USE_EAMXX_SHOC']), decl.start(), GUARDS[SLOW_H])
        arguments, values = call(body, 'erf_slow_rhs_'+kind)
        occurrence = unique(pattern('erf_slow_rhs_'+kind+'('), body, 'RHS call')
        direct(body, occurrence.start(), 'synchronous RHS call')
        require(re.match(r'\s*\)\s*;', body[arguments[-1][1]:]), 'RHS call must end as a statement')
        binding(values, parameters, {'level': ('int', 'level'), 'nrk': ('int', 'nrk'),
            'geom': ('constGeometry', 'fine_geom'),
            'stretched_dz_d': ('Gpu::DeviceVector<Real>&', 'stretched_dz_d[level]')}, 'callback RHS binding')
        spacing = arguments[[key for key, _ in parameters].index('stretched_dz_d')]
        reject_uses(body, ['stretched_dz_d'], [spacing], 'callback device spacing uses')
        reject_uses(body, ['erf_slow_rhs_pre', 'erf_slow_rhs_post'], [occurrence.span()], 'RHS callback callee uses')
        include = includes[0]
        host_scope(body, include.start())
        stack = scopes(body, include.start())
        expected_scope = (PRE_GATE,) if path == CPRE else ()
        require(tuple(scope_header(body, pos) for _, pos in stack) == expected_scope,
                'implicit include must have its supported callback scope')
        require(arguments[-1][1] < include.start(), 'implicit include must follow RHS completion')
        if path == CPOST:
            direct(body, includes[1].start(), 'post limiter include')
    print('  ok   '+kind+' callback: RHS binding and implicit include order in eight configurations')


def dycore(text):
    _, body, definition = signature(text, 'ERF::advance_dycore')
    direct(text, definition.start(), 'dycore definition')
    include_inventory(body, ('ERF_TI_utils.H', 'ERF_TI_no_substep_fun.H', 'ERF_TI_substep_fun.H',
                             'ERF_TI_slow_rhs_pre.H', 'ERF_TI_slow_rhs_post.H'))
    # Validate its directives as well as the registration slice. The configured
    # dycore contains only guarded OpenMP and the inventoried statement includes.
    configured(body, {})
    start = unique(r'#include\s+"ERF_TI_no_substep_fun.H"', body, 'callback definition sequence').start()
    direct(body, start, 'callback definition/registration sequence')
    source = Source(body[start:])
    literal(source, REGISTRATION)
    reject_uses(body, ['mri_integrator', 'mri_integrator_mem', 'slow_rhs_fun_pre', 'slow_rhs_fun_post',
                      'set_slow_rhs_pre', 'set_slow_rhs_post'], [(start, start+source.pos)], 'dycore callback uses')
    print('  ok   both callbacks defined and copied into the same-level integrator before synchronous advance')


def mri(text):
    reject_uses(text, ['MFIter', 'mfi'], [], 'MRI outer iterator inventory')
    cls = unique(pattern('template<class T> class MRISplitIntegrator {'), text, 'MRI class')
    declaration_scope(text, cls.start(), GUARDS[MRI])
    body = brace_block(text, cls.end()-1)
    require('#' not in no_literals(body), 'unsupported MRI class directive')
    advance = unique(pattern('double advance(T& S_old, T& S_new, double time, const double time_step) {'),
                     body, 'MRI advance')
    direct(body, advance.start(), 'MRI advance')
    stage_body = brace_block(body, advance.end()-1)
    allowed, positions = [], {}
    for kind, count in (('pre', 3), ('post', 4)):
        name = 'slow_rhs_'+kind
        typ = 'std::function<void('+', '.join(['T&']*count+['const double']*3+['const int'])+')>'
        member = unique(pattern(typ+' '+name+';'), body, 'callback value member')
        direct(body, member.start(), 'MRI callback member')
        setter = unique(pattern('void set_'+name+'('), body, 'callback setter')
        direct(body, setter.start(), 'MRI setter')
        source = Source(body[setter.start():])
        literal(source, 'void set_'+name+'('+typ+' F) { '+name+' = F; }')
        allowed += [member.span(), (setter.start(), setter.start()+source.pos)]
        statement = name+'(*F_slow, S_old, S_new, '+('*S_sum, ' if kind == 'post' else '')+'time, old_time_stage, time_stage, nrk);'
        matches = list(re.finditer(pattern(statement), stage_body))
        require(len(matches) == 2, name+': expected two stage invocations')
        positions[kind] = []
        for match, expected in zip(matches, (('if(!anelastic)', 'for(intnrk=0;nrk<3;nrk++)'),
                                             ('else', 'for(intnrk=0;nrk<2;nrk++)'))):
            standalone(stage_body, match.start())
            host_scope(stage_body, match.start())
            stack = scopes(stage_body, match.start())
            require(tuple(scope_header(stage_body, pos) for _, pos in stack) == expected,
                    'callback must be a synchronous statement in its MRI stage loop')
            positions[kind].append((stack, match.start()))
            allowed.append((advance.end()+match.start(), advance.end()+match.end()))
    for (pre_scope, pre_pos), (post_scope, post_pos) in zip(positions['pre'], positions['post']):
        require(pre_scope == post_scope and pre_pos < post_pos, 'MRI pre/post stage order changed')
    reject_uses(body, ['slow_rhs_pre', 'slow_rhs_post', 'set_slow_rhs_pre', 'set_slow_rhs_post', 'this'],
                allowed, 'MRI callback aliases/invocations')
    print('  ok   two value-stored callbacks and four synchronous, ordered MRI stage calls')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        dycore(texts[DYCORE])
        mri(texts[MRI])
        for path in CALLBACKS:
            callback(texts, path)
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 callback completion source contracts passed; entry state and other callees remain conditional.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
