#!/usr/bin/env python3
"""Guard selected spacing-reader completion through the acoustic callback.

Four finite contracts cover dycore registration, MRI value storage and direct
substep invocation, the three-way acoustic dispatch with normal-grid spacing
forwarding, and the final-substep implicit include with its stage interval.
Both GPU and implicit-W configurations are checked. This connects the prior
normal-grid substep and implicit-reader completion contracts to callback return.

Assume the prior ownership, spacing, kernel, TileNoZ, MFIter and slow-callback
contracts and inspected pinned AMReX semantics; GPU launch mode, enabled sync,
stable streams, ordinary destruction, no graph capture/concurrent host launches,
valid objects/indices and intended C++/AMReX/std::function/type resolution.
Other code/callees must preserve stream/sync, spacing and captured/step inputs
through aliases/globals and cannot defer host launches. Outputs are disjoint;
external headers cannot rewrite macros or execute runtime statements. No
undefined behavior, nonlocal transfer or invocation of the stored callback
outside the checked dycore advance is modeled.

This is not an arithmetic, GPU execution or whole-program lifetime theorem.
Initial state, no-substep/projection/EB paths, host-spacing consumers, other
readers, setup arrays and regrid/restart/destruction remain obligations. Moving
and variable-terrain numerical bodies are not proved. Standard-library Python.
"""
import itertools
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (DYCORE, MRI, binding, brace_block, call,
    direct, no_literals, reject_uses, scope_header, scopes, signature)
from check_stretched_dz_callback_completion import (REGISTRATION, dycore,
                                                  include_inventory)
from check_stretched_dz_consumers import (FAST, FAST_H, declaration_scope,
                                        host_scope, pattern, standalone)
from check_stretched_dz_mfiter import CONTROLS
from check_stretched_dz_solver_mfiter import IPRE
from momentum_geometry import literal, unique

CALLBACK = 'Source/TimeIntegration/ERF_TI_substep_fun.H'
FILES = (CALLBACK, MRI, DYCORE, FAST, FAST_H, IPRE)
GUARDS = {MRI: 'ERF_MRI_H', FAST_H: 'ERF_FAST_INTEGRATION_H_'}
HEADER = '''auto acoustic_substepping_fun = [&,one_d=one,zero_d=zero](
    int fast_step, int n_sub, int nrk, Vector<MultiFab>& S_slow_rhs,
    const Vector<MultiFab>& S_old, Vector<MultiFab>& S_stage, Vector<MultiFab>& S_data,
    const double dtau, const double, const Real inv_fac,
    const double old_substep_time, const double new_substep_time) {'''
TYPE = '''std::function<void(int, int, int, T&, const T&, T&, T&,
    const double, const double, const amrex::Real, const double, const double)>'''
INVOKE = '''acoustic_substepping(ks, nsubsteps, nrk, *F_slow, S_old, S_new,
    *S_sum, dtau, timestep, inv_fac, time + ks*dtau, time + (ks+1)*dtau);'''
GATE = '''if ((solverChoice.vert_implicit_fac[level][nrk] > zero_d) &&
    !solverChoice.implicit_before_substep && (fast_step == n_sub-1)) {'''
SETUP = '''const Real stage_dt = static_cast<Real>(n_sub * dtau);
    MultiFab scratch(S_data[IntVars::cons], make_alias, 0, 2);
    MultiFab scratch_xmom(S_data[IntVars::xmom], make_alias, 0, 1);
    MultiFab scratch_ymom(S_data[IntVars::ymom], make_alias, 0, 1);'''
W_SETUP = 'MultiFab scratch_zmom(S_data[IntVars::zmom], make_alias, 0, 1);'
BRANCHES = (('erf_substep_MT', 'if (solverChoice.terrain_type == TerrainType::MovingFittedMesh)'),
            ('erf_substep_T', 'if (solverChoice.mesh_type == MeshType::VariableDz)'),
            ('erf_substep_NS', 'else'))


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
        texts[path] = text
    return texts


def configured(body, gpu, implicit_w):
    output, stack = [], []
    options = {'AMREX_USE_GPU': gpu, 'ERF_IMPLICIT_W': implicit_w}
    for line in body.splitlines(keepends=True):
        if '#' not in no_literals(line):
            if all(value for value, _ in stack):
                output.append(line)
            continue
        m = re.fullmatch(r'\s*#\s*(\w+)\s*([^\n]*?)\s*', line)
        require(m, 'unsupported acoustic directive syntax')
        kind, arg = m[1], m[2].strip()
        if kind == 'ifdef' and arg in options:
            stack.append((options[arg], False))
        elif kind == 'else':
            require(stack and not arg and not stack[-1][1], 'invalid acoustic else')
            stack[-1] = (not stack[-1][0], True)
        elif kind == 'endif':
            require(stack and not arg, 'invalid acoustic endif')
            stack.pop()
        else:
            require(kind == 'include' and arg == '"ERF_ImplicitPre.H"', 'unsupported acoustic directive')
            if all(value for value, _ in stack):
                output.append(line)
    require(not stack, 'unclosed acoustic conditional')
    return ''.join(output)


def callback_body(text):
    include_inventory(text, ('ERF_SrcHeaders.H', 'ERF_ImplicitPre.H'))
    source = Source(text)
    literal(source, '#include "ERF_SrcHeaders.H" '+HEADER)
    body = brace_block(text, source.pos-1)
    tail = Source(text[source.pos+len(body):])
    literal(tail, '};')
    tail.done()
    reject_uses(body, ['acoustic_substepping_fun', 'this'], [], 'acoustic callback aliases')
    return body


def registration(text):
    dycore(text)
    _, body, _ = signature(text, 'ERF::advance_dycore')
    sequence = unique(pattern(REGISTRATION), body, 'acoustic registration sequence')
    reject_uses(body, ['acoustic_substepping_fun', 'set_acoustic_substepping'],
                [sequence.span()], 'dycore acoustic callback uses')


def mri(text):
    reject_uses(text, ['MFIter', 'mfi'], [], 'MRI outer iterator inventory')
    cls = unique(pattern('template<class T> class MRISplitIntegrator {'), text, 'MRI class')
    declaration_scope(text, cls.start(), GUARDS[MRI])
    body = brace_block(text, cls.end()-1)
    require('#' not in no_literals(body), 'unsupported MRI class directive')
    member = unique(pattern(TYPE+' acoustic_substepping;'), body, 'acoustic value member')
    direct(body, member.start(), 'acoustic member')
    setter = unique(pattern('void set_acoustic_substepping('), body, 'acoustic setter')
    direct(body, setter.start(), 'acoustic setter')
    source = Source(body[setter.start():])
    literal(source, 'void set_acoustic_substepping('+TYPE+' F) { acoustic_substepping = F; }')
    advance = unique(pattern('double advance(T& S_old, T& S_new, double time, const double time_step) {'),
                     body, 'MRI advance')
    direct(body, advance.start(), 'MRI advance')
    stage = brace_block(body, advance.end()-1)
    invoke = unique(pattern(INVOKE), stage, 'acoustic invocation')
    standalone(stage, invoke.start())
    host_scope(stage, invoke.start())
    expected = ('if(!anelastic)', 'for(intnrk=0;nrk<3;nrk++)',
                'if(!no_substepping)', 'for(intks=0;ks<nsubsteps;++ks)')
    stack = scopes(stage, invoke.start())
    require(tuple(scope_header(stage, pos) for _, pos in stack) == expected, 'acoustic invocation scope changed')
    # The existing slow-callback contract guards the argument lists and scopes
    # at both sites. Here establish the enclosing pre/acoustic/post order.
    pre = list(re.finditer(pattern('slow_rhs_pre('), stage))
    post = list(re.finditer(pattern('slow_rhs_post('), stage))
    require(len(pre) == len(post) == 2 and pre[0].start() < invoke.start() < post[0].start(),
            'acoustic invocation must be between the compressible slow callbacks')
    reject_uses(body, ['acoustic_substepping', 'set_acoustic_substepping', 'this'],
        [member.span(), (setter.start(), setter.start()+source.pos),
         (advance.end()+invoke.start(), advance.end()+invoke.end())], 'MRI acoustic callback uses')
    print('  ok   acoustic callback stored by value and invoked directly inside the substep loop')


def dispatch(body, parameters):
    allowed, ranges, spacing = [], [], None
    for name, condition in BRANCHES:
        occurrence = unique(pattern(name+'('), body, name+' call')
        arguments, values = call(body, name)
        standalone(body, occurrence.start())
        host_scope(body, occurrence.start())
        stack = scopes(body, occurrence.start())
        require(len(stack) == 1, 'substep call must be directly inside its dispatch branch')
        opening = stack[0][1]
        branch = unique(pattern(condition)+r'\s*$', body[:opening], 'acoustic dispatch condition')
        closing = opening+1+len(brace_block(body, opening))
        ranges.append((branch.start(), closing))
        require(re.match(r'\s*\)\s*;', body[arguments[-1][1]:]), 'substep call must end as a statement')
        allowed.append(occurrence.span())
        if name == 'erf_substep_NS':
            binding(values, parameters, {'step': ('int', 'fast_step'), 'nrk': ('int', 'nrk'),
                'level': ('int', 'level'), 'geom': ('constGeometry', 'fine_geom'),
                'stretched_dz_d': ('Gpu::DeviceVector<Real>&', 'stretched_dz_d[level]'),
                'dtau_d': ('constdouble', 'dtau'), 'facinv': ('constReal', 'inv_fac')}, 'normal substep forwarding')
            spacing = arguments[[key for key, _ in parameters].index('stretched_dz_d')]
    direct(body, ranges[0][0], 'acoustic dispatch')
    require(re.fullmatch(r'\s*else\s*', body[ranges[0][1]+1:ranges[1][0]]) and
            not body[ranges[1][1]+1:ranges[2][0]].strip(), 'acoustic branches must form one if/else-if/else chain')
    reject_uses(body, [name for name, _ in BRANCHES], allowed, 'acoustic dispatch callees')
    reject_uses(body, ['stretched_dz_d'], [spacing], 'acoustic device spacing uses')
    return ranges[-1][1]


def implicit(body, implicit_w, after):
    include, = include_inventory(body, ('ERF_ImplicitPre.H',))
    gate = unique(pattern(GATE), body, 'final-substep implicit gate')
    direct(body, gate.start(), 'implicit block')
    require(after < gate.start(), 'implicit diffusion must follow acoustic dispatch')
    source = Source(body[gate.start():])
    literal(source, GATE+SETUP+(W_SETUP if implicit_w else '')+'\n#include "ERF_ImplicitPre.H"\n}')
    require(gate.end() <= include.start() < gate.start()+source.pos, 'implicit include outside its gate')
    reject_uses(body, ['stage_dt'], [(gate.start(), gate.start()+source.pos)], 'implicit interval uses')


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        registration(texts[DYCORE])
        mri(texts[MRI])
        parameters, _, _ = signature(texts[FAST], 'erf_substep_NS')
        declared, _, declaration = signature(texts[FAST_H], 'erf_substep_NS', ';')
        declaration_scope(texts[FAST_H], declaration.start(), GUARDS[FAST_H])
        require([typ for _, typ in parameters] == [typ for _, typ in declared], 'normal substep declaration/definition mismatch')
        raw_body = callback_body(texts[CALLBACK])
        for gpu, implicit_w in itertools.product((False, True), repeat=2):
            body = configured(raw_body, gpu, implicit_w)
            after = dispatch(body, parameters)
            implicit(body, implicit_w, after)
        print('  ok   acoustic dispatch and normal-grid spacing forwarding in four configurations')
        print('  ok   final-substep implicit include and stage interval follow the dispatch')
    except (OSError, ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 acoustic completion source contracts passed; entry state and other callees remain conditional.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
