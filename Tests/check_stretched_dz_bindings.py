#!/usr/bin/env python3
"""Guard spacing ownership and the synchronous slow-RHS callback bindings.

This finite source contract connects ERF's member vectors, constructor setup,
advance_dycore inputs, callback registration, MRI invocation, and the slow-RHS
spacing parameter. check_stretched_dz and check_momentum_domain cover the
constructor arithmetic and downstream momentum bindings separately.

The checked callback runs during advance_dycore, whose level/Geometry locals
are alive. Its std::function remains stored afterward: only the checked path,
which replaces it before each advance, is covered. This is NOT a GPU lifetime
or whole-program alias proof. Preconditions: valid level, initialized member
vector of that level, live ERF object, normal C++/AMReX/std::function semantics,
and no macro rewriting, concurrency or callback invocation from outside this
path. Other calls must preserve the level/Geometry and spacing storage, even
through implicit this or aliases. Their numerical bodies are not interpreted.
Declaration headers, including ERF_SrcHeaders.H, must have no runtime effects
in the enclosing scope; only the enumerated statement fragments are audited.
Checkpoint/WRF refresh timing and GPU completion before resize/destruction are
not established. The known generated-grid positivity gap remains unchanged.

Explicit uses of protected storage are audited in the dycore body and its
statement includes, including both preprocessor branches. They are allowed
only as bare arguments to the listed consumers. Unknown uses fail closed.
This checker needs only the Python standard library.
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require, split_parameters
from check_upwindall_z_reach import brace_block as raw_brace_block
from momentum_geometry import call_arguments, literal, mask, unique

CLASS = 'Source/ERF.H'
CONSTRUCTOR = 'Source/ERF_Constructors.cpp'
ADVANCE = 'Source/TimeIntegration/ERF_Advance.cpp'
DYCORE = 'Source/TimeIntegration/ERF_AdvanceDycore.cpp'
CALLBACK = 'Source/TimeIntegration/ERF_TI_slow_rhs_pre.H'
RHS = 'Source/TimeIntegration/ERF_SlowRhsPre.cpp'
DECLARATIONS = 'Source/TimeIntegration/ERF_TI_slow_headers.H'
MRI = 'Source/TimeIntegration/ERF_MRI.H'
FRAGMENTS = tuple('Source/TimeIntegration/'+name for name in (
    'ERF_TI_utils.H', 'ERF_TI_no_substep_fun.H', 'ERF_TI_substep_fun.H',
    'ERF_TI_slow_rhs_pre.H', 'ERF_TI_slow_rhs_post.H', 'ERF_ImplicitPre.H',
    'ERF_ImplicitPost.H', 'ERF_ScalarLimitPost.H'))
FILES = tuple(dict.fromkeys((CLASS, CONSTRUCTOR, ADVANCE, DYCORE, RHS, DECLARATIONS, MRI, *FRAGMENTS)))
STORAGE = ('stretched_dz_h', 'stretched_dz_d')
# These routines must not mutate or retain an escaping mutable alias to spacing.
CONSUMERS = ('ComputeStrain_S', 'make_mom_sources', 'erf_substep_NS',
             'erf_slow_rhs_pre', 'erf_slow_rhs_post', 'ImplicitDiffForStateLU_S',
             'ImplicitDiffForMomLU_S')
CAPTURE_READERS = (
    'ComputeTurbulentViscosity', 'EBFactory', 'FillIntermediatePatch',
    'ImplicitDiffForMomLU_N', 'ImplicitDiffForMomLU_S', 'ImplicitDiffForMomLU_T',
    'ImplicitDiffForStateLU_N', 'ImplicitDiffForStateLU_S', 'ImplicitDiffForStateLU_T',
    'add_heat_flux_to_source', 'add_momentum_sources', 'add_theta_source',
    'advance_microphysics', 'cons_ave', 'erf_slow_rhs_post', 'erf_slow_rhs_pre',
    'erf_substep_MT', 'erf_substep_NS', 'erf_substep_T', 'getAdvFluxReg', 'get_eb',
    'get_z0', 'init_terrain_surface', 'lsf_at_level', 'make_J', 'make_areas',
    'make_buoyancy', 'make_fast_coeffs', 'make_gradp_pert', 'make_mom_sources',
    'make_sources', 'make_terrain_fitted_coords', 'make_zcc', 'project_momenta',
    'project_velocity_tb', 'realbdy_compute_interior_ghost_rhs', 'redistribute_term',
    'setDomainBndry', 'set_initial_state', 'sounding_at_level', 'u_ave',
    'update_geostrophic_profile', 'update_rhoqt_sources', 'update_rhotheta_sources',
    'update_terrain_stage', 'update_terrain_substep', 'update_w_subsidence', 'v_ave')
STRING = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''


def no_literals(text):
    return re.sub(STRING, lambda m: ' '*len(m[0]), text)


def brace_block(text, start):
    # Preserve offsets while finding the closing brace: logging strings can
    # contain braces, including a lone '}' before a later protected write.
    masked_body = raw_brace_block(no_literals(text), start)
    return text[start+1:start+1+len(masked_body)]


def reject_uses(text, names, spans, label):
    remainder = no_literals(mask(text, spans))
    found = re.search(r'\b(?:'+'|'.join(map(re.escape, names))+r')\b', remainder)
    require(found is None, label+': unmodeled use of '+(found[0] if found else ''))


def scopes(text, end):
    """Return open delimiters at an occurrence; strings preserve source offsets."""
    stack = []
    for pos, char in enumerate(no_literals(text[:end])):
        if char in '([{':
            stack.append((char, pos))
        elif char in ')]}':
            require(stack and stack[-1][0] == {')': '(', ']': '[', '}': '{'}[char], 'unbalanced source scope')
            stack.pop()
    return stack


def direct(text, pos, label):
    require(not scopes(text, pos), label+': binding must be at function/class scope')
    depth = 0
    for directive in re.finditer(r'^\s*#\s*(\w+)[^\n]*', text[:pos], re.M):
        depth += directive[1] in ('if', 'ifdef', 'ifndef')
        depth -= directive[1] == 'endif'
        require(depth >= 0, 'unbalanced preprocessor scope')
    require(depth == 0, label+': binding is conditional on an unsupported preprocessor scope')
    prefix = re.sub(r'^\s*#[^\n]*', '', text[:pos], flags=re.M).rstrip()
    require(not prefix or prefix[-1] in ';}:', label+': binding is not a standalone statement')


def params(text):
    result = []
    for param in split_parameters(text):
        match = re.fullmatch(r'(.*?)\b(\w+)\s*', param, re.S)
        if match:
            result.append((match[2], compact(match[1]).replace('amrex::', '')))
        else:  # unnamed parameters retain their position and type
            result.append((None, compact(param).replace('amrex::', '')))
    return result


def signature(text, name, ending='{'):
    match = unique(r'\bvoid\s+'+re.escape(name)+r'\s*\(([^{}]*?)\)\s*'+re.escape(ending), text, name+' signature')
    return params(match[1]), brace_block(text, match.end()-1) if ending == '{' else '', match


def call(text, name):
    matches = list(call_arguments(text, r'(?<![\w:.>])'+re.escape(name)))
    require(len(matches) == 1, name+': expected exactly one call')
    return matches[0], [compact(text[start:end]) for start, end in matches[0]]


def binding(arguments, parameters, expected, label):
    require(len(arguments) == len(parameters), label+': call arity changed')
    for name, (wanted_type, wanted_arg) in expected.items():
        positions = [index for index, (key, _) in enumerate(parameters) if key == name]
        require(len(positions) == 1, label+': missing/duplicate parameter '+name)
        index = positions[0]
        require(parameters[index][1] == wanted_type, label+': parameter type '+name)
        require(arguments[index] == wanted_arg, label+': argument binding '+name)


def shoc(text, enabled):
    pattern = r'#ifdef\s+ERF_USE_EAMXX_SHOC\s*\n([^#]*?)#endif'
    return re.sub(pattern, lambda m: m[1] if enabled else '', text)


def check_owner(texts):
    text = texts[CLASS]
    match = unique(r'\bclass\s+ERF\s*:\s*public\s+amrex::AmrCore\s*\{', text, 'ERF class')
    body = brace_block(text, match.end()-1)
    for name, vector in [('zlevels_stag', 'amrex::Vector<amrex::Real>'),
                         ('stretched_dz_h', 'amrex::Vector<amrex::Real>'),
                         ('stretched_dz_d', 'amrex::Gpu::DeviceVector<amrex::Real>')]:
        declaration = 'amrex::Vector<'+vector+'> '+name+';'
        pattern = r'\s*'.join(map(re.escape, re.findall(r'\w+|::|[^\s]', declaration)))
        member = unique(pattern, body, name+' owning member')
        direct(body, member.start(), name)
        reject_uses(body, [name], [member.span()], 'class storage')
    print('  ok   nodal, host and device vectors are owning ERF members')

    _, body, _ = signature(texts[CONSTRUCTOR], 'ERF::ERF_shared')
    count = unique(r'\bint\s+nlevs_max\s*=\s*max_level\s*\+\s*1\s*;', body, 'level count')
    direct(body, count.start(), 'level count')
    start = unique(r'\bstretched_dz_h\s*\.\s*resize\s*\(', body, 'host allocation').start()
    direct(body, start, 'spacing setup')
    source = Source(body[start:])
    literal(source, 'stretched_dz_h.resize(nlevs_max); stretched_dz_d.resize(nlevs_max); '
                    'zlevels_stag.resize(max_level+1); '
                    'init_zlevels(zlevels_stag, stretched_dz_h, stretched_dz_d, geom, refRatio(), '
                    'solverChoice.grid_stretching_ratio, solverChoice.zsurf, solverChoice.dz0);')
    require(count.end() < start, 'spacing allocation precedes level count')
    reject_uses(body, STORAGE, [(start, start+source.pos)], 'constructor storage')
    # Check every use before setup of the non-const count. Bare call arguments
    # below are assumed not to mutate it; reference aliases/writes fail.
    prefix = body[:start]
    reject_uses(prefix, ['zlevels_stag'], [], 'constructor nodal ownership')
    spans = [count.span()]
    for arguments in call_arguments(no_literals(prefix),
            r'(?:\.\s*(?:resize|ReSize)|\b(?:initializeMicrophysics|initializeWindFarm|'
            r'validate_sbm_m3_advection|validate_auxiliary_inert_tracer_fixture))'):
        spans.extend((lo, hi) for lo, hi in arguments if compact(prefix[lo:hi]) in ('nlevs_max', 'max_level'))
    spans += [match.span() for pattern in (
        r'\blev\s*<=\s*max_level\b',
        r'(?<=[(,])\s*max_level\s*\+\s*1\s*(?=[),])') for match in re.finditer(pattern, prefix)]
    reject_uses(prefix, ['nlevs_max', 'max_level'], spans, 'constructor level count')
    print('  ok   constructor sizes member vectors and initializes those same objects')


def check_registration(texts):
    parameters, body, _ = signature(texts[DYCORE], 'ERF::advance_dycore')
    _, outer_args = call(texts[ADVANCE], 'advance_dycore')
    binding(outer_args, parameters, {'level': ('int', 'lev'), 'fine_geom': ('constGeometry', 'Geom(lev)')}, 'advance to dycore')
    marker = unique(r'#include\s+"ERF_TI_slow_rhs_pre.H"', body, 'pre callback include')
    direct(body, marker.start(), 'pre callback include')
    start = unique(r'\bMRISplitIntegrator\s*<\s*Vector\s*<\s*MultiFab\s*>\s*>\s*&\s*mri_integrator\s*=', body, 'integrator reference').start()
    direct(body, start, 'integrator reference')
    require(marker.end() < start, 'callback definition must precede registration')
    source = Source(body[start:])
    literal(source, 'MRISplitIntegrator<Vector<MultiFab> >& mri_integrator = *mri_integrator_mem[level]; '
                    'mri_integrator.set_slow_rhs_pre(slow_rhs_fun_pre); '
                    'mri_integrator.set_slow_rhs_post(slow_rhs_fun_post); '
                    'mri_integrator.set_acoustic_substepping(acoustic_substepping_fun); '
                    'mri_integrator.set_slow_fast_timestep_ratio(fixed_mri_dt_ratio > 0 ? fixed_mri_dt_ratio : dt_mri_ratio[level]); '
                    'mri_integrator.set_no_substep(no_substep_fun); '
                    'mri_integrator.advance(state_old, state_new, old_time, dt_advance);')
    reject_uses(body, ['mri_integrator', 'mri_integrator_mem', 'slow_rhs_fun_pre'],
                [(start, start+source.pos)], 'dycore callback registration')
    print('  ok   same-level integrator receives the new callback before synchronous advance')
    return body


def check_callback(texts):
    text = texts[CALLBACK]
    source = Source(text)
    literal(source, '#include "ERF_SrcHeaders.H" '
                    'auto slow_rhs_fun_pre = [&,zero_d=zero,one_d=one](Vector<MultiFab>& S_rhs, '
                    'Vector<MultiFab>& S_old, Vector<MultiFab>& S_data, '
                    'const double old_step_time, const double old_stage_time, '
                    'const double new_stage_time, const int nrk) {')
    body = brace_block(text, source.pos-1)
    tail = Source(text[source.pos+len(body):])
    literal(tail, '};')
    tail.done()
    reject_uses(body, ['slow_rhs_fun_pre', 'this'], [], 'callback capture')
    for enabled in (False, True):
        configured = shoc(body, enabled)
        spans, arguments = call(configured, 'erf_slow_rhs_pre')
        # The call is a standalone statement in the callback's own scope.
        occurrence = configured.rfind('erf_slow_rhs_pre', 0, spans[0][0])
        require(not re.search(r'#\s*(?:pragma|include)\b', configured[:occurrence]),
                'unsupported directive before slow RHS invocation')
        direct(configured, occurrence, 'slow RHS invocation')
        parameters, _, _ = signature(shoc(texts[RHS], enabled), 'erf_slow_rhs_pre')
        declaration, _, _ = signature(shoc(texts[DECLARATIONS], enabled), 'erf_slow_rhs_pre', ';')
        require([typ for _, typ in declaration] == [typ for _, typ in parameters], 'slow RHS declaration/definition type mismatch')
        binding(arguments, parameters, {'level': ('int', 'level'), 'nrk': ('int', 'nrk'),
                'geom': ('constGeometry', 'fine_geom'),
                'stretched_dz_d': ('Gpu::DeviceVector<Real>&', 'stretched_dz_d[level]')}, 'callback to slow RHS')
    print('  ok   reference-captured callback forwards level, stage, Geometry and member spacing in both SHOC configurations')


def scope_header(text, pos):
    prefix = no_literals(text[:pos]).rstrip()
    if prefix.endswith(')'):
        depth = 1
        index = len(prefix)-2
        while index >= 0 and depth:
            depth += (prefix[index] == ')') - (prefix[index] == '(')
            index -= 1
        match = re.search(r'\b(if|for)\s*$', prefix[:index+1])
        require(match is not None, 'MRI callback enclosed by unsupported scope')
        return compact(prefix[match.start():])
    require(re.search(r'\belse$', prefix), 'MRI callback enclosed by unsupported scope')
    return 'else'


def check_mri(texts):
    header = texts[MRI]
    cls = unique(r'\bclass\s+MRISplitIntegrator\s*\{', header, 'MRI class')
    text = brace_block(header, cls.end()-1)
    spans = []
    expected_type = 'std::function<void(T&, T&, T&, const double, const double, const double, const int)>'
    pattern = r'\s*'.join(map(re.escape, re.findall(r'\w+|::|[^\s]', expected_type+' slow_rhs_pre;')))
    member = unique(pattern, text, 'stored callback type')
    direct(text, member.start(), 'MRI callback member')
    spans.append(member.span())
    setter = unique(r'\bvoid\s+set_slow_rhs_pre\s*\(', text, 'callback setter')
    direct(text, setter.start(), 'MRI setter')
    source = Source(text[setter.start():])
    literal(source, 'void set_slow_rhs_pre('+expected_type+' F) { slow_rhs_pre = F; }')
    spans.append((setter.start(), setter.start()+source.pos))
    advance = unique(r'\bdouble\s+advance\s*\(\s*T&\s*S_old\s*,\s*T&\s*S_new\s*,\s*double\s*time\s*,\s*const\s+double\s*time_step\s*\)\s*\{', text, 'MRI advance')
    direct(text, advance.start(), 'MRI advance')
    body = brace_block(text, advance.end()-1)
    require('#' not in body, 'unsupported preprocessor in MRI advance')
    calls = list(re.finditer(r'\bslow_rhs_pre\s*\([^;{}]*\)\s*;', body))
    require(len(calls) == 2, 'MRI advance must contain two supported callback sites')
    for call_match, headers in zip(calls, [('if(!anelastic)', 'for(intnrk=0;nrk<3;nrk++)'), ('else', 'for(intnrk=0;nrk<2;nrk++)')]):
        source = Source(call_match[0])
        literal(source, 'slow_rhs_pre(*F_slow, S_old, S_new, time, old_time_stage, time_stage, nrk);')
        source.done()
        stack = scopes(body, call_match.start())
        require(all(char == '{' for char, _ in stack) and len(stack) == 2, 'MRI callback must be a direct synchronous stage statement')
        require(tuple(scope_header(body, pos) for _, pos in stack) == headers, 'MRI callback stage scope changed')
        require(body[:call_match.start()].rstrip()[-1] in ';}', 'MRI callback is not a standalone stage statement')
        spans.append((advance.end()+call_match.start(), advance.end()+call_match.end()))
    reject_uses(text, ['slow_rhs_pre', 'set_slow_rhs_pre', 'this'], spans, 'MRI callback use')
    print('  ok   MRI stores the callback by value and invokes it directly in both stage loops')


def check_uses(texts, dycore_body):
    allowed_includes = {Path(path).name for path in FRAGMENTS} | {'ERF_SrcHeaders.H'}
    all_fragments = {DYCORE: dycore_body, **{name: texts[name] for name in FRAGMENTS}}
    for name, text in all_fragments.items():
        for directive in re.finditer(r'^\s*#\s*(\w+)([^\n]*)', text, re.M):
            kind, argument = directive.groups()
            require(kind in ('include', 'if', 'ifdef', 'ifndef', 'else', 'elif', 'endif', 'pragma'), 'unsupported directive in spacing fragment')
            if kind == 'include':
                require(argument.strip().strip('"') in allowed_includes, 'unmodeled statement include')
        text = no_literals(text)
        # Allow only bare member-vector arguments to these named consumers.
        spans = []
        for consumer in CONSUMERS:
            for arguments in call_arguments(text, r'(?<![\w:.>])'+consumer+r'(?:\s*<\s*[012]\s*>)?'):
                spans.extend((start, end) for start, end in arguments
                             if compact(text[start:end]) in {value+'[level]' for value in STORAGE})
        reject_uses(text, STORAGE, spans, Path(name).name+' storage')
        # Explicit local writes, shadowing and reference aliases cannot hide
        # behind ordinary index reads or bare call arguments. All other callees
        # receiving these inputs are an explicit nonmutation assumption.
        spans = []
        for arguments in call_arguments(text, r'\b(?:'+'|'.join(CAPTURE_READERS)+r')(?:\s*<\s*[012]\s*>)?'):
            spans.extend((start, end) for start, end in arguments
                         if compact(text[start:end]) in ('level', 'level+1', 'fine_geom'))
        patterns = (r'\[\s*level\s*\]', r'\blevel\s*(?:==|<|>)\s*(?:0\b|finest_level\b|static_cast\s*<\s*int\s*>\s*\(\s*m_ibseb\.size\(\)\s*\))',
                    r'<<\s*level\s*<<',
                    r'\bfine_geom\s*\.\s*(?:Domain|InvCellSizeArray|periodicity)\s*\(\s*\)',
                    r'\bfine_geom\s*\.\s*(?:InvCellSize|CellSize|ProbLo)\s*\(\s*[012]\s*\)')
        spans += [match.span() for pattern in patterns for match in re.finditer(pattern, text)]
        reject_uses(text, ['level', 'fine_geom', 'this'], spans, Path(name).name+' captured input')
    print('  ok   dycore and all eight statement fragments have only supported spacing/input uses')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    texts = {path: strip_comments((root/path).read_text()) for path in FILES}
    check_owner(texts)
    body = check_registration(texts)
    check_callback(texts)
    check_mri(texts)
    check_uses(texts, body)
    print('6 spacing ownership/callback source contracts passed; GPU lifetime remains conditional.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        sys.exit(1)
