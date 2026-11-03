#!/usr/bin/env python3
"""Guard slow-RHS spacing forwarding and the PBL pointer-to-value path.

This standard-library source contract audits explicit spacing uses throughout
erf_slow_rhs_pre, erf_slow_rhs_post and AdvectionSrcForMom. Seven call sites
must forward the same vector to matching reference parameters. The stretched
PBL functor must hold a const pointer, be copied into an inline ParallelFor
kernel, and return a plain three-scalar aggregate. Unsupported uses fail closed.

This is not a numerical theorem, whole-program alias analysis or GPU-lifetime
proof. Valid initialized storage and indices, ordinary C++/AMReX semantics,
disjoint outputs, no concurrent mutation/undefined behavior/external macro
rewriting, and preservation through preexisting aliases are assumptions.
Named callees must preserve spacing: only their signatures are checked here.
check_stretched_dz_consumers separately checks tau's explicit spacing uses;
the diffusion and normal-advection bodies remain external obligations. The
functor copy retains a device pointer, so GPU completion before storage changes
is still required. Domain correctness, positivity, scalar formulas, scheduling
and the rest of the RHS computations are not proved.

usage: check_stretched_dz_rhs.py [ERF_ROOT]
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require
from check_stretched_dz_bindings import (binding, brace_block, call, no_literals,
                                       reject_uses, scopes, shoc, signature)
from check_stretched_dz_consumers import (SLOW_H, STRAIN_H, TAU,
    declaration_scope, host_scope, pattern, preprocess, standalone)
from momentum_geometry import call_arguments, unique

PRE = 'Source/TimeIntegration/ERF_SlowRhsPre.cpp'
POST = 'Source/TimeIntegration/ERF_SlowRhsPost.cpp'
MOM = 'Source/Advection/ERF_AdvectionSrcForMom.cpp'
ADV_H = 'Source/Advection/ERF_Advection.H'
STATE = 'Source/Diffusion/ERF_DiffusionSrcForState_S.cpp'
DIFF_MOM = 'Source/Diffusion/ERF_DiffusionSrcForMom.cpp'
CONSTANT = 'Source/Advection/ERF_AdvectionSrcForMom_ConstantDz.cpp'
STRETCHED = 'Source/Advection/ERF_AdvectionSrcForMom_StretchedDz.cpp'
PBL = 'Source/PBL/ERF_PBLModels.H'
TKE = 'Source/Diffusion/ERF_TurbKESources.H'
FILES = (PRE, POST, MOM, SLOW_H, ADV_H, STRAIN_H, TAU, STATE, DIFF_MOM,
         CONSTANT, STRETCHED, PBL, TKE)
GUARDS = {SLOW_H: 'ERF_SLOW_INTEGRATION_H_', STRAIN_H: 'ERF_DIFFUSION_H_',
          ADV_H: 'ERF_ADVECTION_H_', PBL: 'ERF_PBLMODELS_H_', TKE: 'ERF_TURB_KE_SOURCES_H_'}
VECTOR = 'stretched_dz_d'
MUTABLE = 'Gpu::DeviceVector<Real>&'
CALLEES = {
    'erf_make_tau_terms': (TAU, SLOW_H, MUTABLE),
    'DiffusionSrcForState_S': (STATE, STRAIN_H, 'const'+MUTABLE),
    'DiffusionSrcForMom': (DIFF_MOM, STRAIN_H, 'const'+MUTABLE),
    'AdvectionSrcForMom': (MOM, ADV_H, MUTABLE),
    'AdvectionSrcForMom_ConstantDz': (CONSTANT, ADV_H, 'const'+MUTABLE),
    'AdvectionSrcForMom_StretchedDz': (STRETCHED, ADV_H, 'const'+MUTABLE),
}
FORWARD = {
    PRE: ('erf_make_tau_terms', 'DiffusionSrcForState_S', 'AdvectionSrcForMom', 'DiffusionSrcForMom'),
    POST: ('DiffusionSrcForState_S',),
    MOM: ('AdvectionSrcForMom_ConstantDz', 'AdvectionSrcForMom_StretchedDz'),
}
FUNCTOR = '''const amrex::Real* dz; int klo; int khi;
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
    PBLDerivativeDzInv operator() (int, int, int k) const noexcept {
        const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];
        const amrex::Real dz_hi = dz[(k < khi) ? k+1 : k];
        return {two / (dz_lo + two*dz[k] + dz_hi), two / dz[k], two / dz[k]};
    }'''
INITIALIZERS = {
    'S': 'const PBLDerivativeDzInv_S pbl_dz_inv{stretched_dz_d.data(), domain.smallEnd(2), domain.bigEnd(2)};',
    'T': 'const PBLDerivativeDzInv_T pbl_dz_inv{z_cc};',
    'N': 'const PBLDerivativeDzInv_N pbl_dz_inv{dxInv[2]};',
}


def checked_signature(texts, path, header, name, spacing_type):
    """Check both supported SHOC configurations, even though spacing is shared."""
    enabled = None
    for use_shoc in (False, True):
        params, body, definition = signature(no_literals(shoc(texts[path], use_shoc)), name)
        declared, _, declaration = signature(no_literals(shoc(texts[header], use_shoc)), name, ';')
        require([typ for _, typ in params] == [typ for _, typ in declared], name+': declaration/definition type mismatch')
        require([(key, typ) for key, typ in params if key == VECTOR] == [(VECTOR, spacing_type)], name+': spacing parameter type')
        declaration_scope(shoc(texts[header], use_shoc), declaration.start(), GUARDS[header])
        if use_shoc:
            enabled = params, body, definition
    return enabled


def forwarding(body, name, params):
    arguments, values = call(body, name)
    wanted = dict(params)[VECTOR]
    binding(values, params, {VECTOR: (wanted, VECTOR)}, name+' forwarding')
    start = body.rfind(name, 0, arguments[0][0])
    standalone(body, start)
    host_scope(body, start)
    reject_uses(body, [name], [(start, arguments[0][0])], name+' callee uses')
    # Only the designated argument is exempt: an extra spacing argument must
    # not create an unchecked alias through another parameter.
    return arguments[[key for key, _ in params].index(VECTOR)]


def audited_body(texts, path, header, name):
    _, body, definition = checked_signature(texts, path, header, name, MUTABLE)
    text = shoc(texts[path], True)
    prefix = re.sub(r'(?m)^\s*#include\s*[<"][\w./]+[>"]\s*$', '', text[:definition.start()])
    prefix = re.sub(r'\busing\s+namespace\s+amrex\s*;', '', prefix)
    require(not prefix.strip(), name+': unsupported function prefix')
    require(not text[definition.end()+len(body)+1:].strip(), name+': unsupported function suffix')
    # Before evaluating SHOC directives, reject hidden spacing/functor uses in
    # either conditional region. Both actual signatures are checked above.
    _, raw_body, _ = signature(no_literals(texts[path]), name)
    for conditional in re.finditer(r'#ifdef\s+ERF_USE_EAMXX_SHOC\s*\n([^#]*?)#endif', raw_body):
        reject_uses(conditional[1], [VECTOR, 'pbl_dz_inv'], [], 'SHOC region')
    return preprocess(body)


def check_types(texts):
    for name, expected in (
            ('PBLDerivativeDzInv', 'amrex::Real interior; amrex::Real dirichlet_lo; amrex::Real dirichlet_hi;'),
            ('PBLDerivativeDzInv_S', FUNCTOR)):
        match = unique(r'\bstruct\s+'+name+r'\s*\{', no_literals(texts[PBL]), name+' struct')
        declaration_scope(texts[PBL], match.start(), GUARDS[PBL])
        body = brace_block(texts[PBL], match.end()-1)
        require(re.fullmatch(pattern(expected), body.strip()), 'unsupported source in '+name+' struct body')
    print('  ok   stretched PBL aggregate copies a const pointer and returns only scalar values')


def check_tke(texts):
    text = no_literals(texts[TKE])
    params, body, definition = signature(text, 'AddTurbKESources')
    declaration_scope(text, definition.start(), GUARDS[TKE])
    template = unique(pattern('template <typename PBLDerivativeDzInvFunctor>'), text[:definition.start()], 'TKE template')
    require(not text[template.end():definition.start()].strip(), 'unsupported TKE template prefix')
    require(dict(params).get('pbl_derivative_dz_inv') == 'constPBLDerivativeDzInvFunctor&', 'TKE functor parameter type')
    require([key for key, _ in params].count('pbl_derivative_dz_inv') == 1, 'duplicate TKE functor parameter')
    require('#' not in body, 'unsupported TKE body directive')
    launch = unique(pattern('amrex::ParallelFor(bx,[=,moist_indices=solverChoice.moisture_indices] '
                           'AMREX_GPU_DEVICE(int i,int j,int k) noexcept {'), body, 'TKE value-capture kernel')
    standalone(body, launch.start())
    host_scope(body, launch.start())
    kernel = brace_block(body, launch.end()-1)
    tail = body[launch.end()+len(kernel):]
    require(re.match(r'\}\s*\)\s*;', tail), 'TKE lambda must be a bare launch argument')
    read = unique(pattern('const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);'), body, 'TKE functor evaluation')
    require(launch.end() <= read.start() < launch.end()+len(kernel), 'TKE evaluation is outside its value-captured kernel')
    require(not scopes(kernel, read.start()-launch.end()), 'TKE evaluation has an unsupported nested scope')
    standalone(body, read.start())
    reject_uses(body, ['pbl_derivative_dz_inv'], [read.span()], 'TKE functor uses')
    # Reject local aliases/shadows of the launch function as well.
    launches = [m.span() for m in re.finditer(r'\bamrex::ParallelFor\s*\(', body)]
    reject_uses(body, ['ParallelFor'], launches, 'TKE launcher')
    print('  ok   TKE copies the referenced functor into its GPU kernel before evaluating it')
    return params


def check_post_functor(body, params):
    calls = list(call_arguments(body, r'(?<![\w:.>])AddTurbKESources'))
    require(len(calls) == 3, 'expected three mesh-specific TKE calls')
    reject_uses(body, ['AddTurbKESources'],
                [(body.rfind('AddTurbKESources', 0, args[0][0]), args[0][0]) for args in calls],
                'post-RHS TKE callee uses')
    spans = []
    for grid, statement in INITIALIZERS.items():
        init = unique(pattern(statement), body, grid+' PBL construction')
        standalone(body, init.start())
        host_scope(body, init.start())
        next_call = re.match(r'\s*AddTurbKESources\s*\(', body[init.end():])
        require(next_call is not None, 'PBL construction must directly precede its TKE call')
        first_arg = init.end()+next_call.end()
        candidates = [arguments for arguments in calls if arguments[0][0] == first_arg]
        require(len(candidates) == 1, 'PBL forwarding call not found')
        arguments = candidates[0]
        values = [compact(body[lo:hi]) for lo, hi in arguments]
        binding(values, params, {'pbl_derivative_dz_inv': ('constPBLDerivativeDzInvFunctor&', 'pbl_dz_inv')}, 'PBL to TKE')
        spans += [init.span(), arguments[[key for key, _ in params].index('pbl_derivative_dz_inv')]]
    reject_uses(body, ['pbl_dz_inv', 'PBLDerivativeDzInv_S'], spans, 'post-RHS PBL uses')
    return spans


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw), path+': unsupported C++ lexical form')
        text = strip_comments(raw)
        for macro in re.finditer(r'(?m)^\s*#\s*(define|undef)\b([^\n]*)', text):
            require(macro[1] == 'define' and macro[2].strip() == GUARDS.get(path), path+': unsupported macro')
        texts[path] = text
    callees = {name: checked_signature(texts, path, header, name, typ)[0]
               for name, (path, header, typ) in CALLEES.items()}
    check_types(texts)
    tke_params = check_tke(texts)
    for path, header, name in ((PRE, SLOW_H, 'erf_slow_rhs_pre'),
                               (POST, SLOW_H, 'erf_slow_rhs_post'),
                               (MOM, ADV_H, 'AdvectionSrcForMom')):
        body = audited_body(texts, path, header, name)
        spans = [forwarding(body, callee, callees[callee]) for callee in FORWARD[path]]
        if path == POST:
            spans += check_post_functor(body, tke_params)
        reject_uses(body, [VECTOR], spans, name+' spacing uses')
        print(f'  ok   {name}: every explicit spacing use has a checked forwarding or PBL binding')
    print('5 spacing RHS/PBL source contracts passed; external callees and GPU completion remain conditional.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        sys.exit(1)
