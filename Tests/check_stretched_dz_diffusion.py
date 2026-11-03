#!/usr/bin/env python3
"""Guard explicit spacing preservation through stretched-grid diffusion.

Seven finite source contracts follow the const vector in both diffusion entry
points, its local data pointer, the four scalar-flux dispatch arms, and the
value-captured gradient/divergence kernels. Complete leaf definitions ensure
boundary weights take spacing by value and the gradient returns a scalar.
The state adapter's runtime statement include is expanded before auditing.
Unsupported uses, aliases, captures and source forms fail closed.

This standard-library checker adds no arithmetic theorem. Valid initialized
storage/indices, ordinary C++/AMReX semantics, disjoint output arrays, no undefined
behavior/concurrent mutation/external macro rewriting, and preservation through
other inputs/globals remain assumptions. It is not a C++ parser or whole-program
alias proof. GPU kernels retain pointer copies: completion before storage changes
remains unproved. Numerical accuracy, positivity, bounds, refresh scheduling and
implicit consumers are separate obligations. check_stretched_dz_advection
separately audits the explicit normal-advection pointer paths.

usage: check_stretched_dz_diffusion.py [ERF_ROOT]
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require
from check_stretched_dz_bindings import (binding, brace_block, no_literals,
                                       params, reject_uses, scopes, signature)
from check_stretched_dz_consumers import (declaration_scope, host_scope,
    kernel_regions, pattern, read_context, standalone)
from momentum_geometry import call_arguments, unique

STATE = 'Source/Diffusion/ERF_DiffusionSrcForState_S.cpp'
MOM = 'Source/Diffusion/ERF_DiffusionSrcForMom.cpp'
HEADER = 'Source/Diffusion/ERF_Diffusion.H'
SCALAR = 'Source/Diffusion/ERF_ScalarDiffusion.H'
SETUP = 'Source/Diffusion/ERF_SetupScalarDiffusion.H'
FILES = (STATE, MOM, HEADER, SCALAR, SETUP)
GUARDS = {HEADER: 'ERF_DIFFUSION_H_', SCALAR: 'ERF_SCALAR_DIFFUSION_H_'}
BUILD = 'BuildScalarDiffusionFluxes_S'
IMPL = BUILD+'_Impl'
DIV = 'ApplyScalarDiffusionFluxDivergence_S'
GRAD = 'StretchedScalarGradient'
WEIGHTS = 'StretchedDiffusionDirichletWeights'
RESULT = 'StretchedDiffusionBoundaryWeights'
NAMES = (BUILD, IMPL, DIV, GRAD, WEIGHTS, RESULT)

WEIGHT_DEFINITION = '''AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE StretchedDiffusionBoundaryWeights
StretchedDiffusionDirichletWeights(const amrex::Real dz_near, const amrex::Real dz_next) noexcept {
    const amrex::Real f = dz_next / dz_near + amrex::Real(2.0);
    const amrex::Real f2 = f * f;
    const amrex::Real c3 = amrex::Real(2.0) / (f - f2);
    return {-(amrex::Real(1.0) - f2) * c3, -f2 * c3, c3};
}'''
GRAD_DEFINITION = '''AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
StretchedScalarGradient(const amrex::Array4<const amrex::Real>& scalar,
    const int scalar_comp, const int i, const int j, const int k,
    const amrex::Real* dz, const int klo, const int khi,
    const bool ext_dir_lo, const bool ext_dir_hi) noexcept {
    if (ext_dir_lo) {
        const auto w = StretchedDiffusionDirichletWeights(dz[k], dz[k + 1]);
        return (amrex::Real(1.0) / dz[k]) *
            (w.c1 * scalar(i, j, k - 1, scalar_comp) +
             w.c2 * scalar(i, j, k, scalar_comp) +
             w.c3 * scalar(i, j, k + 1, scalar_comp));
    }
    if (ext_dir_hi) {
        const auto w = StretchedDiffusionDirichletWeights(dz[k - 1], dz[k - 2]);
        return (amrex::Real(1.0) / dz[k - 1]) *
            (-(w.c1 * scalar(i, j, k, scalar_comp) +
               w.c2 * scalar(i, j, k - 1, scalar_comp) +
               w.c3 * scalar(i, j, k - 2, scalar_comp)));
    }
    const amrex::Real inv = (k == klo) ? amrex::Real(1.0) / dz[k]
        : (k == khi + 1) ? amrex::Real(1.0) / dz[k - 1]
        : amrex::Real(2.0) / (dz[k] + dz[k - 1]);
    return inv * (scalar(i, j, k, scalar_comp) - scalar(i, j, k - 1, scalar_comp));
}'''
MOM_READS = (
    '''Real diffContrib = ((tau11(i,j,k)-tau11(i-1,j,k))*dxinv*mfsq
        +(tau12(i,j+1,k)-tau12(i,j,k))*dyinv*mfsq
        +(tau13(i,j,k+1)-tau13(i,j,k))/dz_ptr[k]);''',
    '''Real diffContrib = ((tau21(i+1,j,k)-tau21(i,j,k))*dxinv*mfsq
        +(tau22(i,j,k)-tau22(i,j-1,k))*dyinv*mfsq
        +(tau23(i,j,k+1)-tau23(i,j,k))/dz_ptr[k]);''',
    'Real dzinv_loc = two / (dz_ptr[k]+dz_ptr[k-1]);',
)
GRAD_READ = '''const Real grad = StretchedScalarGradient(
    scalar, scalar_comp, i, j, k, dz, klo, khi, ext_lo, ext_hi);'''
DIV_READ = 'const amrex::Real dzk_inv = amrex::Real(1.0) / dz[k];'


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported C++ lexical form')
        text = strip_comments(raw)
        for macro in re.finditer(r'(?m)^\s*#\s*(define|undef)\b([^\n]*)', text):
            require(macro[1] == 'define' and macro[2].strip() == GUARDS.get(path),
                    path+': unsupported macro')
        texts[path] = text
    return texts


def pointer_parameter(parameters, name, typ):
    require([(key, value) for key, value in parameters if key == name] == [(name, typ)],
            name+': missing, duplicate or changed spacing parameter type')


def entry_body(texts, path, name):
    text = texts[path]
    parameters, body, definition = signature(no_literals(text), name)
    declared, _, declaration = signature(no_literals(texts[HEADER]), name, ';')
    declaration_scope(texts[HEADER], declaration.start(), GUARDS[HEADER])
    require([typ for _, typ in parameters] == [typ for _, typ in declared],
            name+': declaration/definition type mismatch')
    pointer_parameter(parameters, 'stretched_dz_d', 'constGpu::DeviceVector<Real>&')
    reject_uses(no_literals(text)[definition.start():definition.end()],
                [*NAMES, 'dz_ptr', 'ParallelFor'], [], name+' parameter shadows')
    prefix = re.sub(r'(?m)^\s*#include\s*[<"][\w./]+[>"]\s*$', '', text[:definition.start()])
    prefix = re.sub(r'\busing\s+namespace\s+amrex\s*;', '', prefix)
    require(not prefix.strip(), name+': unsupported function prefix')
    require(not text[definition.end()+len(body)+1:].strip(), name+': unsupported function suffix')
    # Expand this statement fragment, rather than assuming it has no effects.
    body = text[definition.end():definition.end()+len(body)]
    if path == STATE:
        include = unique(r'(?m)^\s*#include\s+"ERF_SetupScalarDiffusion.H"\s*$', body, 'state setup include')
        host_scope(body, include.start())
        body = body[:include.start()]+texts[SETUP]+body[include.end():]
    require('#' not in no_literals(body), name+': unsupported body directive')
    return no_literals(body)


def header_function(text, name):
    parameters, body, definition = signature(no_literals(text), name)
    prefix = ('template <bool MultiplyMolecularByDensity, bool AddTurbulence> '
              if name == IMPL else '') + 'inline'
    prefix_match = unique(pattern(prefix)+r'\s*$', text[:definition.start()], name+' definition prefix')
    declaration_scope(text, prefix_match.start(), GUARDS[SCALAR])
    pointer_parameter(parameters, 'dz', 'constReal*')
    reject_uses(definition[1], [*NAMES, 'ParallelFor'], [], name+' parameter shadows')
    require('#' not in body, name+': unsupported body directive')
    return parameters, body, (prefix_match.start(), definition.end()+len(body)+1)


def forward(body, name, parameters, pointer, count=1, template=False, origin=None):
    callee = r'(?<![\w:.>])'+name+(r'\s*<\s*(?:true|false)\s*,\s*(?:true|false)\s*>' if template else '')
    calls = list(call_arguments(body, callee))
    require(len(calls) == count, name+': forwarding call inventory changed')
    uses, spans = [], []
    for arguments in calls:
        values = [compact(body[lo:hi]) for lo, hi in arguments]
        binding(values, parameters, {'dz': ('constReal*', pointer)}, name+' forwarding')
        start = body.rfind(name, 0, arguments[0][0])
        standalone(body, start)
        host_scope(body, start)
        if origin is not None:
            before = scopes(body, origin.start())
            require(origin.end() < start and scopes(body, start)[:len(before)] == before,
                    'spacing pointer does not dominate forwarding')
        uses.append((start, arguments[0][0]))
        spans.append(arguments[[key for key, _ in parameters].index('dz')])
    reject_uses(body, [name], uses, name+' callee uses')
    return spans


def kernel_reads(body, statements, pointer, origin=None):
    # A parameter is available at body entry. A zero-width match expresses that
    # origin for the shared domination/scope checker without changing it.
    if origin is None:
        origin = re.match('', body)
    regions = kernel_regions(body)
    spans = []
    for statement in statements:
        read = unique(pattern(statement), body, pointer+' scalar read')
        read_context(body, read.start(), origin, regions)
        spans.append(read.span())
    return spans


def check(texts):
    text = no_literals(texts[SCALAR])
    definitions = []
    for name, expected in ((RESULT, 'struct '+RESULT+' { amrex::Real c1, c2, c3; };'),
                           (WEIGHTS, WEIGHT_DEFINITION), (GRAD, GRAD_DEFINITION)):
        match = unique(pattern(expected), text, name+' complete definition')
        declaration_scope(text, match.start(), GUARDS[SCALAR])
        definitions.append(match.span())
    print('  ok   boundary weights take scalar copies and return a plain scalar aggregate')
    print('  ok   stretched gradient only reads spacing and returns a scalar')
    functions = {name: header_function(text, name) for name in (BUILD, IMPL, DIV)}
    definitions += [item[2] for item in functions.values()]
    # Additional overloads, specializations, aliases or shadows cannot silently
    # replace the checked functions. Their supported internal uses are below.
    reject_uses(text, NAMES, definitions, 'scalar header declarations')

    _, body, _ = functions[BUILD]
    spans = forward(body, IMPL, functions[IMPL][0], 'dz', count=4, template=True)
    reject_uses(body, ['dz'], spans, 'scalar dispatch spacing uses')
    print('  ok   all four scalar-flux dispatch arms forward the same const pointer')

    _, body, _ = functions[IMPL]
    spans = kernel_reads(body, [GRAD_READ], 'dz')
    # Check position/type against the locked leaf signature as well.
    leaf_params = params(re.search(r'StretchedScalarGradient\((.*?)\) noexcept', GRAD_DEFINITION, re.S)[1])
    args = list(call_arguments(body, r'(?<![\w:.>])'+GRAD))
    require(len(args) == 1, 'gradient call inventory changed')
    binding([compact(body[lo:hi]) for lo, hi in args[0]], leaf_params,
            {'dz': ('constReal*', 'dz')}, 'gradient forwarding')
    reject_uses(body, ['dz', GRAD], spans, 'scalar flux spacing uses')
    print('  ok   scalar-flux kernel captures the pointer by value for gradient evaluation')

    _, body, _ = functions[DIV]
    spans = kernel_reads(body, [DIV_READ], 'dz')
    reject_uses(body, ['dz'], spans, 'scalar divergence spacing uses')
    print('  ok   divergence kernel captures the pointer by value and reads a scalar')

    for path, name in ((STATE, 'DiffusionSrcForState_S'), (MOM, 'DiffusionSrcForMom')):
        body = entry_body(texts, path, name)
        alias = unique(pattern('auto dz_ptr = stretched_dz_d.data();'), body, name+' data origin')
        standalone(body, alias.start())
        host_scope(body, alias.start())
        spans = [alias.span()]
        if path == STATE:
            for callee in (BUILD, DIV):
                spans += forward(body, callee, functions[callee][0], 'dz_ptr', origin=alias)
        else:
            spans += kernel_reads(body, MOM_READS, 'dz_ptr', alias)
        reject_uses(body, ['stretched_dz_d', 'dz_ptr'], spans, name+' spacing uses')
        print('  ok   '+name+': every spacing use has a checked data origin and read/forwarding path')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    check(load(root))
    print('7 spacing diffusion source contracts passed; GPU completion remains conditional.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        sys.exit(1)
