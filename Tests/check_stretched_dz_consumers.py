#!/usr/bin/env python3
"""Guard explicit spacing uses in tau, fast-substep and strain consumers.

This finite source contract audits the complete bodies of erf_make_tau_terms,
erf_substep_NS and ComputeStrain_S. It checks their declaration/definition
types, the tau-to-strain argument, one data pointer per body, and every use
of that pointer or the vector. Supported reads are complete scalar statements
inside value-captured ParallelFor kernels; unsupported uses fail closed.
The fixed statement inventory is a preservation contract, not a proof of the
numerical formulas. This checker uses only the Python standard library.

Assumptions: valid initialized storage; normal C++/AMReX semantics; no macro
rewriting from external headers, concurrent mutation, undefined behavior or
preexisting aliases through other inputs/globals. Output arrays must not alias
spacing. External routines must preserve spacing through those other aliases.
ParallelFor may retain a pointer copy for GPU execution: completion before a
resize/destruction remains an external obligation. This is not a whole-program
alias, index-bound, numerical-accuracy or GPU-lifetime proof. Other consumers,
including slow-RHS and advection/diffusion paths, remain separate obligations.

usage: check_stretched_dz_consumers.py [ERF_ROOT]
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (binding, brace_block, call, no_literals,
                                       reject_uses, scopes, signature)
from momentum_geometry import call_arguments, mask, unique

TAU = 'Source/TimeIntegration/ERF_MakeTauTerms.cpp'
FAST = 'Source/TimeIntegration/ERF_Substep_NS.cpp'
STRAIN = 'Source/Diffusion/ERF_ComputeStrain_S.cpp'
SLOW_H = 'Source/TimeIntegration/ERF_TI_slow_headers.H'
FAST_H = 'Source/TimeIntegration/ERF_TI_fast_headers.H'
STRAIN_H = 'Source/Diffusion/ERF_Diffusion.H'
FUNCTIONS = ((TAU, SLOW_H, 'erf_make_tau_terms', False),
             (FAST, FAST_H, 'erf_substep_NS', False),
             (STRAIN, STRAIN_H, 'ComputeStrain_S', True))
FILES = (TAU, FAST, STRAIN, SLOW_H, FAST_H, STRAIN_H)
PROTECTED = ('stretched_dz_d', 'dz_ptr')
GUARDS = {SLOW_H: 'ERF_SLOW_INTEGRATION_H_', FAST_H: 'ERF_FAST_INTEGRATION_H_',
          STRAIN_H: 'ERF_DIFFUSION_H_'}

# These complete statements use spacing only as scalar values. In particular,
# a reference declaration, assignment to an element, address-of, comma operator,
# cast or call receiving an element cannot match. Counts cover every current
# read, while reject_uses also catches new uses outside this inventory.
TAU_READ = '''er_arr(i,j,k) =
    (u(i+1,j,k)/mf_uy(i+1,j,0)-u(i,j,k)/mf_uy(i,j,0))*dxInv[0]*mfsq +
    (v(i,j+1,k)/mf_vx(i,j+1,0)-v(i,j,k)/mf_vx(i,j,0))*dxInv[1]*mfsq +
    (w(i,j,k+1)-w(i,j,k))/dz_ptr[k];'''
FAST_READ = '''R1_tmp += beta_1 * (
    ((Omega_kp1-Omega_k)/dz_ptr[k]+(Omega_k-Omega_km1)/dz_ptr[k-1])*halfg +
    (-(Omega_kp1*theta_t_hi-Omega_k*theta_t_mid)*coeff_P/dz_ptr[k]
     -(Omega_k*theta_t_mid-Omega_km1*theta_t_lo)*coeff_Q/dz_ptr[k-1]));'''
FACE_READ = '''Real dz_inv = (k == 0) ? one/dz_ptr[0] :
    (k >= nz_dz) ? one/dz_ptr[nz_dz-1] : two/(dz_ptr[k]+dz_ptr[k-1]);'''
READS = {
    TAU: ((TAU_READ, 1),),
    FAST: (('Real h_zeta = dz_ptr[k]*dzi;', 4), (FAST_READ, 1),
           ('Real dz_inv = one/dz_ptr[k];', 1)),
    STRAIN: ((FACE_READ, 10), ('Real dz0 = dz_ptr[k];', 3),
             ('Real dz0 = dz_ptr[k-1];', 2), ('Real dz1 = dz_ptr[k+1];', 2),
             ('Real dz1 = dz_ptr[k-2];', 2), ('Real dz_inv = one/dz_ptr[k];', 1)),
}


def pattern(statement):
    # Match tokens with boundaries; do not hide a read in myReal/other_dz_ptr.
    tokens = re.findall(r'\w+|::|[^\s]', statement)
    return r'(?<![\w:.>])' + r'\s*'.join(
        re.escape(token)+(r'\b' if re.fullmatch(r'\w+', token) else '') for token in tokens)


def standalone(text, start):
    prefix = text[:start].rstrip()
    require(not prefix or prefix[-1] in ';{}', 'protected use is not a standalone statement')


def no_outer_closures(body, stack):
    for kind, pos in stack:
        if kind == '{':
            require(not re.search(r'\][^;{}]*$', body[:pos].rstrip()),
                    'protected spacing use is nested in a closure')


def host_scope(body, start):
    stack = scopes(body, start)
    require(all(kind == '{' for kind, _ in stack), 'protected spacing use must be in host statement scope')
    no_outer_closures(body, stack)


def declaration_scope(text, start, guard):
    require(not scopes(no_literals(text), start), 'consumer declaration is nested in a scope')
    stack = []
    for match in re.finditer(r'(?m)^\s*#\s*(\w+)\b([^\n]*)', text[:start]):
        kind, argument = match[1], match[2].strip()
        if kind in ('if', 'ifdef', 'ifndef'):
            stack.append((kind, argument))
        elif kind == 'endif':
            require(stack, 'unbalanced declaration preprocessor')
            stack.pop()
        elif kind in ('else', 'elif'):
            require(stack, 'unbalanced declaration preprocessor')
            stack[-1] = (kind, argument)
    require(stack == [('ifndef', guard)], 'consumer declaration has an unsupported conditional scope')


def preprocess(body):
    """Audit both branches before removing known, spacing-free directives."""
    text = no_literals(body)
    omp = r'#ifdef\s+_OPENMP\s*\n\s*#pragma\s+omp\s+parallel\s+if\s*\(\s*Gpu::notInLaunchRegion\s*\(\s*\)\s*\)\s*\n\s*#endif'
    spans = [m.span() for m in re.finditer(omp, text)]
    other = r'#ifdef\s+(?:ERF_IMPLICIT_W|AMREX_USE_GPU)\s*\n([^#]*?)#else\s*\n([^#]*?)#endif'
    for match in re.finditer(other, text):
        reject_uses(match[0], PROTECTED, [], 'conditional region')
        spans.extend((match.start()+m.start(), match.start()+m.end())
                     for m in re.finditer(r'#[^\n]*', match[0]))
    text = mask(text, spans)
    require('#' not in text, 'unsupported consumer preprocessor directive')
    return text


def kernel_regions(body):
    """Find complete inline kernels passed as bare ParallelFor arguments."""
    regions = []
    callee = r'(?<![\w:.>])(?:amrex::)?ParallelFor'
    launches = list(re.finditer(callee+r'\s*\(', body))
    reject_uses(body, ['ParallelFor'], [m.span() for m in launches], 'kernel launcher')
    for launch, args in zip(launches, call_arguments(body, callee)):
        opening = args[0][0]-1
        for lo, hi in args:
            argument = body[lo:hi]
            header = re.match(r'\s*\[\s*=\s*\]\s*AMREX_GPU_DEVICE\s*'
                              r'\(\s*int\s+i\s*,\s*int\s+j\s*,\s*int\s+k\s*\)\s*'
                              r'(?:noexcept\s*)?\{', argument)
            if not header:
                continue
            start = lo+header.end()-1
            contents = brace_block(body, start)
            end = start+1+len(contents)
            if body[end:hi].strip() == '}':
                regions.append((start, end, opening, launch.start()))
    return regions


def read_context(body, start, alias, regions):
    enclosing = [region for region in regions if region[0] < start < region[1]]
    require(len(enclosing) == 1, 'spacing read must be in a bare value-captured ParallelFor kernel')
    lo, _, opening, launch = enclosing[0]
    standalone(body, launch)
    stack = scopes(body, start)
    require(stack[-1] == ('{', lo), 'spacing read has an unsupported nested scope')
    require([pos for kind, pos in stack if kind == '('] == [opening],
            'spacing kernel is nested in an unsupported call')
    # Reject outer closures even if they omit their parameter list. They could
    # defer or retain the checked kernel through an implicit pointer capture.
    no_outer_closures(body, stack[:-1])
    alias_scope = scopes(body, alias.start())
    require(alias.end() < start and stack[:len(alias_scope)] == alias_scope,
            'spacing pointer does not dominate its read')
    standalone(body, start)


def check_function(texts, path, header, name, const):
    text = texts[path]
    parameters, raw, definition = signature(no_literals(text), name)
    declared, _, declaration = signature(no_literals(texts[header]), name, ';')
    declaration_scope(texts[header], declaration.start(), GUARDS[header])
    require([typ for _, typ in parameters] == [typ for _, typ in declared],
            name+': declaration/definition type mismatch')
    expected = ('const' if const else '')+'Gpu::DeviceVector<Real>&'
    require([(key, typ) for key, typ in parameters if key == PROTECTED[0]] == [(PROTECTED[0], expected)],
            name+': spacing parameter type')
    # Permit declaration includes/namespace imports before the function, but
    # no extra executable code/macros. The tau file has one unrelated helper.
    prefix = text[:definition.start()]
    prefix = re.sub(r'(?m)^\s*#include\s*[<"][\w./]+[>"]\s*$', '', prefix)
    prefix = re.sub(r'\busing\s+namespace\s+amrex\s*;', '', prefix)
    require(not prefix.strip(), name+': unsupported function prefix')
    end = definition.end()+len(raw)+1
    suffix = no_literals(text[end:])
    if path == TAU:
        _, helper, match = signature(suffix, 'copy_surface_tau_for_implicit')
        reject_uses(helper, PROTECTED, [], 'unrelated tau helper')
        require('#' not in helper, 'unsupported tau helper directive')
        suffix = suffix[:match.start()]+suffix[match.end()+len(helper)+1:]
    require(not suffix.strip(), name+': unsupported function suffix')
    body = preprocess(raw)
    alias = unique(pattern('auto dz_ptr = stretched_dz_d.data();'), body, name+' data pointer')
    standalone(body, alias.start())
    host_scope(body, alias.start())
    spans = [alias.span()]
    if path == STRAIN:
        size = unique(pattern('const int nz_dz = static_cast<int>(stretched_dz_d.size());'), body, 'spacing size')
        standalone(body, size.start())
        spans.append(size.span())
    if path == TAU:
        args, values = call(body, 'ComputeStrain_S')
        callee, _, _ = signature(texts[STRAIN], 'ComputeStrain_S')
        binding(values, callee, {PROTECTED[0]: ('constGpu::DeviceVector<Real>&', PROTECTED[0])}, 'tau to strain')
        at = body.rfind('ComputeStrain_S', 0, args[0][0])
        standalone(body, at)
        host_scope(body, at)
        spans.append(args[[key for key, _ in callee].index(PROTECTED[0])])
    regions = kernel_regions(body)
    reads = 0
    for statement, count in READS[path]:
        matches = list(re.finditer(pattern(statement), body))
        require(len(matches) == count, name+': scalar read inventory changed: '+statement.split('=')[0].strip())
        for match in matches:
            read_context(body, match.start(), alias, regions)
            spans.append(match.span())
        reads += count
    reject_uses(body, PROTECTED, spans, name+' spacing uses')
    print(f'  ok   {name}: matching signature, data origin and {reads} scalar read statements')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported C++ lexical form')
        text = strip_comments(raw)
        for directive in re.finditer(r'(?m)^\s*#\s*(define|undef)\b([^\n]*)', text):
            require(directive[1] == 'define' and directive[2].strip() == GUARDS.get(path),
                    path+': unsupported macro')
        texts[path] = text
    for spec in FUNCTIONS:
        check_function(texts, *spec)
    print('3 spacing consumer source contracts passed; GPU completion remains conditional.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        sys.exit(1)
