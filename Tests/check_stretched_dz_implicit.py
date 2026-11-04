#!/usr/bin/env python3
"""Guard spacing preservation in stretched-grid implicit diffusion.

Check the state and momentum bodies with AMREX_USE_GPU both enabled and disabled.
Expand the state setup fragment, verify const-vector signatures and data origins,
and admit only nine scalar spacing-read statements per routine. GPU reads must
be inside the bare value-captured column kernel; CPU reads must be inside the
direct column loops. Nested ordinary blocks/loops are allowed, closures are not.
The momentum instantiation macro is checked as five logical lines after C++ line
splicing; all other continuations and unsupported directives fail closed.

These five finite source-contract groups add no arithmetic theorem. Assumptions:
valid initialized storage/indices, ordinary C++/AMReX semantics and resolution
to checked declarations, disjoint output arrays, no external macro/overload
rewriting, concurrency or undefined behavior, and preservation through other
inputs/globals. These narrow readers are not a C++ parser or whole-program alias
analysis. GPU completion before spacing storage changes, setup-array lifetimes,
refresh ordering, bounds, positivity, solver formulas and convergence are unproved.
Only the Python standard library is needed.

usage: check_stretched_dz_implicit.py [ERF_ROOT]
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_advection import vector_tokens
from check_stretched_dz_bindings import (brace_block, no_literals, params,
                                       reject_uses, scopes, signature)
from check_stretched_dz_consumers import declaration_scope, host_scope, pattern, standalone
from momentum_geometry import mask, unique

SOURCE = 'Source/Diffusion/ERF_ImplicitDiff_S.cpp'
HEADER = 'Source/Diffusion/ERF_Diffusion.H'
SETUP = 'Source/Diffusion/ERF_SetupVertDiff.H'
FILES = (SOURCE, HEADER, SETUP)
STATE = 'ImplicitDiffForStateLU_S'
MOM = 'ImplicitDiffForMomLU_S'
MACRO = 'INSTANTIATE_IMPLICIT_DIFF_FOR_MOM_LU'
GUARD = 'ERF_DIFFUSION_H_'
VECTOR = 'stretched_dz_d'
SCALARS = ('dz_inv', 'dz_inv_lo', 'dz_inv_hi')
MACRO_BODY = '''#define INSTANTIATE_IMPLICIT_DIFF_FOR_MOM_LU(STAGDIR)
    template void ImplicitDiffForMomLU_S<STAGDIR>(
        const Box&, const Box&, const int, const double,
        const Array4<const int>&, const Array4<const Real>&,
        const Array4<Real>&, const Array4<const Real>&, const Array4<const Real>&,
        const Gpu::DeviceVector<Real>&, const Array4<const Real>&,
        const SolverChoice&, const BCRec*, const bool, const Real, const bool);'''
READS = tuple(statement for index in ('klo', 'k', 'khi') for statement in (
    f'dz_inv = one / dz_ptr[{index}];',
    f'dz_inv_lo = two / (dz_ptr[{index}] + dz_ptr[{index}-1]);',
    f'dz_inv_hi = two / (dz_ptr[{index}] + dz_ptr[{index}+1]);'))
GPU_COLUMN = 'ParallelFor(makeSlab(bx,2,0), [=] AMREX_GPU_DEVICE(int i,int j,int) {'
CPU_COLUMN = 'for(int j(jlo);j<=jhi;++j) { for(int i(ilo);i<=ihi;++i) {'


def instantiations(raw):
    start = unique(r'(?m)^[ \t]*#\s*define\s+'+MACRO+r'\b[^\n]*', raw, 'instantiation macro start')
    end = unique(r'(?m)^[ \t]*#\s*undef\s+'+MACRO+r'\b[^\n]*', raw, 'instantiation macro end')
    require(start.start() < end.start(), 'instantiation macro order')
    block = raw[start.start():end.end()]
    # Splice first, then remove comments, matching C++ translation order.
    logical = strip_comments(re.sub(r'\\\r?\n', '', block))
    lines = [line.strip() for line in logical.splitlines() if line.strip()]
    expected = [MACRO_BODY, *(MACRO+f'({index})' for index in (0, 1, 2)), '#undef '+MACRO]
    require(len(lines) == len(expected) and all(re.fullmatch(pattern(want), line)
            for line, want in zip(lines, expected)), 'unsupported momentum instantiation macro')
    # Require function-like macro syntax: whitespace before '(' changes its meaning.
    require(re.match(r'#\s*define\s+'+MACRO+r'\(', lines[0]), 'instantiation must be a function-like macro')
    require(not strip_comments(raw[end.end():]).strip(), 'instantiations must end the source file')
    remaining = raw[:start.start()]+raw[end.end():]
    reject_uses(remaining, [MACRO], [], 'extra instantiation macro uses')
    print('  ok   exact momentum instantiations for stagdir 0, 1 and 2')
    return remaining


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported C++ lexical form')
        if path == SOURCE:
            raw = instantiations(raw)
        require(not re.search(r'\\\r?\n', raw), path+': unsupported C++ line continuation')
        text = strip_comments(raw)
        for directive in re.finditer(r'(?m)^\s*#\s*(define|undef)\b([^\n]*)', text):
            require(path == HEADER and directive[1] == 'define' and directive[2].strip() == GUARD,
                    path+': unsupported macro')
        texts[path] = text
    return texts


def select_gpu(text, gpu):
    """Evaluate only AMREX_USE_GPU conditionals; reject unknown directives."""
    result, stack = [], []
    for line in text.splitlines(keepends=True):
        match = re.match(r'\s*#\s*(\w+)\b(.*)', line)
        active = all(value for value, _ in stack)
        if not match:
            if active:
                result.append(line)
            continue
        kind, argument = match[1], match[2].strip()
        if kind == 'ifdef':
            require(argument == 'AMREX_USE_GPU', 'unsupported implicit conditional')
            stack.append((gpu, False))
        elif kind == 'else':
            require(stack and not argument and not stack[-1][1], 'invalid implicit else')
            value, _ = stack.pop()
            stack.append((not value, True))
        elif kind == 'endif':
            require(stack and not argument, 'invalid implicit endif')
            stack.pop()
        else:
            require(kind == 'include', 'unsupported implicit directive')
            if active:
                result.append(line)
    require(not stack, 'unclosed implicit conditional')
    return ''.join(result)


def function(texts, configured, name):
    parameters, body, definition = signature(no_literals(configured), name)
    _, _, declaration = signature(no_literals(texts[HEADER]), name, ';')
    declared_text = declaration[1]
    if name == STATE:
        default = unique(pattern('const bool use_mrf_countergradient = false')+r'\s*$',
                         declared_text, 'state default argument')
        declared_text = declared_text[:default.start()]+'const bool use_mrf_countergradient'
    declared = params(declared_text)
    require([typ for _, typ in parameters] == [typ for _, typ in declared],
            name+': declaration/definition type mismatch')
    require([(key, typ) for key, typ in parameters if key == VECTOR] ==
            [(VECTOR, 'constGpu::DeviceVector<Real>&')], name+': spacing parameter type')
    vector_tokens(definition[1])
    vector_tokens(declared_text)
    reject_uses(definition[1], ['dz_ptr', 'ParallelFor', *SCALARS], [], name+' parameter shadows')
    declaration_scope(texts[HEADER], declaration.start(), GUARD)
    reject_uses(texts[HEADER], [name], [declaration.span()], name+' header declarations')
    start = definition.start()
    if name == MOM:
        template = 'template <int stagdir>'
        prefix = unique(pattern(template)+r'\s*$', configured[:start], 'momentum template prefix')
        start = prefix.start()
        unique(pattern(template)+r'\s*$', texts[HEADER][:declaration.start()], 'momentum declaration template')
    require(not scopes(configured, start), name+': definition must be at global scope')
    # Restore strings before resolving the runtime include; masking must not
    # erase its file name or make a hidden directive look harmless.
    raw_body = configured[definition.end():definition.end()+len(body)]
    if name == STATE:
        include = unique(r'(?m)^[ \t]*#include\s+"ERF_SetupVertDiff.H"[ \t]*$', raw_body, 'state setup include')
        host_scope(raw_body, include.start())
        raw_body = raw_body[:include.start()]+texts[SETUP]+raw_body[include.end():]
    require('#' not in no_literals(raw_body), name+': unsupported body directive')
    return no_literals(raw_body), (start, definition.end()+len(body)+1)


def column(body, gpu):
    launch = unique(pattern(GPU_COLUMN if gpu else CPU_COLUMN), body, 'column execution scope')
    standalone(body, launch.start())
    host_scope(body, launch.start())
    contents = brace_block(body, launch.end()-1)
    tail = body[launch.end()+len(contents):]
    require(re.match(r'\}\s*\)\s*;' if gpu else r'\}\s*\}', tail), 'column must close directly')
    allowed = [launch.span()] if gpu else []
    reject_uses(body, ['ParallelFor'], allowed, 'column launcher')
    return launch, contents


def scalar_targets(contents):
    declaration = unique(pattern('Real dz_inv, dz_inv_lo, dz_inv_hi;'), contents, 'scalar inverse locals')
    standalone(contents, declaration.start())
    require(not scopes(contents, declaration.start()), 'inverse locals must be at column scope')
    # The targets must retain their scalar declarations. Reject new declarators,
    # reference bindings, structured bindings or qualified/shadowed targets.
    remainder = mask(contents, [declaration.span()])
    for use in re.finditer(r'\b(?:dz_inv|dz_inv_lo|dz_inv_hi)\b', remainder):
        before = remainder[:use.start()].rstrip()
        require(not before or before[-1] in ';{}=+-*/(,?:', 'unsupported scalar target binding')
    return declaration


def check_body(body, gpu, name):
    alias = unique(pattern('auto dz_ptr = stretched_dz_d.data();'), body, name+' data origin')
    standalone(body, alias.start())
    host_scope(body, alias.start())
    launch, contents = column(body, gpu)
    origin_scope = scopes(body, alias.start())
    require(alias.end() < launch.start() and scopes(body, launch.start())[:len(origin_scope)] == origin_scope,
            'spacing pointer does not dominate column execution')
    local = scalar_targets(contents)
    spans = [alias.span()]
    for statement in READS:
        read = unique(pattern(statement), body, name+' spacing scalar read')
        offset = read.start()-launch.end()
        require(0 <= offset < len(contents), 'spacing read is outside its column')
        require(local.end() < offset, 'spacing read precedes scalar locals')
        host_scope(contents, offset)
        standalone(contents, offset)
        spans.append(read.span())
    reject_uses(body, [VECTOR, 'dz_ptr'], spans, name+' spacing uses')
    print(f'  ok   {name} {"GPU" if gpu else "CPU"}: const vector, data origin and nine scalar reads')


def check(texts):
    for gpu in (False, True):
        configured = select_gpu(texts[SOURCE], gpu)
        spans = []
        for name in (STATE, MOM):
            body, span = function(texts, configured, name)
            spans.append(span)
            check_body(body, gpu, name)
        remainder = mask(configured, spans)
        remainder = re.sub(r'(?m)^\s*#include\s*"[\w.]+"\s*$', '', remainder)
        remainder = re.sub(r'\busing\s+namespace\s+amrex\s*;', '', remainder)
        require(not remainder.strip(), 'unsupported source outside implicit definitions')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    check(load(root))
    print('5 spacing implicit source contracts passed; GPU completion remains conditional.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        sys.exit(1)
