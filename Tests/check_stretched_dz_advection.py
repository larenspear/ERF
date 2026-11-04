#!/usr/bin/env python3
"""Guard spacing preservation through the normal momentum advection path.

Four finite source contracts audit both normal-grid entry points, the vertical
dispatcher and its wrapper. All 36 vector-forwarding sites must bind the same
const-reference vector. Two local data pointers feed six scalar inverse-spacing
statements inside bare value-captured GPU kernels. Every other explicit use of
the vector or pointer fails. This complements the numerical/dispatch checks;
it does not prove the formulas, selector behavior or index bounds.

Assumptions: valid initialized spacing storage/indices; ordinary C++/AMReX
semantics and resolution to the checked declarations; no external macro or
overload rewriting, concurrent mutation or undefined behavior; disjoint output
arrays; and preservation through other inputs/globals. These source readers
are not a C++ parser or whole-program alias analysis. GPU kernels retain pointer
copies, so completion before storage changes remains an external obligation.
check_stretched_dz_implicit audits the stretched implicit-diffusion pointer paths.
Refresh scheduling and generated-spacing positivity are separate obligations.
Only the Python standard library is needed.

usage: check_stretched_dz_advection.py [ERF_ROOT]
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import SCHEMES, compact, require
from check_stretched_dz_bindings import (binding, no_literals, reject_uses,
                                       scopes, signature)
from check_stretched_dz_consumers import (declaration_scope, host_scope,
    kernel_regions, pattern, read_context, standalone)
from momentum_geometry import call_arguments, unique

CONSTANT = 'Source/Advection/ERF_AdvectionSrcForMom_ConstantDz.cpp'
STRETCHED = 'Source/Advection/ERF_AdvectionSrcForMom_StretchedDz.cpp'
HEADER = 'Source/Advection/ERF_Advection.H'
NORMAL = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'
FILES = (CONSTANT, STRETCHED, HEADER, NORMAL)
GUARD = 'ERF_ADVECTION_H_'
VERT = 'AdvectionSrcForMomVert_N'
WRAPPER = 'AdvectionSrcForMomWrapper_N'
VECTOR = 'stretched_dz_d'
VECTOR_TYPE = 'constGpu::DeviceVector<Real>&'
FUNCTIONS = ((CONSTANT, 'AdvectionSrcForMom_ConstantDz'),
             (STRETCHED, 'AdvectionSrcForMom_StretchedDz'),
             (NORMAL, VERT), (NORMAL, WRAPPER))
TEMPLATES = {VERT: 'template<typename InterpType_H>',
             WRAPPER: 'template<typename InterpType_H, typename InterpType_V, typename WallInterpType>'}
READS = (('dzInv = one/dz_ptr[k];', 2),
         ('dzInv = (k == 0) ? one/dz_ptr[k] : two/(dz_ptr[k]+dz_ptr[k-1]);', 1))


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported C++ lexical form')
        text = strip_comments(raw)
        for macro in re.finditer(r'(?m)^\s*#\s*(define|undef)\b([^\n]*)', text):
            require(path == HEADER and macro[1] == 'define' and macro[2].strip() == GUARD,
                    path+': unsupported macro')
        texts[path] = text
    return texts


def function(texts, path, name):
    text = texts[path]
    parameters, body, definition = signature(no_literals(text), name)
    require([(key, typ) for key, typ in parameters if key == VECTOR] == [(VECTOR, VECTOR_TYPE)],
            name+': spacing parameter type')
    vector_tokens(definition[1])
    reject_uses(definition[1], ['dz_ptr', VERT, WRAPPER, 'ParallelFor'], [], name+' parameter shadows')
    require('#' not in body, name+': unsupported body directive')
    end = definition.end()+len(body)+1
    start = definition.start()
    if path == NORMAL:
        prefix = unique(pattern(TEMPLATES[name])+r'\s*$', text[:start], name+' template prefix')
        start = prefix.start()
        require(not scopes(text, start), name+': definition must be at global scope')
    else:
        declared, _, declaration = signature(no_literals(texts[HEADER]), name, ';')
        vector_tokens(declaration[1])
        declaration_scope(texts[HEADER], declaration.start(), GUARD)
        reject_uses(texts[HEADER], [name], [declaration.span()], name+' header declarations')
        require([typ for _, typ in parameters] == [typ for _, typ in declared],
                name+': declaration/definition type mismatch')
        prefix = re.sub(r'(?m)^\s*#include\s*[<"][\w./]+[>"]\s*$', '', text[:start])
        prefix = re.sub(r'\busing\s+namespace\s+amrex\s*;', '', prefix)
        require(not prefix.strip(), name+': unsupported function prefix')
        require(not text[end:].strip(), name+': unsupported function suffix')
    return parameters, body, (start, end)


def vector_tokens(parameters):
    # Compact type comparisons alone would conflate the const keyword with a
    # different namespace named constGpu. Preserve that token boundary here.
    unique(r'(?:^|,)\s*const\s+(?:amrex\s*::\s*)?Gpu\s*::\s*DeviceVector\s*<\s*'
           r'(?:amrex\s*::\s*)?Real\s*>\s*&\s*stretched_dz_d\s*(?=,|$)',
           parameters, 'const spacing-reference parameter tokens')


def forwarding(body, callee, parameters):
    spans, uses = [], []
    for scheme in SCHEMES.values():
        types = scheme if callee == VERT else 'InterpType_H,'+scheme+',UPWINDALL'
        expression = pattern(callee+'<'+types+'>')
        calls = list(call_arguments(body, expression))
        require(len(calls) == 1, callee+'<'+types+'>: forwarding call inventory changed')
        arguments = calls[0]
        binding([compact(body[lo:hi]) for lo, hi in arguments], parameters,
                {VECTOR: (VECTOR_TYPE, VECTOR)}, callee+' forwarding')
        start = body.rfind(callee, 0, arguments[0][0])
        standalone(body, start)
        host_scope(body, start)
        uses.append((start, arguments[0][0]))
        spans.append(arguments[[key for key, _ in parameters].index(VECTOR)])
    reject_uses(body, [callee], uses, callee+' callee uses')
    return spans


def pointer_reads(body, typ):
    alias = unique(pattern('auto dz_ptr = stretched_dz_d.data();'), body, 'spacing data origin')
    standalone(body, alias.start())
    host_scope(body, alias.start())
    regions = kernel_regions(body)
    spans = [alias.span()]
    for statement, count in READS:
        reads = list(re.finditer(pattern(typ+' '+statement), body))
        require(len(reads) == count, 'spacing scalar-read inventory changed')
        for read in reads:
            read_context(body, read.start(), alias, regions)
            spans.append(read.span())
    return spans


def check(texts):
    functions = {name: function(texts, path, name) for path, name in FUNCTIONS}
    header = texts[NORMAL]
    # This header has no guard. Admit only its two declaration includes, each
    # at global scope; all conditional compilation and statement includes fail.
    directives = list(re.finditer(r'(?m)^\s*#[^\n]*', header))
    expected = {'ERF_IndexDefines.H', 'ERF_Interpolation.H'}
    found = []
    for directive in directives:
        include = re.fullmatch(r'\s*#include\s+"([\w.]+)"\s*', directive[0])
        require(include and include[1] in expected and not scopes(header, directive.start()),
                'normal header: unsupported directive or include scope')
        found.append(include[1])
    require(len(found) == 2 and set(found) == expected, 'normal header include inventory changed')
    reject_uses(header, [VERT, WRAPPER, VECTOR, 'dz_ptr'],
                [functions[name][2] for name in (VERT, WRAPPER)], 'normal header declarations')

    for path, name in FUNCTIONS:
        _, body, _ = functions[name]
        spans = []
        if name != WRAPPER:
            callee = WRAPPER if name == VERT else VERT
            spans += forwarding(body, callee, functions[callee][0])
        if path == STRETCHED or name == WRAPPER:
            spans += pointer_reads(body, 'Real' if path == STRETCHED else 'auto')
        reject_uses(body, [VECTOR, 'dz_ptr'], spans, name+' spacing uses')
        print('  ok   '+name+': all spacing uses have checked reference bindings or value-captured scalar reads')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    check(load(root))
    print('4 spacing advection source contracts passed; GPU completion remains conditional.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        sys.exit(1)
