#!/usr/bin/env python3
"""Guard FFT spacing subcolumn ownership and the later member-spacing reader.

Five finite source contracts cover the member interface/solver types, automatic
host/device subcolumns with exact indexing and blocking copy, an automatic
MFIter enclosing four bare value-captured kernels, and the full-level spacing
pointer forwarded only to the bounded inline spacing helper in the stretched
vertical-flux kernel. Physical endpoint faces read one adjacent cell height;
interior faces average two. Given 0 <= lo <= hi, lo <= k <= hi+1, representable
hi+1 and storage at indices 0..hi, these accesses are bounded, including a
one-cell column.

The separately inspected pinned AMReX DeviceVector overload forwards spacing
only to TriA/TriC in solve_z, whose GPU reader has a streamSynchronize directly
after each launch. Thus that temporary's reads finish before the solver returns,
under intended types/callees, stable streams, GPU launch mode, no graph capture,
normal execution and valid data. The later member reads instead rely on the
previously inspected default-sync MFIter completion argument. Both arguments
are conditional dependency audits, not machine-checked AMReX theorems.

Assume a live ERF, initialized consistent host/device spacing, valid nonempty
subdomain/indices, matching initialized solver geometry, disjoint output fields,
no undefined behavior/nonlocal transfer/concurrent launches, and preservation of
spacing and stream/sync state by other code, headers, callees, aliases and globals.
The nonconstant branch must select a supported stretched grid. Header macros
must have their intended meanings. No FFT/tridiagonal numerical correctness,
positivity and caller-established index bounds, solver construction/caller provenance, full solver
completion, boundary-condition internals or whole-program lifetime is proved.
Only the Python standard library is required.
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require
from check_stretched_dz_bindings import (CLASS, brace_block, direct, no_literals,
    reject_uses, scopes, signature)
from check_stretched_dz_consumers import (declaration_scope, kernel_regions,
    pattern, standalone)
from check_stretched_dz_mfiter import CONTROLS, enclosing, header
from momentum_geometry import call_arguments, literal, unique

SOURCE = 'Source/LinearSolvers/ERF_SolveWithFFT.cpp'
UTILS = 'Source/LinearSolvers/ERF_SolverUtils.H'
FILES = (SOURCE, CLASS, UTILS)
DEFINITION = '''void ERF::solve_with_fft(int lev, int isub, const Box& subdomain,
    MultiFab& rhs, MultiFab& phi, Array<MultiFab,AMREX_SPACEDIM>& fluxes) {'''
LOCALS = '''auto const sub_lo = lbound(subdomain);
    auto const sub_hi = ubound(subdomain);
    Vector<Real> stretched_dz_h_sub;
    Gpu::DeviceVector<Real> stretched_dz_d_sub;'''
CONSTANT = 'if (solverChoice.mesh_type == MeshType::ConstantDz) {'
UNIFORM = 'm_3D_poisson[lev][isub]->solve(phi, rhs);'
SUBCOLUMN = '''stretched_dz_h_sub.resize(subdomain.length(2));
    for (int k = sub_lo.z; k <= sub_hi.z; k++) {
        stretched_dz_h_sub[k-sub_lo.z] = stretched_dz_h[lev][k];
    }
    stretched_dz_d_sub.resize(stretched_dz_h_sub.size());
    Gpu::copy(Gpu::hostToDevice, stretched_dz_h_sub.begin(),
        stretched_dz_h_sub.end(), stretched_dz_d_sub.begin());
    m_2D_poisson[lev][isub]->solve(phi, rhs, stretched_dz_d_sub);'''
BC = 'ImposeBCsOnPhi(lev, phi, subdomain);'
GEOMETRY = '''auto dxInv = geom[lev].InvCellSizeArray();
    const Real dx_inv = dxInv[0]; const Real dy_inv = dxInv[1];
    auto const dom_lo = lbound(geom[lev].Domain());
    auto const dom_hi = ubound(geom[lev].Domain());
    auto const bc_type_zlo = domain_bc_type[Orientation(2,Orientation::low)];
    const bool bottom_is_dirichlet = (sub_lo.z == dom_lo.z)
        && (bc_type_zlo == "Outflow" || bc_type_zlo == "Open");
    auto const bc_type_zhi = domain_bc_type[Orientation(2,Orientation::high)];
    const bool top_is_dirichlet = (sub_hi.z == dom_hi.z)
        && (bc_type_zhi == "Outflow" || bc_type_zhi == "Open");'''
LOOP = header('phi', 'TilingIfNotGPU()')
STRETCHED = 'if (solverChoice.mesh_type != MeshType::ConstantDz) {'
ALIAS = 'Real* stretched_dz_d_ptr = stretched_dz_d[lev].data();'
BOUNDARY = 'if ((k == sub_lo.z && !bottom_is_dirichlet) || (k == sub_hi.z+1 && !top_is_dirichlet))'
READ = 'Real dz = projection_vertical_spacing(k, sub_lo.z, sub_hi.z, stretched_dz_d_ptr);'
VERTICAL = BOUNDARY+''' { fz_arr(i,j,k) = zero; } else {
    '''+READ+''' fz_arr(i,j,k) = -(p_arr(i,j,k) - p_arr(i,j,k-1)) / dz; }'''
CONSTANT_VERTICAL = BOUNDARY+''' { fz_arr(i,j,k) = zero; } else {
    fz_arr(i,j,k) = -(p_arr(i,j,k) - p_arr(i,j,k-1)) * dz_inv; }'''
CALLEE = r'(?<![\w:.>])(?:amrex::)?ParallelFor'
MEMBERS = tuple('amrex::Vector<amrex::Vector<std::unique_ptr<amrex::FFT::'+kind+
    '<amrex::MultiFab>>>> '+name+';' for kind, name in
    (('Poisson', 'm_3D_poisson'), ('PoissonHybrid', 'm_2D_poisson')))


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        text = strip_comments(raw)
        for macro in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(path in (CLASS,UTILS) and macro[1] == 'define' and
                    macro[2].strip() == {CLASS:'ERF_H_',UTILS:'ERF_SOLVER_UTILS_H_'}[path],
                    path+': unsupported macro')
        reject_uses(text, CONTROLS, [], path+' stream/sync controls')
        texts[path] = text
    return texts


def fft_member(text, start):
    """Require a direct class member under exactly one positive FFT guard."""
    require(not scopes(text, start), 'FFT member must be directly in ERF')
    stack = []
    for m in re.finditer(r'(?m)^\s*#\s*(\w+)\b([^\n]*)', text[:start]):
        kind, arg = m[1], m[2].strip()
        if kind in ('if', 'ifdef', 'ifndef'):
            stack.append((kind, arg))
        elif kind == 'endif':
            require(stack and not arg, 'unmatched FFT member endif')
            stack.pop()
        elif kind in ('else', 'elif'):
            require(stack, 'unmatched FFT member branch')
            stack[-1] = (kind, arg)
    require(stack == [('ifdef', 'ERF_USE_FFT')], 'FFT member conditional changed')
    prefix = re.sub(r'(?m)^\s*#[^\n]*', '', text[:start])
    standalone(prefix, len(prefix))


def interface(texts):
    text = texts[SOURCE]
    prefix = Source(text)
    literal(prefix, '#include "ERF.H" #include "ERF_SolverUtils.H" using namespace amrex; #ifdef ERF_USE_FFT void ERF::build_fft_solvers(int lev) {')
    _, build, build_def = signature(text, 'ERF::build_fft_solvers')
    require(prefix.pos == build_def.end(), 'unsupported FFT source prefix')
    _, raw, definition = signature(text, 'ERF::solve_with_fft')
    require(not text[build_def.end()+len(build)+1:definition.start()].strip(),
            'unsupported code between FFT definitions')
    tail = Source(text[definition.end()+len(raw)+1:])
    literal(tail, '#endif')
    tail.done()
    expected, _, _ = signature(DEFINITION+'}', 'ERF::solve_with_fft')
    actual, _, _ = signature(text, 'ERF::solve_with_fft')
    require(actual == expected, 'FFT definition parameters changed')
    declared, _, declaration = signature(texts[CLASS], 'solve_with_fft', ';')
    require([typ for _, typ in declared] == [typ for _, typ in expected], 'FFT declaration types changed')
    cls = unique(pattern('class ERF : public amrex::AmrCore {'), texts[CLASS], 'ERF class')
    declaration_scope(texts[CLASS], cls.start(), 'ERF_H_')
    class_body = brace_block(texts[CLASS], cls.end()-1)
    matches = [declaration]+[unique(pattern(member), texts[CLASS], 'FFT solver member') for member in MEMBERS]
    for match in matches:
        require(cls.end() <= match.start() < match.end() <= cls.end()+len(class_body), 'FFT member outside ERF')
        fft_member(class_body, match.start()-cls.end())
    reject_uses(texts[CLASS], ['solve_with_fft','m_2D_poisson','m_3D_poisson'],
                [m.span() for m in matches], 'FFT class uses')
    reject_uses(build, ['stretched_dz_h', 'stretched_dz_d', 'solve_with_fft'], [], 'FFT builder spacing uses')
    require('#' not in no_literals(build), 'unsupported FFT builder directive')
    # Only the established GPU-safe OpenMP wrapper is supported in this body.
    omp = r'#ifdef\s+_OPENMP\s*\n\s*#pragma\s+omp\s+parallel\s+if\s*\(\s*Gpu::notInLaunchRegion\s*\(\s*\)\s*\)\s*\n\s*#endif'
    body = re.sub(omp, '', raw)
    require('#' not in no_literals(body), 'unsupported FFT body directive')
    reject_uses(body, ['goto','try','catch','co_await','co_yield','co_return','return','break','continue','throw'],
                [], 'FFT control transfer')
    print('  ok   matching FFT interface and directly declared Poisson/PoissonHybrid members')
    return body


def subcolumn(body):
    locals_ = unique(pattern(LOCALS), body, 'automatic FFT spacing subcolumns')
    direct(body, locals_.start(), 'FFT spacing locals')
    gate = unique(pattern(CONSTANT), body, 'FFT solve dispatch')
    direct(body, gate.start(), 'FFT solve dispatch')
    then_end = gate.end()+len(brace_block(body, gate.end()-1))+1
    tail = Source(body[then_end:])
    literal(tail, 'else {')
    otherwise = then_end+tail.pos-1
    uniform = unique(pattern(UNIFORM), body, 'uniform solve')
    sub = unique(pattern(SUBCOLUMN), body, 'FFT subcolumn sequence')
    for match, scope in ((uniform, gate.end()-1), (sub, otherwise)):
        standalone(body, match.start())
        require(scopes(body, match.start()) == [('{', scope)], 'FFT solve sequence enclosure changed')
    end = otherwise+1+len(brace_block(body, otherwise))+1
    require(locals_.end() < gate.start(), 'FFT subcolumns must precede dispatch')
    bc = unique(pattern(BC), body, 'FFT boundary call')
    geometry = unique(pattern(GEOMETRY), body, 'FFT geometry/boundary setup')
    for match in (bc, geometry):
        direct(body, match.start(), 'FFT post-solve setup')
    require(end < bc.start() < bc.end() < geometry.start(), 'FFT post-solve order changed')
    print('  ok   automatic subcolumns, exact offset mapping, blocking copy and hybrid forwarding')
    return [locals_, gate, uniform, sub, bc, geometry], geometry


def reader(body, geometry):
    loop = unique(pattern(LOOP), body, 'FFT automatic iterator')
    direct(body, loop.start(), 'FFT iterator')
    require(geometry.end() < loop.start(), 'FFT flux loop must follow solve/geometry setup')
    end = loop.end()+len(brace_block(body, loop.end()-1))
    found = [(loop, end)]
    reject_uses(body, ['MFIter'], [loop.span()], 'FFT iterator inventory')
    allowed = [loop.span()]
    for expression in (r'\b(mfi)\s*\.\s*nodaltilebox\s*\(', r'\barray\s*\(\s*(mfi)\s*\)'):
        for use in re.finditer(expression, no_literals(body)):
            enclosing(body, use.start(), found)
            allowed.append(use.span(1))
    reject_uses(body, ['mfi'], allowed, 'FFT iterator uses')
    matches = list(re.finditer(CALLEE+r'\s*\(', no_literals(body)))
    arguments = list(call_arguments(no_literals(body), CALLEE))
    regions = kernel_regions(no_literals(body))
    require(len(matches) == len(arguments) == len(regions) == 4, 'FFT kernel inventory changed')
    gate = unique(pattern(STRETCHED), body, 'FFT stretched flux branch')
    require(scopes(body, gate.start()) == [('{', loop.end()-1)], 'FFT flux gate enclosure changed')
    standalone(body, gate.start())
    then_end = gate.end()+len(brace_block(body, gate.end()-1))+1
    tail = Source(body[then_end:])
    literal(tail, 'else {')
    otherwise = then_end+tail.pos-1
    for index, (match, args, region) in enumerate(zip(matches, arguments, regions)):
        standalone(body, match.start())
        enclosing(body, match.start(), found)
        expected = [('{', loop.end()-1)]
        if index >= 2:
            expected.append(('{', gate.end()-1 if index == 2 else otherwise))
        require(scopes(body, match.start()) == expected, 'FFT launch enclosure changed')
        require(len(args) == 2 and compact(body[args[0][0]:args[0][1]]) == ('xbx','ybx','zbx','zbx')[index]
                and region[3] == match.start(), 'FFT launch arguments changed')
        require(re.match(r'\s*\)\s*;', body[args[-1][1]:]), 'FFT launch must end as a statement')
    bindings = []
    for statement in ('Array4<Real const> const& p_arr = phi.array(mfi);', *(
            'Box const& '+axis+'bx = mfi.nodaltilebox('+str(index)+'); '+
            'Array4<Real> const& f'+axis+'_arr = fluxes['+str(index)+'].array(mfi);'
            for index, axis in enumerate('xyz'))):
        match = unique(pattern(statement), body, 'FFT flux field binding')
        standalone(body, match.start())
        require(scopes(body, match.start()) == [('{', loop.end()-1)], 'FFT field binding enclosure changed')
        bindings.append(match)
    require(bindings[0].end() < bindings[1].start() and
            all(bindings[i+1].end() < matches[i].start() for i in range(3)), 'FFT field binding order changed')
    reject_uses(body, ['xbx','ybx','zbx'], [m.span() for m in bindings]+[a[0] for a in arguments],
                'FFT flux box uses')
    print('  ok   one automatic iterator encloses four direct value-captured flux kernels')
    return [loop, gate, *bindings], gate, regions


def spacing(body, setup, readers, gate, regions):
    alias = unique(pattern(ALIAS), body, 'FFT same-level member spacing pointer')
    standalone(body, alias.start())
    require(scopes(body, alias.start()) == scopes(body, gate.start())+[('{', gate.end()-1)],
            'FFT pointer must remain in stretched flux branch')
    require(alias.end() < regions[2][3], 'FFT pointer must precede stretched kernel')
    for region, code in zip(regions[2:], (VERTICAL, CONSTANT_VERTICAL)):
        source = Source(body[region[0]+1:region[1]])
        literal(source, code)
        source.done()
    spans = [m.span() for m in setup+readers]+[alias.span()]+[(r[0],r[1]+1) for r in regions[2:]]
    # The only additional subdomain uses are these three diagnostics.
    for message in ('Using the 3D FFT solver on domain ', 'Using the hybrid FFT solver on domain ',
                    'FFT solve complete on domain '):
        spans += [m.span() for m in re.finditer(pattern('amrex::Print() << "'+message+
                                      '" << subdomain << std::endl;'), body)]
    reject_uses(body, ['stretched_dz_h_sub','stretched_dz_d_sub','stretched_dz_h','stretched_dz_d',
                      'stretched_dz_d_ptr','lev','isub','subdomain','sub_lo','sub_hi','top_is_dirichlet','bottom_is_dirichlet','projection_vertical_spacing',
                      'dom_hi','bc_type_zhi','dom_lo','bc_type_zlo','m_2D_poisson','m_3D_poisson','solverChoice',
                      'ImposeBCsOnPhi','phi','rhs','fluxes','this'], spans, 'FFT protected uses')
    print('  ok   full-level device spacing read only as scalars in the stretched vertical kernel')


HELPER = """AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
Real projection_vertical_spacing(int k, int lo, int hi, Real const* dz) noexcept {
    if (k == lo) { return dz[k]; }
    if (k == hi+1) { return dz[k-1]; }
    return Real(0.5)*(dz[k]+dz[k-1]);
}"""


def spacing_helper(text):
    match = unique(pattern(HELPER), text, 'bounded vertical-spacing helper')
    declaration_scope(text, match.start(), 'ERF_SOLVER_UTILS_H_')
    reject_uses(text, ['projection_vertical_spacing'], [match.span()], 'vertical-spacing helper uses')
    print('  ok   first/last faces read one stored cell, interior faces read their adjacent cells')


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        body = interface(texts)
        spacing_helper(texts[UTILS])
        setup, geometry = subcolumn(body)
        readers, gate, regions = reader(body, geometry)
        spacing(body, setup, readers, gate, regions)
    except (OSError, ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('5 FFT spacing source contracts passed; temporary/member completion relies on inspected AMReX and caller state.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
