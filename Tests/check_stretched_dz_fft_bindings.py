#!/usr/bin/env python3
"""Guard FFT solver construction and projection call bindings.

Four finite source contracts cover the builder interface/reset/mesh planner,
its complete per-subdomain Geometry and slot construction, the projection's
FFT/MLMG planner, and two direct FFT calls under their mesh/rectangle gates.
Both FFT preprocessor configurations are checked. Supported builder statements
are consumed completely; projection checks cover only the named regions/uses.

With a prior successful build for the same level, unchanged subdomains/mesh/
Geometry/BCs/solver slots, intended AMReX meanings and valid nonempty cell boxes,
the construction and solve expressions select the same slot and integer domain.
The separately inspected pinned Geometry::define and FFT shift_geom preserve
that domain's lengths while translating its small end to zero. This connects
the previous subcolumn indexing contract to the hybrid solver's local indices.

This does not establish builder invocation/order, state preservation by the
rest of projection or external code/aliases/globals, subdomain partition or
MultiFab alias correctness, numerical accuracy, physical RealBox origins,
get_fft_bc semantics, bounds/positivity, GPU completion or whole-program lifetime.
Assume normal C++ resolution/execution, no macro rewriting from headers, no
undefined behavior/concurrent mutation/nonlocal transfer, and the previous FFT
spacing contract. Dependency semantics are inspected, not parsed by this test.
Only the Python standard library is required.
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import compact, require
from check_stretched_dz_bindings import (CLASS, brace_block, direct, no_literals,
    reject_uses, scope_header, scopes, signature)
from check_stretched_dz_consumers import declaration_scope, standalone
from check_stretched_dz_fft import SOURCE, fft_member
from check_stretched_dz_mfiter import CONTROLS
from check_stretched_dz_projection_divergence import PROJECTION, configured
from momentum_geometry import unique

FILES = (SOURCE, CLASS, PROJECTION)
BUILD = 'void ERF::build_fft_solvers(int lev) {'
RESET = ''.join('''if (MEMBER.size() > lev) {
    for (int n = 0; n < MEMBER[lev].size(); n++) { MEMBER[lev][n].reset(); }
}'''.replace('MEMBER',member) for member in ('m_3D_poisson','m_2D_poisson'))
MESH = '''if ((solverChoice.mesh_type == MeshType::ConstantDz) ||
    (solverChoice.mesh_type == MeshType::StretchedDz)) {'''
LOOP = 'for (int isub = 0; isub < subdomains[lev].size(); ++isub) {'
REGION = 'Box my_region(subdomains[lev][isub].minimalBox());'
RECTANGLES = 'bool all_boxes_ok = true; '+LOOP+REGION+'''
    bool boxes_make_rectangle = (my_region.numPts() == subdomains[lev][isub].numPts());
    if (!boxes_make_rectangle) { all_boxes_ok = false; }
} if (all_boxes_ok) { will_solve_with_mlmg = false; }'''
BUILD_PLAN = 'bool will_solve_with_mlmg = (solverChoice.mesh_type == MeshType::ConstantDz); '+RECTANGLES
BUILD_TOPOLOGY = '''if (has_periodic_subdomain_connections(geom[lev], subdomains[lev])) {
    amrex::Abort("Projection subdomains connected across a periodic boundary are not supported");
}'''
PROJECT_TOPOLOGY = BUILD_TOPOLOGY
GEOMETRY = '''Box subdomain(subdomains[lev][isub].minimalBox());
    auto const dom_lo = lbound(Geom(lev).Domain());
    auto const dom_hi = ubound(Geom(lev).Domain());
    auto const sub_lo = lbound(subdomain);
    auto const sub_hi = ubound(subdomain);
    auto dx = geom[lev].CellSizeArray();
    auto bc_fft = get_fft_bc(geom[lev],domain_bc_type,subdomain,solverChoice.use_real_bcs);
    Geometry my_geom;
    Array<int,AMREX_SPACEDIM> is_per; is_per[0] = 0; is_per[1] = 0; is_per[2] = 0;
    if (geom[lev].isPeriodic(0) && sub_lo.x == dom_lo.x && sub_hi.x == dom_hi.x) { is_per[0] = 1; }
    if (geom[lev].isPeriodic(1) && sub_lo.y == dom_lo.y && sub_hi.y == dom_hi.y) { is_per[1] = 1; }
    int coord_sys = 0;
    if (subdomain == Geom(lev).Domain()) {
        my_geom.define(Geom(lev).Domain(), Geom(lev).ProbDomain(), coord_sys, is_per);
    } else {
        RealBox rb(sub_lo.x*dx[0], sub_lo.y*dx[1], sub_lo.z*dx[2],
                   (sub_hi.x+1)*dx[0], (sub_hi.y+1)*dx[1], (sub_hi.z+1)*dx[2]);
        my_geom.define(subdomain, rb, coord_sys, is_per);
    }
    bool boxes_make_rectangle = (subdomain.numPts() == subdomains[lev][isub].numPts());
    if (!boxes_make_rectangle && !will_solve_with_mlmg) {
        amrex::Abort("FFT won't work unless the union of boxes is rectangular");
    }'''


def slot(member, kind, message):
    return '''if (mg_verbose > 0) {
        amrex::Print() << "MESSAGE" << my_geom.Domain() << std::endl;
    }
    if (MEMBER.size() <= lev) { MEMBER.resize(lev+1); }
    if (MEMBER[lev].size() <= isub) { MEMBER[lev].resize(isub+1); }
    MEMBER[lev][isub] = std::make_unique<FFT::KIND<MultiFab>>(my_geom,bc_fft);
    '''.replace('MEMBER',member).replace('KIND',kind).replace('MESSAGE',message)


SLOTS = '''if (solverChoice.mesh_type == MeshType::ConstantDz && !will_solve_with_mlmg) {
    '''+slot('m_3D_poisson','Poisson','Building the 3D FFT solver to be used on domain ')+'''
    } else if (solverChoice.mesh_type == MeshType::StretchedDz) {
    '''+slot('m_2D_poisson','PoissonHybrid','Building the hybrid FFT solver to be used on domain ')+'}'
PLAN = '''bool will_solve_with_mlmg = false;
    if (solverChoice.mesh_type == MeshType::ConstantDz) {
        will_solve_with_mlmg = true;
#ifdef ERF_USE_FFT
        if (use_fft) { '''+RECTANGLES+''' }
#else
        if (use_fft) {
            amrex::Warning("You set use_fft=true but didn't build with USE_FFT = TRUE; defaulting to MLMG");
        }
#endif
    }'''
CALL = 'solve_with_fft(lev, isub, my_region, rhs_sub[0], phi_sub[0], fluxes_sub[0]);'
DISPATCH = '''
#ifdef ERF_USE_FFT
    '''+REGION+'''
#endif
    if (solverChoice.mesh_type == MeshType::ConstantDz) {
        if (will_solve_with_mlmg) {
            solve_with_mlmg(lev, rhs_sub, phi_sub, fluxes_sub, geom[lev], ref_ratio, domain_bc_type,
                mg_verbose, solverChoice.poisson_reltol, solverChoice.poisson_abstol);
        } else {
#ifdef ERF_USE_FFT
            '''+CALL+'''
#endif
        }
    } else if (solverChoice.mesh_type == MeshType::StretchedDz) {
#ifndef ERF_USE_FFT
        amrex::Abort("Rebuild with USE_FFT = TRUE so you can use the FFT solver");
#else
        bool boxes_make_rectangle = (my_region.numPts() == subdomains[lev][isub].numPts());
        if (!boxes_make_rectangle) {
            amrex::Abort("FFT won't work unless the union of boxes is rectangular");
        } else {
            if (!use_fft) {
                amrex::Warning("Using FFT even though you didn't set use_fft to true; it's the best choice");
            }
            '''+CALL+'''
        }
#endif
    }'''
VARIABLE = '''else if (solverChoice.mesh_type == MeshType::VariableDz) {
#ifdef ERF_USE_FFT
    bool boxes_make_rectangle = (my_region.numPts() == subdomains[lev][isub].numPts());
    if (!boxes_make_rectangle) {
        amrex::Abort("FFT preconditioner for GMRES won't work unless the union of boxes is rectangular");
    } else {
        solve_with_gmres(lev, my_region, rhs_sub[0], phi_sub[0], fluxes_sub[0], ax_sub, ay_sub, az_sub, dJ_sub, znd_sub);
    }
#else
    amrex::Abort("Rebuild with USE_FFT = TRUE so you can use the FFT preconditioner for GMRES");
#endif
'''
NON_EB = 'if (solverChoice.terrain_type != TerrainType::EB) {'



def pattern(code):
    # Keep C++ multi-character operators atomic: + +i must not match ++i.
    tokens = re.findall(r'::|->|<=|>=|==|!=|\+\+|--|&&|\|\||<<|>>|\w+|[^\s]', code)
    return r'(?<![\w:.>])'+r'\s*'.join(re.escape(token)+
        (r'\b' if re.fullmatch(r'\w+',token) else '') for token in tokens)


def literal(source, code):
    source.take(pattern(code))


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        text = strip_comments(raw)
        for macro in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(path == CLASS and macro[1] == 'define' and macro[2].strip() == 'ERF_H_',
                    path+': unsupported macro')
        reject_uses(text, CONTROLS, [], path+' stream/sync controls')
        texts[path] = text
    return texts


def builder(texts):
    text = texts[SOURCE]
    parameters, body, definition = signature(text, 'ERF::build_fft_solvers')
    require(parameters == [('lev','int')], 'FFT builder parameters changed')
    prefix = Source(text[:definition.end()])
    literal(prefix, '#include "ERF.H" #include "ERF_SolverUtils.H" using namespace amrex; #ifdef ERF_USE_FFT '+BUILD)
    prefix.done()
    _, solve, solve_def = signature(text, 'ERF::solve_with_fft')
    require(not text[definition.end()+len(body)+1:solve_def.start()].strip(), 'unsupported code between FFT definitions')
    tail = Source(text[solve_def.end()+len(solve)+1:])
    literal(tail, '#endif')
    tail.done()
    declared, _, declaration = signature(texts[CLASS], 'build_fft_solvers', ';')
    require([typ for _,typ in declared] == ['int'], 'FFT builder declaration type changed')
    cls = unique(pattern('class ERF : public amrex::AmrCore {'), texts[CLASS], 'ERF class')
    declaration_scope(texts[CLASS], cls.start(), 'ERF_H_')
    class_body = brace_block(texts[CLASS], cls.end()-1)
    require(cls.end() <= declaration.start() < declaration.end() <= cls.end()+len(class_body),
            'FFT builder declaration outside ERF')
    fft_member(class_body, declaration.start()-cls.end())
    reject_uses(texts[CLASS], ['build_fft_solvers'], [declaration.span()], 'FFT builder declarations')
    source = Source(body)
    literal(source, 'BL_PROFILE("ERF::build_fft_solvers()"); '+RESET+MESH+BUILD_PLAN+BUILD_TOPOLOGY)
    print('  ok   direct builder interface, same-level reset and supported-mesh planner')
    literal(source, LOOP+GEOMETRY+SLOTS+'} }')
    source.done()
    print('  ok   complete per-subdomain Geometry and same-slot Poisson/PoissonHybrid construction')


def callers(text):
    parameters, raw, definition = signature(text, 'ERF::project_momenta')
    require(parameters == [('lev','int'),('l_time','double'),('l_dt_d','double'),('mom_mf','Vector<MultiFab>&')],
            'projection parameters changed')
    direct(text, definition.start(), 'projection definition')
    reject_uses(text, ['solve_with_fft'], [(definition.end(),definition.end()+len(raw))], 'FFT caller inventory')
    require(len(list(re.finditer(r'\bsolve_with_fft\b', no_literals(raw)))) == 2, 'raw FFT call inventory changed')
    reject_uses(raw, ['build_fft_solvers','m_2D_poisson','m_3D_poisson','stretched_dz_h','stretched_dz_d'],
                [], 'projection solver/spacing aliases')
    reject_uses(raw, ['goto','try','catch','co_await','co_yield','co_return'], [], 'projection control transfer')
    for fft in (False,True):
        body = configured(raw, fft)
        plan = unique(pattern(configured(PLAN,fft)), body, 'projection FFT/MLMG planner')
        direct(body, plan.start(), 'projection FFT/MLMG planner')
        gate = unique(pattern(NON_EB), body, 'non-EB solver gate')
        standalone(body, gate.start())
        stack = scopes(body, gate.start())
        require(len(stack) == 1 and stack[0][0] == '{' and
                scope_header(body,stack[0][1]) == compact(LOOP[:-1]),
                'FFT dispatch must be directly in the subdomain loop')
        loop_open = stack[0][1]
        loop = unique(pattern(LOOP), body[plan.end():loop_open+1], 'projection subdomain loop')
        loop_start = plan.end()+loop.start()
        direct(body, loop_start, 'projection subdomain loop')
        require(plan.end() < loop_start < gate.start(), 'projection planning must precede the loop')
        topology = Source(body[plan.end():loop_start])
        literal(topology, configured(PROJECT_TOPOLOGY,fft))
        topology.done()
        topology_span = (plan.end(), loop_start)
        branch = brace_block(body,gate.end()-1)
        source = Source(branch)
        literal(source,configured(DISPATCH,fft))
        dispatch_end = source.pos
        # Preserve the only other uses of this local region in the adjacent
        # variable-grid branch; its later kernels remain outside this contract.
        literal(source,configured(VARIABLE,fft))
        reject_uses(branch, ['my_region'], [(0,source.pos)], 'projection solver-region uses')
        calls = list(re.finditer(pattern(CALL), no_literals(body)))
        require(len(calls) == (2 if fft else 0), 'configured FFT call inventory changed')
        dispatch_span = (gate.end(),gate.end()+dispatch_end)
        reject_uses(body, ['solve_with_fft','will_solve_with_mlmg'], [plan.span(),topology_span,dispatch_span],
                    'projection planner/FFT uses')
        reject_uses(body, ['has_periodic_subdomain_connections'], [topology_span],
                    'projection topology uses')
        for call in calls:
            require(dispatch_span[0] <= call.start() < call.end() <= dispatch_span[1],
                    'FFT call outside checked dispatch')
        # The local region can be changed only in this checked prefix. Other
        # same-named locals in unrelated projection branches are not this object.
    print('  ok   complete projection FFT/MLMG planner in both FFT configurations')
    print('  ok   two direct FFT calls retain level/subdomain/field bindings and mesh/rectangle gates')


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        builder(texts)
        callers(texts[PROJECTION])
    except (OSError,ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 FFT construction/caller source contracts passed; matching state and AMReX semantics remain premises.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
