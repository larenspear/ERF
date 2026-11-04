#!/usr/bin/env python3
"""Guard selected FFT construction entry points and restart rebuild order.

Three finite source contracts cover init_stuff's interface and FFT gate; the
three level routines' same-level subdomain/grid setup before init_stuff; and
InitData's pre/post sequence, scratch/restart selection, checkpoint invocation,
and post-restart rebuild over all retained levels. Complete FFT gates/loops are
consumed. Other statements are not interpreted. Explicit protected call uses
are inventoried in the selected files/bodies, including disabled branches.

Both NETCDF configurations are selected because its branches cross C++ scopes.
Other preprocessor branches must preserve surrounding delimiter scopes. Protected
sites have exact directive stacks; other supported conditions are not evaluated.
Both FFT configurations are covered without treating the two arms of a
conditional as executing together. This is not a C++ compiler.

Assume successful ordinary C++/AMReX calls and intended resolution, valid levels
and storage, stable selectors/level/grid/BC/solver state through unmodeled code,
and the separate checkpoint-spacing and FFT builder contracts. No external
macro rewriting, undefined behavior, nonlocal transfer or concurrent mutation.
The AmrCore dispatch into level routines and make_subdomains implementation are
not checked. Nor are checkpoint contents/mesh classification, subsequent state
freshness at projection, alias ownership, GPU completion or complete lifetime.
Only the Python standard library is required.
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (CLASS, brace_block, direct, no_literals,
                                       reject_uses, scopes, signature)
from check_stretched_dz_consumers import standalone
from check_stretched_dz_fft_bindings import pattern
from check_stretched_dz_wrf_callers import pp_stack
from momentum_geometry import unique

ARRAYS = 'Source/ERF_MakeNewArrays.cpp'
LEVELS = 'Source/ERF_MakeNewLevel.cpp'
INIT = 'Source/ERF.cpp'
FILES = (CLASS, ARRAYS, LEVELS, INIT)
PARAMS = '''int lev, const BoxArray& ba, const DistributionMapping& dm,
    Vector<MultiFab>& lev_new, Vector<MultiFab>& lev_old,
    MultiFab& tmp_base_state, std::unique_ptr<MultiFab>& tmp_zphys_nd'''
GATE = '''if (((solverChoice.anelastic[lev] == 1) ||
    (solverChoice.project_initial_velocity[lev] == 1)) &&
    ((solverChoice.mesh_type == MeshType::ConstantDz) ||
     (solverChoice.mesh_type == MeshType::StretchedDz))) {
    build_fft_solvers(lev);
}'''
BUILD = '#ifdef ERF_USE_FFT\n'+GATE+'\n#endif'
LOOP = 'for (int lev = 0; lev <= finest_level; lev++) {'
REBUILD = '#ifdef ERF_USE_FFT\n'+LOOP+GATE+'}\n#endif'
SUBDOMAINS = 'make_subdomains(ba.simplified_list(), subdomains[lev]);'
SETTERS = ('SetBoxArray(lev, ba);', 'SetDistributionMap(lev, dm);')
CALLS = {
    'MakeNewLevelFromScratch': 'init_stuff(lev, ba, dm, lev_new, lev_old, base_state[lev], z_phys_nd[lev]);',
    'MakeNewLevelFromCoarse': 'init_stuff(lev, ba, dm, vars_new[lev], vars_old[lev], base_state[lev], z_phys_nd[lev]);',
    'RemakeLevel': 'init_stuff(lev, ba, dm, temp_lev_new, temp_lev_old, temp_base_state, temp_zphys_nd);',
}
RESTART = 'if (!restart_chkfile.empty()) { restart(); }'
START = 'if (restart_chkfile.empty()) { InitFromScratch(zero); } else { init_bcs(); }'
FEATURES = {'ERF_USE_FFT', 'ERF_USE_NETCDF', 'ERF_USE_PARTICLES', 'ERF_USE_WINDFARM',
            'ERF_USE_WW3_COUPLING', 'ERF_IMPLICIT_W', '_OPENMP'}
CALLEES = ('build_fft_solvers', 'init_stuff', 'make_subdomains', 'SetBoxArray', 'SetDistributionMap')


def netcdf_view(body, enabled):
    """Select NETCDF only, preserving offsets and all other feature branches."""
    output, stack = [], []
    for line in body.splitlines(keepends=True):
        match = re.fullmatch(r'[ \t]*#\s*(\w+)\s*([^\n]*?)\s*', no_literals(line))
        active = all(state is not False for state in stack)
        keep = active
        if match:
            kind, arg = match[1], match[2]
            if kind in ('if', 'ifdef', 'ifndef'):
                known = kind == 'ifdef' and arg == 'ERF_USE_NETCDF'
                stack.append(enabled if known else None)
                keep = active and not known
            elif kind == 'else':
                require(stack and not arg, 'invalid NETCDF else')
                if stack[-1] is not None:
                    stack[-1] = not stack[-1]
                    keep = False
            elif kind == 'endif':
                require(stack and not arg, 'invalid NETCDF endif')
                keep = active and stack[-1] is None
                stack.pop()
        output.append(line if keep else re.sub(r'[^\n]', ' ', line))
    require(not stack, 'unclosed NETCDF selection')
    return ''.join(output)


def outline(body):
    """Mask directives, requiring every feature arm to preserve C++ scopes."""
    syntax = no_literals(body)
    output = list(body)
    cpp, pp = [], []
    previous = 0

    def delimiters(end):
        nonlocal previous
        for match in re.finditer(r'[(){}\[\]]', syntax[previous:end]):
            char, pos = match[0], previous+match.start()
            if char in '([{':
                cpp.append((char, pos))
            else:
                require(cpp and cpp[-1][0] == {')':'(', ']':'[', '}':'{'}[char],
                        'unbalanced caller delimiter')
                cpp.pop()
        previous = end

    for match in re.finditer(r'(?m)^[ \t]*#[^\n]*', syntax):
        delimiters(match.start())
        # Balanced braces alone are insufficient: a feature arm ending in
        # "if (ready)" could condition the next protected statement in only
        # one build. Require statement boundaries at every directive too.
        standalone(''.join(output), match.start())
        directive = re.fullmatch(r'[ \t]*#\s*(\w+)\s*(.*?)\s*', match[0])
        require(directive is not None, 'unsupported caller directive syntax')
        kind, arg = directive[1], directive[2]
        if ((kind == 'ifdef' and arg in FEATURES) or
            (kind == 'if' and arg in ('0', 'USE_FC_FACTORY', 'defined(ERF_USE_WINDFARM)'))):
            pp.append((cpp.copy(), False, kind, arg))
        elif kind == 'else':
            require(pp and not arg and not pp[-1][1] and cpp == pp[-1][0],
                    'caller preprocessor arm changes surrounding C++ scope')
            pp[-1] = (pp[-1][0], True, pp[-1][2], pp[-1][3])
        elif kind == 'endif':
            require(pp and not arg and cpp == pp[-1][0],
                    'caller preprocessor arm changes surrounding C++ scope')
            pp.pop()
        elif kind == 'pragma':
            require(arg == 'omp parallel if (amrex::Gpu::notInLaunchRegion())' and pp and
                    pp[-1][2:] == ('ifdef', '_OPENMP') and not pp[-1][1],
                    'unsupported caller pragma')
        else:
            raise ValueError('unsupported caller directive: '+kind+' '+arg)
        output[match.start():match.end()] = ' '*len(match[0])
        previous = match.end()
    delimiters(len(body))
    require(not cpp and not pp, 'unclosed caller scope')
    result = ''.join(output)
    require('#' not in no_literals(result), 'unsupported inline caller directive')
    return result


def transfers(body):
    reject_uses(body, ['goto', 'throw', 'try', 'catch', 'co_await', 'co_yield', 'co_return',
                       'longjmp', 'setjmp'], [], 'caller control transfer')
    # Existing noncapturing local lambdas may return from their own bodies.
    lambdas = [(m.end(), m.end()+len(brace_block(body, m.end()-1))) for m in re.finditer(
        r'\bauto\s+\w+\s*=\s*\[\s*\]\s*\([^{};]*\)(?:\s*->\s*[^{};]+)?\s*\{', no_literals(body))]
    reject_uses(body, ['return'], lambdas, 'caller early return')
    for match in re.finditer(r'\b(break|continue)\b', no_literals(body)):
        enclosing = scopes(body, match.start())
        loops = 'for|while|switch' if match[1] == 'break' else 'for|while'
        require(any(char == '{' and re.search(r'\b(?:'+loops+r')\s*\([^{}]*\)\s*$', no_literals(body[:pos]))
                    for char, pos in enclosing), 'caller loop transfer outside a local loop')


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
        require(not pp_stack(text), path+': unclosed preprocessor scope')
        texts[path] = text
    return texts


def function(text, name, parameters):
    _, body, definition = signature(text, 'ERF::'+name)
    require(re.fullmatch(pattern(parameters), definition[1].strip()), name+': signature changed')
    direct(text, definition.start(), name+' definition')
    require(not pp_stack(text[:definition.start()]), name+': conditional definition')
    plain = tuple(outline(netcdf_view(body, enabled)) for enabled in (False, True))
    for view in plain:
        transfers(view)
    return body, plain, (definition.start(), definition.end()+len(body)+1)


def site(raw, plain, code, stack=(), pp=()):
    match = unique(pattern(code), raw, code)
    for view in plain:
        require(scopes(view, match.start()) == list(stack), code+': enclosing C++ scope changed')
        standalone(view, match.start())
    require(pp_stack(raw[:match.start()]) == list(pp), code+': enclosing preprocessor scope changed')
    return match


def initialization(texts):
    header = texts[CLASS]
    owner = unique(pattern('class ERF : public amrex::AmrCore {'), header, 'ERF class')
    _, _, decl = signature(header, 'init_stuff', ';')
    qualified = re.sub(r'\b(BoxArray|DistributionMapping|Vector|MultiFab)\b', r'amrex::\1', PARAMS)
    require(re.fullmatch(pattern(qualified), decl[1].strip()), 'init_stuff declaration changed')
    require(scopes(header, decl.start()) == [('{', owner.end()-1)] and
            pp_stack(header[:decl.start()]) == [('ifndef', 'ERF_H_')], 'init_stuff member scope changed')
    reject_uses(header, ['init_stuff'], [decl.span()], 'init_stuff declaration inventory')
    raw, plain, span = function(texts[ARRAYS], 'init_stuff', PARAMS)
    build = site(raw, plain, BUILD)
    reject_uses(raw, [*CALLEES, 'm_2D_poisson', 'm_3D_poisson'], [build.span()], 'init_stuff FFT uses')
    reject_uses(texts[ARRAYS], CALLEES, [span], 'array file caller inventory')
    print('  ok   init_stuff interface and same-level FFT build under exact mesh/projection gates')


def levels(text):
    spans = []
    for name, call in CALLS.items():
        parameters = 'int lev, Real time, const BoxArray& ba, const DistributionMapping& dm'
        if name == 'MakeNewLevelFromScratch':
            parameters = parameters.replace('ba,', 'ba_in,').replace('dm', 'dm_in')
        raw, plain, span = function(text, name, parameters)
        spans.append(span)
        setup = 'subdomains[lev].clear();' if name == 'RemakeLevel' else 'subdomains.resize(lev+1);'
        initialize = site(raw, plain, call)
        capacity = site(raw, plain, setup)
        subdomains = site(raw, plain, SUBDOMAINS)
        setters = [site(raw, plain, code) for code in SETTERS]
        require(capacity.end() <= subdomains.start() < initialize.start() and
                all(m.end() <= initialize.start() for m in setters), name+': setup must precede init_stuff')
        if name == 'MakeNewLevelFromScratch':
            aliases = site(raw, plain, 'auto& lev_new = vars_new[lev]; auto& lev_old = vars_old[lev];')
            require(aliases.end() <= initialize.start(), 'scratch aliases must precede init_stuff')
        reject_uses(raw, [*CALLEES, 'subdomains'],
                    [m.span() for m in (initialize, capacity, subdomains, *setters)], name+' setup inventory')
    reject_uses(text, CALLEES, spans, 'level file caller inventory')
    print('  ok   scratch/coarse/remade levels register grids and construct same-level subdomains before init_stuff')


def restart(text):
    names = ('InitData', 'InitData_pre', 'InitData_post', 'restart')
    functions = {name:function(text, name, '') for name in names}
    raw, plain, _ = functions['InitData']
    require(re.fullmatch(pattern('''BL_PROFILE_VAR("ERF::InitData()", InitData);
        InitData_pre(); InitData_post();
        if (!m_seb_surface_layer_checked) { check_seb_surface_layer(); }
        BL_PROFILE_VAR_STOP(InitData);'''), raw.strip()), 'InitData sequence changed')
    raw, plain, _ = functions['InitData_pre']
    start = site(raw, plain, START)
    reject_uses(raw, ['InitFromScratch', 'ReadCheckpointFile', *names, *CALLEES],
                [start.span()], 'pre-initialization call inventory')
    raw, plain, _ = functions['restart']
    read = site(raw, plain, 'ReadCheckpointFile();')
    reject_uses(raw, ['ReadCheckpointFile', 'InitFromScratch', *names, *CALLEES],
                [read.span()], 'restart call inventory')
    raw, plain, _ = functions['InitData_post']
    restore = site(raw, plain, RESTART)
    rebuild = unique(pattern(REBUILD), raw, 'post-restart FFT rebuild')
    enclosing = scopes(plain[0], rebuild.start())
    require(len(enclosing) == 1 and enclosing[0][0] == '{', 'rebuild requires one restart branch')
    opening = enclosing[0][1]
    gate = unique(pattern('if (!restart_chkfile.empty()) {')+r'\s*$', raw[:opening+1], 'rebuild restart gate')
    for view in plain:
        require(scopes(view, rebuild.start()) == enclosing and not scopes(view, gate.start()),
                'restart rebuild branch must be direct in both NETCDF modes')
        standalone(view, gate.start())
        standalone(view, rebuild.start())
    require(not pp_stack(raw[:gate.start()]), 'restart rebuild branch must be unconditional')
    require(not pp_stack(raw[:rebuild.start()]), 'rebuild must depend only on FFT feature')
    require(restore.end() <= gate.start(), 'checkpoint restoration must precede FFT rebuild branch')
    end = opening+1+len(brace_block(raw, opening))
    require(not raw[rebuild.end():end].strip(), 'FFT rebuild must finish the restart branch')
    reject_uses(raw, ['build_fft_solvers', 'ReadCheckpointFile', 'InitFromScratch', *names,
                      'init_stuff', 'make_subdomains', 'm_2D_poisson', 'm_3D_poisson'],
                [restore.span(), rebuild.span()], 'post-initialization call inventory')
    reject_uses(text, [*CALLEES, 'InitFromScratch', 'ReadCheckpointFile', *names],
                [entry[2] for entry in functions.values()], 'initialization file call inventory')
    print('  ok   pre/post selection, direct checkpoint call and all-level FFT rebuild after restart returns')


def check(texts):
    initialization(texts)
    levels(texts[LEVELS])
    restart(texts[INIT])


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        check(load(root))
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('3 FFT rebuild source contracts passed; state freshness and GPU lifetime remain premises.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
