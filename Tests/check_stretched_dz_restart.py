#!/usr/bin/env python3
"""Guard checkpoint spacing refresh before native restart level construction.

Consume the entire nominal-level reload block, with NETCDF disabled and enabled:
read each checkpoint level, skip levels outside retained storage, reject a size
mismatch, replace nominal nodes, then immediately refresh the same level. The
optional z_top read/fallback is checked too. Require this unconditional block
before the direct level-construction loop, and reject any other explicit use of
nominal/spacing storage or the refresh/construction callees in the reader.

These three finite source contracts add no arithmetic theorem. Preconditions:
valid checkpoint counts/data and initialized, consistent member vectors; ordinary
C++/AMReX/stream semantics and resolution to the intended types and callees;
Abort terminates, I/O succeeds, and update_stretched_dz obeys its separately
checked arithmetic/copy contract. Other calls preserve storage through implicit
this, globals and aliases. No external macro rewriting, concurrency or undefined
behavior. The missing-file path keeps initialized storage; checkpoint levels
beyond retained storage are read and skipped. This does not validate malformed
checkpoints, consistency of levels absent from the checkpoint, WRF rebuilding,
the caller chain, projection/EB consumers, or GPU completion before resize.
The source reader is not a C++ compiler or a whole-program alias proof.

usage: check_stretched_dz_restart.py [ERF_ROOT]
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import slice_scope, strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (brace_block, direct, no_literals,
                                       reject_uses, scopes, signature)
from check_stretched_dz_consumers import pattern, standalone
from momentum_geometry import unique

CHECKPOINT = 'Source/IO/ERF_Checkpoint.cpp'
FILES = (CHECKPOINT,)
REFRESH = 'update_stretched_dz(lev, zlevels_stag, stretched_dz_h, stretched_dz_d);'
BUILD = 'MakeNewLevelFromScratch(lev, static_cast<Real>(t_new[lev]), ba_target, dm_target);'
LOOP = 'for (int lev = 0; lev <= finest_level; ++lev) {'
STRING = r'"(?:\\.|[^"\\])*"'
PROTECTED = ('zlevels_stag', 'stretched_dz_h', 'stretched_dz_d', 'update_stretched_dz',
             'MakeNewLevelFromScratch', 'ZLevelsFile', 'ZLevelsfileCharPtr',
             'ZLevelsfileCharPtrString', 'isz', 'nlevs_in_chk', 'nz_stag', 'zlevels_from_chk')


def take(source, statement):
    return source.take(pattern(statement))


def prints(source, operands):
    """Allow diagnostic strings and enumerated reads, never arbitrary calls."""
    operand = STRING + '|' + '|'.join('(?:'+pattern(value)+')' for value in operands)
    count = 0
    while re.match(r'\s*Print\s*\(', source.text[source.pos:]):
        source.take(r'Print\s*\(\s*\)\s*(?:<<\s*(?:'+operand+r')\s*)+;')
        count += 1
    return count


def select_netcdf(text, enabled):
    """The reload block has only nonnested NETCDF guards; reject any drift."""
    output, inside = [], False
    for line in text.splitlines(keepends=True):
        if '#' not in no_literals(line):
            output.append(line if not inside or enabled else '\n')
            continue
        directive = re.fullmatch(r'\s*#\s*(ifdef\s+ERF_USE_NETCDF|endif)\s*', line)
        require(directive is not None, 'unsupported reload directive')
        if directive[1] == 'endif':
            require(inside, 'unmatched reload endif')
            inside = False
        else:
            require(not inside, 'nested reload directive')
            inside = True
    require(not inside, 'unclosed reload directive')
    return ''.join(output)


def reload_contract(block, netcdf):
    source = Source(select_netcdf(block, netcdf))
    take(source, '{ std::string ZLevelsFile(restart_chkfile +')
    source.take(re.escape('"/zlevels"'))
    take(source, ''');
        if (amrex::FileExists(ZLevelsFile)) {
            Vector<char> ZLevelsfileCharPtr;
            ParallelDescriptor::ReadAndBcastFile(ZLevelsFile, ZLevelsfileCharPtr);
            std::string ZLevelsfileCharPtrString(ZLevelsfileCharPtr.dataPtr());
            std::istringstream isz(ZLevelsfileCharPtrString, std::istringstream::in);
            int nlevs_in_chk = 0;
            isz >> nlevs_in_chk;
            for (int lev = 0; lev < nlevs_in_chk; ++lev) {
                int nz_stag = 0;
                isz >> nz_stag;
                Vector<Real> zlevels_from_chk(nz_stag);
                for (int k = 0; k < nz_stag; ++k) {
                    isz >> zlevels_from_chk[k];
                }
                if (lev >= static_cast<int>(zlevels_stag.size())) { continue; }
                if (static_cast<int>(zlevels_stag[lev].size()) != nz_stag) {''')
    prints(source, ('nz_stag', 'lev', 'zlevels_stag[lev].size()', 'std::endl'))
    take(source, 'Abort(')
    source.take(STRING)
    take(source, '); } zlevels_stag[lev] = zlevels_from_chk;')
    take(source, REFRESH)
    take(source, '}')
    if netcdf:
        take(source, 'isz >> z_top;')
    take(source, '}')
    if netcdf:
        take(source, '''if (z_top == zero && !zlevels_stag.empty() && !zlevels_stag[0].empty()) {
            z_top = zlevels_stag[0].back();
            if (solverChoice.init_type == InitType::WRFInput && solverChoice.avg_grid_faces_to_nodes) {''')
        # C++ permits adjacent string literals in this warning.
        operand = r'(?:'+STRING+r'\s*)+|'+pattern('z_top')
        source.take(r'Print\s*\(\s*\)\s*(?:<<\s*(?:'+operand+r')\s*)+;')
        take(source, '} }')
    take(source, '}')
    source.done()


def load(root):
    raw = (root/CHECKPOINT).read_text()
    require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
            'unsupported checkpoint lexical form')
    text = strip_comments(raw)
    # Only existing declaration includes, feature guards and disabled legacy
    # sections are supported. Never silently discard a new macro or pragma.
    for directive in re.finditer(r'(?m)^[ \t]*#[^\n]*', no_literals(text)):
        require(re.fullmatch(r'#\s*(?:include\b[^\n]*|endif\s*|if\s+0\s*|'
                             r'ifdef\s+ERF_USE_(?:NETCDF|PARTICLES)\s*|'
                             r'if\s+defined\s*\(\s*ERF_USE_WINDFARM\s*\)\s*)',
                             directive[0].lstrip()), 'unsupported checkpoint directive')
    _, body, definition = signature(text, 'ERF::ReadCheckpointFile')
    require(not definition[1].strip(), 'checkpoint reader signature changed')
    require(not scopes(text, definition.start()), 'checkpoint definition is nested')
    slice_scope(text[:definition.start()])
    require(not re.search(r'(?m)^\s*#\s*include\b', no_literals(body)),
            'unmodeled checkpoint runtime include')
    require(not re.search(r'\b(?:goto|try|catch|co_await|co_yield|co_return)\b', no_literals(body)),
            'unsupported checkpoint control transfer')
    return body


def check(body):
    anchor = unique(pattern('std::string ZLevelsFile('), body, 'zlevels file declaration')
    stack = scopes(body, anchor.start())
    require(len(stack) == 1 and stack[0][0] == '{', 'reload block must be at reader scope')
    start = stack[0][1]
    direct(body, start, 'reload block')
    block = '{'+brace_block(body, start)+'}'
    end = start+len(block)
    for netcdf in (False, True):
        reload_contract(block, netcdf)
    print('  ok   complete retained-level reload/refresh contract with NETCDF off and on')

    build = unique(pattern(BUILD), body, 'level construction call')
    stack = scopes(body, build.start())
    require(len(stack) == 1 and stack[0][0] == '{', 'construction must be in the direct level loop')
    loop = unique(pattern(LOOP)+r'\s*$', body[:stack[0][1]+1], 'construction loop')
    direct(body, loop.start(), 'construction loop')
    standalone(body, build.start())
    require(end < loop.start(), 'spacing reload must precede level construction')
    print('  ok   spacing refresh block precedes direct level construction')

    reject_uses(body, PROTECTED, [(start, end), build.span()], 'checkpoint reader')
    print('  ok   no other explicit nominal/spacing storage or refresh/construction uses')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    try:
        check(load(root))
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('3 checkpoint spacing source contracts passed; GPU lifetime and WRF ordering remain assumptions.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
