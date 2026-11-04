#!/usr/bin/env python3
"""Fault injection for the checkpoint spacing-refresh source contract."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import brace_block, no_literals, signature, scopes
from check_stretched_dz_consumers import pattern
from check_stretched_dz_restart import BUILD, CHECKPOINT, LOOP, PROTECTED, REFRESH
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_restart.py'
SUCCESS = '3 checkpoint spacing source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    # Comments are irrelevant to the contract; simplify mutation anchors.
    text = strip_comments(original)
    _, body, definition = signature(text, 'ERF::ReadCheckpointFile')
    offset = definition.end()
    anchor = unique(pattern('std::string ZLevelsFile('), body, 'file declaration')
    begin = scopes(body, anchor.start())[0][1]
    block = '{'+brace_block(body, begin)+'}'
    end = begin+len(block)
    build = unique(pattern(BUILD), body, 'build')

    def fixture(name, new_body, fault=True):
        assert new_body != body, name
        return name, text[:offset]+new_body+text[offset+len(body):], fault

    for name, before, after in (
        ('wrong filename', '"/zlevels"', '"/other"'),
        ('inverted existence gate', 'if (amrex::FileExists(ZLevelsFile))', 'if (!amrex::FileExists(ZLevelsFile))'),
        ('unconditional file read', 'if (amrex::FileExists(ZLevelsFile))', 'if (true)'),
        ('wrong broadcast source', 'ReadAndBcastFile(ZLevelsFile,', 'ReadAndBcastFile(File,'),
        ('wrong broadcast destination', 'ZLevelsFile, ZLevelsfileCharPtr)', 'ZLevelsFile, fileCharPtr)'),
        ('wrong stream backing', 'ZLevelsfileCharPtr.dataPtr()', 'fileCharPtr.dataPtr()'),
        ('wrong stream mode', 'std::istringstream::in', 'std::istringstream::out'),
        ('wrong level count stream', 'isz >> nlevs_in_chk;', 'is >> nlevs_in_chk;'),
        ('missing level count read', 'isz >> nlevs_in_chk;', ''),
        ('skip coarse level', 'int lev = 0; lev < nlevs_in_chk', 'int lev = 1; lev < nlevs_in_chk'),
        ('extra level', 'lev < nlevs_in_chk', 'lev <= nlevs_in_chk'),
        ('wrong level bound', 'lev < nlevs_in_chk', 'lev < finest_level'),
        ('wrong level direction', '++lev', '--lev'),
        ('wrong node count stream', 'isz >> nz_stag;', 'is >> nz_stag;'),
        ('wrong temporary size', 'zlevels_from_chk(nz_stag)', 'zlevels_from_chk(nz_stag-1)'),
        ('temporary alias', 'Vector<Real> zlevels_from_chk(nz_stag);', 'auto& zlevels_from_chk = zlevels_stag[lev];'),
        ('skip first node', 'int k = 0; k < nz_stag', 'int k = 1; k < nz_stag'),
        ('extra node', 'k < nz_stag', 'k <= nz_stag'),
        ('wrong node destination', 'isz >> zlevels_from_chk[k];', 'isz >> zlevels_stag[lev][k];'),
        ('wrong node stream', 'isz >> zlevels_from_chk[k];', 'is >> zlevels_from_chk[k];'),
        ('wrong retained-level gate', 'lev >= static_cast<int>(zlevels_stag.size())', 'lev > static_cast<int>(zlevels_stag.size())'),
        ('inverted retained-level gate', 'lev >= static_cast<int>(zlevels_stag.size())', 'lev < static_cast<int>(zlevels_stag.size())'),
        ('skip terminates reading', '{ continue; }', '{ break; }'),
        ('inverted size guard', '.size()) != nz_stag', '.size()) == nz_stag'),
        ('wrong size target', 'zlevels_stag[lev].size()) != nz_stag', 'zlevels_stag[0].size()) != nz_stag'),
        ('nonterminating size mismatch', 'Abort("Cannot restart with a different number of cells in z");', 'Print() << "wrong size";'),
        ('wrong replacement level', 'zlevels_stag[lev] = zlevels_from_chk;', 'zlevels_stag[0] = zlevels_from_chk;'),
        ('wrong replacement data', 'zlevels_stag[lev] = zlevels_from_chk;', 'zlevels_stag[lev] = other;'),
        ('missing refresh', REFRESH, ''),
        ('duplicate refresh', REFRESH, REFRESH+REFRESH),
        ('wrong refresh level', REFRESH, REFRESH.replace('(lev,', '(0,')),
        ('wrong refresh nodes', REFRESH, REFRESH.replace('zlevels_stag', 'other')),
        ('wrong refresh host', REFRESH, REFRESH.replace('stretched_dz_h', 'other')),
        ('wrong refresh device', REFRESH, REFRESH.replace('stretched_dz_d', 'other')),
        ('conditional refresh', REFRESH, 'if (lev == 0) { '+REFRESH+' }'),
        ('deferred refresh', REFRESH, 'auto deferred = [&] { '+REFRESH+' };'),
        ('unbraced conditional refresh', REFRESH, 'if (false) '+REFRESH),
        ('continue before refresh', REFRESH, 'continue; '+REFRESH),
        ('mutate after refresh', REFRESH, REFRESH+' zlevels_stag[lev][0] = 0;'),
        ('resize after refresh', REFRESH, REFRESH+' stretched_dz_d[lev].clear();'),
        ('wrong top read', 'isz >> z_top;', 'is >> z_top;'),
        ('missing top read', 'isz >> z_top;', ''),
        ('inverted fallback', 'z_top == zero', 'z_top != zero'),
        ('missing outer empty guard', '&& !zlevels_stag.empty()', ''),
        ('missing inner empty guard', '&& !zlevels_stag[0].empty()', ''),
        ('wrong fallback level', 'z_top = zlevels_stag[0].back();', 'z_top = zlevels_stag[1].back();'),
        ('wrong fallback node', 'zlevels_stag[0].back()', 'zlevels_stag[0].front()'),
        ('wrong construction level', BUILD, BUILD.replace('(lev,', '(0,')),
        ('missing construction', BUILD, ''),
        ('duplicate construction', BUILD, BUILD+BUILD),
        ('conditional construction', BUILD, 'if (lev == 0) { '+BUILD+' }'),
        ('deferred construction', BUILD, 'auto deferred = [&] { '+BUILD+' };'),
        ('unbraced conditional construction', BUILD, 'if (lev == 0) '+BUILD),
        ('wrong construction loop bound', LOOP, LOOP.replace('<=', '<')),
    ):
        # Restrict edits to the reload block unless they target construction.
        target = body if name.endswith('construction') or name.startswith(('wrong construction', 'missing construction', 'duplicate construction')) else block
        search = pattern(before).removeprefix(r'(?<![\w:.>])') if '"' not in before else re.escape(before)
        match = unique(search, target[:build.start()] if before == LOOP else target, name)
        changed = target[:match.start()]+after+target[match.end():]
        yield fixture(name, changed if target is body else body[:begin]+changed+body[end:])

    refresh = unique(pattern(REFRESH), block, 'refresh')
    replacement = unique(pattern('zlevels_stag[lev] = zlevels_from_chk;'), block, 'replacement')
    changed = block[:replacement.start()]+REFRESH+block[replacement.start():refresh.start()]+block[refresh.end():]
    yield fixture('refresh before replacement', body[:begin]+changed+body[end:])
    # Move the whole checked block past the level builder, preserving valid scopes.
    loop_open = scopes(body, build.start())[0][1]
    loop_end = loop_open+len(brace_block(body, loop_open))+2
    yield fixture('reload after construction', body[:begin]+body[end:loop_end]+block+body[loop_end:])
    for name, prefix, suffix in (
        ('conditional block', 'if (false) ', ''),
        ('deferred block', 'auto deferred = [&] ', ';'),
        ('nested block', '{', '}'),
        ('NETCDF-only block', '\n#ifdef ERF_USE_NETCDF\n', '\n#endif\n'),
        ('disabled block', '\n#if 0\n', '\n#endif\n'),
    ):
        yield fixture(name, body[:begin]+prefix+block+suffix+body[end:])
    for name, code in (
        ('vector escape', 'retain(stretched_dz_d);'),
        ('host alias', 'auto& alias = stretched_dz_h;'),
        ('nominal alias', 'auto& alias = zlevels_stag;'),
        ('callee shadow', 'auto update_stretched_dz = other;'),
        ('builder shadow', 'auto MakeNewLevelFromScratch = other;'),
        ('file shadow', 'auto ZLevelsFile = other;'),
        ('goto bypass', 'goto after_reload;'),
        ('runtime include', '\n#include "unmodeled.H"\n'),
        ('literal hides write', 'Print() << "} /* //"; stretched_dz_d.clear();'),
    ):
        for site in (0, end, len(body)):
            yield fixture(name+f' at {site}', body[:site]+code+body[site:])
    for name, code in (
        ('macro', '#define update_stretched_dz ignore'),
        ('undef', '#undef update_stretched_dz'),
        ('unknown conditional', '#ifdef OTHER\n#endif'),
        ('line splice', '// hidden \\\n#define update_stretched_dz ignore'),
        ('raw string', 'R"(stretched_dz_d)";'),
        ('digraph', '<% stretched_dz_d.clear(); %>'),
    ):
        yield name, text+'\n'+code+'\n', True
    for name, before, after in (
        ('diagnostic side effect', '<< nz_stag', '<< mutate(stretched_dz_d)'),
        ('fallback diagnostic side effect', '<< z_top', '<< mutate(zlevels_stag)'),
        ('wrong NETCDF guard', '#ifdef ERF_USE_NETCDF', '#ifdef ERF_USE_PARTICLES'),
    ):
        assert before in block
        yield fixture(name, body[:begin]+block.replace(before, after, 1)+body[end:])
    for macro in ('ERF_USE_NETCDF', 'ERF_USE_PARTICLES'):
        code = '\n#ifdef '+macro+'\nstretched_dz_d.clear();\n#endif\n'
        yield fixture('extra write under '+macro, body[:end]+code+body[end:])

    yield 'comments removed', text, False
    yield 'CRLF', original.replace('\n', '\r\n'), False
    yield 'token whitespace', text.replace(',', ', /* gap */ ').replace(';', ';\n'), False
    yield fixture('unrelated local', body[:end]+'const int unused = 1;'+body[end:], False)
    yield fixture('quoted protected names', body[:end]+'Print() << "} /* // '+ ' '.join(PROTECTED)+'";'+body[end:], False)
    diagnostic = '"Cannot restart with a different number of cells in z"'
    yield fixture('diagnostic wording', body.replace(diagnostic, '"Checkpoint vertical extent differs"'), False)
    warning = unique(r'Print\(\) << "WARNING:', block, 'fallback warning')
    warning_end = no_literals(block).index(';', warning.start())+1
    changed = block[:warning.start()]+'Print() << "legacy top fallback; } /* // " << z_top;'+block[warning_end:]
    yield fixture('warning wording', body[:begin]+changed+body[end:], False)
    log = unique(r'Print\(\) << "Checkpoint holds[\s\S]*?;', block, 'size warning')
    changed = block[:log.start()]+block[log.end():]
    yield fixture('optional size log removed', body[:begin]+changed+body[end:], False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-restart-mutants-'))
    fixtures = list(cases((ROOT/CHECKPOINT).read_text()))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, source, fault) in enumerate(fixtures):
        root = work/f'{index:03d}'
        target = root/CHECKPOINT
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source)
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught, expected, _ = mutation_outcome(result, fault)
        required = 'FAIL source contract:' if fault else SUCCESS
        expected = expected and required in result.stdout
        faults += fault
        controls += not fault and name != 'baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
