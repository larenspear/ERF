#!/usr/bin/env python3
"""Fault injection for WRF declarations, mode selection and caller ordering."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, signature
from check_stretched_dz_consumers import pattern
from check_stretched_dz_wrf_callers import (CALL, COARSE, FILES, GATE, HEADER, INIT,
                                           PROTECTED, SELECT, SURFACE_CALL, WRF)
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_wrf_callers.py'
SUCCESS = '4 WRF caller source contracts passed;'
FUNCTIONS = {INIT: 'ERF::init_only', COARSE: 'ERF::MakeNewLevelFromCoarse'}


def cases(original):
    yield 'baseline', original, False
    texts = {path: strip_comments(text) for path, text in original.items()}

    def fixture(name, path, text, fault=True):
        assert text != texts[path], name
        return name, {**texts, path: text}, fault

    def replace(name, path, before, after, fault=True):
        match = unique(pattern(before).removeprefix(r'(?<![\w:.>])'), texts[path], name)
        return fixture(name, path, texts[path][:match.start()]+after+texts[path][match.end():], fault)

    declaration = 'void init_from_wrfinput(int lev, amrex::MultiFab& mf_PSFC, bool read_atmos_state = true);'
    for name, before, after in (
        ('false default', 'bool read_atmos_state = true', 'bool read_atmos_state = false'),
        ('missing default', 'bool read_atmos_state = true', 'bool read_atmos_state'),
        ('wrong declaration mode type', 'bool read_atmos_state = true', 'int read_atmos_state = true'),
        ('reference mode', 'bool read_atmos_state = true', 'bool& read_atmos_state = true'),
        ('wrong PSFC type', 'amrex::MultiFab& mf_PSFC', 'amrex::MultiFab* mf_PSFC'),
        ('const PSFC', 'amrex::MultiFab& mf_PSFC', 'const amrex::MultiFab& mf_PSFC'),
        ('overload', declaration, declaration+' void init_from_wrfinput(int lev);'),
        ('nested declaration', declaration, 'struct Nested { '+declaration+' };'),
        ('conditional declaration', declaration, '\n#ifdef OTHER\n'+declaration+'\n#endif\n'),
        ('factory-only declaration', declaration, '\n#if USE_FC_FACTORY\n'+declaration+'\n#endif\n'),
    ):
        yield replace(name, HEADER, before, after)
    for name, before, after in (
        ('definition mode type', 'bool read_atmos_state)', 'int read_atmos_state)'),
        ('definition PSFC type', 'MultiFab& mf_PSFC_lev,', 'MultiFab* mf_PSFC_lev,'),
        ('definition extra default', 'bool read_atmos_state)', 'bool read_atmos_state = false)'),
        ('definition renamed mode', 'bool read_atmos_state)', 'bool read_surface_state)'),
        ('definition guarded', 'void ERF::init_from_wrfinput', '#ifdef OTHER\nvoid ERF::init_from_wrfinput'),
    ):
        item = replace(name, WRF, before, after)
        if name == 'definition guarded':
            item[1][WRF] += '\n#endif\n'
        yield item
    yield fixture('global overload', HEADER, texts[HEADER]+'\nvoid init_from_wrfinput(int lev);\n')

    bodies = {}
    for path, name in FUNCTIONS.items():
        _, body, definition = signature(texts[path], name)
        bodies[path] = body, definition.end()

    def body_fixture(name, path, changed, fault=True):
        body, offset = bodies[path]
        assert changed != body, name
        return fixture(name, path, texts[path][:offset]+changed+texts[path][offset+len(body):], fault)

    def body_replace(name, path, before, after, fault=True):
        body, _ = bodies[path]
        match = unique(pattern(before).removeprefix(r'(?<![\w:.>])'), body, name)
        return body_fixture(name, path, body[:match.start()]+after+body[match.end():], fault)

    for path in (INIT, COARSE):
        label = 'ordinary' if path == INIT else 'coarse'
        for name, replacement in (
            ('missing call', ''), ('duplicate call', CALL+CALL),
            ('wrong level', CALL.replace('(lev,', '(0,')),
            ('wrong PSFC level', CALL.replace('[lev]', '[0]')),
            ('wrong PSFC field', CALL.replace('mf_PSFC', 'other')),
            ('wrong mode', CALL.replace(');', ', false);')),
            ('conditional call', 'if (verbose) { '+CALL+' }'),
            ('unbraced conditional call', 'if (verbose) '+CALL),
            ('deferred call', 'auto later = [&] { '+CALL+' };'),
            ('qualified other call', 'other.'+CALL),
        ):
            yield body_replace(label+' '+name, path, CALL, replacement)

    for name, before, after in (
        ('ordinary inverted file gate', '!nc_init_file[lev].empty()', 'nc_init_file[lev].empty()'),
        ('ordinary wrong dispatch type', 'else if (solverChoice.init_type == InitType::WRFInput && !nc_init_file[lev].empty())',
         'else if (solverChoice.init_type == InitType::Metgrid && !nc_init_file[lev].empty())'),
        ('ordinary wrong gate level', '!nc_init_file[lev].empty()', '!nc_init_file[0].empty()'),
        ('ordinary missing sounding predecessor', 'if (solverChoice.init_type == InitType::Input_Sounding) {',
         'if (false) {'),
    ):
        yield body_replace(name, INIT, before, after)

    init_body, _ = bodies[INIT]
    call = unique(pattern(CALL), init_body, 'ordinary call')
    tail = init_body[call.start():]
    phys = unique(pattern('make_physbcs(lev);'), brace_block('{'+tail, 0), 'ordinary physbc')
    # The matched enclosing WRF branch ends after the boundary application.
    call_end = call.start()+phys.end()
    yield body_fixture('ordinary boundary setup before WRF', INIT,
                       init_body[:call.start()]+'make_physbcs(lev);'+CALL+init_body[call_end:])

    for name, before, after in (
        ('initial mode true', '[[maybe_unused]] bool use_surface_only = false;',
         '[[maybe_unused]] bool use_surface_only = true;'),
        ('initial mode reference', '[[maybe_unused]] bool use_surface_only = false;',
         '[[maybe_unused]] bool& use_surface_only = other;'),
        ('initial mode conditional', '[[maybe_unused]] bool use_surface_only = false;',
         'if (verbose) { [[maybe_unused]] bool use_surface_only = false; }'),
        ('coarse wrong file gate', GATE, GATE.replace(' ||', ' &&')),
        ('coarse inverted file gate', GATE, GATE.replace('!nc_init', 'nc_init')),
        ('coarse wrong file level', GATE, GATE.replace('[lev]', '[0]')),
        ('selector inverted interpolation', SELECT, SELECT.replace('= solverChoice', '= !solverChoice')),
        ('selector wrong level', SELECT, SELECT.replace('lev > 0', 'lev == 0')),
        ('selector wrong initialization type', SELECT, SELECT.replace('WRFInput', 'Metgrid')),
        ('selector or instead of and', SELECT, SELECT.replace('&&', '||', 1)),
        ('conditional selector', SELECT, 'if (verbose) { '+SELECT+' }'),
        ('unbraced conditional selector', SELECT, 'if (verbose) '+SELECT),
        ('deferred selector', SELECT, 'auto later = [&] { '+SELECT+' };'),
        ('surface call wrong mode', SURFACE_CALL, SURFACE_CALL.replace('false', 'true')),
        ('surface call omitted mode', SURFACE_CALL, CALL),
        ('surface call wrong level', SURFACE_CALL, SURFACE_CALL.replace('(lev,', '(0,')),
        ('surface call wrong PSFC level', SURFACE_CALL, SURFACE_CALL.replace('[lev]', '[0]')),
        ('surface missing call', SURFACE_CALL, ''),
        ('surface conditional call', SURFACE_CALL, 'if (verbose) { '+SURFACE_CALL+' }'),
        ('surface deferred call', SURFACE_CALL, 'auto later = [&] { '+SURFACE_CALL+' };'),
        ('surface diagnostic side effect', SURFACE_CALL, 'Print() << mutate(use_surface_only);'+SURFACE_CALL),
        ('radiation mode inverted', 'use_surface_only ? 1 : 0', 'use_surface_only ? 0 : 1'),
        ('SLM condition inverted', 'if (use_surface_only && solverChoice.lsm_type',
         'if (!use_surface_only && solverChoice.lsm_type'),
        ('NoahMP mode inverted', 'initialize_now = use_surface_only &&', 'initialize_now = !use_surface_only &&'),
    ):
        yield body_replace(name, COARSE, before, after)

    coarse_body, _ = bodies[COARSE]
    selection = unique(pattern(SELECT), coarse_body, 'selection')
    selected = coarse_body[selection.start():]
    surface_gate = unique(pattern('if (use_surface_only) {'), selected[:selected.index('init_zphys')], 'surface gate')
    at = selection.start()+surface_gate.start()
    end = selection.start()+surface_gate.end()
    yield body_fixture('reversed surface dispatch', COARSE,
                       coarse_body[:at]+'if (!use_surface_only) {'+coarse_body[end:])
    terrain = unique(pattern('init_zphys(lev, time); update_terrain_arrays(lev);'), selected, 'terrain')
    start, end = selection.start()+terrain.start(), selection.start()+terrain.end()
    terrain_calls = coarse_body[start:end]
    for name, replacement in (
        ('missing terrain init', 'update_terrain_arrays(lev);'),
        ('missing terrain update', 'init_zphys(lev, time);'),
        ('reversed terrain calls', 'update_terrain_arrays(lev); init_zphys(lev, time);'),
        ('wrong terrain level', 'init_zphys(0, time); update_terrain_arrays(0);'),
        ('wrong terrain time', 'init_zphys(lev, 0); update_terrain_arrays(lev);'),
        ('surface-only terrain', 'if (use_surface_only) { '+terrain_calls+' }'),
        ('deferred terrain', 'auto later = [&] { '+terrain_calls+' };'),
    ):
        yield body_fixture(name, COARSE, coarse_body[:start]+replacement+coarse_body[end:])
    without_terrain = coarse_body[:start]+coarse_body[end:]
    yield body_fixture('terrain before WRF', COARSE,
                       without_terrain[:selection.start()]+terrain_calls+without_terrain[selection.start():])

    finish = unique(pattern('if (use_surface_only) {'), coarse_body[end:], 'completion')
    finish_start = end+finish.start()
    finish_end = end+finish.end()
    contents = brace_block(coarse_body, finish_end-1)
    stop = finish_end+len(contents)+1
    completion = coarse_body[finish_start:stop]
    for name, replacement in (
        ('inverted completion', completion.replace('if (use_surface_only)', 'if (!use_surface_only)')),
        ('deferred completion', 'auto later = [&] { '+completion+' };'),
        ('conditional completion', 'if (verbose) { '+completion+' }'),
        ('missing completion', ''),
        ('wrong base-state level', completion.replace('rebuild_base_state_from_wrfinput(lev,', 'rebuild_base_state_from_wrfinput(0,')),
        ('wrong interpolation level', completion.replace('FillCoarsePatch(lev,', 'FillCoarsePatch(0,')),
        ('wrong interpolation time', completion.replace('FillCoarsePatch(lev, time)', 'FillCoarsePatch(lev, 0)')),
        ('interpolation before base-state', completion.replace('rebuild_base_state_from_wrfinput',
         'FillCoarsePatch(lev, time); rebuild_base_state_from_wrfinput').replace('FillCoarsePatch(lev, time);\n', '', 1)),
    ):
        yield body_fixture(name, COARSE, coarse_body[:finish_start]+replacement+coarse_body[stop:])
    for name, code in (
        ('changed level between terrain and completion', 'lev = 0;'),
        ('changed time between terrain and completion', 'time = 0;'),
        ('changed selector between terrain and completion', 'solverChoice.init_type = InitType::Metgrid;'),
    ):
        yield body_fixture(name, COARSE, coarse_body[:end]+code+coarse_body[end:])

    for path in (INIT, COARSE):
        body, _ = bodies[path]
        label = 'ordinary' if path == INIT else 'coarse'
        for name, code in (
            ('PSFC alias', 'auto& alias = mf_PSFC;'),
            ('nominal escape', 'retain(zlevels_stag);'),
            ('host mutation', 'stretched_dz_h.clear();'),
            ('device mutation', 'stretched_dz_d.clear();'),
            ('callee shadow', 'auto init_from_wrfinput = [](auto...) {};'),
            ('early return', 'return;'),
            ('goto', 'goto later;'),
            ('runtime include', '\n#include "unmodeled.H"\n'),
            ('quoted scope escape', 'Print() << "} // /*"; stretched_dz_d.clear();'),
        ):
            for site, at in (('prefix', 0), ('suffix', len(body))):
                yield body_fixture(label+' '+name+' '+site, path, body[:at]+code+body[at:])
        for guard in ('#ifdef ERF_USE_NETCDF', '#if USE_FC_FACTORY'):
            for side in (True, False):
                code = 'stretched_dz_d.clear();'
                injected = '\n'+guard+'\n'+(code if side else '')+'\n#else\n'+('' if side else code)+'\n#endif\n'
                yield body_fixture(label+f' hidden write {guard} {side}', path, body+injected)
        for code in ('#ifdef OTHER\n#endif', '#define init_from_wrfinput ignore',
                     '#undef ERF_USE_NETCDF', '// hidden \\\n#define init_from_wrfinput ignore',
                     'R"(hidden)";', '<% stretched_dz_d.clear(); %>'):
            yield body_fixture(label+' unsupported '+code.splitlines()[0], path, body+'\n'+code+'\n')
    for name, code in (
        ('mode reset', 'use_surface_only = false;'),
        ('mode alias', 'auto& alias = use_surface_only;'),
        ('mode escape', 'retain(use_surface_only);'),
    ):
        yield body_fixture(name, COARSE, coarse_body[:stop]+code+coarse_body[stop:])

    yield 'comments removed', texts, False
    yield 'CRLF', {path: text.replace('\n', '\r\n') for path, text in original.items()}, False
    # Change whitespace only outside string/character literals.
    def whitespace(text):
        parts = re.split('('+STRING+')', text)
        return ''.join(part if i % 2 else part.replace(',', ', /* gap */ ').replace(';', ';\n')
                       for i, part in enumerate(parts))
    yield 'token whitespace', {path: whitespace(text) for path, text in texts.items()}, False
    for path in (INIT, COARSE):
        body, _ = bodies[path]
        label = 'ordinary' if path == INIT else 'coarse'
        for name, code in (
            ('unrelated local', 'const int unused = 1;'),
            ('quoted protected names', 'Print() << "} /* // '+ ' '.join(PROTECTED)+'";'),
        ):
            for site, at in (('prefix', 0), ('suffix', len(body))):
                yield body_fixture(label+' '+name+' '+site, path, body[:at]+code+body[at:], False)
    log = unique(r'amrex::Print\(\) << "Using interp_atmos_from_coarse[^;]*;', coarse_body, 'mode diagnostic')
    for name, code in (('diagnostic removed', ''), ('diagnostic wording', 'amrex::Print() << "Mode; } // /*" << lev;')):
        yield body_fixture(name, COARSE, coarse_body[:log.start()]+code+coarse_body[log.end():], False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-wrf-callers-mutants-'))
    fixtures = list(cases({path: (ROOT/path).read_text() for path in FILES}))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, sources, fault) in enumerate(fixtures):
        root = work/f'{index:03d}'
        for path, source in sources.items():
            target = root/path
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
