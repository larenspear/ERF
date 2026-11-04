#!/usr/bin/env python3
"""Source fault injection for no-substep completion; fixtures are not compiled."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import STRING, brace_block, no_literals, signature
from check_stretched_dz_consumers import pattern
from check_stretched_dz_no_substep_completion import (BCS, CALLBACK, CONTROLS,
    DYCORE, FILES, FRAGMENTS, HEADER, INVOKE, LOOPS, MRI, PROJECTION, REGISTRATION,
    TILING, TYPE)
from momentum_geometry import call_arguments, unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_no_substep_completion.py'
SUCCESS = '4 no-substep completion source contracts passed;'


def cases(original):
    yield 'baseline', original, False
    texts = {path: strip_comments(text) for path, text in original.items()}

    def fixture(name, path, text, fault=True):
        assert text != texts[path], name
        return name, {**texts, path: text}, fault

    def replace(name, path, before, after, fault=True):
        m = unique(pattern(before), texts[path], name)
        return fixture(name, path, texts[path][:m.start()]+after+texts[path][m.end():], fault)

    header = unique(pattern(HEADER), texts[CALLBACK], 'callback header')
    _, body, definition = signature(texts[DYCORE], 'ERF::advance_dycore')
    advance = unique(pattern('double advance(T& S_old, T& S_new, double time, const double time_step) {'),
                     texts[MRI], 'MRI advance')
    bodies = {CALLBACK: (brace_block(texts[CALLBACK], header.end()-1), header.end()),
              DYCORE: (body, definition.end()),
              MRI: (brace_block(texts[MRI], advance.end()-1), advance.end())}

    def body_fixture(name, path, changed, fault=True):
        body, offset = bodies[path]
        return fixture(name, path, texts[path][:offset]+changed+texts[path][offset+len(body):], fault)

    for tag, before, after in (
        ('value capture', '[&', '[='),
        ('missing constant capture', ',bogus_large_value_d=bogus_large_value', ''),
        ('state by value', 'Vector<MultiFab>& S_sum', 'Vector<MultiFab> S_sum'),
        ('wrong time type', 'const double time_for_fp', 'const float time_for_fp'),
        ('stage reference', 'const int nrk', 'const int& nrk'),
    ):
        yield replace('callback '+tag, CALLBACK, HEADER, HEADER.replace(before, after))
    yield replace('MRI reference member', MRI, TYPE+' no_substep;', TYPE+'& no_substep;')
    yield replace('MRI altered member type', MRI, TYPE+' no_substep;',
                  TYPE.replace('int)', 'int&)')+' no_substep;')
    yield replace('MRI setter reference parameter', MRI, 'void set_no_substep('+TYPE+' F)',
                  'void set_no_substep('+TYPE+'& F)')
    for tag, changed in (
        ('setter retains reference', 'no_substep = std::ref(F);'),
        ('deferred setter', 'auto later = [&] { no_substep = F; };'),
        ('conditional setter', 'if (other) { no_substep = F; }'),
        ('wrong setter value', 'no_substep = other;'),
    ):
        yield replace('MRI '+tag, MRI, 'no_substep = F;', changed)

    body, _ = bodies[MRI]
    invokes = list(re.finditer(pattern(INVOKE), body))
    assert len(invokes) == 2
    for index, m in enumerate(invokes):
        for tag, changed in (
            ('deferred call', 'auto later = [&] { '+m[0]+' };'),
            ('conditional call', 'if (other) { '+m[0]+' }'),
            ('unbraced conditional', 'if (other) '+m[0]),
            ('missing call', ''),
            ('duplicate call', m[0]+m[0]),
            ('wrong state', m[0].replace('*S_sum,', 'S_new,')),
            ('wrong old state', m[0].replace('S_old,', 'S_new,')),
            ('wrong RHS', m[0].replace('*F_slow,', 'S_old,')),
            ('wrong time', re.sub(pattern('time + nsubsteps*dtau'), 'time', m[0])),
            ('wrong interval', re.sub(pattern('nsubsteps*dtau, nrk'), 'dtau, nrk', m[0])),
            ('wrong stage', m[0].replace('nrk)', 'nrk+1)')),
            ('outside stage', ''),
        ):
            changed_body = body[:m.start()]+changed+body[m.end():]
            if tag == 'outside stage':
                changed_body += m[0]
            yield body_fixture(f'MRI site {index} '+tag, MRI, changed_body)
        # Move the neighboring slow call, preserving no_substep's exact scopes.
        for kind, after in (('pre', True), ('post', False)):
            slow = list(re.finditer(pattern('slow_rhs_'+kind+'('), body))[index]
            slow_end = body.index(';', slow.end())+1
            statement = body[slow.start():slow_end]
            if after:
                changed_body = body[:slow.start()]+body[slow_end:m.end()]+statement+body[m.end():]
            else:
                changed_body = body[:m.start()]+statement+body[m.start():slow.start()]+body[slow_end:]
            yield body_fixture(f'MRI site {index} order '+kind, MRI, changed_body)
    for before, after in (
        ('if (!anelastic)', 'if (anelastic)'),
        ('for (int nrk = 0; nrk < 3; nrk++)', 'for (int nrk = 0; nrk < 2; nrk++)'),
        ('for (int nrk = 0; nrk < 2; nrk++)', 'for (int nrk = 0; nrk < 1; nrk++)'),
    ):
        yield replace('MRI scope '+after, MRI, before, after)
    gate = list(re.finditer(pattern('if (!no_substepping)'), body[:invokes[0].start()]))[-1]
    yield body_fixture('MRI reversed substep gate', MRI,
                       body[:gate.start()]+'if (no_substepping)'+body[gate.end():])
    # Keep an else scope at the call but attach it to a different if.
    end = gate.end()+len(brace_block(body, body.index('{', gate.end())))
    closing = body.index('}', end)
    yield body_fixture('MRI detached else branch', MRI,
        body[:closing+1]+' if (other) {} '+body[closing+1:])
    for tag, code in (
        ('callback escape', 'retain(no_substep);'),
        ('callback alias', 'auto& alias = no_substep;'),
        ('extra setter', 'set_no_substep(other);'),
        ('outer iterator', 'MFIter outer(other);'),
        ('runtime include', '\n#include "unmodeled.H"\n'),
    ):
        yield body_fixture('MRI '+tag, MRI, code+body)

    body, _ = bodies[DYCORE]
    m = unique(pattern(REGISTRATION), body, 'registration')
    for tag, changed in (
        ('wrong callback', m[0].replace('set_no_substep(no_substep_fun)', 'set_no_substep(other)')),
        ('wrong level', m[0].replace('mri_integrator_mem[level]', 'mri_integrator_mem[level+1]')),
        ('missing definition', m[0].replace('#include "ERF_TI_no_substep_fun.H"', '')),
        ('deferred advance', 'auto later = [&] {\n'+m[0]+'\n};'),
        ('conditional advance', 'if (other) {\n'+m[0]+'\n}'),
    ):
        yield body_fixture('dycore '+tag, DYCORE, body[:m.start()]+changed+body[m.end():])
    for tag, code in (
        ('callback escape', 'retain(no_substep_fun);'),
        ('extra callback invocation', 'no_substep_fun(other);'),
        ('extra setter', 'other.set_no_substep(no_substep_fun);'),
        ('extra advance', 'mri_integrator.advance(state_old,state_new,old_time,dt_advance);'),
    ):
        yield body_fixture('dycore '+tag, DYCORE, body+code)

    body, _ = bodies[CALLBACK]
    loops = []
    for index, statement in enumerate(LOOPS):
        m = unique(pattern(statement), body, 'iterator')
        end = m.end()+len(brace_block(body, m.end()-1))+1
        loops.append((m, end))
        for tag, changed in (
            ('heap iterator', m[0].replace('MFIter mfi', 'auto* mfi = new MFIter')),
            ('wrong state', m[0].replace('S_sum', 'S_old')),
            ('explicit no-sync constructor', m[0].replace('do_tiling' if index == 0 else 'TilingIfNotGPU()',
                                                         'MFItInfo().DisableDeviceSync()')),
            ('wrong increment', m[0].replace('++mfi', 'mfi++')),
        ):
            yield body_fixture(f'loop {index} '+tag, CALLBACK, body[:m.start()]+changed+body[m.end():])
        for tag, changed in (
            ('deferred loop', 'auto later = [&] { '+body[m.start():end]+' };'),
            ('conditional loop', 'if (other) { '+body[m.start():end]+' }'),
            ('duplicate loop', body[m.start():end]*2),
            ('missing loop', ''),
        ):
            yield body_fixture(f'loop {index} '+tag, CALLBACK, body[:m.start()]+changed+body[end:])
        for tag, code in (
            ('iterator alias', 'auto& alias = mfi;'),
            ('iterator move', 'auto moved = std::move(mfi);'),
            ('early Finalize', 'mfi.Finalize();'),
            ('extra increment', '++mfi;'),
            ('stream override', 'Gpu::Device::setStreamIndex(2);'),
        ):
            yield body_fixture(f'loop {index} '+tag, CALLBACK, body[:m.end()]+code+body[m.end():])
    first, second = loops
    yield body_fixture('update loops reversed', CALLBACK,
        body[:first[0].start()]+body[second[0].start():second[1]]+body[first[1]:second[0].start()]+
        body[first[0].start():first[1]]+body[second[1]:])
    yield replace('moving terrain tiled', CALLBACK, TILING, TILING.replace('? false', '? true'))
    yield replace('mutable tiling selector', CALLBACK, TILING, TILING.replace('const bool', 'bool'))
    yield body_fixture('tiling selector mutated', CALLBACK,
        body[:first[0].start()]+'do_tiling = true;'+body[first[0].start():])
    eb = list(re.finditer(pattern('if (solverChoice.terrain_type == TerrainType::EB)'), body))[-1]
    yield body_fixture('wrong second-loop EB gate', CALLBACK,
        body[:eb.start()]+'if (solverChoice.terrain_type != TerrainType::EB)'+body[eb.end():])
    # All eight complete launch sites are independently moved or deferred.
    matches = list(re.finditer(r'\bParallelFor\s*\(', body))
    args = list(call_arguments(no_literals(body), r'\bParallelFor'))
    assert len(matches) == len(args) == 8
    for index, (m, arguments) in enumerate(zip(matches, args)):
        end = body.index(';', arguments[-1][1])+1
        statement = body[m.start():end]
        for tag, changed in (
            ('deferred launch', 'auto later = [&] { '+statement+' };'),
            ('launch outside iterator', ''),
            ('missing launch', ''),
            ('duplicate launch', statement+statement),
        ):
            changed_body = body[:m.start()]+changed+body[end:]
            if tag == 'launch outside iterator':
                changed_body += statement
            yield body_fixture(f'launch {index} '+tag, CALLBACK, changed_body)
        for kernel_index, (lo, hi) in enumerate(arguments):
            if not body[lo:hi].lstrip().startswith('[=]'):
                continue
            for tag, changed in (
                ('reference capture', body[lo:hi].replace('[=]', '[&]', 1)),
                ('host kernel', body[lo:hi].replace('AMREX_GPU_DEVICE', '', 1)),
            ):
                yield body_fixture(f'launch {index} kernel {kernel_index} '+tag, CALLBACK,
                    body[:lo]+changed+body[hi:])
    m = matches[0]
    yield body_fixture('wrong cell box', CALLBACK, body[:m.end()]+'other_box,'+body[args[0][0][1]+1:])
    yield body_fixture('launch callee alias', CALLBACK, body+'retain(ParallelFor);')
    yield body_fixture('iterator outside both loops', CALLBACK, body+'MFIter outer(other);')
    yield body_fixture('iterator use outside loop', CALLBACK, body+'mfi.tilebox();')
    yield body_fixture('callback escape', CALLBACK, body+'retain(no_substep_fun);')
    for spacing in ('stretched_dz_h', 'stretched_dz_d'):
        yield body_fixture('new spacing use '+spacing, CALLBACK, body+'retain('+spacing+'[level]);')

    bcs = unique(pattern(BCS), body, 'boundary call')
    projection = unique(pattern(PROJECTION), body, 'projection')
    for tag, changed in (
        ('deferred boundary call', 'auto later = [&] { '+bcs[0]+' };'),
        ('conditional boundary call', 'if (other) { '+bcs[0]+' }'),
        ('missing boundary call', ''),
        ('duplicate boundary call', bcs[0]*2),
        ('wrong boundary state', bcs[0].replace('S_sum,', 'S_old,', 1)),
        ('wrong boundary time', bcs[0].replace('time_for_fp', 'slow_dt')),
        ('wrong fast flag', bcs[0].replace('fast_only=true', 'fast_only=false')),
        ('wrong sync flag', bcs[0].replace('vel_and_mom_synced=false', 'vel_and_mom_synced=true')),
    ):
        yield body_fixture(tag, CALLBACK, body[:bcs.start()]+changed+body[bcs.end():])
    yield body_fixture('boundary before loop destruction', CALLBACK,
        body[:first[1]-1]+bcs[0]+body[first[1]-1:bcs.start()]+body[bcs.end():])
    yield body_fixture('projection before boundary', CALLBACK,
        body[:bcs.start()]+projection[0]+body[bcs.start():projection.start()]+body[projection.end():])
    for tag, before, after in (
        ('wrong anelastic level', 'anelastic[level]', 'anelastic[level+1]'),
        ('wrong thin-body condition', 'thin_xforce[0]', 'thin_xforce[1]'),
        ('reversed thin-body condition', 'if (have_tb)', 'if (!have_tb)'),
        ('wrong thin-body level', 'project_velocity_tb(level,', 'project_velocity_tb(level+1,'),
        ('wrong thin-body interval', 'project_velocity_tb(level, slow_dt', 'project_velocity_tb(level, time_for_fp'),
        ('wrong thin-body state', 'project_velocity_tb(level, slow_dt, S_sum)', 'project_velocity_tb(level, slow_dt, S_old)'),
        ('wrong momentum time', 'project_momenta(level, time_for_fp', 'project_momenta(level, slow_dt'),
        ('wrong momentum interval', 'time_for_fp, slow_dt', 'time_for_fp, slow_dt_d'),
        ('wrong momentum state', 'time_for_fp, slow_dt, S_sum', 'time_for_fp, slow_dt, S_old'),
    ):
        yield replace('projection '+tag, CALLBACK, PROJECTION, PROJECTION.replace(before, after))
    for name, statement in (
        ('thin body', 'project_velocity_tb(level, slow_dt, S_sum);'),
        ('momentum', 'project_momenta(level, time_for_fp, slow_dt, S_sum);'),
    ):
        for tag, changed in (
            ('deferred', 'auto later = [&] { '+statement+' };'),
            ('missing', ''), ('duplicate', statement*2),
        ):
            yield replace(name+' projection '+tag, CALLBACK, statement, changed)
        yield body_fixture(name+' projection escape', CALLBACK, body+'retain('+statement.split('(')[0]+');')
    yield replace('wrong interval conversion', CALLBACK,
        'amrex::Real slow_dt = static_cast<Real>(slow_dt_d);', 'amrex::Real slow_dt = static_cast<Real>(time_for_fp);')
    for tag, code in (
        ('unknown conditional', '#ifdef OTHER\n#endif'),
        ('unknown directive', '#pragma unmodeled'),
        ('runtime include', '#include "unmodeled.H"'),
        ('macro undef', '#undef AMREX_USE_GPU'),
        ('line splice', '// hidden \\\n#define no_substep other'),
        ('raw string', 'R"(hidden)";'),
        ('digraph', '<% unknown(); %>'),
        ('quoted escape', 'Print() << "} // /*"; Gpu::setNoSyncRegion(true);'),
    ):
        yield body_fixture(tag, CALLBACK, '\n'+code+'\n'+body)
    yield replace('OpenMP without GPU guard', CALLBACK,
        '#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())', '#pragma omp parallel')
    for path in FILES:
        for tag, code in (
            ('NoSyncRegion', 'Gpu::NoSyncRegion region;'),
            ('stream override', 'Gpu::Device::setStreamIndex(2);'),
            ('nonlocal transfer', 'goto outside;'),
            ('macro override', '#define no_substep other'),
        ):
            yield fixture(Path(path).stem+' '+tag, path, code+'\n'+texts[path])
    for path in FRAGMENTS:
        if path != CALLBACK:
            yield fixture(Path(path).stem+' callback shadow', path, texts[path]+'\nauto no_substep_fun = other;')
            yield fixture(Path(path).stem+' extra include', path, texts[path]+'\n#include "unmodeled.H"\n')

    yield 'comments removed', texts, False
    yield 'CRLF', {path: text.replace('\n', '\r\n') for path, text in original.items()}, False

    def whitespace(text):
        parts = re.split('('+STRING+')', text)
        return ''.join(part if i%2 else part.replace(',', ', /* gap */ ').replace(';', ';\n')
                       for i, part in enumerate(parts))

    yield 'token whitespace', {path: whitespace(text) for path, text in texts.items()}, False
    for path, (body, _) in bodies.items():
        for tag, code in (
            ('local', 'const int unused = 1;'),
            ('quoted controls', 'Print() << "} /* // '+' '.join(CONTROLS)+'";'),
            ('additional synchronization', 'Gpu::streamSynchronize();'),
        ):
            yield body_fixture(Path(path).stem+' '+tag, path, code+body, False)
    body, _ = bodies[CALLBACK]
    for index, (loop, _) in enumerate(loops):
        yield body_fixture(f'loop {index} additional synchronization', CALLBACK,
            body[:loop.end()]+'Gpu::streamSynchronize();'+body[loop.end():], False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-no-substep-completion-mutants-'))
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
        expected = expected and ('FAIL source contract:' if fault else SUCCESS) in result.stdout
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
