#!/usr/bin/env python3
"""Source fault injection for acoustic callback completion; fixtures are not compiled."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_acoustic_completion import (BRANCHES, CALLBACK, CONTROLS,
    DYCORE, FAST, FAST_H, FILES, GATE, HEADER, INVOKE, MRI, REGISTRATION, SETUP,
    TYPE, W_SETUP)
from check_stretched_dz_bindings import STRING, brace_block, signature
from check_stretched_dz_consumers import pattern
from momentum_geometry import unique

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_acoustic_completion.py'
SUCCESS = '4 acoustic completion source contracts passed;'


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
        ('missing constant capture', ',zero_d=zero', ''),
        ('step reference', 'int fast_step', 'int& fast_step'),
        ('wrong old-state type', 'const Vector<MultiFab>& S_old', 'Vector<MultiFab>& S_old'),
        ('named full timestep', 'const double,', 'const double slow_dt,'),
        ('wrong interval type', 'const double dtau', 'const float dtau'),
    ):
        yield replace('callback '+tag, CALLBACK, HEADER, HEADER.replace(before, after))
    yield replace('MRI reference member', MRI, TYPE+' acoustic_substepping;', TYPE+'& acoustic_substepping;')
    yield replace('MRI altered member type', MRI, TYPE+' acoustic_substepping;',
                  TYPE.replace('const T&', 'T&')+' acoustic_substepping;')
    for tag, changed in (
        ('setter retains reference', 'acoustic_substepping = std::ref(F);'),
        ('deferred setter', 'auto later = [&] { acoustic_substepping = F; };'),
        ('conditional setter', 'if (other) { acoustic_substepping = F; }'),
        ('wrong setter value', 'acoustic_substepping = other;'),
    ):
        yield replace('MRI '+tag, MRI, 'acoustic_substepping = F;', changed)
    yield replace('MRI setter reference parameter', MRI, 'void set_acoustic_substepping('+TYPE+' F)',
                  'void set_acoustic_substepping('+TYPE+'& F)')

    body, _ = bodies[MRI]
    m = unique(pattern(INVOKE), body, 'MRI acoustic call')
    for tag, changed in (
        ('deferred call', 'auto later = [&] { '+m[0]+' };'),
        ('conditional call', 'if (other) { '+m[0]+' }'),
        ('unbraced conditional call', 'if (other) '+m[0]),
        ('missing call', ''),
        ('duplicate call', m[0]+m[0]),
        ('wrong step', m[0].replace('ks,', 'ks+1,', 1)),
        ('wrong substep count', m[0].replace('nsubsteps,', '1,', 1)),
        ('wrong stage', m[0].replace('nrk,', 'nrk+1,', 1)),
        ('wrong stage state', m[0].replace('S_new,', 'S_old,', 1)),
        ('wrong interval', m[0].replace('dtau,', 'timestep,', 1)),
        ('wrong factor', m[0].replace('inv_fac,', 'one,', 1)),
        ('wrong old time', m[0].replace('time + ks*dtau', 'time')),
        ('wrong new time', re.sub(pattern('time + (ks+1)*dtau'), 'time + dtau', m[0])),
    ):
        yield body_fixture('MRI '+tag, MRI, body[:m.start()]+changed+body[m.end():])
    without = body[:m.start()]+body[m.end():]
    for tag, anchor, after in (
        ('before slow pre', 'slow_rhs_pre(', False),
        ('after slow post', 'slow_rhs_post(', True),
    ):
        pos = re.search(pattern(anchor), without).start()
        if after:
            pos = without.index(';', pos)+1
        yield body_fixture('MRI acoustic '+tag, MRI, without[:pos]+m[0]+without[pos:])
    yield body_fixture('MRI acoustic outside all loops', MRI, without+m[0])
    for before, after in (
        ('if (!anelastic)', 'if (anelastic)'),
        ('for (int nrk = 0; nrk < 3; nrk++)', 'for (int nrk = 0; nrk < 2; nrk++)'),
        ('for (int ks = 0; ks < nsubsteps; ++ks)', 'for (int ks = 1; ks < nsubsteps; ++ks)'),
    ):
        yield replace('MRI scope '+after, MRI, before, after)
    # Select the substep branch containing the invocation, not the earlier setup gate.
    gate = list(re.finditer(pattern('if (!no_substepping)'), body[:m.start()]))[-1]
    yield body_fixture('MRI reversed substep gate', MRI,
                       body[:gate.start()]+'if (no_substepping)'+body[gate.end():])
    opening = body.index('{', gate.end())
    closing = opening+1+len(brace_block(body, opening))
    else_opening = body.index('{', closing+1)
    else_end = else_opening+len(brace_block(body, else_opening))+2
    dispatch = body[gate.start():else_end]
    without_dispatch = body[:gate.start()]+body[else_end:]
    for tag, anchor, after in (
        ('before pre with scopes preserved', 'slow_rhs_pre(', False),
        ('after post with scopes preserved', 'slow_rhs_post(', True),
    ):
        pos = re.search(pattern(anchor), without_dispatch).start()
        if after:
            pos = without_dispatch.index(';', pos)+1
        yield body_fixture('MRI acoustic '+tag, MRI,
            without_dispatch[:pos]+dispatch+without_dispatch[pos:])
    for tag, code in (
        ('callback escape', 'retain(acoustic_substepping);'),
        ('callback alias', 'auto& alias = acoustic_substepping;'),
        ('extra setter call', 'set_acoustic_substepping(other);'),
        ('outer iterator', 'MFIter outer(other);'),
        ('runtime include', '\n#include "unmodeled.H"\n'),
    ):
        yield body_fixture('MRI '+tag, MRI, code+body)

    body, _ = bodies[DYCORE]
    m = unique(pattern(REGISTRATION), body, 'registration')
    for tag, changed in (
        ('wrong callback', m[0].replace('set_acoustic_substepping(acoustic_substepping_fun)',
                                       'set_acoustic_substepping(other)')),
        ('wrong level', m[0].replace('mri_integrator_mem[level]', 'mri_integrator_mem[level+1]')),
        ('missing definition', m[0].replace('#include "ERF_TI_substep_fun.H"', '')),
        ('deferred advance', 'auto later = [&] {\n'+m[0]+'\n};'),
        ('conditional advance', 'if (other) {\n'+m[0]+'\n}'),
    ):
        yield body_fixture('dycore '+tag, DYCORE, body[:m.start()]+changed+body[m.end():])
    for tag, code in (
        ('callback escape', 'retain(acoustic_substepping_fun);'),
        ('extra callback invocation', 'acoustic_substepping_fun(other);'),
        ('extra setter', 'other.set_acoustic_substepping(acoustic_substepping_fun);'),
        ('extra advance', 'mri_integrator.advance(state_old,state_new,old_time,dt_advance);'),
    ):
        yield body_fixture('dycore '+tag, DYCORE, body+code)

    body, _ = bodies[CALLBACK]
    for name, condition in BRANCHES:
        m = unique(pattern(name+'('), body, name)
        end = body.index(';', m.end())+1
        statement = body[m.start():end]
        for tag, changed in (
            ('deferred call', 'auto later = [&] { '+statement+' };'),
            ('conditional call', 'if (other) { '+statement+' }'),
            ('unbraced conditional', 'if (other) '+statement),
            ('missing call', ''),
            ('duplicate call', statement+statement),
            ('nonstatement call', 'retain('+statement[:-1]+');'),
        ):
            yield body_fixture(name+' '+tag, CALLBACK, body[:m.start()]+changed+body[end:])
        yield body_fixture(name+' escaped callee', CALLBACK, body+'retain('+name+');')
        if name == 'erf_substep_NS':
            for tag, before, after in (
                ('wrong spacing level', 'stretched_dz_d[level]', 'stretched_dz_d[level+1]'),
                ('spacing alias', 'stretched_dz_d[level]', 'other_spacing'),
                ('wrong geometry', 'fine_geom', 'other_geom'),
                ('wrong step', 'fast_step,', 'n_sub,'),
                ('wrong stage', 'nrk,', 'nrk+1,'),
                ('wrong level', 'nrk, level,', 'nrk, level+1,'),
                ('wrong interval', 'dtau,', 'dtau*n_sub,'),
                ('wrong factor', 'inv_fac,', 'one_d,'),
                ('extra argument', 'fast_step,', 'fast_step, fast_step,'),
            ):
                assert before in statement, tag
                yield body_fixture('normal substep '+tag, CALLBACK,
                    body[:m.start()]+statement.replace(before, after)+body[end:])
    # Only the final terrain gate selects the dispatch; earlier setup gates are legitimate.
    terrain = list(re.finditer(pattern(BRANCHES[0][1]), body))[-1]
    variable = unique(pattern(BRANCHES[1][1]), body, 'variable branch')
    for tag, changed in (
        ('wrong terrain', 'if (solverChoice.terrain_type == TerrainType::ImmersedForcing)'),
        ('reversed terrain', 'if (solverChoice.terrain_type != TerrainType::MovingFittedMesh)'),
    ):
        yield body_fixture(tag, CALLBACK, body[:terrain.start()]+changed+body[terrain.end():])
    yield replace('wrong variable branch', CALLBACK, BRANCHES[1][1], 'if (solverChoice.mesh_type != MeshType::VariableDz)')
    yield replace('detached variable branch', CALLBACK, 'else '+BRANCHES[1][1], BRANCHES[1][1])
    opening = body.index('{', variable.end())
    closing = opening+1+len(brace_block(body, opening))
    yield body_fixture('detached normal branch', CALLBACK,
        body[:closing+1]+body[closing+1:].replace('else', 'if (other)', 1))
    for tag, code in (
        ('extra device spacing alias', 'auto& alias = stretched_dz_d[level];'),
        ('callback escape', 'retain(acoustic_substepping_fun);'),
    ):
        yield body_fixture(tag, CALLBACK, body+code)
    for path in (FAST, FAST_H):
        yield replace(Path(path).stem+' wrong spacing type', path,
                      'amrex::Gpu::DeviceVector<amrex::Real>& stretched_dz_d',
                      'amrex::Gpu::DeviceVector<amrex::Real> stretched_dz_d')
    yield replace('normal definition wrong interval type', FAST, 'const double dtau_d', 'const float dtau_d')

    gate = unique(pattern(GATE), body, 'implicit gate')
    end = gate.end()+len(brace_block(body, gate.end()-1))+1
    block = body[gate.start():end]
    include = '#include "ERF_ImplicitPre.H"'
    for tag, changed in (
        ('missing include', ''),
        ('duplicate include', include+'\n'+include),
        ('deferred include', 'auto later = [&] {\n'+include+'\n};'),
        ('conditional include', 'if (other) {\n'+include+'\n}'),
        ('unknown include', '#include "unmodeled.H"'),
    ):
        yield replace('implicit '+tag, CALLBACK, include, changed)
    for tag, changed in (
        ('missing block', ''),
        ('deferred block', 'auto later = [&] {\n'+block+'\n};'),
        ('conditional block', 'if (other) {\n'+block+'\n}'),
        ('unbraced conditional block', 'if (other) '+block),
    ):
        yield body_fixture('implicit '+tag, CALLBACK, body[:gate.start()]+changed+body[end:])
    yield body_fixture('implicit before dispatch', CALLBACK,
        body[:terrain.start()]+block+'\n'+body[terrain.start():gate.start()]+body[end:])
    for tag, before, after in (
        ('wrong stage gate', '[level][nrk]', '[level][nrk+1]'),
        ('wrong weight gate', '> zero_d', '>= zero_d'),
        ('wrong implicit mode', '!solverChoice.implicit_before_substep', 'solverChoice.implicit_before_substep'),
        ('wrong final step', 'n_sub-1', 'n_sub'),
        ('first step', 'fast_step == n_sub-1', 'fast_step == 0'),
    ):
        yield replace('implicit '+tag, CALLBACK, GATE, GATE.replace(before, after))
    for tag, before, after in (
        ('substep interval', 'n_sub * dtau', 'dtau'),
        ('full interval', 'n_sub * dtau', 'slow_dt'),
        ('mutable interval', 'const Real stage_dt', 'Real stage_dt'),
        ('wrong state alias', 'S_data[IntVars::cons]', 'S_old[IntVars::cons]'),
        ('wrong state component count', 'make_alias, 0, 2', 'make_alias, 0, 1'),
        ('wrong x alias', 'S_data[IntVars::xmom]', 'S_data[IntVars::ymom]'),
        ('wrong y alias', 'S_data[IntVars::ymom]', 'S_data[IntVars::xmom]'),
    ):
        yield replace('implicit '+tag, CALLBACK, SETUP, SETUP.replace(before, after))
    yield replace('implicit wrong z alias', CALLBACK, W_SETUP, W_SETUP.replace('IntVars::zmom', 'IntVars::ymom'))
    yield replace('implicit missing z alias', CALLBACK, W_SETUP, '')
    yield body_fixture('extra stage interval alias', CALLBACK, body+'retain(stage_dt);')
    for macro in ('AMREX_USE_GPU', 'ERF_IMPLICIT_W'):
        for branch in (True, False):
            prefix = '\n#ifdef '+macro+'\n'+('' if branch else '#else\n')
            yield replace(macro+f' {branch} hidden deferred include', CALLBACK, include,
                prefix+'auto later = [&] {\n#endif\n'+include+prefix+'};\n#endif\n')
            yield body_fixture(macro+f' {branch} hidden override', CALLBACK,
                prefix+'Gpu::setNoSyncRegion(true);\n#endif\n'+body)
    for tag, code in (
        ('unknown conditional', '#ifdef OTHER\n#endif'),
        ('unknown directive', '#pragma unmodeled'),
        ('duplicate else', '#ifdef AMREX_USE_GPU\n#else\n#else\n#endif'),
        ('else argument', '#ifdef AMREX_USE_GPU\n#else OTHER\n#endif'),
        ('unmatched endif', '#endif'),
        ('unclosed conditional', '#ifdef AMREX_USE_GPU'),
        ('runtime include', '#include "unmodeled.H"'),
        ('macro override', '#define erf_substep_NS other'),
        ('macro undef', '#undef AMREX_USE_GPU'),
        ('line splice', '// hidden \\\n#define erf_substep_NS other'),
        ('raw string', 'R"(hidden)";'),
        ('digraph', '<% unknown(); %>'),
        ('quoted escape', 'Print() << "} // /*"; Gpu::setNoSyncRegion(true);'),
    ):
        yield body_fixture(tag, CALLBACK, '\n'+code+'\n'+body)
    for path in FILES:
        for tag, code in (
            ('NoSyncRegion', 'Gpu::NoSyncRegion region;'),
            ('stream override', 'Gpu::Device::setStreamIndex(2);'),
            ('nonlocal transfer', 'goto outside;'),
            ('macro override', '#define acoustic_substepping other'),
        ):
            yield fixture(Path(path).stem+' '+tag, path, code+'\n'+texts[path])

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
    for macro in ('AMREX_USE_GPU', 'ERF_IMPLICIT_W'):
        yield body_fixture(macro+' harmless conditional', CALLBACK,
            '\n#ifdef '+macro+'\nconst int unused = 1;\n#else\nconst int unused = 2;\n#endif\n'+body, False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv)>1 else Path(tempfile.mkdtemp(prefix='erf-acoustic-completion-mutants-'))
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
