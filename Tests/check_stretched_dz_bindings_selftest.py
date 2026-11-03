#!/usr/bin/env python3
"""Mutation checks for the finite spacing ownership/callback contract."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_stretched_dz_bindings import (ADVANCE, CALLBACK, CLASS, CONSTRUCTOR,
        DECLARATIONS, DYCORE, FILES, FRAGMENTS, MRI, RHS)
from check_mapfactor_storage import strip_comments

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT / 'Tests/check_stretched_dz_bindings.py'
SUCCESS = '6 spacing ownership/callback source contracts passed;'


def cases(originals):
    yield 'baseline', {}, False, SUCCESS

    def edit(name, path, before, after, diagnostic='FAIL source contract:', fault=True):
        assert originals[path].count(before) == 1, (name, originals[path].count(before))
        return name, {path: originals[path].replace(before, after)}, fault, diagnostic if fault else SUCCESS

    for name, path, before, after, diagnostic in [
        ('device member is a reference', CLASS, '>> stretched_dz_d;', '>>& stretched_dz_d;', 'stretched_dz_d owning member'),
        ('host member is local', CLASS, 'amrex::Vector<amrex::Vector<amrex::Real>> stretched_dz_h;',
         'void local() { amrex::Vector<amrex::Vector<amrex::Real>> stretched_dz_h; }', 'binding must be at function/class scope'),
        ('conditional owning member', CLASS, 'amrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> stretched_dz_d;',
         '#if 0\namrex::Vector<amrex::Gpu::DeviceVector<amrex::Real>> stretched_dz_d;\n#endif', 'conditional on an unsupported preprocessor scope'),
        ('host/device size mismatch', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);', 'stretched_dz_h.resize(nlevs_max-1);', 'unsupported source'),
        ('node size mismatch', CONSTRUCTOR, 'zlevels_stag.resize(max_level+1);', 'zlevels_stag.resize(max_level);', 'unsupported source'),
        ('level count', CONSTRUCTOR, 'int nlevs_max = max_level + 1;', 'int nlevs_max = max_level;', 'level count'),
        ('level count write', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);', 'nlevs_max = 0; stretched_dz_h.resize(nlevs_max);', 'constructor level count'),
        ('max level write', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);', 'max_level = 0; stretched_dz_h.resize(nlevs_max);', 'constructor level count'),
        ('constructor local device shadow', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);',
         'Vector<Gpu::DeviceVector<Real>> stretched_dz_d; stretched_dz_h.resize(nlevs_max);', 'constructor storage'),
        ('constructor local nodes shadow', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);',
         'Vector<Vector<Real>> zlevels_stag; stretched_dz_h.resize(nlevs_max);', 'constructor nodal ownership'),
        ('constructor count escape', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);',
         'auto escaped = std::ref(nlevs_max); stretched_dz_h.resize(nlevs_max);', 'constructor level count'),
        ('constructor storage escape', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);',
         'auto& other = stretched_dz_d; stretched_dz_h.resize(nlevs_max);', 'constructor storage'),
        ('constructor resize after initialization', CONSTRUCTOR, 'solverChoice.dz0);',
         'solverChoice.dz0); stretched_dz_d[0].clear();', 'constructor storage'),
        ('wrong constructor binding', CONSTRUCTOR, 'init_zlevels(zlevels_stag,', 'init_zlevels(other_nodes,', 'unsupported source'),
        ('conditional initialization', CONSTRUCTOR, 'stretched_dz_h.resize(nlevs_max);',
         'if (max_level > 0) stretched_dz_h.resize(nlevs_max);', 'standalone statement'),
        ('wrong dycore level', ADVANCE, 'advance_dycore(lev, state_old, state_new,', 'advance_dycore(0, state_old, state_new,', 'advance to dycore: argument binding level'),
        ('wrong dycore Geometry', ADVANCE, 'Geom(lev), dt_lev, time);', 'Geom(0), dt_lev, time);', 'advance to dycore: argument binding fine_geom'),
        ('dycore level reference', DYCORE, 'advance_dycore (int level,', 'advance_dycore (int& level,', 'advance to dycore: parameter type level'),
        ('dycore mutable Geometry', DYCORE, 'const Geometry fine_geom,', 'Geometry fine_geom,', 'advance to dycore: parameter type fine_geom'),
        ('wrong integrator level', DYCORE, '*mri_integrator_mem[level];', '*mri_integrator_mem[0];', 'unsupported source'),
        ('integrator copy', DYCORE, 'MRISplitIntegrator<Vector<MultiFab> >& mri_integrator',
         'MRISplitIntegrator<Vector<MultiFab> > mri_integrator', 'integrator reference'),
        ('wrong registered callback', DYCORE, 'set_slow_rhs_pre(slow_rhs_fun_pre);', 'set_slow_rhs_pre(slow_rhs_fun_post);', 'unsupported source'),
        ('advance before registration', DYCORE, 'mri_integrator.set_slow_rhs_pre(slow_rhs_fun_pre);',
         'mri_integrator.advance(state_old, state_new, old_time, dt_advance);\n mri_integrator.set_slow_rhs_pre(slow_rhs_fun_pre);', 'unsupported source'),
        ('callback escapes after advance', DYCORE, 'mri_integrator.advance(state_old, state_new, old_time, dt_advance);',
         'mri_integrator.advance(state_old, state_new, old_time, dt_advance); retain(slow_rhs_fun_pre);', 'dycore callback registration'),
        ('integrator alias escapes', DYCORE, 'mri_integrator.advance(state_old, state_new, old_time, dt_advance);',
         'mri_integrator.advance(state_old, state_new, old_time, dt_advance); retain(mri_integrator);', 'dycore callback registration'),
        ('quoted brace cannot hide later storage write', DYCORE, 'mri_integrator.advance(state_old, state_new, old_time, dt_advance);',
         'mri_integrator.advance(state_old, state_new, old_time, dt_advance); BL_PROFILE("}"); stretched_dz_d[0].clear();', 'unmodeled use of stretched_dz_d'),
        ('guarded callback definition', DYCORE, '#include "ERF_TI_slow_rhs_pre.H"',
         '#if 0\n#include "ERF_TI_slow_rhs_pre.H"\n#endif', 'conditional on an unsupported preprocessor scope'),
        ('value capture', CALLBACK, '[&,zero_d=zero,one_d=one]', '[=,zero_d=zero,one_d=one]', 'unsupported source'),
        ('capture temporary storage', CALLBACK, '[&,zero_d=zero,one_d=one]',
         '[&,stretched_dz_d=Gpu::DeviceVector<Real>(),zero_d=zero,one_d=one]', 'unsupported source'),
        ('wrong stage spacing', CALLBACK, 'stretched_dz_d[level], gradp[level],', 'stretched_dz_d[0], gradp[level],', 'callback to slow RHS: argument binding stretched_dz_d'),
        ('wrong stage level', CALLBACK, 'erf_slow_rhs_pre(level, finest_level, nrk,', 'erf_slow_rhs_pre(0, finest_level, nrk,', 'callback to slow RHS: argument binding level'),
        ('wrong stage number', CALLBACK, 'erf_slow_rhs_pre(level, finest_level, nrk,', 'erf_slow_rhs_pre(level, finest_level, 0,', 'callback to slow RHS: argument binding nrk'),
        ('wrong stage Geometry', CALLBACK, 'Hfx3_EB, fine_geom, solverChoice,', 'Hfx3_EB, geom[0], solverChoice,', 'callback to slow RHS: argument binding geom'),
        ('deferred slow RHS', CALLBACK, 'erf_slow_rhs_pre(level, finest_level, nrk,',
         '#pragma omp task\n erf_slow_rhs_pre(level, finest_level, nrk,', 'unsupported directive before slow RHS invocation'),
        ('conditional slow RHS', CALLBACK, 'erf_slow_rhs_pre(level, finest_level, nrk,',
         'if (nrk == 0) erf_slow_rhs_pre(level, finest_level, nrk,', 'standalone statement'),
        ('copied RHS vector', RHS, 'Gpu::DeviceVector<Real>& stretched_dz_d,', 'Gpu::DeviceVector<Real> stretched_dz_d,', 'declaration/definition type mismatch'),
        ('inconsistent RHS declaration', DECLARATIONS, 'amrex::Gpu::DeviceVector<amrex::Real>& stretched_dz_d,\n                      amrex::Vector<amrex::MultiFab>& gradp,',
         'amrex::Gpu::DeviceVector<amrex::Real> stretched_dz_d,\n                      amrex::Vector<amrex::MultiFab>& gradp,', 'declaration/definition type mismatch'),
        ('setter stores reference to parameter', MRI, 'slow_rhs_pre = F;', 'slow_rhs_pre = [&F](auto&... args) { F(args...); };', 'unsupported source'),
        ('setter empty', MRI, 'slow_rhs_pre = F;', '', 'unsupported source'),
        ('setter conditional', MRI, 'slow_rhs_pre = F;', 'if (false) slow_rhs_pre = F;', 'unsupported source'),
        ('MRI callback escape', MRI, 'slow_rhs_pre = F;', 'slow_rhs_pre = F; retain(slow_rhs_pre);', 'unsupported source'),
        ('disabled setter', MRI, 'void set_slow_rhs_pre (', '#if 0\nvoid set_slow_rhs_pre (', 'conditional on an unsupported preprocessor scope'),
        ('RK3 loop bound', MRI, 'nrk < 3;', 'nrk < 4;', 'MRI callback stage scope changed'),
        ('RK2 loop bound', MRI, 'nrk < 2;', 'nrk < 3;', 'MRI callback stage scope changed'),
        ('conditional MRI invocation', MRI, 'amrex::Real inv_fac = one / static_cast<amrex::Real>(nsubsteps);',
         'slow_rhs_pre = nullptr; amrex::Real inv_fac = one / static_cast<amrex::Real>(nsubsteps);', 'MRI callback use'),
    ]:
        yield edit(name, path, before, after, diagnostic)

    nested = originals[CALLBACK].replace('erf_slow_rhs_pre(level, finest_level, nrk,',
                                        'defer([&] { erf_slow_rhs_pre(level, finest_level, nrk,')
    assert nested != originals[CALLBACK] and nested.count('cloud_chamber_budget.get());') == 1
    nested = nested.replace('cloud_chamber_budget.get());', 'cloud_chamber_budget.get()); });')
    yield 'nested slow RHS', {CALLBACK: nested}, True, 'binding must be at function/class scope'

    # Each complete statement fragment participates in the protected-use audit.
    for path in (DYCORE, *FRAGMENTS):
        anchor = 'const Box& domain = fine_geom.Domain();' if path == DYCORE else None
        for variable in ('stretched_dz_d', 'level', 'fine_geom'):
            statement = {'stretched_dz_d': 'stretched_dz_d[0].clear();',
                         'level': 'int level = 0;', 'fine_geom': 'auto& fine_geom = other_geom;'}[variable]
            if anchor:
                yield edit(Path(path).name+' writes '+variable, path, anchor, statement+'\n'+anchor, 'unmodeled use of '+variable)
            elif path == CALLBACK:
                yield edit('pre callback writes '+variable, path, 'BL_PROFILE("slow_rhs_fun_pre");',
                           'BL_PROFILE("slow_rhs_fun_pre"); '+statement, 'unmodeled use of '+variable)
            else:
                yield Path(path).name+' writes '+variable, {path: originals[path]+'\n'+statement}, True, 'unmodeled use of '+variable
    for name, statement in [('reference escape', 'auto x = std::ref(level);'),
                            ('unknown callee escape', 'retain(level);'),
                            ('postincrement index', 'read(some_array[level++]);'),
                            ('nested storage expression', 'make_mom_sources(stretched_dz_d[level].data());'),
                            ('unknown include', '#include "unmodeled.H"'),
                            ('macro', '#define level 0')]:
        yield edit(name, FRAGMENTS[0], 'auto make_pi_stage =', statement+'\n auto make_pi_stage =')
    base = originals[MRI]
    anchor = 'slow_rhs_pre(*F_slow, S_old, S_new, time, old_time_stage, time_stage, nrk);'
    assert base.count(anchor) == 2
    for index in (0, 1):
        at = [m.start() for m in re.finditer(re.escape(anchor), base)][index]
        for name, after, diagnostic in [
            ('wrong stage', anchor.replace('time_stage, nrk', 'time_stage, 0'), 'unsupported source'),
            ('deferred task', 'defer([&] { '+anchor+' });', 'direct synchronous stage statement'),
            ('preprocessor', '#if 0\n'+anchor+'\n#endif', 'unsupported preprocessor in MRI advance'),
        ]:
            yield f'MRI site {index} '+name, {MRI: base[:at]+after+base[at+len(anchor):]}, True, diagnostic

    # Controls retain the protected bindings while changing names/layout outside
    # them. They are contract-preserving, not whole-program equivalence claims.
    yield 'comments and whitespace', {p: strip_comments(t).replace(',', ', /* gap */ ').replace(';', ';\n')
                                      for p, t in originals.items()}, False, SUCCESS
    for name, path, before, after in [
        ('profile string', CALLBACK, 'BL_PROFILE("slow_rhs_fun_pre");', 'BL_PROFILE("stretched_dz_d.clear(); // /*");'),
        ('unchanged level expression whitespace', CALLBACK, 'stretched_dz_d[level], gradp[level],', 'stretched_dz_d [ level ], gradp[level],'),
        ('qualified RHS parameter', RHS, 'Gpu::DeviceVector<Real>& stretched_dz_d,', 'amrex::Gpu::DeviceVector<amrex::Real>& stretched_dz_d,'),
        ('unrelated callback arithmetic', CALLBACK, 'double slow_dt = new_stage_time - old_step_time;', 'double slow_dt = -old_step_time + new_stage_time;'),
        ('unrelated local in constructor', CONSTRUCTOR, 'int nlevs_max = max_level + 1;', 'int unrelated = 0; int nlevs_max = max_level + 1;'),
        ('unrelated callback local', CALLBACK, 'BL_PROFILE("slow_rhs_fun_pre");', 'BL_PROFILE("slow_rhs_fun_pre"); const int unrelated = 0;'),
        ('unrelated MRI local', MRI, 'timestep = time_step;', 'timestep = time_step; const int unrelated = 0;'),
        ('quoted callback brace', CALLBACK, 'BL_PROFILE("slow_rhs_fun_pre");', 'BL_PROFILE("} harmless literal {");'),
        ('quoted MRI brace', MRI, 'BL_PROFILE_REGION("MRI_advance");', 'BL_PROFILE_REGION("} harmless literal {");'),
    ]:
        yield edit(name, path, before, after, fault=False)
    # A coordinated signature/call permutation retains the spacing binding.
    edits = {}
    for path, before, after in [
        (RHS, 'Gpu::DeviceVector<Real>& stretched_dz_d,\n                       Vector<MultiFab>& gradp,',
         'Vector<MultiFab>& gradp,\n                       Gpu::DeviceVector<Real>& stretched_dz_d,'),
        (DECLARATIONS, 'amrex::Gpu::DeviceVector<amrex::Real>& stretched_dz_d,\n                      amrex::Vector<amrex::MultiFab>& gradp,',
         'amrex::Vector<amrex::MultiFab>& gradp,\n                      amrex::Gpu::DeviceVector<amrex::Real>& stretched_dz_d,'),
        (CALLBACK, 'stretched_dz_d[level], gradp[level],', 'gradp[level], stretched_dz_d[level],')]:
        assert originals[path].count(before) == 1
        edits[path] = originals[path].replace(before, after)
    yield 'coordinated parameter permutation', edits, False, SUCCESS


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-bindings-mutants-'))
    originals = {path: (ROOT/path).read_text() for path in FILES}
    fixtures = list(cases(originals))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, edits, fault, required) in enumerate(fixtures):
        root = work / f'{index:03d}'
        for path, text in originals.items():
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, text))
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught, expected, messages = mutation_outcome(result, fault)
        expected = expected and required in result.stdout
        faults += fault
        controls += not fault and name != 'baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
            print('required: '+required)
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
