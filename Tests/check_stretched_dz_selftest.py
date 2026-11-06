#!/usr/bin/env python3
"""Source faults and equivalence controls for stretched-dz construction.

Arithmetic faults require named SAT diagnostics; structural faults require an
explicit source-contract rejection. Crashes and unknown results never count as
detection. Source rewrites occur only in scratch copies.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import CONST_H, mutation_outcome
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch_selftest import replace_once
from check_upwindall_z_reach import brace_block
from stretched_dz_source import INTERPOLATION, ZLEVELS

WRAPPER = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'
ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_stretched_dz.py')
SUCCESS = 'proved obligations; stretched-dz source contracts passed.'


def cases(originals):
    yield 'baseline', {}, False, (SUCCESS,)

    def edit(name, function, before, after, required=('FAIL source contract:',), fault=True):
        path = INTERPOLATION if function == 'expand_and_interpolate_1d' else ZLEVELS
        text = originals[path]
        match = re.search(r'\bvoid\s+' + function + r'\s*\([^()]*\)\s*\{', text)
        assert match, function
        start = match.end()-1
        body = brace_block(text, start)
        changed = replace_once(body, before, after)
        new = text[:start+1] + changed + text[start+1+len(body):]
        return name, {path: new}, fault, required if fault else (SUCCESS,)

    update = 'update_stretched_dz'
    validation = 'validate_stretched_dz'
    for before, after in [
        ('nodes.size() >= 2', 'nodes.size() >= 1'),
        ('heights.size() == nodes.size()-1', 'heights.size() <= nodes.size()-1'),
        ('std::isfinite(nodes[0])', 'true'),
        ('std::size_t k = 0', 'std::size_t k = 1'),
        ('k < heights.size()', 'k+1 < heights.size()'),
        ('++k', '--k'),
        ('std::isfinite(nodes[k+1])', 'true'),
        ('nodes[k+1] > nodes[k]', 'nodes[k+1] >= nodes[k]'),
        ('nodes[k+1] > nodes[k]', 'nodes[k+1] > nodes[0]'),
        ('std::isfinite(heights[k])', 'true'),
        ('heights[k] > Real(0)', 'heights[k] >= Real(0)'),
        ('AMREX_ALWAYS_ASSERT_WITH_MESSAGE(std::isfinite(heights[k])',
         'AMREX_ASSERT_WITH_MESSAGE(std::isfinite(heights[k])'),
    ]:
        yield edit('validation rejects ' + after, validation, before, after)
    yield edit('negative refresh level unchecked', update, 'AMREX_ALWAYS_ASSERT(lev >= 0);', '')
    validation_call = 'validate_stretched_dz(zlevels_stag[lev], stretched_dz_h[lev]);'
    for function in (update, 'init_zlevels'):
        for after in ('', validation_call.replace('[lev]', '[0]'),
                      'if (false) { '+validation_call+' }',
                      validation_call.replace('zlevels_stag[lev], stretched_dz_h[lev]',
                                              'stretched_dz_h[lev], zlevels_stag[lev]')):
            yield edit(function+' validation call '+after, function, validation_call, after)
    for before, after, diagnostic in [
        ('zlevels_stag[lev].size()) - 1', 'zlevels_stag[lev].size()) - 2', 'update cell count [sat]'),
        ('stretched_dz_h[lev].resize(nz);', 'stretched_dz_h[lev].resize(nz+1);', 'update host extent [sat]'),
        ('stretched_dz_d[lev].resize(nz);', 'stretched_dz_d[lev].resize(nz-1);', 'update device extent [sat]'),
        ('int k = 0;', 'int k = 1;', 'update exact loop domain [sat]'),
        ('k < nz;', 'k < nz-1;', 'update exact loop domain [sat]'),
        ('k < nz;', 'k <= nz;', 'update exact loop domain [sat]'),
        ('stretched_dz_h[lev][k] =', 'stretched_dz_h[lev][k+1] =', 'update destination [sat]'),
        ('stretched_dz_h[lev][k] =', 'stretched_dz_h[lev][0] =', 'update destination [sat]'),
        ('zlevels_stag[lev][k+1] - zlevels_stag[lev][k]', 'zlevels_stag[lev][k] - zlevels_stag[lev][k+1]', 'update adjacent-level difference [sat]'),
        ('zlevels_stag[lev][k+1] - zlevels_stag[lev][k]', 'zlevels_stag[lev][k+1] + zlevels_stag[lev][k]', 'update adjacent-level difference [sat]'),
        ('zlevels_stag[lev][k+1]', 'zlevels_stag[lev][k+2]', 'update read 0 in initialized range [sat]'),
        ('zlevels_stag[lev][k];', 'zlevels_stag[lev][k] + Real(1)/Real(0);', 'update divisor 0 nonzero [sat]'),
    ]:
        yield edit('update ' + diagnostic + ' / ' + after, update, before, after, (diagnostic,))
    for before, after in [
        ('AMREX_ALWAYS_ASSERT(nz > 0);', ''),
        ('AMREX_ALWAYS_ASSERT(lev < static_cast<int>(zlevels_stag.size()));', ''),
        ('stretched_dz_h[lev].resize(nz);', 'stretched_dz_h[lev].resize(stretched_dz_h[lev].size());'),
        ('static_cast<int>(zlevels_stag[lev].size()) - 1', 'nodes - 1'),
        ('stretched_dz_h[lev][k] =', 'stretched_dz_h[lev+1][k] ='),
        ('zlevels_stag[lev][k+1]', 'zlevels_stag[lev+1][k+1]'),
        ('zlevels_stag[lev][k+1] - zlevels_stag[lev][k]', 'zorig[k+1] - zorig[k]'),
        ('k++)', 'k--)'),
        ('stretched_dz_d[lev].resize(nz);', 'return; stretched_dz_d[lev].resize(nz);'),
        ('stretched_dz_d[lev].resize(nz);', 'stretched_dz_h[lev].clear(); stretched_dz_d[lev].resize(nz);'),
    ]:
        yield edit('update rejects ' + after, update, before, after)

    # Both entry points must preserve the copy's direction, range, level and
    # destination; no stale or partial host vector may silently pass.
    for function in (update, 'init_zlevels'):
        for before, after in [
            ('Gpu::hostToDevice', 'Gpu::deviceToHost'),
            ('stretched_dz_h[lev].begin()', 'stretched_dz_h[lev].begin()+1'),
            ('stretched_dz_h[lev].end()', 'stretched_dz_h[lev].end()-1'),
            ('stretched_dz_h[lev].end()', 'stretched_dz_h[lev].begin()'),
            ('stretched_dz_d[lev].begin()', 'stretched_dz_d[lev+1].begin()'),
            ('stretched_dz_d[lev].begin()', 'stretched_dz_h[lev].begin()'),
            ('Gpu::copy(Gpu::hostToDevice,', 'unknown(stretched_dz_h); Gpu::copy(Gpu::hostToDevice,'),
        ]:
            yield edit(function + ' copy ' + after, function, before, after)
    update_copy = re.search(r'    stretched_dz_d\[lev\]\.resize\(nz\);\n    Gpu::copy\([^;]+;', originals[ZLEVELS])[0]
    yield edit('update copy precedes thickness writes', update,
               '    stretched_dz_h[lev].resize(nz);', update_copy + '\n    stretched_dz_h[lev].resize(nz);')
    yield edit('update extra trailing write', update, update_copy, update_copy + '\n    stretched_dz_h[lev].clear();')

    init = 'init_zlevels'
    for before, after, diagnostic in [
        ('zlevels_stag[lev].resize(nz+1);', 'zlevels_stag[lev].resize(nz);', 'constructor nodal extent [sat]'),
        ('stretched_dz_h[lev].resize(domain.length(2));', 'stretched_dz_h[lev].resize(domain.length(2)+1);', 'constructor host extent [sat]'),
        ('stretched_dz_d[lev].resize(stretched_dz_h[lev].size());', 'stretched_dz_d[lev].resize(stretched_dz_h[lev].size()-1);', 'constructor device extent [sat]'),
        ('zlevels_stag[lev][k] = k * dx[2];', 'zlevels_stag[lev][k] = (k+1) * dx[2];', 'uniform node coordinate [sat]'),
        ('stretched_dz_h[lev][k] = dx[2];', 'stretched_dz_h[lev][k] = -dx[2];', 'uniform positive for positive Geometry dz [sat]'),
        ('stretched_dz_h[lev][k] = dx[2];', 'stretched_dz_h[lev][k+1] = dx[2];', 'uniform thickness destination [sat]'),
        ('stretched_dz_h[lev][k-1] = zlevels_stag[lev][k] - zlevels_stag[lev][k-1];', 'stretched_dz_h[lev][k] = zlevels_stag[lev][k] - zlevels_stag[lev][k-1];', 'generated destination [sat]'),
        ('stretched_dz_h[lev][k-1] = zlevels_stag[lev][k] - zlevels_stag[lev][k-1];', 'stretched_dz_h[lev][k-1] = zlevels_stag[lev][k+1] - zlevels_stag[lev][k];', 'generated read 0 in initialized range [sat]'),
        ('Real dz = dz0;', 'Real dz = -dz0;', 'geometric positive induction base [sat]'),
        ('stretched_dz_h[lev][k-1] = dz;', 'stretched_dz_h[lev][k-1] = 2*dz;', 'geometric stored current thickness [sat]'),
        ('zlevels_stag[lev][k-1] + dz;', 'zlevels_stag[lev][k-1] - dz;', 'geometric adjacent-level difference [sat]'),
        ('dz *= grid_stretching_ratio;', 'dz *= Real(0);', 'geometric positive induction step [sat]'),
        ('k < nz_zlevs-1;', 'k < nz_zlevs-2;', 'override exact loop domain [sat]'),
        ('for (int lev = 1; lev <= max_level; lev++)', 'for (int lev = 2; lev <= max_level; lev++)', 'override_refine_levels exact loop domain [sat]'),
    ]:
        yield edit('init ' + diagnostic, init, before, after, (diagnostic,))
    for before, after in [
        ('grid_stretching_ratio == 0 && !(dz0 > 0)', 'grid_stretching_ratio == 0'),
        ('else if (lev == 0 && grid_stretching_ratio == 0)', 'else if (lev == 0 && grid_stretching_ratio > 0)'),
        ('auto dx = geom[lev].CellSizeArray();', 'auto dx = geom[lev].InvCellSizeArray();'),
        ('int nz = domain.length(2);', 'int nz = domain.length(1);'),
        ('zlevels_stag[lev][k] += zsurf;', 'zlevels_stag[lev][k] -= zsurf;'),
        ('if (k > 0)', 'if (k >= 0)'),
        ('Real dz = dz0;', 'Real dz = dz-dz+dz0;'),
        ('build_geometric_zlevels(zlevels_stag[lev], dz0, depth, tol)', 'build_geometric_zlevels(zlevels_stag[lev], depth, dz0, tol)'),
        ('AMREX_ALWAYS_ASSERT_WITH_MESSAGE(gs.converged,', 'AMREX_ALWAYS_ASSERT_WITH_MESSAGE(true,'),
        ('if (n_zlevels != nz+1)', 'if (n_zlevels != nz)'),
        ('pp.getarr("terrain_z_levels", zlevels_stag[0], 0, nz+1);', 'pp.getarr("terrain_z_levels", zlevels_stag[0], 1, nz+1);'),
        ('if (n_zlevels > 0)', 'if (n_zlevels < 0)'),
        ('stretched_dz_d[lev].resize(stretched_dz_h[lev].size());', 'stretched_dz_d[lev].resize(nz);'),
        ('zlevels_stag[lev][k] = k * dx[2];', 'zlevels_stag[lev][k] = k * dx[1];'),
    ]:
        yield edit('init rejects ' + after, init, before, after)
    refined = '''else if (lev > 0) {
            int rr = ref_ratio[lev-1][2];
            expand_and_interpolate_1d(zlevels_stag[lev], zlevels_stag[lev-1], rr, false);'''
    for before, after in [('[lev-1][2]', '[lev-1][1]'), ('rr, false', 'rr, true'),
                          ('zlevels_stag[lev-1], rr', 'zlevels_stag[lev], rr')]:
        yield edit('refinement caller ' + after, init, refined, replace_once(refined, before, after))
    copy_start = ('for (int lev = 0; lev <= max_level; lev++) {\n        '
                  + validation_call + '\n        stretched_dz_d[lev].resize')
    yield edit('copy loop misses last level', init, copy_start, copy_start.replace('<= max_level', '< max_level'),
               ('copy_levels exact loop domain [sat]',))
    # A common scaling error preserves consistency but violates the independent
    # geometric construction specification; require both outcomes explicitly.
    text = originals[ZLEVELS].replace('stretched_dz_h[lev][k-1] = dz;', 'stretched_dz_h[lev][k-1] = 2*dz;')
    text = replace_once(text, 'zlevels_stag[lev][k-1] + dz;', 'zlevels_stag[lev][k-1] + 2*dz;')
    yield 'coherent geometric doubling', {ZLEVELS: text}, True, (
        'geometric stored current thickness [sat]', 'geometric adjacent-level difference [unsat]')

    refinement = 'expand_and_interpolate_1d'
    for before, after, diagnostic in [
        ('refine_fac*(Nz-1) + 1;', 'refine_fac*(Nz-1) + 2;', 'refinement output extent [sat]'),
        ('k < Nz;', 'k < Nz-1;', 'refinement coarse nodes exact loop domain [sat]'),
        ('k < Nz-1;', 'k < Nz;', 'refinement coarse cells exact loop domain [sat]'),
        ('int koff=1;', 'int koff=0;', 'refinement interior offsets exact loop domain [sat]'),
        ('koff < refine_fac;', 'koff <= refine_fac;', 'refinement interior offsets exact loop domain [sat]'),
        ('znew_stag[refine_fac*k] =', 'znew_stag[refine_fac*k+1] =', 'refinement coarse-node destination [sat]'),
        ('znew_stag[refine_fac*k] = zorig[k];', 'znew_stag[refine_fac*k] = zorig[k+1];', 'refinement copied node read 0 in initialized range [sat]'),
        ('amrex::Real z1 = zorig[k];', 'amrex::Real z1 = zorig[k+1];', 'refinement left coordinate binding [sat]'),
        ('zorig[k+1] - z1;', 'zorig[k+1] + z1;', 'refinement coarse width [sat]'),
        ('refine_fac*k + koff]', 'refine_fac*k + koff+1]', 'refinement interior destination [sat]'),
        ('amrex::Real(koff)/refine_fac * dz + z1;', 'amrex::Real(koff)/refine_fac * dz - z1;', 'refinement linear interpolation [sat]'),
        ('amrex::Real(koff)/refine_fac * dz + z1;', 'amrex::Real(koff)/(refine_fac-1) * dz + z1;', 'refinement interpolant divisor 0 nonzero [sat]'),
    ]:
        yield edit('refiner ' + diagnostic, refinement, before, after, (diagnostic,))
    for before, after in [
        ('amrex::Real(koff)/refine_fac', 'koff/refine_fac'),
        ('znew = znew_stag;', 'znew = zorig;'),
        ('amrex::Vector<amrex::Real> znew_stag(Nz_refined);', 'amrex::Vector<amrex::Real> znew_stag(Nz_refined-1);'),
        ('znew.size() == Nz_refined);', 'znew.size() == Nz_refined-1);'),
        ('znew = znew_stag;', 'znew = znew_stag; znew.clear();'),
        ('amrex::Real z1 = zorig[k];', 'amrex::Real z1 = zorig[k]; return;'),
        ('amrex::Real z1 = zorig[k];', 'amrex::Real z1 = zlevels_stag[lev][k];'),
    ]:
        yield edit('refiner rejects ' + after, refinement, before, after)
    for before, after, diagnostic in [
        ('auto dz_ptr = stretched_dz_d.data();', 'auto dz_ptr = other.data();', 'FAIL source contract:'),
        ('auto dzInv = one/dz_ptr[k];', 'auto dzInv = one/dz_ptr[k+1];', 'X consumer read 0 in initialized range [sat]'),
        ('two/(dz_ptr[k] + dz_ptr[k-1])', 'two/(dz_ptr[k] - dz_ptr[k-1])', 'Z consumer divisor 0 nonzero [sat]'),
        ('two/(dz_ptr[k] + dz_ptr[k-1])', 'one/(dz_ptr[k] + dz_ptr[k-1])', 'Z consumer inverse nodal layer spacing [sat]'),
    ]:
        text = originals[WRAPPER]
        # X and Y have identical spacing declarations; mutate both deliberately.
        assert before in text
        yield 'consumer ' + after, {WRAPPER: text.replace(before, after)}, True, (diagnostic,)
    for constant in ('one', 'two'):
        text, count = re.subn(r'(constexpr\s+amrex::Real\s+' + constant + r'\s*=)[^;]+;', r'\g<1> 0;', originals[CONST_H])
        assert count == 1
        yield 'source constant ' + constant, {CONST_H: text}, True, ('source constant ' + constant,)

    # Accept algebraically equivalent source rather than matching arithmetic text.
    for function, before, after in [
        (update, 'k < nz;', 'k <= nz-1;'),
        (update, 'k++)', '++k)'),
        (update, 'zlevels_stag[lev][k+1] - zlevels_stag[lev][k]', '-zlevels_stag[lev][k] + zlevels_stag[lev][1+k]'),
        (update, 'stretched_dz_d[lev].resize(nz);', 'stretched_dz_d[lev].resize(stretched_dz_h[lev].size());'),
        (init, 'zlevels_stag[lev][k] = k * dx[2];', 'zlevels_stag[lev][k] = dx[2] * k;'),
        (init, 'stretched_dz_h[lev].resize(domain.length(2));', 'stretched_dz_h[lev].resize(nz);'),
        (init, 'Real dz = dz0;', 'Real dz = Real(dz0);'),
        (init, 'zlevels_stag[lev][k-1] + dz;', 'dz + zlevels_stag[lev][k-1];'),
        (refinement, 'refine_fac*(Nz-1) + 1;', '1 + (Nz-1)*refine_fac;'),
        (refinement, 'znew_stag[refine_fac*k] =', 'znew_stag[k*refine_fac] ='),
        (refinement, 'amrex::Real(koff)/refine_fac * dz + z1;', 'z1 + dz*(amrex::Real(koff)/refine_fac);'),
    ]:
        yield edit('equivalent ' + after, function, before, after, fault=False)
    yield 'comments and whitespace', {path: strip_comments(text).replace(',', ', /* gap */ ').replace(';', ';\n')
                                       for path, text in originals.items()}, False, (SUCCESS,)
    yield edit('logging text is inert', init, '"Building a stretched grid with dz0: "',
               '"not live: dz=0;"', fault=False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-stretched-dz-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (ZLEVELS, INTERPOLATION, WRAPPER, CONST_H)}
    fixtures = list(cases(originals))
    unexpected = faults = controls = 0
    for index, (name, edits, fault, required) in enumerate(fixtures):
        root = work / f'{index:03d}'
        for path, text in originals.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, text))
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        if result.returncode == 77:
            print(result.stdout.strip())
            return 77
        caught, expected, messages = mutation_outcome(result, fault)
        missing = [item for item in required if item not in result.stdout]
        if '[unknown]' in result.stdout:
            missing.append('all solver results must be decided')
        expected = expected and not missing
        unexpected += not expected
        faults += fault
        controls += not fault and name != 'baseline'
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            for message in messages[:3] + ['missing diagnostic: ' + item for item in missing]:
                print('           ' + message)
    print(f'\n{faults} mutations; {controls} equivalent controls; {unexpected} unexpected outcome(s)')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
