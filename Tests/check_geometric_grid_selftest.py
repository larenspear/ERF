#!/usr/bin/env python3
"""Require deliberate geometric-grid faults to fail with explicit diagnostics."""
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from geometric_grid_source import GRID

ERF = Path(__file__).resolve().parent.parent
VERIFY = ERF / 'Tests/check_geometric_grid.py'
SUCCESS = 'geometric-grid conditional contracts passed.'


def cases(original):
    yield 'baseline', original, False, (SUCCESS,)

    def edit(name, before, after, diagnostic='FAIL source contract:', fault=True):
        assert original.count(before) >= 1, (name, original.count(before))
        if original.count(before) > 1:
            assert before in ('gs.dz0       = dz0;', 'gs.uniform   = false;'), (name, original.count(before))
            location = original.rindex(before)
            mutated = original[:location] + after + original[location+len(before):]
        else:
            mutated = original.replace(before, after)
        return name, mutated, fault, (diagnostic if fault else SUCCESS,)

    structural = [
        ('struct field', 'bool uniform{false};', 'bool uniform{false}; int hidden{0};'),
        ('extra solver side effect', 'GeometricStretch gs;', 'GeometricStretch gs; dz0 = 0;'),
        ('depth count', 'for (int k = 1; k < nz; ++k)', 'for (int k = 1; k <= nz; ++k)'),
        ('depth step count', 'for (int k = 1; k < nz; ++k)', 'for (int k = 1; k < nz; k+=2)'),
        ('real count', 'static_cast<amrex::Real>(nz);', 'static_cast<amrex::Real>(nz+1);'),
        ('invalid count', 'nz < 1 ||', 'nz < 0 ||'),
        ('invalid depth', '!(z_top > amrex::Real(0.0))', '!(z_top >= amrex::Real(0.0))'),
        ('invalid first', '!(dz0 > amrex::Real(0.0))', '!(dz0 >= amrex::Real(0.0))'),
        ('invalid tolerance', '!(tol > amrex::Real(0.0))', '!(tol >= amrex::Real(0.0))'),
        ('finite depth', '|| !(z_top <= largest)', ''),
        ('finite first', '|| !(dz0 <= largest)', ''),
        ('finite tolerance', '!(tol <= largest)', 'false'),
        ('single layer', 'nz == 1 || dz0 >=', 'nz == 0 || dz0 >='),
        ('fallback condition', 'dz0 >= z_top/Nz', 'dz0 > z_top/Nz'),
        ('lower residual binding', 'geometric_depth(dz0, lower, nz) - z_top;', 'geometric_depth(dz0, upper, nz) - z_top;'),
        ('bracket bound', 'const int bracket_limit = std::numeric_limits<amrex::Real>::max_exponent;', 'const int bracket_limit = 0;'),
        ('bracket sign', 'upper_residual < amrex::Real(0.0) &&', 'upper_residual > amrex::Real(0.0) &&'),
        ('bracket iteration bound', 'iter < bracket_limit;', 'iter <= bracket_limit;'),
        ('bracket lower update', 'lower = upper;', 'lower = amrex::Real(1);'),
        ('bracket residual update', 'lower_residual = upper_residual;', 'lower_residual = amrex::Real(0);'),
        ('overflow guard', 'upper <= largest/amrex::Real(2.0)', 'upper <= largest'),
        ('bracket growth', '? upper*amrex::Real(2.0) : largest', '? upper*amrex::Real(1.0) : largest'),
        ('lower-bound termination', 'if (lower_residual >= amrex::Real(0.0))', 'if (lower_residual <= amrex::Real(0.0))'),
        ('lower bracket validity', '!(lower_residual <= amrex::Real(0.0))', '!(lower_residual >= amrex::Real(0.0))'),
        ('upper bracket validity', '!(upper_residual >= amrex::Real(0.0))', '!(upper_residual <= amrex::Real(0.0))'),
        ('solve bound', 'std::numeric_limits<amrex::Real>::digits;', '0;'),
        ('solve iteration bound', 'iter < solve_limit;', 'iter <= solve_limit;'),
        ('adjacency test', 'ratio == lower || ratio == upper', 'ratio != lower || ratio == upper'),
        ('endpoint comparison', 'abs(lower_residual) <= amrex::Math::abs(upper_residual)', 'abs(lower_residual) >= amrex::Math::abs(upper_residual)'),
        ('endpoint choice', '? lower : upper;', '? upper : lower;'),
        ('candidate binding', 'geometric_depth(dz0, ratio, nz) - z_top;', 'geometric_depth(dz0, lower, nz) - z_top;'),
        ('acceptance sign', 'abs(residual) <= target || adjacent', 'abs(residual) >= target || adjacent'),
        ('absolute acceptance', 'amrex::Math::abs(residual) <= target', 'residual <= target'),
        ('negative branch', 'if (residual < amrex::Real(0.0))', 'if (residual > amrex::Real(0.0))'),
        ('negative ratio update', 'lower = ratio;', 'upper = ratio;'),
        ('negative residual update', 'lower_residual = residual;', 'upper_residual = residual;'),
        ('positive ratio update', 'upper = ratio;', 'lower = ratio;'),
        ('positive residual update', 'upper_residual = residual;', 'lower_residual = residual;'),
        ('builder size', 'static_cast<int>(zlevels.size()) - 1;', 'static_cast<int>(zlevels.size()) - 2;'),
        ('builder assertion', 'AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nz >= 1,', 'AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nz >= 0,'),
        ('builder argument binding', 'geometric_stretch(dz0, z_top, nz, tol);', 'geometric_stretch(z_top, dz0, nz, tol);'),
        ('failed solve guard', 'if (!gs.converged) { return gs; }', ''),
        ('builder lower bound', 'for (int k(1); k<=nz; ++k)', 'for (int k(0); k<=nz; ++k)'),
        ('builder upper bound', 'for (int k(1); k<=nz; ++k)', 'for (int k(1); k<nz; ++k)'),
        ('builder destination', 'zlevels[k] =', 'zlevels[k-1] ='),
        ('builder product', 'dz *= gs.ratio;', 'dz += gs.ratio;'),
        ('builder top destination', 'zlevels[nz] = z_top;', 'zlevels[nz-1] = z_top;'),
        ('guard loop bound', 'for (int k = 1; k <= nz; ++k)', 'for (int k = 1; k < nz; ++k)'),
        ('guard monotonicity', 'zlevels[k] > zlevels[k-1]', 'zlevels[k] >= zlevels[k-1]'),
        ('guard finiteness', '!(zlevels[k] <= std::numeric_limits<amrex::Real>::max())', 'false'),
        ('guard result', 'gs.converged = false;', 'gs.converged = true;'),
    ]
    for name, before, after in structural:
        yield edit(name, before, after)
    arithmetic = [
        ('default first', 'dz0{amrex::Real(0.0)}', 'dz0{amrex::Real(1.0)}', 'invalid default dz0 [sat]'),
        ('default ratio', 'ratio{amrex::Real(1.0)}', 'ratio{amrex::Real(0.0)}', 'invalid default ratio [sat]'),
        ('default residual', 'residual{amrex::Real(0.0)}', 'residual{amrex::Real(1.0)}', 'invalid default residual [sat]'),
        ('default uniform', 'bool uniform{false}', 'bool uniform{true}', 'invalid default uniform [sat]'),
        ('default convergence', 'bool converged{false}', 'bool converged{true}', 'invalid default converged [sat]'),
        ('depth seed', 'amrex::Real depth = dz0;', 'amrex::Real depth = 2*dz0;', 'depth seed [sat]'),
        ('depth recurrence sign', 'depth * ratio + dz0;', 'depth * ratio - dz0;', 'Horner recurrence [sat]'),
        ('depth recurrence factor', 'depth * ratio + dz0;', 'depth * (ratio+1) + dz0;', 'Horner recurrence [sat]'),
        ('uniform first', 'gs.dz0       = z_top/Nz;', 'gs.dz0       = z_top/(Nz+1);', 'uniform return dz0 [sat]'),
        ('uniform ratio', 'gs.ratio     = amrex::Real(1.0);', 'gs.ratio     = amrex::Real(1.1);', 'uniform return ratio [sat]'),
        ('uniform residual', 'gs.residual  = amrex::Real(0.0);', 'gs.residual  = amrex::Real(1.0);', 'uniform return residual [sat]'),
        ('uniform flag', 'gs.uniform   = true;', 'gs.uniform   = false;', 'uniform return uniform [sat]'),
        ('uniform positivity', 'gs.converged = (gs.dz0 > amrex::Real(0.0));', 'gs.converged = (gs.dz0 >= amrex::Real(0.0));', 'uniform return converged [sat]'),
        ('target cap', 'dz0/amrex::Real(2.0)', 'dz0*amrex::Real(2.0)', 'capped tolerance [sat]'),
        ('initial lower', 'amrex::Real lower = amrex::Real(1.0);', 'amrex::Real lower = amrex::Real(0.0);', 'initial lower ratio [sat]'),
        ('initial upper', 'amrex::Real upper = amrex::Real(2.0);', 'amrex::Real upper = amrex::Real(1.0);', 'initial upper ratio [sat]'),
        ('midpoint divisor', '(upper-lower)/amrex::Real(2.0)', '(upper-lower)/amrex::Real(3.0)', 'bracket midpoint [sat]'),
        ('lower returned first', 'gs.dz0       = dz0;\n        gs.ratio     = lower;', 'gs.dz0       = dz0+1;\n        gs.ratio     = lower;', 'lower-bound return dz0 [sat]'),
        ('lower returned flag', 'gs.uniform   = false;\n        gs.converged = (amrex::Math::abs(lower_residual)', 'gs.uniform   = true;\n        gs.converged = (amrex::Math::abs(lower_residual)', 'lower-bound return uniform [sat]'),
        ('lower returned ratio', 'gs.ratio     = lower;', 'gs.ratio     = upper;', 'FAIL source contract:'),
        ('lower returned residual', 'gs.residual  = lower_residual;', 'gs.residual  = -lower_residual;', 'lower-bound return residual [sat]'),
        ('lower returned finite', 'gs.converged = (amrex::Math::abs(lower_residual) <= largest);', 'gs.converged = true;', 'lower-bound return converged [sat]'),
        ('returned first', 'gs.dz0       = dz0;', 'gs.dz0       = dz0+1;', 'nonuniform return dz0 [sat]'),
        ('returned ratio', 'gs.ratio     = ratio;', 'gs.ratio     = ratio+1;', 'nonuniform return ratio [sat]'),
        ('returned residual', 'gs.residual  = residual;', 'gs.residual  = -residual;', 'nonuniform return residual [sat]'),
        ('returned flag', 'gs.uniform   = false;', 'gs.uniform   = true;', 'nonuniform return uniform [sat]'),
        ('returned finite', 'gs.converged = (amrex::Math::abs(residual) <= largest);', 'gs.converged = (residual <= largest);', 'nonuniform return converged [sat]'),
        ('builder seed value', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = 2*gs.dz0;', 'builder thickness seed [sat]'),
        ('builder bottom', 'zlevels[0] = amrex::Real(0.0);', 'zlevels[0] = amrex::Real(1.0);', 'builder bottom [sat]'),
        ('builder recurrence', 'zlevels[k-1] + dz;', 'zlevels[k-1] - dz;', 'builder node recurrence [sat]'),
        ('builder factor', 'dz *= gs.ratio;', 'dz *= 2*gs.ratio;', 'builder ratio [sat]'),
        ('builder top', 'zlevels[nz] = z_top;', 'zlevels[nz] = z_top+1;', 'builder forced top [sat]'),
    ]
    for name, before, after, diagnostic in arithmetic:
        yield edit(name, before, after, diagnostic)
    yield edit('uninitialized node read', 'zlevels[k-1] + dz;', 'zlevels[k] + dz;',
               'builder node recurrence initialized read [sat]')
    yield edit('negative node read', 'zlevels[k-1] + dz;', 'zlevels[k-2] + dz;',
               'builder node recurrence initialized read [sat]')
    controls = [
        ('uniform quotient', 'gs.dz0       = z_top/Nz;', 'gs.dz0       = (z_top*amrex::Real(1))/Nz;'),
        ('depth product', 'depth * ratio + dz0;', 'dz0 + ratio*depth;'),
        ('depth seed', 'amrex::Real depth = dz0;', 'amrex::Real depth = amrex::Real(dz0);'),
        ('target cap', 'dz0/amrex::Real(2.0)', 'amrex::Real(0.5)*dz0'),
        ('midpoint', 'lower + (upper-lower)/amrex::Real(2.0)', 'lower + amrex::Real(0.5)*(upper-lower)'),
        ('returned ratio', 'gs.ratio     = ratio;', 'gs.ratio     = ratio + amrex::Real(0);'),
        ('builder seed', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = amrex::Real(gs.dz0);'),
        ('node sum', 'zlevels[k-1] + dz;', 'dz + zlevels[k-1];'),
        ('node index', 'zlevels[k-1] + dz;', 'zlevels[k+(-1)] + dz;'),
        ('builder top', 'zlevels[nz] = z_top;', 'zlevels[nz] = amrex::Real(1)*z_top;'),
        ('assertion string', '"build_geometric_zlevels needs a zlevels array of at least two entries"',
         '"comment markers // and /* are inert; gs.dz0=0;"'),
    ]
    for name, before, after in controls:
        yield edit('equivalent '+name, before, after, fault=False)
    yield 'comments and whitespace', strip_comments(original).replace(';', ';\n').replace(',', ', /* gap */ '), False, (SUCCESS,)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-geometric-grid-mutants-'))
    original = (ERF / GRID).read_text()
    fixtures = list(cases(original))
    faults = controls = unexpected = 0
    for index, (name, text, fault, required) in enumerate(fixtures):
        root = work / f'{index:03d}'
        target = root / GRID
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
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
        faults += fault
        controls += not fault and name != 'baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            for message in messages[:3] + ['missing diagnostic: '+item for item in missing]:
                print('           '+message)
    print(f'\n{faults} mutations; {controls} equivalent controls; {unexpected} unexpected outcome(s)')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
