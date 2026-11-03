#!/usr/bin/env python3
"""Require deliberate geometric-grid faults to fail with explicit diagnostics."""
import re
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
        assert original.count(before) == 1, (name, original.count(before))
        return name, original.replace(before, after), fault, (diagnostic if fault else SUCCESS,)

    # All supported statement sequences are consumed. These are structural
    # failures rather than claims about executing arbitrary mutated C++.
    structural = [
        ('struct field', 'bool uniform{false};', 'bool uniform{false}; int hidden{0};'),
        ('default constructor', 'GeometricStretch gs;', 'GeometricStretch gs{}; gs.dz0 = 1;'),
        ('real count', 'static_cast<amrex::Real>(nz);', 'static_cast<amrex::Real>(nz+1);'),
        ('invalid count', 'nz < 1 ||', 'nz < 0 ||'),
        ('invalid depth', '!(z_top > amrex::Real(0.0))', '!(z_top >= amrex::Real(0.0))'),
        ('invalid thickness', '!(dz0 > amrex::Real(0.0))', '!(dz0 >= amrex::Real(0.0))'),
        ('single layer', 'nz == 1 || dz0 >=', 'nz == 0 || dz0 >='),
        ('fallback condition', 'dz0 >= z_top/Nz', 'dz0 > z_top/Nz'),
        ('iteration limit', 'max_iter = 50;', 'max_iter = 0;'),
        ('initial counter', 'int iter = 0;', 'int iter = 1;'),
        ('loop condition', 'abs(F) > tol && iter < max_iter', 'abs(F) < tol && iter < max_iter'),
        ('iteration bound', 'iter < max_iter)', 'iter <= max_iter)'),
        ('iteration increment', '++iter;', 'iter += 2;'),
        ('correction sign', 'SFact -= F/dFdSF;', 'SFact += F/dFdSF;'),
        ('clamp comparison', '(SFact > SFact_min)', '(SFact < SFact_min)'),
        ('clamp arm', '? SFact : SFact_min;', '? SFact : SFact;'),
        ('extra solver statement', '++iter;', '++iter; dz0 = 0;'),
        ('builder size', 'static_cast<int>(zlevels.size()) - 1;', 'static_cast<int>(zlevels.size()) - 2;'),
        ('builder assertion', 'AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nz >= 1,', 'AMREX_ALWAYS_ASSERT_WITH_MESSAGE(nz >= 0,'),
        ('builder arguments', 'geometric_stretch(dz0, z_top, nz, tol);', 'geometric_stretch(z_top, dz0, nz, tol);'),
        ('builder lower bound', 'for (int k(1); k<=nz; ++k)', 'for (int k(0); k<=nz; ++k)'),
        ('builder upper bound', 'for (int k(1); k<=nz; ++k)', 'for (int k(1); k<nz; ++k)'),
        ('builder increment', 'for (int k(1); k<=nz; ++k)', 'for (int k(1); k<=nz; k+=2)'),
        ('builder destination', 'zlevels[k] =', 'zlevels[k-1] ='),
        ('builder factor operation', 'dz *= gs.ratio;', 'dz += gs.ratio;'),
        ('builder top destination', 'zlevels[nz] = z_top;', 'zlevels[nz-1] = z_top;'),
        ('builder resize', 'zlevels[nz] = z_top;', 'zlevels.resize(nz); zlevels[nz] = z_top;'),
        ('builder extra write', 'zlevels[nz] = z_top;', 'zlevels[nz] = z_top; zlevels[1] = 0;'),
        ('builder order', 'zlevels[k] = zlevels[k-1] + dz;\n        dz *= gs.ratio;',
         'dz *= gs.ratio;\n        zlevels[k] = zlevels[k-1] + dz;'),
        ('builder early return', 'zlevels[nz] = z_top;', 'return gs; zlevels[nz] = z_top;'),
        ('preprocessor', 'amrex::Real dz = gs.dz0;', '#if 0\n amrex::Real dz = gs.dz0;\n#endif'),
        ('side effect in arithmetic', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = (gs.dz0 = 1);'),
        ('out-of-scope seed', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = SFact;'),
        ('uninitialized ratio', 'amrex::Real SFact = amrex::Real(1.03);', 'amrex::Real SFact = SFact;'),
        ('hidden call', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = mutate(gs);'),
        ('unknown power', 'std::pow(SFact,Nz-amrex::Real(1.0))', 'std::pow(SFact,Nz-amrex::Real(2.0))'),
        ('fractional power', 'std::pow(SFact,Nz-amrex::Real(1.0))', 'std::pow(SFact,amrex::Real(2.5))'),
        ('boolean arithmetic', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = true + gs.dz0;'),
        ('boolean unary', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = -true;'),
        ('boolean cast', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = amrex::Real(true);'),
        ('boolean real field', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = true;'),
        ('internal symbol', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = __real(gs.dz0);'),
    ]
    for name, before, after in structural:
        yield edit(name, before, after)

    arithmetic = [
        ('default first', 'dz0{amrex::Real(0.0)}', 'dz0{amrex::Real(1.0)}', 'invalid default dz0 [sat]'),
        ('default ratio', 'ratio{amrex::Real(1.0)}', 'ratio{amrex::Real(0.0)}', 'invalid default ratio [sat]'),
        ('default residual', 'residual{amrex::Real(0.0)}', 'residual{amrex::Real(1.0)}', 'invalid default residual [sat]'),
        ('default uniform', 'bool uniform{false}', 'bool uniform{true}', 'invalid default uniform [sat]'),
        ('default convergence', 'bool converged{false}', 'bool converged{true}', 'invalid default converged [sat]'),
        ('uniform first', 'gs.dz0       = z_top/Nz;', 'gs.dz0       = z_top/(Nz+1);', 'uniform return dz0 [sat]'),
        ('uniform ratio', 'gs.ratio     = amrex::Real(1.0);', 'gs.ratio     = amrex::Real(1.1);', 'uniform return ratio [sat]'),
        ('uniform residual', 'gs.residual  = amrex::Real(0.0);', 'gs.residual  = amrex::Real(1.0);', 'uniform return residual [sat]'),
        ('uniform flag', 'gs.uniform   = true;', 'gs.uniform   = false;', 'uniform return uniform [sat]'),
        ('uniform convergence', 'gs.converged = true;', 'gs.converged = false;', 'uniform return converged [sat]'),
        ('initial ratio value', 'amrex::Real SFact = amrex::Real(1.03);', 'amrex::Real SFact = amrex::Real(1.04);', 'initial ratio [sat]'),
        ('derivative multiplier', 'Nz * std::pow', '(Nz+1) * std::pow', 'analytic derivative [sat]'),
        ('derivative sign', '- std::pow(SFact,Nz) +', '+ std::pow(SFact,Nz) +', 'analytic derivative [sat]'),
        ('derivative denominator', 'std::pow(SFact-amrex::Real(1.0),amrex::Real(2.0))',
         'std::pow(SFact-amrex::Real(1.0),amrex::Real(1.0))', 'analytic derivative [sat]'),
        ('correction divisor', 'SFact -= F/dFdSF;', 'SFact -= F/(dFdSF+1);', 'Newton correction [sat]'),
        ('minimum sign', 'SFact_min = amrex::Real(1.0) + tol;', 'SFact_min = amrex::Real(1.0) - tol;', 'ratio minimum [sat]'),
        ('returned first', 'gs.dz0       = dz0;', 'gs.dz0       = dz0+1;', 'nonuniform return dz0 [sat]'),
        ('returned ratio', 'gs.ratio     = SFact;', 'gs.ratio     = SFact+1;', 'nonuniform return ratio [sat]'),
        ('returned residual', 'gs.residual  = F;', 'gs.residual  = -F;', 'nonuniform return residual [sat]'),
        ('returned uniform', 'gs.uniform   = false;', 'gs.uniform   = true;', 'nonuniform return uniform [sat]'),
        ('returned convergence', 'gs.converged = (amrex::Math::abs(F) <= tol);', 'gs.converged = (F <= tol);', 'nonuniform return converged [sat]'),
        ('builder seed value', 'amrex::Real dz = gs.dz0;', 'amrex::Real dz = 2*gs.dz0;', 'builder thickness seed [sat]'),
        ('builder bottom value', 'zlevels[0] = amrex::Real(0.0);', 'zlevels[0] = amrex::Real(1.0);', 'builder bottom [sat]'),
        ('builder recurrence sign', 'zlevels[k-1] + dz;', 'zlevels[k-1] - dz;', 'builder node recurrence [sat]'),
        ('builder factor value', 'dz *= gs.ratio;', 'dz *= 2*gs.ratio;', 'builder ratio [sat]'),
        ('builder top value', 'zlevels[nz] = z_top;', 'zlevels[nz] = z_top+1;', 'builder forced top [sat]'),
        ('cancelled undefined division', 'zlevels[nz] = z_top;',
         'zlevels[nz] = z_top + amrex::Real(0)/(z_top-z_top);', 'builder forced top divisor 0 nonzero [sat]'),
    ]
    for name, before, after, diagnostic in arithmetic:
        yield edit(name, before, after, diagnostic)
    for prefix, label in [('amrex::Real F     =', 'initial_residual'), ('F      =', 'updated_residual')]:
        match = re.search(re.escape(prefix) + r'[^;]+;', original)
        assert match
        for name, before, after in [
            ('residual sign', '- z_top', '+ z_top'),
            ('sum numerator', 'std::pow(SFact,Nz) - amrex::Real(1.0)', 'std::pow(SFact,Nz) + amrex::Real(1.0)'),
            ('sum denominator', '(SFact - amrex::Real(1.0))', '(SFact + amrex::Real(1.0))'),
        ]:
            yield edit(label+' '+name, match[0], match[0].replace(before, after), label+' [sat]')
    # A changed read is safely rejected; the replay must not crash by indexing
    # a Python list outside the initialized prefix after the symbolic failure.
    yield edit('uninitialized node read', 'zlevels[k-1] + dz;', 'zlevels[k] + dz;',
               'builder node recurrence initialized read [sat]')
    yield edit('negative node read', 'zlevels[k-1] + dz;', 'zlevels[k-2] + dz;',
               'builder node recurrence initialized read [sat]')

    controls = [
        ('uniform quotient', 'gs.dz0       = z_top/Nz;', 'gs.dz0       = (z_top*amrex::Real(1))/Nz;'),
        ('ratio spelling', 'amrex::Real SFact = amrex::Real(1.03);', 'amrex::Real SFact = amrex::Real(103)/amrex::Real(100);'),
        ('derivative product', 'Nz * std::pow(SFact,Nz-amrex::Real(1.0))', 'std::pow(SFact,Nz-amrex::Real(1.0)) * Nz'),
        ('correction', 'SFact -= F/dFdSF;', 'SFact -= (F*amrex::Real(1))/dFdSF;'),
        ('minimum', 'SFact_min = amrex::Real(1.0) + tol;', 'SFact_min = tol + amrex::Real(1);'),
        ('returned ratio', 'gs.ratio     = SFact;', 'gs.ratio     = SFact + amrex::Real(0);'),
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
