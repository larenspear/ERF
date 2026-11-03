"""Read the complete supported geometric solver and nodal builder.

Control flow, types, binding order and side effects are guarded; scalar arithmetic
is retained for exact-real interpretation. This is a narrow, fail-closed reader,
not a C++ frontend. Macro rewriting and changes outside the region are excluded.
"""
import re

from advection_source import Source
from check_mapfactor_storage import strip_comments
from momentum_geometry import literal, unique

GRID = 'Source/Utils/ERF_GridUtils.H'
EXPR = r'([^;{}]+)'
FIELDS = ('dz0', 'ratio', 'residual', 'uniform', 'converged')


def region(text):
    start = unique(r'\bstruct\s+GeometricStretch\s*\{', text, 'GeometricStretch').start()
    end = unique(r'\bstruct\s+InterpolationStencil\s*\{', text, 'InterpolationStencil').start()
    if end <= start:
        raise ValueError('geometric region order changed')
    return text[start:end]


def expression(source, prefix):
    literal(source, prefix)
    return source.take(EXPR + ';')[1]


def fields(source):
    return {name: expression(source, 'gs.' + name + ' =') for name in FIELDS}


def read_program(text):
    source = Source(region(strip_comments(text)))
    model = {}
    literal(source, 'struct GeometricStretch {')
    model['defaults'] = {}
    for name in FIELDS:
        literal(source, ('bool' if name in ('uniform', 'converged') else 'amrex::Real') + ' ' + name + '{')
        model['defaults'][name] = source.take(r'([^;{}]+)\}\s*;')[1]
    literal(source, '}; AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GeometricStretch '
                    'geometric_stretch(amrex::Real dz0, amrex::Real z_top, int nz, amrex::Real tol) noexcept { '
                    'GeometricStretch gs; const amrex::Real Nz = static_cast<amrex::Real>(nz); '
                    'if (nz < 1 || !(z_top > amrex::Real(0.0)) || !(dz0 > amrex::Real(0.0))) { return gs; } '
                    'if (nz == 1 || dz0 >= z_top/Nz) {')
    model['uniform'] = fields(source)
    literal(source, 'return gs; }')
    model['seed'] = expression(source, 'amrex::Real SFact =')
    model['initial_residual'] = expression(source, 'amrex::Real F =')
    literal(source, 'const int max_iter = 50; int iter = 0; '
                    'while (amrex::Math::abs(F) > tol && iter < max_iter) {')
    model['derivative'] = expression(source, 'const amrex::Real dFdSF =')
    model['correction'] = expression(source, 'SFact -=')
    model['minimum'] = expression(source, 'const amrex::Real SFact_min =')
    literal(source, 'SFact = (SFact > SFact_min) ? SFact : SFact_min;')
    model['updated_residual'] = expression(source, 'F =')
    literal(source, '++iter; }')
    model['returned'] = fields(source)
    literal(source, 'return gs; } inline GeometricStretch '
                    'build_geometric_zlevels(amrex::Vector<amrex::Real>& zlevels, '
                    'amrex::Real dz0, amrex::Real z_top, amrex::Real tol) { '
                    'const int nz = static_cast<int>(zlevels.size()) - 1;')
    source.take(r'AMREX_ALWAYS_ASSERT_WITH_MESSAGE\s*\(\s*nz\s*>=\s*1\s*,\s*'
                r'"(?:\\.|[^"\\])*"\s*\)\s*;')
    literal(source, 'const GeometricStretch gs = geometric_stretch(dz0, z_top, nz, tol);')
    model['builder_seed'] = expression(source, 'amrex::Real dz =')
    model['bottom'] = expression(source, 'zlevels[0] =')
    literal(source, 'for (int k(1); k<=nz; ++k) {')
    model['node'] = expression(source, 'zlevels[k] =')
    model['factor'] = expression(source, 'dz *=')
    literal(source, '}')
    model['top'] = expression(source, 'zlevels[nz] =')
    literal(source, 'return gs; }')
    source.done()
    return model
