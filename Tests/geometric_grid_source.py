"""Fail-closed reader for the bounded geometric solver and stored-node guard.

This consumes the complete struct, polynomial evaluator, solver and builder.
Scalar expressions below are interpreted over exact reals by the checker;
floating-point adjacency, finiteness and bounded-loop control are source guards,
not a C++ frontend or a proof of IEEE convergence.
"""
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
    literal(source, '}; AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real '
                    'geometric_depth(amrex::Real dz0, amrex::Real ratio, int nz) noexcept {')
    model['depth_seed'] = expression(source, 'amrex::Real depth =')
    literal(source, 'for (int k = 1; k < nz; ++k) {')
    model['depth_step'] = expression(source, 'depth =')
    literal(source, '} return depth; } AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE GeometricStretch '
                    'geometric_stretch(amrex::Real dz0, amrex::Real z_top, int nz, amrex::Real tol) noexcept { '
                    'GeometricStretch gs; const amrex::Real largest = std::numeric_limits<amrex::Real>::max(); '
                    'const amrex::Real Nz = static_cast<amrex::Real>(nz); '
                    'if (nz < 1 || !(z_top > amrex::Real(0.0)) || !(dz0 > amrex::Real(0.0)) || '
                    '!(tol > amrex::Real(0.0)) || !(z_top <= largest) || !(dz0 <= largest) || '
                    '!(tol <= largest)) { return gs; } '
                    'if (nz == 1 || dz0 >= z_top/Nz) {')
    model['uniform'] = fields(source)
    literal(source, 'return gs; }')
    model['target'] = expression(source, 'const amrex::Real target =')
    model['lower'] = expression(source, 'amrex::Real lower =')
    model['upper'] = expression(source, 'amrex::Real upper =')
    literal(source, 'amrex::Real lower_residual = geometric_depth(dz0, lower, nz) - z_top; '
                    'if (lower_residual >= amrex::Real(0.0)) {')
    model['lower_return'] = fields(source)
    literal(source, 'return gs; } '
                    'amrex::Real upper_residual = geometric_depth(dz0, upper, nz) - z_top; '
                    'const int bracket_limit = std::numeric_limits<amrex::Real>::max_exponent; '
                    'for (int iter = 0; upper_residual < amrex::Real(0.0) && iter < bracket_limit; ++iter) { '
                    'lower = upper; lower_residual = upper_residual; '
                    'upper = (upper <= largest/amrex::Real(2.0)) ? upper*amrex::Real(2.0) : largest; '
                    'upper_residual = geometric_depth(dz0, upper, nz) - z_top; } '
                    'if (!(lower_residual <= amrex::Real(0.0)) || !(upper_residual >= amrex::Real(0.0))) { '
                    'return gs; } '
                    'const int solve_limit = std::numeric_limits<amrex::Real>::max_exponent + '
                    'std::numeric_limits<amrex::Real>::digits; '
                    'for (int iter = 0; iter < solve_limit; ++iter) {')
    model['midpoint'] = expression(source, 'amrex::Real ratio =')
    literal(source, 'const bool adjacent = (ratio == lower || ratio == upper); '
                    'if (adjacent) { ratio = (amrex::Math::abs(lower_residual) <= '
                    'amrex::Math::abs(upper_residual)) ? lower : upper; } '
                    'const amrex::Real residual = geometric_depth(dz0, ratio, nz) - z_top; '
                    'if (amrex::Math::abs(residual) <= target || adjacent) {')
    model['returned'] = fields(source)
    literal(source, 'return gs; } if (residual < amrex::Real(0.0)) { '
                    'lower = ratio; lower_residual = residual; '
                    '} else { upper = ratio; upper_residual = residual; } } return gs; } '
                    'inline GeometricStretch build_geometric_zlevels(amrex::Vector<amrex::Real>& zlevels, '
                    'amrex::Real dz0, amrex::Real z_top, amrex::Real tol) { '
                    'const int nz = static_cast<int>(zlevels.size()) - 1;')
    source.take(r'AMREX_ALWAYS_ASSERT_WITH_MESSAGE\s*\(\s*nz\s*>=\s*1\s*,\s*'
                r'"(?:\\.|[^"\\])*"\s*\)\s*;')
    literal(source, 'GeometricStretch gs = geometric_stretch(dz0, z_top, nz, tol); '
                    'if (!gs.converged) { return gs; }')
    model['builder_seed'] = expression(source, 'amrex::Real dz =')
    model['bottom'] = expression(source, 'zlevels[0] =')
    literal(source, 'for (int k(1); k<=nz; ++k) {')
    model['node'] = expression(source, 'zlevels[k] =')
    model['factor'] = expression(source, 'dz *=')
    literal(source, '}')
    model['top'] = expression(source, 'zlevels[nz] =')
    literal(source, 'for (int k = 1; k <= nz; ++k) { '
                    'if (!(zlevels[k] > zlevels[k-1]) || '
                    '!(zlevels[k] <= std::numeric_limits<amrex::Real>::max())) { '
                    'gs.converged = false; } } return gs; }')
    source.done()
    return model
