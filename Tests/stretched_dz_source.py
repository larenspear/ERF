"""Consume supported z-level constructors and retain their arithmetic for Z3.

This reader rejects leftover statements. Logging accepts only literal strings
and selected pure scalar/array reads. The geometric-grid solver is an external
contract; its call, arguments and convergence assertion are guarded here.
"""
import re
from dataclasses import dataclass

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from momentum_geometry import definition, literal

ZLEVELS = 'Source/Utils/ERF_InitZLevels.cpp'
INTERPOLATION = 'Source/Utils/ERF_Interpolation_1D.H'
STRING = r'"(?:\\.|[^"\\])*"'
EXPR = r'([^;{}]+)'


@dataclass
class Loop:
    start: str
    comparison: str
    stop: str


def loop(source, name='k'):
    start, comparison, stop = source.take(
        rf'for\s*\(\s*int\s+{name}\s*=\s*([^;{{}}]+);\s*{name}\s*(<=|<)\s*([^;{{}}]+);'
        rf'\s*(?:{name}\s*\+\+|\+\+\s*{name})\s*\)\s*\{{').groups()
    return Loop(start, comparison, stop)


def scalar(source, declaration):
    literal(source, declaration + ' =')
    return source.take(EXPR + ';')[1]


def assignment(source, array, level='lev'):
    literal(source, array + f'[{level}]')
    index, expression = source.take(r'\[\s*([^\[\];{}]+)\s*\]\s*=\s*' + EXPR + ';').groups()
    return index, expression


def resize(source, array):
    literal(source, array + '[lev].resize(')
    return source.take(r'([^;{}]+)\)\s*;')[1]


def copy(source):
    literal(source, 'Gpu::copy(Gpu::hostToDevice, stretched_dz_h[lev].begin(), '
                    'stretched_dz_h[lev].end(), stretched_dz_d[lev].begin());')


def prints(source):
    operand = (STRING + r'|std\s*::\s*endl|zlevels_stag\s*\[\s*lev\s*\]\s*\[\s*(?:k|1)\s*\]'
               r'|gs\s*\.\s*(?:dz0|ratio)|nz\s*\+\s*1|dz0|zsurf|lev|n_zlevels')
    while re.match(r'\s*Print\s*\(', source.text[source.pos:]):
        source.take(r'Print\s*\(\s*\)\s*(?:<<\s*(?:' + operand + r')\s*)+;')


def signature(text, name, expected):
    names, types, body = definition(text, name)
    require(names == list(expected) and types == expected, name + ': input signature changed')
    return Source(body)


def read_update(text):
    source = signature(text, 'update_stretched_dz', {
        'lev': 'int', 'zlevels_stag': 'Vector<Vector<Real>>const&',
        'stretched_dz_h': 'Vector<Vector<Real>>&',
        'stretched_dz_d': 'Vector<Gpu::DeviceVector<Real>>&'})
    literal(source, 'AMREX_ALWAYS_ASSERT(lev >= 0);')
    for name in ('zlevels_stag', 'stretched_dz_h', 'stretched_dz_d'):
        literal(source, f'AMREX_ALWAYS_ASSERT(lev < static_cast<int>({name}.size()));')
    size = scalar(source, 'const int nz')
    literal(source, 'AMREX_ALWAYS_ASSERT(nz > 0);')
    host_size = resize(source, 'stretched_dz_h')
    bounds = loop(source)
    target, value = assignment(source, 'stretched_dz_h')
    literal(source, '}')
    literal(source, 'validate_stretched_dz(zlevels_stag[lev], stretched_dz_h[lev]);')
    device_size = resize(source, 'stretched_dz_d')
    copy(source)
    source.done()
    return dict(size=size, host_size=host_size, bounds=bounds, target=target,
                value=value, device_size=device_size)


def refinement_call(source):
    literal(source, 'int rr = ref_ratio[lev-1][2];')
    literal(source, 'expand_and_interpolate_1d(zlevels_stag[lev], zlevels_stag[lev-1], rr, false);')


def read_init(text):
    source = signature(text, 'init_zlevels', {
        'zlevels_stag': 'Vector<Vector<Real>>&', 'stretched_dz_h': 'Vector<Vector<Real>>&',
        'stretched_dz_d': 'Vector<Gpu::DeviceVector<Real>>&', 'geom': 'Vector<Geometry>const&',
        'ref_ratio': 'Vector<IntVect>const&', 'grid_stretching_ratio': 'constReal',
        'zsurf': 'constReal', 'dz0': 'constReal'})
    result = {}
    literal(source, 'int max_level = zlevels_stag.size()-1;')
    result['levels'] = loop(source, 'lev')
    literal(source, 'auto dx = geom[lev].CellSizeArray();')
    literal(source, 'const Box& domain = geom[lev].Domain();')
    literal(source, 'int nz = domain.length(2);')
    result['z_size'] = resize(source, 'zlevels_stag')
    result['host_size'] = resize(source, 'stretched_dz_h')
    literal(source, 'if (grid_stretching_ratio == 0 && !(dz0 > 0)) {')
    result['uniform_z_loop'] = loop(source)
    result['uniform_z'] = assignment(source, 'zlevels_stag')
    literal(source, '}')
    result['uniform_h_loop'] = loop(source)
    result['uniform_h'] = assignment(source, 'stretched_dz_h')
    literal(source, '} } else if (lev == 0 && grid_stretching_ratio == 0) {')
    literal(source, '#ifdef AMREX_USE_FLOAT const Real tol = Real(1.e-4); '
                    '#else const Real tol = Real(1.e-8); #endif')
    literal(source, 'const Real depth = geom[lev].ProbHi(2) - zsurf;')
    literal(source, 'const erf_grid_utils::GeometricStretch gs = '
                    'erf_grid_utils::build_geometric_zlevels(zlevels_stag[lev], dz0, depth, tol);')
    source.take(r'AMREX_ALWAYS_ASSERT_WITH_MESSAGE\s*\(\s*gs\.converged\s*,\s*' + STRING + r'\s*\)\s*;')
    literal(source, 'if (gs.uniform) {')
    prints(source)
    literal(source, '}')
    prints(source)
    result['generated_loop'] = loop(source)
    # The source updates current k before differencing; previous k is already
    # shifted by the preceding iteration. Do not reorder these statements.
    literal(source, 'zlevels_stag[lev][k] += zsurf;')
    literal(source, 'if (k > 0) {')
    result['generated_h'] = assignment(source, 'stretched_dz_h')
    prints(source)
    literal(source, '} }')
    prints(source)
    literal(source, '} else if (lev == 0) {')
    literal(source, 'zlevels_stag[lev][0] = zsurf;')
    result['geometric_seed'] = scalar(source, 'Real dz')
    prints(source)
    result['geometric_first_h'] = assignment(source, 'stretched_dz_h')
    result['geometric_first_z'] = assignment(source, 'zlevels_stag')
    prints(source)
    result['geometric_loop'] = loop(source)
    literal(source, 'dz *=')
    result['geometric_factor'] = source.take(EXPR + ';')[1]
    result['geometric_h'] = assignment(source, 'stretched_dz_h')
    result['geometric_z'] = assignment(source, 'zlevels_stag')
    prints(source)
    literal(source, '}')
    prints(source)
    literal(source, '} else if (lev > 0) {')
    refinement_call(source)
    result['refined_loop'] = loop(source)
    result['refined_h'] = assignment(source, 'stretched_dz_h')
    literal(source, '} } }')
    literal(source, 'ParmParse pp("erf");')
    literal(source, 'int n_zlevels = pp.countval("terrain_z_levels");')
    literal(source, 'if (n_zlevels > 0) {')
    literal(source, 'int nz = geom[0].Domain().length(2);')
    literal(source, 'if (n_zlevels != nz+1) {')
    prints(source)
    source.take(r'Abort\s*\(\s*' + STRING + r'\s*\)\s*;')
    literal(source, '}')
    literal(source, 'if (grid_stretching_ratio > 0) {')
    prints(source)
    literal(source, '} else if (dz0 > 0) {')
    prints(source)
    literal(source, '}')
    literal(source, 'pp.getarr("terrain_z_levels", zlevels_stag[0], 0, nz+1);')
    literal(source, 'AMREX_ALWAYS_ASSERT(zlevels_stag[0][0] == 0);')
    result['override_refine_levels'] = loop(source, 'lev')
    refinement_call(source)
    literal(source, '}')
    result['override_levels'] = loop(source, 'lev')
    literal(source, 'int nz_zlevs = zlevels_stag[lev].size();')
    result['override_loop'] = loop(source)
    result['override_h'] = assignment(source, 'stretched_dz_h')
    literal(source, '} } }')
    result['copy_levels'] = loop(source, 'lev')
    literal(source, 'validate_stretched_dz(zlevels_stag[lev], stretched_dz_h[lev]);')
    result['device_size'] = resize(source, 'stretched_dz_d')
    copy(source)
    literal(source, '}')
    source.done()
    return result


def read_refinement(text):
    source = signature(text, 'expand_and_interpolate_1d', {
        'znew': 'Vector<Real>&', 'zorig': 'constVector<Real>&',
        'refine_fac': 'int', 'destag': 'bool'})
    # definition() deliberately accepts no default argument. The caller of
    # this reader removes only the checked '=false' spelling from the signature.
    literal(source, 'const int Nz = zorig.size();')
    result = {'size': scalar(source, 'const int Nz_refined')}
    literal(source, 'if (destag) { AMREX_ALWAYS_ASSERT(znew.size() == Nz_refined-1); '
                    '} else { AMREX_ALWAYS_ASSERT(znew.size() == Nz_refined); }')
    literal(source, 'amrex::Vector<amrex::Real> znew_stag(Nz_refined);')
    result['nodes_loop'] = loop(source)
    literal(source, 'znew_stag')
    result['node_target'], result['node_value'] = source.take(
        r'\[([^\[\];{}]+)\]\s*=\s*' + EXPR + ';').groups()
    literal(source, '} if (refine_fac > 1) {')
    result['cells_loop'] = loop(source)
    result['left'] = scalar(source, 'amrex::Real z1')
    result['width'] = scalar(source, 'amrex::Real dz')
    result['offset_loop'] = loop(source, 'koff')
    literal(source, 'znew_stag')
    result['inner_target'], result['inner_value'] = source.take(
        r'\[([^\[\];{}]+)\]\s*=\s*' + EXPR + ';').groups()
    literal(source, '} } }')
    # Only destag=false is used by init_zlevels. Consume the other branch but
    # do not claim its arithmetic; this guard detects changes to routing.
    literal(source, 'if (destag) { for (int k=0; k < Nz_refined-1; ++k) { '
                    'znew[k] = myhalf*(znew_stag[k] + znew_stag[k+1]); } '
                    '} else { znew = znew_stag; }')
    source.done()
    return result


def read_validation(text):
    source = signature(text, 'validate_stretched_dz', {
        'nodes': 'constVector<Real>&', 'heights': 'constVector<Real>&'})

    def assertion(predicate):
        literal(source, 'AMREX_ALWAYS_ASSERT_WITH_MESSAGE('+predicate+',')
        source.take(STRING + r'\s*\)\s*;')

    assertion('nodes.size() >= 2 && heights.size() == nodes.size()-1')
    assertion('std::isfinite(nodes[0])')
    literal(source, 'for (std::size_t k = 0; k < heights.size(); ++k) {')
    assertion('std::isfinite(nodes[k+1]) && nodes[k+1] > nodes[k]')
    assertion('std::isfinite(heights[k]) && heights[k] > Real(0)')
    literal(source, '}')
    source.done()


def read_program(root):
    text = strip_comments((root / ZLEVELS).read_text())
    read_validation(text)
    interpolation = strip_comments((root / INTERPOLATION).read_text())
    pattern = r'bool\s+destag\s*=\s*false(?=\s*\))'
    require(len(re.findall(pattern, interpolation)) == 1, 'refinement default argument changed')
    interpolation = re.sub(pattern, 'bool destag', interpolation)
    return read_update(text), read_init(text), read_refinement(interpolation)
