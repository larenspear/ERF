"""Guard production geometry inputs to the normal-grid momentum entry points.

These are finite source-binding checks, not proofs of metric generation, storage
validity or control-volume values. AMReX geometry accessors and const_array have
their ordinary meanings. Named external calls listed below must not mutate these
inputs, including through aliases. Macro rewriting, casts defeating constness,
external alias/concurrency effects and the remainder of SlowRhsPre are excluded.
"""
import re

from advection_source import Source
from check_momentum_dispatch import compact, require, split_parameters
from check_upwindall_z_reach import brace_block

MAPFAC_H = 'Source/DataStructs/ERF_DataStruct.H'
METRICS = {f'mf_{grid}{axis}': f'{grid}_{axis}' for grid in 'muv' for axis in 'xy'}

# These external routines must treat the protected geometry/metric arguments as
# read-only. Their other behavior and arguments are not interpreted here.
READ_ONLY_CALLS = (
    'erf_make_tau_terms', 'OmegaFromW', 'AdvectionSrcForRho', 'AdvectionSrcForScalars',
    'EBAdvectionSrcForRho', 'EBAdvectionSrcForScalars', 'DiffusionSrcForState_S',
    'DiffusionSrcForState_T', 'DiffusionSrcForState_EB', 'DiffusionSrcForState_N',
    'erf_cloud_chamber_wall_flux::apply',
)


def literal(source, text):
    tokens = re.findall(r'\w+|::|[^\s]', text)
    source.take(r'\s*'.join(re.escape(token) for token in tokens))


def unique(pattern, text, description):
    matches = list(re.finditer(pattern, text, re.S))
    require(len(matches) == 1, f'{description}: expected exactly one supported occurrence')
    return matches[0]


def definition(text, name):
    match = unique(r'\bvoid\s+' + name + r'\s*\(([^()]*)\)\s*\{', text, name)
    parameters = split_parameters(match[1])
    names, types = [], {}
    for parameter in parameters:
        binding = re.fullmatch(r'(.+?)\b(\w+)\s*', parameter, re.S)
        require(binding is not None and binding[2] not in types, 'unsupported/duplicate parameter')
        names.append(binding[2])
        types[binding[2]] = compact(binding[1]).replace('amrex::', '')
    return names, types, brace_block(text, match.end() - 1)


def balanced_region(text):
    """Require a complete statement region, not a net-zero scope count."""
    text = re.sub(r'"(?:\\.|[^"\\])*"', 'STRING', text)
    stack = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for char in text:
        if char in '([{':
            stack.append(char)
        elif char in pairs:
            require(stack and stack.pop() == pairs[char], 'caller slice crosses a scope boundary')
    require(not stack, 'caller slice ends inside a nested scope/expression')
    require(not text.strip() or text.rstrip().endswith((';', '}')),
            'caller slice must end at a statement boundary')


def mask(text, spans):
    """Remove approved occurrences without moving offsets of later occurrences."""
    for start, end in sorted(spans, reverse=True):
        text = text[:start] + re.sub(r'[^\n]', ' ', text[start:end]) + text[end:]
    return text


def map_indices(text):
    """Read the complete enum, including intentional same-grid X/Y aliases."""
    match = unique(r'\benum\s+MapFacType\s*\{([^{}]*)\}\s*;', text, 'MapFacType enum')
    source, values, next_value = Source(match[1]), {}, 0
    while source.text[source.pos:].strip():
        name, expression = source.take(r'(\w+)\s*(?:=\s*(\w+)\s*)?(?:,|$)').groups()
        require(name not in values, 'duplicate map-factor index')
        if expression is None:
            value = next_value
        elif expression in values:
            value = values[expression]
        else:
            require(re.fullmatch(r'0|[1-9]\d*', expression), 'unsupported map-factor enum value')
            value = int(expression)
        values[name], next_value = value, value + 1
    require(set(values) == set(METRICS.values()) | {'num'}, 'map-factor enum member set changed')
    require(3 <= values['num'] <= 6, 'map-factor container size must be between three and six')
    slots = {name: value for name, value in values.items() if name != 'num'}
    require(set(slots.values()) == set(range(values['num'])), 'map-factor indices must fill their declared container')
    for left in slots:
        for right in slots:
            require(left[0] == right[0] or slots[left] != slots[right],
                    'map-factor indices alias different staggerings')
    print('  ok   map-factor enum preserves mass/U/V slots and permits same-grid axis aliases')
    return values


def preprocessor(text, protected):
    # These existing SHOC regions may not contain a protected name. Check this
    # BEFORE removing them so neither compile-time branch can hide a binding.
    pattern = r'#ifdef\s+ERF_USE_EAMXX_SHOC\s*\n([^#]*?)#endif'
    for match in re.finditer(pattern, text):
        require(not re.search(r'\b(?:' + '|'.join(protected) + r')\b', match[1]),
                'conditional SHOC region uses protected geometry')
    text = re.sub(pattern, lambda m: mask(m[0], [(0, len(m[0]))]), text)
    text = re.sub(r'#ifdef\s+_OPENMP\s*\n\s*#pragma\s+omp\s+parallel\s+if\s*'
                  r'\(\s*Gpu::notInLaunchRegion\s*\(\s*\)\s*\)\s*\n\s*#endif',
                  lambda m: mask(m[0], [(0, len(m[0]))]), text)
    require('#' not in text, 'unsupported preprocessor in geometry slice')
    return text


def call_arguments(text, pattern):
    """Yield argument spans from supported calls, splitting outside ()/[]/{}."""
    for match in re.finditer(pattern + r'\s*\(', text):
        stack, start, parts = [')'], match.end(), []
        pairs = {'(': ')', '[': ']', '{': '}'}
        for index in range(start, len(text)):
            char = text[index]
            if char in pairs:
                stack.append(pairs[char])
            elif char in ')]}':
                require(stack and stack[-1] == char, 'unbalanced protected call')
                stack.pop()
                if not stack:
                    parts.append((start, index))
                    yield parts
                    break
            elif char == ',' and len(stack) == 1:
                parts.append((start, index))
                start = index + 1
        else:
            raise ValueError('unterminated protected call')


def audit_uses(text, protected, ready):
    """Exempt bare input arguments only; expressions, aliases and writes remain."""
    spans = []
    for name in READ_ONLY_CALLS:
        pattern = r'(?<![\w:.>])' + re.escape(name)
        for arguments in call_arguments(text, pattern):
            for start, end in arguments:
                arg = text[start:end].strip()
                if arg in protected:
                    require(start >= ready.get(arg, 0), f'{arg}: use precedes its geometry binding')
                    spans.append((start, end))
    # Geometry-derived Box/Periodicity values are accepted only at these AMReX
    # read-only API calls. Other uses of geom remain visible to the final audit.
    for arguments in call_arguments(text, r'\.\s*(?:FillBoundary|BuildMask)'):
        for start, end in arguments:
            if compact(text[start:end]) in ('geom.Domain()', 'geom.periodicity()'):
                spans.append((start, end))
    remainder = mask(text, spans)
    found = re.search(r'\b(?:' + '|'.join(protected) + r')\b', remainder)
    require(found is None, 'unmodeled protected geometry use: ' + (found[0] if found else ''))


def audit_iterator(text):
    """The checked const_array(mfi) must still refer to this loop's iterator."""
    patterns = (
        r'\bmfi\s*\.\s*tilebox\s*\(\s*\)',
        r'\bmfi\s*\.\s*nodaltilebox\s*\(\s*[012]\s*\)',
        r'(?:\.|->)\s*(?:array|const_array)\s*\(\s*mfi\s*\)',
        r'\bavg_[xyz]mom\s*\[\s*mfi\s*\]',
        r'\bgetMultiEBCellFlagFab\s*\(\s*\)\s*\[\s*mfi\s*\]',
    )
    spans = [match.span() for pattern in patterns for match in re.finditer(pattern, text)]
    require(not re.search(r'\bmfi\b', mask(text, spans)),
            'unmodeled momentum tile iterator use before call')
    print('  ok   metric tile iterator has only supported nonmutating uses before call')


def geometry_arguments(arguments, signature, label, indices, caller=False):
    names, types, _ = signature
    args = [compact(arg) for arg in arguments.split(',')]
    require(len(args) == len(names), f'{label}: geometry call arity changed')
    expected = {name: name for name in METRICS}
    expected.update(cellSizeInv='dxInv' if caller else 'cellSizeInv', stretched_dz_d='stretched_dz_d')
    for name, value in expected.items():
        wanted = ('constGpuArray<Real,AMREX_SPACEDIM>&' if name == 'cellSizeInv' else
                  ('Gpu::DeviceVector<Real>&' if caller else 'constGpu::DeviceVector<Real>&')
                  if name == 'stretched_dz_d' else 'constArray4<constReal>&')
        require(name in names and types[name] == wanted, f'{label}: {name} geometry parameter type changed')
        actual = args[names.index(name)]
        same = (actual in METRICS and indices[METRICS[actual]] == indices[METRICS[name]]
                if name in METRICS else actual == value)
        require(same, f'{label}: {name} geometry argument binding changed')


def router_geometry_prefix(body):
    """Consume all statements before routing, including the unused EB inverses."""
    source = Source(body)
    literal(source, 'BL_PROFILE_VAR("AdvectionSrcForMom", AdvectionSrcForMom);')
    literal(source, 'AMREX_ALWAYS_ASSERT(bxz.smallEnd(2) > 0);')
    for component, box in (('u', 'bxx'), ('v', 'bxy')):
        literal(source, f'Box box2d_{component}({box}); box2d_{component}.setRange(2,0); '
                        f'box2d_{component}.grow({{3,3,0}});')
    for component in 'uv':
        literal(source, f'FArrayBox mf_{component}_invFAB(box2d_{component},1,The_Async_Arena());')
    for component in 'uv':
        literal(source, f'const Array4<Real>& mf_{component}_inv = mf_{component}_invFAB.array();')
    literal(source, 'const bool use_terrain_fitted_coords = '
                    '(terrain_type == TerrainType::StaticFittedMesh || terrain_type == TerrainType::MovingFittedMesh);')
    literal(source, 'ParallelFor(box2d_u, box2d_v,')
    for component, metric in (('u', 'mf_ux'), ('v', 'mf_vy')):
        literal(source, '[=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {')
        literal(source, f'mf_{component}_inv(i,j,0) = one / {metric}(i,j,0);')
        literal(source, '},' if component == 'u' else '});')
    require(re.match(r'\s*if\b', body[source.pos:]), 'unmodeled statement before geometry routing')


def geometry_bindings(text, router, grids, enum_text):
    """Guard the entry-input -> local -> router -> normal-grid parameter chain."""
    indices = map_indices(enum_text)
    protected = set(METRICS) | {'geom', 'mapfac', 'stretched_dz_d', 'dxInv', 'dx'}
    signature = unique(r'\bvoid\s+erf_slow_rhs_pre\s*\(([^{}]*)\)\s*\{', text, 'slow RHS geometry signature')
    params = split_parameters(preprocessor(signature[1], protected))
    for name, wanted in {'geom': 'constGeometrygeom', 'mapfac': 'Vector<std::unique_ptr<MultiFab>>&mapfac',
                         'stretched_dz_d': 'Gpu::DeviceVector<Real>&stretched_dz_d'}.items():
        matches = [compact(param).replace('amrex::', '') for param in params if re.search(r'\b' + name + r'\b', param)]
        require(matches == [wanted], f'{name}: slow RHS geometry input type/binding changed')
    body = preprocessor(brace_block(text, signature.end() - 1), protected)
    call = unique(r'\bAdvectionSrcForMom\s*\(([^();{}]+)\)\s*;', body, 'momentum geometry call')
    loops = list(re.finditer(r'for\s*\(\s*MFIter\s+mfi\s*\(\s*S_data\s*\[\s*IntVars::cons\s*\]\s*,\s*'
                            r'TileNoZ\s*\(\s*\)\s*\)\s*;\s*mfi\.isValid\s*\(\s*\)\s*;\s*\+\+mfi\s*\)\s*\{', body))
    loops = [loop for loop in loops if loop.end() <= call.start() < loop.end() + len(brace_block(body, loop.end() - 1))]
    require(len(loops) == 1, 'geometry call must belong to one supported tile loop')
    loop = loops[0]
    prefix = body[:call.start()]
    # Earlier local lambdas have returns. The theorem is conditional on reaching
    # the momentum call; the existing tile-slice check rejects control transfers
    # between shrink and call. Nonlocal jumps and const-defeating casts fail here.
    require(not re.search(r'\b(?:goto|throw|const_cast|reinterpret_cast)\b', prefix),
            'unsupported control transfer/cast in geometry slice')
    spans, ready = [], {}
    for name, pattern in {
        'dxInv': r'\bconst\s+GpuArray\s*<\s*Real\s*,\s*AMREX_SPACEDIM\s*>\s+dxInv\s*=\s*geom\s*\.\s*InvCellSizeArray\s*\(\s*\)\s*;',
        'dx': r'\bconst\s+Real\s*\*\s*dx\s*=\s*geom\s*\.\s*CellSize\s*\(\s*\)\s*;',
        'domain': r'\bconst\s+Box\s*&\s*domain\s*=\s*geom\s*\.\s*Domain\s*\(\s*\)\s*;',
    }.items():
        binding = unique(pattern, prefix, name + ' geometry origin')
        require(binding.end() < loop.start(), f'{name}: geometry binding must precede the tile loop')
        balanced_region(prefix[:binding.start()])
        spans.append(binding.span())
        ready[name] = binding.end()
    print('  ok   spacing and domain originate from the same const Geometry input')
    audit_iterator(prefix[loop.end():])

    metric_pattern = (r'\bconst\s+Array4\s*<\s*const\s+Real\s*>\s*&\s*(' + '|'.join(METRICS) + r')\s*=\s*'
                      r'mapfac\s*\[\s*MapFacType\s*::\s*(\w+)\s*\]\s*->\s*const_array\s*\(\s*mfi\s*\)\s*;')
    local = {}
    for binding in re.finditer(metric_pattern, prefix):
        name, slot = binding.groups()
        require(slot in indices and indices[slot] == indices[METRICS[name]], f'{name}: wrong map-factor origin')
        spans.append(binding.span())
        if binding.start() >= loop.end():
            require(name not in local, f'{name}: duplicate metric binding in momentum tile')
            balanced_region(prefix[loop.end():binding.start()])
            local[name] = binding.end()
    require(set(local) == set(METRICS), 'momentum tile must bind all six const metric arrays')
    # Outside this tile there are same-named const views in the earlier Omega
    # loop. Audit those uses too, but only apply the tile-local dominance bounds
    # to the current loop, where each of the six bindings is unique.
    audit_uses(mask(prefix[:loop.end()], spans), protected, ready)
    local_text = mask(prefix, spans)
    audit_uses(local_text[loop.end():], protected,
               {name: end - loop.end() for name, end in local.items()})
    print('  ok   all six momentum metrics use the intended map-factor slots and tile iterator')
    print('  ok   protected geometry has no unmodeled uses before the momentum call')
    geometry_arguments(call[1], router, 'slow RHS', indices, caller=True)
    router_geometry_prefix(router[2])
    for kind, grid in grids.items():
        routed = unique(r'\bAdvectionSrcForMom_' + kind + r'\s*\(([^();{}]+)\)\s*;', router[2], kind + ' geometry route')
        geometry_arguments(routed[1], grid, kind, indices)
    print('  ok   router and both normal-grid callees preserve all eight ordered geometry bindings')

