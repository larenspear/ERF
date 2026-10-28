#!/usr/bin/env python3
"""Exhaustively check the normal momentum scheme dispatch from live C++.

For each grid path, enumerate all 12x12 supported H/V selections: one inline
centered pair and 143 templated pairs. Consume the entire dispatch grammar,
preserve call and template argument order, and check the callee signatures.
The standard library suffices for this finite check; Z3 is not required.

The endpoint is AdvectionSrcForMomWrapper_N's type/parameter bindings. Its
component calls and arithmetic are checked separately. Wall-order reduction,
interpolator implementations, caller input parsing, and terrain/EB are outside
this check. Unsupported selectors must reach the source assertion fallback;
AMREX_ASSERT_WITH_MESSAGE may be disabled in release builds.

usage: check_momentum_dispatch.py [ERF_ROOT]
exit: 0 all checks pass, 1 a source contract fails
"""
import re
import sys
from itertools import product
from pathlib import Path

from advection_source import Source, momentum_component, momentum_lambdas, momentum_setup
from check_upwindall_z_reach import brace_block, strip_comments

ENUM_H = 'Source/ERF_IndexDefines.H'
NORMAL_H = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'
GRID_CPP = {kind: f'Source/Advection/ERF_AdvectionSrcForMom_{kind}.cpp'
            for kind in ('ConstantDz', 'StretchedDz')}
SCHEMES = {
    'Centered_2nd': 'CENTERED2', 'Upwind_3rd': 'UPWIND3', 'Centered_4th': 'CENTERED4',
    'Upwind_5th': 'UPWIND5', 'Centered_6th': 'CENTERED6', 'Weno_3': 'WENO3',
    'Weno_3Z': 'WENO_Z3', 'Weno_3MZQ': 'WENO_MZQ3', 'Weno_5': 'WENO5',
    'Weno_5Z': 'WENO_Z5', 'Weno_7': 'WENO7', 'Weno_7Z': 'WENO_Z7',
}
UNSUPPORTED = {'Upwind_3rd_SL', 'Unknown'}
SHARED_ARGS = ('bxx bxy bxz rho_u_rhs rho_v_rhs rho_w_rhs rho_u rho_v rho_w u v w '
               'cellSizeInv stretched_dz_d mf_mx mf_ux_inv mf_vx_inv mf_my mf_uy_inv '
               'mf_vy_inv upw_frac_h upw_frac_v vert_adv_type lo_z_face hi_z_face').split()
OUTER_ARGS = [dict(rho_w='omega', upw_frac_h='horiz_upw_frac',
                   upw_frac_v='vert_upw_frac').get(name, name) for name in SHARED_ARGS]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compact(text):
    return re.sub(r'\s+', '', text)


def split_parameters(text):
    """Split commas outside C++ template type arguments."""
    result, start, depth = [], 0, 0
    for index, char in enumerate(text):
        depth += (char == '<') - (char == '>')
        require(depth >= 0, 'unbalanced template type in parameter list')
        if char == ',' and depth == 0:
            result.append(text[start:index].strip())
            start = index + 1
    require(depth == 0, 'unbalanced template type in parameter list')
    result.append(text[start:].strip())
    return result


def parameter_type(name):
    if name in ('bxx', 'bxy', 'bxz'):
        return 'const Box&'
    if name.endswith('_rhs'):
        return 'const Array4<Real>&'
    if name == 'cellSizeInv':
        return 'const GpuArray<Real, AMREX_SPACEDIM>&'
    if name == 'stretched_dz_d':
        return 'const Gpu::DeviceVector<Real>&'
    if name.endswith('adv_type'):
        return 'const AdvType'
    if name in ('upw_frac_h', 'upw_frac_v', 'horiz_upw_frac', 'vert_upw_frac'):
        return 'const Real'
    if name in ('lo_z_face', 'hi_z_face'):
        return 'const int'
    if name == 'terrain_type':
        return 'TerrainType&'
    return 'const Array4<const Real>&'


def function(text, name, parameters, template=()):
    declaration = (r'template\s*<([^>]+)>\s*' if template else '')
    pattern = declaration + rf'\bvoid\s+{name}\s*\(([^()]*)\)\s*\{{'
    matches = list(re.finditer(pattern, text))
    require(len(matches) == 1, f'{name}: expected exactly one supported function definition')
    match = matches[0]
    if template:
        actual = [compact(item) for item in match[1].split(',')]
        require(actual == ['typename' + name for name in template],
                f'{name}: template parameter binding/order changed')
    signature = match[2 if template else 1]
    actual = [compact(item).replace('amrex::', '') for item in split_parameters(signature)]
    expected = [compact(parameter_type(item) + ' ' + item) for item in parameters]
    require(actual == expected, f'{name}: ordered function parameter bindings/types changed')
    return brace_block(text, match.end() - 1)


def block(source):
    source.take(r'\{')
    body = brace_block(source.text, source.pos - 1)
    source.pos += len(body) + 1
    return body


def enum_values(text):
    matches = list(re.finditer(r'enum\s+struct\s+AdvType\s*:\s*int\s*\{', text))
    require(len(matches) == 1, 'expected one AdvType enum')
    source = Source(brace_block(text, matches[0].end() - 1))
    values = {}
    while source.text[source.pos:].strip():
        # Only decimal literals are supported; a leading zero is octal in C++.
        name, value = source.take(r'(\w+)\s*=\s*(-?(?:0|[1-9]\d*))\s*(?:,|$)').groups()
        require(name not in values, f'duplicate AdvType member {name}')
        values[name] = int(value)
    source.done()
    require(set(values) == set(SCHEMES) | UNSUPPORTED, 'AdvType member set changed; review dispatch coverage')
    require(len(set(values.values())) == len(values), 'AdvType numeric aliases make scheme conditions overlap')
    return values


def dispatch(body, selector, callee, arguments):
    """Read one complete if/else-if dispatch, exactly one call in each arm."""
    source, table = Source(body), {}
    while True:
        if table:
            source.take(r'else\s+')
            if re.match(r'\s*\{', source.text[source.pos:]):
                fallback = Source(block(source))
                fallback.take(r'AMREX_ASSERT_WITH_MESSAGE\s*\(\s*false\s*,\s*'
                              r'"Unknown advection scheme!"\s*\)\s*;')
                fallback.done()
                source.done()
                break
        scheme = source.take(rf'if\s*\(\s*{selector}\s*==\s*AdvType::(\w+)\s*\)').group(1)
        require(scheme not in table, f'{selector}: duplicate branch {scheme}')
        arm = Source(block(source))
        types, args = arm.take(rf'{callee}\s*<([\w\s,]+)>\s*\(([^()]+)\)\s*;').groups()
        arm.done()
        require([compact(arg) for arg in args.split(',')] == arguments,
                f'{selector}.{scheme}: call argument binding/order changed')
        table[scheme] = tuple(compact(item) for item in types.split(','))
    require(set(table) == set(SCHEMES), f'{selector}: incomplete or unsupported dispatch branch set')
    return table


def horizontal_dispatch(text, kind):
    params = ('bxx bxy bxz rho_u_rhs rho_v_rhs rho_w_rhs u v w rho_u rho_v omega '
              'cellSizeInv stretched_dz_d mf_mx mf_ux mf_vx mf_my mf_uy mf_vy '
              'horiz_adv_type vert_adv_type horiz_upw_frac vert_upw_frac').split()
    params += ([] if kind == 'StretchedDz' else ['terrain_type']) + ['lo_z_face', 'hi_z_face']
    body = function(text, 'AdvectionSrcForMom_' + kind, params)
    first_if = re.search(r'\bif\s*\(', body)
    require(first_if is not None, f'{kind}: missing inline dispatch gate')
    momentum_setup(body[:first_if.start()], kind == 'StretchedDz')
    source = Source(body[first_if.start():])
    condition = source.take(r'if\s*\(([^()]*)\)').group(1)
    clauses = [compact(item) for item in condition.split('&&')]
    require(len(clauses) == 2 and set(clauses) == {
        'horiz_adv_type==AdvType::Centered_2nd', 'vert_adv_type==AdvType::Centered_2nd'},
        f'{kind}: inline gate must select exactly the centered/centered pair')
    inline = momentum_lambdas(block(source))
    for component, lambda_body in zip('uvw', inline):
        momentum_component(lambda_body, f'rho_{component}_rhs', stretched=kind == 'StretchedDz')
    source.take(r'else\s*')
    table = dispatch(block(source), 'horiz_adv_type', 'AdvectionSrcForMomVert_N', OUTER_ARGS)
    source.done()
    return table


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    values = enum_values(strip_comments((root / ENUM_H).read_text()))
    print(f'  ok   {len(values)} distinct AdvType values; {len(SCHEMES)} supported momentum schemes')
    normal = strip_comments((root / NORMAL_H).read_text())
    vertical_body = function(normal, 'AdvectionSrcForMomVert_N', SHARED_ARGS, ('InterpType_H',))
    vertical = dispatch(vertical_body, 'vert_adv_type', 'AdvectionSrcForMomWrapper_N', SHARED_ARGS)
    function(normal, 'AdvectionSrcForMomWrapper_N', SHARED_ARGS,
             ('InterpType_H', 'InterpType_V', 'WallInterpType'))
    for scheme, policy in SCHEMES.items():
        require(vertical[scheme] == ('InterpType_H', policy, 'UPWINDALL'),
                f'vertical {scheme}: expected preserved H, {policy}, UPWINDALL; got {vertical[scheme]}')
    print('  ok   12 vertical mappings preserve horizontal type, select the intended V type and UPWINDALL wall type')
    print('  ok   vertical calls and wrapper signatures preserve all 25 parameter bindings')
    for kind, path in GRID_CPP.items():
        horizontal = horizontal_dispatch(strip_comments((root / path).read_text()), kind)
        for scheme, policy in SCHEMES.items():
            require(horizontal[scheme] == (policy,),
                    f'{kind} horizontal {scheme}: expected {policy}; got {horizontal[scheme]}')
        count = 0
        for horiz, vert in product(SCHEMES, repeat=2):
            if horiz == vert == 'Centered_2nd':
                continue  # The fully consumed inline gate selects this one pair.
            chosen = (horizontal[horiz][0], *vertical[vert][1:])
            require(chosen == (SCHEMES[horiz], SCHEMES[vert], 'UPWINDALL'),
                    f'{kind}: wrong composed policy for {horiz}/{vert}: {chosen}')
            count += 1
        print(f'  ok   {kind}: 144 pairs route to 1 centered inline path and {count} intended template triples')
        print(f'  ok   {kind}: ordered argument bindings preserved; unsupported schemes retain assertion fallback')
    print('\n0 check(s) failed')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError) as exc:
        print(f'FAIL momentum dispatch source contract: {exc}')
        sys.exit(1)
