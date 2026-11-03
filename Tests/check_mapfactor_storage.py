#!/usr/bin/env python3
"""Check the map-factor allocation/initialization slice in ERF::init_stuff.

This finite source contract uses only the standard library. Starting at the
BoxList bl2d declaration, it consumes every active statement through map-factor
initialization. It checks projection to k=0, mass/U/V staggering, distribution
mapping, component/ghost counts, allocation of every live enum slot, and both
initialization paths. Literal #if 0/1 blocks are evaluated; other directives
within the slice fail closed. Same-grid X/Y enum aliases are resolved by value.

The contract is conditional on reaching this slice with a cell-centered ba,
its matching dm, valid lev/container sizes, and ordinary C++/AMReX semantics.
BoxList/BoxArray preserve box order and cardinality; setRange(2,0) sets one cell;
convert applies the stated index type; allocation succeeds and setVal fills the
allocation. define_column_kextent must not mutate these inputs through aliases.
The function prefix/suffix and subsequent metric loading/regridding are outside
the contract. This does not prove lifetime validity, sufficient halo reach,
runtime metric positivity, physical volumes, or the allocation-to-RHS call chain.
It is a structural check, not a Z3 theorem or a C++ execution test.

usage: check_mapfactor_storage.py [ERF_ROOT]
exit: 0 success, 1 source-contract failure
"""
import re
import sys
from fractions import Fraction
from pathlib import Path

from advection_source import Source, checked_constants
from check_momentum_dispatch import compact, require
from momentum_geometry import (MAPFAC_H, balanced_region, definition, literal,
                               map_indices, unique)

ALLOCATOR = 'Source/ERF_MakeNewArrays.cpp'
STAGGER = {'m': (0, 0, 0), 'u': (1, 0, 0), 'v': (0, 1, 0)}
INTEGER = r'(?:0|[1-9]\d*)'


def strip_comments(text):
    # Match in source order: the second slash in //*** is not a block-comment
    # opener. Preserve quoted strings so comment markers inside them are inert.
    tokens = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/'
    return re.sub(tokens, lambda m: ('\n' * m[0].count('\n') + ' ')
                  if m[0].startswith(('/',)) else m[0], text, flags=re.S)


def active_source(text):
    """Only literal non-nested #if 0/1 is supported in this narrow slice."""
    active, inside, output = True, False, []
    for line in text.splitlines(keepends=True):
        if '#' not in line:
            output.append(line if active else '\n')
            continue
        directive = re.fullmatch(r'\s*#\s*(if\s+[01]|endif)\s*', line)
        require(directive is not None, 'unsupported allocator preprocessor directive')
        value = directive[1].split()
        if value[0] == 'if':
            require(not inside, 'nested allocator preprocessor region')
            inside, active = True, value[1] == '1'
        else:
            require(inside, 'unmatched allocator #endif')
            inside, active = False, True
        output.append('\n')
    require(not inside, 'unterminated allocator preprocessor region')
    return ''.join(output)


def slice_scope(prefix):
    """Reject a compile-time region opened outside the modeled slice."""
    depth = 0
    for directive in re.finditer(r'^\s*#\s*(\w+)[^\n]*', prefix, re.M):
        name = directive[1]
        if name in ('if', 'ifdef', 'ifndef'):
            depth += 1
        elif name == 'endif':
            depth -= 1
            require(depth >= 0, 'unmatched allocator prefix #endif')
        elif name in ('else', 'elif'):
            require(depth > 0, 'unmatched allocator prefix branch')
    require(depth == 0, 'allocator preprocessor region crosses slice boundary')
    balanced_region(prefix)


def allocation(source, indices, slots, enabled=True):
    slot = source.take(r'mapfac\s*\[\s*lev\s*\]\s*\[\s*MapFacType\s*::\s*(\w+)\s*\]'
                       r'\s*=\s*std\s*::\s*make_unique\s*<\s*MultiFab\s*>\s*\(')[1]
    require(slot in indices and slot != 'num', 'unsupported allocation slot')
    converted = bool(re.match(r'\s*convert\b', source.text[source.pos:]))
    if converted:
        literal(source, 'convert(')
    literal(source, 'ba2d[lev]')
    stagger = (0, 0, 0)
    if converted:
        stagger = tuple(map(int, source.take(
            rf',\s*IntVect\s*\(\s*({INTEGER})\s*,\s*({INTEGER})\s*,\s*({INTEGER})\s*\)\s*\)').groups()))
    component, gx, gy, gz = map(int, source.take(
        rf',\s*dm\s*,\s*({INTEGER})\s*,\s*IntVect\s*\(\s*({INTEGER})\s*,\s*({INTEGER})\s*,\s*({INTEGER})\s*\)\s*\)\s*;').groups())
    require(component == 1, 'map-factor allocation must have one component')
    require((gx, gy, gz) == (3, 3, 0), 'map-factor ghost layout must be (3,3,0)')
    if enabled:
        number = indices[slot]
        require(number not in slots, 'duplicate allocation of a live map-factor slot')
        slots[number] = stagger


def initialization_loop(source, indices, slots):
    start, stop = source.take(
        rf'for\s*\(\s*int\s+i\s*=\s*({INTEGER})\s*;\s*i\s*<\s*'
        rf'({INTEGER}|mapfac\s*\[\s*lev\s*\]\s*\.\s*size\s*\(\s*\))\s*;\s*i\+\+\s*\)\s*\{{').groups()
    value = source.take(r'mapfac\s*\[\s*lev\s*\]\s*\[\s*i\s*\]\s*->\s*setVal\s*\(\s*(\w+)\s*\)\s*;\s*\}')[1]
    start = int(start)
    stop = indices['num'] if compact(stop) == 'mapfac[lev].size()' else int(stop)
    require(0 <= start <= stop <= indices['num'], 'initialization range exceeds map-factor container')
    require(value in ('one', 'myhalf', 'fourth'), 'unsupported map-factor initial value')
    for number in range(start, stop):
        require(number not in slots, 'duplicate map-factor initialization')
        slots[number] = value


def verify(text, enum_text):
    indices = map_indices(enum_text)
    _, types, body = definition(text, r'ERF::init_stuff')
    require(all(types.get(name) == wanted for name, wanted in
                {'lev': 'int', 'ba': 'constBoxArray&', 'dm': 'constDistributionMapping&'}.items()),
            'allocator input signature changed')
    start = unique(r'\bBoxList\s+bl2d\s*=', body, '2D allocation origin').start()
    end = unique(r'\bif\s*\(\s*solverChoice\s*\.\s*lsm_type\s*==\s*LandSurfaceType\s*::\s*SLM\s*\)',
                 body, 'post-map-factor slice boundary').start()
    require(start < end, 'allocator slice order changed')
    slice_scope(body[:start])
    source = Source(active_source(body[start:end]))
    literal(source, 'BoxList bl2d = ba.boxList();')
    literal(source, 'for (auto& b : bl2d) { b.setRange(2,0); }')
    literal(source, 'ba2d[lev] = BoxArray(std::move(bl2d));')
    # Consume the intervening 1D construction and call rather than skipping them.
    literal(source, 'BoxList bl1d = ba.boxList();')
    literal(source, 'for (auto& b : bl1d) { b.setRange(0,0); b.setRange(1,0); }')
    literal(source, 'ba1d[lev] = BoxArray(std::move(bl1d));')
    literal(source, 'define_column_kextent(lev, ba, dm);')
    print('  ok   ordered 2D boxes preserve horizontal bounds and project to k=0')

    literal(source, 'mapfac[lev].resize(MapFacType::num);')
    slots = {}
    while re.match(r'\s*(?:mapfac\b|if\s*\(\s*MapFacType\b)', source.text[source.pos:]):
        if re.match(r'\s*if\b', source.text[source.pos:]):
            left, right = source.take(r'if\s*\(\s*MapFacType\s*::\s*(\w+)\s*!=\s*'
                                      r'MapFacType\s*::\s*(\w+)\s*\)\s*\{').groups()
            require(left in indices and right in indices, 'unsupported conditional map-factor slot')
            allocation(source, indices, slots, indices[left] != indices[right])
            literal(source, '}')
        else:
            allocation(source, indices, slots)
    require(set(slots) == set(range(indices['num'])), 'not every live map-factor slot is allocated')
    for name, number in indices.items():
        if name != 'num':
            require(slots[number] == STAGGER[name[0]], name + ': allocation has wrong staggering')
    print('  ok   all live metric slots have mass/U/V staggering, matching dm and (3,3,0) ghosts')

    literal(source, 'if (solverChoice.test_mapfactor) {')
    test_values, default_values = {}, {}
    initialization_loop(source, indices, test_values)
    initialization_loop(source, indices, test_values)
    literal(source, '} else {')
    initialization_loop(source, indices, default_values)
    literal(source, '}')
    source.done()
    require(default_values == {number: 'one' for number in slots},
            'default path must initialize every metric to one')
    require(test_values == {number: 'myhalf' if number < 3 else 'fourth' for number in slots},
            'test path must initialize every metric with its slot-specific value')
    print('  ok   both paths initialize every allocated metric slot with the expected positive constant')
    return slots


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    checked_constants(root, {'one': Fraction(1), 'myhalf': Fraction(1, 2), 'fourth': Fraction(1, 4)})
    verify(strip_comments((root / ALLOCATOR).read_text()), strip_comments((root / MAPFAC_H).read_text()))
    print('Map-factor allocation slice contract passed (standard library; no Z3).')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError) as error:
        print(f'FAIL source contract: {error}')
        sys.exit(1)
