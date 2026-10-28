#!/usr/bin/env python3
"""Mutate live momentum dispatch sources; require rejection without Python crashes.

Also accept equivalent branch reordering, whitespace, gate-clause reordering,
and distinct enum renumbering. Uses only the standard library, including when
the underlying arithmetic checkers cannot run because Z3 is unavailable.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_dispatch.py')
ENUM = 'Source/ERF_IndexDefines.H'
NORMAL = 'Source/Advection/ERF_AdvectionSrcForMom_N.H'
GRIDS = {kind: f'Source/Advection/ERF_AdvectionSrcForMom_{kind}.cpp'
         for kind in ('ConstantDz', 'StretchedDz')}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f'expected one mutation anchor: {old!r}')
    return text.replace(old, new)


def swap_names(text, left, right):
    """Swap identifiers without changing longer names or their argument types."""
    if not all(re.search(rf'\b{name}\b', text) for name in (left, right)):
        raise AssertionError(f'missing swap operands: {left}, {right}')
    return re.sub(rf'\b({left}|{right})\b',
                  lambda match: right if match[0] == left else left, text)


def arms(text, selector):
    matches = list(re.finditer(rf'if\s*\(\s*{selector}\s*==\s*AdvType::(\w+)\s*\)'
                              r'\s*\{[^{}]*\}', text))
    if len(matches) != 12:
        raise AssertionError(f'expected 12 {selector} dispatch arms, got {len(matches)}')
    return matches


def replace_span(text, match, replacement):
    return text[:match.start()] + replacement + text[match.end():]


def replace_chain(text, matches, branches):
    return text[:matches[0].start()] + ' else '.join(branches) + text[matches[-1].end():]


def signature_swap(text, function, left, right):
    matches = list(re.finditer(rf'\b{function}\s*\([^()]*\)\s*\{{', text))
    if len(matches) != 1:
        raise AssertionError(f'expected one signature: {function}')
    return replace_span(text, matches[0], swap_names(matches[0][0], left, right))


def cases(originals):
    """Yield (description, changed file, changed source, should reject)."""
    yield 'baseline', None, None, False
    tables = [(kind, path, 'horiz_adv_type', 'AdvectionSrcForMomVert_N')
              for kind, path in GRIDS.items()]
    tables.append(('vertical', NORMAL, 'vert_adv_type', 'AdvectionSrcForMomWrapper_N'))
    for label, path, selector, callee in tables:
        text = originals[path]
        matches = arms(text, selector)
        first = matches[0]

        # Exercise every mapping, including the centered row used with noncentered
        # schemes in the other dimension. Changing one template type compiles but
        # chooses a different interpolation policy.
        for match in matches:
            call = re.search(rf'{callee}<([^>]+)>', match[0])
            if call is None:
                raise AssertionError('missing template call in mutation arm')
            types = call[1].split(',')
            slot = 1 if label == 'vertical' else 0
            types[slot] = 'UPWIND3' if types[slot] == 'CENTERED2' else 'CENTERED2'
            changed = replace_once(match[0], call[0], callee + '<' + ','.join(types) + '>')
            yield f'{label} {match[1]} wrong policy', path, replace_span(text, match, changed), True

        blend = ('upw_frac_h', 'upw_frac_v') if label == 'vertical' else ('horiz_upw_frac', 'vert_upw_frac')
        for left, right in [('bxx', 'bxz'), ('rho_u_rhs', 'rho_v_rhs'), ('rho_u', 'rho_v'),
                            ('u', 'w'), ('mf_mx', 'mf_my'), blend, ('lo_z_face', 'hi_z_face')]:
            yield (f'{label} swap {left}/{right} arguments', path,
                   replace_span(text, first, swap_names(first[0], left, right)), True)

        changed = first[0].replace(selector, 'lo_z_face')
        yield f'{label} wrong selector', path, replace_span(text, first, changed), True
        changed = matches[1][0].replace(matches[1][1], first[1])
        yield f'{label} duplicate condition', path, replace_span(text, matches[1], changed), True
        yield (f'{label} missing branch', path,
               replace_chain(text, matches, [match[0] for match in matches if match != matches[1]]), True)
        changed = first[0].replace('AdvType::Centered_2nd', 'AdvType::Upwind_3rd_SL')
        yield f'{label} unsupported scheme branch', path, replace_span(text, first, changed), True

        statement = first[0][first[0].index('{') + 1:-1]
        changed = first[0][:-1] + statement + '}'
        yield f'{label} duplicate call', path, replace_span(text, first, changed), True
        changed = first[0][:-1] + 'rho_u_rhs(0,0,0) = 0; }'
        yield f'{label} extra write after call', path, replace_span(text, first, changed), True
        changed = first[0].replace('{', '{ return;', 1)
        yield f'{label} return before call', path, replace_span(text, first, changed), True
        fallback = 'AMREX_ASSERT_WITH_MESSAGE(false, "Unknown advection scheme!");'
        yield f'{label} weakened fallback', path, replace_once(text, fallback, fallback.replace('false', 'true')), True
        yield f'{label} omitted fallback', path, replace_once(text, fallback, ''), True
        # In all three source files the selected dispatch function is the final
        # definition, so this inserts a write after its dispatch chain.
        end = text.rindex('}')
        yield f'{label} write after dispatch', path, text[:end] + 'rho_u_rhs(0,0,0) = 0;\n' + text[end:], True
        yield (f'{label} reversed branch order', path,
               replace_chain(text, matches, [match[0] for match in reversed(matches)]), False)

    normal = originals[NORMAL]
    for old, new, label in [
        ('<InterpType_H,CENTERED2,UPWINDALL>', '<CENTERED2,CENTERED2,UPWINDALL>', 'lost horizontal type'),
        ('<InterpType_H,CENTERED2,UPWINDALL>', '<InterpType_H,CENTERED2,CENTERED2>', 'wrong wall type'),
    ]:
        yield f'vertical {label}', NORMAL, replace_once(normal, old, new), True
    for function in ('AdvectionSrcForMomVert_N', 'AdvectionSrcForMomWrapper_N'):
        for left, right in [('rho_u', 'u'), ('upw_frac_h', 'upw_frac_v')]:
            yield f'{function} swapped {left}/{right} parameters', NORMAL, signature_swap(normal, function, left, right), True
    old = ('template<typename InterpType_H, typename InterpType_V, typename WallInterpType>\n'
           'void\nAdvectionSrcForMomWrapper_N')
    yield 'wrapper reversed template bindings', NORMAL, replace_once(normal, old, swap_names(old, 'InterpType_H', 'InterpType_V')), True

    for kind, path in GRIDS.items():
        text = originals[path]
        gate = re.search(r'if\s*\((horiz_adv_type[^{}]+&&[^{}]+)\)\s*\{', text)
        if gate is None:
            raise AssertionError(f'{kind}: missing fast-path gate')
        for label, change in [
            ('OR fast-path gate', gate[0].replace('&&', '||')),
            ('wrong vertical fast-path scheme', gate[0].replace('vert_adv_type == AdvType::Centered_2nd',
                                                               'vert_adv_type == AdvType::Upwind_3rd')),
        ]:
            yield f'{kind} {label}', path, replace_span(text, gate, change), True
        yield (f'{kind} selector overwritten before dispatch', path,
               text[:gate.start()] + 'const_cast<AdvType&>(horiz_adv_type) = AdvType::Centered_2nd;\n'
               + text[gate.start():], True)
        yield (f'{kind} swapped outer velocity parameters', path,
               signature_swap(text, 'AdvectionSrcForMom_' + kind, 'u', 'v'), True)
        clauses = gate[1].split('&&')
        change = 'if (' + ' && '.join(reversed(clauses)) + ') {'
        yield f'{kind} reversed gate clauses', path, replace_span(text, gate, change), False

    enum = originals[ENUM]
    yield 'numeric enum alias', ENUM, replace_once(enum, 'Upwind_3rd    = 102', 'Upwind_3rd    = 101'), True
    # 0145 is octal 101: treating it as decimal would miss an alias of Centered_2nd.
    yield 'octal enum alias', ENUM, replace_once(enum, 'Upwind_3rd    = 102', 'Upwind_3rd    = 0145'), True
    renamed = re.sub(r'(\b\w+\s*=\s*)(\d+)', lambda match: match[1] + str(int(match[2]) + 1000),
                     enum[enum.index('enum struct AdvType'):])
    yield 'distinct enum renumbering', ENUM, enum[:enum.index('enum struct AdvType')] + renamed, False
    yield 'dispatch whitespace', NORMAL, normal.replace('>(bxx, bxy, bxz,', '> ( bxx ,\n bxy , bxz ,'), False


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        tempfile.mkdtemp(prefix='erf-momentum-dispatch-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (ENUM, NORMAL, *GRIDS.values())}
    unexpected = mutations = positives = 0
    for index, (name, path, changed, mutated) in enumerate(cases(originals)):
        root = work / f'{index:03d}'
        for source_path, original in originals.items():
            target = root / source_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(changed if source_path == path else original)
        result = subprocess.run([sys.executable, str(VERIFY), str(root)],
                                capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        caught, expected, failures = mutation_outcome(result, mutated)
        unexpected += not expected
        mutations += mutated
        positives += not mutated and name != 'baseline'
        print(f"[{'CAUGHT' if caught else 'passes' if expected else 'UNEXPECTED'}] {name}")
        for failure in failures[:2]:
            print(f'           {failure}')
    print(f'\n{mutations} mutations; {positives} equivalent rewrites; {unexpected} unexpected outcome(s)')
    return 1 if unexpected else 0


if __name__ == '__main__':
    sys.exit(main())
