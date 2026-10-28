#!/usr/bin/env python3
"""Mutation controls for the integer box proof and the guarded caller slices.

Arithmetic faults must give the specified SAT counterexamples. Structural
faults must be explicit parser/contract rejections, never crashes or unknown.
Positive controls include alternative comparisons under containment, reversed
independent branches, setSmall/setBig and equivalent boundary expressions.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_momentum_dispatch_selftest import replace_once, signature_swap
from check_upwindall_z_reach import brace_block

ERF = Path(__file__).resolve().parent.parent
VERIFY = Path(__file__).with_name('check_momentum_domain.py')
HEADER = 'Source/Advection/ERF_Advection.H'
CALLER = 'Source/TimeIntegration/ERF_SlowRhsPre.cpp'
ROUTER = 'Source/Advection/ERF_AdvectionSrcForMom.cpp'
GRIDS = {kind: f'Source/Advection/ERF_AdvectionSrcForMom_{kind}.cpp'
         for kind in ('ConstantDz', 'StretchedDz')}


def cases(originals):
    yield 'baseline', {}, False, ()
    header = originals[HEADER]
    start = header.index('{', header.index('ShrinkZmomBoxAtDomainEnds ('))
    body = brace_block(header, start)
    end = start + len(body) + 2

    def helper(name, changed, required=(), mutated=True):
        return name, {HEADER: header[:start + 1] + changed + header[end - 1:]}, mutated, required

    intersection = ('shrink equals exact interior intersection [sat]',)
    for edge, read, grow in [('low', 'smallEnd', 'growLo'), ('high', 'bigEnd', 'growHi')]:
        condition = re.search(r'if\s*\(b\.' + read + r'[^{}]+\)\s*\{[^{}]+\}', body)
        statement = f'b.{grow}(2,-1);'
        for amount in ('0', '1', '-2'):
            yield helper(edge + ' wrong shrink amount ' + amount,
                         replace_once(body, statement, f'b.{grow}(2,{amount});'), intersection)
        for axis in (0, 1):
            yield helper(edge + ' wrong update axis ' + str(axis),
                         replace_once(body, statement, f'b.{grow}({axis},-1);'), intersection)
            yield helper(edge + ' wrong gate axis ' + str(axis),
                         replace_once(body, f'b.{read}(2)', f'b.{read}({axis})'), intersection)
        yield helper(edge + ' missing shrink', replace_once(body, condition[0], ''), intersection)
        yield helper(edge + ' unconditional shrink', replace_once(body, condition[0], statement),
                     intersection + ('every original interior node retained [sat]',
                                     'both boxes retain exactly their shared split plane [sat]'))
        yield helper(edge + ' inverted gate', replace_once(body, condition[0], condition[0].replace('==', '!=')), intersection)
        for axis in (0, 1):
            yield helper(edge + ' wrong domain axis ' + str(axis),
                         replace_once(body, f'domain.{read}(2)', f'domain.{read}({axis})'), intersection)
        replacement = 'growHi' if grow == 'growLo' else 'growLo'
        yield helper(edge + ' wrong box end', replace_once(body, statement, f'b.{replacement}(2,-1);'), intersection)
        yield helper(edge + ' nested early return', replace_once(body, statement, 'return b; ' + statement),
                     ('FAIL source contract:',))

    yield helper('top gate misses nodal offset', replace_once(body, 'domain.bigEnd(2) + 1', 'domain.bigEnd(2)'), intersection)
    yield helper('top gate off by two', replace_once(body, 'domain.bigEnd(2) + 1', 'domain.bigEnd(2) + 2'), intersection)
    yield helper('later horizontal box write', replace_once(body, 'return b;', 'b.growHi(0,1); return b;'),
                 ('horizontal endpoints unchanged [sat]',))
    yield helper('later vertical box write', replace_once(body, 'return b;', 'b.growLo(2,1); return b;'), intersection)
    yield helper('early outer return', 'return b;\n' + body, ('FAIL source contract:',))
    yield helper('unconsumed trailing statement', body + 'b.growLo(2,1);', ('FAIL source contract:',))
    yield helper('returned domain instead of nodal box', body.replace('return b;', 'return domain;'), ('FAIL source contract:',))
    yield helper('write to const domain', body.replace('b.growLo', 'domain.growLo'), ('FAIL source contract:',))
    yield helper('unknown box operation', body.replace('b.growLo(2,-1)', 'b.grow(2,-1)'), ('FAIL source contract:',))
    yield helper('side effect in condition', body.replace('b.smallEnd(2)', '(b = domain).smallEnd(2)'), ('FAIL source contract:',))
    yield helper('unsupported C++ negation precedence', body.replace('if (b.smallEnd', 'if (!b.smallEnd'), ('FAIL source contract:',))
    # The second condition sees the updated lower endpoint, not the input one.
    yield helper('sequential lower endpoint dependency', body.replace(
        'if (b.bigEnd(2)', 'if (b.smallEnd(2) == domain.smallEnd(2) && b.bigEnd(2)'), intersection)
    yield 'shrink aliases input by reference', {HEADER: replace_once(header, 'ShrinkZmomBoxAtDomainEnds (amrex::Box b,', 'ShrinkZmomBoxAtDomainEnds (amrex::Box& b,')}, True, ('FAIL source contract:',)
    yield 'duplicate shrink definition', {HEADER: header + '\ninline amrex::Box\nShrinkZmomBoxAtDomainEnds (amrex::Box b, const amrex::Box& domain)\n{' + body + '}\n'}, True, ('FAIL source contract:',)

    branches = list(re.finditer(r'if[^{}]+\{[^{}]+\}', body))
    yield helper('reversed independent shrink branches', branches[1][0] + '\n' + branches[0][0] + '\nreturn b;', mutated=False)
    yield helper('containment-equivalent inequalities', body.replace('== domain.smallEnd', '<= domain.smallEnd').replace('== domain.bigEnd', '>= domain.bigEnd'), mutated=False)
    yield helper('explicit endpoint setters', body.replace('b.growLo(2,-1)', 'b.setSmall(2,b.smallEnd(2)+1)').replace('b.growHi(2,-1)', 'b.setBig(2,b.bigEnd(2)-1)'), mutated=False)
    yield helper('equivalent nodal offset', body.replace('domain.bigEnd(2) + 1', '1 + domain.bigEnd(2)'), mutated=False)
    yield helper('harmless nested condition', body.replace('b.growLo(2,-1);', 'if (domain.smallEnd(2) <= domain.bigEnd(2)) { b.growLo(2,-1); }'), mutated=False)
    yield helper('explicit empty else branches', body.replace('}', '} else { }'), mutated=False)

    caller = originals[CALLER]
    shrink = 'tbz = ShrinkZmomBoxAtDomainEnds(tbz, domain);'
    # All caller edits are restricted to this loop, avoiding earlier Omega bounds.
    call_start = caller.index(shrink)
    low_start = caller.index('int lo_z_face = ', call_start)
    call_end = caller.index('domain, bc_ptr_h);', low_start) + len('domain, bc_ptr_h);')
    region = caller[low_start:call_end]

    def caller_case(name, changed, required=('FAIL source contract:',), mutated=True):
        return name, {CALLER: changed}, mutated, required

    for old, new, description in [
        ('int lo_z_face = domain.smallEnd(2);', 'int lo_z_face = domain.smallEnd(2)+1;', 'lower boundary shifted'),
        ('int lo_z_face = domain.smallEnd(2);', 'int lo_z_face = domain.smallEnd(0);', 'lower boundary wrong axis'),
        ('int hi_z_face = domain.bigEnd(2)+1;', 'int hi_z_face = domain.bigEnd(2);', 'upper boundary not nodal'),
        ('int hi_z_face = domain.bigEnd(2)+1;', 'int hi_z_face = domain.bigEnd(2)+2;', 'upper boundary shifted'),
        ('int hi_z_face = domain.bigEnd(2)+1;', 'int hi_z_face = domain.bigEnd(0)+1;', 'upper boundary wrong axis'),
    ]:
        label = 'caller lower boundary equals domain low node' if 'lo_z_face' in old else 'caller upper boundary equals domain high node'
        yield caller_case(description, caller[:low_start] + replace_once(region, old, new) + caller[call_end:], (label + ' [sat]',))

    for new, name in [(shrink.replace('tbz =', 'tby ='), 'wrong shrink destination'),
                      (shrink.replace('(tbz,', '(tby,'), 'wrong shrink input'),
                      (shrink.replace(', domain)', ', bx)'), 'tile instead of global domain'),
                      ('', 'missing shrink'), ('if (l_use_eb) { ' + shrink + ' }', 'conditional shrink'),
                      (shrink + shrink, 'duplicate shrink')]:
        yield caller_case(name, replace_once(caller, shrink, new))
    for new, name in [('Box tbz = mfi.tilebox();', 'cell-centered input'),
                      ('Box tbz = mfi.nodaltilebox(0);', 'wrong input staggering')]:
        yield caller_case(name, replace_once(caller, 'Box tbz = mfi.nodaltilebox(2);', new))
    for statement in ['tbz.growLo(2,-1);', 'tbz = tbz_grown[2];', 'Box& escaped = tbz;',
                      'unknown(tbz);', 'Box domain = bx;', 'unknown(domain);',
                      'const Box& escaped = domain;', 'continue;', '#define hi_z_face lo_z_face\n']:
        yield caller_case('intervening ' + statement.strip(), replace_once(caller, shrink, shrink + '\n' + statement))
    yield caller_case('shrink in nested scope', replace_once(caller, shrink, '{ ' + shrink + ' }'))
    yield caller_case('geometry domain binding changed', replace_once(caller, 'const Box& domain = geom.Domain();', 'const Box& domain = other.Domain();'))
    for old, new, name in [('mfi, tbx, tby, tbz,', 'mfi, tbx, tbz, tby,', 'swapped caller boxes'),
                           ('lo_z_face, hi_z_face, domain', 'hi_z_face, lo_z_face, domain', 'swapped caller boundaries'),
                           ('lo_z_face, hi_z_face, domain', 'lo_z_face, hi_z_face, bx', 'wrong caller domain')]:
        yield caller_case(name, caller[:low_start] + replace_once(region, old, new) + caller[call_end:])
    changed = replace_once(region, 'int hi_z_face', 'lo_z_face += 1; int hi_z_face')
    yield caller_case('write between boundary definitions', caller[:low_start] + changed + caller[call_end:])
    changed = replace_once(region, 'AdvectionSrcForMom(', 'tbz.growLo(2,1); AdvectionSrcForMom(')
    yield caller_case('late box write', caller[:low_start] + changed + caller[call_end:])
    yield caller_case('conditional momentum call', caller[:low_start] + 'if (l_use_eb) {\n' + region + '\n}' + caller[call_end:])
    changed = region.replace('domain.smallEnd(2);', '(domain.smallEnd(2) + 0);').replace('domain.bigEnd(2)+1;', '1 + domain.bigEnd(2);')
    yield caller_case('equivalent caller bounds', caller[:low_start] + changed + caller[call_end:], (), False)
    yield caller_case('irrelevant caller scalar', replace_once(caller, shrink, shrink + '\nReal harmless = 1;'), (), False)

    router = originals[ROUTER]
    for kind in GRIDS:
        call = re.search(r'AdvectionSrcForMom_' + kind + r'\([^;]+;', router)
        for old, new, label in [('bxx, bxy, bxz', 'bxx, bxz, bxy', 'boxes'),
                                ('lo_z_face, hi_z_face', 'hi_z_face, lo_z_face', 'boundaries')]:
            yield kind + ' swapped ' + label, {ROUTER: replace_once(router, call[0], call[0].replace(old, new))}, True, ('FAIL source contract:',)
        yield kind + ' extra arm write', {ROUTER: replace_once(router, call[0], call[0] + 'rho_w_rhs(0,0,0) = 0;')}, True, ('FAIL source contract:',)
        yield kind + ' changed route', {ROUTER: replace_once(router, 'mesh_type == MeshType::' + kind, 'mesh_type != MeshType::' + kind)}, True, ('FAIL source contract:',)
    for path, name in [(ROUTER, 'AdvectionSrcForMom')] + [(path, 'AdvectionSrcForMom_' + kind) for kind, path in GRIDS.items()]:
        for left, right in [('bxy', 'bxz'), ('lo_z_face', 'hi_z_face')]:
            yield name + ' swapped signature ' + left, {path: signature_swap(originals[path], name, left, right)}, True, ('FAIL source contract:',)
    anchor = 'AMREX_ALWAYS_ASSERT(bxz.smallEnd(2) > 0);'
    for extra in ['Box& alias = bxz;', 'lo_z_face = 0;', 'return;']:
        yield 'router intervening ' + extra, {ROUTER: replace_once(router, anchor, anchor + '\n' + extra)}, True, ('FAIL source contract:',)
    # Reformat every source file; comments and whitespace do not define semantics.
    yield 'whitespace and comments', {path: text.replace(',', ', /* gap */ ').replace(';', ';\n')
                                      for path, text in originals.items()}, False, ()


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-momentum-domain-mutants-'))
    originals = {path: (ERF / path).read_text() for path in (HEADER, CALLER, ROUTER, *GRIDS.values())}
    unexpected = mutants = positives = 0
    for index, (name, edits, mutated, required) in enumerate(cases(originals)):
        root = work / f'{index:03d}'
        for path, original in originals.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, original))
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root / 'stdout.txt').write_text(result.stdout)
        (root / 'stderr.txt').write_text(result.stderr)
        if result.returncode == 77:
            print(result.stdout.strip())
            return 77
        caught, expected, messages = mutation_outcome(result, mutated)
        missing = [item for item in required if item not in result.stdout]
        if '[unknown]' in result.stdout:
            missing.append('all obligations must be decided')
        expected = expected and not missing
        unexpected += not expected
        mutants += mutated
        positives += not mutated and name != 'baseline'
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        for message in messages[:2] + ['missing diagnostic: ' + item for item in missing]:
            print('           ' + message)
    print(f'\n{mutants} mutations; {positives} equivalent rewrites; {unexpected} unexpected outcome(s)')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
