#!/usr/bin/env python3
"""Prove the Z-momentum box contract and check selected caller bindings.

The complete ShrinkZmomBoxAtDomainEnds body is interpreted in source order over
integer box endpoints. For arbitrary contained Z-nodal boxes (also empty ones),
its point set must equal the input intersected with lo < k < hi. Two arbitrary
adjacent cell boxes must retain their shared nodal plane and cover the same
interior as one whole-domain box. This is coverage, not unique ownership.

The SlowRhsPre check consumes the complete tile/shrink setup and boundary/call
sequence. Between those regions it audits every tbz occurrence, allowing only
the existing setVal read. Domain occurrences before the call are restricted to
the const Geometry binding, endpoint reads and a named set of call arguments.
The two normal-grid routing arms and relevant ordered callee parameters are
checked too. Other caller computations and later boundary overwrites are NOT
interpreted: approved calls must not mutate boxes/geometry through aliases.

Assumptions: AMReX inclusive box endpoints, growLo subtracts its amount and
growHi adds it, Box copies preserve index type, nodaltilebox(2) supplies contained
Z-nodal tiles covering the domain, empty boxes launch no work, pure endpoint
reads, ordinary C++/AMReX names (no macro rewriting), and no integer overflow.
The production positive-k assertions additionally require domain.smallEnd(2)
>= 0; the general helper theorem does not. No halo, allocation, floating-point,
terrain/EB arithmetic, boundary budget or whole-RHS conservation proof is made.

usage: check_momentum_domain.py [ERF_ROOT]
exit: 0 success, 1 contract failure, 77 z3 unavailable (CTest skip)
"""
import ast
import re
import sys
from pathlib import Path

from advection_source import Source, pure_expression
from check_momentum_dispatch import block, compact, require, split_parameters
from check_upwindall_z_reach import brace_block, strip_comments

try:
    import z3
except ImportError:
    print('check_momentum_domain.py: z3-solver is not importable; skipping (exit 77)')
    sys.exit(77)

HEADER = 'Source/Advection/ERF_Advection.H'
CALLER = 'Source/TimeIntegration/ERF_SlowRhsPre.cpp'
ROUTER = 'Source/Advection/ERF_AdvectionSrcForMom.cpp'
GRIDS = {kind: f'Source/Advection/ERF_AdvectionSrcForMom_{kind}.cpp'
         for kind in ('ConstantDz', 'StretchedDz')}
HELPER = 'ShrinkZmomBoxAtDomainEnds'
failures = []


def prove(claim, name, assumptions):
    solver = z3.Solver()
    solver.set(timeout=5000)
    solver.add(*assumptions, z3.Not(claim))
    outcome = solver.check()
    print(f"  {'ok  ' if outcome == z3.unsat else 'FAIL'} {name} [{outcome}]")
    if outcome != z3.unsat:
        failures.append(name)
        print(f'       {solver.model() if outcome == z3.sat else solver.reason_unknown()}')


def literal(source, text):
    tokens = re.findall(r'\w+|::|[^\s]', text)
    source.take(r'\s*'.join(re.escape(token) for token in tokens))


def unique(pattern, text, description):
    matches = list(re.finditer(pattern, text, re.S))
    require(len(matches) == 1, f'{description}: expected exactly one supported occurrence')
    return matches[0]


def read_program(text):
    require(len(re.findall(r'\b' + HELPER + r'\b', text)) == 1,
            'expected exactly one shrink declaration/definition')
    signature = unique(r'\binline\s+amrex::Box\s+' + HELPER +
                       r'\s*\(\s*amrex::Box\s+b\s*,\s*const\s+amrex::Box\s*&\s*domain\s*\)\s*\{',
                       text, 'shrink signature')
    source = Source(brace_block(text, signature.end() - 1))

    def nested(source):
        child = Source(block(source))
        result = statements(child)
        child.done()
        return result

    def statements(source):
        result = []
        while source.text[source.pos:].strip():
            if re.match(r'\s*if\b', source.text[source.pos:]):
                condition = source.take(r'if\s*\(([^;{}]+)\)\s*(?=\{)').group(1)
                yes = nested(source)
                no = []
                if re.match(r'\s*else\b', source.text[source.pos:]):
                    source.take(r'else\s*')
                    no = nested(source)
                result.append(('if', condition, yes, no))
            else:
                # Return is deliberately handled only once, at the end of the
                # outer function. Early/conditional returns fail closed.
                match = re.match(r'\s*return\b', source.text[source.pos:])
                if match:
                    break
                obj, operation, axis, expr = source.take(
                    r'(b|domain)\s*\.\s*(growLo|growHi|setSmall|setBig)\s*'
                    r'\(\s*([012])\s*,\s*([^;{}]+)\)\s*;').groups()
                require(obj == 'b', 'shrink must not modify the const domain')
                result.append((operation, int(axis), expr))
        return result

    # Nested regions must be consumed completely; only the outermost region
    # may leave a return for the final consumer.
    program = statements(source)
    result = source.take(r'return\s+(b|domain)\s*;').group(1)
    source.done()
    require(result == 'b', 'shrink must return the Z-nodal box, not the cell-centered domain')
    return program


def expression(text, boxes):
    # Python's `not` precedence differs from C++ unary !. Keep this reader
    # narrow rather than silently translating unparenthesized negations.
    require(not re.search(r'!(?!=)|\bnot\b', text), 'unsupported logical negation in box expression')
    text = text.replace('&&', ' and ').replace('||', ' or ')
    pure_expression(text)
    text = text.strip()

    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return z3.IntVal(node.value)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            obj, method = node.func.value, node.func.attr
            require(isinstance(obj, ast.Name) and obj.id in boxes and
                    method in ('smallEnd', 'bigEnd') and len(node.args) == 1 and
                    not node.keywords, 'unsupported box endpoint read')
            axis = node.args[0]
            require(isinstance(axis, ast.Constant) and type(axis.value) is int and
                    0 <= axis.value < 3, 'box axis must be a literal 0, 1 or 2')
            return boxes[obj.id][method == 'bigEnd'][axis.value]
        if isinstance(node, ast.UnaryOp):
            value = visit(node.operand)
            if z3.is_int(value) and isinstance(node.op, (ast.UAdd, ast.USub)):
                return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = visit(node.left), visit(node.right)
            require(z3.is_int(left) and z3.is_int(right), 'integer box arithmetic required')
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            left, right = visit(node.left), visit(node.comparators[0])
            require(z3.is_int(left) and z3.is_int(right), 'integer endpoint comparison required')
            operations = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                          ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                          ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right}
            if type(node.ops[0]) in operations:
                return operations[type(node.ops[0])]()
        if isinstance(node, ast.BoolOp):
            values = [visit(item) for item in node.values]
            require(all(z3.is_bool(item) for item in values), 'Boolean condition required')
            return (z3.And if isinstance(node.op, ast.And) else z3.Or)(*values)
        raise ValueError(f'unsupported box expression: {text}')

    return visit(ast.parse(text, mode='eval').body)


def execute(program, initial, domain):
    box = [list(side) for side in initial]
    for statement in program:
        if statement[0] == 'if':
            _, condition, yes, no = statement
            predicate = expression(condition, {'b': box, 'domain': domain})
            require(z3.is_bool(predicate), 'Boolean shrink condition required')
            yes_box, no_box = execute(yes, box, domain), execute(no, box, domain)
            box = [[z3.If(predicate, yes_box[side][axis], no_box[side][axis])
                    for axis in range(3)] for side in range(2)]
        else:
            operation, axis, expr = statement
            value = expression(expr, {'b': box, 'domain': domain})
            require(z3.is_int(value), 'integer endpoint update required')
            side = int(operation in ('growHi', 'setBig'))
            box[side][axis] = (box[side][axis] + (value if side else -value)
                               if operation.startswith('grow') else value)
    return box


def contains(box, point):
    return z3.And(*[z3.And(box[0][axis] <= point[axis], point[axis] <= box[1][axis])
                    for axis in range(3)])


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


def check_arguments(arguments, signature, expected, label):
    names, types, _ = signature
    args = [compact(arg) for arg in arguments.split(',')]
    require(len(args) == len(names), f'{label}: call arity changed')
    for name, value in expected.items():
        require(name in names and args[names.index(name)] == value,
                f'{label}: {name} argument binding changed')
        expected_type = 'constint' if name.endswith('_z_face') else 'constBox&'
        require(types[name] == expected_type, f'{label}: {name} parameter type changed')


def audit_domain(text):
    """Restrict explicit uses; named external calls are assumed not to mutate it."""
    text = re.sub(r'\bdomain\s*\.\s*(?:smallEnd|bigEnd)\s*\(\s*[012]\s*\)', 'ENDPOINT', text)
    allowed = ('AdvectionSrcForScalars', 'EBAdvectionSrcForScalars',
               'DiffusionSrcForState_[STN]', 'DiffusionSrcForState_EB',
               'erf_cloud_chamber_wall_flux::apply')
    for pattern in allowed:
        # Only bare argument positions are exempted. Declarations, escapes,
        # writes and unknown calls still leave the protected identifier behind.
        for match in list(re.finditer(r'\b' + pattern + r'\s*\(', text))[::-1]:
            start, depth, end = match.end(), 1, match.end()
            while depth and end < len(text):
                depth += (text[end] == '(') - (text[end] == ')')
                end += 1
            require(depth == 0, 'unbalanced approved domain call')
            args = text[start:end - 1]
            args = re.sub(r'(?<=,)\s*domain\s*(?=,)', ' DOMAIN_ARGUMENT ', args)
            text = text[:start] + args + text[end - 1:]
    require(not re.search(r'\bdomain\b', text), 'unmodeled domain use before momentum call')


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


def caller_bindings(text, router):
    # The large function signature/other computations are not modeled. Select
    # its unique definition, then the exact MFIter loop that owns the call.
    match = unique(r'\bvoid\s+erf_slow_rhs_pre\s*\([^{}]*\)\s*\{', text, 'slow RHS definition')
    body = brace_block(text, match.end() - 1)
    domain = unique(r'\bconst\s+Box\s*&\s*domain\s*=\s*geom\s*\.\s*Domain\s*\(\s*\)\s*;',
                    body, 'const geometry domain binding')
    balanced_region(body[:domain.start()])
    loops = list(re.finditer(r'for\s*\(\s*MFIter\s+mfi\s*\(\s*S_data\s*\[\s*IntVars::cons\s*\]\s*,\s*'
                            r'TileNoZ\s*\(\s*\)\s*\)\s*;\s*mfi\.isValid\s*\(\s*\)\s*;\s*\+\+mfi\s*\)\s*\{', body))
    loops = [loop for loop in loops if re.search(r'\bAdvectionSrcForMom\s*\(', brace_block(body, loop.end() - 1))]
    require(len(loops) == 1, 'expected one supported tile loop containing the momentum call')
    loop = loops[0]
    require(domain.end() < loop.start(), 'domain binding must precede the momentum loop')
    source = Source(brace_block(body, loop.end() - 1))
    for statement in ('Box bx = mfi.tilebox();', 'Box tbx = mfi.nodaltilebox(0);',
                      'Box tby = mfi.nodaltilebox(1);', 'Box tbz = mfi.nodaltilebox(2);'):
        literal(source, statement)
    for component in 'xyz':
        literal(source, f'Vector<Box> tb{component}_grown(AMREX_SPACEDIM);')
    literal(source, 'if (l_use_eb) { for (int dir = 0; dir < AMREX_SPACEDIM; ++dir) {')
    for component in 'xyz':
        literal(source, f'tb{component}_grown[dir] = tb{component};')
    literal(source, 'IntVect iv(1, 1, 1); iv[dir] = 0;')
    for component in 'xyz':
        literal(source, f'tb{component}_grown[dir] = (tb{component}_grown[dir].growHi(dir,1)).grow(iv);')
    literal(source, '} }')
    literal(source, 'tbz = ShrinkZmomBoxAtDomainEnds(tbz, domain);')
    setup_end = source.pos
    bounds = unique(r'\bint\s+lo_z_face\s*=', source.text, 'lower momentum boundary declaration')
    require(bounds.start() >= setup_end, 'boundary declarations must follow shrink')
    middle = source.text[setup_end:bounds.start()]
    balanced_region(middle)
    require(not re.search(r'\b(?:return|continue|break|goto|throw)\b|#', middle),
            'unsupported control transfer/preprocessor in caller slice')
    middle = re.sub(r'\bavg_zmom\s*\[\s*mfi\s*\]\s*\.\s*template\s+setVal\s*<\s*RunOn::Device\s*>\s*'
                    r'\(\s*0\s*,\s*tbz\s*\)\s*;', '', middle)
    require(not re.search(r'\b(?:tbz|lo_z_face|hi_z_face|ShrinkZmomBoxAtDomainEnds)\b', middle),
            'unmodeled momentum box/boundary use before call')
    source.pos = bounds.start()
    low = source.take(r'int\s+lo_z_face\s*=\s*([^;{}]+);').group(1)
    high = source.take(r'int\s+hi_z_face\s*=\s*([^;{}]+);').group(1)
    arguments = source.take(r'AdvectionSrcForMom\s*\(([^();{}]+)\)\s*;').group(1)
    require(len(re.findall(r'\bAdvectionSrcForMom\s*\(', body)) == 1,
            'expected one momentum call in slow RHS')
    check_arguments(arguments, router, {'bxx': 'tbx', 'bxy': 'tby', 'bxz': 'tbz',
                    'lo_z_face': 'lo_z_face', 'hi_z_face': 'hi_z_face', 'domain': 'domain'}, 'slow RHS')
    prefix = body[:loop.end()] + source.text[:setup_end]
    prefix = prefix[:domain.start()] + prefix[domain.end():]
    prefix = re.sub(r'\btbz\s*=\s*ShrinkZmomBoxAtDomainEnds\s*\(\s*tbz\s*,\s*domain\s*\)\s*;', '', prefix)
    audit_domain(prefix + middle)
    print('  ok   caller tile/shrink setup, protected uses and ordered box/boundary bindings')
    return low, high


def route_bindings(router, grids):
    body = router[2]
    first = re.search(r'\bif\s*\(', body)
    require(first is not None, 'missing normal-grid route')
    prefix = body[:first.start()]
    assertion = unique(r'AMREX_ALWAYS_ASSERT\s*\(\s*bxz\.smallEnd\s*\(\s*2\s*\)\s*>\s*0\s*\)\s*;',
                       prefix, 'positive-k assertion')
    prefix = prefix[:assertion.start()] + prefix[assertion.end():]
    require(not re.search(r'\b(?:bxz|lo_z_face|hi_z_face)\b', prefix),
            'unmodeled normal-grid box/boundary use before routing')
    require(not re.search(r'\b(?:return|goto|throw)\b|#', prefix),
            'unsupported control transfer/preprocessor before routing')
    source = Source(body[first.start():])
    for index, kind in enumerate(GRIDS):
        if index:
            literal(source, 'else')
        literal(source, f'if (mesh_type == MeshType::{kind} && terrain_type != TerrainType::EB)')
        arm = Source(block(source))
        args = arm.take(r'AdvectionSrcForMom_' + kind + r'\s*\(([^();{}]+)\)\s*;').group(1)
        arm.done()
        check_arguments(args, grids[kind], {name: name for name in
                        ('bxx', 'bxy', 'bxz', 'lo_z_face', 'hi_z_face')}, kind)
    # The rest is intentionally outside this normal-route binding proof.
    literal(source, 'else if (terrain_type == TerrainType::EB)')
    print('  ok   both normal-grid routing arms preserve ordered box/boundary bindings')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    read = lambda name: strip_comments((root / name).read_text())
    program = read_program(read(HEADER))
    router = definition(read(ROUTER), 'AdvectionSrcForMom')
    low_expr, high_expr = caller_bindings(read(CALLER), router)
    route_bindings(router, {kind: definition(read(path), 'AdvectionSrcForMom_' + kind)
                            for kind, path in GRIDS.items()})
    domain = [list(z3.Ints('domain_lo_x domain_lo_y domain_lo_z')),
              list(z3.Ints('domain_hi_x domain_hi_y domain_hi_z'))]
    initial = [list(z3.Ints('box_lo_x box_lo_y box_lo_z')),
               list(z3.Ints('box_hi_x box_hi_y box_hi_z'))]
    point = z3.Ints('i j k')
    valid_domain = [domain[0][axis] <= domain[1][axis] for axis in range(3)]
    contained = [constraint for axis in range(3) for constraint in
                 (initial[0][axis] >= domain[0][axis],
                  initial[1][axis] <= domain[1][axis] + int(axis == 2))]
    assumptions = valid_domain + contained  # No nonempty-input assumption.
    out = execute(program, initial, domain)
    lo, hi, k = domain[0][2], domain[1][2] + 1, point[2]
    interior = z3.And(lo < k, k < hi)
    member = contains(out, point)
    prove(member == z3.And(contains(initial, point), interior),
          'shrink equals exact interior intersection', assumptions)
    prove(z3.Implies(member, interior), 'physical Z boundary nodes excluded', assumptions)
    prove(z3.Implies(z3.And(contains(initial, point), interior), member),
          'every original interior node retained', assumptions)
    prove(z3.And(*[out[side][axis] == initial[side][axis] for side in range(2) for axis in (0, 1)]),
          'horizontal endpoints unchanged', assumptions)
    prove(z3.Implies(z3.Or(*[initial[0][axis] > initial[1][axis] for axis in range(3)]),
                     z3.Not(member)), 'empty input stays empty', assumptions)
    twice = execute(program, out, domain)
    prove(z3.And(*[twice[side][axis] == out[side][axis] for side in range(2) for axis in range(3)]),
          'shrinking is idempotent', assumptions)
    lower = expression(low_expr, {'domain': domain})
    upper = expression(high_expr, {'domain': domain})
    require(z3.is_int(lower) and z3.is_int(upper), 'integer caller boundary expressions required')
    prove(lower == lo, 'caller lower boundary equals domain low node', valid_domain)
    prove(upper == hi, 'caller upper boundary equals domain high node', valid_domain)
    prove(z3.Implies(member, z3.And(lower < k, k < upper)),
          'caller establishes strict wall-policy bounds', assumptions)
    prove(out[0][2] > 0, 'nonnegative domain origin satisfies positive-k assertions',
          assumptions + [lo >= 0])

    full = [list(domain[0]), [domain[1][a] + int(a == 2) for a in range(3)]]
    full_out = execute(program, full, domain)
    horizontal = z3.And(*[z3.And(domain[0][a] <= point[a], point[a] <= domain[1][a]) for a in (0, 1)])
    prove(z3.Not(contains(full_out, point)), 'one-cell domain has no interior Z nodes',
          valid_domain + [hi == lo + 1])
    prove(contains(full_out, point) == z3.And(horizontal, k == lo + 1),
          'two-cell domain retains its single interior Z plane', valid_domain + [hi == lo + 2])
    split = z3.Int('split_cell_hi')
    bottom = [list(full[0]), [full[1][0], full[1][1], split + 1]]
    top = [[full[0][0], full[0][1], split + 1], list(full[1])]
    bottom_out, top_out = execute(program, bottom, domain), execute(program, top, domain)
    split_assumptions = valid_domain + [lo <= split, split < hi - 1]
    union = z3.Or(contains(bottom_out, point), contains(top_out, point))
    prove(union == z3.And(horizontal, interior), 'arbitrary vertical split covers exactly the interior', split_assumptions)
    prove(union == contains(full_out, point), 'split and whole-domain coverage agree', split_assumptions)
    prove(z3.And(contains(bottom_out, point), contains(top_out, point)) ==
          z3.And(horizontal, k == split + 1), 'both boxes retain exactly their shared split plane', split_assumptions)
    print(f'\n{len(failures)} check(s) failed')
    return bool(failures)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, SyntaxError) as error:
        print(f'  FAIL source contract: {error}')
        sys.exit(1)
