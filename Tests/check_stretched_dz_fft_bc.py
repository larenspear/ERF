#!/usr/bin/env python3
"""Check FFT face classification and its existing builder bindings.

Consume get_fft_bc's complete supported signature/body, including all three
directions and both output slots. Parse its pure Boolean conditions and exhaust
their input partition: periodicity, contact with each domain end, direction,
real lateral BCs and every string literal compared by the source plus an other
class. Compare outputs with an independent per-face specification. Preserve
the previous full builder contract and inventory its explicit helper call.

The specification gives periodic precedence at touching periodic faces; odd at
touching Open/Outflow faces when vertical or not using real BCs; even otherwise.
For a lateral axis of length > 1, the inspected hybrid constructor requires
both faces periodic or neither. The table establishes compatibility iff the
axis is nonperiodic or both ends have the same contact status. The vertical
constructor restriction requires neither face periodic. These restrictions
are conditional properties, not claims that every ERF subdomain satisfies them.

Assume AMREX_SPACEDIM == 3, valid cell boxes/storage, ordinary C++/AMReX accessor
and string semantics, intended name resolution, no external macro rewriting,
concurrency or undefined behavior, and stable builder inputs. Integer geometry
and Boolean/string abstractions are exact under those premises. No arithmetic,
PDE boundary-correctness, subdomain-reachability, state-freshness, GPU-lifetime
or whole-program theorem; the TerrainPoisson caller is outside this contract.
Dependency semantics are inspected separately. Only the standard library is used.
"""
import itertools
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import CLASS, brace_block, no_literals, reject_uses, scopes, signature
from check_stretched_dz_fft_bindings import SOURCE, builder, pattern
from check_stretched_dz_wrf_callers import pp_stack
from momentum_geometry import unique

UTILS = 'Source/LinearSolvers/ERF_SolverUtils.H'
FILES = (UTILS, SOURCE, CLASS)
TYPE = 'Array<std::pair<FFT::Boundary,FFT::Boundary>,AMREX_SPACEDIM>'
SIGNATURE = '''inline '''+TYPE+''' get_fft_bc(Geometry const& lev_geom,
    Array<std::string,2*AMREX_SPACEDIM> l_domain_bc_type,
    Box const& bounding_box, bool use_real_bcs) noexcept {'''
LOOP = 'for (int dir = 0; dir <= 2; dir++) {'
CALL = 'auto bc_fft = get_fft_bc(geom[lev],domain_bc_type,subdomain,solverChoice.use_real_bcs);'


def take(source, code):
    # At an already-consumed token boundary, punctuation may immediately follow
    # the previous token. Identifier boundaries within the pattern still apply.
    source.take(pattern(code).removeprefix(r'(?<![\w:.>])'))


class Expression:
    """Small pure Boolean grammar; source calls/indices have fixed bindings."""
    def __init__(self, text):
        self.source = Source(text)
        self.strings = set()

    def accept(self, token):
        match = re.match(r'\s*'+pattern(token), self.source.text[self.source.pos:])
        if match:
            self.source.pos += match.end()
        return bool(match)

    def atom(self, negated=False):
        if self.accept('!'):
            return ('not', self.atom(negated=True))
        if self.accept('('):
            result = self.disjunction()
            take(self.source, ')')
            return result
        for token, node in (('true', ('const', True)), ('false', ('const', False)),
                            ('lev_geom.isPeriodic(dir)', ('value', 'periodic')),
                            ('use_real_bcs', ('value', 'real'))):
            if self.accept(token):
                return node
        # C++ ! binds more tightly than ==. Without parentheses, !dir != 2
        # negates the integer before comparing; it is not !(dir != 2).
        # Only Boolean leaves and parenthesized predicates support unary !.
        require(not negated, 'negated FFT comparison requires parentheses')
        for endpoint, name in (('smallEnd','low'), ('bigEnd','high')):
            for operator in ('==', '!='):
                if self.accept('lev_geom.Domain().'+endpoint+'(dir) '+operator+' bounding_box.'+endpoint+'(dir)'):
                    node = ('value', name)
                    return node if operator == '==' else ('not', node)
        match = re.match(r'\s*'+pattern('dir')+r'\s*(==|!=)\s*([012])\b', self.source.text[self.source.pos:])
        if match:
            self.source.pos += match.end()
            return ('compare', 'dir', match[1], int(match[2]))
        match = re.match(r'\s*\bbc_type_(lo|hi)\b\s*(==|!=)\s*"([A-Za-z_][A-Za-z_0-9]*)"',
                         self.source.text[self.source.pos:])
        if match:
            self.source.pos += match.end()
            self.strings.add(match[3])
            return ('compare', match[1], match[2], match[3])
        raise ValueError('unsupported FFT boundary condition near '+repr(self.source.text[self.source.pos:self.source.pos+80]))

    def conjunction(self):
        node = self.atom()
        while self.accept('&&'):
            node = ('and', node, self.atom())
        return node

    def disjunction(self):
        node = self.conjunction()
        while self.accept('||'):
            node = ('or', node, self.conjunction())
        return node

    def parse(self):
        node = self.disjunction()
        self.source.done()
        return node


def evaluate(node, inputs):
    if node[0] == 'const': return node[1]
    if node[0] == 'value': return inputs[node[1]]
    if node[0] == 'not': return not evaluate(node[1], inputs)
    if node[0] == 'and': return evaluate(node[1], inputs) and evaluate(node[2], inputs)
    if node[0] == 'or': return evaluate(node[1], inputs) or evaluate(node[2], inputs)
    require(node[0] == 'compare', 'unsupported Boolean node')
    equal = inputs[node[1]] == node[3]
    return equal if node[2] == '==' else not equal


def condition(source):
    take(source, 'if (')
    start, depth = source.pos, 1
    masked = no_literals(source.text)
    for pos in range(start, len(source.text)):
        char = masked[pos]
        depth += (char == '(') - (char == ')')
        if depth == 0:
            expr = Expression(source.text[start:pos])
            node = expr.parse()
            source.pos = pos+1
            take(source, '{')
            return node, expr.strings
    raise ValueError('unterminated FFT boundary condition')


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        text = strip_comments(raw)
        guard = {UTILS:'ERF_SOLVER_UTILS_H_', CLASS:'ERF_H_'}.get(path)
        macros = list(re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)))
        require(len(macros) == int(guard is not None), path+': unexpected macro inventory')
        for macro in macros:
            require(macro[1] == 'define' and macro[2].strip() == guard, path+': unsupported macro')
        if guard:
            take(Source(text), '#ifndef '+guard+' #define '+guard)
        require(not pp_stack(text), path+': unclosed preprocessor scope')
        texts[path] = text
    return texts


def helper(text):
    definition = unique(pattern(SIGNATURE), text, 'FFT boundary helper signature')
    require(not scopes(text, definition.start()), 'FFT boundary helper must be global')
    require(pp_stack(text[:definition.start()]) == [('ifndef','ERF_SOLVER_UTILS_H_'), ('ifdef','ERF_USE_FFT')],
            'FFT boundary helper requires header and FFT guards')
    body = brace_block(text, definition.end()-1)
    reject_uses(text, ['get_fft_bc'], [(definition.start(), definition.end()+len(body)+1)], 'FFT helper inventory')
    source = Source(body)
    take(source, TYPE+' r; FFT::Boundary first, second; '+LOOP)
    branches, strings = [], set()
    for side, slot, orientation in (('lo','first','low'), ('hi','second','high')):
        take(source, 'auto bc_type_'+side+' = l_domain_bc_type[Orientation(dir,Orientation::'+orientation+')];')
        cases = []
        for branch in range(3):
            node = ('const', True)
            if branch: take(source, 'else')
            if branch < 2:
                node, literals = condition(source)
                strings |= literals
            else:
                take(source, '{')
            take(source, slot+' = FFT::Boundary::')
            value = source.take(r'(periodic|odd|even)\b').group(1)
            take(source, '; }')
            cases.append((node,value))
        branches.append(cases)
    take(source, 'r[dir] = std::make_pair(')
    first = source.take(r'(first|second)\b').group(1)
    take(source, ',')
    second = source.take(r'(first|second)\b').group(1)
    take(source, '); } return r;')
    source.done()
    print('  ok   complete FFT helper interface, three-direction loop and both face assignments')
    return branches, strings, (first,second)


def table(model):
    branches, strings, order = model
    classes = set(strings) | {'Outflow', 'Open'}
    other = '__other__'
    while other in classes: other += '_'
    classes = sorted(classes | {other})
    count = mixed = vertical = 0
    for direction, periodic, low, high, real, lo, hi in itertools.product(
            range(3), (False,True), (False,True), (False,True), (False,True), classes, classes):
        inputs = dict(dir=direction, periodic=periodic, low=low, high=high, real=real, lo=lo, hi=hi)
        values = [next(value for node,value in cases if evaluate(node,inputs)) for cases in branches]
        result = tuple(values[('first','second').index(slot)] for slot in order)
        # Independent truth table: no source-derived conditions or outputs.
        expected = []
        for touches, name in ((low,lo),(high,hi)):
            value = 'even'
            if touches:
                if periodic: value = 'periodic'
                elif name in ('Open','Outflow') and (direction == 2 or not real): value = 'odd'
            expected.append(value)
        require(result == tuple(expected), 'FFT boundary classification mismatch: '+str(inputs)+' -> '+str(result)+' expected '+str(expected))
        pair_ok = (result[0] == 'periodic') == (result[1] == 'periodic')
        require(pair_ok == (not periodic or low == high), 'lateral periodic compatibility relation changed')
        vertical_ok = all(value != 'periodic' for value in result)
        require(vertical_ok == (not periodic or not (low or high)), 'vertical periodic compatibility relation changed')
        mixed += direction < 2 and not pair_ok
        vertical += direction == 2 and not vertical_ok
        count += 1
    print(f'  ok   {count} exhaustive input classes match the face specification and conditional periodic constraints')
    print(f'  note hybrid restrictions exclude {mixed} mixed-periodic lateral classes (length > 1) and {vertical} vertical-periodic classes')
    return count, mixed, vertical


def construction(texts):
    # Reuse the complete builder contract, including Geometry, capacities,
    # solver slots and exact bc_fft forwarding to both constructor types.
    builder(texts)
    _, body, definition = signature(texts[SOURCE], 'ERF::build_fft_solvers')
    call = unique(pattern(CALL), body, 'builder boundary call')
    offset = definition.end()
    reject_uses(texts[SOURCE], ['get_fft_bc'], [(offset+call.start(), offset+call.end())], 'builder-file helper inventory')
    reject_uses(texts[CLASS], ['get_fft_bc'], [], 'ERF member shadowing of boundary helper')
    print('  ok   same-level Geometry/BC names/solve box/real flag reach both FFT constructors')


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        table(helper(texts[UTILS]))
        construction(texts)
    except (ValueError, OSError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('3 FFT boundary source/model contracts passed; periodic admissibility and lifetime remain conditional.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
