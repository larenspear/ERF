"""Small, fail-closed source readers shared by the advection checks.

This is not a C++ parser. Callers describe the complete supported statement
sequence of a checked region; any unconsumed text is an error. Flux-building
regions explicitly abstracted by a checker remain outside its arithmetic proof.
Only the standard library is used so the reach check works without Z3.
"""
import ast
import re
from fractions import Fraction

CONST_H = "Source/ERF_NumericalConstants.H"
INITIALIZER = r"(?:amrex::)?Real\s+(\w+)\s*=\s*([^;{}]+);"
NORMALIZE = (r"if\s*\(\s*upw_lo\s*!=\s*amrex::Real\(0\)\s*\)\s*"
             r"upw_lo\s*=\s*\(\s*upw_lo\s*>\s*0\s*\)\s*\?\s*"
             r"amrex::Real\(1\)\s*:\s*-\s*amrex::Real\(1\)\s*;")
LAMBDA = r"\s*\[=\]\s*AMREX_GPU_DEVICE\s*\(\s*int i,\s*int j,\s*int k\s*\)\s*noexcept\s*\{"


class Source:
    def __init__(self, text):
        self.text = text
        self.pos = 0

    def take(self, pattern):
        match = re.compile(r"\s*(?:" + pattern + r")", re.S).match(self.text, self.pos)
        if match is None:
            raise ValueError(f"unsupported source near {self.text[self.pos:self.pos + 100].strip()!r}")
        self.pos = match.end()
        return match

    def done(self):
        if self.text[self.pos:].strip():
            raise ValueError(f"unmodeled statements: {self.text[self.pos:self.pos + 100].strip()!r}")

    def initializers(self):
        values = {}
        while re.match(r"\s*(?:amrex::)?Real\s+\w+\s*=", self.text[self.pos:]):
            name, expr = self.take(INITIALIZER).groups()
            if name in values:
                raise ValueError(f"duplicate initializer: {name}")
            pure_expression(expr)
            values[name] = expr.strip()
        return values


def pure_expression(expr):
    """Reject writes/control flow hidden inside otherwise supported expressions."""
    if re.search(r"(?<![=!<>])=(?!=)|\+\+|--|[;{}&#]", expr):
        raise ValueError(f"unsupported side effect or syntax in expression: {expr}")


def evaluation(block):
    """A sequence of real initializers followed by exactly one final return."""
    source = Source(block)
    assignments = source.initializers()
    result = source.take(r"return\s+([^;{}]+);").group(1)
    pure_expression(result)
    source.done()
    return assignments, result.strip()


def interpolation(block, dispatcher=False):
    """Consume the complete supported UPW interpolation statement sequence."""
    source = Source(block)
    if dispatcher:
        source.take(r"if\s*\(\s*adv_type\s*==\s*AdvType::Centered_2nd\s*\)\s*\{")
        early = source.take(r"val_lo\s*=\s*([^;{}]+);\s*return\s*;\s*\}\s*else\s*\{").group(1)
        pure_expression(early)
    # Limiter normalizes before loading; linear upwind schemes do so afterward.
    before = bool(re.match(r"\s*if\s*\(\s*upw_lo", source.text[source.pos:]))
    if before:
        source.take(NORMALIZE)
    loads = {}
    while re.match(r"\s*amrex::Real\s+s(?:[mp]\d+)?\s*=", source.text[source.pos:]):
        name, expr = source.take(INITIALIZER).groups()
        if name in loads:
            raise ValueError(f"duplicate stencil load: {name}")
        pure_expression(expr)
        read = r"m_phi\s*\([^()]+\)"
        gate = r"\([^?;{}]+\)\s*\?\s*" + read + r"\s*:\s*amrex::Real\(0\)"
        if not re.fullmatch(read, expr.strip()) and not (dispatcher and re.fullmatch(gate, expr.strip())):
            raise ValueError(f"unsupported stencil load: {expr}")
        loads[name] = expr
    if not loads:
        raise ValueError("missing interpolation loads")
    if re.match(r"\s*if\s*\(\s*upw_lo", source.text[source.pos:]):
        if before:
            raise ValueError("duplicate upwind normalization")
        source.take(NORMALIZE)
        source.take(r"upw_lo\s*\*=\s*m_upw_frac\s*;")
    bindings = {}
    while re.match(r"\s*amrex::Real\s+F\w+\s*=", source.text[source.pos:]):
        name, args = source.take(r"amrex::Real\s+(F\w+)\s*=\s*Evaluate\s*\(([^()]*)\)\s*;").groups()
        if name in bindings:
            raise ValueError(f"duplicate Evaluate destination: {name}")
        bindings[name] = tuple(arg.strip() for arg in args.split(','))
    result = source.take(r"val_lo\s*=\s*([^;{}]+);").group(1)
    pure_expression(result)
    if dispatcher:
        source.take(r"\}")
    source.done()
    # Every m_phi occurrence must belong to a checked load or the early return.
    if 'm_phi' in result or any('m_phi' in arg for args in bindings.values() for arg in args):
        raise ValueError("unmodeled array read outside stencil loads")
    if not bindings and not re.fullmatch(r"Evaluate\s*\(\s*\w+(?:\s*,\s*\w+)*\s*\)", result.strip()):
        raise ValueError("linear interpolation must store the complete Evaluate result")
    return bindings, result.strip()


def momentum_lambdas(branch):
    """Read one three-box launch, preserving the box-to-lambda association."""
    source = Source(branch)
    source.take(r"(?:amrex::)?ParallelFor\s*\(\s*bxx\s*,\s*bxy\s*,\s*bxz\s*,")
    bodies = []
    for index in range(3):
        source.take(LAMBDA)
        # Inline fast paths contain no nested statement blocks. Ternary
        # expressions are handled separately by the arithmetic checker.
        body = source.take(r"([^{}]*)\}").group(1)
        bodies.append(body)
        source.take(r"," if index < 2 else r"\)\s*;")
    source.done()
    return bodies


def momentum_component(body, rhs, stretched=False):
    source = Source(body)
    declarations = source.initializers()
    required = {f'{axis}flux_{edge}' for axis in 'xyz' for edge in ('hi', 'lo')}
    required |= {'mfsq', 'advectionSrc'} | ({'dzInv'} if stretched else set())
    if set(declarations) != required:
        raise ValueError(f"unexpected momentum locals: {sorted(set(declarations) ^ required)}")
    # Initializers must precede the source assignment; no reassignments or
    # additional writes are accepted after these declarations.
    names = list(declarations)
    if names[-1] != 'advectionSrc':
        raise ValueError("advectionSrc must follow all flux/metric/spacing definitions")
    value = source.take(rf"{rhs}\s*\(\s*i\s*,\s*j\s*,\s*k\s*\)\s*=\s*([^;{{}}]+);").group(1)
    pure_expression(value)
    source.done()


def spacing_aliases(body, dimensions):
    """Validate initializer provenance and prohibit later writes to aliases.

    Only the prefix before the selected execution path is passed here. Each
    occurrence of a spacing alias must be its declaration, avoiding accepted
    initializers followed by unnoticed assignments or reference escapes.
    """
    for index, name in enumerate(('dxInv', 'dyInv', 'dzInv')[:dimensions]):
        pattern = (rf"(?:\bauto\s+|,\s*){name}\s*=\s*cellSizeInv\s*\[\s*{index}\s*\]"
                   r"(?=\s*[,;])")
        matches = list(re.finditer(pattern, body))
        if len(matches) != 1 or len(re.findall(rf"\b{name}\b", body)) != 1:
            raise ValueError(f"{name} must be initialized only from cellSizeInv[{index}]")


def checked_constants(root, expected):
    """Read source constants as exact rationals and check independent values."""
    text = (root / CONST_H).read_text()
    text = re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)
    definitions = {}
    for match in re.finditer(r'\bconstexpr\s+amrex::Real\s+(\w+)\s*=\s*([^;]+);', text):
        name, expr = match.groups()
        if name in definitions:
            raise ValueError(f'duplicate numerical constant: {name}')
        definitions[name] = expr

    def evaluate(node, pending):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            # Use the source spelling, not a binary float conversion.
            return Fraction(ast.get_source_segment(pending[1], node))
        if isinstance(node, ast.Name):
            return resolve(node.id, pending[0])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand, pending)
            return -value if isinstance(node.op, ast.USub) else value
        if isinstance(node, ast.BinOp):
            left, right = evaluate(node.left, pending), evaluate(node.right, pending)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ValueError('division by zero in numerical constant')
                return left / right
        raise ValueError('unsupported numerical constant expression')

    def resolve(name, pending=()):
        if name not in definitions or name in pending:
            raise ValueError(f'missing or recursive numerical constant: {name}')
        expr = definitions[name].replace('amrex::Real', '').strip()
        return evaluate(ast.parse(expr, mode='eval').body, ((*pending, name), expr))

    result = {name: resolve(name) for name in expected}
    for name, value in result.items():
        if value != Fraction(expected[name]):
            raise ValueError(f'source constant {name} = {value}, expected {expected[name]}')
    return result


def mutation_outcome(result, mutated):
    """An assertion/explicit parse rejection counts; an unexpected crash does not."""
    failures = [line.strip() for line in result.stdout.splitlines()
                if line.strip().startswith('FAIL')]
    caught = result.returncode == 1 and bool(failures) and not result.stderr.strip()
    expected = caught if mutated else result.returncode == 0
    if result.returncode and not caught:
        failures.append('unexpected checker error: ' + (result.stderr.strip() or str(result.returncode)))
    return caught, expected, failures


def momentum_setup(prefix, stretched):
    """Consume the setup prefix and return its four inverse-map expressions."""
    compact = lambda text: re.sub(r"\s+", "", text)
    source = Source(compact(prefix))
    inverses = {}

    def literal(text):
        source.take(re.escape(compact(text)))

    routine = 'AdvectionSrcForMom_' + ('StretchedDz' if stretched else 'ConstantDz')
    literal(f'BL_PROFILE_VAR("{routine}", {routine});')
    literal('AMREX_ALWAYS_ASSERT(bxz.smallEnd(2) > 0);')
    if stretched:
        literal('auto dxInv = cellSizeInv[0]; auto dyInv = cellSizeInv[1];')
    else:
        literal('auto dxInv = cellSizeInv[0], dyInv = cellSizeInv[1], dzInv = cellSizeInv[2];')
        literal('const bool use_terrain_fitted_coords = '
                '(terrain_type == TerrainType::StaticFittedMesh || '
                'terrain_type == TerrainType::MovingFittedMesh);')
        literal('AMREX_ALWAYS_ASSERT(!use_terrain_fitted_coords && '
                '(terrain_type != TerrainType::EB));')

    for component, box in (('u', 'bxx'), ('v', 'bxy')):
        literal(f'Box box2d_{component}({box}); '
                f'box2d_{component}.setRange(2,0); box2d_{component}.grow({{3,3,0}});')
    for component in 'uv':
        for axis in 'xy':
            literal(f'FArrayBox mf_{component}{axis}_invFAB('
                    f'box2d_{component},1,The_Async_Arena());')
        for axis in 'xy':
            literal(f'const Array4<Real>& mf_{component}{axis}_inv = '
                    f'mf_{component}{axis}_invFAB.array();')

    literal('ParallelFor(box2d_u, box2d_v,')
    for component in 'uv':
        literal('[=] AMREX_GPU_DEVICE (int i, int j, int) noexcept {')
        for axis in 'xy':
            name = f'mf_{component}{axis}'
            literal(f'{name}_inv(i,j,0) =')
            expression = source.take(r'([^;{}]+);').group(1)
            pure_expression(expression)
            inverses[name] = expression
        literal('},' if component == 'u' else '});')
    if stretched:
        literal('auto dz_ptr = stretched_dz_d.data();')
    source.done()

    return inverses
