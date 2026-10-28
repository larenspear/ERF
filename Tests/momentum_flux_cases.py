"""Source edits shared by the constant/stretched momentum mutation runners.

These cases only locate C++ statements; they do not use the proof's geometric
specification. Every one of the eighteen flux initializers gets a scaling fault.
Selected cases require the new shared-face proofs to produce their own solver
results, so another assertion cannot mask a missing or inconclusive check.
"""
import re


def flux_cases(source):
    statements = re.findall(r'\bReal\s+([xyz]flux_(?:hi|lo))\s*=\s*([^;]+);', source)
    if len(statements) != 18:
        raise ValueError(f'expected eighteen inline flux initializers, found {len(statements)}')
    located = {}
    for index, (name, expr) in enumerate(statements):
        component = 'XYZ'[index // 6]
        located[component, name] = expr
    required = {(component, f'{axis}flux_{edge}')
                for component in 'XYZ' for axis in 'xyz' for edge in ('hi', 'lo')}
    if set(located) != required:
        raise ValueError('unexpected flux initializer layout')

    mutations, controls, diagnostics = {}, {}, {}

    def face_results(component, axis, agrees, cancels):
        return tuple(f'{"ok" if passed else "FAIL"} {component}.{axis}: shared interior face '
                     f'{property_name} [{"unsat" if passed else "sat"}]'
                     for property_name, passed in (('flux agrees', agrees),
                                                   ('cancels in weighted RHS', cancels)))

    diagnostics['baseline (no change)'] = tuple(
        line for component in 'XYZ' for axis in 'xyz'
        for line in face_results(component, axis, True, True))
    # Altering an external wall flux must not invalidate the interior identities.
    diagnostics['z top flux is averaged instead of using the wall value'] = face_results(
        'Z', 'z', True, True)

    def edit(label, original, replacement, target=mutations):
        if original == replacement or source.count(original) != 1:
            raise ValueError(f'{label}: expected one changed, uniquely located expression')
        target[label] = (original, replacement, 0, 1)

    for (component, name), expr in located.items():
        if expr.count('fourth') != 1:
            raise ValueError(f'{component}.{name}: expected one averaging coefficient')
        label = f'{component}.{name} reconstruction is doubled'
        edit(label, expr, expr.replace('fourth', '(2 * fourth)'))
        diagnostics[label] = face_results(component, name[0], False, False)

    for component, axis in zip('XYZ', 'xyz'):
        # Scaling BOTH sides is still conservative, but violates the previously
        # checked reconstruction rule. This distinguishes the two properties.
        pair = [located[component, f'{axis}flux_{edge}'] for edge in ('hi', 'lo')]
        start = min(source.index(expr) for expr in pair)
        end = max(source.index(expr) + len(expr) for expr in pair)
        original = source[start:end]
        changed = original
        for expr in pair:
            changed = changed.replace(expr, expr.replace('fourth', '(2 * fourth)'))
        label = f'{component}.{axis} both face reconstructions are doubled'
        edit(label, original, changed)
        diagnostics[label] = face_results(component, axis, True, True)

        # A coefficient defect leaves the shared flux unchanged but breaks its
        # cancellation between the two RHS values.
        original = f'({axis}flux_hi - {axis}flux_lo)'
        if source.count(original) != 3:
            raise ValueError(f'{component}.{axis}: expected three divergence terms')
        label = f'{component}.{axis} divergence adds the two face fluxes'
        mutations[label] = (original, original.replace(' - ', ' + '), 'XYZ'.index(component), 3)
        diagnostics[label] = face_results(component, axis, True, False)

    for label, component, name, pattern, replacement in (
        ('X y-flux transports the wrong mass component', 'X', 'yflux_hi', r'\brho_v\b', 'rho_u'),
        ('Y low z-flux reads velocity above instead of below', 'Y', 'zflux_lo',
         r'v\(i,j,k-1\)', 'v(i,j,k+1)'),
        ('Z low x-flux reads mass above instead of below', 'Z', 'xflux_lo', r'k-1', 'k+1'),
        ('Z y-flux uses the wrong inverse map', 'Z', 'yflux_hi', r'mf_vx_inv', 'mf_vy_inv'),
        ('X low x-flux uses the central map at its neighbor', 'X', 'xflux_lo',
         r'mf_uy_inv\(i-1,j,0\)', 'mf_uy_inv(i,j,0)'),
        ('X high z-flux uses the wrong transverse neighbor', 'X', 'zflux_hi', r'i-1', 'i+1'),
    ):
        expr = located[component, name]
        changed, count = re.subn(pattern, replacement, expr, count=1)
        if count != 1:
            raise ValueError(f'{label}: mutation target missing')
        edit(label, expr, changed)

    for name in ('mf_ux', 'mf_uy', 'mf_vx', 'mf_vy'):
        original = f'{name}_inv(i,j,0) = one / {name}(i,j,0);'
        edit(f'{name} inverse setup multiplies instead of divides', original,
             original.replace('one /', 'one *'))
    edit('inverse map aliases the wrong buffer',
         'mf_uy_inv = mf_uy_invFAB.array();', 'mf_uy_inv = mf_ux_invFAB.array();')
    edit('inverse map component launch boxes are swapped',
         'ParallelFor(box2d_u, box2d_v,', 'ParallelFor(box2d_v, box2d_u,')
    original = 'mf_uy_inv(i,j,0) = one / mf_uy(i,j,0);'
    edit('inverse map is negated after initialization', original,
         original + '\n        mf_uy_inv(i,j,0) *= -1;')

    expr = located['X', 'xflux_hi']
    edit('equivalent averaging as two halves', expr,
         expr.replace('fourth', '(1.0 / 2) * (1.0 / 2)'), controls)
    edit('equivalent reordered velocity average', expr,
         expr.replace('(u(i+1,j,k) + u(i,j,k))', '(u(i,j,k) + u(i+1,j,k))'), controls)
    edit('equivalent reciprocal numerator and denominator', original,
         'mf_uy_inv(i,j,0) = (one + one) / (2 * mf_uy(i,j,0));', controls)
    return mutations, controls, diagnostics


def check_diagnostics(result, required):
    lines = {re.sub(r'\s+', ' ', line.strip()) for line in result.stdout.splitlines()}
    return [f'missing required solver result: {line}' for line in required if line not in lines]
