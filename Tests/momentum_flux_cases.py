"""Source edits shared by the constant/stretched momentum mutation runners.

These cases only locate C++ statements; they do not use the proof's geometric
specification. Every one of the eighteen flux initializers gets a scaling fault.
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

    mutations, controls = {}, {}

    def edit(label, original, replacement, target=mutations):
        if original == replacement or source.count(original) != 1:
            raise ValueError(f'{label}: expected one changed, uniquely located expression')
        target[label] = (original, replacement, 0, 1)

    for (component, name), expr in located.items():
        if expr.count('fourth') != 1:
            raise ValueError(f'{component}.{name}: expected one averaging coefficient')
        edit(f'{component}.{name} reconstruction is doubled', expr,
             expr.replace('fourth', '(2 * fourth)'))

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
    return mutations, controls
