"""Read WENO interpolation and compose its checked scalar kernel modularly.

The existing candidate/weight proofs check the complete scalar Evaluate body.
The transport proof treats that pure kernel as an arbitrary function of its
ordered samples. A universal function identity also holds for the checked
nonlinear expression, without expanding its large rational formulas at every
momentum call site. Counterexamples to that identity are interface-model
counterexamples, not necessarily numerical witnesses for a specific kernel.
"""
import re

from advection_source import NORMALIZE, Source, pure_expression
from check_momentum_dispatch import compact, require
import check_weno_candidates as scalar
from check_upwindall_z_reach import brace_block
from linear_z_interpolation import LinearZ, sign

z3 = scalar.z3
WENO = {'Weno_3': 'WENO3', 'Weno_3Z': 'WENO_Z3', 'Weno_3MZQ': 'WENO_MZQ3',
        'Weno_5': 'WENO5', 'Weno_5Z': 'WENO_Z5', 'Weno_7': 'WENO7', 'Weno_7Z': 'WENO_Z7'}


class WenoZ:
    # Keep the original name/default for the existing vertical checker.
    def __init__(self, name, body, direction='Z'):
        require(direction in ('X', 'Y', 'Z'), 'unsupported interpolation direction')
        self.direction = direction
        self.name = name
        self.radius = (int(name[-1]) - 1) // 2
        self.canonical = ['s' if offset == 0 else f's{"p" if offset > 0 else "m"}{abs(offset)}'
                          for offset in range(-self.radius, self.radius + 1)]
        constructors = re.findall(rf'\b{name}\s*\(([^()]*)\)\s*:\s*([^{{}}]*)\{{([^{{}}]*)\}}', body)
        require(len(constructors) == 1, f'{name}: expected one constructor')
        params, initializers, constructor_body = constructors[0]
        require(compact(params) == 'constamrex::Array4<constamrex::Real>&phi,constamrex::Real',
                f'{name}: constructor parameters changed')
        require(compact(initializers) == 'm_phi(phi)' and not constructor_body.strip(),
                f'{name}: constructor must bind phi and ignore the blend fraction')

        signatures = list(re.finditer(rf'void\s+InterpolateIn{direction}\s*\(([^()]*)\)\s*const\s*\{{', body))
        require(len(signatures) == 1, f'{name}: expected one {direction} method')
        require(compact(signatures[0][1]) ==
                'constint&i,constint&j,constint&k,constint&qty_index,amrex::Real&val_lo,amrex::Realupw_lo',
                f'{name}: {direction} method parameter bindings changed')
        source = Source(brace_block(body, signatures[0].end() - 1))
        source.take(NORMALIZE)
        # Preserve declaration order, including the binding of each Evaluate
        # result to Fhi/Flo. Reject late writes and unmodeled expressions.
        self.declarations = []
        available = set()
        for local, expression in source.initializers().items():
            if re.fullmatch(r's(?:[mp]\d+)?', local):
                self.declarations.append(('load', local, LinearZ.read_offsets(expression)))
            else:
                require(local in ('Fhi', 'Flo'), f'{name}: unsupported local {local}')
                match = re.fullmatch(r'Evaluate\s*\(\s*(\w+(?:\s*,\s*\w+)*)\s*\)', expression)
                require(match is not None, f'{name}: unsupported Evaluate call')
                args = [arg.strip() for arg in match[1].split(',')]
                require(set(args) <= available, f'{name}: Evaluate uses an undeclared sample')
                require(all(re.fullmatch(r's(?:[mp]\d+)?', arg) for arg in args),
                        f'{name}: Evaluate needs scalar samples')
                self.declarations.append(('call', local, args))
            available.add(local)
        require({'Fhi', 'Flo'} <= available, f'{name}: missing one-sided reconstruction')
        self.result = source.take(r'val_lo\s*=\s*([^;{}]+);').group(1)
        pure_expression(self.result)
        source.done()

        signatures = list(re.finditer(r'amrex::Real\s+Evaluate\s*\(([^()]*)\)\s*const\s*\{', body))
        require(len(signatures) == 1, f'{name}: expected one scalar Evaluate method')
        self.params, env, assignments, raw, returned = scalar.parse_evaluate(name, body)
        require(len(self.params) == len(self.canonical) and set(self.params) == set(self.canonical),
                f'{name}: Evaluate sample parameters changed')
        # Exclude shadowed input parameters/constants: the scalar reader must
        # represent a pure function of precisely these samples and fixed eps.
        require(not set(assignments) & (set(self.params) | {'eps'}),
                f'{name}: Evaluate shadows an input')
        for kind, _, args in self.declarations:
            if kind == 'call':
                require(len(args) == len(self.params), f'{name}: Evaluate arity changed')
        self.kernel = z3.Function(name + '_Evaluate', *([z3.RealSort()] * (len(self.params) + 1)))

        # These are prerequisites, run here as well as in the standalone suite.
        # They prove candidate support/moments, independent beta derivative
        # integrals and ideal reconstruction, beta PSD, positive weights and
        # complete return normalization (including the MZQ correction), plus
        # definedness and actual nonlinear exactness on constant/affine fields.
        expected = scalar.EXPECTED[name]
        candidates = {key: value for key, value in assignments.items() if re.fullmatch(r'v\d+', key)}
        require(set(candidates) == set(expected), f'{name}: candidate set changed')
        scale = scalar.candidate_scale(name, body)
        for label, order in expected.items():
            scalar.prove_candidate(name, label, candidates[label], env, self.params, order, scale,
                                   scalar.EXPECTED_SUPPORT[name][label])
        scalar.prove_weights(name, body, assignments, raw, env, self.params, returned, len(expected), scale)

    def formula(self, field, i, j, q, component, flow):
        env = {'upw_lo': sign(flow)}
        for kind, local, value in self.declarations:
            if kind == 'load':
                dx, dy, dz = value
                env[local] = field(i + dx, j + dy, q + dz, component)
            else:
                bound = dict(zip(self.params, [env[arg] for arg in value]))
                env[local] = self.kernel(*[bound[param] for param in self.canonical])
        try:
            return scalar.to_z3(self.result, env)
        except NameError as exc:
            raise ValueError(f'{self.name}: unsupported interpolation result dependency: {exc}') from exc

    def reference(self, field, i, j, q, component, flow):
        # Evaluate's center is the cell to the left of the face for positive
        # flow; negative flow reflects the sample order around that face.
        def sample(offset):
            point = [i, j, q]
            point['XYZ'.index(self.direction)] += offset
            return field(*point, component)
        offsets = range(-self.radius, self.radius + 1)
        left = self.kernel(*[sample(offset - 1) for offset in offsets])
        right = self.kernel(*[sample(-offset) for offset in offsets])
        return z3.If(flow > 0, left, z3.If(flow < 0, right, (left + right) / 2))
