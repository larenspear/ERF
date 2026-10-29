# Spack's builtin amrex recipe, plus the one thing ERF needs that it lacks: a
# particle-data precision of its own (AMReX_PARTICLES_PRECISION), so that an
# ERF built with particles_precision different from precision, as ERF's own
# Windows CI does, has an AMReX to link against. The same change is proposed
# for spack/spack-packages (branch larenspear/spack-packages:amrex-particles-
# precision); once it is merged there, delete this file and the erf recipe's
# `amrex particles_precision=` dependencies keep working unchanged.
#
# Registering this repository therefore shadows builtin.amrex; the override
# changes nothing unless particles_precision is set to double or single.

from spack_repo.builtin.packages.amrex.package import Amrex as BuiltinAmrex

from spack.package import *


class Amrex(BuiltinAmrex):
    """AMReX, Spack's builtin recipe plus a particle-precision variant (see above)."""

    # AMReX_PARTICLES_PRECISION; AMReX's CMake defaults it to the mesh precision.
    variant(
        "particles_precision",
        default="same",
        values=("same", "double", "single"),
        multi=False,
        description="Precision of the particle data ('same' follows precision)",
        when="+particles",
    )

    def cmake_args(self):
        args = super().cmake_args()
        if self.spec.satisfies("+particles"):
            precision = self.spec.variants["particles_precision"].value
            if precision == "same":
                precision = self.spec.variants["precision"].value
            args.append(self.define("AMReX_PARTICLES_PRECISION", precision.upper()))
        return args
