# ERF's Spack package. It lives in the ERF repository so that it changes
# together with ERF's CMake options and submodule pins; CI
# (.github/workflows/spack-package.yml) checks the pins against HEAD, sweeps
# the variant space through the concretizer and builds representative
# configurations. See Docs/sphinx_doc/buildingSpack.rst.
#
# Register it with `spack repo add <ERF>/Build/spack/spack_repo/erf` (any
# Spack 1.x) or `spack repo add https://github.com/erf-model/ERF.git`
# (Spack 1.1+). Spack's builtin repository (spack/spack-packages) must be
# registered too: this recipe uses its build systems and its amrex, mpi,
# netcdf and googletest packages. A +particles build of ERF 26.06 or older
# needs a spack-packages checkout from 2026-10-19 or later, whose amrex
# recipe installs the AMReX_TracerParticle_mod_K.H header ERF includes.

import itertools
import os

from spack_repo.builtin.build_systems.cmake import CMakePackage
from spack_repo.builtin.build_systems.cuda import CudaPackage
from spack_repo.builtin.build_systems.rocm import ROCmPackage

from spack.package import *


class Erf(CMakePackage, CudaPackage, ROCmPackage):
    """ERF solves the compressible Navier-Stokes equations on an Arakawa C-grid
    for large-scale weather modeling.
    """

    homepage = "https://erf.readthedocs.io/en/latest/index.html"
    url = "https://github.com/erf-model/ERF/archive/refs/tags/26.09.tar.gz"
    git = "https://github.com/erf-model/ERF.git"

    maintainers("larenspear", "jmsexton03")

    license("BSD-3-Clause", checked_by="larenspear")

    # Each ERF release vendors a same-numbered AMReX submodule (ERF YY.MM tracks
    # AMReX YY.MM) and follows AMReX's fast-moving API, which offers no
    # cross-release compatibility guarantee: e.g. ERF 25.12 uses amrex::*::Build
    # overloads missing from amrex@25.02 and the ArrayND ncomp() accessor removed
    # by amrex@26.04. Declare each release together with its matching AMReX pin
    # so a new version cannot be added without one. The development branch
    # builds the AMReX commit its submodule points at, as ERF's own build does:
    # amrex@develop with Spack's commit variant set to the gitlink (a git-commit
    # version in a depends_on is not a candidate for the concretizer, the variant
    # is); check_pins.py keeps it equal to the gitlink and rewrites it with --fix.
    _amrex_develop = "5308689e82930103102c6c07dca7c7b5f08e4344"
    version("develop", branch="development")
    depends_on(f"amrex@develop commit={_amrex_develop}", when="@develop", type=("build", "link"))

    for _v, _sha in (
        ("26.09", "d86372f6b9d85996f1b30beb7353c01b33f1bf0e454c0e63c913b255e17dadc0"),
        ("26.06", "379ffdebb30242cacddde95357511d5de1da29332c6624d5470088927743db78"),
        ("26.05", "198f8093b7afb2e405f509beca3b6a6da94c299af564e889e3c80daecc376e65"),
        ("26.04", "93a7ef8e6b55c4f830fac85129d6cf36c0b0e931eb060d7f8a1579afea8f909d"),
        ("26.03", "90399f4a7904e2b0d3c8e785484bebb89f2a5c46826fa4d751f939c0678d2d47"),
        ("26.02", "22b6570b9f022e58e0952801a85d4634b3076b40953422e0bd4df3de1d22e44d"),
        ("26.01", "1b40ec52b225e6bed1455f074cec04db1337121683ae1300c3adb3d048596d4d"),
        ("25.12", "0d11389c59561dea51f6b12253e1af7ab157599e0657bde603db7c102713e158"),
        ("25.11", "bd4160686eb52a49173ba07fe68ef86c2cd20b17283eaacf01b6c2d61a82982f"),
        ("25.10", "92575d25a8e87266f1b3a9a0c16a5755b84c45d489a060abeec6ddea8b8d8fe0"),
        ("25.08", "c8723384e00fb4bbb694385bdbc5187f38ee1509a0a18a23efc1094212fd21f4"),
        ("25.07", "bee8c136872c4e5400fdf2f6a432c8dc17f9b8384715158d9780a2c561a51642"),
        ("25.06", "dd1d627faf67477e3fa1f58b006f7e2b00316c66e8c99baf8b6cfbedce37b5f2"),
        ("25.05", "eb56a44d915d6af3ad30438dff6b96d9c489ae7c610fe4cdd0c17f2f26cc4b28"),
        ("25.04", "fc6837b252c1cebd0f2d38cdb3cea8ad55c837e4163c76e5d3508d973b282a42"),
        ("25.03", "3068c0c5f66538bcc1a12ee8c3eb142e5d02b73b71ab1cfbb244a053e8d5cf4a"),
        ("25.01", "11e48aadfc420c3f9adb05a0c5778776d6cb0915476532fd7a095d9f167ef584"),
        ("24.11", "19e7ce4829a46c98baf55b1dc4b2cefc508f2db41182e47c0ca788fe9280783f"),
    ):
        version(_v, sha256=_sha)
        depends_on(f"amrex@{_v}", when=f"@{_v}", type=("build", "link"))

    # ERF 24.10's AMReX submodule is five commits past amrex@24.10, and
    # Source/Utils/ERF_ChopGrids.cpp calls amrex::decompose(), which one of
    # those commits added: amrex@24.11 is the first release that declares it.
    version("24.10", sha256="7c9ba35374e71103fab8c5ba3ea107b4574e3270038bc3e8cbdcbe30eba26e7a")
    depends_on("amrex@24.11", when="@24.10", type=("build", "link"))

    # Releases before 24.10 always build their AMReX submodule (there is no
    # ERF_USE_INTERNAL_AMREX switch) and predate ERF's find_package(AMReX
    # COMPONENTS) integration, so they are not offered here.

    variant("mpi", default=True, description="Enable MPI support")
    variant("openmp", default=False, description="Enable OpenMP support")
    variant("sycl", default=False, description="Enable the SYCL GPU backend (needs %oneapi)")
    variant("netcdf", default=False, description="Enable NetCDF I/O (parallel, needs MPI)")
    # ERF's CMake defaults ERF_ENABLE_HDF5 to the NetCDF setting; keep that default
    # while leaving both combinations available.
    variant("hdf5", default=True, description="Enable AMReX HDF5 plotfile I/O", when="+netcdf")
    variant("hdf5", default=False, description="Enable AMReX HDF5 plotfile I/O", when="~netcdf")
    variant("particles", default=False, description="Enable Lagrangian particles")
    variant("multiblock", default=False, description="Enable multiblock support")
    # ERF_ENABLE_FFT appeared in 25.01 (AMReX's FFT module in amrex@24.11).
    variant("fft", default=False, description="Enable the AMReX FFT-based solvers", when="@25.01:")
    variant("tests", default=False, description="Build the regression and unit tests")
    variant("fcompare", default=False, description="Build AMReX's fcompare alongside ERF")
    # Single precision is selectable from 26.06, where the AMReX components ERF
    # requests follow ERF_PRECISION; earlier releases hard-code double.
    variant(
        "precision",
        default="double",
        values=("double",),
        multi=False,
        description="Floating point precision of the mesh and particle data",
        when="@:26.05",
    )
    variant(
        "precision",
        default="double",
        values=("double", "single"),
        multi=False,
        description="Floating point precision of the mesh and particle data",
        when="@26.06:",
    )
    # Each of these appeared in ERF's CMake with the release named here.
    variant("tools", default=False, description="Build the additional tools", when="@25.07:")
    variant(
        "rrtmgp",
        default=False,
        description="Enable RTE-RRTMGP radiation (builds EKAT and Kokkos)",
        when="@25.10:",
    )
    variant(
        "noahmp",
        default=False,
        description="Enable the Noah-MP land surface model",
        when="@25.12:",
    )
    variant(
        "windfarm",
        default=False,
        description="Enable the wind farm parameterizations",
        when="@26.02:",
    )
    variant(
        "ml_uphys_diagnostics",
        default=False,
        description="Enable the ML microphysics diagnostics in the SDM particle microphysics",
        when="@26.02:",
    )
    variant(
        "wsm6_fort",
        default=False,
        description="Enable the WSM6 Fortran microphysics bridge",
        when="@26.05:",
    )
    variant(
        "wdm6_fort",
        default=False,
        description="Enable the WDM6 Fortran microphysics bridge",
        when="@26.09:",
    )
    variant(
        "eamxx_shoc",
        default=False,
        description="Build the EAMxx SHOC turbulence parameterization (EKAT, Kokkos and "
        "a sparse checkout of E3SM's EAMxx sources at the commit ERF pins)",
        when="@26.09:",
    )
    # ERF_ENABLE_FIRE is on the development branch only at the time of writing;
    # the first release to carry it will be 26.10 or later.
    variant(
        "fire",
        default=False,
        description="Enable the fire coupling (atmosphere side; fire engines are external)",
        when="@26.10:",
    )
    # ERF 26.06 split the gtest unit tests from the regression tests
    # (ERF_ENABLE_UNIT_TESTS, on by default whenever the tests are on).
    variant(
        "unit_tests",
        default=True,
        description="Build the GoogleTest unit tests alongside the regression tests",
        when="@26.06:+tests",
    )
    # ERF 26.06 added ERF_PARTICLES_PRECISION, defaulting to ERF_PRECISION.
    variant(
        "particles_precision",
        default="same",
        values=("same", "double", "single"),
        multi=False,
        description="Floating point precision of the particle data ('same' follows precision)",
        when="@26.06:",
    )

    with default_args(type="build"):
        depends_on("cmake@3.20:")
        depends_on("c")
        depends_on("cxx")
        # ERF's project() enables Fortran only for the Fortran-based parameterizations;
        # EKAT's own project() enables it for RRTMGP builds.
        depends_on("fortran", when="+noahmp")
        depends_on("fortran", when="+wsm6_fort")
        depends_on("fortran", when="+wdm6_fort")
        depends_on("fortran", when="+rrtmgp")
        depends_on("fortran", when="+eamxx_shoc")
        # The Noah-MP ERF driver generates part of its interface with Python (26.09+).
        depends_on("python@3:", when="+noahmp")
        # ERF's FindNetCDF tries pkg-config before nc-config.
        depends_on("pkgconfig", when="+netcdf")
        # ERF 26.06 added gtest-based unit tests, enabled by default alongside the
        # regression tests. Supply GoogleTest from Spack rather than the bundled
        # Submodules/googletest checkout (see patch()).
        depends_on("googletest", when="@26.06:+tests+unit_tests")

    with default_args(type=("build", "link")):
        # AMReX must be built for the same programming model as ERF.
        for _v in ("mpi", "openmp", "cuda", "rocm", "sycl"):
            depends_on(f"amrex+{_v}", when=f"+{_v}")
            depends_on(f"amrex~{_v}", when=f"~{_v}")
        for _sm in CudaPackage.cuda_arch_values:
            depends_on(f"amrex+cuda cuda_arch={_sm}", when=f"+cuda cuda_arch={_sm}")
        for _tgt in ROCmPackage.amdgpu_targets:
            depends_on(f"amrex+rocm amdgpu_target={_tgt}", when=f"+rocm amdgpu_target={_tgt}")

        # ERF only includes AMReX's HDF5 and FFT headers; the libraries behind them
        # are AMReX's dependencies.
        depends_on("amrex+hdf5", when="+hdf5")
        depends_on("amrex+fft", when="+fft")
        # ERF.H includes AMReX_EBFabFactory.H unconditionally since 25.03 (the
        # EB-aware Poisson solvers); AMReX's default has EB on, so state it.
        depends_on("amrex+eb", when="@25.03:")

        # ERF's CMake unconditionally requests the AMReX PIC, PARTICLES and
        # LSOLVERS components via find_package(AMReX ... COMPONENTS), regardless of
        # ERF's own +particles variant, so AMReX must always provide them.
        # ERF is three-dimensional and asks for the 3D component only. AMReX's
        # default (dimensions=1,2,3) satisfies this; a 3D-only AMReX
        # (^amrex dimensions:=3) saves libamrex_1d and libamrex_2d, 460 MB on
        # a CUDA build, and is accepted too.
        depends_on("amrex dimensions=3 +pic+particles+linear_solvers")

        # The same find_package() call pins the AMReX precision components:
        # "PDOUBLE" "DOUBLE" is hard-coded through 26.05, and 26.06 derives it
        # from ERF_PRECISION, which defaults to DOUBLE.
        depends_on("amrex precision=double", when="@:26.05")
        for _p in ("double", "single"):
            depends_on(f"amrex precision={_p}", when=f"@26.06:precision={_p}")
        # The same call asks for the PSINGLE/PDOUBLE particle component chosen by
        # ERF_PARTICLES_PRECISION, so AMReX's particle precision must match it.
        # Spack's builtin amrex has no such variant yet; this repository's amrex
        # (packages/amrex) adds it until spack-packages does.
        for _p in ("double", "single"):
            depends_on(f"amrex particles_precision={_p}", when=f"@26.06:particles_precision={_p}")
        # Every ERF regression test pipes its plotfile through AMReX's fcompare,
        # and ERF looks for it at <amrex prefix>/bin/amrex_fcompare, which an
        # installed AMReX only ships with +plotfile_tools.
        depends_on("amrex+plotfile_tools", when="+tests")
        depends_on("amrex+plotfile_tools", when="+fcompare")

        depends_on("mpi", when="+mpi")
        # ERF opens NetCDF files with nc_create_par (NC_NETCDF4 | NC_MPIIO): the
        # HDF5-backed parallel mode, not PnetCDF.
        depends_on("netcdf-c+mpi", when="+netcdf")
        # The Noah-MP ERF driver (built in-tree, see the resources below) is Fortran
        # and reads NetCDF through the Fortran interface.
        depends_on("netcdf-fortran", when="+noahmp")
        # EKAT's bundled Kokkos (see _rrtmgp_stack) imports rocThrust for its HIP backend
        # and oneDPL for its SYCL backend (Kokkos_ENABLE_ROCTHRUST and Kokkos_ENABLE_ONEDPL
        # default on). A ROCm or oneAPI installation carries them; Spack's kokkos package
        # depends on the same two.
        for _ekat in ("+rrtmgp", "+eamxx_shoc"):
            depends_on("rocthrust", when=f"{_ekat}+rocm")
            depends_on("intel-oneapi-dpl", when=f"{_ekat}+sycl")

    # Spack cuda_arch / amdgpu_target -> Kokkos_ARCH_<name> for the Kokkos 4.5.1
    # that EKAT bundles (its cmake/kokkos_arch.cmake option names; Spack's own
    # kokkos package maps newer releases, e.g. VEGA900 and NAVI1100 are not
    # options in 4.5.1).
    _kokkos_cuda_arch = {
        "50": "MAXWELL50",
        "52": "MAXWELL52",
        "53": "MAXWELL53",
        "60": "PASCAL60",
        "61": "PASCAL61",
        "70": "VOLTA70",
        "72": "VOLTA72",
        "75": "TURING75",
        "80": "AMPERE80",
        "86": "AMPERE86",
        "89": "ADA89",
        "90": "HOPPER90",
    }
    _kokkos_amdgpu_arch = {
        "gfx906": "VEGA906",
        "gfx908": "VEGA908",
        "gfx90a": "VEGA90A",
        "gfx940": "AMD_GFX940",
        "gfx942": "AMD_GFX942",
        "gfx1030": "NAVI1030",
        "gfx1100": "AMD_GFX1100",
        "gfx1103": "AMD_GFX1103",
    }

    # ERF's CMake accepts at most one of OpenMP, CUDA, HIP and SYCL.
    conflicts("+openmp", when="+cuda", msg="ERF supports one of OpenMP, CUDA, HIP and SYCL")
    conflicts("+openmp", when="+rocm", msg="ERF supports one of OpenMP, CUDA, HIP and SYCL")
    conflicts("+openmp", when="+sycl", msg="ERF supports one of OpenMP, CUDA, HIP and SYCL")
    conflicts("+cuda", when="+rocm", msg="ERF supports one of OpenMP, CUDA, HIP and SYCL")
    conflicts("+cuda", when="+sycl", msg="ERF supports one of OpenMP, CUDA, HIP and SYCL")
    conflicts("+rocm", when="+sycl", msg="ERF supports one of OpenMP, CUDA, HIP and SYCL")
    # CudaPackage defaults cuda_arch to "none" without a conflict (ROCmPackage
    # has one for amdgpu_target=none); cmake_args needs a real compute capability,
    # and the amrex recipe would otherwise build for the build node's GPU.
    conflicts("cuda_arch=none", when="+cuda", msg="ERF needs an explicit cuda_arch")
    # SYCL means Intel's compilers, for AMReX and for ERF's own kernels.
    requires("%c,cxx=oneapi", when="+sycl", msg="the SYCL backend needs the oneAPI compilers")
    # What ERF's sources and CMake require.
    conflicts("+netcdf", when="~mpi", msg="ERF's NetCDF I/O is parallel and needs MPI")
    conflicts("+noahmp", when="~netcdf", msg="Noah-MP requires NetCDF")
    conflicts("+rrtmgp", when="~netcdf", msg="RRTMGP requires NetCDF")
    conflicts("+rrtmgp", when="~mpi", msg="RRTMGP (EKAT) requires MPI")
    conflicts("+eamxx_shoc", when="~mpi", msg="SHOC (EKAT) requires MPI")
    # The checked-in eamxx_config.h builds EAMxx in double precision.
    conflicts("+eamxx_shoc", when="precision=single", msg="EAMxx SHOC is double precision only")
    conflicts("+tools", when="~netcdf", msg="The tools require NetCDF")
    conflicts(
        "+ml_uphys_diagnostics",
        when="~particles",
        msg="the ML microphysics diagnostics live in the particle (SDM) microphysics",
    )
    conflicts(
        "+wsm6_fort",
        when="precision=single",
        msg="the WSM6 Fortran bridge is double precision only",
    )
    conflicts(
        "+wdm6_fort",
        when="precision=single",
        msg="the WDM6 Fortran bridge is double precision only",
    )
    # +rrtmgp and +eamxx_shoc build the Kokkos bundled with EKAT, which needs its
    # GPU architecture as Kokkos_ARCH_<name> (see cmake_args): it ignores
    # CMAKE_CUDA_ARCHITECTURES, autodetects otherwise (and fails on a node
    # without the GPU) and accepts exactly one GPU architecture.
    for _ekat in ("+rrtmgp", "+eamxx_shoc"):
        for _sm in CudaPackage.cuda_arch_values:
            if _sm not in _kokkos_cuda_arch:
                conflicts(
                    f"cuda_arch={_sm}",
                    when=f"{_ekat}+cuda",
                    msg="EKAT's bundled Kokkos has no architecture for this cuda_arch",
                )
        for _tgt in ROCmPackage.amdgpu_targets:
            if _tgt not in _kokkos_amdgpu_arch:
                conflicts(
                    f"amdgpu_target={_tgt}",
                    when=f"{_ekat}+rocm",
                    msg="EKAT's bundled Kokkos has no architecture for this amdgpu_target",
                )
        for _a, _b in itertools.combinations(sorted(_kokkos_cuda_arch), 2):
            conflicts(
                f"cuda_arch={_a},{_b}", when=f"{_ekat}+cuda", msg="Kokkos takes one cuda_arch"
            )
        for _a, _b in itertools.combinations(sorted(_kokkos_amdgpu_arch), 2):
            conflicts(
                f"amdgpu_target={_a},{_b}",
                when=f"{_ekat}+rocm",
                msg="Kokkos takes one amdgpu_target",
            )

    # RRTMGP, EKAT and EKAT's bundled Kokkos, yaml-cpp, spdlog and Catch2, at the
    # commits ERF pins as submodules (Submodules/RRTMGP and Submodules/ekat in
    # ERF, extern/* in EKAT; unchanged from 25.10 through development).
    # Build/spack/check_pins.py compares these with the gitlinks in ERF's tree
    # and CI runs it, so a submodule bump fails until this list follows.
    # ERF's CMake add_subdirectory()s all of them, forces EKAT's own spdlog
    # (CMAKE_DISABLE_FIND_PACKAGE_spdlog) and passes Kokkos options meant for an
    # in-tree Kokkos, and E3SM's Kokkos fork carries E3SM patches, so Spack's
    # kokkos, spdlog and yaml-cpp packages are not drop-in replacements. They are
    # resources rather than submodules because EKAT lists its submodules over
    # SSH, which an unattended recursive fetch cannot follow. They land in a
    # staging directory and patch() moves them into place: release tarballs and
    # git checkouts both carry an empty placeholder directory at each submodule
    # path, and Spack leaves a resource alone when its destination already exists.
    _rrtmgp_stack = (
        ("rrtmgp", "rte-rrtmgp", "71693e7490db0908a625eadb6ac3f23967de2c7d", "Submodules/RRTMGP"),
        ("ekat", "EKAT", "10ab9d34ceba4eead542702a29676adf77ad91af", "Submodules/ekat"),
        (
            "kokkos",
            "kokkos",
            "6a19eb9454da490dbf785443d8f021028baeb161",
            "Submodules/ekat/extern/kokkos",
        ),
        (
            "yaml-cpp",
            "yaml-cpp",
            "95088a0a2b6f2dec0b3e6e59020cdcc0d4f3c658",
            "Submodules/ekat/extern/yaml-cpp",
        ),
        (
            "spdlog",
            "spdlog",
            "bdd1dff3788ebfe520f48f9ad216c60da6dd8f00",
            "Submodules/ekat/extern/spdlog",
        ),
        # EKAT's testing-support library needs Catch2's single header even with
        # EKAT_ENABLE_TESTS off; nothing in ERF uses it, but EKAT always builds it.
        (
            "catch2",
            "Catch2",
            "958944d27a2d2fb82aa008377bf4f8752f6b848e",
            "Submodules/ekat/extern/Catch2",
        ),
    )
    # EAMxx SHOC needs EKAT too (ERF's CMake turns EKAT on for either); the two
    # conditions are disjoint so a +rrtmgp+eamxx_shoc build fetches each once.
    for _name, _repo, _commit, _ in _rrtmgp_stack:
        for _when in ("+rrtmgp", "~rrtmgp+eamxx_shoc"):
            resource(
                name=_name,
                git=f"https://github.com/E3SM-Project/{_repo}.git",
                commit=_commit,
                destination="spack-resources",
                placement=_name,
                when=_when,
            )

    # The EAMxx sources SHOC is compiled from: the two directories ERF's CMake
    # reads under external/E3SM/components/eamxx/src, out of E3SM at the commit
    # Build/GNU_Ekat/eamxx_clone.sh checks out (check_pins.py compares the two).
    # Spack 1.1+ makes a sparse, blob-filtered clone; older Spack clones all of E3SM.
    _e3sm_commit = "2eb52d9b3d452e593f3825f9408507cae3519bf6"
    resource(
        name="e3sm-eamxx",
        git="https://github.com/E3SM-Project/E3SM.git",
        commit=_e3sm_commit,
        git_sparse_paths=["components/eamxx/src/physics", "components/eamxx/src/share"],
        destination="spack-resources",
        placement="e3sm",
        when="+eamxx_shoc",
    )

    # Noah-MP's ERF coupling layer (drivers/erf) is versioned with ERF, not with
    # Noah-MP's own releases: ERF's pin is well past the last Noah-MP tag and
    # moves with most ERF releases, and ERF compiles the driver in-tree with
    # its own precision macro (DOUBLE_PREC selects the Fortran real kind). So ERF
    # builds Noah-MP from its pinned commit, like the RRTMGP stack above, rather
    # than from Spack's standalone noahmp package, whose tagged releases predate
    # the interface ERF calls. Pins are the Submodules/Noah-MP gitlink of each
    # release (`git ls-tree <tag> Submodules/Noah-MP`); check_pins.py keeps the
    # develop entry equal to the gitlink on the development branch.
    _noahmp_pins = (
        ("25.12:26.04", "6dc16027de56f2abc9fe430016acd086625e9ae2"),
        ("26.05", "c08dd406761330ac823a3e268bd2e98eb6889b40"),
        ("26.06", "41587582bf9b73786ce693660b0a56ff412d1eff"),
        ("26.09", "e03243d2c5eac19b1c6a02a2c3ca12f03f92cceb"),
        ("develop", "fbaa38f0753a0f199dcebd508e2ef64a5dc74bfd"),
    )
    for _versions, _commit in _noahmp_pins:
        resource(
            name="noahmp",
            git="https://github.com/erf-model/noahmp.git",
            commit=_commit,
            destination="spack-resources",
            placement="noahmp",
            when=f"@{_versions}+noahmp",
        )

    @staticmethod
    def _move_into_place(src, dst):
        """Move a staged resource onto the submodule path ERF's CMake expects."""
        # An uninitialised submodule leaves an empty directory behind.
        if os.path.isdir(dst) and not os.listdir(dst):
            os.rmdir(dst)
        if os.path.exists(dst):
            raise InstallError(f"{dst} is already populated; refusing to replace it")
        mkdirp(os.path.dirname(dst))
        os.rename(src, dst)

    def patch(self):
        if self.spec.satisfies("+rrtmgp") or self.spec.satisfies("+eamxx_shoc"):
            for name, _, _, target in self._rrtmgp_stack:
                self._move_into_place(join_path("spack-resources", name), target)
        if self.spec.satisfies("+eamxx_shoc"):
            self._move_into_place(
                join_path("spack-resources", "e3sm"), join_path("external", "E3SM")
            )
        if self.spec.satisfies("+noahmp"):
            self._move_into_place(join_path("spack-resources", "noahmp"), "Submodules/Noah-MP")

        # ERF 26.06+ unit tests add_subdirectory() the bundled googletest
        # checkout, which is not fetched. Redirect them to the Spack-provided
        # GoogleTest: both test executables already link GTest::gtest, the target
        # name that find_package(GTest) defines, and Tests/Unit already
        # include(GoogleTest) for gtest_discover_tests.
        if self.spec.satisfies("@26.06:+tests+unit_tests"):
            unit_cmakelists = join_path("Tests", "Unit", "CMakeLists.txt")
            marker = (
                "add_subdirectory(${PROJECT_SOURCE_DIR}/Submodules/googletest\n"
                "                 ${PROJECT_BINARY_DIR}/Submodules/googletest\n"
                "                 EXCLUDE_FROM_ALL)"
            )
            with open(unit_cmakelists, encoding="utf-8") as f:
                contents = f.read()
            if marker not in contents:
                raise InstallError("ERF unit tests no longer match the googletest patch marker")
            with open(unit_cmakelists, "w", encoding="utf-8") as f:
                f.write(contents.replace(marker, "find_package(GTest REQUIRED)"))

    def cmake_args(self):
        spec = self.spec
        args = [
            self.define("ERF_USE_INTERNAL_AMREX", False),
            self.define_from_variant("ERF_ENABLE_MPI", "mpi"),
            self.define_from_variant("ERF_ENABLE_OPENMP", "openmp"),
            self.define_from_variant("ERF_ENABLE_CUDA", "cuda"),
            self.define_from_variant("ERF_ENABLE_HIP", "rocm"),
            self.define_from_variant("ERF_ENABLE_SYCL", "sycl"),
            self.define_from_variant("ERF_ENABLE_NETCDF", "netcdf"),
            self.define_from_variant("ERF_ENABLE_HDF5", "hdf5"),
            self.define_from_variant("ERF_ENABLE_PARTICLES", "particles"),
            self.define_from_variant("ERF_ENABLE_MULTIBLOCK", "multiblock"),
            self.define_from_variant("ERF_ENABLE_FFT", "fft"),
            self.define_from_variant("ERF_ENABLE_TESTS", "tests"),
            self.define_from_variant("ERF_ENABLE_FCOMPARE", "fcompare"),
            # Conditional variants absent from this version define nothing.
            self.define_from_variant("ERF_ENABLE_TOOLS", "tools"),
            self.define_from_variant("ERF_ENABLE_RRTMGP", "rrtmgp"),
            self.define_from_variant("ERF_ENABLE_NOAHMP", "noahmp"),
            self.define_from_variant("ERF_ENABLE_WINDFARM", "windfarm"),
            self.define_from_variant("ERF_ENABLE_ML_UPHYS_DIAGNOSTICS", "ml_uphys_diagnostics"),
            self.define_from_variant("ERF_ENABLE_WSM6_FORT", "wsm6_fort"),
            self.define_from_variant("ERF_ENABLE_WDM6_FORT", "wdm6_fort"),
            self.define_from_variant("ERF_ENABLE_EAMXX_SHOC", "eamxx_shoc"),
            self.define_from_variant("ERF_ENABLE_FIRE", "fire"),
            self.define("ERF_PRECISION", spec.variants["precision"].value.upper()),
        ]

        # ERF 26.06 keys its gtest unit tests off a separate option that
        # defaults to ON whenever the regression tests are on; state it
        # explicitly so ~tests never drags in a GoogleTest build.
        if spec.satisfies("@26.06:"):
            args.append(self.define("ERF_ENABLE_UNIT_TESTS", spec.satisfies("+tests+unit_tests")))
            particles_precision = spec.variants["particles_precision"].value
            if particles_precision == "same":
                particles_precision = spec.variants["precision"].value
            args.append(self.define("ERF_PARTICLES_PRECISION", particles_precision.upper()))

        # ERF_ENABLE_NVHPC (25.11+) makes ERF's CMake look for the NVHPC SDK's
        # CUDA and math libraries; only meaningful with the NVHPC compilers.
        if spec.satisfies("@25.11:+cuda%nvhpc"):
            args.append(self.define("ERF_ENABLE_NVHPC", True))

        # Only ERF's test harness reads MPIEXEC_PREFLAGS, and --oversubscribe is
        # an Open MPI spelling that other MPI launchers reject.
        if spec.satisfies("+tests+mpi ^[virtuals=mpi] openmpi"):
            args.append(self.define("MPIEXEC_PREFLAGS", "--oversubscribe"))

        # ERF_ENABLE_EKAT was an explicit option in the first RRTMGP releases;
        # 25.12 and later derive it from ERF_ENABLE_RRTMGP.
        if spec.satisfies("@25.10:25.11+rrtmgp"):
            args.append(self.define("ERF_ENABLE_EKAT", True))

        if spec.satisfies("+rrtmgp") or spec.satisfies("+eamxx_shoc"):
            # The bundled Kokkos's GPU architecture (see the conflicts above).
            if spec.satisfies("+cuda"):
                (sm,) = spec.variants["cuda_arch"].value
                args.append(self.define("Kokkos_ARCH_" + self._kokkos_cuda_arch[sm], True))
            if spec.satisfies("+rocm"):
                (tgt,) = spec.variants["amdgpu_target"].value
                args.append(self.define("Kokkos_ARCH_" + self._kokkos_amdgpu_arch[tgt], True))
            if spec.satisfies("+sycl"):
                # Kokkos's FindTPLONEDPL falls back to find_package(oneDPL) when the
                # compiler does not see the headers; the config lives in the component.
                dpl = spec["intel-oneapi-dpl"].package.component_prefix
                args.append(self.define("oneDPL_DIR", dpl.lib.cmake.oneDPL))

        # ERF's CMake prepends NETCDF_DIR to CMAKE_PREFIX_PATH before its FindNetCDF.
        if spec.satisfies("+netcdf"):
            args.append(self.define("NETCDF_DIR", spec["netcdf-c"].prefix))

        # The Noah-MP driver's FindNetCDF looks for the Fortran interface under the
        # C library's prefix only; Spack keeps netcdf-fortran apart, so hand it the
        # answers it would search for.
        if spec.satisfies("+noahmp"):
            netcdf_fortran = spec["netcdf-fortran"]
            args.extend(
                [
                    self.define("NETCDF_INCLUDES_F90", netcdf_fortran.prefix.include),
                    self.define("NETCDF_LIBRARIES_F90", netcdf_fortran.libs.joined(";")),
                ]
            )

        if spec.satisfies("+cuda"):
            # SASS for every requested architecture, PTX only for the newest:
            # AMReX's own "all" policy. SASS+PTX for each of three architectures
            # pushed erf_exec past the 2 GB small code model at link time
            # ("relocation truncated to fit: R_X86_64_PC32"). Compressed fatbins
            # are transparent at load time.
            archs = sorted(spec.variants["cuda_arch"].value, key=self._cuda_arch_number)
            spelled = [a if a == archs[-1] else f"{a}-real" for a in archs]
            args.extend(
                [
                    self.define("CMAKE_CUDA_ARCHITECTURES", ";".join(spelled)),
                    self.define("CMAKE_CUDA_FLAGS", "-Xfatbin=-compress-all"),
                    self.define("CUDAToolkit_ROOT", spec["cuda"].prefix),
                ]
            )

        if spec.satisfies("+rocm"):
            args.append(self.define("CMAKE_CXX_COMPILER", spec["hip"].hipcc))

        return args

    @staticmethod
    def _cuda_arch_number(arch):
        """Numeric part of a compute capability ("90a" -> 90), for ordering."""
        return int("".join(c for c in arch if c.isdigit()))

    @run_after("install")
    def install_examples(self):
        """Ship two self-contained cases so an installed ERF can be run without a
        checkout: prefix/share/erf/examples/<case>/inputs_*."""
        examples = join_path(self.prefix.share, "erf", "examples")
        for case in ("ScalarAdvDiff", "IsentropicVortex"):
            src = join_path(self.stage.source_path, "Exec", "RegTests", case)
            if os.path.isdir(src):
                install_tree(src, join_path(examples, case))

    def test_help(self):
        """erf_exec prints its usage"""
        erf_exec = Executable(self.prefix.bin.erf_exec)
        out = erf_exec("--help", output=str.split, error=str.split)
        assert "usage" in out.lower()

    def test_scalar_advection(self):
        """run two steps of the uniform-velocity scalar advection case"""
        inputs = join_path(
            self.prefix.share, "erf", "examples", "ScalarAdvDiff", "inputs_advect_uniformU"
        )
        if not os.path.isfile(inputs):
            raise SkipTest("no example inputs installed for this version")
        # A wind-farm-enabled erf_exec aborts on any input that names no
        # windfarm_type (WindFarm's constructor rejects the None default).
        if self.spec.satisfies("+windfarm"):
            raise SkipTest("ERF +windfarm cannot run a case without a wind farm")
        erf_exec = Executable(self.prefix.bin.erf_exec)
        with working_dir(self.test_suite.current_test_cache_dir, create=True):
            erf_exec(inputs, "max_step=2", "erf.plot_int_1=-1", "erf.check_int=-1")
