"""The build matrix of .github/workflows/spack-package.yml, as JSON.

The lists below hold variant strings. The workflow's plan job prefixes each with
``erf@<version>`` (``develop`` by default, a release tag from the ``version`` input), so one
manual run builds the whole matrix at a release. Entries that name a version themselves are
built as written on development runs and dropped on release runs: ``erf@develop`` entries use
options the development branch alone has, and the pinned-release entries keep a release path
proven alongside development. A ``spec`` input builds that one spec alone: in the ROCm or
SYCL job when it asks for that backend, otherwise in the install job.

    python3 Build/spack/ci_matrix.py                   # development, every job
    python3 Build/spack/ci_matrix.py --version 26.09   # the same at the 26.09 release
    python3 Build/spack/ci_matrix.py --spec "erf@develop +fft"
"""

import argparse
import json
import re

# Every variant that changes what is compiled, each with the ERF workflow, build script or
# documentation page that builds that configuration. CUDA mirrors cuda-ci.yml (CUDA 12,
# particles); NetCDF + Noah-MP + RRTMGP + FFT on a GPU is what Build/Perlmutter and the
# Kestrel quickstart script configure.
INSTALL = [
    "+mpi",  # the default build
    "~mpi",
    "+openmp +particles +tests +fcompare",
    "+netcdf +noahmp +particles",  # Build/cmake_with_noahmp.sh
    "+netcdf ~hdf5 +tests +fcompare",  # Build/cmake_with_netcdf.sh
    "+netcdf +rrtmgp",  # gcc-rrtmgp.yml
    "+netcdf +fft +rrtmgp +tests +fcompare",  # Build/cmake_with_kokkos_many.sh
    "+netcdf +tools",
    "+fft +tests +fcompare",  # Build/cmake_with_fft.sh
    "+multiblock +tests +fcompare",  # Build/cmake_multiblock.sh
    "+windfarm +wsm6_fort +wdm6_fort",
    "precision=single particles_precision=single +particles",  # gcc.yml
    "precision=double particles_precision=single +particles",  # gcc.yml
    "+particles +ml_uphys_diagnostics",
    "+eamxx_shoc +tests ~unit_tests",  # Build/cmake_with_shoc.sh
    "+particles +tests +fcompare ^mpich",  # ci.yml's MPI
    "+particles +cuda cuda_arch=80 ^cuda@12",  # cuda-ci.yml
    "+netcdf +noahmp +rrtmgp +fft +cuda cuda_arch=80 ^cuda@12",  # Build/Perlmutter/cmake_with_RRTMGP_NOAHMP.sh
    "erf@develop +fire",  # ERF_ENABLE_FIRE exists on development only
    "erf@26.09 +particles +cuda cuda_arch=80 ^cuda@12",  # the last release, CUDA
    "erf@26.09 +netcdf +noahmp +particles +tests",  # the last release, CPU
    "erf@23.10 +mpi",  # the oldest release, bundled AMReX
    "erf@24.09 +netcdf +particles +tests",  # the last bundled-AMReX release
    "erf@24.10 +mpi",  # the one release paired with the next AMReX (24.11)
    "erf@25.01 +netcdf",  # the last release with AMReX embedded boundaries off
]
# macos.yml: Apple Clang, MPI on, particles, double and single precision.
MACOS = ["+particles +tests +fcompare", "+particles precision=single"]
# hip.yml: ROCm from AMD's apt repository, gfx908, particles; and the radiation build of
# Build/cmake_with_kokkos_many_hip.sh.
ROCM = [
    "+rocm amdgpu_target=gfx908 +particles",
    "+netcdf +rrtmgp +fft +rocm amdgpu_target=gfx908",
]
# sycl.yml: Intel oneAPI, Intel MPI, particles; and the Aurora configuration of
# Docs/sphinx_doc/aurora_build_run.rst (+netcdf +noahmp +rrtmgp +sycl) on development only:
# at 26.09 and earlier EKAT's bundled Kokkos 4.5.1 requires SYCL_EXT_INTEL_USM_ADDRESS_SPACES,
# which the oneAPI 2026 releases the runner image installs no longer define; the Kokkos 5.1
# that development's EKAT bundles (#4293) dropped that check.
SYCL = [
    "+sycl +particles %oneapi ^[virtuals=mpi] intel-oneapi-mpi",
    "erf@develop +netcdf +noahmp +rrtmgp +sycl %oneapi ^[virtuals=mpi] intel-oneapi-mpi",
]
JOBS = {"install": INSTALL, "macos": MACOS, "rocm": ROCM, "sycl": SYCL}


def specs(entries, version):
    out = []
    for entry in entries:
        if entry.startswith("erf@"):
            if version == "develop":
                out.append(entry)
        else:
            out.append(f"erf@{version} {entry}")
    return out


def matrix(version="", spec=""):
    if spec:
        job = (
            "macos"
            if "platform=darwin" in spec
            else "rocm"
            if "+rocm" in spec
            else "sycl"
            if "+sycl" in spec
            else "install"
        )
        return {name: [spec] if name == job else [] for name in JOBS}
    version = version or "develop"
    return {job: specs(entries, version) for job, entries in JOBS.items()}


def validate_request(spec, runner, build_mode, run_tests):
    """Reject requests that would silently skip the requested validation."""
    if build_mode != "install" and not spec:
        raise ValueError("develop/reuse validation requires one explicit spec")
    if spec and not re.match(r"^erf(?:@|\s|$)", spec):
        raise ValueError("the spec must name erf explicitly")
    gpu = any(backend in spec for backend in ("+cuda", "+rocm", "+sycl"))
    if runner == "ubuntu-24.04-arm" and (gpu or "platform=darwin" in spec or not spec):
        raise ValueError("the Linux ARM64 runner requires one CPU Linux spec")
    if run_tests and spec and (gpu or "+tests" not in spec):
        raise ValueError("run_tests requires a CPU spec with +tests")
    if build_mode == "develop" and not re.match(r"^erf@develop(?:\s|$)", spec):
        raise ValueError("develop mode requires erf@develop to match the checked-out source")
    if build_mode != "install" and (gpu or "platform=darwin" in spec):
        raise ValueError("develop/reuse validation currently supports Linux CPU specs")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="", help="ERF version to build (default: develop)")
    ap.add_argument("--spec", default="", help="build this one spec instead of the matrix")
    ap.add_argument("--runner", default="ubuntu-22.04")
    ap.add_argument("--build-mode", choices=("install", "develop", "reuse"), default="install")
    ap.add_argument("--run-tests", action="store_true")
    a = ap.parse_args()
    try:
        validate_request(a.spec, a.runner, a.build_mode, a.run_tests)
    except ValueError as exc:
        ap.error(str(exc))
    print(json.dumps(matrix(a.version, a.spec)))


if __name__ == "__main__":
    main()
