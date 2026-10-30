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
]
# macos.yml: Apple Clang, MPI on, particles, double and single precision.
MACOS = ["+particles +tests +fcompare", "+particles precision=single"]
# hip.yml: ROCm from AMD's apt repository, gfx908, particles; and the radiation build of
# Build/cmake_with_kokkos_many_hip.sh.
ROCM = [
    "+rocm amdgpu_target=gfx908 +particles",
    "+netcdf +rrtmgp +fft +rocm amdgpu_target=gfx908",
]
# sycl.yml: Intel oneAPI, Intel MPI, particles. The Aurora configuration of
# Docs/sphinx_doc/aurora_build_run.rst (+netcdf +noahmp +rrtmgp +sycl) is refused with the
# oneAPI the runner image installs (2026.x): EKAT's bundled Kokkos 4.5.1 requires
# SYCL_EXT_INTEL_USM_ADDRESS_SPACES, which those releases no longer define (Kokkos dropped
# the check in 5.1, the version EKAT now bundles upstream); 2025.x releases build it.
SYCL = ["+sycl +particles %oneapi ^[virtuals=mpi] intel-oneapi-mpi"]
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
        job = "rocm" if "+rocm" in spec else "sycl" if "+sycl" in spec else "install"
        return {name: [spec] if name == job else [] for name in JOBS}
    version = version or "develop"
    return {job: specs(entries, version) for job, entries in JOBS.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--version", default="", help="ERF version to build (default: develop)")
    ap.add_argument("--spec", default="", help="build this one spec instead of the matrix")
    a = ap.parse_args()
    print(json.dumps(matrix(a.version, a.spec)))


if __name__ == "__main__":
    main()
