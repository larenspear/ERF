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
    '+mpi',
    '~mpi',
    '+openmp',
    '+netcdf',
    '+netcdf ~hdf5',
    '+hdf5 ~netcdf',
    '+particles',
    '+multiblock',
    '+fft',
    '+tests',
    '+tests ~unit_tests',
    '+fcompare',
    'precision=single',
    '+particles particles_precision=single',
    '+particles precision=single particles_precision=double',
    '+netcdf +tools',
    '+netcdf +rrtmgp',
    '+netcdf +noahmp',
    '+windfarm',
    '+particles +ml_uphys_diagnostics',
    '+wsm6_fort',
    '+wdm6_fort',
    '+eamxx_shoc',
    '+openmp +particles +tests +fcompare',
    '+netcdf +noahmp +particles',
    '+netcdf ~hdf5 +tests +fcompare',
    '+netcdf +fft +rrtmgp +tests +fcompare',
    '+fft +tests +fcompare',
    '+multiblock +tests +fcompare',
    '+windfarm +wsm6_fort +wdm6_fort',
    'precision=single particles_precision=single +particles',
    'precision=double particles_precision=single +particles',
    '+eamxx_shoc +tests ~unit_tests',
    '+particles +tests +fcompare ^mpich',
    '~mpi +openmp +particles +tests +fcompare',
    '+particles +cuda cuda_arch=80 ^cuda@12',
    '+netcdf +noahmp +rrtmgp +fft +cuda cuda_arch=80 ^cuda@12',
    '+openmp +netcdf +hdf5 +particles +multiblock +fft +tests +fcompare +tools +rrtmgp +noahmp +windfarm +wsm6_fort +wdm6_fort +eamxx_shoc +ml_uphys_diagnostics',
    '+openmp +netcdf +hdf5 +particles +multiblock +fft +tests +fcompare +tools +rrtmgp +noahmp +windfarm +wsm6_fort +wdm6_fort +eamxx_shoc +ml_uphys_diagnostics ^mpich',
    '+netcdf +particles +fft +tests +fcompare +tools +rrtmgp +noahmp +windfarm +wsm6_fort +wdm6_fort +eamxx_shoc +ml_uphys_diagnostics +cuda cuda_arch=80 ^cuda@12',
    '+netcdf +particles +fft +tests +fcompare +tools +rrtmgp +noahmp +windfarm +wsm6_fort +wdm6_fort +eamxx_shoc +ml_uphys_diagnostics +cuda cuda_arch=70,80,90 ^cuda@12',
]
MACOS = [
    '+particles +tests +fcompare',
    '+particles precision=single',
    '+netcdf +noahmp +particles',
    '+openmp',
    '~mpi',
    '+netcdf +fft +multiblock +tools +tests +fcompare',
    '+netcdf +particles +windfarm +wsm6_fort +wdm6_fort +ml_uphys_diagnostics +tests +fcompare',
]
ROCM = [
    '+rocm amdgpu_target=gfx908 +particles',
    '+netcdf +rrtmgp +fft +rocm amdgpu_target=gfx908',
    '+netcdf +noahmp +particles +tests +fcompare +rocm amdgpu_target=gfx90a',
    '+netcdf +particles +fft +tests +fcompare +tools +rrtmgp +noahmp +windfarm +wsm6_fort +wdm6_fort +eamxx_shoc +ml_uphys_diagnostics +rocm amdgpu_target=gfx90a',
]
SYCL = [
    '+sycl +particles %oneapi ^[virtuals=mpi] intel-oneapi-mpi',
    '+netcdf +noahmp +particles +tests +fcompare +sycl %oneapi ^[virtuals=mpi] intel-oneapi-mpi',
    '+netcdf +rrtmgp +sycl %oneapi ^[virtuals=mpi] intel-oneapi-mpi',
    '+netcdf +particles +fft +tests +fcompare +tools +noahmp +windfarm +wsm6_fort +wdm6_fort +ml_uphys_diagnostics +sycl %oneapi ^[virtuals=mpi] intel-oneapi-mpi',
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
        job = ("macos" if "platform=darwin" in spec else "rocm" if "+rocm" in spec
               else "sycl" if "+sycl" in spec else "install")
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
