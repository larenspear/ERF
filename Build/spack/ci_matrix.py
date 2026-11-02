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
    "erf@24.10 +netcdf +particles +tests",
    "erf@24.11 +netcdf +particles +tests",
    "erf@25.01 +netcdf +particles +tests",
    "erf@25.03 +netcdf +particles +tests",
    "erf@25.04 +netcdf +particles +tests",
    "erf@25.05 +netcdf +particles +tests",
    "erf@25.06 +netcdf +particles +tests",
    "erf@25.07 +netcdf +particles +tests",
    "erf@25.08 +netcdf +particles +tests",
    "erf@25.10 +netcdf +particles +tests",
    "erf@25.11 +netcdf +particles +tests",
    "erf@25.12 +netcdf +particles +tests",
    "erf@26.01 +netcdf +particles +tests",
    "erf@26.02 +netcdf +particles +tests",
    "erf@26.03 +netcdf +particles +tests",
    "erf@26.04 +netcdf +particles +tests",
    "erf@26.05 +netcdf +particles +tests",
    "erf@26.06 +netcdf +particles +tests",
]
MACOS = []
ROCM = []
SYCL = []
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
