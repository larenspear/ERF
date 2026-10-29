#!/usr/bin/env python3
"""Concretization sweep for ERF's Spack recipe.

Every spec in POS must concretize; every spec in NEG must be refused on the recipe's own
terms (a conflict, a dependency that cannot be met, or a variant value the version does
not offer). Concretizing takes under a minute per spec and needs no build, so this covers
the whole variant product cheaply: a missing conflict, a bad depends_on, a variant that no
release has, all show up here. The install matrix in .github/workflows/spack-package.yml
then builds a representative subset.

Run it with a Spack that has the in-tree recipe registered ahead of builtin:

    spack repo add Build/spack/spack_repo/erf
    python3 Build/spack/erf_sweep.py

ERF_SWEEP_JOBS sets the number of concurrent solves (default 3).
"""

import concurrent.futures as cf
import os
import re
import subprocess
import sys
import time

POS = [
    "erf@develop",
    "erf@develop ~mpi",
    "erf@develop +openmp",
    "erf@develop +netcdf",
    "erf@develop +netcdf ~hdf5",
    "erf@develop +netcdf +noahmp",
    "erf@develop +netcdf +rrtmgp",
    "erf@develop +netcdf +tools",
    "erf@develop +particles",
    "erf@develop +particles +ml_uphys_diagnostics",
    "erf@develop +tests +fcompare",
    "erf@develop +tests ~unit_tests",
    "erf@develop precision=single",
    "erf@develop precision=single particles_precision=single +particles",
    "erf@develop +eamxx_shoc +tests",
    "erf@develop +netcdf +rrtmgp +eamxx_shoc",
    "erf@develop +fire",
    "erf@26.09 +eamxx_shoc",
    "erf@develop +windfarm",
    "erf@develop +wsm6_fort +wdm6_fort",
    "erf@develop +fft",
    "erf@develop +cuda cuda_arch=80",
    "erf@develop +netcdf +cuda cuda_arch=70,80,90",
    "erf@develop +rocm amdgpu_target=gfx90a",
    "erf@develop +netcdf +rrtmgp +cuda cuda_arch=80",
    "erf@develop +netcdf +rrtmgp +rocm amdgpu_target=gfx90a",
    "erf@26.09",
    "erf@26.09 +netcdf +noahmp +particles +tests",
    "erf@26.09 +netcdf +hdf5 +particles +fcompare +openmp",
    "erf@26.09 +netcdf +rrtmgp +cuda cuda_arch=80",
    "erf@26.06 +netcdf +rrtmgp",
    "erf@26.05",
    "erf@25.12 +netcdf +noahmp",
    "erf@25.10 +netcdf +rrtmgp",
    "erf@25.10 +particles",
    "erf@25.01 +fft",
    "erf@24.11",
    "erf@24.10 +particles +tests",
]
# +sycl requires the oneAPI compilers and Spack only accepts an external or installed
# compiler for a %-requirement, so on gcc-only machines it is refused; it concretizes
# wherever `spack compiler find` has registered oneAPI.
NEG = [
    "erf@develop +sycl",
    "erf@develop +rrtmgp ~netcdf",
    "erf@develop +rrtmgp ~mpi",
    "erf@develop +cuda cuda_arch=80 +rocm amdgpu_target=gfx90a",
    "erf@develop +openmp +cuda cuda_arch=80",
    "erf@develop +cuda",
    "erf@develop +netcdf +rrtmgp +cuda cuda_arch=80,90",
    "erf@develop +ml_uphys_diagnostics ~particles",
    "erf@develop +eamxx_shoc precision=single",
    "erf@develop +eamxx_shoc ~mpi",
    "erf@develop precision=double particles_precision=single +particles",
    "erf@26.06 +eamxx_shoc",
    "erf@26.09 +fire",
    "erf@develop +netcdf ~mpi",
    "erf@develop +tools ~netcdf",
    "erf@26.09 +wsm6_fort precision=single",
    "erf@26.05 precision=single",
    "erf@26.01 +windfarm",
    "erf@25.11 +netcdf +noahmp",
    "erf@25.11 +noahmp",
    "erf@25.06 +tools",
    "erf@25.03 ^amrex~eb",
    "erf@24.10 +fft",
    "erf@24.10 ^amrex@24.10",
]


# Spack offers no CUDA or ROCm toolchain on macOS, so GPU specs cannot concretize there
# whatever the recipe says; a local run on a Mac skips them and CI covers them.
SKIP_ON_DARWIN = ("+cuda", "+rocm")
if sys.platform == "darwin":
    skipped = [s for s in POS + NEG if any(k in s for k in SKIP_ON_DARWIN)]
    POS = [s for s in POS if s not in skipped]
    NEG = [s for s in NEG if s not in skipped]
    print(f"## skipping {len(skipped)} GPU specs on macOS (no CUDA/ROCm packages here)")


def run(spec):
    t = time.time()
    r = subprocess.run(
        ["spack", "spec", "--fresh", "--format", "{name}@{version}{variants}", spec],
        capture_output=True,
        text=True,
        timeout=900,
    )
    full = r.stderr + r.stdout
    err = full.strip().split("\n")
    key = re.compile(r"Error|error|conflict|unsatisfiable|cannot|requires|needs|variant", re.I)
    msg = [line for line in err if key.search(line)]
    return (
        spec,
        r.returncode,
        round(time.time() - t),
        (msg[-1] if msg else err[-1] if err else "")[:150],
        full,
    )


def refused(rc, full):
    """A NEG spec counts only if the recipe refused it: the solver found a conflict or an
    unmet dependency, or the version offers no such variant value. "No such variant" means
    the recipe under test lacks the variant altogether, so the spec proves nothing."""
    return (
        rc != 0
        and "No such variant" not in full
        and (
            "failed to concretize" in full
            or "concretization failed" in full
            or "invalid values for variant" in full
            or "Cannot satisfy" in full
            or "cannot satisfy" in full
        )
    )


# One solve first so the bootstrap and the repository index are built once, then the rest
# a few at a time: concurrent first solves have been killed without any output on a
# 7 GB runner.
run(POS[0])
with cf.ThreadPoolExecutor(int(os.environ.get("ERF_SWEEP_JOBS", "3"))) as ex:
    res = list(ex.map(run, POS + NEG))
# A positive spec that failed with no output at all was most likely killed; solve it
# again on its own and show everything it says.
for i, (spec, rc, dt, msg, full) in enumerate(res):
    if i < len(POS) and rc != 0 and not full.strip():
        print(f"## retrying {spec} alone (no output the first time)")
        res[i] = run(spec)
        if res[i][1] != 0:
            print(res[i][4][-3000:])
ok = 0
print("## positive (must concretize):")
for spec, rc, dt, msg, _ in res[: len(POS)]:
    st = "ok  " if rc == 0 else "FAIL"
    ok += rc == 0
    print(f"   {st} {dt:4d}s  {spec:55s} {'' if rc == 0 else msg}")
print("## negative (must be refused with the recipe's message):")
for spec, rc, dt, msg, full in res[len(POS) :]:
    good = refused(rc, full)
    st = "ok  " if good else "FAIL"
    ok += good
    print(
        f"   {st} {dt:4d}s  {spec:55s} {msg if rc != 0 else 'concretized (should have been refused)'}"
    )
print(f"## {ok}/{len(res)} as expected")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write("### Concretization sweep\n\n| | spec | result |\n|---|---|---|\n")
        for i, (spec, rc, dt, msg, full) in enumerate(res):
            good = (rc == 0) if i < len(POS) else refused(rc, full)
            f.write(
                f"| {'ok' if good else 'FAIL'} | `{spec}` | {('concretized' if rc == 0 else msg.replace('|', '/'))} |\n"
            )
        f.write(f"\n{ok}/{len(res)} as expected\n")
sys.exit(0 if ok == len(res) else 1)
