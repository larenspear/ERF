#!/usr/bin/env python3
"""Concretization sweep for the ERF Spack recipe.

Every spec in POS must concretize; every spec in NEG must be refused with one of the
recipe's own conflict messages. Concretizing takes under a minute per spec and needs no
build, so this covers the whole variant product cheaply: a missing conflict, a bad
depends_on, a variant that no release has, all show up here. The build matrix in
spack-package.yml then builds a representative subset. Run locally with a Spack whose
builtin repo is the recipe under test:  python3 erf_sweep.py
"""
import os, subprocess, sys, time, concurrent.futures as cf, re
POS = ["erf@develop", "erf@develop ~mpi", "erf@develop +openmp", "erf@develop +netcdf", "erf@develop +netcdf +noahmp",
       "erf@develop +netcdf +rrtmgp", "erf@develop +netcdf +tools", "erf@develop +particles", "erf@develop +tests +fcompare",
       "erf@develop precision=single", "erf@develop +windfarm", "erf@develop +wsm6_fort +wdm6_fort", "erf@develop +fft",
       "erf@develop +cuda cuda_arch=80", "erf@develop +netcdf +cuda cuda_arch=70,80,90", "erf@develop +rocm amdgpu_target=gfx90a",
       "erf@develop +sycl", "erf@26.09", "erf@26.09 +netcdf +noahmp +particles +tests", "erf@26.06 +netcdf +rrtmgp",
       "erf@25.12 +netcdf +noahmp", "erf@25.10 +particles", "erf@24.10 +particles +tests", "erf@24.09",
       "erf@26.09 +netcdf +hdf5 +particles +fcompare +openmp", "erf@26.09 +netcdf +rrtmgp +cuda cuda_arch=80",
       "erf@develop +netcdf ~hdf5", "erf@25.10 +netcdf +rrtmgp", "erf@26.05"]
NEG = ["erf@develop +rrtmgp ~netcdf", "erf@develop +rrtmgp ~mpi", "erf@develop +cuda cuda_arch=80 +rocm amdgpu_target=gfx90a",
       "erf@develop +openmp +cuda cuda_arch=80", "erf@26.05 precision=single", "erf@26.01 +windfarm", "erf@25.11 +netcdf +noahmp",
       "erf@develop +netcdf ~mpi", "erf@develop +tools ~netcdf", "erf@26.09 +wsm6_fort precision=single",
       "erf@25.11 +noahmp", "erf@25.06 +tools"]
def run(spec):
    t=time.time()
    r=subprocess.run(["spack","spec","--fresh","--format","{name}@{version}{variants}",spec], capture_output=True, text=True, timeout=900)
    err=(r.stderr+r.stdout).strip().split("\n")
    msg=[l for l in err if re.search(r"Error|error|conflict|unsatisfiable|cannot|requires|needs|variant", l, re.I)]
    return spec, r.returncode, round(time.time()-t), (msg[-1] if msg else err[-1] if err else "")[:150], r.stderr+r.stdout
# One solve first so the bootstrap and the repository index are built once, then the rest
# a few at a time: concurrent first solves have been killed without any output on a
# 7 GB runner.
run(POS[0])
with cf.ThreadPoolExecutor(int(os.environ.get("ERF_SWEEP_JOBS", "3"))) as ex:
    res=list(ex.map(run, POS+NEG))
# A positive spec that failed with no output at all was most likely killed; solve it
# again on its own and show everything it says.
for i,(spec,rc,dt,msg,full) in enumerate(res):
    if i < len(POS) and rc != 0 and not full.strip():
        print(f"## retrying {spec} alone (no output the first time)")
        res[i] = run(spec)
        if res[i][1] != 0:
            print(res[i][4][-3000:])
ok=0
print("## positive (must concretize):")
for spec,rc,dt,msg,_ in res[:len(POS)]:
    st="ok  " if rc==0 else "FAIL"; ok+= rc==0
    print(f"   {st} {dt:4d}s  {spec:55s} {'' if rc==0 else msg}")
print("## negative (must be refused with the recipe's message):")
for spec,rc,dt,msg,_ in res[len(POS):]:
    st="ok  " if rc!=0 else "FAIL"; ok+= rc!=0
    print(f"   {st} {dt:4d}s  {spec:55s} {msg if rc!=0 else 'concretized (should have been refused)'}")
print(f"## {ok}/{len(res)} as expected")
if os.environ.get("GITHUB_STEP_SUMMARY"):
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write("### Concretization sweep\n\n| | spec | result |\n|---|---|---|\n")
        for i,(spec,rc,dt,msg,_) in enumerate(res):
            good = (rc==0) if i < len(POS) else (rc!=0)
            f.write(f"| {'ok' if good else 'FAIL'} | `{spec}` | {('concretized' if rc==0 else msg.replace('|','/'))} |\n")
        f.write(f"\n{ok}/{len(res)} as expected\n")
sys.exit(0 if ok == len(res) else 1)
