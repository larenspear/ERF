# Running ERF from the prebuilt images

One script per machine runs a case from the images that
`.github/workflows/spack-containers.yml` publishes (`ghcr.io/erf-model/erf` for
CPUs, `ghcr.io/erf-model/erf-cuda` for A100 GPUs). Each script copies the case
into a run directory, validates the inputs when the image can, records the
image's build description, runs, and prints the run record when the image
writes one. Without a case it runs the `ScalarAdvDiff` example from the image
for ten steps, which is the smoke test for a new machine.

| Script | Machine | Engine | Launch |
| --- | --- | --- | --- |
| `cpu-anywhere.sh` | any x86_64 host, or Apple Silicon under emulation | Docker, Podman or Apptainer | `mpiexec` inside the image |
| `deception-a100.sbatch` | Deception (PNNL), one node | Apptainer | `mpiexec` inside the image |
| `perlmutter-a100.sbatch` | Perlmutter (NERSC), any node count | podman-hpc `--mpi --gpu` | `srun`, Cray MPICH |
| `polaris-a100.pbs` | Polaris (ALCF), any node count | Apptainer with `cray-mpich-abi` | `mpiexec`, Cray MPICH |

The variables they read (`ERF_IMAGE`, `ERF_CASE`, `ERF_INPUTS`, `ERF_ARGS`,
`ERF_RANKS`, `ERF_WORK`, `ERF_BIND`) are described at the top of
`cpu-anywhere.sh` and in `Docs/sphinx_doc/containers.rst`.
