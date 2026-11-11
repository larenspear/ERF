#!/bin/bash
# Run one ERF case from the prebuilt CUDA image on Perlmutter (NERSC) with podman-hpc:
# one rank per A100, any number of nodes.
#
#   podman-hpc pull ghcr.io/erf-model/erf-cuda:development-sm80      # once, on a login node
#   sbatch -A <project>_g perlmutter-a100.sbatch                      # smoke test on 4 GPUs
#   ERF_CASE=$SCRATCH/mycase ERF_INPUTS=inputs sbatch -A <project>_g -N 2 -q regular -t 2:00:00 perlmutter-a100.sbatch
#
# --mpi swaps the image's MPICH for Cray MPICH, so ranks reach the Slingshot network
# (the libmpi line below must say Cray; Spack's RPATH may defeat the swap, in which case bind
# Cray's lib-abi-mpich over the image's MPICH directory as the Polaris script does);
# --gpu exposes the GPUs. The GPU allocation of a project is its account name with _g.
#
# ERF_IMAGE, ERF_CASE, ERF_INPUTS, ERF_ARGS, ERF_WORK, ERF_BIND as in cpu-anywhere.sh.
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH -t 00:20:00
#SBATCH -J erf-image
#SBATCH -o erf-image-%j.out
set -euo pipefail

IMAGE=${ERF_IMAGE:-ghcr.io/erf-model/erf-cuda:development-sm80}
WORK=${ERF_WORK:-$SCRATCH/erf-run-$SLURM_JOB_ID}
INPUTS=${ERF_INPUTS:-inputs_advect_uniformU}
ARGS=${ERF_ARGS:-}
export SLURM_CPU_BIND=cores

mkdir -p "$WORK"
podman-hpc image exists "$IMAGE" || podman-hpc pull "$IMAGE"

# One container on this node, for the steps before the run.
in_image () {
    podman-hpc run --rm --gpu --mpi -v "$WORK:/work" -w /work ${ERF_BIND:+-v "$ERF_BIND:$ERF_BIND"} "$IMAGE" "$@"
}

if [ -n "${ERF_CASE:-}" ]; then
    cp -a "$ERF_CASE/." "$WORK/"
else
    in_image bash -c 'cp -a "$(dirname "$(dirname "$(command -v erf_exec)")")/share/erf/examples/ScalarAdvDiff/." /work/'
    ARGS=${ERF_ARGS:-"max_step=10 erf.check_int=-1 erf.plot_int_1=10"}
fi

in_image erf_exec --describe-json > "$WORK/erf_build.json" 2> /dev/null || rm -f "$WORK/erf_build.json"
help=$(in_image erf_exec --help 2>&1 || true)
case $help in *--validate-input*) in_image erf_exec --validate-input "$INPUTS" ;; esac

# Prove the ranks span the nodes and that Cray MPICH, not the image's MPICH, is what loads.
srun hostname | sort | uniq -c | sed 's/^ */ranks on node: /'
in_image bash -c 'ldd "$(command -v erf_exec)" | grep -m1 libmpi' | sed 's/^ */libmpi: /'

# Cray MPICH names its settings, provider and NIC per rank: the ranks must talk over
# Slingshot (provider cxi), since a silent fall-back to TCP runs correctly and slowly.
# One container per rank, launched by Slurm; the image's MPICH is not CUDA-aware.
# shellcheck disable=SC2086  # ERF_ARGS is a list of inputs
srun podman-hpc run --rm --gpu --mpi -v "$WORK:/work" -w /work ${ERF_BIND:+-v "$ERF_BIND:$ERF_BIND"} \
    --env MPICH_ENV_DISPLAY=1 --env MPICH_OFI_NIC_VERBOSE=1 --env MPICH_OFI_VERBOSE=1 \
    --env MPICH_SMP_SINGLE_COPY_MODE=NONE "$IMAGE" \
    erf_exec "$INPUTS" $ARGS amrex.use_gpu_aware_mpi=0 2>&1 | tee "$WORK/run.log"
grep -i -E 'provider|nic|cxi' "$WORK/run.log" | sort | uniq -c | sort -rn | head -n 8 | sed 's/^ */mpi: /'
if grep -q -i -E 'provider[^a-z]*(tcp|sockets)' "$WORK/run.log"; then
    echo "MPI fell back to a TCP provider instead of Slingshot"; exit 1
fi
[ -f "$WORK/erf_run.json" ] && cat "$WORK/erf_run.json"
echo "run directory: $WORK"
