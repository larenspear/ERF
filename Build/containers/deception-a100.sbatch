#!/bin/bash
# Run one ERF case from the prebuilt CUDA image on Deception (PNNL): one node, one rank per GPU.
#
#   sbatch deception-a100.sbatch                                    # smoke test on one A100
#   ERF_CASE=/rcfs/scratch/$USER/mycase ERF_INPUTS=inputs sbatch deception-a100.sbatch
#   sbatch --gres=gpu:4 -n 4 -c 32 deception-a100.sbatch            # four GPUs, four ranks
#
# The image is built for A100 (sm80): use a100_80_shared or a100_shared, not dlv (V100) or
# h100. The shared partitions bill only the GPUs and cores requested. MPI runs inside the
# container, so this script is for one node. Compute nodes may not reach ghcr.io, so pull
# the image once on a login node, with Apptainer's temporary directory on the node's local
# disk (/rcfs refuses the hard links inside the image):
#   module load apptainer && mkdir -p /scratch/$USER/apptainer-tmp /rcfs/scratch/$USER/erf-sif
#   APPTAINER_TMPDIR=/scratch/$USER/apptainer-tmp apptainer pull \
#       /rcfs/scratch/$USER/erf-sif/ghcr.io_erf-model_erf-cuda_development-sm80.sif \
#       docker://ghcr.io/erf-model/erf-cuda:development-sm80
# Compute nodes cannot mount a SIF either, so the job unpacks it onto the node's local disk
# on first use (about a minute), under ERF_LOCAL_SCRATCH or /scratch/$USER.
#
# ERF_IMAGE, ERF_CASE, ERF_INPUTS, ERF_ARGS, ERF_RANKS, ERF_WORK, ERF_BIND as in cpu-anywhere.sh.
#SBATCH -A erf_ci
#SBATCH -p a100_80_shared
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 00:20:00
#SBATCH -J erf-image
#SBATCH -o erf-image-%j.out
set -euo pipefail

IMAGE=${ERF_IMAGE:-ghcr.io/erf-model/erf-cuda:development-sm80}
SIF=${ERF_SIF:-/rcfs/scratch/$USER/erf-sif/$(echo "$IMAGE" | tr '/:' '__').sif}
WORK=${ERF_WORK:-$SLURM_SUBMIT_DIR/erf-run-$SLURM_JOB_ID}
RANKS=${ERF_RANKS:-$SLURM_NTASKS}
INPUTS=${ERF_INPUTS:-inputs_advect_uniformU}
ARGS=${ERF_ARGS:-}

if ! command -v apptainer > /dev/null; then
    # A batch shell has no module function; the profile that defines it is neither -e nor -u clean.
    if ! type module > /dev/null 2>&1; then set +eu; source /etc/profile; set -eu; fi
    module load apptainer
fi
for node_disk in "${ERF_LOCAL_SCRATCH:-}" "/scratch/$USER" "/local/scratch/$USER" "/tmp/$USER"; do
    [ -n "$node_disk" ] && mkdir -p "$node_disk" 2> /dev/null && [ -w "$node_disk" ] && break
done
export APPTAINER_TMPDIR=$node_disk/apptainer-tmp APPTAINER_CACHEDIR=$node_disk/apptainer-cache
SANDBOX=$node_disk/erf-sandbox/$(basename "${SIF%.sif}")
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR" "$(dirname "$SANDBOX")" "$(dirname "$SIF")" "$WORK"
[ -s "$SIF" ] || apptainer pull "$SIF" "docker://$IMAGE"
if [ ! -d "$SANDBOX/opt" ] || [ "$SIF" -nt "$SANDBOX" ]; then   # first use, or a newer image
    rm -rf "$SANDBOX" && apptainer build --sandbox "$SANDBOX" "$SIF"
fi

in_image () {
    apptainer exec --nv --bind "$WORK:/work" --pwd /work ${ERF_BIND:+--bind "$ERF_BIND"} "$SANDBOX" "$@"
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

# One rank per GPU; the image's MPICH is not CUDA-aware. Inside the job its launcher would
# otherwise look for srun, which the container does not have.
# shellcheck disable=SC2086  # ERF_ARGS is a list of inputs
in_image mpiexec -launcher fork -n "$RANKS" erf_exec "$INPUTS" $ARGS amrex.use_gpu_aware_mpi=0
[ -f "$WORK/erf_run.json" ] && cat "$WORK/erf_run.json"
echo "run directory: $WORK"
