#!/bin/bash
# Run one ERF case from the prebuilt CUDA image on Perlmutter (NERSC) with podman-hpc:
# one rank per A100, any number of nodes.
#
#   podman-hpc pull ghcr.io/erf-model/erf-cuda:development-sm80      # once, on a login node
#   sbatch -A <project>_g perlmutter-a100.sbatch                      # smoke test on 4 GPUs
#   ERF_CASE=$SCRATCH/mycase ERF_INPUTS=inputs sbatch -A <project>_g -N 2 -q regular -t 2:00:00 perlmutter-a100.sbatch
#
# --mpi supplies the Slurm, PMI and Slingshot device plumbing for Cray MPICH inside the
# container, and --gpu exposes the GPUs. The MPICH it bundles (8.1) predates the
# MPIX_GPU_query_support call AMReX makes, so the script takes the host's current Cray MPICH
# and its dependencies instead, and binds them over the image's own MPICH and libfabric
# directories, because Spack links erf_exec with an RPATH naming those and an RPATH beats the
# library path. The GPU allocation of a project is its account name with _g.
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

# The directories erf_exec resolves its MPICH and libfabric from inside the image.
image_lib () {
    podman-hpc run --rm "$IMAGE" bash -c 'dirname "$(ldd "$(command -v erf_exec)" | awk -v l="$1" "\$1 == l {print \$3}")"' _ "$1"
}
image_mpich=$(image_lib libmpi.so.12)
image_fabric=$(image_lib libfabric.so.1)
# The host's newest Cray MPICH, ABI flavour, and everything it loads apart from glibc: libfabric,
# the Slingshot libraries, PMI and their dependencies, gathered where the RPATH will look.
cray_mpich=$(ls -d /opt/cray/pe/mpich/*/ofi/gnu/*/lib-abi-mpich | sort -V | tail -n 1)
mkdir -p "$WORK/craylib"
ldd "$cray_mpich/libmpi.so.12" | awk '$3 ~ "^/" {print $3}' \
    | grep -v -E '/(libc|libm|libpthread|libdl|librt|libresolv|ld-linux)[.-]|lib-abi-mpich/' \
    | xargs -r -I{} cp -L {} "$WORK/craylib/"
echo "Cray MPICH from $cray_mpich with $(ls "$WORK/craylib" | wc -l) libraries ($(ls "$WORK/craylib" | tr '\n' ' ' | cut -c1-100)...)"
binds=(-v "$WORK:/work" -v "$cray_mpich:$image_mpich" -v "$WORK/craylib:$image_fabric")
mpi_env=(--env MPICH_ENV_DISPLAY=1 --env MPICH_OFI_NIC_VERBOSE=1 --env MPICH_OFI_VERBOSE=1
         --env MPICH_SMP_SINGLE_COPY_MODE=NONE)

# One container on this node, for the steps before the run.
in_image () {
    podman-hpc run --rm --gpu --mpi "${binds[@]}" -w /work ${ERF_BIND:+-v "$ERF_BIND:$ERF_BIND"} "$IMAGE" "$@"
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

# Prove the ranks span the nodes and that Cray's MPICH and libfabric, not the image's, load.
srun hostname | sort | uniq -c | sed 's/^ */ranks on node: /'
in_image bash -c 'lib=$(ldd "$(command -v erf_exec)" | awk "/libmpi.so.12 =>/ {print \$3}")
    v=$(grep -a -o -m1 "CRAY MPICH version [0-9.]*" "$lib" || true)
    if [ -n "$v" ]; then echo "libmpi: $lib is $v"; else echo "libmpi: $lib is the image MPICH"; exit 1; fi
    lib=$(ldd "$(command -v erf_exec)" | awk "/libfabric.so.1 =>/ {print \$3}")
    if cmp -s "$lib" /work/craylib/libfabric.so.1; then echo "fabric: $lib is the host libfabric"; else echo "fabric: $lib is the image libfabric"; exit 1; fi
    missing=$(ldd "$(command -v erf_exec)" | grep "not found" || true)
    if [ -n "$missing" ]; then echo "libraries missing inside the container:"; echo "$missing"; exit 1; fi'

# Cray MPICH names its settings, provider and NIC per rank: the ranks must talk over
# Slingshot (provider cxi), since a silent fall-back to TCP runs correctly and slowly.
# One container per rank, launched by Slurm; the image's MPICH is not CUDA-aware.
# shellcheck disable=SC2086  # ERF_ARGS is a list of inputs
srun podman-hpc run --rm --gpu --mpi "${binds[@]}" -w /work ${ERF_BIND:+-v "$ERF_BIND:$ERF_BIND"} \
    "${mpi_env[@]}" "$IMAGE" erf_exec "$INPUTS" $ARGS amrex.use_gpu_aware_mpi=0 2>&1 | tee "$WORK/run.log"
grep -i -E 'provider|nic|cxi' "$WORK/run.log" | sort | uniq -c | sort -rn | head -n 8 | sed 's/^ */mpi: /'
if grep -q -i -E 'provider[^a-z]*(tcp|sockets)' "$WORK/run.log"; then
    echo "MPI fell back to a TCP provider instead of Slingshot"; exit 1
fi
[ -f "$WORK/erf_run.json" ] && cat "$WORK/erf_run.json"
echo "run directory: $WORK"
