#!/bin/bash
# Run one ERF case from the prebuilt CPU image with Docker, Podman or Apptainer: a laptop,
# a workstation or a cluster node. On Apple Silicon the image runs under x86_64 emulation.
#
#   ./cpu-anywhere.sh                                       # smoke test: ScalarAdvDiff, 2 ranks
#   ERF_CASE=~/mycase ERF_INPUTS=inputs ERF_RANKS=8 ./cpu-anywhere.sh
#
# ERF_IMAGE   image (default: the development CPU image)
# ERF_ENGINE  docker, podman or apptainer (default: the first one found)
# ERF_CASE    directory with the inputs file and its data, copied to the run directory;
#             without it the ScalarAdvDiff example from the image runs for ten steps
# ERF_INPUTS  inputs file name inside the case
# ERF_ARGS    extra inputs for the command line
# ERF_RANKS   MPI ranks
# ERF_WORK    run directory (default: erf-run-<timestamp> under the current directory)
# ERF_BIND    an extra host directory to make visible inside the container, for data the
#             inputs name by absolute path
set -euo pipefail

IMAGE=${ERF_IMAGE:-ghcr.io/erf-model/erf:development}
ENGINE=${ERF_ENGINE:-}
WORK=${ERF_WORK:-$PWD/erf-run-$(date +%Y%m%d-%H%M%S)}
RANKS=${ERF_RANKS:-2}
INPUTS=${ERF_INPUTS:-inputs_advect_uniformU}
ARGS=${ERF_ARGS:-}

if [ -z "$ENGINE" ]; then
    for candidate in docker podman apptainer; do
        if command -v "$candidate" > /dev/null; then ENGINE=$candidate; break; fi
    done
fi
[ -n "$ENGINE" ] || { echo "no container engine found; set ERF_ENGINE" >&2; exit 1; }
mkdir -p "$WORK"

# in_image runs one command inside the image with the run directory at /work.
case $ENGINE in
    docker|podman)
        in_image () {
            "$ENGINE" run --rm --platform linux/amd64 -v "$WORK:/work" -w /work \
                ${ERF_BIND:+-v "$ERF_BIND:$ERF_BIND"} "$IMAGE" "$@"
        }
        ;;
    apptainer)
        SIF=${ERF_SIF:-${APPTAINER_CACHEDIR:-$HOME/.apptainer}/erf-sif/$(echo "$IMAGE" | tr '/:' '__').sif}
        mkdir -p "$(dirname "$SIF")"
        [ -s "$SIF" ] || apptainer pull "$SIF" "docker://$IMAGE"
        in_image () {
            apptainer exec --bind "$WORK:/work" --pwd /work ${ERF_BIND:+--bind "$ERF_BIND"} "$SIF" "$@"
        }
        ;;
    *) echo "unknown engine $ENGINE" >&2; exit 1 ;;
esac

if [ -n "${ERF_CASE:-}" ]; then
    cp -a "$ERF_CASE/." "$WORK/"
else
    in_image bash -c 'cp -a "$(dirname "$(dirname "$(command -v erf_exec)")")/share/erf/examples/ScalarAdvDiff/." /work/'
    ARGS=${ERF_ARGS:-"max_step=10 erf.check_int=-1 erf.plot_int_1=10"}
fi

in_image erf_exec --describe-json > "$WORK/erf_build.json" 2> /dev/null || rm -f "$WORK/erf_build.json"
help=$(in_image erf_exec --help 2>&1 || true)
case $help in *--validate-input*) in_image erf_exec --validate-input "$INPUTS" ;; esac

# The fork launcher keeps the image's MPICH from looking for srun when run inside a Slurm job.
# shellcheck disable=SC2086  # ERF_ARGS is a list of inputs
in_image mpiexec -launcher fork -n "$RANKS" erf_exec "$INPUTS" $ARGS
[ -f "$WORK/erf_run.json" ] && cat "$WORK/erf_run.json"
echo "run directory: $WORK"
