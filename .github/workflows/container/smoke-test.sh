#!/usr/bin/env bash
# Exercise the runtime image as a user: bundled input, MPI, and writable output.
# Usage: bash smoke-test.sh IMAGE [OUTPUT_DIRECTORY]
set -euo pipefail

image=${1:?usage: smoke-test.sh IMAGE [OUTPUT_DIRECTORY]}
output_root=${2:-$(mktemp -d "${TMPDIR:-/tmp}/erf-container-smoke.XXXXXX")}
mkdir -p "$output_root"
output_root=$(cd "$output_root" && pwd -P)

for ranks in 1 2; do
  run_dir="$output_root/np$ranks"
  # Do not let output from an earlier test satisfy this run's assertions.
  mkdir "$run_dir"
  docker run --rm --platform linux/amd64 \
    --user "$(id -u):$(id -g)" \
    --mount "type=bind,source=$run_dir,target=/work" \
    --workdir /work "$image" \
    mpiexec -n "$ranks" erf_exec /usr/local/share/erf/inputs_scalar_advection \
    max_step=5 'amr.n_cell=16 16 8' amr.max_grid_size=8 \
    erf.plot_int_1=5 erf.check_int=-1 2>&1 | tee "$run_dir/run.log"

  grep -Fq 'Coarse STEP 5 ends' "$run_dir/run.log"
  test -s "$run_dir/plt00005/Header"
  grep -Eq "^number of MPI processes: $ranks$" "$run_dir/plt00005/job_info"
  echo "Runtime image passed with $ranks MPI rank(s); output: $run_dir"
done
