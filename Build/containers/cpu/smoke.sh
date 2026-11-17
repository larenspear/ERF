#!/usr/bin/env bash
# Run inside the development image with networking disabled.
set -euo pipefail
work=$(mktemp -d "${TMPDIR:-/tmp}/erf-cpu-dev.XXXXXX")
src=/erf
build=/tmp/erf-build
collect_logs=false

finish() {
  status=$?
  trap - EXIT
  if ! ccache -s; then
    echo 'Could not report ccache statistics.' >&2
    if [[ "$status" -eq 0 ]]; then status=1; fi
  fi
  if [[ "$status" -ne 0 && "$collect_logs" == true && -n "${ERF_DEV_DIAGNOSTICS:-}" ]]; then
    for entry in \
      "$build/Testing/Temporary/LastTest.log:ctest-last.log" \
      "$build/Tests/test_files/FFTProjection_StretchedOutflowTop/run/simulation.log:fft-simulation.log" \
      "$build/Tests/test_files/SDM_SpeciesDiffusion_column/implicit/simulation.log:particle-implicit.log" \
      "$build/Tests/test_files/SDM_SpeciesDiffusion_column/explicit/simulation.log:particle-explicit.log" \
      "$work/run/simulation.log:scalar-simulation.log"; do
      source_path=${entry%:*}
      if [[ -f "$source_path" ]]; then
        cp "$source_path" "$ERF_DEV_DIAGNOSTICS/${entry##*:}" || echo "Could not copy $source_path" >&2
      fi
    done
  fi
  rm -rf "$work" || echo "Could not remove $work" >&2
  exit "$status"
}
trap finish EXIT

if [[ -e "$build" ]]; then
  echo "Expected an empty build path: $build" >&2
  exit 1
fi
collect_logs=true
ccache -z

# Change one translation unit in this disposable container.
printf '\nstatic_assert(true, "CPU development image rebuild smoke");\n' >> "$src/Exec/ERF_Prob.cpp"
"$src/Build/containers/cpu/configure.sh" "$src" "$build"
cmake --build "$build" --parallel 2
ctest --test-dir "$build" -L unit -LE parallel --parallel 2 --output-on-failure --no-tests=error
# The listener self-test fails unless an MPI rank 1 is present and reports its failure.
ctest --test-dir "$build" -R '^erf_mpi_gtest_listener_self_test$' --output-on-failure --no-tests=error
# Run both independent checks even if one fails; neither failure is suppressed.
test_status=0
ctest --test-dir "$build" -R '^FFTProjection_StretchedOutflowTop$' --output-on-failure --no-tests=error || test_status=$?
ctest --test-dir "$build" -R '^SDM_SpeciesDiffusion_column$' --output-on-failure --no-tests=error || test_status=$?
if [[ "$test_status" -ne 0 ]]; then exit "$test_status"; fi

exe="$build/Exec/erf_exec"
"$exe" --describe-json > "$work/description.json"
python3 - "$work/description.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    description = json.load(source)
assert description["application"] == "ERF"
assert description["gpu_backend"] == "none"
assert description["features"]["mpi"] is True
assert description["features"]["particles"] is True
assert description["features"]["fft"] is True
PY
mkdir "$work/run"
cd "$work/run"
"$exe" "$src/Exec/RegTests/ScalarAdvDiff/inputs_advect_uniformU" \
  max_step=2 'amr.n_cell=16 16 8' amr.max_grid_size=8 \
  erf.plot_int_1=-1 erf.check_int=-1 > simulation.log 2>&1
grep -Fq 'Coarse STEP 2 ends' simulation.log
printf '%s\n' 'Offline modified-source build, unit tests and two-step CPU smoke passed.'
