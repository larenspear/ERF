#!/usr/bin/env bash
# Run inside the development image with networking disabled.
set -euo pipefail
work=$(mktemp -d "${TMPDIR:-/tmp}/erf-cpu-dev.XXXXXX")
trap 'rm -rf "$work"' EXIT
src=/erf
build=/tmp/erf-build
if [[ -e "$build" ]]; then
  echo "Expected an empty build path: $build" >&2
  exit 1
fi
ccache -z

# Change one translation unit in this disposable container.
printf '\nstatic_assert(true, "CPU development image rebuild smoke");\n' >> "$src/Exec/ERF_Prob.cpp"
"$src/Build/containers/cpu/configure.sh" "$src" "$build"
cmake --build "$build" --parallel 2
ctest --test-dir "$build" -L unit -LE parallel --parallel 2 --output-on-failure --no-tests=error
ctest --test-dir "$build" -R '^FFTProjection_StretchedOutflowTop$' --output-on-failure --no-tests=error
ctest --test-dir "$build" -R '^SDM_SpeciesDiffusion_column$' --output-on-failure --no-tests=error
ccache -s

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
