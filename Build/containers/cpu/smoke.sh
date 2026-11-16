#!/usr/bin/env bash
# Run inside the development image with networking disabled.
set -euo pipefail
work=$(mktemp -d "${TMPDIR:-/tmp}/erf-cpu-dev.XXXXXX")
trap 'rm -rf "$work"' EXIT
cp -R /erf "$work/source"
export CCACHE_DIR="$work/ccache"
mkdir -p "$CCACHE_DIR"
cp -R /opt/erf-ccache/. "$CCACHE_DIR/"
chmod -R u+rwX "$CCACHE_DIR"

# Change one translation unit without changing the model's scientific behavior.
printf '\nstatic_assert(true, "CPU development image rebuild smoke");\n' >> "$work/source/Exec/ERF_Prob.cpp"
"$work/source/Build/containers/cpu/configure.sh" "$work/source" "$work/build"
cmake --build "$work/build" --parallel 2
ctest --test-dir "$work/build" -L unit -LE parallel --parallel 2 --output-on-failure --no-tests=error
ccache -s

exe="$work/build/Exec/erf_exec"
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
PY
mkdir "$work/run"
cd "$work/run"
"$exe" "$work/source/Exec/RegTests/ScalarAdvDiff/inputs_advect_uniformU" \
  max_step=2 'amr.n_cell=16 16 8' amr.max_grid_size=8 \
  erf.plot_int_1=-1 erf.check_int=-1 > simulation.log 2>&1
grep -Fq 'Coarse STEP 2 ends' simulation.log
printf '%s\n' 'Offline modified-source build, unit tests and two-step CPU smoke passed.'
