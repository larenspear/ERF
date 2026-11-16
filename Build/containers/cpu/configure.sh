#!/usr/bin/env bash
# Configure the CPU development profile. Pass a writable source and build tree.
set -euo pipefail
src=${1:?usage: configure.sh SOURCE BUILD}
build=${2:?usage: configure.sh SOURCE BUILD}

exec cmake -S "$src" -B "$build" \
  -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO='-O2 -g1 -DNDEBUG' \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DAMReX_ASSERTIONS:BOOL=ON \
  -DERF_DIM:STRING=3 \
  -DERF_PRECISION:STRING=DOUBLE \
  -DERF_ENABLE_MPI:BOOL=ON \
  -DMPI_CXX_COMPILE_OPTIONS=-fno-lto \
  -DMPI_CXX_LINK_FLAGS=-fno-lto \
  -DERF_ENABLE_PARTICLES:BOOL=ON \
  -DERF_ENABLE_NETCDF:BOOL=OFF \
  -DERF_ENABLE_FFT:BOOL=OFF \
  -DERF_ENABLE_TESTS:BOOL=ON \
  -DERF_ENABLE_FCOMPARE:BOOL=ON
