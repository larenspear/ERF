#!/usr/bin/env bash
# Configures the CPU images: double precision, MPI and particles, without NetCDF
# or the CI-only example fire engine. Keep these options and the image docs in sync.
#
# Usage: Build/container-configure.sh <source dir> <build dir>
# In the erf-dev image: Build/container-configure.sh /erf /erf/build
set -euo pipefail

src=${1:?usage: container-configure.sh <source dir> <build dir>}
build=${2:?usage: container-configure.sh <source dir> <build dir>}

# Ubuntu's MPICH wrapper advertises -flto=auto -ffat-lto-objects, which FindMPI
# would put on every target; -fno-lto overrides it (see ci.yml).
exec cmake -S "$src" -B "$build" \
  -DCMAKE_BUILD_TYPE:STRING=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS_RELWITHDEBINFO="-O2 -g1 -DNDEBUG" \
  -DAMReX_ASSERTIONS:BOOL=ON \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DERF_DIM:STRING=3 \
  -DERF_PRECISION:STRING=DOUBLE \
  -DERF_ENABLE_MPI:BOOL=ON \
  -DMPI_CXX_COMPILE_OPTIONS=-fno-lto \
  -DMPI_CXX_LINK_FLAGS=-fno-lto \
  -DERF_PARALLEL_TEST_ALLOW_OVERSUBSCRIBE:BOOL=ON \
  '-DERF_PARALLEL_TEST_NRANKS:STRING=1;2;4' \
  -DERF_ENABLE_PARTICLES:BOOL=ON \
  -DERF_ENABLE_NETCDF:BOOL=OFF \
  -DERF_ENABLE_FIRE:BOOL=OFF \
  -DERF_FIRE_ENGINE_DIRS:STRING= \
  -DERF_ENABLE_TESTS:BOOL=ON \
  -DERF_ENABLE_ALL_WARNINGS:BOOL=ON \
  -DERF_ENABLE_FCOMPARE:BOOL=ON
