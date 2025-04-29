#!/bin/bash

#By default, Apple aliases GCC to their clang
#If you install GCC via homebrew (brew install gcc), you need to use the command gcc-14
#Set the environment variables below for OpenMPI or MPICH (both installable via brew)

export OMPI_CC=gcc-14
export OMPI_CXX=g++-14
export OMPI_FC=gfortran-14

#export MPICH_CC=gcc-14
#export MPICH_CXX=g++-14
#export MPICH_FC=gfortran-14

cmake -DCMAKE_INSTALL_PREFIX:PATH=./install \
      -DCMAKE_CXX_COMPILER:STRING=mpicxx \
      -DCMAKE_C_COMPILER:STRING=mpicc \
      -DCMAKE_Fortran_COMPILER:STRING=mpifort \
      -DCMAKE_OSX_ARCHITECTURES=arm64 \
      -DMPIEXEC_PREFLAGS:STRING=--oversubscribe \
      -DCMAKE_BUILD_TYPE:STRING=Release \
      -DERF_DIM:STRING=3 \
      -DERF_ENABLE_MPI:BOOL=ON \
      -DERF_ENABLE_TESTS:BOOL=ON \
      -DERF_ENABLE_FCOMPARE:BOOL=ON \
      -DERF_ENABLE_DOCUMENTATION:BOOL=OFF \
      -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=ON \
      .. && make -j8
