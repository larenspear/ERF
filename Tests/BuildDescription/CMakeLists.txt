# These checks never initialize the simulation or require an input deck.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(_description_test ${CMAKE_CURRENT_SOURCE_DIR}/test_description.py)
set(_description_args "")
if(TARGET erf_exec)
  set(_backend none)
  foreach(_gpu IN ITEMS CUDA HIP SYCL)
    if(ERF_ENABLE_${_gpu})
      string(TOLOWER "${_gpu}" _backend)
    endif()
  endforeach()
  set(_bits 64)
  if(ERF_PRECISION STREQUAL "SINGLE")
    set(_bits 32)
  endif()
  list(APPEND _description_args --executable $<TARGET_FILE:erf_exec>
    --expect-backend ${_backend} --expect-real-bits ${_bits})
  foreach(_feature IN ITEMS MPI PARTICLES FFT)
    string(TOLOWER "${_feature}" _name)
    if(ERF_ENABLE_${_feature})
      list(APPEND _description_args --expect-${_name} ON)
    else()
      list(APPEND _description_args --expect-${_name} OFF)
    endif()
  endforeach()
endif()

add_test(NAME erf_build_description
  COMMAND ${Python3_EXECUTABLE} ${_description_test} ${_description_args})
set(_description_tests erf_build_description)
if(TARGET erf_exec AND ERF_ENABLE_MPI)
  include(${PROJECT_SOURCE_DIR}/Tests/MPILauncher.cmake)
  erf_mpi_launcher_command(_launcher
    LAUNCHER "${MPIEXEC_EXECUTABLE}" NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS "${ERF_TEST_NRANKS}" PREFLAGS "${MPIEXEC_PREFLAGS}")
  add_test(NAME erf_build_description_mpi
    COMMAND ${Python3_EXECUTABLE} ${_description_test} ${_description_args}
      --launcher ${_launcher})
  set_tests_properties(erf_build_description_mpi PROPERTIES PROCESSORS ${ERF_TEST_NRANKS})
  list(APPEND _description_tests erf_build_description_mpi)
endif()
foreach(_test IN LISTS _description_tests)
  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${_test}")
  set_tests_properties(${_test} PROPERTIES LABELS "infrastructure;description"
    TIMEOUT 120 ENVIRONMENT
      "CXX=\"${CMAKE_CXX_COMPILER}\";TMPDIR=${CMAKE_CURRENT_BINARY_DIR}/${_test}")
endforeach()
