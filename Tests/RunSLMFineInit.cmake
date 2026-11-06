cmake_minimum_required(VERSION 3.24)

include("${CMAKE_CURRENT_LIST_DIR}/MPILauncher.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/ResolveExecutable.cmake")

foreach(required TEST_EXE INPUT_DIR WORKING_DIRECTORY)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "RunSLMFineInit.cmake requires ${required}")
    endif()
endforeach()

if(NOT DEFINED MAX_STEP)
    set(MAX_STEP 0)
endif()
if(NOT MAX_STEP MATCHES "^[0-9]+$")
    message(FATAL_ERROR "RunSLMFineInit.cmake requires a nonnegative MAX_STEP")
endif()

erf_resolve_executable(test_exe "${TEST_EXE}" CONFIG "${CONFIG}" CONTEXT "SLM fine-level startup")
erf_mpi_launcher_command(command
    LAUNCHER "${MPIEXEC}"
    NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}"
    NRANKS 1
    PREFLAGS "${MPIEXEC_PREFLAGS}"
    CONTEXT "SLM fine-level startup"
    OPTIONAL)

file(MAKE_DIRECTORY "${WORKING_DIRECTORY}")
file(COPY "${INPUT_DIR}/input_sounding" DESTINATION "${WORKING_DIRECTORY}")

# In a build without NetCDF, the optional file-name table has only its default
# level-zero entry. A file-less SLM fine level must inherit its coarse state
# without indexing that table at level one.
list(APPEND command "${test_exe}" "${INPUT_DIR}/SLM_Restart.i"
    "max_step=${MAX_STEP}" "erf.plot_lsm=false"
    "amr.max_level=1" "amr.ref_ratio_vect=2 2 1"
    "amr.max_grid_size=32" "amr.blocking_factor=2" "amr.n_error_buf=0"
    "erf.refinement_indicators=patch" "erf.patch.max_level=1"
    "erf.patch.in_box_lo_indices_crse=0 0 0"
    "erf.patch.in_box_hi_indices_crse=3 7 15"
    "amrex.throw_exception=1" "amrex.signal_handling=0")

# The first fine land timestep must finish its deferred atmospheric setup
# after inheriting the coarse soil state. Keep this runtime check short and
# leave grid-remake preservation to its separate tests.
if(MAX_STEP GREATER 0)
    list(APPEND command "erf.fixed_dt=0.01" "erf.regrid_int=-1")
endif()

execute_process(COMMAND ${command}
    WORKING_DIRECTORY "${WORKING_DIRECTORY}"
    OUTPUT_VARIABLE output ERROR_VARIABLE errors
    RESULT_VARIABLE result TIMEOUT 90)
file(WRITE "${WORKING_DIRECTORY}/SLM_FineInit.log"
    "RESULT=${result}\n${output}\n${errors}")

if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "SLM fine-level startup failed (${result}); see SLM_FineInit.log")
endif()
if(NOT output MATCHES "BA FROM SCRATCH AT LEVEL 1")
    message(FATAL_ERROR "SLM startup did not create the requested fine level; see SLM_FineInit.log")
endif()
if(MAX_STEP GREATER 0)
    if(NOT output MATCHES "\\[Level 1 step 1\\] Advanced [1-9][0-9]* cells" OR
       NOT output MATCHES "Coarse STEP ${MAX_STEP} ends")
        message(FATAL_ERROR "SLM did not complete the requested coarse and fine timesteps; see SLM_FineInit.log")
    endif()
endif()
