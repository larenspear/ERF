.. _BuildDescription:

Inspect an ERF build
===================

Use the executable's build description to check compiled capabilities without
an input deck, source checkout, or build directory.

Read compiled capabilities
--------------------------

.. code-block:: console

   erf_exec --describe-json

The command emits one JSON object on standard output and exits with status zero.
It needs no input file and does not initialize AMReX, Kokkos, or a GPU device.
MPI builds still initialize MPI; the installed MPI libraries and their launcher
must work. The executable also needs its linked libraries; a CUDA build may
require the driver library even though it does not initialize a device. Under
MPI, only rank zero writes the object. ``--describe`` retains the human-readable
build report.

For a human-readable YAML representation of the same data, use:

.. code-block:: console

   erf_exec --describe-yaml

Both formats use the same field definitions and schema version. YAML emits
indented mappings, quotes every string value, and uses no custom tags or aliases.
Version strings and GPU target tokens therefore keep their string types when
loaded by a YAML parser. ERF does not need a YAML library or Python to emit either
format. Python 3 is required at build time to generate source metadata.
Prefer JSON for automated calls: its smaller parsing rules and wide
tool support make it the primary machine interface.

Schema version 1 has the following fields. Consumers must check
``schema_version`` and can ignore additional fields within a supported version.

.. list-table:: Executable metadata
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``application``, ``dimension``
     - ``ERF`` and the number of spatial dimensions.
   * - ``real_bits``, ``particle_real_bits``
     - Compiled floating-point widths. Particle precision is null when particles
       are disabled.
   * - ``gpu_backend``
     - ``none``, ``cuda``, ``hip``, or ``sycl``.
   * - ``features``
     - Boolean compile-time switches, including MPI, OpenMP, particles, FFT,
       NetCDF, AMReX HDF5, Noah-MP, RRTMGP, native/EAMxx SHOC, wind farms,
       Fortran microphysics bridges, multiblock, and WW3 coupling. These describe
       available code, not the physics selected by an input deck.
   * - ``build.build_system``
     - ``cmake`` or ``gnu-make``.
   * - ``build.source``, ``build.amrex_source``
     - Full Git ``revision`` and ``tracked_dirty`` when available. Null means
       unknown, including a source archive or externally installed AMReX.
       Dirty status covers tracked files and submodule differences; it does not
       hash untracked files, build inputs, dependencies, or the executable.
   * - ``build.gpu_target_settings``
     - Architecture settings supplied by the build system, such as ``["80"]``
       or ``["gfx90a"]``. CPU builds report ``[]``; unavailable GPU settings,
       including SYCL without ahead-of-time targets, report null. Tokens such
       as ``native`` remain unresolved. This field is not a device probe or
       an inspection of embedded device code.
   * - ``source_describe``
     - Existing ERF Git description. This may contain a tag or dirty suffix;
       it is not a full revision identifier.
   * - ``compiler``, ``dependencies``
     - Compiler name/version and available dependency versions/descriptions.
       The FFT provider follows the compiled backend: FFTW, cuFFT, rocFFT, or
       oneMKL. Unavailable versions are null. This is not a dependency lockfile.
   * - ``mpi_standard``
     - Compile-time MPI standard major/minor, or null. This does not identify
       the MPI implementation or establish ABI compatibility with host MPI.

CMake captures source identity when configuring; reconfigure after changing
source. GNU Make captures it when regenerating build information before linking.
Neither a matching revision nor these capabilities establish an exact reusable
build. Keep the container digest, recipe/dependency identity, driver requirements,
and site qualification record in the downstream selection policy.
For Spack packages, retain the concrete spec alongside this description to record
the resolved dependencies, compiler, variants, and architecture target. The spec
and its DAG hash identify the concretized build; hash the shipped executable
separately when an exact artifact identity is needed.

Use the description from an agent
--------------------------------

The machine-readable schema is installed as
``share/erf/schema/erf-build-description-v1.schema.json``. It uses JSON Schema
Draft 2020-12 and validates either parsed JSON or the object obtained by safely
loading YAML. The schema describes required fields, types, allowed values, and
the relationship between enabled MPI/particles and their detail fields.
It permits additional fields for compatible extensions. Changing the type or
meaning of an existing field requires a new ``schema_version``.

An automated caller should:

1. Invoke ``--describe-json`` with an argument list, a timeout, and separate
   stdout/stderr capture. Check the exit status before parsing stdout. Keep
   stderr with the diagnostic evidence.
2. Parse one document, check ``schema_version``, and validate against the matching
   schema. Treat malformed output or an unsupported version as an inspection
   error; these do not establish that ERF needs rebuilding.
3. Use exact field names and types. ``false`` means a capability is disabled;
   ``null`` means information is unknown or not applicable as described by that
   field. Unknown information must not satisfy a compatibility requirement.
   Keep revision identifiers and GPU targets as strings.
4. Compare required capabilities and precision. Use the full container build
   identity and digest for reuse decisions. A matching description alone does
   not prove dependency, driver, or host-MPI compatibility.
5. Test the selected application in the intended execution environment:
   compile-time availability does not establish site compatibility.

For example, a Python caller can validate the description using the optional
``jsonschema`` package. ``schema_path`` names the installed schema and
``executable`` names the selected ERF binary:

.. code-block:: python

   import json
   import subprocess
   from pathlib import Path
   from jsonschema import Draft202012Validator

   result = subprocess.run(
       [executable, "--describe-json"],
       capture_output=True, text=True, check=True, timeout=30,
   )
   description = json.loads(result.stdout)
   schema = json.loads(Path(schema_path).read_text())
   Draft202012Validator(schema).validate(description)
   if not all(description["features"][name] is True
              for name in ("particles", "fft")):
       raise RuntimeError("Selected ERF lacks required particles/FFT support")

Test the metadata formats
-------------------------

Generator and executable tests use Python's standard library. Optional format
tests use a C++17 compiler, PyYAML, and jsonschema to check string escaping, scalar
types, schema failures, and equivalence of the installed executable's outputs:

.. code-block:: console

   python3 Tests/BuildDescription/test_description.py \
     --executable /path/to/install/bin/erf_exec

Use ``--launcher mpiexec -n 2`` as the final arguments to check single-document
output under MPI.
These are developer checks, not additional ERF build or runtime dependencies.

CMake registers these checks with the ``description`` CTest label when ERF tests
are enabled. Set ``Python3_EXECUTABLE`` to a Python environment containing
``PyYAML`` and ``jsonschema`` to enable the optional parser/schema checks;
otherwise those checks are reported as skipped. For example:

.. code-block:: console

   ctest --test-dir build -L description --output-on-failure
