.. _sec:build:spack:

Installing ERF with Spack
=========================

ERF ships its own `Spack <https://spack.io>`_ package recipe in ``Build/spack``. One
command builds ERF together with its dependencies: AMReX, MPI, NetCDF and HDF5, GoogleTest
for the unit tests, the RTE-RRTMGP/EKAT radiation stack and Noah-MP, each at the version
ERF pins. The recipe is versioned with ERF, so it follows ERF's CMake options and submodule
pins, and the ``Spack package`` CI workflow checks that it does.

Spack's builtin package repository also carries an older ``erf`` recipe. The in-tree one is
the maintained one; once registered it takes precedence.

Requirements
------------

* Spack 1.0 or newer. Registering the recipe straight from ERF's git URL needs Spack 1.1 or
  newer.
* Spack's builtin repository (``spack/spack-packages``) from 2026-10-19 or later, which
  Spack sets up by default. The recipe uses its build systems and its ``amrex``, ``mpi``,
  ``netcdf-c``, ``netcdf-fortran`` and ``googletest`` packages.

Registering the recipe
----------------------

From an ERF checkout:

.. code-block:: bash

   spack repo add /path/to/ERF/Build/spack/spack_repo/erf

Without a checkout, Spack clones ERF into its own cache (``~/.spack/package_repos``) and
finds the recipe through ``spack-repo-index.yaml``. The clone is shallow but it is a full
working tree, about 700 MB, most of it the regression-test gold files, so prefer the path
form when a checkout exists:

.. code-block:: bash

   spack repo add https://github.com/erf-model/ERF.git

Either way ``spack repo list`` then shows ``erf`` ahead of ``builtin``, and ``spack info erf``
lists the versions and variants below. ``spack repo update erf`` refreshes a URL-registered
copy (that fetch brings ERF's full history).

Installing
----------

.. code-block:: bash

   spack install erf                                  # latest release, MPI on, double precision
   spack install erf@develop                          # ERF's development branch with AMReX's
   spack install erf +netcdf +rrtmgp                  # NetCDF I/O and RTE-RRTMGP radiation
   spack install erf +netcdf +noahmp +particles       # Noah-MP land surface, Lagrangian particles
   spack install erf +tests +fcompare                 # the regression and unit tests, AMReX's fcompare
   spack install erf +cuda cuda_arch=80               # NVIDIA A100
   spack install erf +rocm amdgpu_target=gfx90a       # AMD MI250X
   spack install erf +sycl %oneapi                    # Intel GPUs (needs the oneAPI compilers)

``spack spec -I erf <variants>`` shows what a spec would build before building it.

After ``spack load erf``, ``erf_exec`` is on the ``PATH`` and two example cases are in
``$(spack location -i erf)/share/erf/examples``. ``spack test run erf`` runs the recipe's
stand-alone tests (``erf_exec --help`` and two steps of scalar advection) on an installed ERF.

Versions
--------

``develop`` (ERF's ``development`` branch, built against AMReX's ``development`` branch) and
every release from 24.10 on. Each release is pinned to the AMReX release it vendors, since
ERF follows AMReX's API release by release; ``^amrex@<other>`` is refused. Releases before
24.10 always build their bundled AMReX and are not offered.

Variants
--------

Defaults are in parentheses. A variant is offered only for the releases whose CMake has the
option, so for example ``erf@26.01 +windfarm`` is refused.

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Variant
     - CMake option
     - Notes
   * - ``mpi`` (on)
     - ``ERF_ENABLE_MPI``
     - NetCDF and RRTMGP need it.
   * - ``openmp``, ``cuda``, ``rocm``, ``sycl`` (off)
     - ``ERF_ENABLE_OPENMP`` / ``_CUDA`` / ``_HIP`` / ``_SYCL``
     - At most one. ``cuda`` needs ``cuda_arch``; ``rocm`` needs ``amdgpu_target``; ``sycl``
       needs the oneAPI compilers. AMReX is built for the same backend and architectures.
   * - ``netcdf`` (off)
     - ``ERF_ENABLE_NETCDF``
     - Parallel NetCDF-4 (``netcdf-c+mpi``).
   * - ``hdf5`` (follows ``netcdf``)
     - ``ERF_ENABLE_HDF5``
     - AMReX HDF5 plotfiles.
   * - ``particles`` (off)
     - ``ERF_ENABLE_PARTICLES``
     - Lagrangian particles.
   * - ``ml_uphys_diagnostics`` (off, 26.02+)
     - ``ERF_ENABLE_ML_UPHYS_DIAGNOSTICS``
     - ML microphysics diagnostics in the SDM particle microphysics; needs ``+particles``.
   * - ``multiblock`` (off)
     - ``ERF_ENABLE_MULTIBLOCK``
     -
   * - ``fft`` (off, 25.01+)
     - ``ERF_ENABLE_FFT``
     - AMReX's FFT-based solvers (``amrex+fft``).
   * - ``tests`` (off)
     - ``ERF_ENABLE_TESTS``
     - Regression tests, and from 26.06 the unit tests too; the Open MPI launcher gets
       ``--oversubscribe`` for the parallel tests.
   * - ``unit_tests`` (on, 26.06+, with ``+tests``)
     - ``ERF_ENABLE_UNIT_TESTS``
     - The GoogleTest unit tests; GoogleTest comes from Spack. ``~unit_tests`` is ERF's
       regression-tests-only build.
   * - ``fcompare`` (off)
     - ``ERF_ENABLE_FCOMPARE``
     - AMReX's ``fcompare`` (``amrex+plotfile_tools``).
   * - ``precision=double`` (26.06+: ``single``)
     - ``ERF_PRECISION``
     - AMReX is built with the same precision.
   * - ``particles_precision=same`` (26.06+: ``double``, ``single``)
     - ``ERF_PARTICLES_PRECISION``
     - ``same`` follows ``precision``. AMReX is built with the same particle precision
       (this repository carries an ``amrex`` recipe adding that variant until Spack's own
       has it).
   * - ``tools`` (off, 25.07+)
     - ``ERF_ENABLE_TOOLS``
     - Needs ``+netcdf``.
   * - ``rrtmgp`` (off, 25.10+)
     - ``ERF_ENABLE_RRTMGP``
     - RTE-RRTMGP radiation. Builds EKAT with its bundled Kokkos, yaml-cpp, spdlog and Catch2
       from the commits ERF pins; needs ``+netcdf``. On GPUs, Kokkos takes one architecture
       and needs rocThrust (``rocthrust``) with ``+rocm`` or oneDPL (``intel-oneapi-dpl``)
       with ``+sycl``.
   * - ``noahmp`` (off, 25.12+)
     - ``ERF_ENABLE_NOAHMP``
     - Noah-MP from the commit ERF pins for that release; needs ``+netcdf``, adds Fortran,
       ``netcdf-fortran`` and Python.
   * - ``windfarm`` (off, 26.02+)
     - ``ERF_ENABLE_WINDFARM``
     - Wind farm parameterizations.
   * - ``wsm6_fort`` (off, 26.05+), ``wdm6_fort`` (off, 26.09+)
     - ``ERF_ENABLE_WSM6_FORT`` / ``_WDM6_FORT``
     - Fortran microphysics bridges; double precision only.
   * - ``eamxx_shoc`` (off, 26.09+)
     - ``ERF_ENABLE_EAMXX_SHOC``
     - The EAMxx SHOC turbulence scheme. Builds EKAT like ``rrtmgp`` and fetches the two
       EAMxx source directories from E3SM at the commit ``Build/GNU_Ekat/eamxx_clone.sh``
       pins (a sparse clone on Spack 1.1+). Needs ``+mpi``; double precision only.
   * - ``fire`` (off, development)
     - ``ERF_ENABLE_FIRE``
     - The fire coupling, atmosphere side. Fire engines are separate code
       (``ERF_FIRE_ENGINE_DIRS``) and are not built by the recipe.

``%nvhpc +cuda`` builds also set ``ERF_ENABLE_NVHPC`` (25.11+); no CI covers them yet.

Not covered, on purpose: the documentation build (``ERF_ENABLE_DOCUMENTATION``), code
coverage, the warning flags, fire engine directories, the bundled-AMReX build
(``ERF_USE_INTERNAL_AMREX``; Spack always builds against its own ``amrex``), the GNU Make
build, WW3 coupling, and Windows.

Build cache
-----------

The CI workflow pushes ERF's dependencies to an OCI build cache on GHCR. To reuse them:

.. code-block:: bash

   spack mirror add --unsigned erf-buildcache oci://ghcr.io/erf-model/spack-buildcache

Maintaining the recipe
----------------------

The recipe is ``Build/spack/spack_repo/erf/packages/erf/package.py``. The ``Spack package``
workflow (``.github/workflows/spack-package.yml``) runs on pull requests that touch the
recipe, ERF's CMake, ``.gitmodules`` or a submodule, on merges of those, weekly and on
demand:

* ``submodule pins`` runs ``Build/spack/check_pins.py``: the RRTMGP/EKAT and Noah-MP commits
  the recipe pins for ``@develop`` must be the gitlinks in the tree, and the release pins
  must match their tags. A submodule bump fails this check until the recipe follows.
* ``concretization sweep`` runs ``Build/spack/erf_sweep.py`` and ``spack audit``: every
  supported variant combination concretizes, every unsupported one is refused with the
  recipe's own message. Minutes, no build.
* ``spack install`` builds every configuration that changes what is compiled (the lists in
  ``Build/spack/ci_matrix.py``, each traced to the ERF workflow, build script or
  documentation page that builds it) and runs the stand-alone tests; ``macOS``, ``ROCm``
  and ``oneAPI SYCL`` do the same on those platforms. Merges, weekly and on demand. A
  manual run (Actions, *Spack package*, *Run workflow*) builds the whole matrix at a release
  with the ``version`` input (``26.09``) or one spec with the ``spec`` input (a ``+rocm`` or
  ``+sycl`` spec runs in that platform's job).

What to do when:

* **A submodule moves on development.** ``python3 Build/spack/check_pins.py --fix`` rewrites
  the develop Noah-MP pin. The RRTMGP/EKAT stack is one pin set shared by all releases;
  if ``Submodules/RRTMGP`` or ``Submodules/ekat`` moves, update ``_rrtmgp_stack`` with the
  new gitlinks and EKAT's ``extern/*`` gitlinks at the new EKAT commit.
* **A release is tagged.** Add it to the version table with the tarball's sha256
  (``spack checksum erf <version>``), and a Noah-MP pin line for it
  (``git ls-tree <tag> Submodules/Noah-MP``). The weekly run fails until the newest tag is
  offered.
* **A CMake option is added.** Add a variant with ``when="@<first release>:"``, its
  ``define_from_variant`` in ``cmake_args``, any conflict ERF's CMake enforces, and a
  positive and a negative spec in ``erf_sweep.py``.

Before opening the pull request: ``ruff format`` and ``ruff check`` on the Python files,
and, with the recipe registered, ``spack audit packages erf`` and
``python3 Build/spack/erf_sweep.py``.
