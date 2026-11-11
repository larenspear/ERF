 .. role:: cpp(code)
    :language: c++

.. _containers:

Containers
==========

ERF publishes runnable images. This page shows which images exist, how to run a
case from them on Deception, Perlmutter, Polaris and any machine with a
container engine, and how to build an image of your own on Perlmutter.

Prebuilt images
---------------

``.github/workflows/spack-containers.yml`` builds images from the Spack recipe in
``Build/spack`` every Monday and at every release:

* ``ghcr.io/erf-model/erf:development`` and ``ghcr.io/erf-model/erf:<release>``:
  the CPU image, for ``x86_64_v3`` (AVX2) processors, on Ubuntu 24.04.
* ``ghcr.io/erf-model/erf-cuda:development-sm80`` and
  ``ghcr.io/erf-model/erf-cuda:<release>-sm80``: the same stack compiled for CUDA
  on A100 GPUs (compute capability 8.0), on NVIDIA's CUDA 12.8 runtime image for
  Ubuntu 24.04. It does not run on V100 or H100 GPUs.

The Ubuntu 24.04 bases carry glibc 2.39, which matters on Cray systems: the
site's Cray MPICH, bound into the container for multi-node runs, needs glibc
2.38, so images on older bases cannot use it.

Both carry MPI (MPICH), NetCDF and HDF5, Noah-MP, RRTMGP, FFT and particles, so
every capability an inputs file can select is available. ``erf_exec`` is on
``PATH`` and the examples are under ``share/erf/examples``. Every tag has a
dated copy that never moves, ``development-20261110`` for example: use the dated
tag for anything that must be reproducible, and ``erf_exec --describe-json`` to
record what an image contains.

Running a case from an image
----------------------------

The scripts in ``Build/containers`` run one case from an image on each machine.
They read the same environment variables:

* ``ERF_IMAGE``: the image; the default is the development image for the machine.
* ``ERF_CASE``: a directory holding the inputs file and the data it names, copied
  into the run directory. Without it the script runs the ``ScalarAdvDiff`` example
  from the image for ten steps, which is the smoke test for a new machine.
* ``ERF_INPUTS``: the inputs file name inside the case.
* ``ERF_ARGS``: extra inputs for the command line, ``max_step=100 erf.plot_int_1=10`` for example.
* ``ERF_RANKS``: MPI ranks; on the GPU machines one per GPU.
* ``ERF_WORK``: the run directory; the default is a new directory beside the job.
* ``ERF_BIND``: an extra directory to make visible inside the container, for data
  the inputs name by absolute path.

Each script validates the inputs with ``erf_exec --validate-input`` when the
image has it, records ``erf_exec --describe-json`` as ``erf_build.json`` in the
run directory, passes ``amrex.use_gpu_aware_mpi=0`` because the image's MPICH is
not CUDA-aware, and prints ``erf_run.json`` when the image writes one.

Deception (PNNL)
~~~~~~~~~~~~~~~~

Slurm and Apptainer. Submit from a directory on ``/rcfs``. The image is pulled
once into ``/rcfs/scratch/$USER/erf-sif`` on a login node, as the script's
header shows; compute nodes cannot mount a SIF, so each job unpacks it onto the
node's local disk on first use.
Use an A100 partition, ``a100_80_shared`` or ``a100_shared``, which bill only the
GPUs and cores requested. MPI runs inside the container, so this script is for
one node.

.. literalinclude:: ../../Build/containers/deception-a100.sbatch
   :language: bash

Perlmutter (NERSC)
~~~~~~~~~~~~~~~~~~

Slurm and ``podman-hpc``. Pull the image once on a login node, which also
migrates it to ``$SCRATCH``. ``--mpi`` swaps the image's MPICH for Cray MPICH, so
runs span nodes, and ``--gpu`` exposes the A100s. Submit with the project's GPU
account, ``sbatch -A m1234_g`` for example.

.. literalinclude:: ../../Build/containers/perlmutter-a100.sbatch
   :language: bash

Polaris (ALCF)
~~~~~~~~~~~~~~

PBS and Apptainer, which on Polaris runs only on compute nodes, behind the ALCF
proxy and with ``--fakeroot``. The image is pulled on first use. Cray MPICH and
Cray's libfabric are bound over the image's own MPICH and libfabric directories,
together with the host libraries they need, because Spack links ``erf_exec``
with an RPATH that would otherwise keep the image's copies; the script checks
that the libraries it then loads are Cray's and that the Slingshot provider is
in use. It binds ``/opt/cray`` rather than ALCF's example ``/opt``, which would
hide the image's own ``/opt/software``. Verified with eight ranks on two nodes.

.. literalinclude:: ../../Build/containers/polaris-a100.pbs
   :language: bash

Any machine with a CPU
~~~~~~~~~~~~~~~~~~~~~~

Docker, Podman or Apptainer, from a laptop to a cluster node. On Apple Silicon
the image runs under x86_64 emulation. For Perlmutter's CPU nodes use the
Perlmutter script with ``ERF_IMAGE=ghcr.io/erf-model/erf:development`` and
``sbatch -C cpu --ntasks-per-node=128 -c 1``, without ``--gpu``.

.. literalinclude:: ../../Build/containers/cpu-anywhere.sh
   :language: bash

Building your own image on Perlmutter
-------------------------------------

The following is a brief introduction to building ERF containers on the NERSC
Perlmutter platform. For more details, please see `NERSC's detailed containers
documentation <https://docs.nersc.gov/development/containers>`_, which also
includes containers tutorials.

Container images can be built on one's desktop/laptop using a standard container
framework such as Docker, Podman (Pod Manager), etc. or directly on a Perlmutter
login node using ``podman-hpc``.

Example ERF containerfile
~~~~~~~~~~~~~~~~~~~~~~~~~

The following is an example ERF containerfile with filename ``erf_containerfile``:

.. code:: shell

     1  FROM nvcr.io/nvidia/cuda:12.2.0-devel-ubuntu22.04
     2
     3  WORKDIR /app
     4
     5  RUN apt-get update -y && \
     6      DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
     7      g++-12 gcc-12 gfortran-12 git libtool make tar autoconf automake wget python3 cmake
     8
     9  # MPICH to be swapped out later for Cray MPI
    10
    11  RUN wget https://www.mpich.org/static/downloads/4.2.2/mpich-4.2.2.tar.gz && \
    12      tar xzf mpich-4.2.2.tar.gz && cd mpich-4.2.2 && \
    13      ./configure CC=/usr/bin/gcc-12 CXX=/usr/bin/g++-12 F77=/usr/bin/gfortran-12 FC=/usr/bin/gfortran-12 && \
    14      make -j8 && make install && \
    15      cd .. && rm -rf mpich-4.2.2 mpich-4.2.2.tar.gz
    16
    17  RUN mkdir /app/erf && cd /app/erf && wget https://github.com/erf-model/ERF/archive/refs/tags/24.06.tar.gz && \
    18    tar xzf 24.06.tar.gz && cd ERF-24.06/Submodules && \
    19    wget https://github.com/AMReX-Codes/amrex/releases/download/24.06/amrex-24.06.tar.gz && \
    20    tar xzf amrex-24.06.tar.gz && rmdir AMReX && mv amrex AMReX && cd .. && mkdir MyBuild && cd MyBuild && \
    21    cmake \
    22    -DCMAKE_C_COMPILER=mpicc \
    23    -DCMAKE_CXX_COMPILER=mpicxx \
    24    -DCMAKE_Fortran_COMPILER=mpif90 \
    25    -DCMAKE_BUILD_TYPE:STRING=Release \
    26    -DCMAKE_CUDA_ARCHITECTURES=80 \
    27    -DERF_ENABLE_MPI:BOOL=ON \
    28    -DERF_ENABLE_CUDA=ON \
    29    .. && \
    30    make -j8

Line numbers were added for instructional purposes but should not appear in containerfile.
The containerfile is available at https://github.com/erf-model/ERF/blob/development/Build/erf_containerfile

* Line 1 downloads a container base image from NVIDIA's container registry that contains the Ubuntu 22.04 operating system and CUDA 12.2.0
* Line 3 sets the working directory
* Lines 5-7 download the GNU 12 compilers and all the necessary utilities for building ERF in the container
* Lines 11-15 download the MPICH source code and build it.  At runtime this MPICH will get replaced by Perlmutter's Cray MPI
* Lines 17-30 wget the ERF and AMReX 24.06 tarballs and build with cmake/make



Build the container on Perlmutter using ``podman-hpc`` and using the containerfile ``erf_containerfile`` with name ``erf`` and tag ``1.00`` (``-t <name>:<tag>``)

   .. code:: shell

    podman-hpc build -t erf:1.00 -f erf_containerfile

In order to use this image in a job or access it from any other login node, one needs to migrate the image onto the $SCRATCH filesystem by issuing the following command:

.. code:: shell

  podman-hpc migrate erf:1.00

``podman-hpc images`` will display the following

.. code:: shell

  user@perlmutter:login12:~> podman-hpc images
  REPOSITORY                    TAG                         IMAGE ID      CREATED       SIZE        R/O
  localhost/erf                 1.00                        893605c3ee9b  5 hours ago   16.1 GB     true
  localhost/erf                 1.00                        893605c3ee9b  5 hours ago   16.1 GB     false

Note that ``localhost`` will not be needed for podman-hpc commands.

Run container on Perlmutter
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Submit the following slurm batch script in order to use the image in a job

.. code:: shell

  #!/bin/bash

  #SBATCH --account=<proj>
  #SBATCH --constraint=gpu
  #SBATCH --job-name=erf
  #SBATCH --nodes=1
  #SBATCH --time=0:05:00
  #SBATCH -q regular

  srun -N 1 -n 4 -c 32 --ntasks-per-node=4 --gpus-per-node=4 ./device_wrapper \
  podman-hpc run --rm --mpi --gpu -v /pscratch/sd/u/user/erf/abl:/run -w /run erf:1.00 \
  /app/erf/ERF-24.06/MyBuild/Exec/erf_exec CanonicalTests/ABL/inputs_smagorinsky amrex.use_gpu_aware_mpi=0

``device_wrapper`` script:

.. code:: shell

      #!/bin/bash
      # select_cpu_device wrapper script
      export CUDA_VISIBLE_DEVICES=$((3-$SLURM_LOCALID))
      exec $*

Arguments for ``podman-hpc run`` used above

* ``--rm`` removes the container after exit
* ``--mpi`` enables Cray MPI support (swaps MPICH in the container for Perlmutter's Cray MPI)
* ``--gpu`` enables NVIDIA GPU support
* ``-v /pscratch/sd/u/user/erf/abl:/run`` mounts ``/pscratch/sd/u/user/erf/abl`` on Perlmutter onto ``/run`` in the container
* ``-w /run`` makes the ``/run`` directory inside the container the working directory, i.e. any output from the ERF run will be written to the ``/run`` directory in the container, which will appear in the ``/pscratch/sd/u/user/erf/abl`` directory on Perlmutter.
* ``erf:1.00`` container name and tag
* ``/app/erf/ERF-24.06/MyBuild/Exec/erf_exec`` ERF binary in container

The remaining arguments are the normal ERF command line arguments.

Please issue ``podman-hpc --help`` for the help page and ``podman-hpc run --help`` for the ``podman-hpc run`` help page.

Container image libraries
~~~~~~~~~~~~~~~~~~~~~~~~~

Container image libraries provide a convenient way to store and share images.
The best known one is probably Docker Hub.  NERSC provides a private registry to its users via `registry.nersc.gov <https://docs.nersc.gov/development/containers/registry>`_.

In order to push the ERF image to Docker Hub, need to modify the tag, login to Docker Hub, and then push the image.

.. code:: shell

   podman-hpc tag erf:1.00 docker.io/docker_hub_username/erf:1.00
   podman-hpc login docker.io
   podman-hpc push docker.io/docker_hub_username/erf:1.00

Shifter container runtime
~~~~~~~~~~~~~~~~~~~~~~~~~

Shifter is a NERSC-developed tool that provides an alternative method for running containers on Perlmutter.   Recall that Shifter cannot be used to build images. `NERSC's containers documentation <https://docs.nersc.gov/development/containers>`_ provides an introduction to shifter including a tutorial.

Use the ``shifterimg pull`` command to pull images directly from Docker Hub and it will automatically convert your Docker image into Shifter format.

Need to first login to Docker Hub, then pull.  ``--user nersc_user`` will restrict usage of the image to only ``nersc_user``.

.. code:: shell

   shifterimg login docker.io
   shifterimg -v --user nersc_user pull docker_hub_username/erf:1.00

Submit the following slurm batch script in order to use the image in a job

.. code:: shell

  #!/bin/bash

  #SBATCH --account=<proj>
  #SBATCH --constraint=gpu
  #SBATCH --job-name=erf
  #SBATCH --nodes=1
  #SBATCH --time=0:05:00
  #SBATCH -q regular
  #SBATCH --image=docker_hub_username/erf:1.00 # for shifter, not podman-hpc
  #SBATCH --module=mpich,gpu  # for shifter, not podman-hpc; CPU-only MPI
  ##SBATCH --module=cuda-mpich  # for shifter, not podman-hpc; GPU-Aware MPI
  #SBATCH --volume="<$SCRATCH/my_erf_run_dir>:/run"  # for shifter, not podman-hpc

  srun -N 1 -n 4 -c 32 --ntasks-per-node=4 --gpus-per-node=4 ./device_wrapper \
  shifter \
  /app/erf/ERF-24.06/MyBuild/Exec/erf_exec CanonicalTests/ABL/inputs_smagorinsky amrex.use_gpu_aware_mpi=0



Common Issues
~~~~~~~~~~~~~

* Using ``podman`` rather than ``podman-hpc`` on Perlmutter (best to always use ``podman-hpc``)
* Before issuing ``podman-hpc migrate <name>:<tag>`` after having issued the command earlier with identical ``<name>:<tag>``, if want to keep the same name, please change the ``<tag>`` to one that has not been used previously.  If want to use an identical ``<name>:<tag>`` used in a previous ``podman-hpc migrate`` command, please first issue ``podman-hpc rmsqi <name>:<tag>`` to delete the old image.  Otherwise could potentially end up with errors such as

    .. code:: shell

      Error: read-only image store assigns the same name to multiple images

  and will have resort to `various methods <https://docs.nersc.gov/development/containers/podman-hpc/overview/#troubleshooting>`_ to get out of the bad configuration state.

* The default containerfile is a file called ``Containerfile`` (case sensitive).  When that file is being used, can replace ``-f erf_containerfile`` with a period:

   .. code:: shell

    podman-hpc build -t erf:1.00 .

  Note that for this case, the period is mandatory. Here it does not denote the end of a sentence.
