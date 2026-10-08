# Build environment that matches an AWS HPC node: Amazon Linux 2023 with the EFA
# installer's libfabric and Open MPI (/opt/amazon/openmpi), GCC and CMake from the
# distribution. Used by .github/workflows/aws.yml to build and test ERF on GitHub's
# runners against the AWS software stack; no EFA hardware is needed to compile.
FROM amazonlinux:2023

RUN dnf -y update && dnf -y install \
      gcc gcc-c++ gcc-gfortran make git tar gzip xz which \
      python3 python3-pip diffutils hostname procps-ng \
    && dnf clean all

# AMReX needs CMake 3.25+; Amazon Linux 2023 ships 3.22. The Kitware binary is what
# AWS ParallelCluster users typically install too.
ARG CMAKE_VERSION=3.31.6
RUN curl -fsSL -o /tmp/cmake.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
    && sh /tmp/cmake.sh --skip-license --prefix=/usr/local \
    && rm -f /tmp/cmake.sh

# EFA installer: libfabric + Open MPI built the way AWS ships them. Kernel module and
# ulimit tweaks are skipped (no hardware in a container; the user-space libraries are
# what ERF links against).
RUN curl -fsSL -o /tmp/efa.tar.gz https://efa-installer.amazonaws.com/aws-efa-installer-latest.tar.gz \
    && tar -xzf /tmp/efa.tar.gz -C /tmp \
    && cd /tmp/aws-efa-installer \
    && ./efa_installer.sh -y --skip-kmod --skip-limit-conf --no-verify \
    && rm -rf /tmp/efa.tar.gz /tmp/aws-efa-installer

ENV PATH=/opt/amazon/openmpi/bin:/opt/amazon/efa/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/amazon/openmpi/lib64:/opt/amazon/openmpi/lib:/opt/amazon/efa/lib64:/opt/amazon/efa/lib \
    OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ARG CCACHE_VERSION=4.12.2
RUN curl -fsSL -o /tmp/ccache.tar.xz https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}-linux-x86_64.tar.xz \
    && tar -xJf /tmp/ccache.tar.xz -C /tmp \
    && install -m 0755 /tmp/ccache-${CCACHE_VERSION}-linux-x86_64/ccache /usr/local/bin/ccache \
    && rm -rf /tmp/ccache*
