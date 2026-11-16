# CPU development image

This image provides a CPU build environment, an ERF source snapshot, and a
compiler cache. It is intended for testing source changes without installing
ERF's build dependencies on the host. The image does not contain a configured
build tree or an installed executable.

The profile enables MPI, particles, double precision, unit tests, and
`fcompare`. It disables NetCDF and FFT. It does not provide RRTMGP, Noah-MP,
EAMxx, or a GPU toolchain. Select a different build for a case that requires
those features. The image is not an EARL container executor or a qualified
scientific baseline.

## Build and test

Start with a clean checkout and initialize the required submodules:

```sh
git submodule update --init --depth 1 Submodules/AMReX Submodules/googletest
docker build --platform linux/amd64 -f Build/containers/cpu/Containerfile --target dev -t erf-cpu-dev:local .
docker run --rm --network=none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$PWD/Build/containers/cpu/smoke.sh,target=/tmp/smoke.sh,readonly" \
  erf-cpu-dev:local bash /tmp/smoke.sh
```

The image build runs the serial unit-test selection. The smoke script copies
source and cache to writable temporary storage, changes one C++ translation
unit, builds and runs the unit tests offline, checks the executable's native
build description, and runs a two-step CPU case. The cache's effectiveness for
changed source and other hosts must be measured; the test does not require a
specific hit rate.

## Scope

The image does not contain Git metadata or attest the identity of a modified
executable. Retain the source revision, modifications, image digest, build
configuration, and executable hash separately when using it in an experiment.
Do not treat a successful compile or smoke test as scientific validation.

The accompanying GitHub workflow builds and tests the image on container-file
changes or manual dispatch. It does not publish an image. Publication, a
versioned provenance contract, additional physics dependencies, and EARL's
container execution integration are follow-up work.
