# ERF's Spack package

`spack_repo/erf` is a [Spack](https://spack.io) package repository with one
package, `erf`. It is kept here, next to ERF's CMake, so that a new release, a
new CMake option or a submodule bump and the recipe that follows it land in the
same pull request. How to use it: `Docs/sphinx_doc/buildingSpack.rst`
(rendered at https://erf.readthedocs.io/en/latest/buildingSpack.html).

```
spack repo add /path/to/ERF/Build/spack/spack_repo/erf   # any Spack 1.x
spack repo add https://github.com/erf-model/ERF.git        # Spack 1.1+, no checkout needed
spack install erf
```

| File | Purpose |
|---|---|
| `spack_repo/erf/packages/erf/package.py` | the recipe |
| `spack_repo/erf/repo.yaml` | repository namespace (`erf`) and package API version |
| `check_pins.py` | the recipe's RRTMGP/EKAT and Noah-MP pins must equal ERF's submodule gitlinks; `--fix` updates the develop Noah-MP pin |
| `erf_sweep.py` | concretization sweep: supported variant combinations concretize, unsupported ones are refused |
| `ci_matrix.py` | the workflows' build lists, as JSON for a version (`--version 26.09`) or one spec (run in the ROCm or SYCL job when it asks for that backend), and the container image list |
| `../../spack-repo-index.yaml` | tells `spack repo add <git URL>` where this repository is |
| `../../.github/workflows/spack-package.yml` | runs the pin check and the sweep on relevant PRs, and the build matrix on merges, weekly and on demand (at a release with the `version` input) |
| `../../.github/workflows/spack-containers.yml` | builds the image list through the recipe and pushes runnable images to `ghcr.io/<owner>/erf:<version>` (Spack layers on `ubuntu:22.04`), weekly and on demand |

## Policies

* **This recipe is canonical.** Spack's builtin repository (`spack/spack-packages`)
  also carries an `erf` package; it is a snapshot that may lag. Registering this
  repository puts it ahead of builtin, so `spack install erf` uses this one.
* **Pins move with the tree.** The RRTMGP/EKAT, Noah-MP and E3SM commits the recipe
  fetches, and the AMReX commit `@develop` builds, must equal ERF's gitlinks: the pin
  check fails a pull request that bumps a submodule without the recipe
  (`check_pins.py --fix` rewrites the develop Noah-MP and AMReX pins). Releases build
  the AMReX release of the same month, the one their submodule vendors.
* **A release adds one step.** Add the version with the tarball's sha256 and its
  Noah-MP pin (the docs page says how); the weekly run fails until the newest tag is
  offered.
* **Overrides are temporary.** `packages/amrex` exists only for `particles_precision`;
  delete it once spack-packages carries that variant.

Other projects keep their recipe the same way, for example ParaView
(`Utilities/spack/repo`), Axom (`scripts/spack`), COSMA (`spack_repo/cosma`), Rayleigh
(`spack-repo`) and Amanzi (`config/spack`).
