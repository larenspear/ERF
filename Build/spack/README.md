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
| `ci_matrix.py` | the workflow's build lists, as JSON for a version (`--version 26.09`) or one spec (run in the ROCm or SYCL job when it asks for that backend) |
| `ci_build.py` | concretizes one environment, builds its lockfile and saves build/test diagnostics |
| `../../spack-repo-index.yaml` | tells `spack repo add <git URL>` where this repository is |
| `../../.github/workflows/spack-package.yml` | runs the pin check and the sweep on relevant PRs, and the build matrix on merges, weekly and on demand (at a release with the `version` input) |

## Policies

* **This recipe is canonical.** Spack's builtin repository (`spack/spack-packages`)
  also carries an `erf` package; it is a snapshot that may lag. Registering this
  repository puts it ahead of builtin, so `spack install erf` uses this one.
* **Pins move with the tree.** The RRTMGP/EKAT, Noah-MP and E3SM commits the recipe
  fetches, and the AMReX commit `@develop` builds, must equal ERF's gitlinks: the pin
  check fails a pull request that bumps a submodule without the recipe
  (`check_pins.py --fix` rewrites the develop Noah-MP and AMReX pins). Releases from
  24.10 build the AMReX release of the same month (24.10 builds 24.11: its submodule
  is five commits past the 24.10 tag), with embedded boundaries off before 25.03 as
  ERF's CMake sets them; 23.10 to 24.09 build the AMReX they bundle.
* **A release adds one step.** Add the version with the tarball's sha256 and its
  Noah-MP pin (the docs page says how); the weekly run fails until the newest tag is
  offered.
* **Overrides are temporary.** `packages/amrex` exists only for `particles_precision`;
  delete it once spack-packages carries that variant.

Other projects keep their recipe the same way, for example ParaView
(`Utilities/spack/repo`), Axom (`scripts/spack`), COSMA (`spack_repo/cosma`), Rayleigh
(`spack-repo`) and Amanzi (`config/spack`).

## Optional validation on hosted runners

The `Spack package` workflow accepts these manual inputs in addition to `spec`
and `version`. Each job has a timeout and uploads diagnostics for 14 days, even
when its build or tests fail. Runs continue while your computer is off.

| Input | Use |
|---|---|
| `run_tests=true` | Run the full CTest suite during installation for CPU specs with `+tests`. GPU jobs compile only; a single GPU spec with this input is rejected. |
| `spack_ref` / `packages_ref` | Try another Spack or package repository commit without changing the normal pins. Use commit SHAs for reproducible comparisons. |
| `runner=ubuntu-24.04-arm` | Build one Linux CPU spec on a native ARM64 hosted runner. |
| `build_mode=develop` | Build the checked-out ERF source with `spack develop`. Requires one `erf@develop` CPU spec. |
| `build_mode=reuse` | Install a single-precision, non-MPI ERF first, then solve and build the requested CPU spec in that populated store. |

An explicit `platform=darwin` routes a single spec to the macOS runner. To compare
the regression results with ERF's generic Linux CI, use
`erf@26.09 +particles +tests +fcompare target=x86_64`. FMA-capable targets can
change results relative to that baseline; diagnostics preserve the numerical
failure rather than relaxing a tolerance or silently retrying it.

Each build uses one `spack.yaml` and `spack.lock` for installation, stand-alone
tests and cache publication. Its artifact contains those files, the requested
spec and tool revisions, installed specs, CMake/CTest logs and any AMReX
`Backtrace.*` files. Build environment dumps and mirror configuration are
excluded. Full builds retain their stages until the runner is discarded.

Run the workflow helper checks locally with:

```sh
python3 -m unittest discover -s Build/spack/tests
```
