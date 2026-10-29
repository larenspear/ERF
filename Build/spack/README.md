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
| `ci_matrix.py` | the workflow's build lists, as JSON for a version (`--version 26.09`) or one spec |
| `../../spack-repo-index.yaml` | tells `spack repo add <git URL>` where this repository is |
| `../../.github/workflows/spack-package.yml` | runs the pin check and the sweep on relevant PRs, and the build matrix on merges, weekly and on demand (at a release with the `version` input) |

An older revision of this recipe is also in Spack's builtin repository
(`spack/spack-packages`). This one is the maintained one and is used instead
of it once registered.
