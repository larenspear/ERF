# ccache trunks on GHCR

This action stores one rolling ccache snapshot per build configuration at
`ghcr.io/<repository-owner>/erf-ccache:<sanitized-key>`. All eight compiler-cache
workflows restore from GHCR. Only default-branch runs save; pull requests only read.
Saves happen before tests so an unrelated test failure does not freeze the cache.
These are compiler objects for incremental builds, not runnable ERF installations.

During migration, each workflow falls back to its existing Actions cache on a GHCR
miss or outage. This fallback is read-only: new snapshots go only to GHCR. The old
Actions entries eventually expire under GitHub's cache policy. A first run with
neither cache available builds normally and seeds its GHCR trunk on the default branch.

## Configuration names

`CCACHE_TRUNK_KEY` supplies the same key to restore and save, independently of workflow
and job names. Keys describe the runner environment, architecture, compiler/MPI or GPU
profile, precision, and relevant options. The current development matrix has 21 trunks:

| Workflow | Profile | Trunks |
| --- | --- | ---: |
| `ci.yml` | Ubuntu 22.04, x64, GCC, MPICH, double, particles off/on | 2 |
| `gcc.yml` | Ubuntu 22.04, x64, GCC, Open MPI, precision/particles | 3 |
| `gcc-rrtmgp.yml` | Ubuntu 22.04, x64, GCC, Spack/Open MPI, Kokkos, NetCDF/RRTMGP | 3 |
| `cuda-ci.yml` | Ubuntu 22.04, x64, CUDA 12.2, target 8.0, Open MPI, precision | 2 |
| `hip.yml` | Ubuntu 22.04, x64, ROCm 6.3.2, gfx908, Open MPI, precision | 2 |
| `sycl.yml` | Ubuntu 24.04, x64, oneAPI/SYCL, Intel MPI, precision | 2 |
| `macos.yml` | macOS latest, actual runner architecture, Apple Clang, Open MPI, precision | 2 |
| `windows.yml` | Windows latest, actual runner architecture, MSVC, MPI/precision/particles | 5 |

For example, `ubuntu-22.04-x64-gcc-mpich-double-particles-off` and
`ubuntu-22.04-x64-gcc-mpich-double-particles-on` retain the Linux pilot's names.
Matching mesh/particle precision and the other fixed build options are defined by
those workflows. Include new independent matrix dimensions in the key if coverage
expands. Update explicit version/target names when dependency profiles change.

`latest` describes the selected native runner image, not a pinned OS or toolchain.
ccache still checks source, compiler, headers, and flags; a toolchain or large header
update can cause a cold build even when the trunk downloads successfully. GHCR improves
cache availability and retention, not ccache's ability to reuse incompatible objects.

## Transports and failure handling

With `transport: auto`, Linux uses the existing Docker scratch-image transport;
macOS and Windows use an OCI artifact containing `ccache.tar` through ORAS. Tags must
not be shared between transports: their formats differ. The explicit `docker` and
`oras` settings are available for transport checks.

The native helper downloads ORAS 1.3.0 from its official GitHub release and verifies a
platform-specific SHA256 before execution. It needs Python 3, curl, and gh, supplied by
GitHub's hosted runners. Linux Docker also needs Bash and GNU coreutils. No Python
packages or Docker daemon are required on native runners.

Login and transfer commands get two attempts. `transfer-timeout` defaults to 180 seconds
per transfer command (maximum 600), allowing for the existing CUDA cache limit of
1100 MB. Login has a 30-second limit and pruning has a 60-second overall budget.
Registry, download, and filesystem failures warn and let compilation/tests continue;
invalid action inputs fail. `hit` is true only after a complete restore.

Native command failures report the operation, exit status or timeout, and up to
1024 characters of diagnostic output. Tokens, authentication fields, URL credentials
and query strings are redacted before truncation. Explicit missing-manifest responses
are normal cache misses; permission errors and ambiguous failures remain warnings.
Failed pruning includes the command's diagnostic details and remains nonfatal.

Both transports stage beside the destination before replacing it, keeping the previous
cache for rollback. The path must be a dedicated cache directory, not home or the root.
The native helper fetches an immutable blob digest from the manifest, verifies its
size/checksum, and rejects archive traversal, links, and special files. It preserves
file timestamps so CI's existing stale-object eviction remains effective. Credentials
use a private temporary configuration removed after the transfer.

## Access and retention

Publishing jobs need `contents: read` and `packages: write`. Source labels/annotations
link a new package to its repository. If it already exists, grant the publishing
repository access; deleting superseded versions also requires package admin access.
Private packages need an authorized token. Public visibility permits anonymous reads.
Verify package access from fork-origin PRs when adopting this upstream: those tokens
are read-only, and a denied restore warns and falls back to the old cache or a cold build.

After a successful push, `prune.py` keeps every named configuration tag and deletes
superseded untagged versions. Failed pruning is retried after a later save. There is no
age-based expiration of the active trunks. Renaming/retiring configurations requires
separate cleanup of their named tags. Concurrent development runs can publish in
completion order; revision metadata records the producer but does not enforce ordering.
All snapshots are transferred as complete archives/images, not per-object registry diffs.

## Local checks

Run all fault-injection tests on Linux:

```sh
python3 -m unittest discover -s .github/actions/ccache-trunk -p 'test_*.py' -v
```

Run only the portable tests on macOS or Windows (use `python` on Windows):

```sh
python3 -m unittest discover -s .github/actions/ccache-trunk -p 'test_native*.py' -v
```

The tests need no network or credentials. They cover retries/timeouts, failed or
partial restores, rollback, archive integrity/safety, timestamp preservation, failed
saves, pruning retention/races, credential cleanup, redacted diagnostics, and invalid
inputs. The full suite runs in the existing `shellcheck` job in `lint.yml` for pull
requests and development pushes that change `.github/**`, and on manual dispatch.
It uses the runner's Python
standard library, Bash, and GNU coreutils, without compiling ERF or accessing GHCR.
