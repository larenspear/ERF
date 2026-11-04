# Fork-only ccache compression comparison

This branch temporarily replaces `gcc.yml` with a manual benchmark. It is not an
upstream PR candidate. Run `gcc.yml` on `ci-ccache-compression-compare` in
`larenspear/ERF`. The workflow uses one Ubuntu 22.04 runner with a 15-minute limit.

It restores the fork's GCC/MPICH and HIP/ROCm cache images, records their immutable
digests and producer labels, and selects up to 256 MiB of uncompressed result
entries from each. Selection uses a deterministic hash order. Oversized entries
are skipped; the artifact records selected paths, sizes, hashes and population
counts, so this sample must not be treated as every cache or build configuration.

Each trial starts with a fresh copy of the same uncompressed entries. After an
untimed warmup pair, four measured pairs alternate levels 1 and 10 on the same
runner. ccache 4.12.2, matching current ERF Linux CI, uses one recompression thread.
Measurements include elapsed time, CPU time, stored bytes and decode/rewrite time.
Every decompressed result must match the baseline byte for byte, and original
entries must remain unchanged. A cache miss fails the benchmark explicitly.

The workflow does not compile ERF or publish caches. It has read-only contents and
package permissions. It uploads only measurements and metadata, retained for 14
days, and writes a comparison table to the Actions job summary.

Recompression measures cache processing, including file I/O. It does not measure
compiler overhead, cache-hit latency, Docker packaging, network transfer or
end-to-end CI time. Larger compressed entries may make cache transfers slower.
Use the result to decide whether a full-build comparison is worthwhile.
