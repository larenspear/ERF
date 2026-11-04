#!/usr/bin/env python3
"""Compare real cache entries without rebuilding ERF or publishing any cache.

This measures ccache recompression and decode/rewrite, not build speed or network
transfer. Each timed trial starts with a fresh copy of the same uncompressed
entries. The source cache is read-only. Cache-hit behavior is not benchmarked.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import statistics
import subprocess
import tempfile
import time


def run(command, **kwargs):
    return subprocess.run(command, check=True, capture_output=True, text=True,
                          timeout=120, **kwargs).stdout


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def recompress(ccache, directory, level, log):
    # No source-cache configuration or ambient ccache setting can alter a trial.
    env = {k: v for k, v in os.environ.items() if not k.startswith('CCACHE_')}
    env.update(CCACHE_DIR=str(directory), CCACHE_CONFIGPATH=os.devnull)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    output = run([ccache, '--recompress', str(level), '--recompress-threads', '1'],
                 env=env)
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    log.write_text(output)
    return {'wall_seconds': wall,
            'cpu_seconds': after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime}


def compare(args):
    ccache = shutil.which(args.ccache)
    if not ccache:
        raise RuntimeError('ccache executable not found')
    source = args.cache.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    budget = args.max_mib * 1024 * 1024
    paths = [p for p in source.rglob('*R') if p.is_file() and not p.is_symlink()]
    # Hash order avoids lexicographic or file-size selection bias and is stable.
    paths.sort(key=lambda p: hashlib.sha256(str(p.relative_to(source)).encode()).digest())
    selected = []
    uncompressed_size = 0
    for path in paths:
        info = run([ccache, '--inspect', str(path)])
        if 'Self-contained: yes' not in info or 'Entry type: 0 (result)' not in info:
            raise RuntimeError(f'unsupported cache entry: {path.relative_to(source)}')
        match = re.search(r'^Entry size: (\d+)$', info, re.MULTILINE)
        if not match:
            raise RuntimeError('could not read cache entry size')
        size = int(match[1])
        if uncompressed_size + size > budget:
            continue
        selected.append({'path': str(path.relative_to(source)), 'entry_size': size,
                         'source_bytes': path.stat().st_size, 'source_sha256': digest(path)})
        uncompressed_size += size
        if uncompressed_size >= budget * 0.95:
            break
    if not selected:
        raise RuntimeError('no self-contained result entries fit the sample budget')

    report = {'label': args.label, 'ccache_version': run([ccache, '--version']).splitlines()[0],
              'platform': platform.platform(), 'cpu_count': os.cpu_count(), 'threads': 1,
              'available_result_entries': len(paths), 'sample_entries': selected,
              'sample_uncompressed_entry_bytes': uncompressed_size,
              'max_sample_mib': args.max_mib, 'trials': [],
              'method': 'Uncompressed baseline; fresh copies; untimed warmup pair; four alternating-order pairs; round-trip SHA256 verification. Compression/decode+rewrite only, not build or transfer times.'}
    if args.image_ref:
        image = json.loads(run(['docker', 'image', 'inspect', args.image_ref]))[0]
        report['source_image'] = {'ref': args.image_ref, 'digests': image['RepoDigests'],
                                  'labels': image['Config'].get('Labels', {})}
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'{args.label}: {len(selected)}/{len(paths)} results, '
          f'{uncompressed_size / 1024**2:.1f} MiB before compression', flush=True)

    with tempfile.TemporaryDirectory(prefix='ccache-compare-', dir=source.parent) as temporary:
        root = Path(temporary)
        seed = root / 'seed'
        for item in selected:
            destination = seed / item['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / item['path'], destination)
        recompress(ccache, seed, 'uncompressed', output / 'normalize.txt')
        baseline = {item['path']: digest(seed / item['path']) for item in selected}
        (output / 'baseline-sha256.json').write_text(json.dumps(baseline, indent=2) + '\n')
        for pair in range(5):
            order = (1, 10) if pair % 2 == 0 else (10, 1)
            for level in order:
                trial = root / 'trial'
                shutil.copytree(seed, trial)
                prefix = f'{"warmup" if pair == 0 else f"pair-{pair}"}-level-{level}'
                compression = recompress(ccache, trial, level, output / f'{prefix}-compress.txt')
                size = sum((trial / item['path']).stat().st_size for item in selected)
                # Inspect one result outside the timed interval to verify the level.
                info = run([ccache, '--inspect', str(trial / selected[0]['path'])])
                if not re.search(rf'^Compression level: {level}$', info, re.MULTILINE):
                    raise RuntimeError('requested compression level was not applied')
                decode = recompress(ccache, trial, 'uncompressed', output / f'{prefix}-decode.txt')
                if {item['path']: digest(trial / item['path']) for item in selected} != baseline:
                    raise RuntimeError('decompressed cache contents changed')
                if pair:
                    report['trials'].append({'pair': pair, 'order': list(order), 'level': level,
                                             'compression': compression, 'compressed_bytes': size,
                                             'decode_and_rewrite': decode, 'round_trip_verified': True})
                    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
                print(f'{args.label} {prefix}: compress={compression["wall_seconds"]:.3f}s, '
                      f'size={size / 1024**2:.2f} MiB; identical round trip', flush=True)
                shutil.rmtree(trial)

    for item in selected:
        if digest(source / item['path']) != item['source_sha256']:
            raise RuntimeError('source cache was modified')
    report['source_unchanged'] = True
    report['summary'] = {}
    for level in (1, 10):
        trials = [t for t in report['trials'] if t['level'] == level]
        sizes = {t['compressed_bytes'] for t in trials}
        if len(sizes) != 1:
            raise RuntimeError('compressed size varied between identical trials')
        walls = [t['compression']['wall_seconds'] for t in trials]
        report['summary'][str(level)] = {
            'compression_median_seconds': statistics.median(walls),
            'compression_min_seconds': min(walls), 'compression_max_seconds': max(walls),
            'compression_median_cpu_seconds': statistics.median(t['compression']['cpu_seconds'] for t in trials),
            'compressed_bytes': sizes.pop(),
            'decode_rewrite_median_seconds': statistics.median(t['decode_and_rewrite']['wall_seconds'] for t in trials)}
    lines = [f'### {args.label}: ccache compression comparison', '',
             f'{len(selected)} of {len(paths)} result entries; {uncompressed_size / 1024**2:.1f} MiB uncompressed. '
             'Four measured pairs on one runner, one thread, alternating order.', '',
             '| Level | Median compression | Range | Stored size | Median decode + rewrite |',
             '| --- | ---: | ---: | ---: | ---: |']
    for level, row in report['summary'].items():
        lines.append(f'| {level} | {row["compression_median_seconds"]:.3f} s | '
                     f'{row["compression_min_seconds"]:.3f}–{row["compression_max_seconds"]:.3f} s | '
                     f'{row["compressed_bytes"] / 1024**2:.2f} MiB | '
                     f'{row["decode_rewrite_median_seconds"]:.3f} s |')
    lines += ['', 'All decompressed entries match the baseline byte for byte; source entries are unchanged. '
              'These are compression measurements, not measured build-time or network-transfer savings.', '']
    summary = '\n'.join(lines)
    (output / 'summary.md').write_text(summary)
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(summary)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
            stream.write(summary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--ccache', default='ccache')
    parser.add_argument('--max-mib', type=int, default=256)
    parser.add_argument('--image-ref')
    arguments = parser.parse_args()
    if arguments.max_mib <= 0:
        parser.error('--max-mib must be positive')
    compare(arguments)
