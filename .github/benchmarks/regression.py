"""Fork-only serial/concurrent comparison; not part of the production PR."""
import collections
import json
import os
from pathlib import Path
import re
import shutil
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path.cwd()
BUILD = ROOT / 'build'
OUT = ROOT / 'regression-evidence'
OUT.mkdir(exist_ok=True)
CORES = int(subprocess.check_output(['nproc'], text=True))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')


def inventory():
    result = json.loads(subprocess.check_output(
        ['ctest', '-L', 'regression', '--show-only=json-v1'], cwd=BUILD, text=True))
    save('inventory.json', result)
    return result['tests']


def audit():
    tests = inventory()
    assert len(tests) > 300, 'Unexpected regression selection'
    dirs = {}
    counts = collections.Counter()
    exceptions = {'CloudChamber_Bulk_FixedDtGuard', 'SHOC_Unstable_Cloud_SatAdj_vs_NoCond'}
    dependencies = {}
    for test in tests:
        name = test['name']
        props = {p['name']: p['value'] for p in test['properties']}
        if props.get('DISABLED'):
            continue
        count = int(props.get('PROCESSORS', 1))
        assert 1 <= count <= CORES, (name, count, CORES)
        counts[count] += 1
        command = test['command']
        rank_args = [match[1] for arg in command
                     if (match := re.fullmatch(r'-D(?:NRANKS|CHK_NRANKS|RESTART_NRANKS)=(\d+)', arg))]
        tokens = shlex.split(command[-1]) if command[:2] == ['/usr/bin/sh', '-c'] else command
        for index, token in enumerate(tokens[:-2]):
            if Path(token).name in ('mpiexec', 'mpirun') and tokens[index + 1] == '-n':
                rank_args.append(tokens[index + 2])
        assert all(int(n) <= count for n in rank_args), (name, count, rank_args)
        work = props.get('WORKING_DIRECTORY', '')
        for arg in command:
            if arg.startswith('-DWORKING_DIRECTORY='):
                work = arg.split('=', 1)[1]
        if name not in exceptions:
            path = Path(work).resolve()
            assert path.is_relative_to(BUILD / 'Tests/test_files'), (name, work)
            assert path not in dirs, (name, dirs.get(path), work)
            assert not any(path in p.parents or p in path.parents for p in dirs), (name, work)
            dirs[path] = name
        dependencies[name] = props.get('DEPENDS', [])
    assert set(dependencies['SHOC_Unstable_Cloud_SatAdj_vs_NoCond']) == {
        'SHOC_Unstable_Cloud_SatAdj_Property', 'SHOC_Unstable_Cloud_NoCond_Property'}
    save('audit.json', {'registered': len(tests), 'enabled': sum(counts.values()),
        'processor_counts': dict(counts), 'cpu_budget': CORES,
        'unique_output_directories': len(dirs), 'reviewed_exceptions': sorted(exceptions),
        'dependencies': {k: v for k, v in dependencies.items() if v}})
    print((OUT / 'audit.json').read_text(), flush=True)


def prepare():
    audit()
    off = json.loads((OUT / 'particles-off-inventory.json').read_text())
    on_names = {t['name'] for t in inventory()}
    off_names = {t['name'] for t in off['tests']}
    save('coverage.json', {'particles_off': len(off_names), 'particles_on': len(on_names),
        'off_only': sorted(off_names - on_names), 'on_only': sorted(on_names - off_names)})
    assert off_names <= on_names, 'Particles OFF has tests absent from ON validation'
    shutil.copytree(BUILD / 'Tests/test_files', OUT / 'pristine-test-files')
    save('environment.json', {
        'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'nproc': CORES, 'cpu': subprocess.check_output(['lscpu'], text=True),
        'memory': Path('/proc/meminfo').read_text(),
        'mpi': subprocess.check_output(['mpiexec', '--version'], text=True),
        'cmake': subprocess.check_output(['cmake', '--version'], text=True),
        'ctest_parallel_level': os.environ.get('CTEST_PARALLEL_LEVEL'),
    })
    shutil.copyfile(BUILD / 'CMakeCache.txt', OUT / 'CMakeCache.txt')


def run_trial(name):
    # Recreate the configured inputs at the same absolute paths for every trial.
    # Also discard CTest's learned costs, so order is not biased by the prior run.
    shutil.rmtree(BUILD / 'Tests/test_files')
    shutil.copytree(OUT / 'pristine-test-files', BUILD / 'Tests/test_files')
    for path in (BUILD / 'Testing/Temporary').glob('*'):
        if path.is_file():
            path.unlink()
    args = ['ctest', '-L', 'regression', '-VV', '--output-junit', str(OUT / (name + '.xml'))]
    if name != 'serial':
        args.extend(['--parallel', str(CORES)])
    start = time.monotonic()
    min_available = None
    with (OUT / (name + '.log')).open('w') as log:
        process = subprocess.Popen(args, cwd=BUILD, stdout=log, stderr=subprocess.STDOUT)
        next_update = start
        while process.poll() is None:
            mem = int(re.search(r'MemAvailable:\s+(\d+)', Path('/proc/meminfo').read_text())[1])
            min_available = mem if min_available is None else min(min_available, mem)
            now = time.monotonic()
            if now >= next_update:
                print(f'{name}: {now - start:.0f}s elapsed; {mem // 1024} MiB available', flush=True)
                next_update = now + 60
            time.sleep(1)
    results = {}
    for case in ET.parse(OUT / (name + '.xml')).getroot().iter('testcase'):
        status = 'passed'
        if case.find('failure') is not None or case.find('error') is not None:
            status = 'failed'
        elif case.get('status') == 'disabled':
            status = 'disabled'
        elif case.find('skipped') is not None:
            status = 'skipped'
        assert case.attrib['name'] not in results
        results[case.attrib['name']] = {'status': status, 'seconds': float(case.get('time', 0))}
    measurement = {'name': name, 'seconds': time.monotonic() - start,
        'returncode': process.returncode, 'command': args, 'min_available_kib': min_available,
        'results': results}
    save(name + '.json', measurement)
    print(json.dumps({k: v for k, v in measurement.items() if k != 'results'}, indent=2), flush=True)
    print('Failures:', [k for k, v in results.items() if v['status'] == 'failed'], flush=True)
    evidence = OUT / name
    evidence.mkdir()
    for path in (BUILD / 'Tests').rglob('*'):
        if path.is_file() and (path.suffix == '.log' or path.name.startswith('Backtrace.')):
            target = evidence / path.relative_to(BUILD)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    # Return zero here so baseline failures cannot suppress the comparisons.
    # The final comparison step reports both equivalence and any suite failures.


def compare():
    measurements = [json.loads((OUT / (name + '.json')).read_text())
                    for name in ('serial', 'parallel-1', 'parallel-2')]
    expected = {t['name'] for t in inventory()}
    baseline = {k: v['status'] for k, v in measurements[0]['results'].items()}
    summary = []
    equivalent = True
    for measurement in measurements:
        statuses = {k: v['status'] for k, v in measurement['results'].items()}
        differences = {k: [baseline.get(k), statuses.get(k)]
                       for k in baseline.keys() | statuses.keys() if baseline.get(k) != statuses.get(k)}
        equivalent &= not differences and set(statuses) == expected
        summary.append({'trial': measurement['name'], 'seconds': measurement['seconds'],
            'returncode': measurement['returncode'], 'counts': dict(collections.Counter(statuses.values())),
            'differences_from_serial': differences, 'complete_selection': set(statuses) == expected,
            'min_available_mib': measurement['min_available_kib'] / 1024})
    save('comparison.json', {'equivalent': equivalent, 'trials': summary})
    print((OUT / 'comparison.json').read_text())
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as stream:
        stream.write('```json\n' + (OUT / 'comparison.json').read_text() + '```\n')
    assert equivalent, 'Selected tests or outcomes differ'
    assert all(m['returncode'] == 0 for m in measurements), 'Matching suite failures; inspect retained logs'


if sys.argv[1] == 'prepare':
    prepare()
elif sys.argv[1] == 'compare':
    compare()
else:
    run_trial(sys.argv[1])
