"""Small real compiler-cache round trip; validation branch only."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

cache = Path(os.environ['CCACHE_DIR'])
work = Path(os.environ['RUNNER_TEMP']) / 'compiler-smoke'
work.mkdir(exist_ok=True)
windows = os.environ['RUNNER_OS'] == 'Windows'
compiler = 'cl' if windows else ('clang++' if os.environ['RUNNER_OS'] == 'macOS' else 'g++')
source = work / 'smoke.cpp'
obj = work / ('smoke.obj' if windows else 'smoke.o')
exe = work / ('smoke.exe' if windows else 'smoke')

def run(*args):
    return subprocess.check_output(args, cwd=work, text=True, stderr=subprocess.STDOUT)

def compile_and_run(value):
    if windows:
        run('ccache', compiler, '/nologo', '/c', str(source), '/Fo' + str(obj))
        run(compiler, '/nologo', str(obj), '/Fe' + str(exe))
    else:
        run('ccache', compiler, '-c', str(source), '-o', str(obj))
        run(compiler, str(obj), '-o', str(exe))
    assert run(str(exe)).strip() == str(value)

def write_source(value):
    source.write_text(f'#include <stdio.h>\nint main() {{ printf("%d\\n", {value}); return 0; }}\n')

def stats():
    values = dict(line.split() for line in run('ccache', '--print-stats').splitlines())
    return {key: int(value) for key, value in values.items()}

mode = sys.argv[1]
if mode == 'populate':
    write_source(42)
    run('ccache', '-z')
    compile_and_run(42)
    assert stats()['cache_miss'] == 1
    print(run('ccache', '-s'))
elif mode == 'empty':
    shutil.rmtree(cache)
    obj.unlink()
    exe.unlink()
elif mode == 'verify':
    run('ccache', '-z')
    compile_and_run(42)
    warm = stats()
    assert warm['direct_cache_hit'] + warm['preprocessed_cache_hit'] == 1, warm
    assert warm['cache_miss'] == 0, warm
    write_source(43)
    compile_and_run(43)
    changed = stats()
    assert changed['cache_miss'] == 1, changed
    print(json.dumps({'warm': warm, 'changed_source': changed}, indent=2))
    print('PASS: GHCR cache hit; changed source rebuilt and produced the new output')
