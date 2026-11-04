"""Temporary fork-only comparison; not part of the proposed workflow change."""

import collections
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def inventory(build):
    return json.loads(subprocess.check_output(
        ["ctest", "-L", "regression", "--show-only=json-v1"], cwd=build, text=True
    ))


def links(root):
    return {str(p.relative_to(root)): os.readlink(p)
            for p in root.rglob("*") if p.is_symlink()}


def outcomes(path):
    result = {}
    for test in ET.parse(path).iter("testcase"):
        name = test.attrib["name"]
        if name in result:
            raise RuntimeError(f"Duplicate result: {name}")
        if test.find("failure") is not None or test.find("error") is not None:
            state = "failed"
        elif test.attrib.get("status") == "disabled":
            state = "disabled"
        elif test.find("skipped") is not None:
            state = "skipped"
        else:
            state = "passed"
        result[name] = state
    return result


def main():
    build, output = (Path(p).resolve() for p in sys.argv[1:3])
    slots = int(sys.argv[3])
    output.mkdir(parents=True, exist_ok=False)
    baseline = inventory(build)
    tests = baseline["tests"]
    if not tests:
        raise RuntimeError("No regression tests registered")
    save_json(output / "inventory.json", baseline)
    props = {t["name"]: {p["name"]: p["value"] for p in t["properties"]}
             for t in tests}
    save_json(output / "scheduling.json", {
        "slots": slots,
        "processors": dict(collections.Counter(p.get("PROCESSORS", 1) for p in props.values())),
        "working_directories": {n: p.get("WORKING_DIRECTORY") for n, p in props.items()},
        "environments": {n: p.get("ENVIRONMENT") for n, p in props.items()},
        "dependencies": {n: p["DEPENDS"] for n, p in props.items() if "DEPENDS" in p},
    })
    inputs = build / "Tests/test_files"
    snapshot = output / "pristine-inputs"
    shutil.copytree(inputs, snapshot, symlinks=True)
    expected_links = links(inputs)
    if links(snapshot) != expected_links:
        raise RuntimeError("Snapshot changed symbolic links")
    results = []
    # Bracket the serial run with two parallel runs to expose first-run effects.
    for label, parallel in [("parallel-1", True), ("serial", False), ("parallel-2", True)]:
        trial = output / label
        trial.mkdir()
        shutil.rmtree(inputs)
        shutil.copytree(snapshot, inputs, symlinks=True)
        if links(inputs) != expected_links:
            raise RuntimeError("Restoring inputs changed symbolic links")
        # Do not let CTest's learned durations from one trial reorder the next.
        if (build / "Testing").exists():
            shutil.rmtree(build / "Testing")
        current = inventory(build)
        save_json(trial / "inventory.json", current)
        if current["tests"] != tests:
            raise RuntimeError("Test selection or properties changed between trials")
        command = ["ctest", "-L", "regression"]
        if parallel:
            command += ["--parallel", str(slots)]
        command += ["-VV", "--output-junit", str(trial / "results.xml")]
        memory = []
        stop = threading.Event()

        def sample_memory():
            while not stop.is_set():
                if Path("/proc/meminfo").exists():
                    match = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
                    memory.append(int(match[1]))
                stop.wait(1)

        sampler = threading.Thread(target=sample_memory)
        sampler.start()
        print(f"BEGIN {label}: {' '.join(command)}", flush=True)
        start = time.monotonic()
        with (trial / "ctest.log").open("w") as log:
            process = subprocess.Popen(command, cwd=build, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            for line in process.stdout:
                log.write(line)
                if re.search(r"Test\s+#|tests passed|Total Test time", line):
                    print(line, end="", flush=True)
            code = process.wait()
        elapsed = time.monotonic() - start
        stop.set()
        sampler.join()
        state = outcomes(trial / "results.xml")
        if set(state) != {t["name"] for t in tests}:
            raise RuntimeError("JUnit results do not match selected tests")
        for path in (build / "Tests").rglob("*"):
            if path.is_file() and (path.suffix == ".log" or path.name.startswith("Backtrace.")):
                dest = trial / "logs" / path.relative_to(build / "Tests")
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
        if (build / "Testing").exists():
            shutil.copytree(build / "Testing", trial / "Testing", symlinks=True)
        result = {"trial": label, "command": command, "elapsed_seconds": elapsed,
                  "exit_code": code, "outcomes": state,
                  "counts": dict(collections.Counter(state.values())),
                  "minimum_available_memory_kib": min(memory) if memory else None}
        save_json(trial / "summary.json", result)
        results.append(result)
        print(f"END {label}: {elapsed:.2f}s, exit={code}, {result['counts']}", flush=True)
    same = all(r["outcomes"] == results[0]["outcomes"] for r in results)
    comparison = {"same_outcomes": same, "trials": results}
    save_json(output / "comparison.json", comparison)
    shutil.rmtree(snapshot)
    lines = ["| Trial | Seconds | Exit | Outcomes |", "| --- | ---: | ---: | --- |"]
    lines += [f"| {r['trial']} | {r['elapsed_seconds']:.2f} | {r['exit_code']} | {r['counts']} |"
              for r in results]
    lines += ["", f"Identical outcomes: {same}. Same build, restored inputs, {slots} parallel CPU slots."]
    summary = "\n".join(lines) + "\n"
    print(summary, flush=True)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as handle:
            handle.write(summary)
    # Equal failures are useful evidence, but must keep the workflow red.
    return int(not same or any(r["exit_code"] for r in results))


if __name__ == "__main__":
    sys.exit(main())
