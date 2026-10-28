#!/usr/bin/env python3
"""Evidence for the isolated fork-only full-build cache benchmark."""

import json
import os
from pathlib import Path
import subprocess
import sys

OUT = Path(os.environ["BENCHMARK_METRICS"])
CACHE = Path.home() / ".cache/ccache"


def capture(*args):
    return subprocess.check_output(args, text=True).strip()


def write(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not CACHE.exists() or not any(CACHE.rglob("*")), "runner cache must start empty"
    source = Path("Source/ERF.cpp")
    if os.environ["BENCHMARK_PHASE"] == "changed":
        content = source.read_text()
        assert content.count("Coarse STEP ") == 2
        source.write_text(content.replace("Coarse STEP ", "Coarse STEP [cache-benchmark] "))
    write("environment.json", {
        "configuration": os.environ["BENCHMARK_CONFIGURATION"],
        "phase": os.environ["BENCHMARK_PHASE"],
        "commit": capture("git", "rev-parse", "HEAD"),
        "submodules": capture("git", "submodule", "status"),
        "source_diff": capture("git", "diff", "--", "Source/ERF.cpp"),
        "ccache": capture("ccache", "--version"),
        "compiler": capture("c++", "--version"),
        "cmake": capture("cmake", "--version"),
        "cpu": capture("lscpu"),
        "memory": capture("free", "-b"),
        "image_os": os.environ.get("ImageOS"),
        "image_version": os.environ.get("ImageVersion"),
        "cache_config": capture("ccache", "--show-config"),
    })


def verify():
    ref = "ghcr.io/larenspear/erf-ccache:" + os.environ["CCACHE_TRUNK_KEY"]
    # The production action warns on transfer failures. Verify the actual pushed
    # image independently and anonymously before permitting the next phase.
    manifest = json.loads(capture("docker", "manifest", "inspect", ref))
    local_id = capture("docker", "image", "inspect", "--format", "{{.Id}}", ref)
    assert manifest["config"]["digest"] == local_id, "published image differs from local build"
    binaries = list(Path(os.environ["BENCHMARK_BUILD"]).rglob("erf_exec"))
    assert len(binaries) == 1, binaries
    marker = b"Coarse STEP [cache-benchmark] "
    assert (marker in binaries[0].read_bytes()) == (os.environ["BENCHMARK_PHASE"] == "changed")
    write("publication.json", {
        "ref": ref,
        "config_digest": local_id,
        "compressed_layer_bytes": sum(layer["size"] for layer in manifest["layers"]),
        "manifest": manifest,
        "executable": str(binaries[0]),
        "changed_source_marker_verified": True,
    })
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        stream.write("ready=true\n")


def summarize():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {}
    stats_file = OUT / "ccache-after.txt"
    if stats_file.exists():
        for line in stats_file.read_text().splitlines():
            key, value = line.split()
            stats[key] = int(value)
    timings = {}
    for start in OUT.glob("*-start-ns"):
        end = OUT / start.name.replace("-start-ns", "-end-ns")
        if end.exists():
            timings[start.name.removesuffix("-start-ns")] = (int(end.read_text()) - int(start.read_text())) / 1e9
    hits = stats.get("direct_cache_hit", 0) + stats.get("preprocessed_cache_hit", 0)
    calls = hits + stats.get("cache_miss", 0)
    result = {
        "configuration": os.environ["BENCHMARK_CONFIGURATION"],
        "phase": os.environ["BENCHMARK_PHASE"],
        "restore_hit": os.environ.get("BENCHMARK_RESTORE_HIT"),
        "build_outcome": os.environ.get("BENCHMARK_BUILD_OUTCOME"),
        "publication_verified": (OUT / "publication.json").exists(),
        "timings_seconds": timings,
        "ccache_stats": stats,
        "ccache_hit_percent": 100 * hits / calls if calls else None,
        "cache_file_bytes": sum(p.stat().st_size for p in CACHE.rglob("*") if p.is_file()),
    }
    write("metrics.json", result)
    print(json.dumps(result, indent=2))
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as stream:
        stream.write("```json\n" + json.dumps(result, indent=2) + "\n```\n")


{"prepare": prepare, "verify": verify, "summarize": summarize}[sys.argv[1]]()
