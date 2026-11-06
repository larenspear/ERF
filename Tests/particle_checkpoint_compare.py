"""Read this serial native-double AMReX 2.1 particle checkpoint by stable ID.

Format follows pinned AMReX_ParticleIO.H::ReadParticles: per-grid integer
records followed by real records. This diagnostic supports native little-
endian double files and does not claim a general checkpoint reader.
"""
from pathlib import Path
import json
import math
import struct
import sys


def read(directory):
    directory = Path(directory)
    lines = iter((directory / "Header").read_text().splitlines())
    assert next(lines) == "Version_Two_Dot_One_double"
    ndim = int(next(lines))
    assert ndim == 3, "reader is restricted to the 3D runtime fixture"
    nreal = int(next(lines))
    real_names = [next(lines) for _ in range(nreal)]
    nint = int(next(lines))
    int_names = [next(lines) for _ in range(nint)]
    assert int(next(lines)) == 1
    total = int(next(lines))
    next_id = int(next(lines))
    finest = int(next(lines))
    grids = [int(next(lines)) for _ in range(finest + 1)]
    particles = {}
    for lev, ngrids in enumerate(grids):
        for grid in range(ngrids):
            number, count, offset = map(int, next(lines).split())
            if not count:
                continue
            data = (directory / f"Level_{lev}" / f"DATA_{number:05d}").read_bytes()
            ints = struct.unpack_from("<" + "i" * count * (2 + nint), data, offset)
            reals = struct.unpack_from("<" + "d" * count * (ndim + nreal), data, offset + count * (2 + nint) * 4)
            for p in range(count):
                iv = ints[p * (2 + nint):(p + 1) * (2 + nint)]
                rv = reals[p * (ndim + nreal):(p + 1) * (ndim + nreal)]
                assert all(math.isfinite(value) for value in rv), "nonfinite particle attribute"
                key = iv[:2]
                assert key not in particles
                particles[key] = (lev, grid, iv[2:], rv)
    assert len(particles) == total
    return {"real_names": ["x", "y", "z", *real_names], "int_names": int_names,
            "next_id": next_id, "particles": particles}


def compare(left, right):
    a, b = read(left), read(right)
    assert a["real_names"] == b["real_names"] and a["int_names"] == b["int_names"]
    assert a["particles"].keys() == b["particles"].keys()
    changed_real = {}
    changed_int = 0
    changed_location = 0
    for key, old in a["particles"].items():
        new = b["particles"][key]
        changed_location += old[:2] != new[:2]
        changed_int += old[2] != new[2]
        for name, x, y in zip(a["real_names"], old[3], new[3]):
            if x != y:
                summary = changed_real.setdefault(name, {"count": 0, "max_abs_diff": 0, "max_abs_reference": 0})
                summary["count"] += 1
                summary["max_abs_diff"] = max(summary["max_abs_diff"], abs(y-x))
                summary["max_abs_reference"] = max(summary["max_abs_reference"], abs(x))
    return {"particles": len(a["particles"]), "changed_level_or_grid": changed_location,
            "changed_integer_records": changed_int, "changed_real_fields": changed_real,
            "next_id_equal": a["next_id"] == b["next_id"]}


if __name__ == "__main__":
    print(json.dumps(compare(sys.argv[1], sys.argv[2]), indent=2))
