#!/usr/bin/env python3
"""Emit build-only metadata as C++; shared by CMake and GNU Make."""

import argparse
from pathlib import Path
import re
import subprocess


def source_identity(directory):
    """Do not accidentally report a parent repository for an unpacked dependency."""
    unknown = {"revision": None, "tracked_dirty": None}
    if not directory or not (Path(directory) / ".git").exists():
        return unknown
    try:
        revision = subprocess.check_output(
            ["git", "-C", directory, "rev-parse", "--verify", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                directory,
                "diff",
                "--quiet",
                "--ignore-submodules=untracked",
                "HEAD",
                "--",
            ],
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
    except (OSError, subprocess.CalledProcessError):
        return unknown
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        return unknown
    return {
        "revision": revision,
        "tracked_dirty": bool(status) if status in (0, 1) else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--amrex", default="")
    parser.add_argument("--build-system", required=True, choices=("cmake", "gnu-make"))
    parser.add_argument(
        "--gpu-backend", default="none", choices=("none", "cuda", "hip", "sycl")
    )
    parser.add_argument("--gpu-targets", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
    targets = [t for t in re.split(r"[;,\s]+", args.gpu_targets) if t]
    metadata = {
        "build_system": args.build_system,
        "source": source_identity(args.source),
        "amrex_source": source_identity(args.amrex),
        # These are build-system settings, not device discovery or a cubin audit.
        "gpu_target_settings": targets
        if targets or args.gpu_backend == "none"
        else None,
    }
    code = emit_cpp(metadata)
    path = Path(args.output)
    if args.append:
        with path.open("a", encoding="utf-8") as stream:
            stream.write(code)
    elif not path.exists() or path.read_text(encoding="utf-8") != code:
        path.write_text(code, encoding="utf-8")


def emit_cpp(metadata):
    """Emit one set of typed fields, consumed by both JSON and YAML writers."""
    lines = [
        "\n#include <ERF_DescriptionWriter.H>",
        "void erf_write_build_metadata (erf_description::Writer& writer)",
        "{",
    ]

    def field(name, value):
        key = cpp_string(name)
        if isinstance(value, dict):
            lines.append(f"    writer.begin_object({key});")
            for child, item in value.items():
                field(child, item)
            lines.append("    writer.end_object();")
        elif isinstance(value, list):
            values = ", ".join(cpp_string(item) for item in value)
            lines.append(f"    writer.strings({key}, {{{values}}});")
        else:
            if value is None:
                literal = "nullptr"
            elif isinstance(value, bool):
                literal = "true" if value else "false"
            elif isinstance(value, str):
                literal = cpp_string(value)
            else:
                raise TypeError(f"unsupported build-metadata value: {type(value)}")
            lines.append(f"    writer.value({key}, {literal});")

    field("build", metadata)
    return "\n".join(lines) + "\n}\n"


def cpp_string(value):
    # Fixed-width octal escapes preserve UTF-8 bytes and cannot absorb a digit
    # from the following character, unlike a variable-width C++ hex escape.
    return (
        '"'
        + "".join(
            chr(byte) if 32 <= byte < 127 and byte not in (34, 92) else f"\\{byte:03o}"
            for byte in value.encode("utf-8")
        )
        + '"'
    )


if __name__ == "__main__":
    main()
