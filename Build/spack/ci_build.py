"""Build one locked Spack environment and retain diagnostics for hosted CI.

The workflow calls prepare, install and collect in separate steps so collection
still runs after a failed configure, build or test. No build environment dumps or
credential-bearing Spack mirror configuration are copied into artifacts.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


def command(*args, capture=False):
    return subprocess.run(args, check=True, text=True, capture_output=capture).stdout


def paths():
    workspace = Path(os.environ["GITHUB_WORKSPACE"])
    return workspace, workspace / "spack-ci-env", workspace / "spack-ci-artifacts"


def prepare():
    workspace, env, artifacts = paths()
    artifacts.mkdir(exist_ok=True)
    spec = os.environ["SPEC"]
    mode = os.environ.get("BUILD_MODE", "install")
    metadata = {
        "requested_spec": spec,
        "build_mode": mode,
        "run_tests": os.environ.get("RUN_TESTS", "false"),
        "recipe_commit": command("git", "rev-parse", "HEAD", capture=True).strip(),
        "spack_commit": command(
            "git", "-C", str(workspace / "spack_install_dir"), "rev-parse", "HEAD", capture=True
        ).strip(),
        "packages_ref": os.environ["PACKAGES_REF"],
        "packages_commit": command(
            "git", "-C", str(workspace / "spack-packages"), "rev-parse", "HEAD", capture=True
        ).strip(),
        "runner_arch": os.environ.get("RUNNER_ARCH", ""),
        "runner_os": os.environ.get("RUNNER_OS", ""),
    }
    (artifacts / "request.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # These paths are within this ephemeral runner workspace on Linux and macOS.
    # Do not copy the general Spack config: it contains the registry credentials.
    command("spack", "config", "add", f"config:build_stage:[{workspace / 'spack-ci-stage'}]")
    command("spack", "config", "add", f"config:test_stage:{workspace / 'spack-ci-tests'}")
    if mode == "reuse":
        # Seed an incompatible root before solving the requested double/MPI build.
        # Its dependencies can be reused; the root and AMReX precision must not be.
        seed = "erf@26.09 ~mpi +particles precision=single target=x86_64"
        command("spack", "install", "--fail-fast", seed)
        (artifacts / "seed.json").write_text(
            command("spack", "find", "--json", seed, capture=True)
        )
    command("spack", "env", "create", "--without-view", "-d", str(env))
    command("spack", "-e", str(env), "add", spec)
    if mode == "develop":
        command(
            "spack", "-e", str(env), "develop", "--no-clone", "-p", str(workspace), "erf@develop"
        )
    command("spack", "-e", str(env), "concretize")
    command("spack", "-e", str(env), "spec", "-I", "-N")


def install():
    _, env, _ = paths()
    spec = os.environ["SPEC"]
    args = [
        "spack",
        "-e",
        str(env),
        "install",
        "--verbose",
        "--fail-fast",
        "--keep-stage",
        "--use-buildcache=package:never,dependencies:auto",
    ]
    gpu = any(backend in spec for backend in ("+cuda", "+rocm", "+sycl"))
    if os.environ.get("RUN_TESTS") == "true" and "+tests" in spec and not gpu:
        args.append("--test=root")
    command(*args)


def wanted(path):
    """Allow only build/test diagnostics, never an environment or config dump."""
    return (
        path.name
        in {
            "spack-build-out.txt",
            "spack-build-01-cmake-out.txt",
            "CMakeCache.txt",
            "CMakeConfigureLog.yaml",
            "CMakeError.log",
            "CMakeOutput.log",
            "LastTest.log",
            "LastTestsFailed.log",
            "CTestCostData.txt",
            "Test.xml",
            "results.txt",
            "test_suite.lock",
        }
        or path.name.startswith("Backtrace.")
        or path.name.endswith("-test-out.txt")
    )


def copy_diagnostics(root, destination):
    """Keep paths for identically named logs; skip links and binary/large files."""
    if not root.is_dir():
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in {".git", "spack_install_dir", "spack-ci-artifacts"}]
        for name in files:
            path = Path(directory) / name
            if not wanted(path) or path.is_symlink() or path.stat().st_size > 20 * 1024 * 1024:
                continue
            target = destination / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            if name == "LastTestsFailed.log" or name.startswith("Backtrace."):
                print(
                    f"--- {path.relative_to(root)} ---\n{path.read_text(errors='replace')[:12000]}"
                )


def collect():
    workspace, env, artifacts = paths()
    artifacts.mkdir(exist_ok=True)
    for name in ("spack.yaml", "spack.lock"):
        if (env / name).is_file():
            shutil.copyfile(env / name, artifacts / name)
    for name in ("spack-ci-stage", "spack-ci-tests"):
        copy_diagnostics(workspace / name, artifacts / name)
    if os.environ.get("BUILD_MODE") == "develop":
        for root in workspace.glob("spack-build-*"):
            copy_diagnostics(root, artifacts / root.name)
    try:
        installed = command("spack", "-e", str(env), "find", "--json", capture=True)
        (artifacts / "installed.json").write_text(installed)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("No installed environment available; keeping earlier diagnostics.")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as stream:
            stream.write(
                f"### Spack validation\n\nSpec: `{os.environ['SPEC']}`\n\n"
                f"Mode: `{os.environ.get('BUILD_MODE', 'install')}`; "
                f"CTest requested: `{os.environ.get('RUN_TESTS', 'false')}`.\n\n"
                "The job result includes build and test failures. Download the diagnostics "
                "artifact for the lockfile, source/tool revisions, CMake/CTest logs and "
                "AMReX backtraces when produced. GPU jobs are compile-only.\n"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "install", "collect"))
    args = parser.parse_args()
    {"prepare": prepare, "install": install, "collect": collect}[args.phase]()


if __name__ == "__main__":
    main()
