"""Dispatch routing, failure propagation and useful CI artifacts."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ci_build
import ci_matrix


class RequestTests(unittest.TestCase):
    def test_darwin_spec_reaches_macos(self):
        spec = "erf@develop +tests platform=darwin"
        result = ci_matrix.matrix(spec=spec)
        self.assertEqual(result["macos"], [spec])
        self.assertEqual(result["install"], [])

    def test_gpu_routing(self):
        for backend, job in (("cuda", "install"), ("rocm", "rocm"), ("sycl", "sycl")):
            result = ci_matrix.matrix(spec=f"erf@26.09 +{backend}")
            self.assertEqual([name for name, specs in result.items() if specs], [job])

    def test_requests_cannot_silently_skip_validation(self):
        invalid = [
            ("+tests", "ubuntu-22.04", "install", True),
            ("erf@26.09 +cuda +tests", "ubuntu-22.04", "install", True),
            ("erf@26.09 +mpi", "ubuntu-22.04", "install", True),
            ("erf@26.09 +rocm", "ubuntu-24.04-arm", "install", False),
            ("", "ubuntu-24.04-arm", "install", False),
            ("", "ubuntu-22.04", "develop", False),
            ("erf@26.09", "ubuntu-22.04", "develop", False),
            ("erf@develop +rocm", "ubuntu-22.04", "develop", False),
            ("erf@develop +sycl", "ubuntu-22.04", "develop", False),
            ("erf@develop platform=darwin", "ubuntu-22.04", "develop", False),
            ("erf@26.09 +cuda", "ubuntu-22.04", "reuse", False),
            ("erf@26.09 +tests", "macos-15", "install", True),
            ("erf@26.09 +cuda platform=darwin", "macos-15", "install", False),
        ]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                ci_matrix.validate_request(*request)

    def test_valid_cpu_development_request(self):
        ci_matrix.validate_request("erf@develop +tests", "ubuntu-22.04", "develop", True)

    def test_valid_cuda_development_build(self):
        ci_matrix.validate_request("erf@develop +cuda cuda_arch=80", "ubuntu-22.04", "develop", False)

    def test_explicit_macos_runner(self):
        ci_matrix.validate_request("erf@26.09 +tests platform=darwin", "macos-15", "install", True)


class BuildTests(unittest.TestCase):
    def test_incompatible_spec_formats_use_separate_caches(self):
        urls = []
        for version in ("5", "6"):
            with (
                patch.dict(os.environ, {"GITHUB_REPOSITORY_OWNER": "example"}),
                patch.object(ci_build, "command", side_effect=[version + "\n", None]) as run,
            ):
                ci_build.configure_cache()
                urls.append(run.call_args.args[-1])
                self.assertIn("GHCR_TOKEN", run.call_args.args)
        self.assertEqual(urls, ["oci://ghcr.io/example/spack-buildcache-v5",
                                "oci://ghcr.io/example/spack-buildcache-v6"])

    def test_failed_build_or_ctest_is_not_hidden(self):
        with patch.dict(
            os.environ, {"GITHUB_WORKSPACE": "/tmp/ci", "SPEC": "erf +tests", "RUN_TESTS": "true"}
        ):
            with patch.object(
                ci_build, "command", side_effect=subprocess.CalledProcessError(1, "spack")
            ) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    ci_build.install()
                self.assertIn("--test=root", run.call_args.args)

    def test_full_matrix_never_runs_gpu_executables(self):
        for backend in ("cuda", "rocm", "sycl"):
            with (
                self.subTest(backend=backend),
                patch.dict(
                    os.environ,
                    {
                        "GITHUB_WORKSPACE": "/tmp/ci",
                        "SPEC": f"erf +tests +{backend}",
                        "RUN_TESTS": "true",
                    },
                ),
                patch.object(ci_build, "command") as run,
            ):
                ci_build.install()
                self.assertNotIn("--test=root", run.call_args.args)

    def test_backtraces_and_failures_survive_without_environment_dumps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "stage"
            output = Path(temp) / "artifacts"
            root.mkdir()
            for relative in (
                "build/Testing/Temporary/LastTestsFailed.log",
                "build/Testing/Temporary/LastTest.log.tmp",
                "spack-stage-erf-build.log",
                "build/Tests/Backtrace.0",
                "spack-build-out.txt",
                "spack-build-env.txt",
                "mirrors.yaml",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(relative)
            (root / "Backtrace.1").symlink_to(root / "spack-build-env.txt")
            ci_build.copy_diagnostics(root, output)
            self.assertEqual(
                {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()},
                {
                    "build/Testing/Temporary/LastTestsFailed.log",
                    "build/Testing/Temporary/LastTest.log.tmp",
                    "spack-stage-erf-build.log",
                    "build/Tests/Backtrace.0",
                    "spack-build-out.txt",
                },
            )


if __name__ == "__main__":
    unittest.main()
