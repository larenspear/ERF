#!/usr/bin/env python3
"""Round-trip ERF descriptions with independent YAML and JSON Schema parsers.

Generator and executable checks use the Python standard library. Optional format
checks use PyYAML, jsonschema, and a C++17 compiler. ERF has no YAML dependency.
Pass --executable to check a real installed executable too.
"""

import argparse
import copy
import json
import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import sys
from unittest import mock
import unittest

try:
    import jsonschema
    import yaml
except ImportError:
    jsonschema = yaml = None

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "build_metadata", ROOT / "Tools/build_metadata.py"
)
build_metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_metadata)


def metadata():
    schema = json.loads(
        (ROOT / "Docs/schema/erf-build-description-v1.schema.json").read_text()
    )
    return {
        "schema_version": 1,
        "application": "ERF",
        "dimension": 3,
        "real_bits": 64,
        "gpu_backend": "none",
        "features": {
            feature: feature in ("mpi", "particles", "fft")
            for feature in schema["properties"]["features"]["required"]
        },
        "build": {
            "build_system": "cmake",
            "source": {"revision": "a" * 40, "tracked_dirty": True},
            "amrex_source": {"revision": None, "tracked_dirty": None},
            "gpu_target_settings": [],
        },
    }


EXECUTABLE = None
LAUNCHER = []


@unittest.skipUnless(
    yaml and jsonschema, "install PyYAML and jsonschema for format/schema tests"
)
class FormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="erf-description-formats-")
        cls.addClassCleanup(cls.directory.cleanup)
        work = Path(cls.directory.name)
        cls.special = [
            "80",
            "26.09",
            "yes",
            "null",
            "01",
            "1e3",
            "",
            'quote" slash\\\n\t\r\b\f\x01\x7f',
            "café \u0085 \u009f \u2028 \u2029 😀",
            ": # --- !!python/object &anchor *alias",
        ]
        cls.build = {
            "build_system": "cmake",
            "source": {"revision": "1" * 40, "tracked_dirty": False},
            "amrex_source": {"revision": None, "tracked_dirty": None},
            "gpu_target_settings": cls.special,
        }
        (work / "generated.cpp").write_text(build_metadata.emit_cpp(cls.build))
        (work / "main.cpp").write_text("""#include <ERF_DescriptionWriter.H>
#include <iostream>
#include <locale>
struct Grouped : std::numpunct<char> {
    char do_thousands_sep () const override { return ','; }
    std::string do_grouping () const override { return "\\3"; }
};
void erf_write_build_metadata (erf_description::Writer&);
int main (int argc, char**) {
    // Formatting must not inherit an application's stream flags or locale.
    std::cout.imbue(std::locale(std::locale::classic(), new Grouped));
    std::cout << std::hex << std::showbase << std::boolalpha;
    erf_description::Writer writer(std::cout, argc > 1
        ? erf_description::Format::YAML : erf_description::Format::JSON);
    erf_write_build_metadata(writer);
    writer.begin_object("scalars");
    writer.value("number", 1234);
    writer.value("enabled", true);
    writer.value("disabled", false);
    writer.value("unknown", nullptr);
    writer.end_object();
    writer.finish();
}
""")
        cls.probe = work / ("probe.exe" if os.name == "nt" else "probe")
        compiler = shlex.split(os.environ.get("CXX", "c++"))
        sources = [str(work / "main.cpp"), str(work / "generated.cpp")]
        if Path(compiler[0]).stem.lower() in ("cl", "clang-cl"):
            flags = ["/nologo", "/std:c++17", "/EHsc", "/I" + str(ROOT / "Source/IO")]
            output = ["/Fe:" + str(cls.probe)]
        else:
            flags = ["-std=c++17", "-I" + str(ROOT / "Source/IO")]
            output = ["-o", str(cls.probe)]
        subprocess.run(
            compiler + flags + sources + output,
            cwd=work,
            check=True,
            capture_output=True,
            text=True,
        )
        cls.schema = json.loads(
            (ROOT / "Docs/schema/erf-build-description-v1.schema.json").read_text()
        )
        jsonschema.Draft202012Validator.check_schema(cls.schema)
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

    def test_json_and_yaml_round_trip_without_scalar_coercion(self):
        json_text = subprocess.check_output([str(self.probe)], text=True)
        yaml_text = subprocess.check_output([str(self.probe), "yaml"], text=True)
        expected = {
            "build": self.build,
            "scalars": {
                "number": 1234,
                "enabled": True,
                "disabled": False,
                "unknown": None,
            },
        }
        self.assertEqual(json.loads(json_text), expected)
        parsed = yaml.safe_load(yaml_text)
        self.assertEqual(parsed, expected)
        self.assertIs(parsed["scalars"]["enabled"], True)
        self.assertIs(parsed["scalars"]["disabled"], False)
        self.assertIsNone(parsed["scalars"]["unknown"])
        self.assertIs(type(parsed["scalars"]["number"]), int)
        self.assertTrue(
            all(type(v) is str for v in parsed["build"]["gpu_target_settings"])
        )
        self.assertIn("build:\n  build_system:", yaml_text)
        self.assertEqual(len(list(yaml.safe_load_all(yaml_text))), 1)

    def test_schema_accepts_unknowns_and_additive_fields(self):
        sample = self.complete_sample()
        sample["future_field"] = {"example": 42}
        sample["features"]["future_feature"] = True
        self.validator.validate(sample)

    @staticmethod
    def complete_sample():
        sample = metadata()
        sample.update(
            {
                "source_describe": None,
                "particle_real_bits": 64,
                "compiler": {"name": None, "version": None},
                "dependencies": {
                    "amrex": {"version": None, "describe": None},
                    "noahmp": {"describe": None},
                    "netcdf": {"version": None},
                    "fft": {"provider": "fftw", "version": None},
                },
                "mpi_standard": {"major": 3, "minor": 1},
            }
        )
        return sample

    def test_schema_rejects_ambiguous_or_contradictory_metadata(self):
        for field, value in [
            ("schema_version", 2),
            ("schema_version", True),
            ("real_bits", "64"),
            ("particle_real_bits", None),
            ("mpi_standard", None),
            ("gpu_backend", "unknown"),
        ]:
            sample = self.complete_sample()
            sample[field] = value
            with (
                self.subTest(field=field, value=value),
                self.assertRaises(jsonschema.ValidationError),
            ):
                self.validator.validate(sample)
        sample = self.complete_sample()
        del sample["build"]["source"]["revision"]
        with self.assertRaises(jsonschema.ValidationError):
            self.validator.validate(sample)
        sample = self.complete_sample()
        sample["features"]["fft"] = "false"
        with self.assertRaises(jsonschema.ValidationError):
            self.validator.validate(sample)

    def test_schema_accepts_disabled_features_with_null_details(self):
        sample = self.complete_sample()
        sample["features"]["particles"] = False
        sample["particle_real_bits"] = None
        sample["features"]["mpi"] = False
        sample["mpi_standard"] = None
        self.validator.validate(sample)

    def test_installed_executable_formats_and_schema(self):
        if EXECUTABLE is None:
            self.skipTest("pass --executable to test a real ERF binary")
        json_result = subprocess.run(
            LAUNCHER + [EXECUTABLE, "--describe-json"],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        yaml_result = subprocess.run(
            LAUNCHER + [EXECUTABLE, "--describe-yaml"],
            text=True,
            capture_output=True,
            check=True,
            timeout=30,
        )
        description = json.loads(json_result.stdout)
        self.assertEqual(description, yaml.safe_load(yaml_result.stdout))
        self.assertEqual(len(list(yaml.safe_load_all(yaml_result.stdout))), 1)
        self.validator.validate(description)
        # A schema-version mismatch must remain a hard compatibility failure.
        incompatible = copy.deepcopy(description)
        incompatible["schema_version"] += 1
        with self.assertRaises(jsonschema.ValidationError):
            self.validator.validate(incompatible)


class GeneratorTests(unittest.TestCase):
    def test_dirty_status_and_untracked_files(self):
        with tempfile.TemporaryDirectory(prefix="erf-metadata-git-") as directory:

            def git(*args):
                return subprocess.check_output(
                    ["git", "-C", directory, *args], text=True, stderr=subprocess.PIPE
                ).strip()

            git("init")
            source = Path(directory) / "source.txt"
            source.write_text("original\n")
            git("add", "source.txt")
            git(
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.invalid",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-m",
                "test source",
            )
            revision = git("rev-parse", "HEAD")
            self.assertEqual(
                build_metadata.source_identity(directory),
                {"revision": revision, "tracked_dirty": False},
            )
            (Path(directory) / "untracked.txt").write_text("untracked\n")
            self.assertIs(
                build_metadata.source_identity(directory)["tracked_dirty"], False
            )
            source.write_text("changed\n")
            self.assertIs(
                build_metadata.source_identity(directory)["tracked_dirty"], True
            )
            git("add", "source.txt")
            self.assertIs(
                build_metadata.source_identity(directory)["tracked_dirty"], True
            )

    def test_source_archive_does_not_inherit_parent_git_identity(self):
        # Tools is inside a repository but is not itself a repository root.
        self.assertEqual(
            build_metadata.source_identity(str(ROOT / "Tools")),
            {"revision": None, "tracked_dirty": None},
        )

    def test_actual_source_full_revision(self):
        if not (ROOT / ".git").exists():
            self.skipTest("source archive has no Git identity")
        expected = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
        self.assertEqual(
            build_metadata.source_identity(str(ROOT))["revision"], expected
        )

    def test_generator_targets_and_unknown_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.cpp"
            for backend, targets, expected in (
                ("cuda", "80;90", ["80", "90"]),
                ("hip", "gfx90a:xnack-", ["gfx90a:xnack-"]),
                ("sycl", "", None),
                ("none", "", []),
            ):
                with (
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "build_metadata.py",
                            "--source",
                            directory,
                            "--build-system",
                            "gnu-make",
                            "--gpu-backend",
                            backend,
                            "--gpu-targets",
                            targets,
                            "--output",
                            str(path),
                        ],
                    ),
                    mock.patch.object(
                        build_metadata, "emit_cpp", side_effect=json.dumps
                    ),
                ):
                    build_metadata.main()
                # FormatTests checks serialization; this tests input collection.
                data = json.loads(path.read_text())
                self.assertEqual(data["gpu_target_settings"], expected)
                self.assertIsNone(data["source"]["revision"])
                self.assertEqual(data["build_system"], "gnu-make")


class ExecutableTests(unittest.TestCase):
    def test_json_capabilities_from_empty_directory(self):
        if EXECUTABLE is None:
            self.skipTest("pass --executable to test ERF")
        with tempfile.TemporaryDirectory(prefix="erf-describe-empty-") as directory:
            result = subprocess.run(
                LAUNCHER + [EXECUTABLE, "--describe-json"],
                cwd=directory,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            # json.loads rejects banners, trailing text and duplicate rank documents.
            data = json.loads(result.stdout)
            self.assertEqual(list(Path(directory).iterdir()), [])
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["application"], "ERF")
        self.assertEqual(data["dimension"], 3)
        self.assertIn(data["real_bits"], (32, 64))
        for name in ("mpi", "particles", "fft"):
            self.assertIs(type(data["features"][name]), bool)
            expected = getattr(ARGS, "expect_" + name)
            if expected is not None:
                self.assertEqual(data["features"][name], expected == "ON")
        self.assertEqual(data["mpi_standard"] is not None, data["features"]["mpi"])
        self.assertEqual(
            data["particle_real_bits"] is not None, data["features"]["particles"]
        )
        if ARGS.expect_backend:
            self.assertEqual(data["gpu_backend"], ARGS.expect_backend)
        if ARGS.expect_real_bits:
            self.assertEqual(data["real_bits"], ARGS.expect_real_bits)
        if not data["features"]["fft"]:
            self.assertIsNone(data["dependencies"]["fft"]["provider"])
        elif data["gpu_backend"] == "none":
            self.assertEqual(data["dependencies"]["fft"]["provider"], "fftw")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable")
    for feature in ("mpi", "particles", "fft"):
        parser.add_argument("--expect-" + feature, choices=("ON", "OFF"))
    parser.add_argument("--expect-backend", choices=("none", "cuda", "hip", "sycl"))
    parser.add_argument("--expect-real-bits", type=int, choices=(32, 64))
    parser.add_argument(
        "--launcher",
        nargs=argparse.REMAINDER,
        default=[],
        help="MPI launcher and arguments; must be the last option",
    )
    ARGS, remaining = parser.parse_known_args()
    EXECUTABLE = str(Path(ARGS.executable).resolve()) if ARGS.executable else None
    LAUNCHER = ARGS.launcher
    unittest.main(argv=[__file__] + remaining)
