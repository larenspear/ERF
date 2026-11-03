"""Reduce ERF's ARM compiler crash to its GoogleTest parameter type."""

import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path("probe-results")
root.mkdir()
source = root / "probe.cpp"
source.write_text(r'''
#include <gtest/gtest.h>
#include <string>
#include <tuple>

namespace eos_test {
struct EOSState {
    double temperature;
    double pressure;
    double qv;
};
}

namespace {
using EOSStateCase = std::tuple<std::string, eos_test::EOSState>;
class ERFEOSRepresentativeStateTest : public ::testing::TestWithParam<EOSStateCase> {};
}

TEST_P(ERFEOSRepresentativeStateTest, StateIsFinite) {
    EXPECT_GT(std::get<1>(GetParam()).temperature, 0.0);
}

INSTANTIATE_TEST_SUITE_P(
    RepresentativeStates,
    ERFEOSRepresentativeStateTest,
    ::testing::Values(
        EOSStateCase{"DryWarm", {300.0, 90000.0, 0.0}},
        EOSStateCase{"MoistWarm", {300.0, 90000.0, 0.01}},
        EOSStateCase{"ColdUpperAir", {260.0, 60000.0, 0.002}},
        EOSStateCase{"ModerateMoist", {285.0, 80000.0, 0.006}}),
    [](const ::testing::TestParamInfo<EOSStateCase>& info) {
        return std::get<0>(info.param);
    });
''')

results = []
for compiler in ("g++-13", "g++-14"):
    if not shutil.which(compiler):
        raise SystemExit(f"Required runner compiler is missing: {compiler}")
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[0]
    for standard in ("c++17", "c++20"):
        for target in ("generic", "neoverse-n2"):
            name = f"{compiler}-{standard}-{target}"
            args = [compiler, f"-std={standard}", "-O3", "-DNDEBUG",
                    "-DGTEST_LINKED_AS_SHARED_LIBRARY=1",
                    "-Igoogletest-1.18.0/googletest/include"]
            if target != "generic":
                args.append(f"-mcpu={target}")
            args += ["-c", str(source), "-o", str(root / f"{name}.o")]
            run = subprocess.run(args, text=True, capture_output=True, timeout=90)
            (root / f"{name}.log").write_text(run.stdout + run.stderr)
            record = {"compiler": version, "standard": standard, "target": target,
                      "command": args, "returncode": run.returncode,
                      "internal_compiler_error": "internal compiler error" in run.stderr}
            results.append(record)
            print(json.dumps(record), flush=True)
            if run.returncode:
                print(run.stderr[:8000], flush=True)

(root / "results.json").write_text(json.dumps(results, indent=2) + "\n")
with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as summary:
    summary.write("Compiler-only diagnostic; no ERF runtime coverage.\n\n")
    summary.write("| Compiler | Standard | Target | Result |\n|---|---|---|---|\n")
    for result in results:
        status = "ICE" if result["internal_compiler_error"] else "failure" if result["returncode"] else "compiled"
        summary.write(f"| {result['compiler']} | {result['standard']} | {result['target']} | {status} |\n")
