#!/usr/bin/env python3
"""Check that ERF's Spack recipe pins the same submodule commits as ERF's tree.

The recipe (Build/spack/spack_repo/erf/packages/erf/package.py) fetches the RRTMGP/EKAT
stack and Noah-MP as pinned git resources instead of submodules, so release tarballs can
be built. This compares those pins with the gitlinks in ERF's own tree:

  * the @develop pins against the ref given with --ref (default HEAD);
  * each tagged release the recipe offers against that tag, for the tags available locally
    (`git fetch --tags` to get them all);
  * EKAT's own extern/* pins against EKAT's tree at the pinned EKAT commit, read from a
    local Submodules/ekat checkout when it has that commit and from the GitHub API otherwise.

It also lists release tags that the recipe does not offer yet (a warning, or a failure
with --strict).

    python3 Build/spack/check_pins.py            # report; exit 1 on a mismatch
    python3 Build/spack/check_pins.py --fix      # also rewrite the develop Noah-MP pin

Needs only git and the standard library. Run it from anywhere inside the ERF checkout.
"""

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE_PY = os.path.join(HERE, "spack_repo", "erf", "packages", "erf", "package.py")
EKAT_EXTERNS = ("kokkos", "yaml-cpp", "spdlog", "Catch2")
RELEASE_TAG = re.compile(r"^\d\d\.\d\d$")


def git(*args, cwd=None, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def gitlinks(ref, path, cwd):
    """{path: commit} for the submodule gitlinks under path at ref."""
    out = {}
    for line in git("ls-tree", ref, path + "/", cwd=cwd).splitlines():
        mode, kind, sha, name = line.split(None, 3)
        if mode == "160000":
            out[name] = sha
    return out


def recipe_pins(path):
    """The pin tables and the versions the recipe declares, read from its AST."""
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Erf")
    tables = {}
    versions = []
    for node in cls.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], "id", None)
            if name in ("_rrtmgp_stack", "_noahmp_pins"):
                tables[name] = ast.literal_eval(node.value)
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            for entry in ast.literal_eval(node.iter):
                if isinstance(entry, tuple) and len(entry) == 2 and RELEASE_TAG.match(entry[0]):
                    versions.append(entry[0])
    for node in ast.walk(cls):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "version"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            versions.append(node.args[0].value)
    missing = {"_rrtmgp_stack", "_noahmp_pins"} - set(tables)
    if missing:
        raise RuntimeError(f"{path}: could not read {', '.join(sorted(missing))}")
    return tables["_rrtmgp_stack"], tables["_noahmp_pins"], sorted(set(versions))


def vtuple(v):
    return tuple(int(x) for x in v.split("."))


def noahmp_pin_for(version, pins):
    """The Noah-MP pin whose version range covers a release, or None."""
    for spec, commit in pins:
        if spec == "develop":
            continue
        lo, _, hi = spec.partition(":")
        hi = hi or lo
        if vtuple(lo) <= vtuple(version) <= vtuple(hi):
            return commit
    return None


def ekat_externs(ekat_commit, root):
    """{name: commit} for EKAT's extern/* submodules at a commit, or None if unavailable."""
    ekat_dir = os.path.join(root, "Submodules", "ekat")
    probe = subprocess.run(
        ["git", "-C", ekat_dir, "cat-file", "-e", f"{ekat_commit}^{{commit}}"], capture_output=True
    )
    if probe.returncode == 0:
        return gitlinks(ekat_commit, "extern", cwd=ekat_dir)
    url = f"https://api.github.com/repos/E3SM-Project/EKAT/contents/extern?ref={ekat_commit}"
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json"})
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            entries = json.load(resp)
    except Exception as exc:  # network, rate limit: report, do not fail
        print(f"warning: could not read EKAT's extern pins ({exc}); skipping that check")
        return None
    # A directory listing reports a gitlink's commit as its sha (and calls it a file).
    return {e["name"]: e["sha"] for e in entries if e.get("name") in EKAT_EXTERNS}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ref", default="HEAD", help="ref whose tree the @develop pins must match")
    ap.add_argument(
        "--fix", action="store_true", help="rewrite the develop Noah-MP pin to match --ref"
    )
    ap.add_argument(
        "--strict", action="store_true", help="fail when a release tag is missing from the recipe"
    )
    args = ap.parse_args()

    root = git("rev-parse", "--show-toplevel", cwd=HERE).strip()
    stack, noahmp, versions = recipe_pins(PACKAGE_PY)
    stack_by_path = {dest: commit for _, _, commit, dest in stack}
    failures = []
    rows = []

    def check(label, expected, actual):
        ok = expected == actual
        rows.append((label, expected, actual, ok))
        if not ok:
            failures.append(label)

    # @develop: the tree at --ref.
    links = gitlinks(args.ref, "Submodules", cwd=root)
    develop_noahmp = dict(noahmp).get("develop")
    check("develop Noah-MP", links.get("Submodules/Noah-MP"), develop_noahmp)
    for path in ("Submodules/RRTMGP", "Submodules/ekat"):
        check(f"develop {path}", links.get(path), stack_by_path.get(path))

    # Each release the recipe offers, where the tag is available locally.
    skipped_tags = []
    for v in versions:
        if not RELEASE_TAG.match(v):
            continue
        if (
            subprocess.run(
                ["git", "-C", root, "rev-parse", "-q", "--verify", f"refs/tags/{v}^{{commit}}"],
                capture_output=True,
            ).returncode
            != 0
        ):
            skipped_tags.append(v)
            continue
        tag_links = gitlinks(v, "Submodules", cwd=root)
        if vtuple(v) >= (25, 10):
            for path in ("Submodules/RRTMGP", "Submodules/ekat"):
                check(f"{v} {path}", tag_links.get(path), stack_by_path.get(path))
        if vtuple(v) >= (25, 12):
            check(f"{v} Noah-MP", tag_links.get("Submodules/Noah-MP"), noahmp_pin_for(v, noahmp))

    # EKAT's own submodules at the pinned EKAT commit.
    externs = ekat_externs(stack_by_path["Submodules/ekat"], root)
    if externs is not None:
        for name in EKAT_EXTERNS:
            check(
                f"EKAT extern/{name}",
                externs.get(name),
                stack_by_path.get(f"Submodules/ekat/extern/{name}"),
            )

    # Release tags the recipe does not offer.
    tags = [t for t in git("tag", "--list", cwd=root).split() if RELEASE_TAG.match(t)]
    newest = max((vtuple(v) for v in versions if RELEASE_TAG.match(v)), default=(0, 0))
    unoffered = sorted((t for t in tags if vtuple(t) > newest), key=vtuple)

    width = max(len(r[0]) for r in rows)
    for label, expected, actual, ok in rows:
        print(
            f"{'ok  ' if ok else 'FAIL'} {label:{width}}  tree {expected or '-'}  recipe {actual or '-'}"
        )
    if skipped_tags:
        print(f"note: tags not available locally, not checked: {', '.join(skipped_tags)}")
    if unoffered:
        print(
            f"{'FAIL' if args.strict else 'warning'}: release tags the recipe does not offer: {', '.join(unoffered)}"
        )
        if args.strict:
            failures.append("unoffered releases")

    if (
        args.fix
        and links.get("Submodules/Noah-MP")
        and links["Submodules/Noah-MP"] != develop_noahmp
    ):
        src = open(PACKAGE_PY, encoding="utf-8").read()
        new, n = re.subn(
            r'\("develop", "[0-9a-f]{40}"\)', f'("develop", "{links["Submodules/Noah-MP"]}")', src
        )
        if n != 1:
            raise RuntimeError("could not find the develop Noah-MP pin to rewrite")
        open(PACKAGE_PY, "w", encoding="utf-8").write(new)
        print(f"fixed: develop Noah-MP pin -> {links['Submodules/Noah-MP']}")
        failures.remove("develop Noah-MP")
    if any(f.startswith("develop Submodules/") for f in failures):
        print(
            "The RRTMGP/EKAT pins moved: update _rrtmgp_stack in package.py (ERF's RRTMGP and ekat gitlinks, "
            "then EKAT's extern/* gitlinks at the new EKAT commit) and rerun this check."
        )

    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write("### Spack recipe pins\n\n| | check | tree | recipe |\n|---|---|---|---|\n")
            for label, expected, actual, ok in rows:
                f.write(
                    f"| {'ok' if ok else 'FAIL'} | {label} | `{expected or '-'}` | `{actual or '-'}` |\n"
                )
            if unoffered:
                f.write(f"\nRelease tags the recipe does not offer: {', '.join(unoffered)}\n")
    if failures:
        print(f"{len(failures)} mismatch(es): {', '.join(failures)}")
        return 1
    print("all pins match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
