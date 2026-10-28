#!/usr/bin/env python3
"""Transfer a ccache archive with ORAS on runners without Docker.

Only input errors fail the action. Registry, installation, and filesystem failures
leave compilation free to proceed. No third-party Python packages are needed.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile

ORAS_VERSION = "1.3.0"
ORAS_RELEASES = {
    ("macOS", "X64"): ("darwin_amd64.tar.gz", "82c33f7da8430ea7fa7e7bdf7721be0a0d0481e5ccb2472ea438490d5e8641a9"),
    ("macOS", "ARM64"): ("darwin_arm64.tar.gz", "e10c6552c02d5a7c7eaf7170d3b6f7f094b675a98a1e0edf4d4478a909447245"),
    ("Windows", "X64"): ("windows_amd64.zip", "b050e93aa0dc7a79a61fa8e4074dfa302c41d4af01b634fe393c5dd687536aee"),
    ("Linux", "X64"): ("linux_amd64.tar.gz", "6cdc692f929100feb08aa8de584d02f7bcc30ec7d88bc2adc2054d782db57c64"),
    ("Linux", "ARM64"): ("linux_arm64.tar.gz", "7649738b48fde10542bcc8b0e9b460ba83936c75fb5be01ee6d4443764a14352"),
}
ARTIFACT_TYPE = "application/vnd.erf.ccache.v1"
LAYER_TYPE = "application/vnd.oci.image.layer.v1.tar"
ARCHIVE_NAME = "ccache.tar"
# Well above the existing 150-1100 MB cache limits; rejects damaged/hostile archives.
MAX_ARCHIVE_BYTES = 8 * 1024**3
MAX_MEMBERS = 500_000


def warning(message):
    print(f"::warning::{message}", file=sys.stderr)


def run(command, *, timeout=60, attempts=2, **kwargs):
    """Retry only bounded commands. Keep credentials and command output out of errors."""
    for attempt in range(attempts):
        try:
            return subprocess.run(command, check=True, capture_output=True,
                                  text=True, timeout=timeout, **kwargs).stdout
        except (subprocess.SubprocessError, OSError):
            if attempt + 1 == attempts:
                raise RuntimeError(f"{Path(command[0]).name} {command[1]} failed or timed out") from None
            time.sleep(2)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def install_oras(directory, platform):
    filename, checksum = ORAS_RELEASES[platform]
    asset = f"oras_{ORAS_VERSION}_{filename}"
    archive = directory / asset
    run(["curl", "--fail", "--silent", "--show-error", "--location", "--max-time", "60",
         "--output", str(archive),
         f"https://github.com/oras-project/oras/releases/download/v{ORAS_VERSION}/{asset}"],
        timeout=65)
    if sha256(archive) != checksum:
        raise ValueError("ORAS download checksum mismatch")
    executable = "oras.exe" if platform[0] == "Windows" else "oras"
    # Extract only the verified executable, never arbitrary archive paths.
    if filename.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            contents = bundle.read(executable)
    else:
        with tarfile.open(archive) as bundle:
            contents = bundle.extractfile(executable).read()
    binary = directory / executable
    binary.write_bytes(contents)
    binary.chmod(0o700)
    return binary


def archive_path(name):
    """Accept ordinary relative paths on both POSIX and Windows."""
    path = PurePosixPath(name)
    if (not path.parts or path.is_absolute() or ".." in path.parts
            or PureWindowsPath(name).drive or "\\" in name or ":" in name
            or any(p.endswith((" ", ".")) for p in path.parts)
            or any(re.fullmatch(r"(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])", p.split(".")[0])
                   for p in path.parts)):
        raise ValueError("unsafe cache archive path")
    return path


def unpack(archive, destination):
    """Write regular files/directories ourselves; never follow archive links."""
    total = 0
    with tarfile.open(archive, "r:") as bundle:
        for count, member in enumerate(bundle, 1):
            if count > MAX_MEMBERS or not (member.isfile() or member.isdir()):
                raise ValueError("unsupported cache archive member")
            relative = archive_path(member.name)
            total += member.size
            if total > MAX_ARCHIVE_BYTES:
                raise ValueError("cache archive exceeds size limit")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                # CI evicts objects not touched by the build; restoring must not
                # make every old object appear freshly used.
                os.utime(target, (member.mtime, member.mtime))


def pack(cache, archive):
    with tarfile.open(archive, "w") as bundle:
        for path in sorted(cache.rglob("*")):
            relative = path.relative_to(cache).as_posix()
            archive_path(relative)
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise ValueError("cache contains a link or special file")
            bundle.add(path, arcname=relative, recursive=False)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("cache archive exceeds size limit")


def promote(staged, cache):
    """Keep the previous cache until the replacement is installed successfully."""
    previous = staged.with_name(staged.name + ".previous")
    if cache.exists():
        cache.rename(previous)
    try:
        staged.rename(cache)
    except OSError:
        if previous.exists():
            try:
                previous.rename(cache)
            except OSError:
                warning(f"original cache remains at {previous}")
        raise
    if previous.exists():
        shutil.rmtree(previous)


def layer_descriptor(manifest):
    if (not isinstance(manifest, dict)
            or not isinstance(manifest.get("layers"), list)
            or manifest.get("artifactType") != ARTIFACT_TYPE
            or len(manifest.get("layers", [])) != 1):
        raise ValueError("unexpected cache artifact format")
    layer = manifest["layers"][0]
    if (not isinstance(layer, dict)
            or layer.get("mediaType") != LAYER_TYPE
            or not isinstance(layer.get("digest"), str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", layer["digest"])
            or not isinstance(layer.get("size"), int)
            or not 0 < layer["size"] <= MAX_ARCHIVE_BYTES):
        raise ValueError("invalid cache layer")
    return layer


def transfer(mode, cache, ref, directory, platform, timeout):
    binary = install_oras(directory, platform)
    config = directory / "registry.json"
    config.write_text('{"auths":{}}', encoding="utf-8")
    config.chmod(0o600)
    registry_args = ["--registry-config", str(config)]
    token = os.environ.get("GHCR_TOKEN", "")
    if token:
        run([str(binary), "login", "ghcr.io", *registry_args,
             "--username", os.environ["GITHUB_ACTOR"], "--password-stdin"],
            input=token, timeout=30)

    archive = directory / ARCHIVE_NAME
    if mode == "restore":
        manifest = json.loads(run([str(binary), "manifest", "fetch", *registry_args, ref],
                                  timeout=timeout))
        layer = layer_descriptor(manifest)
        # Fetch the immutable blob from this manifest, not the moving configuration tag.
        # An artifact-supplied filename never reaches the filesystem.
        blob_ref = ref.rsplit(":", 1)[0] + "@" + layer["digest"]
        run([str(binary), "blob", "fetch", *registry_args, "--output", str(archive), blob_ref],
            timeout=timeout)
        if (archive.stat().st_size != layer["size"]
                or "sha256:" + sha256(archive) != layer["digest"]):
            raise ValueError("cache archive digest/size mismatch")
        cache.parent.mkdir(parents=True, exist_ok=True)
        staged = Path(tempfile.mkdtemp(prefix=cache.name + ".ghcr-", dir=cache.parent))
        try:
            unpack(archive, staged)
            promote(staged, cache)
        finally:
            if staged.exists():
                shutil.rmtree(staged)
        output("hit=true")
        print(f"restored {ref}")
    else:
        pack(cache, archive)
        run([str(binary), "push", *registry_args, "--artifact-type", ARTIFACT_TYPE,
             "--annotation", "org.opencontainers.image.source=" + os.environ["CACHE_SOURCE"],
             "--annotation", "org.opencontainers.image.revision=" + os.environ["GITHUB_SHA"],
             ref, ARCHIVE_NAME + ":" + LAYER_TYPE], cwd=directory, timeout=timeout)
        print(f"saved {ref}")
        env = dict(os.environ, GH_TOKEN=token, OWNER=os.environ["GITHUB_REPOSITORY_OWNER"],
                   OWNER_TYPE=os.environ["CACHE_OWNER_TYPE"])
        try:
            print(run([sys.executable, str(Path(__file__).with_name("prune.py")), "erf-ccache:0"],
                      env=env, timeout=65, attempts=1), end="")
        except RuntimeError:
            warning("could not prune old GHCR ccache versions; a later save will retry")


def output(value):
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as stream:
        stream.write(value + "\n")


def main():
    try:
        mode = os.environ["CACHE_MODE"]
        if mode not in ("restore", "save"):
            raise ValueError("mode must be restore or save")
        if mode == "save" and not os.environ.get("GHCR_TOKEN"):
            raise ValueError("token is required when saving")
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", os.environ["CACHE_KEY"]).lower()
        if not re.fullmatch(r"[a-z0-9_][a-z0-9_.-]{0,127}", tag):
            raise ValueError("key must produce a valid GHCR tag")
        raw_path = os.environ["CACHE_PATH"]
        cache = Path(raw_path).expanduser().resolve()
        if not raw_path or cache in (Path(cache.anchor), Path.home().resolve()):
            raise ValueError("path must name a dedicated ccache directory")
        if cache.exists() and not cache.is_dir():
            raise ValueError("cache path is not a directory")
        platform = (os.environ["RUNNER_OS"], os.environ["RUNNER_ARCH"])
        if platform not in ORAS_RELEASES:
            raise ValueError("unsupported ORAS runner platform")
        timeout = int(os.environ.get("CACHE_TIMEOUT", "180"))
        if not 1 <= timeout <= 600:
            raise ValueError("transfer timeout must be between 1 and 600 seconds")
    except (KeyError, ValueError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    if mode == "restore":
        output("hit=false")
    ref = f"ghcr.io/{os.environ['GITHUB_REPOSITORY_OWNER'].lower()}/erf-ccache:{tag}"
    try:
        if mode == "save" and not any(p.is_file() for p in cache.rglob("*")):
            print(f"no ccache files in {cache}: skip save")
            return 0
        with tempfile.TemporaryDirectory(prefix="erf-ccache-", dir=os.environ["RUNNER_TEMP"]) as tmp:
            transfer(mode, cache, ref, Path(tmp), platform, timeout)
    except (OSError, RuntimeError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        warning(f"could not {mode} GHCR ccache ({exc}); continuing with the local cache")
    return 0


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
