#!/usr/bin/env python3
"""Transfer a ccache archive with ORAS on runners without Docker.

Only input errors fail the action. Registry, installation, and filesystem failures
leave compilation free to proceed. No third-party Python packages are needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
TOKEN_VARIABLES = ("GHCR_TOKEN", "GH_TOKEN", "GITHUB_TOKEN")


@dataclass(frozen=True)
class Config:
    mode: str
    cache: Path
    ref: str
    platform: tuple[str, str]
    timeout: int
    runner_temp: Path
    output_path: Path | None
    token: str = field(repr=False)
    actor: str
    source: str
    revision: str
    prune_env: dict[str, str] = field(repr=False)


def read_config(env):
    """Validate action inputs once, before any installation or registry access."""
    mode = env["CACHE_MODE"]
    if mode not in ("restore", "save"):
        raise ValueError("mode must be restore or save")
    token = env.get("GHCR_TOKEN", "")
    if mode == "save" and not token:
        raise ValueError("token is required when saving")
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", env["CACHE_KEY"]).lower()
    if not re.fullmatch(r"[a-z0-9_][a-z0-9_.-]{0,127}", tag):
        raise ValueError("key must produce a valid GHCR tag")
    raw_path = env["CACHE_PATH"]
    cache = Path(raw_path).expanduser().resolve()
    if not raw_path or cache in (Path(cache.anchor), Path.home().resolve()):
        raise ValueError("path must name a dedicated ccache directory")
    if cache.exists() and not cache.is_dir():
        raise ValueError("cache path is not a directory")
    platform = (env["RUNNER_OS"], env["RUNNER_ARCH"])
    if platform not in ORAS_RELEASES:
        raise ValueError("unsupported ORAS runner platform")
    timeout = int(env.get("CACHE_TIMEOUT", "180"))
    if not 1 <= timeout <= 600:
        raise ValueError("transfer timeout must be between 1 and 600 seconds")
    owner = env["GITHUB_REPOSITORY_OWNER"]
    saving = mode == "save"
    return Config(
        mode=mode, cache=cache, ref=f"ghcr.io/{owner.lower()}/erf-ccache:{tag}",
        platform=platform, timeout=timeout, runner_temp=Path(env["RUNNER_TEMP"]),
        output_path=Path(env["GITHUB_OUTPUT"]) if not saving else None,
        token=token, actor=env["GITHUB_ACTOR"] if token else "",
        source=env["CACHE_SOURCE"] if saving else "",
        revision=env["GITHUB_SHA"] if saving else "",
        prune_env=dict(env, GH_TOKEN=token, OWNER=owner, OWNER_TYPE=env["CACHE_OWNER_TYPE"])
        if saving else {},
    )


def diagnostic(value, secrets=(), limit=1024):
    """Redact before truncating, then keep subprocess output on one log line."""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value or "")
    credentials = [*secrets, *(os.environ.get(name) for name in TOKEN_VARIABLES)]
    for secret in sorted({s for s in credentials if s}, key=len, reverse=True):
        if isinstance(secret, bytes):
            secret = secret.decode("utf-8", errors="replace")
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"(?i)\b(Bearer|Basic)\s+[^\s,;\"']+", r"\1 [redacted]", text)
    # Cover JSON/key=value credentials without hiding HTTP status after /token:.
    text = re.sub(
        r'''(?i)((?<![\w/])(?:token|access_token|refresh_token|password|auth)\b'''
        r'''["']?\s*[:=]\s*["']?)[^\s,;&"']+''', r"\1[redacted]", text)
    text = re.sub(r"(https?://)[^/\s]+@", r"\1[redacted]@", text)
    text = re.sub(r'''(https?://[^\s?"']+)\?[^\s"']+''', r"\1?[redacted]", text)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + " [truncated]"


class CommandError(RuntimeError):
    def __init__(self, message, detail, returncode=None):
        super().__init__(message + (f"; {detail}" if detail else ""))
        self.detail = detail
        self.returncode = returncode

    def missing_manifest(self, ref):
        # A generic 404 may be an authentication endpoint failure. Only explicit
        # manifest errors or ORAS's exact reference-not-found response are misses.
        if self.returncode is None or self.returncode <= 0 or "[truncated]" in self.detail:
            return False
        detail = self.detail.lower()
        if re.search(r"\b(401|403|5\d\d|unauthorized|forbidden|denied)\b", detail):
            return False
        return (bool(re.search(r"(?<![\w/:-])manifest[ _]unknown\b", detail))
                or f"{ref.lower()}: not found" in detail)


def warning(message):
    print("::warning::" + diagnostic(message, limit=2048).replace("%", "%25"), file=sys.stderr)


def run(command, *, timeout=60, attempts=2, label=None, **kwargs):
    """Retry bounded commands; report the final failure without argv or stdin."""
    for attempt in range(attempts):
        try:
            return subprocess.run(command, check=True, capture_output=True,
                                  text=True, timeout=timeout, **kwargs).stdout
        except (subprocess.SubprocessError, OSError) as exc:
            if attempt + 1 == attempts:
                if isinstance(exc, subprocess.CalledProcessError):
                    reason = f"exit status {exc.returncode}"
                elif isinstance(exc, subprocess.TimeoutExpired):
                    reason = f"timed out after {timeout}s"
                else:
                    reason = f"could not start ({type(exc).__name__})"
                detail = ((getattr(exc, "stderr", None) or "").strip()
                          or getattr(exc, "stdout", None) or getattr(exc, "strerror", ""))
                env = kwargs.get("env") or os.environ
                secrets = [kwargs.get("input"), *(env.get(name) for name in TOKEN_VARIABLES)]
                count = f"{attempts} attempt" + ("s" if attempts != 1 else "")
                message = f"{label or Path(command[0]).name}: {reason} after {count}"
                raise CommandError(message, diagnostic(detail, secrets),
                                   getattr(exc, "returncode", None)) from None
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
        timeout=65, label="download ORAS")
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


def setup_registry(config, directory):
    """Install ORAS and log in using an isolated, temporary credential file."""
    binary = install_oras(directory, config.platform)
    registry_file = directory / "registry.json"
    registry_file.write_text('{"auths":{}}', encoding="utf-8")
    registry_file.chmod(0o600)
    registry_args = ["--registry-config", str(registry_file)]
    if config.token:
        run([str(binary), "login", "ghcr.io", *registry_args,
             "--username", config.actor, "--password-stdin"],
            input=config.token, timeout=30, label="ORAS login")
    return binary, registry_args


def restore(config, binary, registry_args, directory):
    """Verify a downloaded snapshot, then replace the cache with rollback support."""
    cache, ref = config.cache, config.ref
    archive = directory / ARCHIVE_NAME
    try:
        manifest = json.loads(run([str(binary), "manifest", "fetch", *registry_args, ref],
                                  timeout=config.timeout, label="fetch cache manifest"))
    except CommandError as exc:
        if not exc.missing_manifest(ref):
            raise
        print(f"no cache at {ref}; continuing with the local cache")
        return
    layer = layer_descriptor(manifest)
    # Fetch the immutable blob from this manifest, not the moving configuration tag.
    # An artifact-supplied filename never reaches the filesystem.
    blob_ref = ref.rsplit(":", 1)[0] + "@" + layer["digest"]
    run([str(binary), "blob", "fetch", *registry_args, "--output", str(archive), blob_ref],
        timeout=config.timeout, label="fetch cache archive")
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
    output(config.output_path, "hit=true")
    print(f"restored {ref}")


def save(config, binary, registry_args, directory):
    """Package and publish the cache; leave pruning to the successful-save path."""
    archive = directory / ARCHIVE_NAME
    pack(config.cache, archive)
    run([str(binary), "push", *registry_args, "--artifact-type", ARTIFACT_TYPE,
         "--annotation", "org.opencontainers.image.source=" + config.source,
         "--annotation", "org.opencontainers.image.revision=" + config.revision,
         config.ref, ARCHIVE_NAME + ":" + LAYER_TYPE], cwd=directory, timeout=config.timeout,
        label="publish cache")
    print(f"saved {config.ref}")


def prune(config):
    """Pruning is best effort and must not turn a successful save into a failure."""
    try:
        print(run([sys.executable, str(Path(__file__).with_name("prune.py")), "erf-ccache:0"],
                  env=config.prune_env, timeout=65, attempts=1, label="prune cache versions"), end="")
    except RuntimeError as exc:
        warning(f"could not prune old GHCR ccache versions ({exc}); a later save will retry")


def output(path, value):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(value + "\n")


def main():
    try:
        config = read_config(dict(os.environ))
    except (KeyError, ValueError, OSError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 2

    if config.mode == "restore":
        output(config.output_path, "hit=false")
    try:
        if config.mode == "save" and not any(p.is_file() for p in config.cache.rglob("*")):
            print(f"no ccache files in {config.cache}: skip save")
            return 0
        with tempfile.TemporaryDirectory(prefix="erf-ccache-", dir=config.runner_temp) as tmp:
            directory = Path(tmp)
            binary, registry_args = setup_registry(config, directory)
            if config.mode == "restore":
                restore(config, binary, registry_args, directory)
            else:
                save(config, binary, registry_args, directory)
                prune(config)
    except (OSError, RuntimeError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        warning(f"could not {config.mode} GHCR ccache ({exc}); continuing with the local cache")
    return 0


if __name__ == "__main__":
    def interrupted(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
