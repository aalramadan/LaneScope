import hashlib
import json
import pathlib
import re
import shutil
import tarfile
import tempfile
import urllib.request

REDIST_URL = "https://developer.download.nvidia.com/compute/cuda/redist/"
PLATFORM = "linux-x86_64"
MANIFEST_NAME = re.compile(r"redistrib_(\d+)\.(\d+)\.(\d+)\.json")
CURRENT = "current"
PREFIX = "nsight-compute-"
TIMEOUT = 60
CHUNK = 1 << 20

Release = tuple[int, int, int]


def releases(index: str) -> list[Release]:
    """Return every CUDA release that an index page of the redistributables lists."""
    found = {(int(a), int(b), int(c)) for a, b, c in MANIFEST_NAME.findall(index)}
    return sorted(found)


def pick(found: list[Release], cuda: tuple[int, int]) -> Release:
    """Return the newest release at or below the CUDA version that the driver supports."""
    usable = [release for release in found if release[:2] <= cuda]
    if not usable:
        raise ValueError(f"NVIDIA lists no CUDA release at or below {cuda[0]}.{cuda[1]}")
    return max(usable)


def archive(manifest: object) -> tuple[str, str, str]:
    """Return the version, the relative path, and the SHA-256 of the Nsight Compute archive."""
    entry = manifest.get("nsight_compute") if isinstance(manifest, dict) else None
    target = entry.get(PLATFORM) if isinstance(entry, dict) else None
    if not isinstance(entry, dict) or not isinstance(target, dict):
        raise ValueError(f"the manifest names no Nsight Compute archive for {PLATFORM}")
    version, path, digest = entry.get("version"), target.get("relative_path"), target.get("sha256")
    if not all(isinstance(value, str) for value in (version, path, digest)):
        raise ValueError("the Nsight Compute entry of the manifest lacks its version or files")
    assert isinstance(version, str) and isinstance(path, str) and isinstance(digest, str)
    if not re.fullmatch(r"[0-9.]+", version) or ".." in path or path.startswith("/"):
        raise ValueError("the Nsight Compute entry of the manifest has an unsafe name")
    return version, path, digest


def install(
    cuda: tuple[int, int],
    folder: pathlib.Path,
    base_url: str = REDIST_URL,
    replace: bool = False,
) -> pathlib.Path:
    """Download and unpack the Nsight Compute that suits the CUDA version, returning its ncu.

    An unpacked copy of the same version is used as it is, unless `replace` asks
    for a fresh download, such as after a damaged unpack. The old copy stays until
    the new one is checked and unpacked, so a failed download loses nothing.
    """
    release = pick(releases(_text(base_url)), cuda)
    manifest = json.loads(_text(f"{base_url}redistrib_{'.'.join(map(str, release))}.json"))
    version, path, digest = archive(manifest)
    target = folder / f"{PREFIX}{version}"
    found = None if replace else find_ncu(target)
    if found is None:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder, prefix=".download-") as scratch:
            package = pathlib.Path(scratch) / "archive.tar.xz"
            _download(base_url + path, package, digest)
            unpacked = pathlib.Path(scratch) / "unpacked"
            _unpack(package, unpacked)
            if find_ncu(unpacked) is None:
                raise ValueError(f"the Nsight Compute archive of {version} holds no ncu")
            if target.exists():
                # The old copy moves into the scratch folder, which goes away with it.
                target.rename(pathlib.Path(scratch) / "old")
            shutil.move(str(unpacked), str(target))
        found = find_ncu(target)
        assert found is not None
    (folder / CURRENT).write_text(target.name)
    return found


def cached_ncu(folder: pathlib.Path) -> pathlib.Path | None:
    """Return the ncu that the last install chose, or else the newest one in the cache."""
    try:
        chosen = (folder / CURRENT).read_text().strip()
    except OSError:
        chosen = ""
    if chosen.startswith(PREFIX) and (found := find_ncu(folder / chosen)) is not None:
        return found
    versions = sorted(
        (_version(path.name), path) for path in folder.glob(f"{PREFIX}*") if _version(path.name)
    )
    for _, path in reversed(versions):
        if (found := find_ncu(path)) is not None:
            return found
    return None


def _version(name: str) -> tuple[int, ...]:
    text = name.removeprefix(PREFIX)
    return tuple(int(part) for part in text.split(".")) if re.fullmatch(r"[0-9.]+", text) else ()


def find_ncu(folder: pathlib.Path) -> pathlib.Path | None:
    """Return the ncu program inside an unpacked archive, at most three folders deep."""
    if not folder.is_dir():
        return None
    for depth in range(4):
        for candidate in sorted(folder.glob("/".join(["*"] * depth + ["ncu"]))):
            if candidate.is_file():
                return candidate
    return None


def _text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        return response.read().decode()


def _download(url: str, path: pathlib.Path, digest: str) -> None:
    hasher = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response, open(path, "wb") as handle:
        while chunk := response.read(CHUNK):
            hasher.update(chunk)
            handle.write(chunk)
    if hasher.hexdigest() != digest:
        raise ValueError(f"the archive at {url} does not match the SHA-256 of its manifest")


def _unpack(package: pathlib.Path, folder: pathlib.Path) -> None:
    # The data filter refuses absolute paths, links out of the folder, and device files.
    if not hasattr(tarfile, "data_filter"):
        raise ValueError("this Python cannot unpack archives safely; use Python 3.10.12 or later")
    with tarfile.open(package, "r:xz") as archive_file:
        archive_file.extractall(folder, filter="data")
