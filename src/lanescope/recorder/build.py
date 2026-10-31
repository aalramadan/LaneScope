import dataclasses
import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from collections.abc import Sequence

from lanescope.cuda import nvrtc
from lanescope.cuda.toolchain import Toolchain, find

RECORDER_DIR = pathlib.Path(__file__).parent
LIBRARY_NAME = "liblanescope_recorder.so"
SANITIZER_LIBRARY = "libsanitizer-public.so"
# A change to one of these files makes a cached build stale. This file holds the options.
SOURCES = ("recorder.cpp", "record.h", "cubin.h", "patches.cu", "build.py")


class BuildError(RuntimeError):
    """A compiler step failed, with its output as the message."""


@dataclasses.dataclass(frozen=True)
class RecorderBuild:
    """Files that a recorded run loads."""

    library: pathlib.Path
    patches: pathlib.Path
    sanitizer_library: pathlib.Path


def check_arch(arch: str) -> str:
    """Return the compute capability digits of an architecture name such as sm_89."""
    match = re.fullmatch(r"sm_(\d{2,3})[af]?", arch)
    if match is None:
        raise ValueError(f"an architecture looks like sm_89, not {arch}")
    return match.group(1)


def build_recorder(
    out_dir: pathlib.Path,
    arch: str,
    tools: Toolchain | None = None,
    cache: pathlib.Path | None = None,
) -> RecorderBuild:
    """Build the recorder library and the patches for `arch` into `out_dir`.

    With a cache folder, the files of an earlier build from the same sources and tools for
    the same architecture are copied in place of a new compile, which takes a second or more.
    """
    number = check_arch(arch)
    tools = tools or find()
    tools.require("sanitizer_include", "sanitizer_lib", "cuda_include", "compiler")
    assert tools.sanitizer_lib and tools.sanitizer_include and tools.cuda_include and tools.compiler
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache is None:
        return _compile(out_dir, arch, number, tools)
    entry = cache / "recorder" / _build_key(arch, tools)
    try:
        _fill(entry, arch, number, tools)
        for path in entry.iterdir():
            # A new file in place of the old one spares a program that still maps the library.
            staged = out_dir / f".{path.name}.part"
            shutil.copy2(path, staged)
            os.replace(staged, out_dir / path.name)
    except (OSError, BuildError):
        # A cache that cannot serve the build, such as one on a full disk, leaves the
        # compile to out_dir, where a real compiler error shows again.
        return _compile(out_dir, arch, number, tools)
    patches = out_dir / _patches_name(arch, tools)
    return RecorderBuild(out_dir / LIBRARY_NAME, patches, tools.sanitizer_lib / SANITIZER_LIBRARY)


def _build_key(arch: str, tools: Toolchain) -> str:
    digest = hashlib.sha256(arch.encode())
    for name in SOURCES:
        digest.update((RECORDER_DIR / name).read_bytes())
    assert tools.sanitizer_lib and tools.sanitizer_include and tools.cuda_include
    used = (
        tools.compiler, tools.nvcc, tools.nvrtc, tools.ptxas,
        tools.sanitizer_lib / SANITIZER_LIBRARY,
        tools.sanitizer_include / "sanitizer.h",
        tools.cuda_include / "cuda.h",
    )  # fmt: skip
    for path in used:
        if path is not None and path.exists():
            info = path.stat()
            digest.update(f"{path.resolve()} {info.st_size} {info.st_mtime_ns}".encode())
    return digest.hexdigest()[:20]


def _fill(entry: pathlib.Path, arch: str, number: str, tools: Toolchain) -> None:
    built = (entry / LIBRARY_NAME, entry / _patches_name(arch, tools))
    if all(path.is_file() and path.stat().st_size > 0 for path in built):
        return
    # An entry that a crash left empty or half written gets built again.
    shutil.rmtree(entry, ignore_errors=True)
    entry.parent.mkdir(parents=True, exist_ok=True)
    # The build takes the name of the entry only when complete and on disk, so no
    # recording sees half of one.
    staging = pathlib.Path(tempfile.mkdtemp(prefix=".build-", dir=entry.parent))
    try:
        _compile(staging, arch, number, tools)
        for path in (*staging.iterdir(), staging):
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        try:
            staging.rename(entry)
        except OSError:
            # Another recording filled the entry first.
            if not all(path.is_file() for path in built):
                raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _compile(out_dir: pathlib.Path, arch: str, number: str, tools: Toolchain) -> RecorderBuild:
    assert tools.sanitizer_lib and tools.sanitizer_include and tools.cuda_include and tools.compiler
    library = out_dir / LIBRARY_NAME
    _run([
        str(tools.compiler), "-std=c++17", "-O2", "-fPIC", "-shared",
        f"-I{RECORDER_DIR}", f"-I{tools.sanitizer_include}", f"-I{tools.cuda_include}",
        str(RECORDER_DIR / "recorder.cpp"),
        f"-L{tools.sanitizer_lib}", "-lsanitizer-public",
        f"-Wl,-rpath,{tools.sanitizer_lib}", "-ldl",
        "-o", str(library),
    ])  # fmt: skip
    patches = _build_patches(out_dir, arch, number, tools)
    return RecorderBuild(library, patches, tools.sanitizer_lib / SANITIZER_LIBRARY)


def _patches_name(arch: str, tools: Toolchain) -> str:
    return f"patches-{arch}.fatbin" if tools.nvcc is not None else f"patches-{arch}.cubin"


def _build_patches(out_dir: pathlib.Path, arch: str, number: str, tools: Toolchain) -> pathlib.Path:
    source = RECORDER_DIR / "patches.cu"
    patches = out_dir / _patches_name(arch, tools)
    if tools.nvcc is not None:
        _run([
            str(tools.nvcc), "--fatbin", "--compile-as-tools-patch",
            f"-gencode=arch=compute_{number},code={arch}", f"-I{RECORDER_DIR}",
            "-o", str(patches), "-c", str(source),
        ])  # fmt: skip
        return patches
    tools.require("nvrtc", "ptxas")
    assert tools.nvrtc and tools.ptxas
    program = nvrtc.compile_cuda(
        source.read_text(),
        source.name,
        [
            f"--gpu-architecture=compute_{number}",
            "--relocatable-device-code=true",
            f"--include-path={RECORDER_DIR}",
            "-std=c++17",
        ],
        tools.nvrtc,
    )
    ptx = out_dir / f"patches-{arch}.ptx"
    ptx.write_bytes(program.ptx)
    # ptxas refuses --compile-only together with the tools-patch mode.
    _run(
        [
            str(tools.ptxas),
            f"-arch={arch}",
            "--compile-as-tools-patch",
            str(ptx),
            "-o",
            str(patches),
        ]
    )
    return patches


def _run(command: Sequence[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise BuildError(f"{command[0]} failed:\n{result.stderr or result.stdout}")
