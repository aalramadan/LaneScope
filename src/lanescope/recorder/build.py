import dataclasses
import pathlib
import re
import subprocess
from collections.abc import Sequence

from lanescope.cuda import nvrtc
from lanescope.cuda.toolchain import Toolchain, find

RECORDER_DIR = pathlib.Path(__file__).parent
LIBRARY_NAME = "liblanescope_recorder.so"


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
    out_dir: pathlib.Path, arch: str, tools: Toolchain | None = None
) -> RecorderBuild:
    """Build the recorder library and the patches for `arch` into `out_dir`."""
    number = check_arch(arch)
    tools = tools or find()
    tools.require("sanitizer_include", "sanitizer_lib", "cuda_include", "compiler")
    assert tools.sanitizer_lib and tools.sanitizer_include and tools.cuda_include and tools.compiler
    out_dir.mkdir(parents=True, exist_ok=True)
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
    return RecorderBuild(library, patches, tools.sanitizer_lib / "libsanitizer-public.so")


def _build_patches(out_dir: pathlib.Path, arch: str, number: str, tools: Toolchain) -> pathlib.Path:
    source = RECORDER_DIR / "patches.cu"
    if tools.nvcc is not None:
        patches = out_dir / f"patches-{arch}.fatbin"
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
    patches = out_dir / f"patches-{arch}.cubin"
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
