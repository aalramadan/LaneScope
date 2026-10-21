import ctypes
import dataclasses
import pathlib
from collections.abc import Sequence

_SUCCESS = 0


class CompileError(RuntimeError):
    """NVRTC rejected a program, with its log as the message."""


@dataclasses.dataclass(frozen=True)
class Program:
    """Output of one NVRTC compilation."""

    ptx: bytes
    cubin: bytes | None
    log: str


def compile_cuda(source: str, name: str, options: Sequence[str], library: pathlib.Path) -> Program:
    """Compile CUDA C++ source with the NVRTC library at the given path.

    The cubin is present only when the options name a real architecture, such as
    --gpu-architecture=sm_89.
    """
    nvrtc = _load(library)
    program = ctypes.c_void_p()
    _check(
        nvrtc,
        nvrtc.nvrtcCreateProgram(
            ctypes.byref(program), source.encode(), name.encode(), 0, None, None
        ),
    )
    try:
        encoded = [option.encode() for option in options]
        array = (ctypes.c_char_p * len(encoded))(*encoded)
        status = nvrtc.nvrtcCompileProgram(program, len(encoded), array)
        log = (
            _read(nvrtc, program, "nvrtcGetProgramLogSize", "nvrtcGetProgramLog")
            .decode(errors="replace")
            .rstrip("\x00")
        )
        if status != _SUCCESS:
            raise CompileError(f"NVRTC could not compile {name}:\n{log}")
        ptx = _read(nvrtc, program, "nvrtcGetPTXSize", "nvrtcGetPTX").rstrip(b"\x00")
        cubin = None
        if any(option.startswith(("--gpu-architecture=sm_", "-arch=sm_")) for option in options):
            cubin = _read(nvrtc, program, "nvrtcGetCUBINSize", "nvrtcGetCUBIN")
        return Program(ptx, cubin, log)
    finally:
        nvrtc.nvrtcDestroyProgram(ctypes.byref(program))


def version(library: pathlib.Path) -> tuple[int, int]:
    """Return the CUDA version of an NVRTC library, which is the version of its toolkit."""
    nvrtc = _load(library)
    major, minor = ctypes.c_int(), ctypes.c_int()
    _check(nvrtc, nvrtc.nvrtcVersion(ctypes.byref(major), ctypes.byref(minor)))
    return major.value, minor.value


def _load(library: pathlib.Path) -> ctypes.CDLL:
    try:
        nvrtc = ctypes.CDLL(str(library))
    except OSError as error:
        raise CompileError(f"cannot load NVRTC from {library}: {error}") from error
    nvrtc.nvrtcGetErrorString.restype = ctypes.c_char_p
    return nvrtc


def _check(nvrtc: ctypes.CDLL, status: int) -> None:
    if status != _SUCCESS:
        raise CompileError(f"NVRTC error: {nvrtc.nvrtcGetErrorString(status).decode()}")


def _read(nvrtc: ctypes.CDLL, program: ctypes.c_void_p, size_name: str, read_name: str) -> bytes:
    size = ctypes.c_size_t()
    _check(nvrtc, getattr(nvrtc, size_name)(program, ctypes.byref(size)))
    buffer = ctypes.create_string_buffer(size.value)
    _check(nvrtc, getattr(nvrtc, read_name)(program, buffer))
    return buffer.raw
