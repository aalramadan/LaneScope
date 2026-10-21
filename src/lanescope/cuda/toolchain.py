import dataclasses
import glob
import importlib.util
import os
import pathlib
import shutil
from collections.abc import Mapping, Sequence

from lanescope.nsight.installer import cached_ncu

Path = pathlib.Path


class ToolchainError(RuntimeError):
    """A tool that a step needs is missing."""


@dataclasses.dataclass(frozen=True)
class Toolchain:
    """Paths of the CUDA tools, with None for a tool that was not found."""

    sanitizer_include: Path | None
    sanitizer_lib: Path | None
    cuda_include: Path | None
    ptxas: Path | None
    nvcc: Path | None
    nvrtc: Path | None
    ncu: Path | None
    nvdisasm: Path | None
    compiler: Path | None
    disassemblers: tuple[Path, ...] = ()

    def require(self, *names: str) -> None:
        """Raise ToolchainError naming every listed tool that is missing.

        A missing C++ compiler gets a remedy of its own, since neither a CUDA
        toolkit nor the NVIDIA wheels bring one.
        """
        missing = [name for name in names if getattr(self, name) is None]
        cuda = [name for name in missing if name != "compiler"]
        problems = []
        if cuda:
            problems.append(
                f"missing CUDA tools: {', '.join(cuda)}. Install a CUDA toolkit, "
                "or run pip install 'lanescope[gpu]' for every tool but ncu."
            )
        if "compiler" in missing:
            problems.append(
                "C++ compiler not found, neither g++ nor the command that CXX names. "
                "Install g++ with: sudo apt install g++"
            )
        if problems:
            raise ToolchainError(" ".join(problems))


def find(environ: Mapping[str, str] | None = None) -> Toolchain:
    """Return the tools of the first complete toolkit, or of the NVIDIA wheels.

    LANESCOPE_USE_WHEELS=1 takes the wheels even where a complete toolkit exists,
    such as a toolkit newer than the CUDA version of the driver.
    """
    env = os.environ if environ is None else environ
    path = env.get("PATH", "")
    roots = _toolkit_roots(env)
    cache = cache_folder(env)
    cached = cached_ncu(cache) if cache else None
    ncu = _first_file(
        ([cached] if cached else [])
        + [root / "bin" / "ncu" for root in roots]
        + _on_path("ncu", path)
        + [Path(p) for p in sorted(glob.glob("/opt/nvidia/nsight-compute/*/ncu"), reverse=True)]
    )
    disassemblers = tuple(
        dict.fromkeys(
            candidate
            for candidate in [d / "bin" / "nvdisasm" for d in _wheel_dirs("nvidia.cu13")]
            + [root / "bin" / "nvdisasm" for root in roots]
            + _on_path("nvdisasm", path)
            if candidate.is_file() and os.access(candidate, os.X_OK)
        )
    )
    nvdisasm = disassemblers[0] if disassemblers else None
    compiler = _first_file(_on_path(env.get("CXX", "g++"), path))
    use_wheels = env.get("LANESCOPE_USE_WHEELS") == "1"
    root = None if use_wheels else next((root for root in roots if _complete(root)), None)
    if root is not None:
        return Toolchain(
            sanitizer_include=root / "compute-sanitizer" / "include",
            sanitizer_lib=root / "compute-sanitizer",
            cuda_include=root / "include",
            ptxas=_first_file([root / "bin" / "ptxas"]),
            nvcc=root / "bin" / "nvcc",
            nvrtc=_first_file(
                [Path(p) for p in sorted(glob.glob(str(root / "lib64" / "libnvrtc.so*")))]
            ),
            ncu=ncu,
            nvdisasm=nvdisasm,
            compiler=compiler,
            disassemblers=disassemblers,
        )
    sanitizer = _wheel_dirs("nvidia.cuda_sanitizer_api")
    return Toolchain(
        sanitizer_include=_first_dir(
            [d / "include" for d in sanitizer]
            + [d / "compute-sanitizer" / "include" for d in sanitizer],
            "sanitizer.h",
        ),
        sanitizer_lib=_first_dir(
            [d / "lib" for d in sanitizer] + [d / "compute-sanitizer" for d in sanitizer],
            "libsanitizer-public.so",
        ),
        cuda_include=_first_dir(
            [d / "include" for d in _wheel_dirs("nvidia.cuda_runtime")], "cuda.h"
        ),
        ptxas=_first_file([d / "bin" / "ptxas" for d in _wheel_dirs("nvidia.cuda_nvcc")]),
        nvcc=None,
        nvrtc=_first_file(
            [
                Path(p)
                for d in _wheel_dirs("nvidia.cuda_nvrtc")
                for p in sorted(glob.glob(str(d / "lib" / "libnvrtc.so.*")))
            ]
        ),
        ncu=ncu,
        nvdisasm=nvdisasm,
        compiler=compiler,
        disassemblers=disassemblers,
    )


def cache_folder(environ: Mapping[str, str] | None = None) -> Path | None:
    """Return the folder for downloads such as Nsight Compute.

    The folder is LANESCOPE_CACHE when set, and ~/.cache/lanescope otherwise.
    """
    env = os.environ if environ is None else environ
    if env.get("LANESCOPE_CACHE"):
        return Path(env["LANESCOPE_CACHE"])
    return Path(env["HOME"]) / ".cache" / "lanescope" if env.get("HOME") else None


def _complete(root: Path) -> bool:
    return all(
        (root / part).is_file()
        for part in (
            "bin/nvcc",
            "include/cuda.h",
            "compute-sanitizer/include/sanitizer.h",
            "compute-sanitizer/libsanitizer-public.so",
        )
    )


def _toolkit_roots(env: Mapping[str, str]) -> list[Path]:
    names = [env.get("CUDA_HOME"), env.get("CUDA_PATH"), "/usr/local/cuda"]
    return [Path(name) for name in names if name]


def _wheel_dirs(package: str) -> list[Path]:
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    return [Path(location) for location in spec.submodule_search_locations]


def _on_path(name: str, path: str) -> list[Path]:
    found = shutil.which(name, path=path)
    return [Path(found)] if found else []


def _first_dir(candidates: Sequence[Path], marker: str) -> Path | None:
    return next((d for d in candidates if (d / marker).is_file()), None)


def _first_file(candidates: Sequence[Path]) -> Path | None:
    return next((f for f in candidates if f.is_file()), None)
