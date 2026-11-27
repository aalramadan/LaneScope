import importlib.util
import os
import pathlib
import py_compile
import subprocess
import sys
import sysconfig

import pytest

from lanescope.cuda import nvrtc, toolchain
from lanescope.cuda.driver import Driver, DriverError

EXAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "triton_matmul.py"
CONFLICT_KINDS = EXAMPLE.parent / "conflict_kinds.py"
NVRTC = toolchain.find_toolchain().nvrtc


def _python_headers() -> bool:
    folders = [sysconfig.get_paths()["include"], *os.environ.get("CPATH", "").split(":")]
    return any(pathlib.Path(folder, "Python.h").is_file() for folder in folders if folder)


def _gpu_with_triton() -> bool:
    if importlib.util.find_spec("triton") is None or importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


def test_the_triton_example_is_valid_python(tmp_path):
    py_compile.compile(str(EXAMPLE), cfile=str(tmp_path / "example.pyc"), doraise=True)


@pytest.mark.skipif(
    not _python_headers() or not _gpu_with_triton(),
    reason="Triton, PyTorch with a GPU, and the Python headers are needed",
)
def test_the_triton_example_matches_the_product_of_pytorch():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE), "--size", "128"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "the 128 x 128 product matches PyTorch" in result.stdout


def _has_gpu() -> bool:
    try:
        Driver().close()
    except DriverError:
        return False
    return True


def test_the_conflict_kinds_example_is_valid_python(tmp_path):
    py_compile.compile(str(CONFLICT_KINDS), cfile=str(tmp_path / "kinds.pyc"), doraise=True)


@pytest.mark.skipif(NVRTC is None, reason="NVRTC is needed")
def test_the_kernels_of_the_conflict_kinds_example_compile_with_line_tables():
    source = CONFLICT_KINDS.with_suffix(".cu")
    assert NVRTC is not None
    options = ["--gpu-architecture=sm_89", "-lineinfo"]
    program = nvrtc.compile_cuda(source.read_text(), str(source), options, NVRTC)
    assert program.cubin


@pytest.mark.skipif(NVRTC is None or not _has_gpu(), reason="NVRTC and a GPU are needed")
def test_the_conflict_kinds_example_launches_its_eight_kernels():
    result = subprocess.run(
        [sys.executable, str(CONFLICT_KINDS)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "launched 8 kernels"
