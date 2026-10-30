import importlib.util
import os
import pathlib
import py_compile
import subprocess
import sys
import sysconfig

import pytest

EXAMPLE = pathlib.Path(__file__).parent.parent / "examples" / "triton_matmul.py"


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
