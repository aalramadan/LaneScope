import shutil
import subprocess

import pytest

from lanescope.cuda import nvrtc, toolchain
from lanescope.recorder import build

TOOLS = toolchain.find()
needs_compiler = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")
# The recorder reads the cubin of each module through this function of cubin.h.
PROGRAM = r"""
#include "cubin.h"

#include <fstream>
#include <iterator>
#include <vector>

int main(int argc, char** argv) {
    std::ifstream file(argv[1], std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
    return lanescope::NamesFunction(bytes.data(), bytes.size(), argv[2]) ? 0 : 1;
}
"""
KERNELS = """
namespace at { namespace native {
template <int N> __global__ void vectorized_elementwise_kernel(float* out) { out[threadIdx.x] = N; }
template __global__ void vectorized_elementwise_kernel<4>(float*);
} }
extern "C" __global__ void read_tile(float* out) { out[threadIdx.x] = 1.0f; }
__device__ float launch_counter[4];
"""


@pytest.fixture(scope="module")
def names_function(tmp_path_factory):
    """Return a check of whether a function of a cubin has a name that holds a text."""
    folder = tmp_path_factory.mktemp("cubin")
    source = folder / "names_function.cpp"
    source.write_text(PROGRAM)
    program = folder / "names_function"
    command = ["g++", "-std=c++17", f"-I{build.RECORDER_DIR}", str(source), "-o", str(program)]
    subprocess.run(command, check=True)

    def names(data: bytes, text: str) -> bool:
        cubin = folder / "module.cubin"
        cubin.write_bytes(data)
        return subprocess.run([str(program), str(cubin), text], check=False).returncode == 0

    return names


@needs_compiler
@pytest.mark.skipif(TOOLS.nvrtc is None, reason="NVRTC is needed")
def test_a_kernel_matches_by_any_part_of_its_mangled_name(names_function):
    assert TOOLS.nvrtc is not None
    cubin = nvrtc.compile_cuda(
        KERNELS, "kernels.cu", ["--gpu-architecture=sm_89"], TOOLS.nvrtc
    ).cubin
    assert cubin is not None
    assert names_function(cubin, "vectorized_elementwise")
    assert names_function(cubin, "read_tile")
    assert not names_function(cubin, "absent_kernel")
    # A variable is no function, even where its name sits in the cubin.
    assert b"launch_counter" in cubin
    assert not names_function(cubin, "launch_counter")
    assert names_function(cubin[: len(cubin) // 2], "absent_kernel")


@needs_compiler
@pytest.mark.parametrize("data", [b"", b"\x7fELF\x02" + bytes(64), b"no cubin" * 16])
def test_a_file_that_is_no_cubin_keeps_its_module_patched(names_function, data):
    assert names_function(data, "any_kernel")
