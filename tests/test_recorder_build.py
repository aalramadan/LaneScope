import pytest

from lanescope.cuda import toolchain
from lanescope.recorder import build

TOOLS = toolchain.find()
needs_tools = pytest.mark.skipif(
    TOOLS.sanitizer_include is None or TOOLS.compiler is None or TOOLS.nvrtc is None,
    reason="the Sanitizer API, a C++ compiler, and NVRTC are needed",
)


@needs_tools
def test_recorder_library_and_patches_build_for_sm_89(tmp_path):
    result = build.build_recorder(tmp_path, "sm_89", TOOLS)
    assert result.library.stat().st_size > 0
    assert b"LaneScopeSharedAccess" in result.patches.read_bytes()
    assert result.sanitizer_library.name == "libsanitizer-public.so"


@pytest.mark.parametrize(("arch", "digits"), [("sm_89", "89"), ("sm_120", "120"), ("sm_90a", "90")])
def test_check_arch_returns_the_compute_capability(arch, digits):
    assert build.check_arch(arch) == digits


@pytest.mark.parametrize("arch", ["89", "compute_89", "sm_8", "sm89"])
def test_check_arch_rejects_other_names(arch):
    with pytest.raises(ValueError, match="sm_89"):
        build.check_arch(arch)


EMPTY = toolchain.Toolchain(None, None, None, None, None, None, None, None, None)


def test_missing_tools_are_named():
    with pytest.raises(toolchain.ToolchainError, match="sanitizer_include, cuda_include"):
        EMPTY.require("sanitizer_include", "cuda_include")


def test_a_missing_compiler_gets_the_command_that_installs_it():
    with pytest.raises(toolchain.ToolchainError) as error:
        EMPTY.require("compiler")
    assert "sudo apt install g++" in str(error.value)
    assert "CUDA" not in str(error.value)


def test_a_missing_compiler_is_named_apart_from_missing_cuda_tools():
    with pytest.raises(toolchain.ToolchainError) as error:
        EMPTY.require("sanitizer_include", "compiler")
    assert "missing CUDA tools: sanitizer_include." in str(error.value)
    assert "sudo apt install g++" in str(error.value)


def make_toolkit(root):
    for part in (
        "bin/nvcc",
        "bin/ptxas",
        "include/cuda.h",
        "compute-sanitizer/include/sanitizer.h",
        "compute-sanitizer/libsanitizer-public.so",
        "lib64/libnvrtc.so.12",
    ):
        (root / part).parent.mkdir(parents=True, exist_ok=True)
        (root / part).write_text("")


def test_find_takes_every_tool_from_one_complete_toolkit(tmp_path):
    make_toolkit(tmp_path)
    found = toolchain.find({"CUDA_HOME": str(tmp_path), "PATH": ""})
    assert found.nvcc == tmp_path / "bin" / "nvcc"
    assert found.sanitizer_lib == tmp_path / "compute-sanitizer"
    assert found.nvrtc == tmp_path / "lib64" / "libnvrtc.so.12"


def make_wheels(root):
    """Lay out the files of the CUDA wheels and return the folder of each package."""
    folders = {
        "nvidia.cuda_sanitizer_api": root / "sanitizer",
        "nvidia.cuda_runtime": root / "runtime",
        "nvidia.cuda_nvcc": root / "nvcc",
        "nvidia.cuda_nvrtc": root / "nvrtc",
    }
    for package, part in (
        ("nvidia.cuda_sanitizer_api", "include/sanitizer.h"),
        ("nvidia.cuda_sanitizer_api", "lib/libsanitizer-public.so"),
        ("nvidia.cuda_runtime", "include/cuda.h"),
        ("nvidia.cuda_nvcc", "bin/ptxas"),
        ("nvidia.cuda_nvrtc", "lib/libnvrtc.so.12"),
    ):
        (folders[package] / part).parent.mkdir(parents=True, exist_ok=True)
        (folders[package] / part).write_text("")
    return folders


def test_find_takes_the_wheels_over_a_complete_toolkit_on_request(tmp_path, monkeypatch):
    make_toolkit(tmp_path / "toolkit")
    wheels = make_wheels(tmp_path / "wheels")
    monkeypatch.setattr(
        toolchain, "_wheel_dirs", lambda package: [wheels[package]] if package in wheels else []
    )
    environ = {"CUDA_HOME": str(tmp_path / "toolkit"), "PATH": ""}
    assert toolchain.find(environ).nvcc == tmp_path / "toolkit" / "bin" / "nvcc"
    found = toolchain.find(environ | {"LANESCOPE_USE_WHEELS": "1"})
    assert found.nvcc is None
    assert found.sanitizer_lib == wheels["nvidia.cuda_sanitizer_api"] / "lib"
    assert found.nvrtc == wheels["nvidia.cuda_nvrtc"] / "lib" / "libnvrtc.so.12"


def test_find_skips_an_incomplete_toolkit_and_an_nvcc_on_the_path(tmp_path):
    (tmp_path / "toolkit" / "bin").mkdir(parents=True)
    (tmp_path / "toolkit" / "bin" / "nvcc").write_text("")
    (tmp_path / "path").mkdir()
    stray = tmp_path / "path" / "nvcc"
    stray.write_text("")
    stray.chmod(0o755)
    found = toolchain.find({"CUDA_HOME": str(tmp_path / "toolkit"), "PATH": str(tmp_path / "path")})
    assert found.nvcc is None


def test_find_takes_nvdisasm_from_the_wheel_before_a_toolkit(tmp_path, monkeypatch):
    make_toolkit(tmp_path / "toolkit")
    (tmp_path / "toolkit" / "bin" / "nvdisasm").write_text("")
    wheel = tmp_path / "wheel"
    (wheel / "bin").mkdir(parents=True)
    (wheel / "bin" / "nvdisasm").write_text("")
    real = toolchain._wheel_dirs
    environ = {"CUDA_HOME": str(tmp_path / "toolkit"), "PATH": ""}

    def wheels(present):
        return lambda package: (
            ([wheel] if present else []) if package == "nvidia.cu13" else real(package)
        )

    for copy in (tmp_path / "toolkit" / "bin" / "nvdisasm", wheel / "bin" / "nvdisasm"):
        copy.chmod(0o755)
    monkeypatch.setattr(toolchain, "_wheel_dirs", wheels(True))
    found = toolchain.find(environ)
    assert found.nvdisasm == wheel / "bin" / "nvdisasm"
    assert found.disassemblers == (
        wheel / "bin" / "nvdisasm",
        tmp_path / "toolkit" / "bin" / "nvdisasm",
    )
    monkeypatch.setattr(toolchain, "_wheel_dirs", wheels(False))
    assert toolchain.find(environ).nvdisasm == tmp_path / "toolkit" / "bin" / "nvdisasm"
    (tmp_path / "toolkit" / "bin" / "nvdisasm").chmod(0o644)
    assert toolchain.find(environ).disassemblers == ()
