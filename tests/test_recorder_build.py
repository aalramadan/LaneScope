import dataclasses
import os
import re
import shutil

import pytest

from lanescope.cuda import nvrtc, toolchain
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


@needs_tools
@pytest.mark.parametrize(("digits", "splits"), [("61", False), ("89", True)])
def test_the_patches_split_records_by_pc_from_sm_70_on(digits, splits):
    assert TOOLS.nvrtc is not None
    options = [
        f"--gpu-architecture=compute_{digits}",
        "--relocatable-device-code=true",
        f"--include-path={build.RECORDER_DIR}",
        "-std=c++17",
    ]
    source = (build.RECORDER_DIR / "patches.cu").read_text()
    program = nvrtc.compile_cuda(source, "patches.cu", options, TOOLS.nvrtc)
    assert (b"match.any.sync.b64" in program.ptx) == splits


@needs_tools
def test_a_cached_build_serves_a_new_folder_without_a_compile(tmp_path, monkeypatch):
    first = build.build_recorder(tmp_path / "first", "sm_89", TOOLS, cache=tmp_path / "cache")
    monkeypatch.setattr(build, "_compile", lambda *args: pytest.fail("the build ran again"))
    second = build.build_recorder(tmp_path / "second", "sm_89", TOOLS, cache=tmp_path / "cache")
    assert second.library == tmp_path / "second" / build.LIBRARY_NAME
    assert second.library.read_bytes() == first.library.read_bytes()
    assert second.patches.read_bytes() == first.patches.read_bytes()


@needs_tools
def test_a_cache_that_cannot_take_the_build_leaves_the_build_to_the_folder(tmp_path):
    blocked = tmp_path / "cache"
    blocked.write_text("a file where the cache folder belongs")
    result = build.build_recorder(tmp_path / "out", "sm_89", TOOLS, cache=blocked)
    assert result.library.parent == tmp_path / "out"
    assert result.library.stat().st_size > 0


def stand_in_tools(folder):
    """Return a toolchain of empty files, for builds whose compile a test replaces."""
    compiler = folder / "g++"
    compiler.write_text("one compiler")
    folders = dict.fromkeys(("sanitizer_include", "sanitizer_lib", "cuda_include"), folder)
    return dataclasses.replace(EMPTY, compiler=compiler, **folders)


def stand_in_compile(text, fails_under=None):
    """Return a compile that writes the library and the patches with the given text."""

    def compile_files(out_dir, arch, number, tools):
        if fails_under is not None and fails_under in out_dir.parents:
            raise build.BuildError("g++ failed:\nNo space left on device")
        library, patches = out_dir / build.LIBRARY_NAME, out_dir / f"patches-{arch}.cubin"
        library.write_text(text)
        patches.write_text(text)
        return build.RecorderBuild(library, patches, tools.sanitizer_lib / build.SANITIZER_LIBRARY)

    return compile_files


def test_the_build_key_changes_with_the_architecture_and_the_tools(tmp_path):
    tools = stand_in_tools(tmp_path)
    key = build._build_key("sm_89", tools)
    assert build._build_key("sm_89", tools) == key
    assert build._build_key("sm_90", tools) != key
    assert tools.compiler is not None
    tools.compiler.write_text("another compiler")
    assert build._build_key("sm_89", tools) != key


def test_the_build_key_changes_with_every_source_and_local_include(tmp_path, monkeypatch):
    seen, waiting = set(), ["recorder.cpp", "patches.cu"]
    while waiting:
        name = waiting.pop()
        seen.add(name)
        text = (build.RECORDER_DIR / name).read_text()
        waiting += [found for found in re.findall(r'#include "([^"]+)"', text) if found not in seen]
    assert seen <= set(build.SOURCES)
    for name in build.SOURCES:
        shutil.copy(build.RECORDER_DIR / name, tmp_path / name)
    monkeypatch.setattr(build, "RECORDER_DIR", tmp_path)
    tools = stand_in_tools(tmp_path)
    key = build._build_key("sm_89", tools)
    for name in build.SOURCES:
        (tmp_path / name).write_text((tmp_path / name).read_text() + "\n")
        assert build._build_key("sm_89", tools) != key
        key = build._build_key("sm_89", tools)


def test_a_cache_entry_that_another_recording_fills_first_serves_the_build(tmp_path, monkeypatch):
    tools, cache = stand_in_tools(tmp_path), tmp_path / "cache"
    entry = cache / "recorder" / build._build_key("sm_89", tools)
    mine = stand_in_compile("mine")

    def racing_compile(out_dir, arch, number, tools):
        stand_in_compile("theirs")(entry.parent / "theirs", arch, number, tools)
        (entry.parent / "theirs").rename(entry)
        return mine(out_dir, arch, number, tools)

    (entry.parent / "theirs").mkdir(parents=True)
    monkeypatch.setattr(build, "_compile", racing_compile)
    result = build.build_recorder(tmp_path / "out", "sm_89", tools, cache=cache)
    assert result.library.read_text() == "theirs"
    assert sorted(path.name for path in entry.iterdir()) == [
        "liblanescope_recorder.so",
        "patches-sm_89.cubin",
    ]
    assert [path.name for path in entry.parent.iterdir()] == [entry.name]


def test_a_failed_compile_leaves_neither_an_entry_nor_its_staging_folder(tmp_path, monkeypatch):
    tools, cache = stand_in_tools(tmp_path), tmp_path / "cache"
    monkeypatch.setattr(build, "_compile", stand_in_compile("", fails_under=tmp_path))
    with pytest.raises(build.BuildError, match="No space left"):
        build.build_recorder(tmp_path / "out", "sm_89", tools, cache=cache)
    assert list((cache / "recorder").iterdir()) == []


def test_a_compile_that_fails_in_the_cache_falls_back_to_the_folder(tmp_path, monkeypatch):
    tools, cache = stand_in_tools(tmp_path), tmp_path / "cache"
    monkeypatch.setattr(build, "_compile", stand_in_compile("local", fails_under=cache))
    result = build.build_recorder(tmp_path / "out", "sm_89", tools, cache=cache)
    assert result.library == tmp_path / "out" / build.LIBRARY_NAME
    assert result.library.read_text() == "local"
    assert list((cache / "recorder").iterdir()) == []


def test_a_cache_entry_with_an_empty_library_is_built_again(tmp_path, monkeypatch):
    tools, cache = stand_in_tools(tmp_path), tmp_path / "cache"
    entry = cache / "recorder" / build._build_key("sm_89", tools)
    entry.mkdir(parents=True)
    (entry / build.LIBRARY_NAME).write_text("")
    monkeypatch.setattr(build, "_compile", stand_in_compile("fresh"))
    result = build.build_recorder(tmp_path / "out", "sm_89", tools, cache=cache)
    assert result.library.read_text() == "fresh"
    assert result.patches.read_text() == "fresh"
    assert (entry / build.LIBRARY_NAME).read_text() == "fresh"


def test_a_build_from_the_cache_puts_a_new_file_in_place_of_the_library(tmp_path, monkeypatch):
    tools, cache, out = stand_in_tools(tmp_path), tmp_path / "cache", tmp_path / "out"
    monkeypatch.setattr(build, "_compile", stand_in_compile("library"))
    first = build.build_recorder(out, "sm_89", tools, cache=cache)
    with open(first.library) as mapped:
        inode = os.fstat(mapped.fileno()).st_ino
        second = build.build_recorder(out, "sm_89", tools, cache=cache)
        assert second.library.stat().st_ino != inode
        assert mapped.read() == "library"
    assert sorted(path.name for path in out.iterdir()) == [
        "liblanescope_recorder.so",
        "patches-sm_89.cubin",
    ]


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
