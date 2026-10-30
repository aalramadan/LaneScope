import pathlib
import re
import subprocess

import pytest

from lanescope.cuda import toolchain
from lanescope.cuda.driver import Driver, DriverError
from lanescope.model import shared_memory
from lanescope.nsight import calibration
from lanescope.trace import Kind, Space

FORUM_WAVEFRONTS = [2, 2, 4, 4, 4, 4, 8, 4, 8]
TOOLS = toolchain.find()
MEMORY_OPCODES = {"LD", "ST", "LDG", "STG", "LDS", "STS", "LDL", "STL", "ATOM", "ATOMG", "ATOMS"}
MEMORY_OPCODES |= {"RED", "LDGSTS", "LDSM"}
OPCODE_OF = {
    (Space.SHARED, Kind.LOAD): "LDS",
    (Space.SHARED, Kind.STORE): "STS",
    (Space.GLOBAL, Kind.LOAD): "LDG",
    (Space.GLOBAL, Kind.STORE): "STG",
}
WIDTH_MODIFIERS = {1: {"U8"}, 2: {"U16"}, 4: set(), 8: {"64"}, 16: {"128"}}


def _sass_opcodes(cubin: pathlib.Path) -> dict[str, list[str]]:
    text = subprocess.run(
        [str(TOOLS.nvdisasm), "-c", str(cubin)], capture_output=True, text=True, check=True
    ).stdout
    functions: dict[str, list[str]] = {}
    current: list[str] = []
    for line in text.splitlines():
        if header := re.match(r"\.text\.(\w+):", line):
            current = functions.setdefault(header.group(1), [])
        elif instruction := re.match(r"\s+/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][\w.]*)", line):
            current.append(instruction.group(1))
    return functions


def test_case_names_are_unique():
    names = [case.name for case in calibration.cases()]
    assert len(names) == len(set(names))


def test_kernel_source_defines_exactly_the_kernels_that_run():
    defined = set(re.findall(r"__global__ void (\w+)\(", calibration.kernel_source()))
    launched = {case.kernel for case in calibration.cases()} | set(calibration.TRANSPOSE_KERNELS)
    assert defined == launched


def test_forum_cases_keep_the_wavefronts_measured_on_h100():
    forum = [
        case for case in calibration.cases() if "forum" in case.name and case.kind.name == "LOAD"
    ]
    found = [
        shared_memory.analyze(case.addresses(), case.width, arch="sm_90").wavefronts
        for case in forum
    ]
    assert found == FORUM_WAVEFRONTS


def test_masked_lanes_have_no_address():
    case = next(case for case in calibration.cases() if case.name == "shared-load-4-half-warp")
    assert case.addresses()[15] == 128 * 15
    assert case.addresses()[16] is None


def test_shared_cases_fit_the_shared_buffer():
    for case in calibration.cases():
        if case.space == Space.SHARED:
            assert max(case.offsets) + case.width <= calibration.SHARED_BYTES


@pytest.mark.skipif(TOOLS.nvrtc is None, reason="NVRTC is needed")
def test_every_kernel_compiles_for_sm_89():
    cubin = calibration.compile_kernels("sm_89", TOOLS)
    names = {name.decode() for name in re.findall(rb"\.text\.([A-Za-z0-9_]+)", cubin)}
    expected = {case.kernel for case in calibration.cases()} | set(calibration.TRANSPOSE_KERNELS)
    assert expected <= names


def test_kernel_source_uses_one_vector_load_for_16_bytes():
    source = calibration.kernel_source()
    body = source.split("void shared_load_16(")[1].split("extern")[0]
    assert body.count("ld.volatile.shared.v4.u32") == 1


@pytest.mark.skipif(
    TOOLS.nvrtc is None or TOOLS.nvdisasm is None, reason="NVRTC and nvdisasm are needed"
)
def test_each_case_kernel_keeps_one_weak_memory_instruction_of_its_shape(tmp_path):
    cubin = tmp_path / "bench.cubin"
    cubin.write_bytes(calibration.compile_kernels("sm_89", TOOLS))
    functions = _sass_opcodes(cubin)
    for case in calibration.cases():
        memory = [op for op in functions[case.kernel] if op.split(".")[0] in MEMORY_OPCODES]
        assert [op.split(".")[0] for op in memory] == [OPCODE_OF[case.space, case.kind]], case.name
        modifiers = set(memory[0].split(".")[1:])
        assert modifiers & {"U8", "S8", "U16", "S16", "64", "128"} == WIDTH_MODIFIERS[case.width]
        assert "STRONG" not in modifiers, case.name


@pytest.mark.skipif(
    TOOLS.nvrtc is None or TOOLS.nvdisasm is None, reason="NVRTC and nvdisasm are needed"
)
def test_kernels_built_from_a_source_file_name_that_file_in_their_line_information(tmp_path):
    source = tmp_path / "bench.cu"
    cubin = tmp_path / "bench.cubin"
    cubin.write_bytes(calibration.compile_kernels("sm_89", TOOLS, source))
    assert source.read_text() == calibration.kernel_source()
    listing = subprocess.run(
        [str(TOOLS.nvdisasm), "--print-line-info", str(cubin)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert f'File "{source}"' in listing


def test_random_cases_repeat_for_one_seed_and_change_with_another():
    assert calibration.random_cases(20, 1) == calibration.random_cases(20, 1)
    assert calibration.random_cases(20, 1) != calibration.random_cases(20, 2)


def test_random_cases_fit_the_calibration_kernels():
    kernels = {case.kernel for case in calibration.cases()}
    for case in calibration.random_cases(300, 5):
        limit = calibration.SHARED_BYTES if case.space == Space.SHARED else calibration.GLOBAL_BYTES
        assert case.kernel in kernels
        assert case.mask
        assert all(offset % case.width == 0 and offset < limit for offset in case.offsets)


def test_launches_put_the_random_cases_after_the_fixed_ones():
    assert calibration.launches(3, 4) == calibration.cases() + calibration.random_cases(3, 4)


def _has_gpu() -> bool:
    try:
        Driver().close()
    except DriverError:
        return False
    return True


def test_transpose_bandwidth_counts_one_read_and_one_write_per_element():
    time = calibration.TransposeTime("transpose_tile", milliseconds=1.0, size=1024)
    assert time.gigabytes_per_second == pytest.approx(2 * 4 * 1024 * 1024 / 1e-3 / 1e9)


def test_timing_rejects_a_matrix_size_that_is_not_a_multiple_of_32(tmp_path):
    with pytest.raises(ValueError, match="multiple of 32"):
        calibration.time_transposes(tmp_path / "bench.cubin", size=100, repeats=1)


@pytest.mark.skipif(TOOLS.nvrtc is None or not _has_gpu(), reason="NVRTC and a GPU are needed")
def test_each_transpose_gets_a_positive_time_on_the_local_gpu(tmp_path):
    driver = Driver()
    arch = driver.arch()
    driver.close()
    cubin = tmp_path / "bench.cubin"
    cubin.write_bytes(calibration.compile_kernels(arch, TOOLS))
    times = calibration.time_transposes(cubin, size=64, repeats=2)
    assert [time.kernel for time in times] == list(calibration.TRANSPOSE_KERNELS)
    assert all(time.milliseconds > 0 for time in times)
