import ctypes
import json
import os
import shutil
import struct
import subprocess
import sys

import numpy as np
import pytest

from lanescope.cuda import nvrtc, toolchain
from lanescope.cuda.driver import Driver, DriverError
from lanescope.model import counters
from lanescope.recorder import capture
from lanescope.recorder.build import RECORDER_DIR, RecorderBuild, build_recorder
from lanescope.trace import WARP_SIZE, Kind, Space, Trace

BASE_PC = 0x7F00_0000_1000
# A library that writes its name to loaded.txt in LANESCOPE_OUT when the loader maps it.
PROBE_LIBRARY = r"""
#include <cstdio>
#include <cstdlib>
#include <string>

__attribute__((constructor)) static void mark_loaded() {
    if (const char* out = std::getenv("LANESCOPE_OUT")) {
        if (FILE* file = std::fopen((std::string(out) + "/loaded.txt").c_str(), "a")) {
            std::fputs(NAME "\n", file);
            std::fclose(file);
        }
    }
}
"""
needs_compiler = pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")


def write_launch(folder, index, rows, **info):
    records = np.zeros(len(rows), dtype=capture.RECORD_DTYPE)
    for record, row in zip(records, rows, strict=True):
        for key, value in row.items():
            record[key] = value
    (folder / f"launch-{index}.bin").write_bytes(records.tobytes())
    description = {
        "format": 1,
        "kernel": "k",
        "launch": index,
        "function_pc": BASE_PC,
        "arch": "sm_89",
        "grid": [1, 1, 1],
        "block": [32, 1, 1],
        "records": len(rows),
        "dropped": 0,
        "record_bytes": 280,
    }
    description.update(info)
    (folder / f"launch-{index}.json").write_text(json.dumps(description))
    return folder / f"launch-{index}.json"


def row(pc_offset, addresses, width=4, space=Space.SHARED, kind: int = Kind.LOAD, mask=0xFFFFFFFF):
    return {
        "pc": BASE_PC + pc_offset,
        "addr": addresses,
        "width": width,
        "space": int(space),
        "kind": int(kind),
        "mask": mask,
    }


def test_record_layout_is_280_bytes():
    assert capture.RECORD_DTYPE.itemsize == 280


def test_import_turns_records_into_a_trace_with_kernel_relative_pcs(tmp_path):
    write_launch(tmp_path, 0, [row(0x120, [128 * lane for lane in range(WARP_SIZE)])])
    (path,) = capture.import_recording(tmp_path)
    trace = capture.Trace.load(path)
    assert int(trace.pc[0]) == 0x120
    assert trace.metadata["arch"] == "sm_89"
    (counter,) = counters.count(trace)
    assert (counter.achieved, counter.conflicts) == (32, 31)


def test_atomics_reach_the_trace_as_atomic_accesses(tmp_path):
    info = write_launch(
        tmp_path,
        0,
        [row(0x10, [4 * lane for lane in range(WARP_SIZE)]), row(0x20, [0] * WARP_SIZE, kind=2)],
    )
    trace = capture.to_trace(*capture.load_launch(info))
    assert [Kind(int(kind)) for kind in trace.kind] == [Kind.LOAD, Kind.ATOMIC]
    assert trace.metadata["skipped"] == 0


def test_inactive_lanes_are_cleared(tmp_path):
    addresses = [4 * lane for lane in range(WARP_SIZE)]
    info = write_launch(tmp_path, 0, [row(0x10, addresses, mask=0x0000FFFF)])
    trace = capture.to_trace(*capture.load_launch(info))
    assert trace.lane_addresses(0)[16] is None
    assert int(trace.addr[0, 16]) == 0


def test_launch_files_are_sorted_by_launch_number(tmp_path):
    for index in (10, 2, 1):
        write_launch(tmp_path, index, [row(0x10, [4 * lane for lane in range(WARP_SIZE)])])
    assert [path.stem for path in capture.launch_files(tmp_path)] == [
        "launch-1",
        "launch-2",
        "launch-10",
    ]


def test_import_skips_launch_files_whose_name_holds_no_launch_number(tmp_path):
    write_launch(tmp_path, 0, [row(0x10, [4 * lane for lane in range(WARP_SIZE)])])
    for name in ("launch-x.json", "launch-1.orig.json"):
        (tmp_path / name).write_text("{}")
    assert [path.name for path in capture.launch_files(tmp_path)] == ["launch-0.json"]
    assert [path.name for path in capture.import_recording(tmp_path)] == ["trace-0.npz"]


def test_import_keeps_the_launch_order_of_many_launches(tmp_path):
    lanes = [4 * lane for lane in range(WARP_SIZE)]
    for index in range(12):
        write_launch(tmp_path, index, [row(0x10 * (index + 1), lanes)] * (index + 1))
    paths = capture.import_recording(tmp_path)
    assert [path.name for path in paths] == [f"trace-{index}.npz" for index in range(12)]
    for index, path in enumerate(paths):
        trace = Trace.load(path)
        assert len(trace) == index + 1
        assert set(trace.pc.tolist()) == {0x10 * (index + 1)}


def test_import_lists_only_the_kernels_that_ran_in_each_module(tmp_path, monkeypatch):
    lanes = [4 * lane for lane in range(WARP_SIZE)]
    for index, (module, kernel) in enumerate([(0, "a"), (0, "b"), (1, "c"), (0, "a")]):
        write_launch(tmp_path, index, [row(0x10, lanes)], module=module, kernel=kernel)
    for module in (0, 1):
        (tmp_path / f"module-{module}.cubin").write_bytes(b"cubin")
    listed = {}

    def listing(disassemblers, cubin, kernels=()):
        listed[cubin.name] = set(kernels)
        line = '\t//## File "/src/k.cu", line 3\n        /*0010*/ LDS R0, [R1] ;\n'
        return "".join(f".text.{name}:\n{line}" for name in kernels)

    monkeypatch.setattr(capture.line_info, "module_listing", listing)
    paths = capture.import_recording(tmp_path, [tmp_path / "nvdisasm"])
    assert listed == {"module-0.cubin": {"a", "b"}, "module-1.cubin": {"c"}}
    assert [Trace.load(path).metadata["lines"] for path in paths] == [{"0x10": 3}] * 4


def test_import_takes_one_launch_at_a_time_when_records_fill_its_memory(tmp_path, monkeypatch):
    paths = [write_launch(tmp_path, index, [row(0x10, [0] * WARP_SIZE)]) for index in range(4)]
    assert capture._workers(paths) == min(capture.IMPORT_THREADS, os.cpu_count() or 1)
    monkeypatch.setattr(capture, "IMPORT_BYTES", capture.RECORD_DTYPE.itemsize)
    assert capture._workers(paths) == 1


def test_record_replaces_the_files_of_an_earlier_recording(tmp_path):
    for index in (0, 5):
        write_launch(tmp_path, index, [row(0x10, [4 * lane for lane in range(WARP_SIZE)])])
    (tmp_path / "module-5.cubin").write_bytes(b"old module")
    capture.import_recording(tmp_path)
    (tmp_path / "notes.txt").write_text("keep")
    (tmp_path / "launch-1.orig.json").write_text("{}")
    # The run stands in for a recorder that writes the description of one launch.
    one_launch = (
        "import os, pathlib; "
        "pathlib.Path(os.environ['LANESCOPE_OUT'], 'launch-0.json').write_text('new')"
    )
    build = RecorderBuild(
        tmp_path / "build" / "lib.so", tmp_path / "build" / "patches.cubin", tmp_path / "libsan.so"
    )
    assert capture.record([sys.executable, "-c", one_launch], build, tmp_path) == 0
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "launch-0.json",
        "launch-1.orig.json",
        "notes.txt",
    ]
    assert (tmp_path / "launch-0.json").read_text() == "new"


def build_probe(source, library, name):
    library.parent.mkdir(parents=True, exist_ok=True)
    command = ["g++", "-shared", "-fPIC", f'-DNAME="{name}"', "-o", str(library), str(source)]
    subprocess.run(command, check=True)


@needs_compiler
def test_recorder_loads_from_folders_whose_paths_hold_spaces(tmp_path):
    source = tmp_path / "probe.cpp"
    source.write_text(PROBE_LIBRARY)
    library = tmp_path / "my project" / "build" / "liblanescope_recorder.so"
    sanitizer = tmp_path / "cache folder" / "libsanitizer-public.so"
    build_probe(source, library, "recorder")
    build_probe(source, sanitizer, "sanitizer")
    out = tmp_path / "my project" / "record"
    build = RecorderBuild(library, library.parent / "patches.cubin", sanitizer)
    assert capture.record([sys.executable, "-c", ""], build, out) == 0
    assert sorted((out / "loaded.txt").read_text().split()) == ["recorder", "sanitizer"]


@pytest.mark.parametrize("folder", ["a:b", "a;b"])
def test_environment_refuses_a_library_folder_that_the_search_path_would_split(tmp_path, folder):
    build = RecorderBuild(
        tmp_path / folder / "lib.so", tmp_path / "patches.cubin", tmp_path / "libsan.so"
    )
    with pytest.raises(ValueError, match="colons and semicolons"):
        capture.environment(build, tmp_path / "out")


def test_pcs_stay_absolute_without_a_kernel_start(tmp_path):
    info = write_launch(
        tmp_path, 0, [row(0x10, [4 * lane for lane in range(WARP_SIZE)])], function_pc=0
    )
    trace = capture.to_trace(*capture.load_launch(info))
    assert int(trace.pc[0]) == BASE_PC + 0x10
    assert trace.metadata["pc_base"] == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [({"format": 2}, "launch format"), ({"record_bytes": 64}, "280")],
)
def test_bad_launch_descriptions_are_rejected(tmp_path, change, message):
    info = write_launch(tmp_path, 0, [row(0x10, [4 * lane for lane in range(WARP_SIZE)])], **change)
    with pytest.raises(ValueError, match=message):
        capture.load_launch(info)


def test_truncated_records_are_rejected(tmp_path):
    info = write_launch(tmp_path, 0, [row(0x10, [4 * lane for lane in range(WARP_SIZE)])])
    data = info.with_suffix(".bin")
    data.write_bytes(data.read_bytes()[:100])
    with pytest.raises(ValueError, match="inside a record"):
        capture.load_launch(info)


def test_environment_preloads_the_recorder_before_other_libraries(tmp_path):
    build = RecorderBuild(
        tmp_path / "lib.so", tmp_path / "patches.cubin", tmp_path / "san" / "libsan.so"
    )
    base = {"LD_PRELOAD": "other.so", "LD_LIBRARY_PATH": "/opt/lib"}
    env = capture.environment(build, tmp_path / "out", ctas=(4, 8), kernel="gemm", base=base)
    assert env["LD_PRELOAD"] == "lib.so:libsan.so:other.so"
    assert env["LD_LIBRARY_PATH"] == f"{tmp_path.resolve()}:{(tmp_path / 'san').resolve()}:/opt/lib"
    assert (env["LANESCOPE_CTAS"], env["LANESCOPE_KERNEL"]) == ("4:8", "gemm")


def test_launch_description_that_is_no_object_is_rejected(tmp_path):
    info = write_launch(tmp_path, 0, [row(0x10, [4 * lane for lane in range(WARP_SIZE)])])
    info.write_text("[1, 2]")
    with pytest.raises(ValueError, match="no launch description"):
        capture.load_launch(info)


def test_pcs_outside_the_kernel_stay_absolute(tmp_path):
    info = write_launch(
        tmp_path, 0, [row(0x400, [4 * lane for lane in range(WARP_SIZE)])], function_size=0x100
    )
    trace = capture.to_trace(*capture.load_launch(info))
    assert int(trace.pc[0]) == BASE_PC + 0x400


def _has_gpu() -> bool:
    try:
        Driver().close()
    except DriverError:
        return False
    return True


TOOLS = toolchain.find()
needs_gpu = pytest.mark.skipif(
    TOOLS.nvrtc is None or shutil.which("g++") is None or not _has_gpu(),
    reason="NVRTC, g++, and a GPU are needed",
)


def record_script(tmp_path, text, **settings):
    """Run a launch script under a new recorder and return its traces and their folder."""
    driver = Driver()
    arch = driver.arch()
    driver.close()
    build = build_recorder(tmp_path / "build", arch, TOOLS)
    script = tmp_path / "launch.py"
    script.write_text(text)
    out_dir = tmp_path / "record"
    settings.setdefault("ctas", (0, 1 << 30))
    assert capture.record([sys.executable, str(script)], build, out_dir, **settings) == 0
    paths = capture.import_recording(out_dir, TOOLS.disassemblers)
    return [Trace.load(path) for path in paths], out_dir


LDMATRIX_LAUNCH = r'''
import ctypes
import sys

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import find

SOURCE = """
extern "C" __global__ void ldmatrix_x4(unsigned sink) {
    __shared__ __align__(128) unsigned char smem[4096];
    const unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(smem + 16 * threadIdx.x));
    unsigned a, b, c, d;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(address) : "memory");
    if ((a ^ b ^ c ^ d) == sink) asm volatile("trap;");
}
"""
driver = Driver()
options = [f"--gpu-architecture={driver.arch()}"]
program = nvrtc.compile_cuda(SOURCE, "ldmatrix.cu", options, find().nvrtc)
module = driver.load(program.cubin)
kernel = driver.function(module, "ldmatrix_x4")
driver.launch(kernel, (1, 1, 1), (32, 1, 1), [ctypes.c_uint32(0xFFFFFFFF)])
driver.close()
'''


@needs_gpu
def test_recorder_records_the_row_address_of_each_lane_of_an_ldmatrix(tmp_path):
    (trace,), _ = record_script(tmp_path, LDMATRIX_LAUNCH)
    rows = [row for row in range(len(trace)) if trace.space[row] == Space.SHARED]
    assert len(rows) == 1
    assert int(trace.width[rows[0]]) == 16
    assert trace.lane_addresses(rows[0]) == tuple(16 * lane for lane in range(WARP_SIZE))
    assert trace.metadata["opcodes"] == {f"{int(trace.pc[rows[0]]):#x}": "LDSM.16.M88.4"}


TWO_CONTEXTS_LAUNCH = r'''
import ctypes

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import find

SOURCE = """
extern "C" __global__ void read_tile(unsigned sink) {
    __shared__ unsigned tile[32];
    if (tile[threadIdx.x] == sink) asm volatile("trap;");
}
"""
for _ in range(2):
    # Each pass releases the primary context, so the second pass runs in a new context.
    driver = Driver()
    options = [f"--gpu-architecture={driver.arch()}"]
    program = nvrtc.compile_cuda(SOURCE, "read_tile.cu", options, find().nvrtc)
    kernel = driver.function(driver.load(program.cubin), "read_tile")
    driver.launch(kernel, (1, 1, 1), (32, 1, 1), [ctypes.c_uint32(0xFFFFFFFF)])
    driver.close()
'''


@needs_gpu
def test_recorder_records_a_kernel_of_a_context_that_replaces_a_destroyed_one(tmp_path):
    traces, _ = record_script(tmp_path, TWO_CONTEXTS_LAUNCH)
    assert [len(trace) for trace in traces] == [1, 1]


TWO_MODULES_LAUNCH = r'''
import ctypes

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import find

driver = Driver()
options = [f"--gpu-architecture={driver.arch()}"]
for name in ("skipped_tile", "recorded_tile"):
    source = f"""
extern "C" __global__ void {name}(unsigned sink) {{
    __shared__ unsigned tile[32];
    if (tile[threadIdx.x] == sink) asm volatile("trap;");
}}
"""
    program = nvrtc.compile_cuda(source, f"{name}.cu", options, find().nvrtc)
    kernel = driver.function(driver.load(program.cubin), name)
    driver.launch(kernel, (1, 1, 1), (32, 1, 1), [ctypes.c_uint32(0xFFFFFFFF)])
driver.close()
'''


@needs_gpu
def test_recorder_leaves_a_module_without_the_named_kernel_unpatched(tmp_path):
    traces, out_dir = record_script(tmp_path, TWO_MODULES_LAUNCH, kernel="recorded")
    assert [trace.metadata["kernel"] for trace in traces] == ["recorded_tile"]
    module = json.loads((out_dir / "launch-0.json").read_text())["module"]
    assert [path.name for path in out_dir.glob("module-*.cubin")] == [f"module-{module}.cubin"]


ZERO_FILL_LAUNCH = r'''
import ctypes

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import find

SOURCE = """
extern "C" __global__ void copy_odd_lanes(const unsigned char* buffer) {
    __shared__ __align__(128) unsigned char smem[512];
    const unsigned address =
        static_cast<unsigned>(__cvta_generic_to_shared(smem + 16 * threadIdx.x));
    const unsigned zeros = (threadIdx.x + 1) & 1u;
    asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;"
                 " cp.async.cg.shared.global [%0], [%1], 16, p; }"
                 :: "r"(address), "l"(buffer + 16 * threadIdx.x), "r"(zeros) : "memory");
    asm volatile("cp.async.wait_all;" ::: "memory");
}
"""
driver = Driver()
options = [f"--gpu-architecture={driver.arch()}"]
program = nvrtc.compile_cuda(SOURCE, "copy.cu", options, find().nvrtc)
kernel = driver.function(driver.load(program.cubin), "copy_odd_lanes")
buffer = driver.alloc(512)
driver.launch(kernel, (1, 1, 1), (32, 1, 1), [ctypes.c_uint64(buffer)])
driver.free(buffer)
driver.close()
'''


@needs_gpu
def test_recorder_records_the_shared_side_of_a_partly_zero_filling_copy_as_one_access(tmp_path):
    (trace,), _ = record_script(tmp_path, ZERO_FILL_LAUNCH)
    rows = sorted(zip(trace.space.tolist(), trace.mask.tolist(), strict=True))
    assert rows == [(Space.SHARED, 0xFFFFFFFF), (Space.GLOBAL, 0xAAAAAAAA)]


EIGHT_CTAS_LAUNCH = r'''
import ctypes

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import find

SOURCE = """
extern "C" __global__ void read_tile(unsigned sink) {
    __shared__ unsigned tile[32];
    if (tile[threadIdx.x] == sink) asm volatile("trap;");
}
"""
driver = Driver()
options = [f"--gpu-architecture={driver.arch()}"]
program = nvrtc.compile_cuda(SOURCE, "read_tile.cu", options, find().nvrtc)
kernel = driver.function(driver.load(program.cubin), "read_tile")
driver.launch(kernel, (8, 1, 1), (32, 1, 1), [ctypes.c_uint32(0xFFFFFFFF)])
driver.close()
'''


@needs_gpu
def test_recorder_records_only_the_ctas_of_its_window(tmp_path):
    (trace,), _ = record_script(tmp_path, EIGHT_CTAS_LAUNCH, ctas=(2, 3))
    assert sorted(trace.cta.tolist()) == [2, 3, 4]


DIVERGED_LOADS_LAUNCH = r'''
import ctypes

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import find

SOURCE = """
// A call keeps the two sides of the branch apart, where the compiler would otherwise
// predicate the two loads.
__device__ __noinline__ unsigned load_shared(unsigned address) {
    unsigned value;
    asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(value) : "r"(address));
    return value;
}

extern "C" __global__ void diverged_loads(unsigned* out) {
    __shared__ unsigned tile[96];
    const unsigned lane = threadIdx.x % 32;
    const unsigned base = static_cast<unsigned>(__cvta_generic_to_shared(tile));
    const unsigned odd = base + 4 * lane;
    const unsigned even = base + 128 + 4 * (lane / 2);
    unsigned sum = 0;
    #pragma unroll 1
    for (unsigned round = 0; round < 8; ++round) {
        unsigned value;
        asm volatile("{ .reg .pred p; setp.ne.u32 p, %3, 0; @p ld.volatile.shared.u32 %0, [%1];"
                     " @!p ld.volatile.shared.u32 %0, [%2]; }"
                     : "=r"(value) : "r"(odd), "r"(even), "r"((lane ^ round) & 1u));
        sum += value;
        if ((lane ^ round) & 1u) {
            value = load_shared(odd);
        } else {
            asm volatile("ld.volatile.shared.u32 %0, [%1];" : "=r"(value) : "r"(even));
        }
        sum += value;
    }
    out[threadIdx.x] = sum;
}
"""
driver = Driver()
options = [f"--gpu-architecture={driver.arch()}"]
program = nvrtc.compile_cuda(SOURCE, "diverged.cu", options, find().nvrtc)
kernel = driver.function(driver.load(program.cubin), "diverged_loads")
out = driver.alloc(4 * 128)
driver.launch(kernel, (1, 1, 1), (128, 1, 1), [ctypes.c_uint64(out)])
driver.free(out)
driver.close()
'''


@needs_gpu
def test_recorder_keeps_the_lanes_of_two_diverged_loads_in_records_of_their_own(tmp_path):
    (trace,), _ = record_script(tmp_path, DIVERGED_LOADS_LAUNCH)
    loads = trace.mask[trace.space == Space.SHARED].tolist()
    # Four warps run two pairs of loads in eight rounds, each load on half of the lanes.
    assert len(loads) == 4 * 8 * 4
    assert set(loads) == {0x55555555, 0xAAAAAAAA}


SPLIT_SOURCE = """
#include "patches.cu"

extern "C" __global__ void two_pcs(void* buffer) {
    const unsigned lane = threadIdx.x % 32;
    LaneScopeSharedAccess(buffer, 0x1000 + 0x10 * (lane & 1),
                          reinterpret_cast<void*>(4ull * lane), 4, 0x1, nullptr);
}
"""


@needs_gpu
def test_patches_split_the_lanes_of_one_call_by_pc():
    assert TOOLS.nvrtc is not None
    driver = Driver()
    try:
        options = [f"--gpu-architecture={driver.arch()}", f"--include-path={RECORDER_DIR}"]
        program = nvrtc.compile_cuda(SPLIT_SOURCE, "two_pcs.cu", options, TOOLS.nvrtc)
        assert program.cubin is not None
        kernel = driver.function(driver.load(program.cubin), "two_pcs")
        size = 32 + 4 * capture.RECORD_DTYPE.itemsize
        buffer = driver.alloc(size)
        # count, dropped, capacity, first CTA, CTA count, reserved, then the records.
        driver.upload(buffer, struct.pack("<6IQ", 0, 0, 4, 0, 1, 0, buffer + 32))
        driver.launch(kernel, (1, 1, 1), (32, 1, 1), [ctypes.c_uint64(buffer)])
        data = driver.download(buffer, size)
        driver.free(buffer)
    finally:
        driver.close()
    (count,) = struct.unpack_from("<I", data)
    records = np.frombuffer(data[32:], dtype=capture.RECORD_DTYPE)[:count]
    assert sorted(zip(records["pc"].tolist(), records["mask"].tolist(), strict=True)) == [
        (0x1000, 0x55555555),
        (0x1010, 0xAAAAAAAA),
    ]
