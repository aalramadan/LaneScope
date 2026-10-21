import json
import re
import subprocess

import numpy as np
import pytest

from lanescope.cuda import line_info, nvrtc, toolchain
from lanescope.recorder import capture
from lanescope.trace import WARP_SIZE, Kind, Space, Trace

TOOLS = toolchain.find()
LISTING = """
//--------------------- .text.first               --------------------------
	.section	.text.first,"ax",@progbits
first:
.text.first:
	//## File "/src/kernel.cu", line 5
        /*0000*/                   IMAD.MOV.U32 R1, RZ, RZ, c[0x0][0x28] ;
	//## File "/src/kernel.cu", line 7
        /*0010*/                   S2R R11, SR_TID.X ;
        /*0020*/                   LDG.E R6, [R6.64] ;
	//## File "/src/helper.cuh", line 3
        /*0030*/                   LDS R0, [R0] ;
.L_x_0:
        /*0040*/                   BRA `(.L_x_0);
//--------------------- .text.second              --------------------------
.text.second:
        /*0000*/                   EXIT ;
"""
KERNEL = """extern "C" __global__ void probe(const float* in, float* out) {
  __shared__ float tile[32][32];
  tile[threadIdx.y][threadIdx.x] = in[threadIdx.x];
  __syncthreads();
  out[threadIdx.x] = tile[threadIdx.x][threadIdx.y];
}
"""


HELPER = """__device__ __forceinline__ void put(float* tile, int i, float v) {
  tile[i] = v;
}
"""
INLINED = """#include "helper.cuh"
extern "C" __global__ void inlined(const float* in, float* out) {
  __shared__ float tile[32 * 32];
  put(tile, threadIdx.x * 32, in[threadIdx.x]);
  __syncthreads();
  out[threadIdx.x] = tile[threadIdx.x];
}
"""


def test_parse_maps_each_offset_to_the_last_location_before_it():
    functions = line_info.parse(LISTING)
    assert functions["first"] == {
        0x00: ("/src/kernel.cu", 5),
        0x10: ("/src/kernel.cu", 7),
        0x20: ("/src/kernel.cu", 7),
        0x30: ("/src/helper.cuh", 3),
        0x40: ("/src/helper.cuh", 3),
    }
    assert functions["second"] == {}


def test_metadata_keeps_the_lines_of_the_main_file_with_its_text(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text("a\nb\nc\n")
    lines = {0x0: (str(source), 2), 0x10: (str(source), 3), 0x20: ("/other.cuh", 9)}
    data = line_info.metadata(lines)
    assert data == {
        "source_file": str(source),
        "lines": {"0x0": 2, "0x10": 3},
        "source": ["a", "b", "c"],
    }


def test_metadata_without_a_readable_file_keeps_only_the_lines():
    data = line_info.metadata({0x0: ("/missing/kernel.cu", 4)})
    assert data == {"source_file": "/missing/kernel.cu", "lines": {"0x0": 4}}
    assert line_info.metadata({}) == {}


@pytest.mark.skipif(
    TOOLS.nvrtc is None or TOOLS.nvdisasm is None, reason="NVRTC and nvdisasm are needed"
)
def test_import_maps_a_recorded_shared_load_to_its_source_line(tmp_path):
    source = tmp_path / "probe.cu"
    source.write_text(KERNEL)
    assert TOOLS.nvrtc is not None and TOOLS.nvdisasm is not None
    program = nvrtc.compile_cuda(
        KERNEL, str(source), ["--gpu-architecture=sm_89", "-lineinfo"], TOOLS.nvrtc
    )
    assert program.cubin is not None
    folder = tmp_path / "run"
    folder.mkdir()
    (folder / "module-0.cubin").write_bytes(program.cubin)
    table = line_info.module_lines(TOOLS.disassemblers, folder / "module-0.cubin")["probe"]
    offset = max(offset for offset, (_, line) in table.items() if line == 5)
    base = 0x7F00_0000_0000
    records = np.zeros(1, dtype=capture.RECORD_DTYPE)
    records[0]["pc"] = base + offset
    records[0]["addr"] = [128 * lane for lane in range(WARP_SIZE)]
    records[0]["mask"] = 0xFFFFFFFF
    records[0]["width"] = 4
    records[0]["space"] = int(Space.SHARED)
    records[0]["kind"] = int(Kind.LOAD)
    (folder / "launch-0.bin").write_bytes(records.tobytes())
    info = {
        "format": 1,
        "kernel": "probe",
        "launch": 0,
        "module": 0,
        "function_pc": base,
        "function_size": 4096,
        "record_bytes": 280,
    }
    (folder / "launch-0.json").write_text(json.dumps(info))
    (path,) = capture.import_recording(folder, TOOLS.disassemblers)
    trace = Trace.load(path)
    assert trace.source_line(offset) == (5, "out[threadIdx.x] = tile[threadIdx.x][threadIdx.y];")
    assert trace.metadata["source_file"] == str(source)


def write_unreadable_module(folder):
    records = np.zeros(1, dtype=capture.RECORD_DTYPE)
    records[0]["pc"] = 0x1010
    records[0]["mask"] = 1
    records[0]["width"] = 4
    (folder / "launch-0.bin").write_bytes(records.tobytes())
    info = {"format": 1, "kernel": "k", "module": 0, "function_pc": 0x1000, "record_bytes": 280}
    (folder / "launch-0.json").write_text(json.dumps(info))
    (folder / "module-0.cubin").write_bytes(b"not a cubin")


def test_import_without_nvdisasm_leaves_the_lines_out(tmp_path):
    write_unreadable_module(tmp_path)
    (path,) = capture.import_recording(tmp_path)
    assert "lines" not in Trace.load(path).metadata


@pytest.mark.skipif(TOOLS.nvdisasm is None, reason="nvdisasm is needed")
def test_import_keeps_a_trace_whose_module_nvdisasm_cannot_read(tmp_path):
    write_unreadable_module(tmp_path)
    with pytest.warns(UserWarning, match="carry no source lines"):
        (path,) = capture.import_recording(tmp_path, TOOLS.disassemblers)
    assert "lines" not in Trace.load(path).metadata


def test_metadata_reads_only_absolute_paths_of_source_files(tmp_path, monkeypatch):
    notes = tmp_path / "notes.txt"
    notes.write_text("key = 1")
    assert "source" not in line_info.metadata({0x0: (str(notes), 1)})
    (tmp_path / "kernel.cu").write_text("x")
    monkeypatch.chdir(tmp_path)
    assert "source" not in line_info.metadata({0x0: ("kernel.cu", 1)})


@pytest.mark.skipif(
    TOOLS.nvrtc is None or TOOLS.nvdisasm is None, reason="NVRTC and nvdisasm are needed"
)
def test_a_store_inlined_from_a_header_takes_the_line_of_its_call(tmp_path):
    assert TOOLS.nvrtc is not None and TOOLS.nvdisasm is not None
    (tmp_path / "helper.cuh").write_text(HELPER)
    source = tmp_path / "inlined.cu"
    source.write_text(INLINED)
    options = ["--gpu-architecture=sm_89", "-lineinfo", f"-I{tmp_path}"]
    program = nvrtc.compile_cuda(INLINED, str(source), options, TOOLS.nvrtc)
    assert program.cubin is not None
    cubin = tmp_path / "inlined.cubin"
    cubin.write_bytes(program.cubin)
    code = subprocess.run(
        [str(TOOLS.nvdisasm), "--print-code", str(cubin)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    found = re.search(r"/\*([0-9a-f]{4})\*/\s+STS\b", code)
    assert found is not None
    table = line_info.module_lines(TOOLS.disassemblers, cubin)["inlined"]
    assert table[int(found.group(1), 16)] == (str(source), 4)


def test_module_lines_turn_to_the_next_nvdisasm_when_one_fails(tmp_path):
    failing = tmp_path / "old-nvdisasm"
    failing.write_text("#!/bin/sh\necho 'CUDA architecture is deprecated' >&2\nexit 1\n")
    failing.chmod(0o755)
    working = tmp_path / "nvdisasm"
    listing = '.text.k:\\n//## File "/src/k.cu", line 3\\n /*0000*/ EXIT ;\\n'
    working.write_text(f"#!/bin/sh\nprintf '{listing}'\n")
    working.chmod(0o755)
    cubin = tmp_path / "module.cubin"
    cubin.write_bytes(b"")
    assert line_info.module_lines([failing, working], cubin) == {"k": {0: ("/src/k.cu", 3)}}
    with pytest.raises(ValueError, match="CUDA architecture is deprecated"):
        line_info.module_lines([failing], cubin)


def test_module_lines_pass_over_an_nvdisasm_that_cannot_start(tmp_path):
    broken = tmp_path / "nvdisasm"
    broken.write_text("not a program")
    cubin = tmp_path / "module.cubin"
    cubin.write_bytes(b"")
    with pytest.raises(ValueError, match="cannot start"):
        line_info.module_lines([broken], cubin)
