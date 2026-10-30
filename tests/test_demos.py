import numpy as np

from lanescope import analysis, demos
from lanescope.fixes import layouts
from lanescope.model.counters import count
from lanescope.trace import Kind, Space


def test_transpose_trace_has_four_instructions_of_128_accesses():
    trace = demos.transpose_trace(tiles_per_side=2)
    pcs, counts = np.unique(trace.pc, return_counts=True)
    assert pcs.tolist() == sorted(demos.TRANSPOSE_LINES)
    assert counts.tolist() == [128] * 4


def test_shared_accesses_stay_inside_the_tile():
    trace = demos.transpose_trace()
    shared = trace.addr[trace.space == Space.SHARED]
    assert shared.max() < demos.TILE * demos.TILE * demos.FLOAT_BYTES


def test_each_pc_maps_to_its_source_line():
    trace = demos.transpose_trace()
    source = trace.source_line(demos.PC_LOAD_TILE)
    assert source is not None
    line, text = source
    assert line == 10
    assert text.startswith("out[(y + j) * n + x] = tile[threadIdx.x]")


def test_warp_access_reads_one_address_per_lane():
    access = demos.warp_access(0x10, Space.SHARED, demos.Kind.LOAD, 4, lambda lane: 4 * lane)
    assert access.addresses[:3] == (0, 4, 8)
    assert len(access.addresses) == 32


def test_row_sums_reads_put_each_quarter_warp_into_four_banks():
    trace = demos.row_sums_trace(blocks=1)
    row = next(
        index
        for index in range(len(trace))
        if trace.space[index] == Space.SHARED and trace.kind[index] == Kind.LOAD
    )
    data = analysis.access(trace, row)
    assert [phase["wavefronts"] for phase in data["phases"]] == [8, 8, 8, 8]
    assert (data["achieved"], data["ideal"]) == (32, 4)


def test_row_sums_tile_needs_one_vector_of_padding_per_row():
    trace = demos.row_sums_trace(blocks=1)
    buffer = layouts.Buffer(0, 32 * demos.ROW_BYTES)
    data = analysis.whatif(trace, buffer, layouts.Padding(demos.ROW_BYTES, demos.VECTOR_BYTES))
    assert all(item["after"]["excessive"] == 0 for item in data["instructions"])


def test_column_sums_reads_take_one_sector_per_lane():
    trace = demos.column_sums_trace(blocks=1)
    data = analysis.access(trace, 0)
    assert (data["achieved"], data["lines"], data["ideal"]) == (32, 32, 4)


def test_demo_traces_name_their_files_and_kernels():
    traces = demos.demo_traces()
    assert {name: trace.metadata["kernel"] for name, trace in traces.items()} == {
        "1_transpose.npz": "transpose",
        "2_row_sums.npz": "row_sums",
        "3_column_sums.npz": "column_sums",
        "4_matmul_tile.npz": "matmul_tile",
    }


def test_matmul_tile_ldmatrix_puts_the_eight_rows_of_each_matrix_on_one_bank():
    trace = demos.matmul_tile_trace()
    (read,) = [counter for counter in count(trace) if counter.pc == demos.PC_READ_TILE]
    assert (read.achieved, read.ideal) == (32 * read.requests, 4 * read.requests)


def test_matmul_tile_needs_the_swizzle_that_cute_uses_for_128_byte_rows():
    best = analysis.suggest(demos.matmul_tile_trace())["changes"][0]
    assert best["change"] == "XOR swizzle Swizzle<3,4,3> on byte offsets"
    assert best["excessive_after"] == 0


def test_matmul_tile_maps_the_copy_and_the_ldmatrix_to_their_source_lines():
    trace = demos.matmul_tile_trace()
    copy, read = (trace.source_line(pc) for pc in (demos.PC_COPY_TILE, demos.PC_READ_TILE))
    assert copy is not None and copy[1].startswith("cp_async_16(&tile")
    assert read is not None and read[1].startswith("ldmatrix_x4(fragment[k]")
