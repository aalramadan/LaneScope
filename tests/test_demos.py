import numpy as np

from lanescope import analysis, demos
from lanescope.fixes import layouts
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
    }
