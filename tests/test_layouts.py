import numpy as np
import pytest

from lanescope import demos
from lanescope.fixes import layouts
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

TILE_BYTES = demos.TILE * demos.TILE * demos.FLOAT_BYTES
ROW_BYTES = demos.TILE * demos.FLOAT_BYTES
TILE_BUFFER = layouts.Buffer(0, TILE_BYTES)


def after_change(change):
    return {
        item.before.pc: item
        for item in layouts.compare(demos.transpose_trace(), TILE_BUFFER, change)
    }


def shared_trace(*accesses):
    return Trace.from_accesses(
        [
            Access(pc, Space.SHARED, Kind.LOAD, width, tuple(addresses))
            for pc, width, addresses in accesses
        ]
    )


def test_padding_one_float_per_row_removes_the_column_conflicts():
    result = after_change(layouts.Padding(ROW_BYTES, 4))
    assert result[demos.PC_LOAD_TILE].before.excessive == 3968
    assert result[demos.PC_LOAD_TILE].after.excessive == 0
    assert result[demos.PC_STORE_TILE].after.excessive == 0


def test_xor_swizzle_removes_the_column_conflicts_without_extra_bytes():
    change = layouts.XorSwizzle(5, 2, 5)
    result = after_change(change)
    assert result[demos.PC_LOAD_TILE].after.achieved == 128
    assert result[demos.PC_STORE_TILE].after.achieved == 128
    assert change.extra_bytes(TILE_BUFFER) == 0


def test_comparison_covers_only_instructions_on_the_buffer():
    result = after_change(layouts.Padding(ROW_BYTES, 4))
    assert set(result) == {demos.PC_STORE_TILE, demos.PC_LOAD_TILE}


def test_worst_rows_of_a_comparison_point_into_the_full_trace():
    trace = demos.transpose_trace()
    for item in layouts.compare(trace, TILE_BUFFER, layouts.XorSwizzle(5, 2, 5)):
        assert int(trace.pc[item.before.worst_row]) == item.before.pc
        assert int(trace.pc[item.after.worst_row]) == item.after.pc


def test_global_accesses_stay_in_place():
    trace = Trace.from_accesses(
        [Access(0x20, Space.GLOBAL, Kind.LOAD, 4, tuple(4 * lane for lane in range(WARP_SIZE)))]
    )
    moved = layouts.apply(trace, TILE_BUFFER, layouts.Padding(ROW_BYTES, 4))
    assert np.array_equal(moved.addr, trace.addr)


def test_padding_moves_each_row_by_its_pad():
    trace = shared_trace((0x10, 4, [ROW_BYTES * lane for lane in range(WARP_SIZE)]))
    moved = layouts.apply(trace, TILE_BUFFER, layouts.Padding(ROW_BYTES, 4))
    assert moved.lane_addresses(0)[:3] == (0, 132, 264)


def test_padding_moves_later_buffers_up_by_the_rounded_extra_bytes():
    trace = shared_trace((0x10, 4, [TILE_BYTES + 4 * lane for lane in range(WARP_SIZE)]))
    change = layouts.Padding(ROW_BYTES, 4)
    moved = layouts.apply(trace, TILE_BUFFER, change)
    assert layouts.later_shift(TILE_BUFFER, change) == 128
    assert moved.lane_addresses(0)[0] == TILE_BYTES + 128


def test_comparison_includes_instructions_on_moved_later_buffers():
    trace = shared_trace(
        (0x10, 4, [ROW_BYTES * lane for lane in range(WARP_SIZE)]),
        (0x20, 4, [TILE_BYTES + 8 * lane for lane in range(WARP_SIZE)]),
    )
    pcs = [
        item.before.pc
        for item in layouts.compare(trace, TILE_BUFFER, layouts.Padding(ROW_BYTES, 4))
    ]
    assert pcs == [0x10, 0x20]


def test_padding_reports_the_bytes_the_buffer_gains():
    assert layouts.Padding(ROW_BYTES, 4).extra_bytes(TILE_BUFFER) == 128


def test_swizzle_is_a_permutation_of_the_buffer():
    offsets = np.arange(0, TILE_BYTES, 4, dtype=np.uint64)
    moved = layouts.XorSwizzle(5, 2, 5).move(offsets)
    assert sorted(moved.tolist()) == offsets.tolist()


def test_cute_swizzle_on_elements_equals_the_byte_swizzle_with_a_larger_base():
    elements = np.arange(1024, dtype=np.uint64)
    on_elements = layouts.XorSwizzle(3, 3, 3).move(elements)
    on_bytes = layouts.XorSwizzle(3, 4, 3).move(elements * np.uint64(2))
    assert np.array_equal(on_elements * np.uint64(2), on_bytes)


def test_padding_that_misaligns_wide_accesses_names_the_change():
    trace = shared_trace((0x10, 8, [8 * lane for lane in range(WARP_SIZE)]))
    with pytest.raises(ValueError, match="pad each 128-byte row by 4 bytes misaligns the 8-byte"):
        layouts.apply(trace, layouts.Buffer(0, 1024), layouts.Padding(128, 4))


def test_padding_that_splits_wide_accesses_is_rejected():
    trace = shared_trace((0x10, 16, [16 * lane for lane in range(WARP_SIZE)]))
    with pytest.raises(ValueError, match="splits the 16-byte accesses of pc 0x10"):
        layouts.apply(trace, layouts.Buffer(0, 960), layouts.Padding(40, 16))


def test_swizzle_below_the_access_width_is_rejected():
    trace = shared_trace((0x10, 16, [16 * lane for lane in range(WARP_SIZE)]))
    with pytest.raises(ValueError, match="splits the 16-byte accesses"):
        layouts.apply(trace, layouts.Buffer(0, 4096), layouts.XorSwizzle(3, 2, 3))


def test_access_across_the_buffer_edge_is_rejected():
    trace = shared_trace((0x10, 16, [16 * lane for lane in range(WARP_SIZE)]))
    with pytest.raises(ValueError, match="cross the edge of the buffer"):
        layouts.apply(trace, layouts.Buffer(8, 256), layouts.Padding(8, 8))


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (layouts.XorSwizzle(3, 4, 2), "shift"),
        (layouts.XorSwizzle(5, 8, 5), "multiple of 8192"),
        (layouts.XorSwizzle(8, 30, 20), "below 2\\*\\*48"),
        (layouts.Padding(100, 4), "whole number"),
        (layouts.Padding(0, 4), "positive row size"),
        (layouts.Padding(128, 1 << 60), "padded row"),
    ],
)
def test_changes_that_do_not_fit_the_buffer_are_rejected(change, message):
    with pytest.raises(ValueError, match=message):
        change.check(TILE_BUFFER)


@pytest.mark.parametrize(("start", "size"), [(0, 0), (-4, 16), (0, 1 << 60)])
def test_buffer_needs_a_valid_range(start, size):
    with pytest.raises(ValueError, match="buffer"):
        layouts.Buffer(start, size)


def test_infer_buffer_covers_the_accesses_of_one_pc_in_whole_lines():
    trace = demos.transpose_trace()
    assert layouts.infer_buffer(trace, demos.PC_LOAD_TILE) == layouts.Buffer(0, 4096)


def test_infer_buffer_needs_a_shared_access():
    trace = demos.transpose_trace()
    with pytest.raises(ValueError, match="no shared access"):
        layouts.infer_buffer(trace, demos.PC_LOAD_INPUT)


def test_infer_buffer_stops_at_the_array_and_not_at_a_line():
    # Twenty rows of 80 bytes fill 1,600 bytes, and a second array starts right after them.
    edges = [
        Access(
            0x20,
            Space.SHARED,
            Kind.LOAD,
            4,
            tuple(80 * lane + column if lane < 20 else None for lane in range(WARP_SIZE)),
        )
        for column in (0, 76)
    ]
    neighbor = Access(
        0x40, Space.SHARED, Kind.LOAD, 4, tuple(1600 + 4 * lane for lane in range(WARP_SIZE))
    )
    trace = Trace.from_accesses([*edges, neighbor])
    assert layouts.infer_buffer(trace, 0x20) == layouts.Buffer(0, 1600)


def test_padding_that_would_move_addresses_past_the_window_is_refused():
    with pytest.raises(ValueError, match="padded buffer must end below"):
        layouts.Padding(16, (1 << 47) - 16).check(layouts.Buffer(0, 1 << 47))
