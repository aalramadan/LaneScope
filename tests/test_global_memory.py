import pytest

from lanescope.model import global_memory
from lanescope.trace import WARP_SIZE

BASE = 0x7F00_0000_0000


def lanes(address_of):
    return [address_of(lane) for lane in range(WARP_SIZE)]


def test_coalesced_floats_touch_four_sectors_of_one_line():
    result = global_memory.analyze(lanes(lambda lane: BASE + 4 * lane), 4)
    assert (result.sectors, result.lines, result.ideal, result.excessive) == (4, 1, 4, 0)


def test_stride_of_two_floats_doubles_the_sectors():
    result = global_memory.analyze(lanes(lambda lane: BASE + 8 * lane), 4)
    assert (result.sectors, result.ideal, result.excessive) == (8, 4, 4)


def test_offset_of_one_float_adds_a_sector_and_a_line():
    result = global_memory.analyze(lanes(lambda lane: BASE + 4 + 4 * lane), 4)
    assert (result.sectors, result.lines) == (5, 2)


def test_column_access_touches_one_sector_per_lane():
    result = global_memory.analyze(lanes(lambda lane: BASE + 4096 * lane), 4)
    assert (result.sectors, result.lines, result.excessive) == (32, 32, 28)


def test_broadcast_touches_one_sector_without_excess():
    result = global_memory.analyze([BASE + 64] * WARP_SIZE, 4)
    assert (result.sectors, result.excessive) == (1, 0)


def test_coalesced_128_bit_accesses_touch_sixteen_sectors():
    result = global_memory.analyze(lanes(lambda lane: BASE + 16 * lane), 16)
    assert (result.sectors, result.lines, result.ideal) == (16, 4, 16)


def test_bytes_of_one_sector_need_one_sector():
    result = global_memory.analyze(lanes(lambda lane: BASE + lane), 1)
    assert (result.sectors, result.ideal) == (1, 1)


def test_sector_lanes_name_the_lanes_of_each_sector():
    result = global_memory.analyze(lanes(lambda lane: BASE + 4 * lane), 4)
    assert result.sector_lanes[1] == (1, tuple(range(8, 16)))
    assert result.sector_address(1) == BASE + 32


def test_counts_do_not_depend_on_the_line_of_the_access():
    first = global_memory.analyze(lanes(lambda lane: BASE + 12 + 8 * lane), 4)
    second = global_memory.analyze(lanes(lambda lane: BASE + 128 * 9 + 12 + 8 * lane), 4)
    assert (first.sectors, first.lines) == (second.sectors, second.lines)
    assert second.base - first.base == 128 * 9


def test_inactive_lanes_are_left_out():
    addresses = lanes(lambda lane: BASE + 4096 * lane)
    addresses[8:] = [None] * 24
    result = global_memory.analyze(addresses, 4)
    assert (result.active, result.sectors, result.ideal) == (8, 8, 1)


def test_access_without_active_lanes_is_rejected():
    with pytest.raises(ValueError, match="active lane"):
        global_memory.analyze([None] * WARP_SIZE, 4)


def test_coalesced_floats_from_the_last_sector_of_a_line_touch_two_lines():
    result = global_memory.analyze(lanes(lambda lane: BASE + 96 + 4 * lane), 4)
    assert (result.sectors, result.lines) == (4, 2)


@pytest.mark.parametrize(("width", "sectors", "lines"), [(2, 2, 1), (8, 8, 2)])
def test_coalesced_accesses_of_two_and_eight_bytes(width, sectors, lines):
    result = global_memory.analyze(lanes(lambda lane: BASE + width * lane), width)
    assert (result.sectors, result.lines, result.ideal) == (sectors, lines, sectors)
