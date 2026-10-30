import itertools

import pytest

from lanescope.model import async_copy
from lanescope.trace import WARP_SIZE

LANE_SOURCES = [16 * lane for lane in range(WARP_SIZE)]
# The rotations and reflections of the four sectors of a line, as source to target maps.
ROTATIONS_AND_REFLECTIONS = {"0123", "1230", "2301", "3012", "0321", "1032", "2103", "3210"}


def first_group(rows):
    """Return the destinations when lanes 0 to 7 write these rows and the other lanes idle."""
    return [16 * row for row in rows] + [None] * (WARP_SIZE - len(rows))


def wavefronts(destinations, sources=LANE_SOURCES):
    return len(async_copy.wavefronts(async_copy.copies(sources, destinations)))


def test_a_lane_pair_that_copies_one_sector_into_one_aligned_sector_makes_one_write():
    straight = async_copy.copies(LANE_SOURCES, first_group([0, 1]))
    swapped = async_copy.copies(LANE_SOURCES, first_group([1, 0]))
    apart = async_copy.copies(LANE_SOURCES, first_group([1, 2]))
    assert [(copy.lanes, copy.swapped) for copy in straight] == [((0, 1), False)]
    assert [(copy.lanes, copy.swapped) for copy in swapped] == [((0, 1), True)]
    assert [copy.lanes for copy in apart] == [(0,), (1,)]


def test_a_row_that_an_earlier_write_covers_takes_no_write_of_its_own():
    assert len(async_copy.copies(LANE_SOURCES, [0] * WARP_SIZE)) == 1
    covered = async_copy.copies(LANE_SOURCES, first_group([0, 1, 1, 1, 0, 0, 1, 1]))
    assert [copy.lanes for copy in covered] == [(0, 1)]


@pytest.mark.parametrize("order", ["".join(map(str, p)) for p in itertools.permutations(range(4))])
def test_one_wavefront_moves_the_sectors_of_a_line_by_one_rotation_or_reflection(order):
    rows = [2 * int(order[lane // 2]) + lane % 2 for lane in range(8)]
    assert wavefronts(first_group(rows)) == (1 if order in ROTATIONS_AND_REFLECTIONS else 2)


@pytest.mark.parametrize(
    ("lanes", "rows", "expected"),
    [
        ((0, 2), (0, 2), 1),
        ((0, 2), (0, 6), 1),
        ((1, 2), (1, 2), 2),
        ((0, 2), (0, 4), 2),
        ((0, 2), (0, 1), 2),
    ],
)
def test_lone_rows_share_a_wavefront_only_with_equal_halves_and_one_sector_map(
    lanes, rows, expected
):
    destinations = [None] * WARP_SIZE
    for lane, row in zip(lanes, rows, strict=True):
        destinations[lane] = 16 * row
    assert wavefronts(destinations) == expected


def test_each_global_line_and_each_shared_line_take_wavefronts_of_their_own():
    assert wavefronts([16 * lane for lane in range(WARP_SIZE)]) == 4
    assert wavefronts(first_group(range(2, 10))) == 2


def test_lanes_that_read_separate_global_sectors_write_one_row_per_wavefront():
    apart = [32 * lane for lane in range(WARP_SIZE)]
    assert wavefronts(first_group(range(8)), apart) == 8
