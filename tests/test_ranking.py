import numpy as np
import pytest

from lanescope import demos
from lanescope.fixes import layouts, ranking
from lanescope.model import counters
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

TILE_BUFFER = layouts.Buffer(0, 4096)
LANES = np.arange(WARP_SIZE)


def ranked_by_counting_every_row(trace, buffer):
    """Rank the candidates by moving and counting every access of the buffer for each one."""
    accesses = trace.select(layouts.lanes_inside(trace, buffer).any(axis=1))

    def totals(rows):
        shared = [c for c in counters.count(rows) if c.space == Space.SHARED]
        return sum(c.achieved for c in shared), sum(c.excessive for c in shared)

    wavefronts, excessive = totals(accesses)
    found = []
    for change in ranking.candidates(buffer, int(accesses.width.max())):
        try:
            after = totals(layouts.apply(accesses, buffer, change))
        except ValueError:
            continue
        extra = change.extra_bytes(buffer)
        found.append(ranking.Suggestion(change, wavefronts, after[0], excessive, after[1], extra))
    better = [s for s in found if s.excessive_after < s.excessive_before]
    return sorted(better, key=ranking._rank)


def random_trace(seed, widths, arch, rows=4000):
    """Return repeats of random lane patterns on the tile buffer and the buffer above it.

    Each of twelve shared instructions gets one width and one direction and 20
    patterns, such as strides, random words, lane pairs, and lanes split between
    the two buffers. Some rows leave lanes inactive, and a global load adds rows
    that no layout change moves.
    """
    rng = np.random.default_rng(seed)
    patterns = []
    for pc in range(0x10, 0xD0, 0x10):
        width, kind = int(rng.choice(widths)), int(rng.integers(2))
        slots = TILE_BUFFER.size // width
        for shape in range(20):
            start = int(rng.integers(slots))
            if shape % 4 == 0:
                slot = (start + LANES * int(rng.choice([1, 2, 3, 8, 32, 33]))) % slots
            elif shape % 4 == 1:
                slot = rng.integers(0, slots, WARP_SIZE)
            elif shape % 4 == 2:
                slot = (start + LANES // 2 * int(rng.choice([1, 8, 32]))) % slots
            else:
                slot = rng.integers(0, slots, WARP_SIZE) + np.where(LANES < 16, 0, slots)
            mask = 0xFFFFFFFF if rng.random() < 0.7 else int(rng.integers(1, 1 << 32))
            patterns.append((pc, Space.SHARED, kind, width, mask, slot * width))
    for _ in range(10):
        lines = demos.INPUT_BASE + rng.integers(0, 1 << 20, WARP_SIZE) * 128
        patterns.append((0xF0, Space.GLOBAL, Kind.LOAD, 4, 0xFFFFFFFF, lines))
    picked = [patterns[index] for index in rng.integers(0, len(patterns), rows)]
    active = np.array([[mask >> lane & 1 for lane in range(WARP_SIZE)] for *_, mask, _ in picked])
    return Trace(
        pc=np.array([row[0] for row in picked]),
        space=np.array([int(row[1]) for row in picked]),
        kind=np.array([int(row[2]) for row in picked]),
        width=np.array([row[3] for row in picked]),
        mask=np.array([row[4] for row in picked]),
        cta=rng.integers(0, 64, rows),
        warp=rng.integers(0, 8, rows),
        addr=np.where(active == 1, np.stack([row[5] for row in picked]), 0).astype(np.uint64),
        metadata={"arch": arch},
    )


@pytest.mark.parametrize(
    ("name", "buffer"),
    [
        ("1_transpose.npz", TILE_BUFFER),
        ("1_transpose.npz", layouts.Buffer(0, 2048)),
        ("2_row_sums.npz", layouts.Buffer(0, 8192)),
        ("2_row_sums.npz", layouts.Buffer(4096, 4096)),
    ],
)
def test_search_ranks_the_demo_traces_as_counting_every_row_does(name, buffer):
    trace = demos.demo_traces()[name]
    expected = ranked_by_counting_every_row(trace, buffer)
    assert expected
    assert ranking.suggest(trace, buffer) == expected


@pytest.mark.parametrize(
    ("seed", "widths", "arch"),
    [(1, (4,), "sm_89"), (2, (4, 8, 16), "sm_89"), (3, (8, 16), "sm_90"), (4, (4, 8), "")],
)
def test_search_ranks_a_large_random_trace_as_counting_every_row_does(seed, widths, arch):
    trace = random_trace(seed, widths, arch)
    expected = ranked_by_counting_every_row(trace, TILE_BUFFER)
    assert expected
    assert ranking.suggest(trace, TILE_BUFFER) == expected


def test_search_moves_each_distinct_access_once_per_change(monkeypatch):
    sizes = []

    def apply(trace, buffer, change):
        sizes.append(len(trace))
        return layouts.apply(trace, buffer, change)

    monkeypatch.setattr(ranking, "apply", apply)
    trace = demos.transpose_trace()
    ranking.suggest(trace, TILE_BUFFER)
    shared = trace.select(trace.space == Space.SHARED)
    assert set(sizes) == {len(np.unique(counters.row_keys(shared), axis=0))}


def test_search_finds_the_swizzle_and_the_pad_of_the_transpose():
    found = ranking.suggest(demos.transpose_trace(), TILE_BUFFER)
    best, second = found[:2]
    assert best.change == layouts.XorSwizzle(5, 2, 5)
    assert second.change == layouts.Padding(128, 4)
    assert (best.excessive_after, second.excessive_after, second.extra_bytes) == (0, 0, 128)


def test_every_suggestion_lowers_the_excess():
    found = ranking.suggest(demos.transpose_trace(), TILE_BUFFER)
    assert all(s.excessive_after < s.excessive_before for s in found)


def test_candidates_keep_16_byte_accesses_whole():
    for change in ranking.candidates(layouts.Buffer(0, 8192), 16):
        assert change.keeps_whole(16)
        if isinstance(change, layouts.Padding):
            assert change.pad_bytes % 16 == 0


def test_buffer_without_shared_accesses_is_rejected():
    trace = Trace.from_accesses(
        [Access(0x10, Space.GLOBAL, Kind.LOAD, 4, tuple(4 * lane for lane in range(WARP_SIZE)))]
    )
    with pytest.raises(ValueError, match="inside the buffer"):
        ranking.suggest(trace, TILE_BUFFER)


def test_buffer_cutting_through_accesses_names_the_crossing():
    with pytest.raises(ValueError, match="cross the edge of the buffer"):
        ranking.suggest(demos.transpose_trace(), layouts.Buffer(2, 1024))


def test_rows_that_are_no_power_of_two_are_tried():
    accesses = [
        Access(
            0x20,
            Space.SHARED,
            Kind.LOAD,
            4,
            tuple(192 * lane + 4 * column for lane in range(WARP_SIZE)),
        )
        for column in range(48)
    ]
    found = ranking.suggest(Trace.from_accesses(accesses), layouts.Buffer(0, 6144))
    assert any(s.change == layouts.Padding(192, 4) and s.excessive_after == 0 for s in found)


def test_divisors_come_in_ascending_order_once_each():
    assert ranking.divisors(36) == [1, 2, 3, 4, 6, 9, 12, 18, 36]
    assert ranking.divisors(4096) == [1 << power for power in range(13)]
    assert ranking.divisors(1) == [1]


def test_candidates_of_a_huge_buffer_come_from_its_divisors_alone():
    found = ranking.candidates(layouts.Buffer(0, 1 << 40), 4)
    rows = {change.row_bytes for change in found if isinstance(change, layouts.Padding)}
    assert rows == {1 << power for power in range(4, 41)}
