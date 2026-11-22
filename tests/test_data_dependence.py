import numpy as np

from lanescope import analysis, demos
from lanescope.fixes.layouts import Buffer
from lanescope.model.counters import count
from lanescope.model.data_dependence import MIN_ACCESSES, data_dependent
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace


def random_bins(accesses, warps, seed=7):
    """Return a trace of shared atomics whose 32 lanes add to random bins of 1,024."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(accesses):
        bins = rng.integers(0, 1024, WARP_SIZE)
        lanes = tuple(4 * int(value) for value in bins)
        rows.append(Access(0x40, Space.SHARED, Kind.ATOMIC, 4, lanes, index % 2, index % warps))
    return Trace.from_accesses(rows, {"kernel": "bins"})


def flagged(trace):
    return {key[0] for key in data_dependent(trace, count(trace))}


def test_the_bins_of_the_histogram_demo_follow_its_input_values():
    assert flagged(demos.histogram_trace()) == {demos.PC_ADD_BIN}


def test_conflicts_that_a_layout_sets_never_follow_the_input_values():
    for trace in (demos.transpose_trace(), demos.matmul_tile_trace(), demos.row_sums_trace()):
        assert flagged(trace) == set()


def test_random_bins_of_many_warps_follow_the_input_values():
    assert flagged(random_bins(64, 8)) == {0x40}


def test_too_few_accesses_tell_nothing_about_the_input_values():
    assert flagged(random_bins(MIN_ACCESSES - 1, 8)) == set()


def test_accesses_of_one_warp_alone_tell_nothing_about_the_input_values():
    rows = random_bins(64, 1).select(np.arange(64))
    rows.cta[:] = 0
    assert flagged(rows) == set()


def test_explain_marks_the_instruction_and_gives_the_excess_of_its_warps():
    records = analysis.explain(demos.histogram_trace())["instructions"]
    atomic = next(record for record in records if record["pc"] == hex(demos.PC_ADD_BIN))
    assert atomic["data_dependent"] is True
    fewest, most = atomic["excess_range"]
    assert 0 <= fewest < most
    others = [record for record in records if record["pc"] != hex(demos.PC_ADD_BIN)]
    assert not any(record["data_dependent"] for record in others)


def test_suggest_says_whether_the_conflicts_of_its_instruction_follow_the_input_values():
    assert analysis.suggest(demos.histogram_trace())["data_dependent"] is True
    assert analysis.suggest(demos.transpose_trace())["data_dependent"] is False


def test_suggest_for_a_given_buffer_says_whether_its_conflicts_follow_the_input_values():
    bins = Buffer(0, demos.HISTOGRAM_BINS * 4)
    assert analysis.suggest(demos.histogram_trace(), bins)["data_dependent"] is True
    tile = Buffer(0, 4096)
    assert analysis.suggest(demos.transpose_trace(), tile)["data_dependent"] is False
