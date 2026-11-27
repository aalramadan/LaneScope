import numpy as np
import pytest

from lanescope import demos, trace_reports
from lanescope.fixes.layouts import Buffer
from lanescope.model.counters import count_instructions
from lanescope.model.data_dependence import (
    MAX_PATTERNS,
    MIN_ACCESSES,
    data_dependent,
    follows_input_over_launches,
    input_evidence,
)
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
    return {key[0] for key in data_dependent(trace, count_instructions(trace))}


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
    records = trace_reports.explain(demos.histogram_trace())["instructions"]
    atomic = next(record for record in records if record["pc"] == hex(demos.PC_ADD_BIN))
    assert atomic["data_dependent"] is True
    fewest, most = atomic["excess_range"]
    assert 0 <= fewest < most
    others = [record for record in records if record["pc"] != hex(demos.PC_ADD_BIN)]
    assert not any(record["data_dependent"] for record in others)


def test_suggest_says_whether_the_conflicts_of_its_instruction_follow_the_input_values():
    assert trace_reports.suggest(demos.histogram_trace())["data_dependent"] is True
    assert trace_reports.suggest(demos.transpose_trace())["data_dependent"] is False


def test_suggest_for_a_given_buffer_says_whether_its_conflicts_follow_the_input_values():
    bins = Buffer(0, demos.HISTOGRAM_BINS * 4)
    assert trace_reports.suggest(demos.histogram_trace(), bins)["data_dependent"] is True
    tile = Buffer(0, 4096)
    assert trace_reports.suggest(demos.transpose_trace(), tile)["data_dependent"] is False


def test_the_evidence_of_a_shared_instruction_gives_its_rows_patterns_and_excess_counts():
    trace = random_bins(24, 8)
    (counter,) = count_instructions(trace)
    (found,) = input_evidence(trace, [counter]).values()
    assert found.rows == 24
    assert found.patterns is not None and len(found.patterns) == 24
    assert found.excess_counts == frozenset(counter.excess_counts)
    assert found.data_dependent is False


def test_the_evidence_of_one_trace_keeps_no_patterns_past_its_limit():
    trace = random_bins(MAX_PATTERNS + 1, 8)
    (found,) = input_evidence(trace, count_instructions(trace)).values()
    assert found.patterns is None
    assert found.data_dependent is True


def test_instructions_without_excess_and_global_ones_carry_no_evidence():
    trace = demos.transpose_trace()
    keys = {key[0] for key in input_evidence(trace, count_instructions(trace))}
    assert keys == {demos.PC_LOAD_TILE}


@pytest.mark.parametrize(
    ("rows", "launches", "patterns", "levels", "follows"),
    [
        (48, 2, 24, 3, True),
        (48, 1, 48, 3, False),
        (MIN_ACCESSES - 1, 2, 31, 3, False),
        (48, 2, 23, 3, False),
        (48, 2, 48, 2, False),
        (48, 2, None, 3, False),
    ],
)
def test_the_launches_of_a_line_follow_the_input_when_they_differ_enough(
    rows, launches, patterns, levels, follows
):
    hashes = None if patterns is None else frozenset(range(patterns))
    found = follows_input_over_launches(rows, launches, hashes, frozenset(range(levels)))
    assert found is follows
