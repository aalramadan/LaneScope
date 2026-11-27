import dataclasses
import json

import pytest

from lanescope import demos, recording_comparison, summary
from lanescope.fixes import layouts

TILE_4 = "void ns::tile<4>(float*)"
TILE_8 = "void ns::tile<8>(float*)"


def padded_transpose():
    """Return the transpose with each row of its tile padded by one float, free of conflicts."""
    trace = demos.transpose_trace()
    return layouts.apply(trace, layouts.infer_buffer(trace), layouts.Padding(128, 4))


def timing(**kernels):
    """Return the kernel times of lanescope time, with the nanoseconds of each kernel per round."""
    rounds = len(next(iter(kernels.values())))
    return {
        "format": 1,
        "rounds": rounds,
        "kernels": {
            name: {"launches": [1] * rounds, "nanoseconds": values, "shortest": 1, "longest": 1}
            for name, values in kernels.items()
        },
    }


def side(run, traces, times=None, kernel=None):
    """Return the summary of one run of traces, with the kernel name and the times given."""
    items = []
    for index, trace in enumerate(traces):
        found = summary.trace_facts(trace)
        if kernel is not None:
            found = dataclasses.replace(found, kernel=kernel[index])
        items.append((f"{run}/trace-{index}.npz", run, found))
    return summary.summarize(items, times={run: times} if times else None)


def test_a_fix_shows_its_counts_before_and_after_and_its_speedup():
    before = side("a", [demos.transpose_trace()], timing(transpose=[1_600, 1_700, 1_650]))
    after = side("b", [padded_transpose()], timing(transpose=[400, 420, 410]))
    data = recording_comparison.compare(before, after)
    (row,) = data["kernels"]
    assert (row["template"], row["name"]) == ("transpose", "transpose")
    assert row["before"]["wavefronts"] == {"achieved": 4224, "ideal": 256, "excessive": 3968}
    assert row["after"]["wavefronts"] == {"achieved": 256, "ideal": 256, "excessive": 0}
    assert (row["before"]["time_ns"], row["before"]["time_range_ns"]) == (1_650, [1_600, 1_700])
    assert row["time"] == {"speedup": 1_650 / 410, "within_spread": False}
    assert data["timed"] == {"before": 3, "after": 3}


def test_times_whose_rounds_overlap_lie_within_the_spread():
    before = side("a", [demos.transpose_trace()], timing(transpose=[500, 520, 510]))
    after = side("b", [demos.transpose_trace()], timing(transpose=[505, 515, 500]))
    (row,) = recording_comparison.compare(before, after)["kernels"]
    assert row["time"]["within_spread"] is True


def test_a_side_without_times_gives_no_speedup():
    before = side("a", [demos.transpose_trace()], timing(transpose=[500]))
    after = side("b", [padded_transpose()])
    data = recording_comparison.compare(before, after)
    assert data["kernels"][0]["time"] == {"speedup": None, "within_spread": None}
    assert data["total"]["time"] == {"speedup": None, "within_spread": None}
    assert data["timed"] == {"before": 1, "after": None}


def test_the_templates_of_one_side_alone_are_listed_apart():
    before = side("a", [demos.transpose_trace(), demos.row_sums_trace()])
    after = side("b", [padded_transpose(), demos.histogram_trace()])
    data = recording_comparison.compare(before, after)
    assert [row["name"] for row in data["kernels"]] == ["transpose"]
    assert [entry["name"] for entry in data["only_before"]] == ["row_sums"]
    assert [entry["name"] for entry in data["only_after"]] == ["histogram"]
    assert data["only_after"][0]["launches"] == 1


def test_the_total_adds_up_every_kernel_of_each_side_with_the_time_of_the_whole_run():
    before = side(
        "a",
        [demos.transpose_trace(), demos.row_sums_trace()],
        timing(transpose=[1_000, 1_200], row_sums=[100, 100], untraced=[50, 50]),
    )
    after = side("b", [padded_transpose()], timing(transpose=[300, 320]))
    total = recording_comparison.compare(before, after)["total"]
    assert total["before"]["launches"] == 2
    assert total["before"]["wavefronts"]["excessive"] == before["totals"]["wavefronts"]["excessive"]
    assert (total["before"]["time_ns"], total["before"]["time_range_ns"]) == (1_250, [1_150, 1_350])
    assert total["after"]["time_ns"] == 310
    assert total["time"]["within_spread"] is False


def test_the_instances_of_a_template_compare_where_their_names_agree():
    traces = [demos.transpose_trace(tiles_per_side=1), demos.transpose_trace()]
    before = side("a", traces, kernel=[TILE_4, TILE_8])
    after = side("b", traces, kernel=[TILE_4, "void ns::tile<16>(float*)"])
    (row,) = recording_comparison.compare(before, after)["kernels"]
    assert row["template"] == "ns::tile"
    assert [(found["kernel"], found["name"]) for found in row["instances"]] == [(TILE_4, "tile<4>")]


def test_a_template_of_one_instance_whose_arguments_change_compares_under_its_template():
    before = side("a", [demos.transpose_trace()], kernel=[TILE_4])
    after = side("b", [padded_transpose()], kernel=[TILE_8])
    (row,) = recording_comparison.compare(before, after)["kernels"]
    assert (row["name"], row["full_name"], row["instances"]) == ("tile", "ns::tile", [])


def test_a_side_adds_up_every_trace_of_its_folder_as_one_run_with_its_times(tmp_path):
    (tmp_path / "nested").mkdir()
    demos.transpose_trace().save(tmp_path / "trace-0.npz")
    demos.transpose_trace().save(tmp_path / "nested" / "trace-0.npz")
    (tmp_path / "kernel-times.json").write_text(json.dumps(timing(transpose=[70])))
    data = recording_comparison.read_side(tmp_path)
    assert data["runs"] == [str(tmp_path)]
    assert (data["kernels"][0]["launches"], data["kernels"][0]["time_ns"]) == (2, 70)


def test_a_side_of_one_trace_file_takes_the_times_beside_it(tmp_path):
    demos.transpose_trace().save(tmp_path / "transpose.npz")
    (tmp_path / "kernel-times.json").write_text(json.dumps(timing(transpose=[70])))
    data = recording_comparison.read_side(tmp_path / "transpose.npz")
    assert data["kernels"][0]["time_ns"] == 70
    assert recording_comparison.times_folder(tmp_path / "transpose.npz") == tmp_path


@pytest.mark.parametrize(
    ("make", "reason"),
    [(lambda path: None, "no such file or folder"), (lambda path: path.mkdir(), "found no trace")],
)
def test_a_side_names_a_path_without_traces(tmp_path, make, reason):
    make(tmp_path / "side")
    with pytest.raises(ValueError, match=reason):
        recording_comparison.read_side(tmp_path / "side")
