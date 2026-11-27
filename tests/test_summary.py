import concurrent.futures
import dataclasses

import numpy as np
import pytest

from lanescope import demos, summary, summary_text
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

TILE_4 = "void ns::tile<4>(float*)"
TILE_8 = "void ns::tile<8>(float*)"


def facts(trace):
    return summary.trace_facts(trace)


def shared_lines(data):
    return [line for line in data["lines"] if line["unit"] == "wavefronts"]


def instance(trace, kernel):
    return dataclasses.replace(facts(trace), kernel=kernel)


def two_instances():
    """Return the items of one run: a small transpose as tile<4>, two large ones as tile<8>."""
    small = instance(demos.transpose_trace(tiles_per_side=1), TILE_4)
    large = instance(demos.transpose_trace(), TILE_8)
    return [
        ("r/trace-0.npz", "r", small),
        ("r/trace-1.npz", "r", large),
        ("r/trace-2.npz", "r", large),
    ]


def scattered_bins(seed, accesses=24):
    """Return the facts of a launch of shared atomics whose lanes add to random bins."""
    rng = np.random.default_rng(seed)
    rows = [
        Access(
            0x40,
            Space.SHARED,
            Kind.ATOMIC,
            4,
            tuple(4 * int(v) for v in rng.integers(0, 1024, WARP_SIZE)),
            index % 2,
            index % 8,
        )
        for index in range(accesses)
    ]
    return facts(Trace.from_accesses(rows, {"kernel": "bins"}))


def test_the_launches_of_one_kernel_in_a_run_add_up():
    one = facts(demos.transpose_trace())
    data = summary.summarize([("a/trace-0.npz", "a", one), ("a/trace-1.npz", "a", one)])
    (kernel,) = data["kernels"]
    assert kernel["launches"] == 2
    assert kernel["totals"]["wavefronts"]["excessive"] == 2 * 3968
    (line,) = shared_lines(data)
    assert (line["line"], line["launches"], line["excessive"]) == (10, 2, 2 * 3968)
    assert line["kernel_achieved"] == kernel["totals"]["wavefronts"]["achieved"]
    assert data["totals"]["wavefronts"]["excessive"] == 2 * 3968


def test_kernels_of_the_same_name_in_two_runs_stay_apart():
    one = facts(demos.transpose_trace())
    data = summary.summarize([("x.npz", "x.npz", one), ("y.npz", "y.npz", one)])
    assert [kernel["run"] for kernel in data["kernels"]] == ["x.npz", "y.npz"]
    assert data["runs"] == ["x.npz", "y.npz"]


def test_lines_come_worst_first_and_keep_only_extra_passes_or_sectors():
    items = [
        ("t.npz", "t.npz", facts(demos.transpose_trace())),
        ("r.npz", "r.npz", facts(demos.row_sums_trace())),
        ("c.npz", "c.npz", facts(demos.column_sums_trace())),
    ]
    data = summary.summarize(items)
    shared = shared_lines(data)
    assert [line["excessive"] for line in shared] == sorted(
        (line["excessive"] for line in shared), reverse=True
    )
    assert all(line["excessive"] > 0 for line in data["lines"])
    sectors = [line for line in data["lines"] if line["unit"] == "sectors"]
    assert sectors and sectors[0]["name"] == "column_sums"


def test_each_line_names_the_trace_and_the_pc_of_its_worst_access():
    small = facts(demos.transpose_trace(tiles_per_side=1))
    large = facts(demos.transpose_trace(tiles_per_side=2))
    data = summary.summarize([("s/trace-0.npz", "s", small), ("s/trace-1.npz", "s", large)])
    (line,) = shared_lines(data)
    assert line["worst"] == {
        "id": "s/trace-1.npz",
        "pc": hex(demos.PC_LOAD_TILE),
        "space": "shared",
    }
    (kernel,) = data["kernels"]
    assert kernel["worst"]["wavefronts"]["id"] == "s/trace-1.npz"
    assert kernel["worst"]["sectors"] is None


def test_a_line_that_follows_the_input_values_carries_the_spread_of_its_warps():
    data = summary.summarize([("h.npz", "h.npz", facts(demos.histogram_trace()))])
    (line,) = shared_lines(data)
    assert line["data_dependent"] is True
    assert line["kind"] == "atomic"
    fewest, most = line["excess_range"]
    assert fewest < most


def test_instructions_without_a_source_line_group_by_their_pc():
    rows = [
        Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(32))),
        Access(0x20, Space.SHARED, Kind.LOAD, 4, tuple(256 * lane for lane in range(32))),
    ]
    data = summary.summarize([("p.npz", "p.npz", facts(Trace.from_accesses(rows, {})))])
    assert [(line["line"], line["pc"]) for line in shared_lines(data)] == [
        (None, "0x10"),
        (None, "0x20"),
    ]
    assert data["without_lines"] == 1


def test_files_without_a_trace_and_launches_with_lost_records_count_apart():
    lossy = dataclasses.replace(facts(demos.transpose_trace()), dropped=12)
    data = summary.summarize([("bad.npz", "bad.npz", None), ("l.npz", "l.npz", lossy)])
    assert data["unreadable"] == 1
    assert data["traces"] == 1
    assert data["dropped"] == {"traces": 1, "records": 12}


def test_a_summary_keeps_only_the_worst_lines_of_each_unit(monkeypatch):
    monkeypatch.setattr(summary, "MAX_LINES", 1)
    items = [(name, name, facts(trace)) for name, trace in demos.demo_traces().items()]
    data = summary.summarize(items)
    assert len(shared_lines(data)) == 1
    assert len([line for line in data["lines"] if line["unit"] == "sectors"]) == 1


def test_kernel_names_in_a_summary_are_short_with_the_full_name_beside():
    long_name = "ns::detail::kernel<" + ", ".join(f"ns::T{index}<int>" for index in range(9)) + ">"
    renamed = dataclasses.replace(facts(demos.transpose_trace()), kernel=long_name)
    (kernel,) = summary.summarize([("k.npz", "k.npz", renamed)])["kernels"]
    assert kernel["name"] == "kernel<…>"
    assert kernel["full_name"] == long_name


def test_trace_files_skip_hidden_and_cache_folders_and_stop_at_the_limit(tmp_path):
    for folder in ("run", "run/.hidden", "run/__pycache__", "run/deeper"):
        (tmp_path / folder).mkdir(parents=True, exist_ok=True)
        (tmp_path / folder / "trace-0.npz").write_bytes(b"")
    (tmp_path / "run" / "notes.txt").write_text("")
    found = summary.trace_files(tmp_path / "run", 10)
    assert [path.relative_to(tmp_path).as_posix() for path in found] == [
        "run/trace-0.npz",
        "run/deeper/trace-0.npz",
    ]
    assert len(summary.trace_files(tmp_path / "run", 1)) == 1
    assert summary.trace_files(tmp_path / "run" / "trace-0.npz", 5) == [
        tmp_path / "run" / "trace-0.npz"
    ]


@pytest.mark.parametrize(
    ("name", "run"), [("trace-12.npz", "folder"), ("transpose.npz", "folder/transpose.npz")]
)
def test_a_recorded_trace_belongs_to_its_folder_and_any_other_file_to_itself(tmp_path, name, run):
    path = tmp_path / "folder" / name
    assert summary.run_of(path) == tmp_path / run


def test_read_facts_gives_none_for_a_file_that_holds_no_trace(tmp_path):
    broken = tmp_path / "broken.npz"
    broken.write_bytes(b"not a trace")
    assert summary.read_facts(broken) is None
    assert summary.read_facts(tmp_path / "missing.npz") is None


def test_the_text_of_a_summary_ranks_kernels_and_lines_and_names_the_next_command():
    items = [
        ("run/trace-0.npz", "run", facts(demos.transpose_trace())),
        ("run/trace-1.npz", "run", facts(demos.histogram_trace())),
    ]
    text = summary_text.render(summary.summarize(items))
    assert "run: 2 launches of 2 kernels" in text
    assert "Bank conflicts add" in text
    lines = text.splitlines()
    first = lines[lines.index("Lines by extra passes") + 2]
    assert "transpose" in first and "line 10 load" in first
    assert any("line 6 atomic *" in line for line in lines)
    assert "No padding or swizzle removes them." in text
    assert text.endswith(
        "Worst line: lanescope suggest run/trace-0.npz --pc 0x120 ranks its fixes, and "
        "lanescope gui run opens it in the browser."
    )


def test_the_text_of_global_memory_points_to_the_lanes_of_the_worst_line():
    items = [("c.npz", "c.npz", facts(demos.column_sums_trace()))]
    text = summary_text.render(summary.summarize(items), "sectors")
    assert "Scattered accesses add" in text
    assert "Worst line: lanescope lanes c.npz --pc 0xc0 shows its threads" in text


def test_the_notes_of_a_summary_name_missing_lines_lost_records_and_bad_files():
    lossy = dataclasses.replace(facts(demos.transpose_trace()), dropped=5, has_lines=False)
    notes = summary_text.notes(summary.summarize([("a", "a", lossy), ("b", "b", None)]))
    assert notes[0].startswith("No source lines in 1 of 1 launch. Build the program with -lineinfo")
    assert notes[1].startswith(
        "The record buffer filled up in 1 of 1 launch and left out 5 accesses"
    )
    assert notes[2] == "skipped 1 file without a trace"


def test_percentages_keep_one_decimal_and_mark_a_sliver():
    assert summary_text.percent(3968, 4096) == "96.9%"
    assert summary_text.percent(1, 100_000) == "<0.1%"
    assert summary_text.percent(1, 0) == "-"


class FakePool:
    """Stand in for a pool of processes: keep the traces it takes and how it shuts down."""

    made: list["FakePool"] = []

    def __init__(self, workers, mp_context=None):
        self.submitted = []
        self.cancelled = False
        FakePool.made.append(self)

    def submit(self, function, path):
        self.submitted.append(path)
        future = concurrent.futures.Future()
        future.set_result(f"facts of {path.name}")
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.cancelled |= cancel_futures


def test_reading_in_processes_holds_a_bounded_window_and_cancels_the_rest_when_closed(
    tmp_path, monkeypatch
):
    FakePool.made = []
    monkeypatch.setattr(summary.concurrent.futures, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(summary.os, "cpu_count", lambda: 2)
    paths = [tmp_path / f"trace-{index}.npz" for index in range(40)]
    reader = summary.read_many(paths, processes=True)
    first = [next(reader) for _ in range(3)]
    assert first == ["facts of trace-0.npz", "facts of trace-1.npz", "facts of trace-2.npz"]
    (pool,) = FakePool.made
    assert len(pool.submitted) == summary.WINDOW_PER_WORKER * 2 + 3
    reader.close()
    assert pool.cancelled


def test_the_instances_of_a_template_add_up_in_one_entry_that_lists_them():
    data = summary.summarize(two_instances())
    (entry,) = data["kernels"]
    assert (entry["template"], entry["name"], entry["full_name"]) == (
        "ns::tile",
        "tile",
        "ns::tile",
    )
    assert [found["full_name"] for found in entry["instances"]] == [TILE_8, TILE_4]
    assert [found["launches"] for found in entry["instances"]] == [2, 1]
    assert entry["launches"] == 3
    excess = [found["totals"]["wavefronts"]["excessive"] for found in entry["instances"]]
    assert entry["totals"]["wavefronts"]["excessive"] == sum(excess)
    assert entry["worst"]["wavefronts"] == {
        "id": "r/trace-1.npz",
        "pc": hex(demos.PC_LOAD_TILE),
        "space": "shared",
    }


def test_an_entry_of_one_instance_keeps_the_names_of_its_kernel():
    data = summary.summarize([("t.npz", "t.npz", instance(demos.transpose_trace(), TILE_4))])
    (entry,) = data["kernels"]
    assert (entry["template"], entry["name"], entry["full_name"]) == ("ns::tile", "tile<4>", TILE_4)
    (only,) = entry["instances"]
    assert (only["kernel"], only["name"], only["launches"]) == (TILE_4, "tile<4>", 1)


def test_a_source_line_adds_up_over_the_instances_of_its_template():
    data = summary.summarize(two_instances())
    (line,) = shared_lines(data)
    (entry,) = data["kernels"]
    assert (line["line"], line["template"], line["instances"], line["launches"]) == (
        10,
        "ns::tile",
        2,
        3,
    )
    assert line["excessive"] == entry["totals"]["wavefronts"]["excessive"]
    assert (line["kernel"], line["name"]) == (TILE_8, "tile")
    assert line["kernel_achieved"] == entry["totals"]["wavefronts"]["achieved"]
    assert line["worst"]["id"] == "r/trace-1.npz"


def test_lines_without_a_line_table_stay_apart_per_instance():
    rows = [Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(32)))]
    trace = Trace.from_accesses(rows, {})
    items = [("a.npz", "r", instance(trace, TILE_4)), ("b.npz", "r", instance(trace, TILE_8))]
    data = summary.summarize(items)
    assert len(data["kernels"]) == 1
    lines = shared_lines(data)
    assert sorted(line["kernel"] for line in lines) == [TILE_4, TILE_8]
    assert all((line["pc"], line["instances"]) == ("0x10", 1) for line in lines)


def test_without_templates_each_instance_stands_alone():
    data = summary.summarize(two_instances(), by_template=False)
    assert [(entry["name"], entry["template"]) for entry in data["kernels"]] == [
        ("tile<8>", "ns::tile"),
        ("tile<4>", "ns::tile"),
    ]
    assert all(len(entry["instances"]) == 1 for entry in data["kernels"])
    assert [line["name"] for line in shared_lines(data)] == ["tile<8>", "tile<4>"]


def test_the_text_lists_a_template_once_with_the_count_of_its_instances():
    text = summary_text.render(summary.summarize(two_instances()))
    lines = text.splitlines()
    assert lines[0] == "r: 3 launches of 2 kernels in 1 template"
    kernels = lines[lines.index("Kernels by extra passes") + 2 :]
    assert kernels[0].endswith("  tile (2 instances)")
    assert not kernels[1].strip()


def test_the_text_of_the_instances_view_lists_each_instance_as_its_own_kernel():
    text = summary_text.render(summary.summarize(two_instances(), by_template=False))
    lines = text.splitlines()
    assert lines[0] == "r: 3 launches of 2 kernels in 1 template"
    kernels = lines[lines.index("Kernels by extra passes") + 2 :][:2]
    assert [row.split()[-1] for row in kernels] == ["tile<8>", "tile<4>"]


def test_a_line_of_few_accesses_per_launch_follows_the_input_when_its_patterns_differ_each_time():
    launches = [scattered_bins(seed) for seed in (1, 2, 3)]
    assert not any(item.data_dependent for found in launches for item in found.instructions)
    data = summary.summarize((f"b/trace-{n}.npz", "b", found) for n, found in enumerate(launches))
    (line,) = shared_lines(data)
    assert line["data_dependent"] is True
    fewest, most = line["excess_range"]
    assert fewest < most


def test_a_line_whose_launches_repeat_their_patterns_does_not_follow_the_input():
    same = scattered_bins(1)
    data = summary.summarize((f"b/trace-{n}.npz", "b", same) for n in range(3))
    (line,) = shared_lines(data)
    assert line["data_dependent"] is False
    assert line["excess_range"] is None


def test_a_launch_with_too_many_patterns_to_keep_leaves_the_line_unmarked():
    launches = [scattered_bins(seed) for seed in (1, 2, 3)]
    (item,) = launches[0].instructions
    launches[0] = dataclasses.replace(
        launches[0], instructions=(dataclasses.replace(item, patterns=None),)
    )
    data = summary.summarize((f"b/trace-{n}.npz", "b", found) for n, found in enumerate(launches))
    assert shared_lines(data)[0]["data_dependent"] is False


def test_a_line_that_one_trace_marks_stays_marked_over_all_launches():
    marked = facts(demos.histogram_trace())
    unkept = tuple(dataclasses.replace(item, patterns=None) for item in marked.instructions)
    items = [
        ("h/trace-0.npz", "h", marked),
        ("h/trace-1.npz", "h", dataclasses.replace(marked, instructions=unkept)),
    ]
    (line,) = shared_lines(summary.summarize(items))
    assert line["data_dependent"] is True
