import concurrent.futures
import dataclasses
import json

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
    text = summary_text.render_summary_text(summary.summarize(items))
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
    text = summary_text.render_summary_text(summary.summarize(items), "sectors")
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
    text = summary_text.render_summary_text(summary.summarize(two_instances()))
    lines = text.splitlines()
    assert lines[0] == "r: 3 launches of 2 kernels in 1 template"
    kernels = lines[lines.index("Kernels by extra passes") + 2 :]
    assert kernels[0].endswith("  tile (2 instances)")
    assert not kernels[1].strip()


def test_the_text_of_the_instances_view_lists_each_instance_as_its_own_kernel():
    text = summary_text.render_summary_text(summary.summarize(two_instances(), by_template=False))
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


def column_loads(origins):
    """Return the facts of two conflicted loads on call line 7, with the origins they are given."""
    rows = [
        Access(pc, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(WARP_SIZE)))
        for pc in (0x10, 0x20)
    ]
    metadata = {
        "kernel": "sort",
        "source_file": "/src/sort.cu",
        "source": [""] * 6 + ["    block_sort(keys);"],
        "lines": {"0x10": 7, "0x20": 7},
        "opcodes": {"0x10": "LDS", "0x20": "LDS"},
    }
    if origins:
        metadata |= {
            "origin_files": ["/src/include/block_sort.cuh"],
            "origins": {pc: [0, line] for pc, line in origins.items()},
            "origin_text": [{"41": "  a = keys[i];", "58": "  b = keys[j];"}],
        }
    return facts(Trace.from_accesses(rows, metadata))


def test_instructions_inlined_from_two_lines_of_a_header_stay_apart_under_their_origins():
    data = summary.summarize([("s.npz", "s.npz", column_loads({"0x10": 41, "0x20": 58}))])
    lines = shared_lines(data)
    assert [line["origin"] for line in lines] == [
        {"file": "/src/include/block_sort.cuh", "line": 41, "text": "a = keys[i];"},
        {"file": "/src/include/block_sort.cuh", "line": 58, "text": "b = keys[j];"},
    ]
    assert all(
        (line["file"], line["line"], line["source"]) == ("/src/sort.cu", 7, "block_sort(keys);")
        for line in lines
    )


def test_instructions_inlined_from_one_line_of_a_header_add_up_in_one_line():
    data = summary.summarize([("s.npz", "s.npz", column_loads({"0x10": 41, "0x20": 41}))])
    (line,) = shared_lines(data)
    assert line["origin"]["line"] == 41
    assert line["excessive"] == 2 * 31


def test_instructions_of_one_call_line_without_origins_add_up_in_the_call_line():
    data = summary.summarize([("s.npz", "s.npz", column_loads({}))])
    (line,) = shared_lines(data)
    assert (line["file"], line["line"], line["origin"]) == ("/src/sort.cu", 7, None)
    assert line["excessive"] == 2 * 31


def test_an_inlined_line_of_one_template_adds_up_over_its_instances():
    loads = column_loads({"0x10": 41, "0x20": 58})
    items = [
        ("r/trace-0.npz", "r", dataclasses.replace(loads, kernel=TILE_4)),
        ("r/trace-1.npz", "r", dataclasses.replace(loads, kernel=TILE_8)),
    ]
    lines = shared_lines(summary.summarize(items))
    assert [(line["origin"]["line"], line["instances"]) for line in lines] == [(41, 2), (58, 2)]


def test_the_text_names_the_header_line_that_an_inlined_line_comes_from():
    text = summary_text.render_summary_text(
        summary.summarize([("s.npz", "s.npz", column_loads({"0x10": 41, "0x20": 58}))])
    )
    lines = text.splitlines()
    rows = lines[lines.index("Lines by extra passes") + 2 :][:2]
    assert [row.split()[4:6] for row in rows] == [
        ["block_sort.cuh:41", "load"],
        ["block_sort.cuh:58", "load"],
    ]


@pytest.mark.parametrize(
    ("nanoseconds", "text"),
    [(512, "512 ns"), (1_500, "1.5 us"), (2_340_000, "2.3 ms"), (3_100_000_000, "3.1 s")],
)
def test_durations_take_the_largest_unit_that_keeps_a_whole_number(nanoseconds, text):
    assert summary_text.duration(nanoseconds) == text


def test_a_range_takes_the_unit_of_its_slowest_round():
    assert summary_text.duration_range(900, 1_200) == "0.9-1.2 us"


def timing(**kernels):
    """Return the kernel times of lanescope time, with the nanoseconds of each kernel per round."""
    rounds = len(next(iter(kernels.values())))
    return {
        "format": 1,
        "rounds": rounds,
        "kernels": {
            name: {
                "launches": [1] * rounds,
                "nanoseconds": list(values),
                "shortest": 1,
                "longest": 9,
            }
            for name, values in kernels.items()
        },
    }


def test_each_kernel_of_a_timed_run_takes_its_median_time_over_the_rounds():
    items = [("r/trace-0.npz", "r", facts(demos.transpose_trace()))]
    times = {"r": timing(transpose=[400, 500, 460], copy=[100, 100, 100])}
    data = summary.summarize(items, times=times)
    (kernel,) = data["kernels"]
    assert (kernel["time_ns"], kernel["timed_launches"], kernel["round_ns"]) == (
        460,
        1,
        [400, 500, 460],
    )
    assert kernel["instances"][0]["time_ns"] == 460
    assert data["times"] == [
        {"run": "r", "rounds": 3, "total_ns": 560, "round_ns": [500, 600, 560]}
    ]


def test_a_template_takes_the_median_of_the_rounds_of_its_instances_added_up():
    times = {"r": timing(**{TILE_4: [10, 30, 20], TILE_8: [300, 100, 200]})}
    (entry,) = summary.summarize(two_instances(), times=times)["kernels"]
    assert entry["round_ns"] == [310, 130, 220]
    assert (entry["time_ns"], entry["timed_launches"]) == (220, 2)
    assert [found["time_ns"] for found in entry["instances"]] == [200, 20]


def test_a_kernel_without_a_time_and_a_run_without_times_give_none():
    items = [
        ("a/trace-0.npz", "a", facts(demos.transpose_trace())),
        ("b/trace-0.npz", "b", facts(demos.histogram_trace())),
    ]
    data = summary.summarize(items, times={"a": timing(other=[5])})
    assert all(kernel["time_ns"] is None for kernel in data["kernels"])
    assert [timed["run"] for timed in data["times"]] == ["a"]
    assert summary.summarize(items)["times"] == []


def test_the_text_of_a_timed_run_gives_the_time_of_each_kernel_and_its_part_of_the_run():
    items = [
        ("r/trace-0.npz", "r", facts(demos.transpose_trace())),
        ("r/trace-1.npz", "r", facts(demos.histogram_trace())),
    ]
    times = {"r": timing(transpose=[1_500_000], histogram=[2_500_000], scale=[1_000_000])}
    lines = summary_text.render_summary_text(summary.summarize(items, times=times)).splitlines()
    assert lines[1] == "Kernel time: 5.0 ms, the median of 1 round without the recorder."
    table = lines[lines.index("Kernels by extra passes") + 1 :]
    assert table[0].split() == "extra passes of kernel launches time of run kernel".split()
    assert table[1].split()[-4:] == ["1.5", "ms", "30.0%", "transpose"]
    assert table[2].split()[-4:] == ["2.5", "ms", "50.0%", "histogram"]


def test_the_text_orders_the_kernels_by_time_on_request():
    items = [
        ("r/trace-0.npz", "r", facts(demos.transpose_trace())),
        ("r/trace-1.npz", "r", facts(demos.histogram_trace())),
        ("r/trace-2.npz", "r", facts(demos.column_sums_trace())),
    ]
    times = {"r": timing(transpose=[1_000], histogram=[3_000])}
    lines = summary_text.render_summary_text(
        summary.summarize(items, times=times), order="time"
    ).splitlines()
    table = lines[lines.index("Kernels by time") + 2 :][:3]
    assert [row.split()[-1] for row in table] == ["histogram", "transpose", "column_sums"]
    assert table[2].split()[-3:] == ["-", "-", "column_sums"]


def test_the_text_of_an_untimed_run_has_no_time_column():
    data = summary.summarize([("t.npz", "t.npz", facts(demos.transpose_trace()))])
    text = summary_text.render_summary_text(data)
    assert "Kernel time" not in text
    assert "of run" not in text


def test_the_times_of_a_recording_lie_in_its_folder_and_those_of_a_trace_file_beside_it(tmp_path):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "kernel-times.json").write_text(json.dumps(timing(k=[7])))
    found = summary.run_times(
        {"run": tmp_path / "run", "demo": tmp_path / "run" / "demo.npz", "bare": tmp_path}
    )
    assert sorted(found) == ["demo", "run"]
    assert found["run"]["kernels"]["k"]["nanoseconds"] == [7]


def test_a_summary_of_files_takes_one_run_for_all_on_request(tmp_path):
    folder = tmp_path / "side"
    (folder / "nested").mkdir(parents=True)
    demos.transpose_trace().save(folder / "trace-0.npz")
    demos.transpose_trace().save(folder / "nested" / "trace-0.npz")
    (folder / "kernel-times.json").write_text(json.dumps(timing(transpose=[9])))
    paths = summary.trace_files(folder, 10)
    found = [summary.read_facts(path) for path in paths]
    apart = summary.summarize_files(paths, found)
    assert apart["runs"] == [str(folder), str(folder / "nested")]
    together = summary.summarize_files(paths, found, run=folder)
    assert together["runs"] == [str(folder)]
    assert together["kernels"][0]["launches"] == 2
    assert together["kernels"][0]["time_ns"] == 9
