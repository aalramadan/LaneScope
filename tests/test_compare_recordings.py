import json
import pathlib
import shutil
import sys

import pytest

from lanescope import demos, kernel_times
from lanescope.cli import main
from lanescope.cuda import toolchain
from lanescope.cuda.driver import Driver, DriverError
from lanescope.fixes import layouts
from lanescope.recorder import capture
from lanescope.recorder.build import build_kernel_timer, build_recorder

CONFLICT_KINDS = pathlib.Path(__file__).parent.parent / "examples" / "conflict_kinds.py"


def padded_transpose():
    """Return the transpose with each row of its tile padded by one float, free of conflicts."""
    trace = demos.transpose_trace()
    return layouts.apply_layout_change(trace, layouts.infer_buffer(trace), layouts.Padding(128, 4))


def write_times(folder, **kernels):
    """Write the kernel-times.json of lanescope time, with the nanoseconds of each round."""
    rounds = len(next(iter(kernels.values())))
    result = {
        "format": 1,
        "rounds": rounds,
        "kernels": {
            name: {"launches": [1] * rounds, "nanoseconds": values, "shortest": 1, "longest": 1}
            for name, values in kernels.items()
        },
    }
    (folder / "kernel-times.json").write_text(json.dumps(result))


@pytest.fixture
def sides(tmp_path):
    """Return two folders: the transpose and a histogram, then the padded transpose and a sum."""
    before, after = tmp_path / "before", tmp_path / "after"
    for folder in (before, after):
        folder.mkdir()
        demos.histogram_trace().save(folder / "trace-1.npz")
    demos.transpose_trace().save(before / "trace-0.npz")
    padded_transpose().save(after / "trace-0.npz")
    demos.row_sums_trace().save(before / "trace-2.npz")
    return before, after


def test_compare_prints_the_passes_and_the_times_of_each_kernel_before_and_after(sides, capsys):
    before, after = sides
    write_times(
        before,
        transpose=[1_600_000, 1_700_000, 1_650_000],
        histogram=[500_000, 520_000, 510_000],
        row_sums=[10_000, 10_000, 10_000],
    )
    write_times(after, transpose=[410_000, 420_000, 400_000], histogram=[505_000, 515_000, 500_000])
    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[:4] == [
        f"before: {before}, kernel times of 3 rounds",
        f"after: {after}, kernel times of 3 rounds",
        "Shared memory: 6,894 passes before and 622 after. Bank conflicts add 5,934 before and "
        "174 after.",
        "Kernel time: 2.17 ms before and 915 us after, a speedup of 2.37x.",
    ]
    counts = out[out.index("Extra passes by kernel") + 1 :]
    assert counts[0].split() == (
        "launches passes before passes after extra before extra after kernel".split()
    )
    assert counts[1].split() == ["1", "4,224", "256", "3,968", "0", "transpose"]
    assert counts[3].split() == ["3/2", "6,894", "622", "5,934", "174", "all", "kernels"]
    times = out[out.index("Kernel time by kernel") + 1 :]
    assert times[1].split() == [
        "1.65", "ms", "1.60-1.70", "ms", "410", "us", "400-420", "us", "4.02x", "transpose"
    ]  # fmt: skip
    assert times[2].split()[-3:] == ["within", "spread", "histogram"]
    assert any(line.startswith("within spread: the rounds of the two sides") for line in out)
    alone = out[out.index("Kernels of one side alone") + 2]
    assert alone.split() == ["before", "only", "1", "1,792", "10.0", "us", "row_sums"]
    assert not any(line.startswith("Next step") for line in out)


def test_compare_without_times_names_lanescope_time_as_the_next_step(sides, capsys):
    before, after = sides
    for folder in sides:
        kernel_times.write_stored_command(folder, ["./my_app"])
    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out
    assert "Kernel time by kernel" not in out
    assert out.rstrip().endswith(
        f"Next step: lanescope time {before} {after} times the kernels of both sides in turns, "
        "and lanescope compare then gives their times."
    )


def test_compare_of_trace_files_names_their_folders_for_lanescope_time(tmp_path, capsys):
    for name, trace in (("a", demos.transpose_trace()), ("b", padded_transpose())):
        (tmp_path / name).mkdir()
        trace.save(tmp_path / name / "transpose.npz")
        kernel_times.write_stored_command(tmp_path / name, ["./my_app"])
    files = [str(tmp_path / name / "transpose.npz") for name in ("a", "b")]
    assert main(["compare", *files]) == 0
    out = capsys.readouterr().out
    assert f"Next step: lanescope time {tmp_path / 'a'} {tmp_path / 'b'} times" in out


def test_compare_of_folders_without_their_command_names_the_step_that_times_each(sides, capsys):
    before, after = sides
    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out.rstrip().splitlines()
    assert out[-2:] == [
        f"Next step for the before side: {before} holds no run.json with the command of its "
        f"recording. lanescope time {before} -- ./my_app times the kernels of the program given "
        f"after --, and a new recording with lanescope record --out {before} keeps its command "
        "for lanescope time.",
        f"Next step for the after side: {after} holds no run.json with the command of its "
        f"recording. lanescope time {after} -- ./my_app times the kernels of the program given "
        f"after --, and a new recording with lanescope record --out {after} keeps its command "
        "for lanescope time.",
    ]


def test_compare_names_lanescope_time_for_the_untimed_side_alone_when_the_other_lacks_run_json(
    sides, capsys
):
    before, after = sides
    write_times(before, transpose=[1_000])
    kernel_times.write_stored_command(after, ["./my_app"])
    assert main(["compare", str(before), str(after)]) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == (
        f"Next step for the after side: lanescope time {after} times its kernels with the "
        "command of its recording, and lanescope compare then gives their times."
    )


def test_compare_lists_an_instance_of_one_side_alone_and_compares_the_shared_ones(tmp_path, capsys):
    before, after = tmp_path / "before", tmp_path / "after"
    scale_4, scale_8 = "void scale<4>(float*)", "void scale<8>(float*)"
    for folder, kernels in ((before, (scale_4, scale_8)), (after, (scale_4,))):
        folder.mkdir()
        for index, kernel in enumerate(kernels):
            trace = demos.transpose_trace()
            trace.metadata["kernel"] = kernel
            trace.save(folder / f"trace-{index}.npz")
        write_times(folder, **{kernel: [4_200, 4_300, 4_250] for kernel in kernels})
    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out.splitlines()
    counts = out[out.index("Extra passes by kernel") + 2 :]
    assert counts[0].split() == "1 4,224 4,224 3,968 3,968 scale<4> (shared instances)".split()
    times = out[out.index("Kernel time by kernel") + 2 :]
    assert times[0].split()[-5:] == "within spread scale<4> (shared instances)".split()
    assert any(line.startswith("shared instances: the row adds up only") for line in out)
    alone = out[out.index("Kernels of one side alone") + 2]
    assert alone.split() == ["before", "only", "1", "3,968", "4.25", "us", "scale<8>"]


def test_compare_keeps_the_counts_of_a_side_whose_file_of_kernel_times_is_broken(sides, capsys):
    before, after = sides
    (after / "kernel-times.json").write_text('{"format": 1, "rounds": 2, "kern')
    assert main(["compare", str(before), str(after)]) == 0
    captured = capsys.readouterr()
    assert "Shared memory: 6,894 passes before and 622 after." in captured.out
    assert f"lanescope: {after / 'kernel-times.json'} holds no kernel times (" in captured.err
    assert f"The counts of {after} stay, without its kernel times." in captured.err
    assert main(["compare", str(before), str(after), "--json"]) == 0
    (unread,) = json.loads(capsys.readouterr().out)["unreadable_times"]
    assert unread["file"] == str(after / "kernel-times.json")


def test_compare_of_global_memory_lists_the_kernels_with_sectors(sides, capsys):
    before, after = sides
    assert main(["compare", str(before), str(after), "--global"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[2].startswith("Global memory: ")
    counts = out[out.index("Extra sectors by kernel") + 1 :]
    assert counts[0].split()[1:3] == ["sectors", "before"]


def test_compare_limits_the_kernels_it_lists(sides, capsys):
    before, after = sides
    assert main(["compare", str(before), str(after), "--limit", "1"]) == 0
    out = capsys.readouterr().out.splitlines()
    counts = out[out.index("Extra passes by kernel") + 2 :]
    assert counts[0].split()[-1] == "transpose"
    assert counts[1].split() == ["and", "1", "more", "kernel"]


def test_compare_json_gives_the_whole_comparison_for_tools(sides, capsys):
    before, after = sides
    assert main(["compare", str(before), str(after), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert (data["before"], data["after"]) == (str(before), str(after))
    assert [row["name"] for row in data["kernels"]] == ["transpose", "histogram"]
    assert data["total"]["after"]["launches"] == 2


def test_compare_names_a_side_without_traces(sides, tmp_path, capsys):
    before, _ = sides
    assert main(["compare", str(before), str(tmp_path / "missing")]) == 2
    assert f"no such file or folder: {tmp_path / 'missing'}" in capsys.readouterr().err


def _has_gpu() -> bool:
    try:
        Driver().close()
    except DriverError:
        return False
    return True


TOOLS = toolchain.find_toolchain()
needs_gpu = pytest.mark.skipif(
    TOOLS.nvrtc is None or TOOLS.cupti_lib is None or shutil.which("g++") is None or not _has_gpu(),
    reason="NVRTC, CUPTI, g++, and a GPU are needed",
)


@needs_gpu
def test_the_example_and_its_padded_tile_timed_together_compare_kernel_by_kernel(tmp_path, capsys):
    driver = Driver()
    arch = driver.arch()
    driver.close()
    built = build_recorder(tmp_path / "build", arch, TOOLS)
    plain = [sys.executable, str(CONFLICT_KINDS)]
    folders = [tmp_path / "before", tmp_path / "after"]
    for folder, command in zip(folders, (plain, [*plain, "--padded"]), strict=True):
        assert capture.record_command(command, built, folder) == 0
        capture.import_recording(folder, TOOLS.disassemblers)
        kernel_times.write_stored_command(folder, command)
    timer = build_kernel_timer(tmp_path / "build", TOOLS)
    kernel_times.time_folders(folders, timer, 2)
    assert main(["compare", *map(str, folders), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert sorted(row["name"] for row in data["kernels"]) == [
        "histogram",
        "scale",
        "transpose",
        "tree_sum",
    ]
    assert data["only_before"] == data["only_after"] == []
    transpose = next(row for row in data["kernels"] if row["name"] == "transpose")
    assert transpose["before"]["wavefronts"]["excessive"] == 31_744
    assert transpose["after"]["wavefronts"]["excessive"] == 0
    for row in data["kernels"]:
        assert row["before"]["launches"] == row["after"]["launches"]
        assert row["before"]["time_ns"] > 0 and row["after"]["time_ns"] > 0
        assert row["time"]["speedup"] is not None
    assert main(["compare", *map(str, folders)]) == 0
    out = capsys.readouterr().out
    assert "Kernel time by kernel" in out
    assert "Next step" not in out
