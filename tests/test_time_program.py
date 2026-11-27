import json
import sys

import pytest

from lanescope import kernel_times
from lanescope.cli import main
from lanescope.commands import time_program
from lanescope.cuda import toolchain
from lanescope.summary_text import duration, duration_range

# A stand-in for a timed program that writes the file of the kernel timer itself. Its
# argument holds the kernels of each run, taken in turn.
FAKE_PROGRAM = r"""
import json
import os
import pathlib
import sys

plan = pathlib.Path(sys.argv[1])
rounds = json.loads(plan.read_text())
counter = plan.with_suffix(".count")
done = int(counter.read_text()) if counter.exists() else 0
counter.write_text(str(done + 1))
path = f"{os.environ['LANESCOPE_KERNEL_TIMES']}.{os.getpid()}"
pathlib.Path(path).write_text(json.dumps({"kernels": rounds[done % len(rounds)]}))
"""


def kernel(launches, nanoseconds):
    each = nanoseconds // launches
    return {"launches": launches, "nanoseconds": nanoseconds, "shortest": each, "longest": each}


@pytest.fixture
def no_timer_build(monkeypatch, tmp_path):
    built = []

    def build(out_dir, cache=None):
        built.append(out_dir)
        return tmp_path / "timer.so"

    monkeypatch.setattr(time_program, "build_kernel_timer", build)
    return built


def fake_command(tmp_path, name, rounds):
    script = tmp_path / "fake_program.py"
    script.write_text(FAKE_PROGRAM)
    plan = tmp_path / f"{name}.json"
    plan.write_text(json.dumps(rounds))
    return [sys.executable, str(script), str(plan)]


def stored_folder(tmp_path, name, rounds):
    folder = tmp_path / name
    kernel_times.write_stored_command(folder, fake_command(tmp_path, name, rounds))
    return folder


def test_time_prints_the_kernels_of_each_folder_by_median_time(tmp_path, no_timer_build, capsys):
    rounds = [
        {"transpose": kernel(2, 4000), "scale": kernel(1, 900)},
        {"transpose": kernel(2, 5000), "scale": kernel(1, 1000)},
        {"transpose": kernel(2, 4600), "scale": kernel(1, 1100)},
    ]
    folder = stored_folder(tmp_path, "run1", rounds)
    assert main(["time", str(folder), "--rounds", "3"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert [line.split(":")[0] for line in out[:3]] == [f"round {n} of 3" for n in (1, 2, 3)]
    table = out[out.index("") + 1 :]
    assert table[0] == f"{folder}: 2 kernels, {duration(5700)} of kernel time, median of 3 rounds"
    assert table[1].split() == "median of time launches range over rounds kernel".split()
    transpose_range = duration_range(4000, 5000, 4600).split()
    assert table[2].split() == [
        *duration(4600).split(),
        "80.7%",
        "2",
        *transpose_range,
        "transpose",
    ]
    scale_range = duration_range(900, 1100, 1000).split()
    assert table[3].split() == [*duration(1000).split(), "17.5%", "1", *scale_range, "scale"]
    assert table[4] == f"wrote {folder / 'kernel-times.json'}"
    assert json.loads((folder / "kernel-times.json").read_text())["rounds"] == 3


def test_time_gives_the_median_of_the_round_sums_as_the_kernel_time_of_a_folder(
    tmp_path, no_timer_build, capsys
):
    rounds = [
        {"transpose": kernel(1, 4000), "scale": kernel(1, 3000)},
        {"transpose": kernel(1, 1000), "scale": kernel(1, 5000)},
        {"transpose": kernel(1, 5000), "scale": kernel(1, 1000)},
    ]
    folder = stored_folder(tmp_path, "run", rounds)
    assert main(["time", str(folder), "--rounds", "3"]) == 0
    out = capsys.readouterr().out.splitlines()
    header = f"{folder}: 2 kernels, {duration(6000)} of kernel time, median of 3 rounds"
    assert header in out


def test_time_names_one_kernel_in_the_singular(tmp_path, no_timer_build, capsys):
    folder = stored_folder(tmp_path, "run", [{"k": kernel(1, 10)}])
    assert main(["time", str(folder), "--rounds", "1"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert f"{folder}: 1 kernel, {duration(10)} of kernel time, median of 1 round" in out


def test_time_takes_five_rounds_by_default_in_turns_over_the_folders(
    tmp_path, no_timer_build, capsys
):
    first = stored_folder(tmp_path, "first", [{"k": kernel(1, 10)}])
    second = stored_folder(tmp_path, "second", [{"k": kernel(1, 20)}])
    assert main(["time", str(first), str(second)]) == 0
    progress = [line for line in capsys.readouterr().out.splitlines() if line.startswith("round")]
    assert [line.split(" ran in")[0] for line in progress] == [
        f"round {n} of 5: {folder}" for n in range(1, 6) for folder in (first, second)
    ]
    assert len(no_timer_build) == 1
    assert not no_timer_build[0].exists()


def test_time_with_a_program_after_the_folder_stores_it_as_the_command(
    tmp_path, no_timer_build, capsys
):
    command = fake_command(tmp_path, "plan", [{"k": kernel(1, 10)}])
    folder = tmp_path / "fresh"
    assert main(["time", str(folder), "--rounds", "2", "--", *command]) == 0
    assert kernel_times.read_stored_command(folder)[0] == command
    assert json.loads((folder / "kernel-times.json").read_text())["rounds"] == 2


def test_time_takes_the_rounds_option_between_the_folders(tmp_path, no_timer_build, capsys):
    first = stored_folder(tmp_path, "first", [{"k": kernel(1, 10)}])
    second = stored_folder(tmp_path, "second", [{"k": kernel(1, 20)}])
    assert main(["time", str(first), "--rounds", "2", str(second)]) == 0
    for folder in (first, second):
        assert json.loads((folder / "kernel-times.json").read_text())["rounds"] == 2


def test_a_failed_timing_of_a_new_program_keeps_the_stored_command_and_its_times(
    tmp_path, no_timer_build, capsys
):
    folder = stored_folder(tmp_path, "run", [{"k": kernel(1, 10)}])
    assert main(["time", str(folder), "--rounds", "1"]) == 0
    stored = {name: (folder / name).read_text() for name in ("run.json", "kernel-times.json")}
    failing = [sys.executable, "-c", "import sys; sys.exit(3)"]
    assert main(["time", str(folder), "--rounds", "1", "--", *failing]) == 2
    assert "exited with status 3" in capsys.readouterr().err
    assert {path.name: path.read_text() for path in folder.iterdir()} == stored


def test_a_failed_timing_of_a_program_stores_no_command_in_a_new_folder(
    tmp_path, no_timer_build, capsys
):
    folder = tmp_path / "fresh"
    assert main(["time", str(folder), "--", sys.executable, "-c", "import sys; sys.exit(3)"]) == 2
    assert not folder.exists()


def test_time_warns_when_a_process_ended_without_its_exit_handlers(
    tmp_path, no_timer_build, capsys
):
    script = tmp_path / "unfinished.py"
    script.write_text(
        "import json, os\n"
        "times = {'k': {'launches': 1, 'nanoseconds': 5, 'shortest': 5, 'longest': 5}}\n"
        "base = os.environ['LANESCOPE_KERNEL_TIMES']\n"
        "for pid, exited in ((11, True), (12, False)):\n"
        "    written = {'exited': exited, 'kernels': times}\n"
        "    open(f'{base}.{pid}', 'w').write(json.dumps(written))\n"
    )
    folder = tmp_path / "run"
    assert main(["time", str(folder), "--rounds", "2", "--", sys.executable, str(script)]) == 0
    err = capsys.readouterr().err
    assert (
        f"lanescope: 2 processes in the 2 rounds of {folder} ended through _exit or a signal, "
        "without the exit handler of the timer. Their kernel times stop at the last flush of "
        "the timer, which runs every 0.1 s." in err
    )


def test_time_with_a_program_takes_exactly_one_folder(tmp_path, no_timer_build, capsys):
    assert main(["time", str(tmp_path / "a"), str(tmp_path / "b"), "--", "./app"]) == 2
    assert "a program after -- takes exactly one folder" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


def test_time_needs_a_folder(capsys):
    with pytest.raises(SystemExit) as stop:
        main(["time", "--", "./app"])
    assert stop.value.code == 2
    assert "lanescope time" in capsys.readouterr().err


def test_time_of_a_folder_without_a_stored_command_fails_before_the_build(
    tmp_path, no_timer_build, capsys
):
    assert main(["time", str(tmp_path)]) == 2
    assert f"{tmp_path} holds no run.json" in capsys.readouterr().err
    assert no_timer_build == []


def test_time_warns_when_the_launches_of_a_kernel_differ_between_rounds(
    tmp_path, no_timer_build, capsys
):
    folder = stored_folder(tmp_path, "run", [{"k": kernel(1, 10)}, {"k": kernel(2, 20)}])
    assert main(["time", str(folder), "--rounds", "3"]) == 0
    err = capsys.readouterr().err
    assert f"the launches of k in {folder} differ between rounds (1, 2, 1)" in err


def test_time_without_cupti_names_the_missing_tools(tmp_path, monkeypatch, capsys):
    folder = stored_folder(tmp_path, "run", [{"k": kernel(1, 10)}])
    empty = toolchain.Toolchain(*[None] * 9, disassemblers=())
    monkeypatch.setattr(time_program, "cache_folder", lambda: None)
    monkeypatch.setattr("lanescope.recorder.build.find_toolchain", lambda: empty)
    assert main(["time", str(folder)]) == 2
    err = capsys.readouterr().err
    assert "missing CUDA tools: cupti_include, cupti_lib, cuda_include" in err
    assert "pip install 'lanescope[gpu]'" in err
