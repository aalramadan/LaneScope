import json
import sys

import pytest

from lanescope import kernel_times
from lanescope.cli import main
from lanescope.commands import time_program
from lanescope.cuda import toolchain

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
    kernel_times.write_run(folder, fake_command(tmp_path, name, rounds))
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
    assert table[0] == f"{folder}: 2 kernels, 5.6 us of kernel time, median of 3 rounds"
    assert table[1].split() == "median of time launches range over rounds kernel".split()
    assert table[2].split() == ["4.6", "us", "82.1%", "2", "4.0-5.0", "us", "transpose"]
    assert table[3].split() == ["1.0", "us", "17.9%", "1", "0.9-1.1", "us", "scale"]
    assert table[4] == f"wrote {folder / 'kernel-times.json'}"
    assert json.loads((folder / "kernel-times.json").read_text())["rounds"] == 3


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
    assert no_timer_build == [first / "build"]


def test_time_with_a_program_after_the_folder_stores_it_as_the_command(
    tmp_path, no_timer_build, capsys
):
    command = fake_command(tmp_path, "plan", [{"k": kernel(1, 10)}])
    folder = tmp_path / "fresh"
    assert main(["time", str(folder), "--rounds", "2", "--", *command]) == 0
    assert kernel_times.read_run(folder)[0] == command
    assert json.loads((folder / "kernel-times.json").read_text())["rounds"] == 2


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
    monkeypatch.setattr("lanescope.recorder.build.find", lambda: empty)
    assert main(["time", str(folder)]) == 2
    err = capsys.readouterr().err
    assert "missing CUDA tools: cupti_include, cupti_lib, cuda_include" in err
    assert "pip install 'lanescope[gpu]'" in err
