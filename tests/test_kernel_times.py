import json
import shutil
import sys

import pytest

from lanescope import kernel_times
from lanescope.cuda import toolchain
from lanescope.cuda.driver import Driver, DriverError
from lanescope.nsight import calibration
from lanescope.recorder import capture
from lanescope.recorder.build import build_kernel_timer, build_recorder
from lanescope.trace import read_metadata

# A stand-in for a timed program. It writes the file of the kernel timer itself, with the
# times that its argument names for each run, and logs each run to the file of LOG.
FAKE_PROGRAM = r"""
import json
import os
import pathlib
import sys

plan = json.loads(pathlib.Path(sys.argv[1]).read_text())
log = pathlib.Path(plan["log"])
runs = log.read_text().splitlines() if log.exists() else []
mine = [run for run in runs if run.startswith(plan["name"] + " ")]
with log.open("a") as file:
    file.write(f"{plan['name']} {os.getcwd()}\n")
kernels = plan["rounds"][len(mine) % len(plan["rounds"])]
if kernels is not None:
    path = f"{os.environ['LANESCOPE_KERNEL_TIMES']}.{os.getpid()}"
    written = {"format": 1, "pid": os.getpid(), "skipped": plan.get("skipped", 0),
               "dropped": 0, "kernels": kernels}
    pathlib.Path(path).write_text(json.dumps(written))
sys.stderr.write(plan.get("stderr", ""))
sys.exit(plan.get("status", 0))
"""


def times(launches, nanoseconds, shortest=None, longest=None):
    return {
        "launches": launches,
        "nanoseconds": nanoseconds,
        "shortest": shortest if shortest is not None else nanoseconds // max(launches, 1),
        "longest": longest if longest is not None else nanoseconds // max(launches, 1),
    }


def fake_folder(tmp_path, name, rounds, **plan):
    """Return a folder whose stored command is the fake program with these rounds of times."""
    script = tmp_path / "fake_program.py"
    script.write_text(FAKE_PROGRAM)
    folder = tmp_path / name
    folder.mkdir()
    plan_path = tmp_path / f"{name}-plan.json"
    plan_path.write_text(
        json.dumps({"name": name, "log": str(tmp_path / "log.txt"), "rounds": rounds} | plan)
    )
    kernel_times.write_stored_command(
        folder, [sys.executable, str(script), str(plan_path)], tmp_path
    )
    return folder


def logged_runs(tmp_path):
    return [line.split(" ")[0] for line in (tmp_path / "log.txt").read_text().splitlines()]


def test_write_run_stores_the_command_and_its_folder(tmp_path):
    kernel_times.write_stored_command(tmp_path / "run1", ["./app", "--size", "64"], tmp_path)
    assert json.loads((tmp_path / "run1" / "run.json").read_text()) == {
        "format": 1,
        "command": ["./app", "--size", "64"],
        "cwd": str(tmp_path),
    }
    assert kernel_times.read_stored_command(tmp_path / "run1") == (
        ["./app", "--size", "64"],
        tmp_path,
    )


def test_write_run_removes_the_times_of_an_earlier_command(tmp_path):
    (tmp_path / "kernel-times.json").write_text("{}")
    kernel_times.write_stored_command(tmp_path, ["./app"])
    assert not (tmp_path / "kernel-times.json").exists()


def test_read_run_of_a_folder_without_a_stored_command_names_both_remedies(tmp_path):
    with pytest.raises(ValueError, match="holds no run.json") as error:
        kernel_times.read_stored_command(tmp_path)
    assert "lanescope record" in str(error.value)
    assert f"lanescope time {tmp_path} -- ./my_app" in str(error.value)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        "[]",
        '{"format": 2, "command": ["./app"], "cwd": "/"}',
        '{"format": 1, "command": [], "cwd": "/"}',
        '{"format": 1, "command": "./app", "cwd": "/"}',
        '{"format": 1, "command": ["./app", 3], "cwd": "/"}',
        '{"format": 1, "command": ["./app"]}',
    ],
)
def test_read_run_rejects_a_stored_command_of_another_shape(tmp_path, text):
    (tmp_path / "run.json").write_text(text)
    with pytest.raises(ValueError, match="is no stored command"):
        kernel_times.read_stored_command(tmp_path)


def test_timing_runs_the_folders_in_turns_once_per_round(tmp_path):
    rounds = [{"k": times(1, 100)}]
    first = fake_folder(tmp_path, "first", rounds)
    second = fake_folder(tmp_path, "second", rounds)
    seen = []
    kernel_times.time_folders(
        [first, second], tmp_path / "timer.so", 3, lambda index, folder, _: seen.append(index)
    )
    assert logged_runs(tmp_path) == ["first", "second"] * 3
    assert seen == [0, 0, 1, 1, 2, 2]


def test_timing_writes_the_launches_and_nanoseconds_of_each_round(tmp_path):
    rounds = [
        {"k": times(2, 300, 100, 200), "j": times(1, 50)},
        {"k": times(2, 280, 120, 160), "j": times(1, 60)},
    ]
    folder = fake_folder(tmp_path, "run", rounds)
    (result,) = kernel_times.time_folders([folder], tmp_path / "timer.so", 2)
    written = json.loads((folder / "kernel-times.json").read_text())
    assert written == result
    assert written["format"] == 1
    assert written["rounds"] == 2
    assert len(written["wall_seconds"]) == 2
    assert all(seconds > 0 for seconds in written["wall_seconds"])
    assert written["untimed_launches"] == [0, 0]
    assert written["kernels"] == {
        "j": {"launches": [1, 1], "nanoseconds": [50, 60], "shortest": 50, "longest": 60},
        "k": {"launches": [2, 2], "nanoseconds": [300, 280], "shortest": 100, "longest": 200},
    }


def test_timing_runs_each_command_in_its_stored_folder(tmp_path):
    folder = fake_folder(tmp_path, "run", [{"k": times(1, 10)}])
    command, _ = kernel_times.read_stored_command(folder)
    kernel_times.write_stored_command(folder, command, folder)
    kernel_times.time_folders([folder], tmp_path / "timer.so", 1)
    assert (tmp_path / "log.txt").read_text() == f"run {folder}\n"


def test_a_kernel_missing_from_a_round_counts_no_launch_there(tmp_path):
    rounds = [{"k": times(1, 10), "j": times(1, 5)}, {"k": times(1, 12)}]
    folder = fake_folder(tmp_path, "run", rounds)
    (result,) = kernel_times.time_folders([folder], tmp_path / "timer.so", 2)
    assert result["kernels"]["j"]["launches"] == [1, 0]
    assert result["kernels"]["j"]["nanoseconds"] == [5, 0]


def test_the_times_of_every_process_of_a_run_add_up(tmp_path):
    folder = tmp_path / "run"
    script = tmp_path / "two_processes.py"
    script.write_text(
        "import json, os, sys\n"
        "base = os.environ['LANESCOPE_KERNEL_TIMES']\n"
        "for pid, ns in ((101, 300), (102, 100)):\n"
        "    kernels = {'k': {'launches': 1, 'nanoseconds': ns, 'shortest': ns, 'longest': ns}}\n"
        "    with open(f'{base}.{pid}', 'w') as file:\n"
        "        json.dump({'format': 1, 'skipped': 1, 'dropped': 2, 'kernels': kernels}, file)\n"
        "open(f'{base}.103.part', 'w').write('half written')\n"
    )
    kernel_times.write_stored_command(folder, [sys.executable, str(script)])
    (result,) = kernel_times.time_folders([folder], tmp_path / "timer.so", 1)
    assert result["kernels"]["k"] == {
        "launches": [2],
        "nanoseconds": [400],
        "shortest": 100,
        "longest": 300,
    }
    assert result["untimed_launches"] == [6]


@pytest.mark.parametrize(("status", "ending"), [(3, "exited with status 3"), (-9, "signal 9")])
def test_a_failing_command_names_its_folder_and_status_and_writes_no_times(
    tmp_path, status, ending
):
    first = fake_folder(tmp_path, "first", [{"k": times(1, 10)}])
    failing = fake_folder(tmp_path, "failing", [{"k": times(1, 10)}], stderr="out of memory\n")
    if status < 0:
        kernel_times.write_stored_command(failing, ["sh", "-c", f"kill -{-status} $$"], tmp_path)
    else:
        plan = json.loads((tmp_path / "failing-plan.json").read_text())
        (tmp_path / "failing-plan.json").write_text(json.dumps(plan | {"status": status}))
    with pytest.raises(ValueError) as error:
        kernel_times.time_folders([first, failing], tmp_path / "timer.so", 2)
    assert f"the command of {failing} in round 1 " in str(error.value)
    assert ending in str(error.value)
    if status > 0:
        assert str(error.value).endswith(":\nout of memory")
    assert not (first / "kernel-times.json").exists()
    assert not (failing / "kernel-times.json").exists()


def test_a_command_that_leaves_no_times_stops_the_timing(tmp_path):
    folder = fake_folder(tmp_path, "run", [{"k": times(1, 10)}, None])
    with pytest.raises(ValueError, match="in round 2 exited with status 0 and left no kernel"):
        kernel_times.time_folders([folder], tmp_path / "timer.so", 3)
    assert logged_runs(tmp_path) == ["run", "run"]
    assert not (folder / "kernel-times.json").exists()


def test_a_run_without_times_names_processes_that_end_without_exit_handlers(tmp_path):
    folder = fake_folder(tmp_path, "run", [None])
    with pytest.raises(ValueError) as error:
        kernel_times.time_folders([folder], tmp_path / "timer.so", 1)
    assert (
        "A process that ends through _exit or a signal, such as a worker of a multiprocessing "
        "Pool, may also have lost the times of the kernels it launched in its last 0.1 s"
    ) in str(error.value)


def test_a_timing_counts_the_processes_that_ended_without_exit_handlers(tmp_path):
    folder = tmp_path / "run"
    script = tmp_path / "three_processes.py"
    script.write_text(
        "import json, os\n"
        "base = os.environ['LANESCOPE_KERNEL_TIMES']\n"
        "kernels = {'k': {'launches': 1, 'nanoseconds': 5, 'shortest': 5, 'longest': 5}}\n"
        "for pid, exited in ((101, True), (102, False), (103, False)):\n"
        "    with open(f'{base}.{pid}', 'w') as file:\n"
        "        json.dump({'format': 1, 'exited': exited, 'kernels': kernels}, file)\n"
    )
    kernel_times.write_stored_command(folder, [sys.executable, str(script)])
    (result,) = kernel_times.time_folders([folder], tmp_path / "timer.so", 2)
    assert result["unfinished_processes"] == [2, 2]
    assert result["kernels"]["k"]["launches"] == [3, 3]


def test_a_timing_stores_a_new_command_with_its_times_once_every_round_succeeded(tmp_path):
    folder = fake_folder(tmp_path, "run", [{"k": times(1, 10)}])
    stored, _ = kernel_times.read_stored_command(folder)
    failing = [sys.executable, "-c", "import sys; sys.exit(3)"]
    with pytest.raises(ValueError, match="exited with status 3"):
        kernel_times.time_folders([folder], tmp_path / "timer.so", 1, commands=[failing])
    assert kernel_times.read_stored_command(folder)[0] == stored
    fresh = tmp_path / "fresh"
    (result,) = kernel_times.time_folders([fresh], tmp_path / "timer.so", 1, commands=[stored])
    assert kernel_times.read_stored_command(fresh)[0] == stored
    assert kernel_times.read_times(fresh) == result


def test_an_interrupted_write_of_the_times_keeps_the_earlier_ones(tmp_path, monkeypatch):
    folder = fake_folder(tmp_path, "run", [{"k": times(1, 10)}])
    (earlier,) = kernel_times.time_folders([folder], tmp_path / "timer.so", 1)
    kept = sorted(path.name for path in folder.iterdir())

    def interrupt(*_):
        raise KeyboardInterrupt

    monkeypatch.setattr(kernel_times.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        kernel_times.time_folders([folder], tmp_path / "timer.so", 2)
    assert kernel_times.read_times(folder) == earlier
    assert sorted(path.name for path in folder.iterdir()) == kept


def test_a_command_that_cannot_start_names_its_folder(tmp_path):
    kernel_times.write_stored_command(tmp_path, [str(tmp_path / "missing-program")])
    with pytest.raises(ValueError, match=f"the command of {tmp_path} cannot start"):
        kernel_times.time_folders([tmp_path], tmp_path / "timer.so", 1)


def test_a_timer_file_that_is_no_json_stops_the_timing(tmp_path):
    script = tmp_path / "broken.py"
    script.write_text(
        "import os\nopen(os.environ['LANESCOPE_KERNEL_TIMES'] + '.7', 'w').write('{')\n"
    )
    kernel_times.write_stored_command(tmp_path, [sys.executable, str(script)])
    with pytest.raises(ValueError, match="unreadable file"):
        kernel_times.time_folders([tmp_path], tmp_path / "timer.so", 1)


def test_the_timed_program_gets_the_timer_and_the_file_for_its_times(tmp_path):
    script = tmp_path / "environment.py"
    script.write_text(
        "import json, os\n"
        "base = os.environ['LANESCOPE_KERNEL_TIMES']\n"
        "kernel = os.environ['CUDA_INJECTION64_PATH']\n"
        "times = {'launches': 1, 'nanoseconds': 1, 'shortest': 1, 'longest': 1}\n"
        "open(f'{base}.1', 'w').write(json.dumps({'kernels': {kernel: times}}))\n"
    )
    kernel_times.write_stored_command(tmp_path, [sys.executable, str(script)])
    (result,) = kernel_times.time_folders([tmp_path], tmp_path / "timer.so", 1)
    assert list(result["kernels"]) == [str(tmp_path / "timer.so")]


def test_ranked_orders_the_kernels_by_their_median_time_over_the_rounds():
    result = {
        "rounds": 3,
        "kernels": {
            "short": times_over_rounds([1, 1, 1], [10, 30, 20]),
            "long": times_over_rounds([2, 2, 3], [100, 50, 60]),
        },
    }
    long, short = kernel_times.ranked(result)
    assert (long.name, long.median_ns, long.launches) == ("long", 60, 2)
    assert (long.fastest_round_ns, long.slowest_round_ns) == (50, 100)
    assert not long.even_launches
    assert (short.name, short.median_ns) == ("short", 20)
    assert short.even_launches


def times_over_rounds(launches, nanoseconds):
    return {"launches": launches, "nanoseconds": nanoseconds, "shortest": 1, "longest": 1}


def test_read_times_gives_the_times_that_time_folders_wrote(tmp_path):
    folder = fake_folder(tmp_path, "a", [{"k": times(2, 40)}, {"k": times(2, 60)}])
    (written,) = kernel_times.time_folders([folder], tmp_path / "timer.so", 2)
    assert kernel_times.read_times(folder) == written


def test_read_times_of_a_folder_without_times_gives_none(tmp_path):
    assert kernel_times.read_times(tmp_path) is None


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps({"format": 2, "rounds": 1, "kernels": {}}),
        json.dumps({"format": 1, "rounds": 2, "kernels": {"k": times_over_rounds([1], [5])}}),
        json.dumps({"format": 1, "rounds": 1, "kernels": {"k": times_over_rounds([1], ["5"])}}),
    ],
)
def test_read_times_names_a_file_that_holds_no_kernel_times(tmp_path, text):
    (tmp_path / "kernel-times.json").write_text(text)
    with pytest.raises(ValueError, match="kernel-times.json holds no kernel times"):
        kernel_times.read_times(tmp_path)


def test_round_totals_add_up_the_named_kernels_or_every_kernel_in_each_round():
    result = {
        "rounds": 2,
        "kernels": {
            "a": times_over_rounds([1, 1], [10, 20]),
            "b": times_over_rounds([1, 1], [1, 2]),
        },
    }
    assert kernel_times.round_totals(result) == [11, 22]
    assert kernel_times.round_totals(result, ["b", "missing"]) == [1, 2]


def test_median_total_is_the_median_of_the_round_sums_over_every_kernel():
    result = {
        "rounds": 3,
        "kernels": {
            "a": times_over_rounds([1, 1, 1], [40, 10, 50]),
            "b": times_over_rounds([1, 1, 1], [30, 50, 10]),
        },
    }
    assert kernel_times.median_total(result) == 60


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
def test_the_timer_times_every_kernel_that_a_recording_of_the_bench_traces(tmp_path):
    driver = Driver()
    arch = driver.arch()
    driver.close()
    cubin = tmp_path / "bench.cubin"
    cubin.write_bytes(calibration.compile_kernels(arch, TOOLS))
    command = [sys.executable, "-m", "lanescope", "bench", "run", "--cubin", str(cubin)]
    recording = tmp_path / "recording"
    built = build_recorder(tmp_path / "build", arch, TOOLS)
    assert capture.record_command(command, built, recording) == 0
    traced: dict[str, int] = {}
    for path in capture.import_recording(recording, TOOLS.disassemblers):
        kernel = read_metadata(path)["kernel"]
        traced[kernel] = traced.get(kernel, 0) + 1
    kernel_times.write_stored_command(recording, command)
    timer = build_kernel_timer(tmp_path / "build", TOOLS)
    (result,) = kernel_times.time_folders([recording], timer, 2)
    assert set(result["kernels"]) == set(traced)
    for name, launches in traced.items():
        assert result["kernels"][name]["launches"] == [launches, launches], name
        assert all(nanoseconds > 0 for nanoseconds in result["kernels"][name]["nanoseconds"])
    assert result["untimed_launches"] == [0, 0]


# Leaving the block terminates the worker, which ends without its exit handlers.
POOL_PROGRAM = r"""
import multiprocessing
import pathlib
import sys
import time

from lanescope.nsight import calibration


def run_bench(cubin):
    calibration.run_calibration(pathlib.Path(cubin))
    time.sleep(1)


if __name__ == "__main__":
    with multiprocessing.get_context(sys.argv[1]).Pool(1) as pool:
        pool.apply(run_bench, (sys.argv[2],))
"""


@needs_gpu
@pytest.mark.parametrize("start_method", ["spawn", "forkserver"])
def test_the_timer_keeps_the_times_of_a_pool_worker_that_ends_without_exit_handlers(
    tmp_path, start_method
):
    driver = Driver()
    arch = driver.arch()
    driver.close()
    cubin = tmp_path / "bench.cubin"
    cubin.write_bytes(calibration.compile_kernels(arch, TOOLS))
    script = tmp_path / "pool_bench.py"
    script.write_text(POOL_PROGRAM)
    direct, pooled = tmp_path / "direct", tmp_path / "pooled"
    command = [sys.executable, "-m", "lanescope", "bench", "run", "--cubin", str(cubin)]
    kernel_times.write_stored_command(direct, command)
    pool_command = [sys.executable, str(script), start_method, str(cubin)]
    kernel_times.write_stored_command(pooled, pool_command)
    timer = build_kernel_timer(tmp_path / "build", TOOLS)
    alone, in_worker = kernel_times.time_folders([direct, pooled], timer, 1)
    assert {name: kernel["launches"] for name, kernel in in_worker["kernels"].items()} == {
        name: kernel["launches"] for name, kernel in alone["kernels"].items()
    }
    assert in_worker["unfinished_processes"] == [1]
    assert alone["unfinished_processes"] == [0]
