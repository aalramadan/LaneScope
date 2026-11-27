import ctypes
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import urllib.request

import numpy as np
import pytest

from lanescope import demos, mcp_server, summary
from lanescope.cli import main
from lanescope.commands import nsight_check, record_program
from lanescope.cuda import toolchain
from lanescope.fixes import ranking
from lanescope.gui import server as gui
from lanescope.nsight import calibration
from lanescope.recorder import capture
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

SOURCE = pathlib.Path(__file__).parent.parent / "src"


@pytest.fixture
def trace_path(tmp_path):
    path = tmp_path / "transpose.npz"
    demos.transpose_trace().save(path)
    return path


def test_demo_writes_the_trace_and_prints_both_changes(tmp_path, capsys):
    out = tmp_path / "demo.npz"
    assert main(["demo", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert out.exists()
    assert "31 bank conflicts" in text
    assert "pad each 128-byte row by 4 bytes" in text
    assert "Swizzle<5,2,5>" in text


def test_explain_lists_the_worst_instruction_first(trace_path, capsys):
    assert main(["explain", str(trace_path)]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("0x")]
    assert rows[0].startswith("0x0120")


def test_explain_json_gives_counters_for_tools(trace_path, capsys):
    assert main(["explain", str(trace_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    first = report["instructions"][0]
    assert (first["pc"], first["line"], first["excessive"]) == ("0x120", 10, 3968)


def test_lanes_shows_the_bank_grid_of_the_worst_access(trace_path, capsys):
    assert main(["lanes", str(trace_path), "--pc", "0x120"]) == 0
    assert "32 wavefronts against an ideal of 1" in capsys.readouterr().out


def test_lanes_rejects_an_unknown_pc(trace_path, capsys):
    assert main(["lanes", str(trace_path), "--pc", "0x999"]) == 2
    assert "no access at pc 0x999" in capsys.readouterr().err


def test_lanes_rejects_a_row_of_another_pc(trace_path, capsys):
    assert main(["lanes", str(trace_path), "--pc", "0x120", "--row", "0"]) == 2
    assert "row 0 is no access of pc 0x120" in capsys.readouterr().err


def test_whatif_json_gives_counters_before_and_after(trace_path, capsys):
    assert (
        main(["whatif", str(trace_path), "--buffer", "0:4096", "--swizzle", "5,2,5", "--json"]) == 0
    )
    report = json.loads(capsys.readouterr().out)
    worst = [item for item in report["instructions"] if item["before"]["pc"] == "0x120"][0]
    assert (worst["before"]["excessive"], worst["after"]["excessive"]) == (3968, 0)
    assert report["extra_bytes"] == 0


def test_whatif_rejects_a_buffer_without_accesses(trace_path, capsys):
    assert main(["whatif", str(trace_path), "--buffer", "8192:4096", "--pad", "128:4"]) == 2
    assert "inside the buffer" in capsys.readouterr().err


def test_whatif_reports_a_change_that_does_not_fit(trace_path, capsys):
    assert main(["whatif", str(trace_path), "--buffer", "0:4096", "--pad", "100:4"]) == 2
    assert "whole number" in capsys.readouterr().err


def test_whatif_keeps_the_kind_of_an_atomic_apart_from_its_unit(tmp_path, capsys):
    lanes = tuple(128 * lane for lane in range(WARP_SIZE))
    path = tmp_path / "atomics.npz"
    Trace.from_accesses([Access(0x10, Space.SHARED, Kind.ATOMIC, 4, lanes)]).save(path)
    assert main(["whatif", str(path), "--buffer", "0:4096", "--pad", "128:4"]) == 0
    assert "shared atomic wavefronts" in " ".join(capsys.readouterr().out.split())


def test_suggest_names_the_global_sectors_only_of_a_change_that_alters_them(tmp_path, capsys):
    path = tmp_path / "matmul_tile.npz"
    demos.matmul_tile_trace().save(path)
    assert main(["suggest", str(path), "--limit", "50"]) == 0
    lines = capsys.readouterr().out.splitlines()
    (padding,) = [line for line in lines if line.startswith("pad each 128-byte row by 16 bytes")]
    assert "excessive global sectors 0 -> 256" in padding
    assert "sectors" not in lines[1]


def test_report_command_writes_the_html_file(trace_path, tmp_path):
    out = tmp_path / "report.html"
    assert main(["report", str(trace_path), "-o", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_rejects_a_negative_limit(trace_path, tmp_path):
    with pytest.raises(SystemExit) as stop:
        main(["report", str(trace_path), "-o", str(tmp_path / "r.html"), "--limit", "-1"])
    assert stop.value.code == 2


def test_whatif_json_worst_rows_point_into_the_trace(trace_path, capsys):
    assert main(["whatif", str(trace_path), "--buffer", "0:4096", "--pad", "128:4", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    trace = Trace.load(trace_path)
    for item in report["instructions"]:
        for side in ("before", "after"):
            row = item[side]["worst_row"]
            assert f"{int(trace.pc[row]):#x}" == item[side]["pc"]


def test_lanes_shows_the_worse_space_of_a_generic_pc(tmp_path, capsys):
    accesses = [
        Access(0x40, Space.SHARED, Kind.LOAD, 4, tuple(4 * lane for lane in range(WARP_SIZE))),
        Access(0x40, Space.GLOBAL, Kind.LOAD, 4, tuple(4096 * lane for lane in range(WARP_SIZE))),
    ]
    path = tmp_path / "generic.npz"
    Trace.from_accesses(accesses).save(path)
    assert main(["lanes", str(path), "--pc", "0x40"]) == 0
    assert "32 sectors against an ideal of 4" in capsys.readouterr().out


def test_explain_replaces_control_characters_of_the_kernel_name(tmp_path, capsys):
    path = tmp_path / "evil.npz"
    access = Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(4 * lane for lane in range(WARP_SIZE)))
    Trace.from_accesses([access], {"kernel": "k\x1b[2Jx"}).save(path)
    assert main(["explain", str(path)]) == 0
    out = capsys.readouterr().out
    assert "\x1b" not in out
    assert "kernel k?[2Jx" in out


def test_malformed_trace_file_gives_a_message(tmp_path, capsys):
    path = tmp_path / "broken.npz"
    path.write_bytes(b"PK\x03\x04 not a zip")
    assert main(["explain", str(path)]) == 2
    assert "no readable trace file" in capsys.readouterr().err


def test_suggest_prints_the_best_changes(trace_path, capsys):
    assert main(["suggest", str(trace_path), "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "Swizzle<5,2,5>" in out
    assert "pad each 128-byte row by 4 bytes" in out


def test_suggest_json_gives_the_inferred_buffer(trace_path, capsys):
    assert main(["suggest", str(trace_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["buffer"] == {"start": 0, "size": 4096}


def test_bench_list_prints_every_case(capsys):
    assert main(["bench", "list"]) == 0
    assert len(capsys.readouterr().out.splitlines()) == len(calibration.launches())


@pytest.mark.skipif(toolchain.find().nvrtc is None, reason="NVRTC is needed")
def test_bench_build_writes_the_cubin_and_its_source_without_the_driver(
    tmp_path, monkeypatch, capsys
):
    def no_driver():
        raise AssertionError("the CUDA driver was loaded")

    monkeypatch.setattr(record_program, "Driver", no_driver)
    monkeypatch.setattr(nsight_check, "Driver", no_driver)
    monkeypatch.setattr(calibration, "Driver", no_driver)
    cubin = tmp_path / "bench-sm_89.cubin"
    source = tmp_path / "bench-sm_89.cu"
    assert main(["bench", "build", "--arch", "sm_89", "--out", str(cubin)]) == 0
    assert cubin.read_bytes()[:4] == b"\x7fELF"
    assert source.read_text() == calibration.kernel_source()
    assert capsys.readouterr().out.splitlines() == [f"wrote {cubin}", f"wrote {source}"]


def test_bench_build_keeps_the_cubin_apart_from_its_source(tmp_path, capsys):
    assert main(["bench", "build", "--arch", "sm_89", "--out", str(tmp_path / "bench.cu")]) == 2
    assert "such as bench.cubin" in capsys.readouterr().err
    assert not (tmp_path / "bench.cu").exists()


def test_bench_run_help_names_the_step_that_builds_its_cubin(capsys):
    with pytest.raises(SystemExit):
        main(["bench", "run", "--help"])
    assert "bench build" in " ".join(capsys.readouterr().out.split())


def test_import_of_a_folder_without_launches_fails_with_a_message(tmp_path, capsys):
    assert main(["import", str(tmp_path)]) == 2
    assert "no launch files" in capsys.readouterr().err


def test_record_needs_a_program(tmp_path, capsys):
    assert main(["record", "--out", str(tmp_path), "--arch", "sm_89"]) == 2
    assert "after --" in capsys.readouterr().err


def fake_recorder(monkeypatch, status, launches):
    """Replace the recorder with a run that leaves launch files and exits with `status`.

    Return the settings that the run receives.
    """
    settings = {}

    def run(command, built, out_dir, **given):
        settings.update(given)
        out_dir.mkdir(parents=True, exist_ok=True)
        records = np.zeros(1, dtype=capture.RECORD_DTYPE)
        records[0]["pc"], records[0]["mask"], records[0]["width"] = 0x1010, 1, 4
        info = {"format": 1, "kernel": "k", "function_pc": 0x1000, "record_bytes": 280}
        for launch in range(launches):
            (out_dir / f"launch-{launch}.bin").write_bytes(records.tobytes())
            (out_dir / f"launch-{launch}.json").write_text(json.dumps(info))
        return status

    monkeypatch.setattr(record_program, "build_recorder", lambda out, arch, cache: None)
    monkeypatch.setattr(capture, "record", run)
    tools = toolchain.Toolchain(*[None] * 9, disassemblers=())
    monkeypatch.setattr(record_program, "find", lambda: tools)
    return settings


def run_record(out, *options):
    return main(["record", "--out", str(out), "--arch", "sm_89", *options, "--", "./app"])


@pytest.mark.parametrize("status", [0, 1])
def test_record_fails_when_the_recorder_saw_no_kernel_launch(tmp_path, monkeypatch, capsys, status):
    fake_recorder(monkeypatch, status=status, launches=0)
    assert run_record(tmp_path) == 2
    out, err = capsys.readouterr()
    assert "launched no kernel that the recorder saw" in err
    assert ("exited with status 1" in err) == bool(status)
    assert "wrote" not in out


@pytest.mark.parametrize(
    ("status", "ending"),
    [(3, "the program exited with status 3."), (-11, "the program was killed by signal 11.")],
)
def test_record_of_a_failing_program_says_how_it_ended_and_writes_no_trace(
    tmp_path, monkeypatch, capsys, status, ending
):
    fake_recorder(monkeypatch, status=status, launches=1)
    assert run_record(tmp_path) == 2
    out, err = capsys.readouterr()
    assert ending in err
    assert f"lanescope import {tmp_path}" in err
    assert "wrote" not in out
    assert not (tmp_path / "trace-0.npz").exists()


def test_record_imports_each_launch_of_a_program_that_succeeds(tmp_path, monkeypatch, capsys):
    fake_recorder(monkeypatch, status=0, launches=2)
    assert run_record(tmp_path) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"wrote {tmp_path / 'trace-0.npz'}",
        f"wrote {tmp_path / 'trace-1.npz'}",
        f"Next, lanescope summary {tmp_path} ranks the conflicts of all 2 launches, "
        f"and lanescope gui {tmp_path} opens them in the browser.",
    ]


@pytest.mark.parametrize("capacity", ["0", str((1 << 26) + 1), "many"])
def test_record_rejects_a_capacity_that_the_recorder_would_ignore(
    tmp_path, monkeypatch, capsys, capacity
):
    fake_recorder(monkeypatch, status=0, launches=1)
    with pytest.raises(SystemExit) as stop:
        run_record(tmp_path, "--capacity", capacity)
    assert stop.value.code == 2
    assert "from 1 to 67108864" in capsys.readouterr().err


def test_record_passes_the_largest_capacity_that_the_recorder_keeps(tmp_path, monkeypatch):
    settings = fake_recorder(monkeypatch, status=0, launches=1)
    assert run_record(tmp_path, "--capacity", str(1 << 26)) == 0
    assert settings["capacity"] == 1 << 26


@pytest.mark.parametrize(
    ("arguments", "reason"),
    [
        (["lanes", "t.npz", "--pc", "zz"], "expected a number such as 0x120, got 'zz'"),
        (["suggest", "t.npz", "--buffer", "0:zz"], "expected a number such as 0x120, got 'zz'"),
        (["whatif", "t.npz", "--buffer", "0:4096", "--swizzle", "1,zz,3"], "got 'zz'"),
        (["whatif", "t.npz", "--buffer", "0:4096", "--pad", "128:zz"], "got 'zz'"),
        (["whatif", "t.npz", "--buffer", "0:0", "--pad", "128:4"], "and a positive size"),
        (["whatif", "t.npz", "--buffer=-4:4096", "--pad", "128:4"], "a start of zero or more"),
        (["whatif", "t.npz", "--buffer", f"0:{(1 << 48) + 1}", "--pad", "128:4"], "end at or"),
        (["suggest", "t.npz", "--limit", "x"], "expected a count of zero or more"),
        (["report", "t.npz", "-o", "r.html", "--limit", "-1"], "expected a count of zero or more"),
        (["gui", "--port", "x"], "expected a port from 0 to 65535"),
    ],
)
def test_argument_errors_give_their_reason(capsys, arguments, reason):
    with pytest.raises(SystemExit) as stop:
        main(arguments)
    assert stop.value.code == 2
    error = capsys.readouterr().err
    assert reason in error
    assert "invalid" not in error


def test_gpu_info_without_a_driver_gives_a_message(capsys):
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        pass
    else:
        pytest.skip("a CUDA driver is present")
    assert main(["gpu-info"]) == 2
    assert "cannot load the CUDA driver" in capsys.readouterr().err


def test_suggest_with_limit_zero_lists_no_change_without_a_false_note(trace_path, capsys):
    assert main(["suggest", str(trace_path), "--limit", "0"]) == 0
    assert "no padding or swizzle" not in capsys.readouterr().out


def test_gui_demo_serves_the_demo_traces_and_the_llama_cpp_recordings(monkeypatch):
    served = {}

    def fake_serve(paths, host, port, allowed_hosts, open_browser, processes):
        served["files"] = sorted(path.name for path in paths[0].iterdir())
        served["address"] = (host, port, allowed_hosts, open_browser)
        served["processes"] = processes

    monkeypatch.setattr(gui, "serve", fake_serve)
    assert main(["gui", "--demo", "--host", "0.0.0.0", "--allow-host", "box"]) == 0
    assert served["files"] == [
        "1_transpose.npz",
        "2_row_sums.npz",
        "3_column_sums.npz",
        "4_matmul_tile.npz",
        "5_histogram.npz",
        "6_llama_cpp_indexer.npz",
        "7_llama_cpp_indexer_fixed.npz",
    ]
    assert served["address"] == ("0.0.0.0", gui.DEFAULT_PORT, ["box"], False)
    assert served["processes"] is True


def test_gui_serves_this_folder_by_default(monkeypatch, tmp_path):
    served = []
    monkeypatch.setattr(gui, "serve", lambda paths, *rest, **options: served.extend(paths))
    monkeypatch.chdir(tmp_path)
    assert main(["gui"]) == 0
    assert served == [tmp_path]


def test_gui_rejects_a_port_outside_the_range(capsys):
    with pytest.raises(SystemExit):
        main(["gui", "--port", "70000"])
    assert "65535" in capsys.readouterr().err


def start_demo(hangup=signal.SIG_DFL):
    """Start `lanescope gui --demo` with the given SIGHUP handler and return it with its folder."""
    environment = dict(os.environ, PYTHONPATH=str(SOURCE), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "lanescope", "gui", "--demo", "--port", "0"]
    process = subprocess.Popen(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        preexec_fn=lambda: signal.signal(signal.SIGHUP, hangup),
    )
    assert process.stdout is not None
    url = re.search(r"http://\S+/", process.stdout.readline())
    assert url is not None
    with urllib.request.urlopen(url.group(0) + "api/info", timeout=10) as answer:
        demo = pathlib.Path(json.loads(answer.read())["roots"][0])
    return process, demo


@pytest.mark.parametrize("stop", [signal.SIGTERM, signal.SIGHUP])
def test_gui_demo_stopped_by_a_signal_removes_its_demo_folder(stop):
    process, demo = start_demo()
    with process:
        try:
            names = sorted(path.name for path in demo.iterdir())
            assert names == [
                "1_transpose.npz",
                "2_row_sums.npz",
                "3_column_sums.npz",
                "4_matmul_tile.npz",
                "5_histogram.npz",
                "6_llama_cpp_indexer.npz",
                "7_llama_cpp_indexer_fixed.npz",
            ]
            process.send_signal(stop)
            assert process.wait(timeout=10) == 0
            assert not demo.exists()
        finally:
            if process.poll() is None:
                process.kill()


def test_gui_demo_finishes_its_cleanup_when_the_terminal_sends_a_second_hangup():
    process, demo = start_demo()
    with process:
        try:
            process.send_signal(signal.SIGHUP)
            process.send_signal(signal.SIGHUP)
            assert process.wait(timeout=10) == 0
            assert process.stderr is not None and process.stderr.read() == ""
            assert not demo.exists()
        finally:
            if process.poll() is None:
                process.kill()


def test_gui_demo_started_with_hangups_ignored_keeps_serving_after_one():
    process, demo = start_demo(hangup=signal.SIG_IGN)
    with process:
        try:
            process.send_signal(signal.SIGHUP)
            time.sleep(0.5)
            assert process.poll() is None
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) == 0
            assert not demo.exists()
        finally:
            if process.poll() is None:
                process.kill()


def test_import_reads_lines_with_the_next_nvdisasm_when_one_fails(tmp_path, monkeypatch):
    failing = tmp_path / "old-nvdisasm"
    failing.write_text("#!/bin/sh\nexit 1\n")
    failing.chmod(0o755)
    working = tmp_path / "nvdisasm"
    listing = '.text.k:\\n//## File "/src/k.cu", line 3\\n /*0010*/ LDS R0, [R0] ;\\n'
    working.write_text(f"#!/bin/sh\nprintf '{listing}'\n")
    working.chmod(0o755)
    records = np.zeros(1, dtype=capture.RECORD_DTYPE)
    records[0]["pc"], records[0]["mask"], records[0]["width"] = 0x1010, 1, 4
    (tmp_path / "launch-0.bin").write_bytes(records.tobytes())
    info = {"format": 1, "kernel": "k", "module": 0, "function_pc": 0x1000, "record_bytes": 280}
    (tmp_path / "launch-0.json").write_text(json.dumps(info))
    (tmp_path / "module-0.cubin").write_bytes(b"")
    tools = toolchain.Toolchain(*[None] * 9, disassemblers=(failing, working))
    monkeypatch.setattr(record_program, "find", lambda: tools)
    assert main(["import", str(tmp_path)]) == 0
    assert Trace.load(tmp_path / "trace-0.npz").metadata["lines"] == {"0x10": 3}


def test_import_without_nvdisasm_says_that_instruction_rules_fall_back(
    tmp_path, monkeypatch, capsys
):
    records = np.zeros(1, dtype=capture.RECORD_DTYPE)
    records[0]["pc"], records[0]["mask"], records[0]["width"] = 0x1010, 1, 4
    (tmp_path / "launch-0.bin").write_bytes(records.tobytes())
    info = {"format": 1, "kernel": "k", "module": 0, "function_pc": 0x1000, "record_bytes": 280}
    (tmp_path / "launch-0.json").write_text(json.dumps(info))
    tools = toolchain.Toolchain(*[None] * 9, disassemblers=())
    monkeypatch.setattr(record_program, "find", lambda: tools)
    assert main(["import", str(tmp_path)]) == 0
    assert "nvdisasm was not found" in capsys.readouterr().err


def test_validate_takes_random_patterns_only_for_the_bench(tmp_path, capsys):
    command = ["validate", "--out", str(tmp_path), "--random", "5", "--", "./app"]
    assert main(command) == 2
    assert "only to the bench" in capsys.readouterr().err


def test_bench_time_rejects_a_matmul_size_that_is_not_a_multiple_of_64(tmp_path, capsys):
    command = ["bench", "time", "--cubin", str(tmp_path / "bench.cubin"), "--matmul-size", "96"]
    with pytest.raises(SystemExit) as stop:
        main(command)
    assert stop.value.code == 2
    assert "multiple of 64" in capsys.readouterr().err


def test_bench_time_rejects_a_matrix_size_that_is_not_a_multiple_of_32(tmp_path, capsys):
    assert main(["bench", "time", "--cubin", str(tmp_path / "bench.cubin"), "--size", "100"]) == 2
    assert "multiple of 32" in capsys.readouterr().err


def test_mcp_serves_the_tools_of_the_mcp_server(monkeypatch):
    served = []
    monkeypatch.setattr(mcp_server, "serve", lambda: served.append(True))
    assert main(["mcp"]) == 0
    assert served == [True]


def test_mcp_names_the_package_extra_that_brings_the_sdk(monkeypatch, capsys):
    monkeypatch.setattr(mcp_server, "MCPServer", None)
    assert main(["mcp"]) == 2
    assert "pip install 'lanescope[mcp]'" in capsys.readouterr().err


def test_the_command_line_starts_without_the_mcp_sdk():
    modules = "{'mcp', 'lanescope.mcp_server'}"
    code = f"import sys, lanescope.cli; print(sorted({modules} & set(sys.modules)))"
    environment = dict(os.environ, PYTHONPATH=str(SOURCE))
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


@pytest.fixture
def recording(tmp_path):
    """Return a recording folder with two launches of the transpose and one of the histogram."""
    folder = tmp_path / "run1"
    folder.mkdir()
    demos.transpose_trace().save(folder / "trace-0.npz")
    demos.transpose_trace(tiles_per_side=1).save(folder / "trace-1.npz")
    demos.histogram_trace().save(folder / "trace-2.npz")
    return folder


def test_summary_ranks_the_kernels_and_lines_of_a_recording(recording, capsys):
    assert main(["summary", str(recording)]) == 0
    out = capsys.readouterr().out
    assert f"{recording}: 3 launches of 2 kernels" in out
    kernels = out.split("Kernels by extra passes")[1].split("Lines by extra passes")[0]
    assert kernels.index("transpose") < kernels.index("histogram")
    assert "line 10 load" in out
    assert "line 6 atomic *" in out
    assert f"lanescope suggest {recording / 'trace-0.npz'} --pc 0x120 ranks its fixes" in out


def test_summary_json_gives_the_whole_summary_for_tools(recording, capsys):
    assert main(["summary", str(recording), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["traces"] == 3
    assert {kernel["name"]: kernel["launches"] for kernel in data["kernels"]} == {
        "transpose": 2,
        "histogram": 1,
    }


def test_summary_of_global_memory_ranks_extra_sectors(tmp_path, capsys):
    demos.column_sums_trace().save(tmp_path / "trace-0.npz")
    assert main(["summary", str(tmp_path), "--global"]) == 0
    out = capsys.readouterr().out
    assert "Global memory:" in out
    assert "Lines by extra sectors" in out


def test_summary_limits_the_rows_it_lists(recording, capsys):
    assert main(["summary", str(recording), "--limit", "1"]) == 0
    out = capsys.readouterr().out
    assert "and 1 more kernel" in out


@pytest.mark.parametrize(
    ("make", "reason"),
    [
        (lambda path: path / "missing", "no such file or folder"),
        (lambda path: path, "found no trace files"),
    ],
)
def test_summary_names_a_path_without_traces(tmp_path, capsys, make, reason):
    assert main(["summary", str(make(tmp_path))]) == 2
    assert reason in capsys.readouterr().err


def test_summary_skips_files_without_a_trace_and_says_so(recording, capsys):
    (recording / "notes.npz").write_bytes(b"not a trace")
    assert main(["summary", str(recording)]) == 0
    assert "skipped 1 file without a trace" in capsys.readouterr().err


def test_summary_reads_a_large_recording_in_several_processes(recording, monkeypatch, capsys):
    assert main(["summary", str(recording), "--json"]) == 0
    serial = json.loads(capsys.readouterr().out)
    monkeypatch.setattr(summary, "PARALLEL_TRACES", 1)
    assert main(["summary", str(recording), "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == serial


def test_explain_gives_each_excess_as_a_part_of_the_kernel_and_marks_input_values(tmp_path, capsys):
    path = tmp_path / "histogram.npz"
    demos.histogram_trace().save(path)
    assert main(["explain", str(path)]) == 0
    out = capsys.readouterr().out
    assert "of kernel" in out.splitlines()[2]
    atomic = next(line for line in out.splitlines() if line.startswith("0x0170"))
    assert atomic.endswith("% *")
    assert "No layout of the buffer removes them." in out


def test_suggest_warns_that_a_fix_for_input_values_may_help_less_on_other_inputs(tmp_path, capsys):
    path = tmp_path / "histogram.npz"
    demos.histogram_trace().save(path)
    assert main(["suggest", str(path)]) == 0
    assert "may help less on other inputs" in capsys.readouterr().out


def test_suggest_explains_input_values_when_no_layout_lowers_them(tmp_path, monkeypatch, capsys):
    path = tmp_path / "histogram.npz"
    demos.histogram_trace().save(path)
    monkeypatch.setattr(ranking, "suggest", lambda trace, buffer: [])
    assert main(["suggest", str(path)]) == 0
    out = capsys.readouterr().out
    assert "no padding or swizzle lowers the excess wavefronts of this buffer" in out
    assert "They fall only when each lane, or each group of lanes" in out


def test_record_of_many_launches_counts_its_traces_and_points_to_the_summary(
    tmp_path, monkeypatch, capsys
):
    fake_recorder(monkeypatch, status=0, launches=12)
    assert run_record(tmp_path) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"wrote 12 traces to {tmp_path}"
    assert out[1].startswith(f"Next, lanescope summary {tmp_path} ranks the conflicts of all 12")


def test_record_of_one_launch_points_to_its_instructions(tmp_path, monkeypatch, capsys):
    fake_recorder(monkeypatch, status=0, launches=1)
    assert run_record(tmp_path) == 0
    out = capsys.readouterr().out
    assert (
        f"Next, lanescope explain {tmp_path / 'trace-0.npz'} lists its memory instructions" in out
    )


def test_record_with_lost_records_advises_a_larger_capacity(tmp_path, monkeypatch, capsys):
    fake_recorder(monkeypatch, status=0, launches=2)
    original = capture.to_trace
    monkeypatch.setattr(
        capture, "to_trace", lambda info, records: original(info | {"dropped": 7}, records)
    )
    assert run_record(tmp_path, "--capacity", "1000") == 0
    err = capsys.readouterr().err
    assert "the record buffer filled up in 2 of 2 launches and left out 14 accesses" in err
    assert "such as --capacity 2000" in err


def test_record_with_a_disassembler_but_no_line_tables_advises_lineinfo(
    tmp_path, monkeypatch, capsys
):
    fake_recorder(monkeypatch, status=0, launches=1)
    tools = toolchain.Toolchain(*[None] * 9, disassemblers=(pathlib.Path("nvdisasm"),))
    monkeypatch.setattr(record_program, "find", lambda: tools)
    assert run_record(tmp_path) == 0
    err = capsys.readouterr().err
    assert "no source lines in 1 of 1 launch, since their kernels were built without" in err
    assert "nvcc -lineinfo" in err
