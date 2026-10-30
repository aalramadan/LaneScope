import ctypes
import dataclasses
import json
import pathlib

import pytest

from lanescope import demos
from lanescope.cuda.driver import DriverError
from lanescope.cuda.toolchain import find
from lanescope.nsight import calibration, compare, validate
from lanescope.recorder.build import RecorderBuild
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

COLUMNS = ("pc", "space", "kind", "width", "mask", "cta", "warp", "addr")


def filler_trace():
    access = Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(4 * lane for lane in range(WARP_SIZE)))
    return Trace.from_accesses([access])


def kernel(name, store_wavefronts, load_wavefronts):
    return compare.KernelCounters(
        kernel=name,
        metrics={"memory_l1_wavefronts_shared": {0x40: store_wavefronts, 0x80: load_wavefronts}},
        sass={0x40: "STS [R2], R3", 0x80: "LDS R4, [R5]"},
    )


def test_launch_names_cover_every_case_the_transposes_and_the_matmuls():
    names = validate.launch_names()
    fixed = list(calibration.TRANSPOSE_KERNELS) + list(calibration.MATMUL_KERNELS)
    assert len(names) == len(calibration.launches()) + len(fixed)
    assert names[-len(fixed) :] == fixed


def matmul_trace(kernel_name):
    """Return a plain matmul: a cp.async that fills a 512-byte tile and an ldmatrix on one bank."""
    rows = tuple(16 * lane for lane in range(WARP_SIZE))
    accesses = [
        Access(0x40, Space.GLOBAL, Kind.LOAD, 16, rows),
        Access(0x40, Space.SHARED, Kind.STORE, 16, rows),
        Access(0x80, Space.SHARED, Kind.LOAD, 16, tuple(128 * (lane % 4) for lane in range(32))),
    ]
    opcodes = {"0x40": "LDGSTS.E.BYPASS.128", "0x80": "LDSM.16.M88.4"}
    return Trace.from_accesses(accesses, {"kernel": kernel_name, "opcodes": opcodes})


def traces_and_kernels():
    names, kernels = validate.launch_names(), validate.launch_kernels()
    traces = []
    for kernel_name in kernels:
        if kernel_name == "matmul_plain":
            traces.append(matmul_trace(kernel_name))
            continue
        trace = demos.transpose_trace() if kernel_name == "transpose_tile" else filler_trace()
        traces.append(
            Trace(**{n: getattr(trace, n) for n in COLUMNS}, metadata={"kernel": kernel_name})
        )
    counters = [kernel(name, 0.0, 0.0) for name in kernels]
    counters[names.index("transpose_padded")] = kernel("transpose_padded", 128.0, 128.0)
    counters[names.index("transpose_swizzled")] = kernel("transpose_swizzled", 128.0, 4096.0)
    counters[names.index("matmul_swizzled")] = compare.KernelCounters(
        kernel="matmul_swizzled",
        metrics={"memory_l1_wavefronts_shared": {0x40: 4.0, 0x80: 4.0}},
        sass={0x40: "LDGSTS.E.BYPASS.128 [R1][R2.64]", 0x80: "LDSM.16.M88.4 R4, [R5]"},
    )
    return traces, counters


def test_predictions_set_the_what_if_counts_against_the_measured_ones():
    launches, problems = validate.pair_launches(*traces_and_kernels())
    assert problems == []
    found = {
        (p.kernel, p.kind): (p.predicted, p.measured) for p in validate.predict_layouts(launches)
    }
    assert found[("matmul_swizzled", Kind.STORE)] == (4, 4.0)
    assert found[("matmul_swizzled", Kind.LOAD)] == (4, 4.0)
    assert found[("transpose_padded", Kind.LOAD)] == (128, 128.0)
    assert found[("transpose_swizzled", Kind.LOAD)] == (128, 4096.0)
    assert found[("transpose_swizzled", Kind.STORE)] == (128, 128.0)


def test_launches_of_another_kernel_are_unpaired_and_named():
    traces, counters = traces_and_kernels()
    counters[3] = kernel("something_else", 0.0, 0.0)
    launches, problems = validate.pair_launches(traces, counters)
    assert launches[3].counters is None
    assert problems == [
        f"Launch 3 ({validate.launch_names()[3]}) has Nsight Compute counters of something_else."
    ]


def test_missing_launches_are_reported():
    traces, counters = traces_and_kernels()
    launches, problems = validate.pair_launches(traces[:-1], counters)
    assert launches[-1].trace is None
    assert problems[0].startswith(f"The recorder wrote {len(traces) - 1} launches")


def test_predictions_need_the_tiled_transpose():
    launches, _ = validate.pair_launches([filler_trace()], None)
    assert validate.predict_layouts(launches) == []


def test_validation_stops_with_a_clear_error_without_a_gpu(tmp_path):
    try:
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        pass
    else:
        pytest.skip("a CUDA driver is present")
    with pytest.raises(DriverError, match="cannot load the CUDA driver"):
        validate.validate(tmp_path)


class StandInDriver:
    """The calls of the CUDA driver that the check makes, without a GPU."""

    def name(self):
        return "Stand-in GPU"

    def arch(self):
        return "sm_89"

    def version(self):
        return 12090

    def close(self):
        pass


@pytest.fixture
def offline_check(monkeypatch, tmp_path):
    """Replace the GPU, the builds, and the recorded run of the check with stand-ins.

    The returned list holds the trace paths that the import of the recording gives.
    """
    traces = []
    recorder = RecorderBuild(tmp_path / "lib.so", tmp_path / "patches.cubin", tmp_path / "san.so")
    monkeypatch.setattr(validate, "Driver", StandInDriver)
    monkeypatch.setattr(validate, "check_versions", lambda tools, version: None)
    monkeypatch.setattr(validate, "build_recorder", lambda out_dir, arch, tools: recorder)
    monkeypatch.setattr(validate.calibration, "compile_kernels", lambda arch, tools, source: b"")
    monkeypatch.setattr(validate.capture, "record", lambda command, build, out, **kw: 0)
    monkeypatch.setattr(validate.capture, "import_recording", lambda out, tools: list(traces))
    return traces


def test_validation_stops_when_the_recorded_run_writes_no_launch_files(tmp_path, offline_check):
    tools = dataclasses.replace(find(), ncu=None)
    with pytest.raises(validate.ValidationError, match="no launch files"):
        validate.validate(tmp_path / "check", tools=tools)


def test_random_patterns_reach_the_recorded_run_and_the_launch_names(
    tmp_path, offline_check, monkeypatch
):
    commands = []

    def record(command, build, out, **settings):
        commands.append(command)
        return 0

    monkeypatch.setattr(validate.capture, "record", record)
    trace = tmp_path / "trace-0.npz"
    filler_trace().save(trace)
    offline_check.append(trace)
    tools = dataclasses.replace(find(), ncu=None)
    validate.validate(tmp_path / "check", tools=tools, random_count=5, seed=7)
    assert commands[0][-4:] == ["--random", "5", "--seed", "7"]
    names = validate.launch_names(5, 7)
    assert names[: -len(calibration.TRANSPOSE_KERNELS) - len(calibration.MATMUL_KERNELS)] == [
        case.name for case in calibration.launches(5, 7)
    ]


def summary_row(launch, nsight):
    return compare.Row(launch, "k", 0x10, "LDS R0, [R0]", "shared_wavefronts", 1, nsight)


def test_summary_gives_the_matched_counts_of_each_group_of_launches():
    groups = calibration.launch_groups(2, 3)
    starts = [sum(len(cases) for _, cases in groups[:index]) for index in range(len(groups) + 1)]
    rows = [summary_row(start, 1.0) for start in starts] + [summary_row(starts[1], 2.0)]
    lines = validate._summary_lines(rows, groups)
    assert lines[0] == "LaneScope matches Nsight Compute in 7 of 8 instruction counts."
    assert lines[1:7] == [
        f"The {len(groups[0][1])} calibration launches give 1 of 1.",
        f"The {len(groups[1][1])} ldmatrix and cp.async launches give 1 of 2.",
        f"The {len(groups[2][1])} atomic launches give 1 of 1.",
        "The 2 random lane patterns of seed 3 give 1 of 1.",
        "The 2 random ldmatrix and cp.async patterns of seed 3 give 1 of 1.",
        "The 2 random atomic patterns of seed 3 give 1 of 1.",
    ]


def test_validation_without_nsight_compute_says_how_to_get_it(tmp_path, offline_check):
    trace = tmp_path / "trace-0.npz"
    filler_trace().save(trace)
    offline_check.append(trace)
    tools = dataclasses.replace(find(), ncu=None)
    summary = validate.validate(tmp_path / "check", tools=tools).read_text()
    assert "`lanescope get-ncu`" in summary
    assert "install Nsight Compute" in summary


def program_check(monkeypatch, tmp_path, counters):
    """Stand in for the recording and the profile of a program with one launch of kernel k."""
    trace = tmp_path / "trace-0.npz"
    access = Access(0x80, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(32)))
    Trace.from_accesses([access], {"kernel": "k"}).save(trace)
    calls = []
    monkeypatch.setattr(validate.capture, "import_recording", lambda out, tools: [trace])
    monkeypatch.setattr(
        validate, "_profile", lambda tools, command, report, kernel="": calls.append(kernel)
    )
    monkeypatch.setattr(validate.compare, "read_report", lambda report: counters)
    return calls


def test_validating_a_program_compares_each_of_its_launches_with_its_counters(
    tmp_path, offline_check, monkeypatch
):
    counters = compare.KernelCounters(
        kernel="k(float *)",
        metrics={
            "memory_l1_wavefronts_shared": {0x80: 32.0},
            "memory_l1_wavefronts_shared_ideal": {0x80: 1.0},
        },
        sass={0x80: "LDS R0, [R1]"},
    )
    calls = program_check(monkeypatch, tmp_path, [counters])
    tools = dataclasses.replace(find(), ncu=pathlib.Path("ncu"))
    summary = validate.validate_program(tmp_path / "check", ["./app"], "k", tools=tools)
    text = summary.read_text()
    assert "LaneScope matches Nsight Compute in 2 of 2 instruction counts." in text
    assert calls == ["k"]


def test_validating_a_program_names_a_launch_without_counters_of_its_kernel(
    tmp_path, offline_check, monkeypatch
):
    other = compare.KernelCounters(kernel="other(float *)", metrics={}, sass={})
    program_check(monkeypatch, tmp_path, [other])
    tools = dataclasses.replace(find(), ncu=pathlib.Path("ncu"))
    text = validate.validate_program(tmp_path / "check", ["./app"], tools=tools).read_text()
    assert "Launch 0 (k) has Nsight Compute counters of other(float *)." in text


def test_a_launch_pairs_with_counters_that_carry_its_mangled_name():
    trace = Trace.from_accesses([], {"kernel": "_Z1kPf"})
    counters = compare.KernelCounters(kernel="k(float *)", metrics={}, sass={}, mangled="_Z1kPf")
    launches, problems = validate._pair([trace], [counters], ["_Z1kPf"], ["_Z1kPf"])
    assert problems == []
    assert launches[0].counters is counters


def test_validating_a_program_needs_nsight_compute(tmp_path, offline_check):
    tools = dataclasses.replace(find(), ncu=None)
    with pytest.raises(validate.ValidationError, match="lanescope get-ncu"):
        validate.validate_program(tmp_path / "check", ["./app"], tools=tools)


def test_cuda_tools_newer_than_the_driver_name_the_switch_to_the_wheels(monkeypatch):
    monkeypatch.setattr(validate.nvrtc, "version", lambda path: (12, 9))
    tools = dataclasses.replace(find(), nvrtc=pathlib.Path("libnvrtc.so.12"))
    with pytest.raises(validate.ValidationError, match="LANESCOPE_USE_WHEELS=1"):
        validate.check_versions(tools, 12060)


NCU_2026 = """#!/bin/sh
while [ "$#" -gt 0 ]; do
  if [ "$1" = --export ]; then out="$2"; fi
  shift
done
case "$out" in
  *.ncu-rep) touch "$out" ;;
  *) touch "$out.ncu-repz" ;;
esac
"""


def test_profile_finds_the_report_of_an_nsight_compute_that_compresses_by_default(tmp_path):
    ncu = tmp_path / "ncu"
    ncu.write_text(NCU_2026)
    ncu.chmod(0o755)
    report = tmp_path / "out" / "bench.ncu-rep"
    validate._profile(dataclasses.replace(find(), ncu=ncu), ["true"], report)
    assert report.exists()


NCU_ARGUMENTS = """#!/bin/sh
printf '%s\\n' "$@" > "$(dirname "$0")/arguments.txt"
while [ "$#" -gt 0 ]; do
  if [ "$1" = --export ]; then touch "$2"; fi
  shift
done
"""


def test_profile_collects_only_the_source_counters_by_the_mangled_kernel_name(tmp_path):
    ncu = tmp_path / "ncu"
    ncu.write_text(NCU_ARGUMENTS)
    ncu.chmod(0o755)
    report = tmp_path / "out" / "program.ncu-rep"
    validate._profile(dataclasses.replace(find(), ncu=ncu), ["true"], report, "cudnn")
    arguments = (tmp_path / "arguments.txt").read_text().splitlines()
    assert arguments[arguments.index("--kernel-name-base") + 1] == "mangled"
    assert arguments[arguments.index("--kernel-name") + 1] == "regex:.*cudnn.*"
    assert arguments[arguments.index("--section") + 1] == "SourceCounters"
    assert "--set" not in arguments


def test_metric_table_names_a_metric_that_only_later_kernels_report():
    global_kernel = compare.KernelCounters(
        kernel="global_load_4",
        metrics={"memory_l2_theoretical_sectors_global": {0x80: 4.0}},
        sass={0x80: "LDG.E R2, [R2.64]"},
    )
    lines = validate._metric_lines([kernel("shared_load_4", 1, 1), global_kernel])
    assert "| global_sectors | memory_l2_theoretical_sectors_global |" in lines


def write_check(folder, rows):
    record = folder / "record"
    record.mkdir()
    for launch in (0, 1, 10):
        demos.transpose_trace(tiles_per_side=1).save(record / f"trace-{launch}.npz")
    (folder / "validation.json").write_text(json.dumps({"rows": rows, "predictions": []}))
    return record


def check_row(launch, nsight=1024.0, kernel="transpose"):
    row = {"launch": launch, "offset": 0x120, "quantity": "shared_wavefronts", "nsight": nsight}
    return row | {"kernel": kernel, "sass": "LDS R0, [R0]"}


def test_measured_counts_number_the_traces_in_launch_order(tmp_path):
    record = write_check(tmp_path, [check_row(2), check_row(1, 7.0)])
    counts = validate.measured_counts(record / "trace-10.npz", "transpose")
    assert counts == {(0x120, "shared_wavefronts"): (1024.0, "LDS R0, [R0]")}
    assert validate.measured_counts(record / "trace-0.npz", "transpose") == {}


def test_measured_counts_skip_rows_of_another_kernel(tmp_path):
    rows = [check_row(0, kernel="transpose_tile"), check_row(1, kernel="transpose(float *, int)")]
    record = write_check(tmp_path, rows)
    assert validate.measured_counts(record / "trace-0.npz", "transpose") == {}
    assert validate.measured_counts(record / "trace-1.npz", "transpose") != {}


def test_measured_counts_keep_only_finite_counts(tmp_path):
    record = write_check(tmp_path, [check_row(0, float("nan"))])
    counts = validate.measured_counts(record / "trace-0.npz", "transpose")
    assert counts == {(0x120, "shared_wavefronts"): (None, "LDS R0, [R0]")}


def test_measured_counts_skip_traces_outside_a_check_and_malformed_rows(tmp_path):
    record = write_check(tmp_path, [{"launch": 0, "offset": "0x120"}, "row", {"launch": 0}])
    assert validate.measured_counts(record / "trace-0.npz", "transpose") == {}
    loose = tmp_path / "loose.npz"
    demos.transpose_trace(tiles_per_side=1).save(loose)
    assert validate.measured_counts(loose, "transpose") is None
    other = tmp_path / "runs"
    other.mkdir()
    demos.transpose_trace(tiles_per_side=1).save(other / "trace-0.npz")
    assert validate.measured_counts(other / "trace-0.npz", "transpose") is None
    (tmp_path / "validation.json").write_text("{broken")
    assert validate.measured_counts(record / "trace-0.npz", "transpose") is None


@pytest.mark.parametrize("nsight", [10**400, float("inf")])
def test_measured_counts_keep_a_count_that_no_float_holds_as_missing(tmp_path, nsight):
    record = write_check(tmp_path, [check_row(0, nsight)])
    counts = validate.measured_counts(record / "trace-0.npz", "transpose")
    assert counts == {(0x120, "shared_wavefronts"): (None, "LDS R0, [R0]")}


def test_measured_counts_skip_a_summary_with_a_number_of_too_many_digits(tmp_path):
    record = write_check(tmp_path, [])
    (tmp_path / "validation.json").write_text('{"rows": [{"nsight": ' + "9" * 5000 + "}]}")
    assert validate.measured_counts(record / "trace-0.npz", "transpose") is None


def test_measured_counts_skip_a_summary_nested_too_deep_to_read(tmp_path):
    record = write_check(tmp_path, [])
    (tmp_path / "validation.json").write_text("[" * 100_000 + "]" * 100_000)
    assert validate.measured_counts(record / "trace-0.npz", "transpose") is None
