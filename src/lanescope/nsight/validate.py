import dataclasses
import json
import math
import pathlib
import re
import subprocess
import sys

from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver, version_text
from lanescope.cuda.toolchain import Toolchain, find
from lanescope.fixes.layouts import Padding, XorSwizzle, apply, infer_buffer
from lanescope.model.counters import count
from lanescope.nsight import calibration, compare
from lanescope.recorder import capture
from lanescope.recorder.build import build_recorder
from lanescope.trace import Kind, Space, Trace

TRACE_NAME = re.compile(r"trace-(\d+)")
SUMMARY = "validation.json"


class ValidationError(RuntimeError):
    """A step of the GPU check failed."""


@dataclasses.dataclass(frozen=True)
class Prediction:
    """Predicted and measured shared-memory wavefronts of one transpose kernel."""

    kernel: str
    change: str
    kind: Kind
    predicted: int
    measured: float | None


def check_versions(tools: Toolchain, driver_version: int) -> None:
    """Raise ValidationError when the CUDA tools are newer than the driver supports."""
    tools.require("nvrtc")
    assert tools.nvrtc is not None
    major, minor = nvrtc.version(tools.nvrtc)
    if major * 1000 + minor * 10 > driver_version:
        raise ValidationError(
            f"the CUDA tools are version {major}.{minor}, newer than the "
            f"{version_text(driver_version)} that the driver supports. "
            "Update the NVIDIA driver, point CUDA_HOME at an older toolkit, or set "
            "LANESCOPE_USE_WHEELS=1 to use the CUDA wheels of the driver's version, "
            "which scripts/gpu_validate.sh installs."
        )


@dataclasses.dataclass(frozen=True)
class Launch:
    """One launch of the calibration run with its trace and its counters, when they match it."""

    index: int
    name: str
    trace: Trace | None
    counters: compare.KernelCounters | None


def launch_names(random_count: int = 0, seed: int = 0) -> list[str]:
    """Return the case or kernel name of each launch of the calibration run, in order."""
    cases = calibration.launches(random_count, seed)
    return [case.name for case in cases] + list(calibration.TRANSPOSE_KERNELS)


def launch_kernels(random_count: int = 0, seed: int = 0) -> list[str]:
    """Return the kernel of each launch of the calibration run, in order."""
    cases = calibration.launches(random_count, seed)
    return [case.kernel for case in cases] + list(calibration.TRANSPOSE_KERNELS)


def pair_launches(
    traces: list[Trace],
    kernels: list[compare.KernelCounters] | None,
    random_count: int = 0,
    seed: int = 0,
) -> tuple[list[Launch], list[str]]:
    """Pair each launch with the trace and the counters of its own kernel, and list each gap."""
    names, expected = launch_names(random_count, seed), launch_kernels(random_count, seed)
    problems = []
    if len(traces) != len(names):
        problems.append(
            f"The recorder wrote {len(traces)} launches, while the run has {len(names)}."
        )
    if kernels is not None and len(kernels) != len(names):
        problems.append(
            f"Nsight Compute reported {len(kernels)} launches, while the run has {len(names)}."
        )
    launches = []
    for index, (name, kernel) in enumerate(zip(names, expected, strict=True)):
        trace = traces[index] if index < len(traces) else None
        if trace is not None and trace.metadata.get("kernel") != kernel:
            problems.append(f"Launch {index} ({name}) recorded {trace.metadata.get('kernel')}.")
            trace = None
        counters = kernels[index] if kernels is not None and index < len(kernels) else None
        if counters is not None and counters.kernel.split("(")[0].strip() != kernel:
            problems.append(
                f"Launch {index} ({name}) has Nsight Compute counters of {counters.kernel}."
            )
            counters = None
        launches.append(Launch(index, name, trace, counters))
    return launches, problems


def validate(
    out_dir: pathlib.Path,
    arch: str | None = None,
    tools: Toolchain | None = None,
    random_count: int = 0,
    seed: int = 0,
) -> pathlib.Path:
    """Run the GPU check and return the path of its summary.

    With `random_count`, the run adds that many random lane patterns of `seed`
    after the fixed calibration cases, which tests the model on patterns that
    played no part in fitting it.
    """
    tools = tools or find()
    driver = Driver()
    arch = arch or driver.arch()
    supported = driver.version()
    gpu = f"{driver.name()} ({arch}), driver CUDA version {version_text(supported)}"
    driver.close()
    check_versions(tools, supported)
    out_dir.mkdir(parents=True, exist_ok=True)
    build = build_recorder(out_dir / "build", arch, tools)
    cubin = out_dir / f"bench-{arch}.cubin"
    cubin.write_bytes(calibration.compile_kernels(arch, tools, out_dir / "bench.cu"))
    command = [sys.executable, "-m", "lanescope", "bench", "run", "--cubin", str(cubin)]
    if random_count:
        command += ["--random", str(random_count), "--seed", str(seed)]
    names = launch_names(random_count, seed)
    record_dir = out_dir / "record"
    if capture.record(command, build, record_dir, ctas=(0, 1 << 30)) != 0:
        raise ValidationError(f"the recorded run failed. Its launch files are in {record_dir}.")
    paths = capture.import_recording(record_dir, tools.disassemblers)
    if not paths:
        raise ValidationError(
            f"the recorder wrote no launch files in {record_dir}. "
            "The output of the recorded run above names the cause."
        )
    traces = [Trace.load(path) for path in paths]
    lines = [f"# LaneScope validation on {gpu}", "", *_trace_lines(traces, names)]
    if tools.ncu is None:
        lines += [
            "",
            "Nsight Compute was not found, so the comparison was skipped. Run "
            "`lanescope get-ncu` or install Nsight Compute, then run the check again.",
        ]
        return _write(out_dir, lines, [], [])
    report = out_dir / "ncu" / "bench.ncu-rep"
    _profile(tools, command, report)
    kernels = compare.read_report(report)
    compare.dump(kernels, out_dir / "ncu-metrics.json")
    launches, problems = pair_launches(traces, kernels, random_count, seed)
    rows = [
        row
        for launch in launches
        if launch.trace is not None and launch.counters is not None
        for row in compare.compare(launch.index, launch.trace, launch.counters)
    ]
    predictions = predict_transposes(launches)
    lines += ["", *problems]
    lines += [
        "",
        *_metric_lines(kernels),
        "",
        *_summary_lines(rows, len(calibration.cases()), random_count, seed),
        "",
        *_prediction_lines(predictions),
    ]
    lines += ["", *_mismatch_lines(rows, names)]
    return _write(out_dir, lines, rows, predictions)


def summary_path(trace_path: pathlib.Path) -> pathlib.Path | None:
    """Return the validation.json of the GPU check that wrote a trace, or None for no check.

    The check writes its traces as record/trace-<n>.npz and its summary in the
    folder above.
    """
    if trace_path.parent.name != "record" or not TRACE_NAME.fullmatch(trace_path.stem):
        return None
    summary = trace_path.parent.parent / SUMMARY
    return summary if summary.is_file() else None


def measured_counts(
    trace_path: pathlib.Path, kernel: str
) -> dict[tuple[int, str], tuple[float | None, str]] | None:
    """Return the Nsight Compute counts of a checked trace by offset and quantity, with its SASS.

    The check numbers its launches in the order of its trace files, and each row
    names its kernel, which must be the kernel of the trace. A trace of no check
    gives None.
    """
    summary = summary_path(trace_path)
    if summary is None:
        return None
    siblings = sorted(
        (path for path in trace_path.parent.glob("trace-*.npz") if TRACE_NAME.fullmatch(path.stem)),
        key=lambda path: int(path.stem.split("-")[1]),
    )
    if trace_path not in siblings:
        return None
    launch = siblings.index(trace_path)
    try:
        data = json.loads(summary.read_text())
    except (OSError, ValueError, RecursionError):
        # ValueError also covers a number with more digits than Python converts, and
        # RecursionError a summary nested deeper than the JSON reader goes.
        return None
    rows = data.get("rows") if isinstance(data, dict) else None
    found: dict[tuple[int, str], tuple[float | None, str]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or row.get("launch") != launch:
            continue
        offset, quantity, nsight = row.get("offset"), row.get("quantity"), row.get("nsight")
        # Nsight Compute names a C++ kernel with its signature, the recorder without it.
        name = row.get("kernel")
        if not isinstance(name, str) or name.split("(")[0].strip() != kernel:
            continue
        if isinstance(offset, int) and isinstance(quantity, str):
            value = None
            if isinstance(nsight, (int, float)) and not isinstance(nsight, bool):
                try:
                    value = float(nsight)
                except OverflowError:
                    value = None
                if value is not None and not math.isfinite(value):
                    value = None
            found[offset, quantity] = (value, str(row.get("sass", "")))
    return found


def predict_transposes(launches: list[Launch]) -> list[Prediction]:
    """Predict the padded and swizzled transposes from the tiled one, next to their counters."""
    by_name = {launch.name: launch for launch in launches}
    tile = by_name["transpose_tile"].trace if "transpose_tile" in by_name else None
    if tile is None:
        return []
    buffer = infer_buffer(tile)
    predictions = []
    for kernel, change in (
        ("transpose_padded", Padding(128, 4)),
        ("transpose_swizzled", XorSwizzle(5, 2, 5)),
    ):
        changed = apply(tile, buffer, change)
        counters = by_name[kernel].counters if kernel in by_name else None
        for kind, opcode in ((Kind.STORE, "STS"), (Kind.LOAD, "LDS")):
            predicted = sum(
                c.achieved for c in count(changed) if c.space == Space.SHARED and c.kind == kind
            )
            measured = None
            if counters is not None:
                values = counters.quantity("shared_wavefronts")[1]
                hits = [
                    value
                    for offset, value in values.items()
                    if compare.opcode(counters.sass.get(offset, "")).startswith(opcode)
                ]
                measured = sum(hits) if hits else None
            predictions.append(Prediction(kernel, change.describe(), kind, predicted, measured))
    return predictions


def _profile(tools: Toolchain, command: list[str], report: pathlib.Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    assert tools.ncu is not None
    result = subprocess.run(
        [str(tools.ncu), "--set", "full", "--import-source", "no", "--force-overwrite",
         "--export", str(report), *command],
        capture_output=True, text=True, check=False,
    )  # fmt: skip
    (report.parent / "ncu.log").write_text(result.stdout + result.stderr)
    if "ERR_NVGPUCTRPERM" in result.stdout + result.stderr:
        raise ValidationError(
            "Nsight Compute has no access to the GPU counters. On WSL2, enable "
            "'Allow access to the GPU performance counters to all users' in the NVIDIA "
            "Control Panel of Windows (Desktop, Enable Developer Settings) and run again."
        )
    if result.returncode != 0 or not report.exists():
        raise ValidationError(
            f"Nsight Compute failed. Its output is in {report.parent / 'ncu.log'}."
        )


def _trace_lines(traces: list[Trace], names: list[str]) -> list[str]:
    records = sum(len(trace) for trace in traces)
    dropped = sum(int(trace.metadata.get("dropped", 0)) for trace in traces)
    lines = [
        f"The recorder wrote {len(traces)} launches of {len(names)} "
        f"with {records} warp-level accesses."
    ]
    if dropped:
        lines.append(f"It dropped {dropped} accesses for lack of buffer space.")
    empty = [
        names[i] if i < len(names) else str(i) for i, trace in enumerate(traces) if len(trace) == 0
    ]
    if empty:
        lines.append(f"These launches have no records: {', '.join(empty)}.")
    return lines


def _metric_lines(kernels: list[compare.KernelCounters]) -> list[str]:
    if not kernels:
        return ["Nsight Compute reported no kernels."]
    lines = ["| Quantity | Nsight Compute metric |", "|---|---|"]
    for quantity in compare.QUANTITIES:
        # A kernel without global accesses reports no global metric at all.
        names = (kernel.quantity(quantity)[0] for kernel in kernels)
        lines.append(f"| {quantity} | {next(filter(None, names), None) or 'not found'} |")
    return lines


def _summary_lines(
    rows: list[compare.Row], calibration: int, random_count: int, seed: int
) -> list[str]:
    matched = sum(row.matches for row in rows)
    lines = [f"LaneScope matches Nsight Compute in {matched} of {len(rows)} instruction counts."]
    if random_count:
        held_out = [row for row in rows if calibration <= row.launch < calibration + random_count]
        lines.append(
            f"The {random_count} random lane patterns of seed {seed} give "
            f"{sum(row.matches for row in held_out)} of {len(held_out)}."
        )
    lines.append("")
    lines += ["| Opcode and quantity | Matched | Total |", "|---|---|---|"]
    for key, entry in compare.summarize(rows).items():
        lines.append(f"| {key} | {entry['matched']} | {entry['total']} |")
    return lines


def _prediction_lines(predictions: list[Prediction]) -> list[str]:
    lines = [
        "| Kernel | Change | Kind | Predicted wavefronts | Measured |",
        "|---|---|---|---|---|",
    ]
    for p in predictions:
        measured = "n/a" if p.measured is None else f"{p.measured:g}"
        lines.append(
            f"| {p.kernel} | {p.change} | {p.kind.name.lower()} | {p.predicted} | {measured} |"
        )
    return lines


def _mismatch_lines(rows: list[compare.Row], names: list[str]) -> list[str]:
    lines = [
        "| Launch | Offset | SASS | Quantity | LaneScope | Nsight Compute |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        if not row.matches:
            name = names[row.launch] if row.launch < len(names) else str(row.launch)
            nsight = "n/a" if row.nsight is None else f"{row.nsight:g}"
            lines.append(
                f"| {name} | {row.offset:#x} | {row.sass} | {row.quantity} "
                f"| {row.lanescope} | {nsight} |"
            )
    return lines


def _write(
    out_dir: pathlib.Path, lines: list[str], rows: list[compare.Row], predictions: list[Prediction]
) -> pathlib.Path:
    payload = {
        "rows": [dataclasses.asdict(row) | {"matches": row.matches} for row in rows],
        "predictions": [dataclasses.asdict(p) | {"kind": p.kind.name.lower()} for p in predictions],
    }
    (out_dir / SUMMARY).write_text(json.dumps(payload, indent=1))
    summary = out_dir / "validation.md"
    summary.write_text("\n".join(lines) + "\n")
    return summary
