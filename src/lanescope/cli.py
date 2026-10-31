import argparse
import json
import math
import pathlib
import sys
import tempfile
from collections.abc import Sequence

from lanescope import analysis, demos, html_report, mcp_server, text_view
from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver, DriverError, version_text
from lanescope.cuda.nvrtc import CompileError
from lanescope.cuda.toolchain import ToolchainError, cache_folder, find
from lanescope.fixes import layouts
from lanescope.gui import server
from lanescope.model.counters import count, worst_first
from lanescope.nsight import calibration, installer
from lanescope.nsight.validate import ValidationError, validate, validate_program
from lanescope.recorder import capture
from lanescope.recorder.build import BuildError, build_recorder
from lanescope.trace import Trace

TABLE_HEADER = (
    f"{'pc':8}{'line':>5}  {'space':7}{'kind':6}{'width':>5}{'requests':>10}  {'unit':11}"
    f"{'achieved':>9}{'ideal':>7}{'excessive':>10}"
)
ERRORS = (
    ValueError, OSError, ImportError, BuildError, CompileError, DriverError, ToolchainError,
    ValidationError,
)  # fmt: skip
# The recorder keeps from 1 to 2**26 records per launch and ignores any other capacity.
MAX_CAPACITY = 1 << 26


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except ERRORS as error:
        print(f"lanescope: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lanescope",
        description="Explain the shared and global memory counters of a CUDA kernel trace.",
    )
    commands = parser.add_subparsers(required=True, metavar="command")

    explain = commands.add_parser(
        "explain", help="counters of every memory instruction, worst first"
    )
    explain.add_argument("trace", type=pathlib.Path)
    explain.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    explain.set_defaults(run=_explain)

    lanes = commands.add_parser("lanes", help="lane-by-lane view of one access of an instruction")
    lanes.add_argument("trace", type=pathlib.Path)
    lanes.add_argument(
        "--pc", type=_number, required=True, help="PC of the instruction, such as 0x120"
    )
    lanes.add_argument("--row", type=int, help="trace row to show instead of the worst access")
    lanes.set_defaults(run=_lanes)

    change = commands.add_parser(
        "whatif", help="counters after a layout change of one shared buffer"
    )
    change.add_argument("trace", type=pathlib.Path)
    change.add_argument("--buffer", type=_buffer, required=True, help="START:SIZE in bytes")
    kinds = change.add_mutually_exclusive_group(required=True)
    kinds.add_argument("--pad", type=_padding, help="ROW_BYTES:PAD_BYTES")
    kinds.add_argument("--swizzle", type=_swizzle, help="BITS,BASE,SHIFT on byte offsets")
    change.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    change.set_defaults(run=_whatif)

    search = commands.add_parser(
        "suggest", help="padding and swizzle changes ranked by predicted excess"
    )
    search.add_argument("trace", type=pathlib.Path)
    search.add_argument(
        "--pc", type=_number, help="instruction whose buffer to search, the worst by default"
    )
    search.add_argument("--buffer", type=_buffer, help="START:SIZE in bytes, inferred by default")
    search.add_argument("--limit", type=_count, default=5, help="most changes to list")
    search.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    search.set_defaults(run=_suggest)

    html_report = commands.add_parser("report", help="write an HTML report with lane grids")
    html_report.add_argument("trace", type=pathlib.Path)
    html_report.add_argument("-o", "--out", type=pathlib.Path, required=True, help="HTML file")
    html_report.add_argument("--limit", type=_count, default=20, help="most instructions to draw")
    html_report.set_defaults(run=_report)

    demo = commands.add_parser("demo", help="write and analyze the trace of a tiled transpose")
    demo.add_argument("--out", type=pathlib.Path, default=pathlib.Path("transpose.npz"))
    demo.set_defaults(run=_demo)

    recorder = commands.add_parser("build", help="build the recorder for a GPU architecture")
    recorder.add_argument("--out", type=pathlib.Path, required=True, help="folder for the build")
    recorder.add_argument("--arch", help="such as sm_89, the local GPU by default")
    recorder.set_defaults(run=_build)

    run = commands.add_parser("record", help="run a CUDA program under the recorder")
    run.add_argument(
        "--out", type=pathlib.Path, required=True, help="folder for launches and traces"
    )
    run.add_argument("--arch", help="such as sm_89, the local GPU by default")
    run.add_argument(
        "--ctas", type=_ctas, default=(0, 64), help="FIRST:COUNT of the CTAs to record"
    )
    run.add_argument("--kernel", default="", help="record only kernels whose name holds this text")
    run.add_argument(
        "--capacity", type=_capacity, default=1 << 18, help="records kept per launch, up to 2^26"
    )
    run.add_argument("command", nargs=argparse.REMAINDER, help="-- program and its arguments")
    run.set_defaults(run=_record)

    imports = commands.add_parser("import", help="turn the launch files of a recording into traces")
    imports.add_argument("folder", type=pathlib.Path)
    imports.set_defaults(run=_import)

    calibration = commands.add_parser("bench", help="calibration kernels with known lane patterns")
    steps = calibration.add_subparsers(required=True, metavar="step")
    listing = steps.add_parser("list", help="print every case in launch order")
    listing.set_defaults(run=_bench_list)
    compiling = steps.add_parser("build", help="compile the kernels with NVRTC, without a GPU")
    compiling.add_argument("--arch", required=True, help="such as sm_89")
    compiling.add_argument(
        "--out", type=pathlib.Path, required=True, help="cubin file, with its .cu source beside it"
    )
    compiling.set_defaults(run=_bench_build)
    running = steps.add_parser("run", help="run every case once on the local GPU")
    running.add_argument(
        "--cubin", type=pathlib.Path, required=True, help="kernels for this GPU from bench build"
    )
    _random_options(running)
    running.set_defaults(run=_bench_run)
    timing = steps.add_parser(
        "time", help="time the transposes and the matmuls with their plain and changed layouts"
    )
    timing.add_argument(
        "--cubin", type=pathlib.Path, required=True, help="kernels for this GPU from bench build"
    )
    timing.add_argument(
        "--size", type=_size, default=8192, help="rows and columns of the matrix, a multiple of 32"
    )
    timing.add_argument(
        "--matmul-size",
        type=_matmul_size,
        default=4096,
        help="rows and columns of the matrices to multiply, a multiple of 64",
    )
    timing.add_argument("--repeats", type=_size, default=50, help="launches timed per kernel")
    timing.set_defaults(run=_bench_time)

    check = commands.add_parser(
        "validate", help="compare LaneScope with Nsight Compute on the local GPU"
    )
    check.add_argument("--out", type=pathlib.Path, required=True, help="folder for every result")
    check.add_argument("--arch", help="such as sm_89, the local GPU by default")
    _random_options(check)
    check.add_argument(
        "--kernel",
        default="",
        help="with a program, compare only kernels whose name holds this text",
    )
    check.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="-- program and its arguments, the bench by default",
    )
    check.set_defaults(run=_validate)

    info = commands.add_parser(
        "gpu-info", help="print the GPU, its driver, and the CUDA tools found"
    )
    info.add_argument(
        "--cuda-version", action="store_true", help="print only the driver CUDA version"
    )
    info.set_defaults(run=_gpu_info)

    fetch = commands.add_parser(
        "get-ncu",
        help="download Nsight Compute for the local driver into ~/.cache/lanescope, "
        "or LANESCOPE_CACHE, without root",
    )
    fetch.add_argument(
        "--cuda-version",
        type=_cuda_version,
        help="such as 12.9, the driver's version by default; a download even when ncu is found",
    )
    fetch.add_argument(
        "--force", action="store_true", help="download again, even when ncu is found"
    )
    fetch.set_defaults(run=_get_ncu)

    agents = commands.add_parser(
        "mcp", help="serve the analyses to kernel agents over MCP on stdio"
    )
    agents.set_defaults(run=_mcp)

    window = commands.add_parser("gui", help="explore traces and try fixes in the browser")
    window.add_argument(
        "paths", nargs="*", type=pathlib.Path, help="trace files or folders, this folder by default"
    )
    window.add_argument(
        "--host", default="127.0.0.1", help="address to listen on, 0.0.0.0 inside a container"
    )
    window.add_argument("--port", type=_port, default=server.DEFAULT_PORT, help="port to listen on")
    window.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="another host name that the page may use, such as a machine name",
    )
    window.add_argument("--open", action="store_true", help="open the page in the browser")
    window.add_argument("--demo", action="store_true", help="add the traces of four demo kernels")
    window.set_defaults(run=_gui)
    return parser


def _explain(args: argparse.Namespace) -> int:
    trace = Trace.load(args.trace)
    data = analysis.explain(trace)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    kernel = _printable(trace.metadata.get("kernel", "?"))
    arch = _printable(trace.metadata.get("arch", "?"))
    print(f"kernel {kernel} ({arch}), {len(trace)} warp-level accesses")
    print()
    print(TABLE_HEADER)
    for record in data["instructions"]:
        print(
            f"{int(record['pc'], 16):#06x}  {record['line'] or '-':>5}  {record['space']:7}"
            f"{record['kind']:6}{record['width']:5d}{record['requests']:10d}  {record['unit']:11}"
            f"{record['achieved']:9d}{record['ideal']:7d}{record['excessive']:10d}"
        )
    return 0


def _lanes(args: argparse.Namespace) -> int:
    print(analysis.lanes(Trace.load(args.trace), args.pc, args.row))
    return 0


def _whatif(args: argparse.Namespace) -> int:
    data = analysis.whatif(Trace.load(args.trace), args.buffer, args.pad or args.swizzle)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        _print_comparison(args.buffer, data)
    return 0


def _suggest(args: argparse.Namespace) -> int:
    data = analysis.suggest(Trace.load(args.trace), args.buffer, args.pc, args.limit)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"buffer {data['buffer']['start']:#x}+{data['buffer']['size']}")
    if data["found"] == 0:
        print("no padding or swizzle lowers the excess wavefronts of this buffer")
    for item in data["changes"]:
        before, after = item["excessive_sectors_before"], item["excessive_sectors_after"]
        sectors = f", excessive global sectors {before} -> {after}" if before != after else ""
        print(
            f"{item['change']}: excessive wavefronts {item['excessive_before']} -> "
            f"{item['excessive_after']}{sectors}, {item['extra_bytes']} extra bytes"
        )
    return 0


def _report(args: argparse.Namespace) -> int:
    args.out.write_text(html_report.render(Trace.load(args.trace), args.limit), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def _demo(args: argparse.Namespace) -> int:
    trace = demos.transpose_trace()
    trace.save(args.out)
    print(f"wrote {args.out}")
    print()
    _explain(argparse.Namespace(trace=args.out, json=False))
    worst = worst_first(count(trace))[0]
    print()
    print(text_view.describe_row(trace, worst.worst_row))
    buffer = layouts.Buffer(0, demos.TILE * demos.TILE * demos.FLOAT_BYTES)
    row_bytes = demos.TILE * demos.FLOAT_BYTES
    for change in (layouts.Padding(row_bytes, demos.FLOAT_BYTES), layouts.XorSwizzle(5, 2, 5)):
        print()
        _print_comparison(buffer, analysis.whatif(trace, buffer, change))
    return 0


def _build(args: argparse.Namespace) -> int:
    built = build_recorder(args.out, args.arch or _local_arch())
    print(f"library {built.library}")
    print(f"patches {built.patches}")
    return 0


def _record(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("give the program to record after --")
    built = build_recorder(args.out / "build", args.arch or _local_arch())
    status = capture.record(
        command, built, args.out, capacity=args.capacity, ctas=args.ctas, kernel=args.kernel
    )
    launches = capture.launch_files(args.out)
    if status:
        # subprocess reports a program that a signal kills by the negative signal number.
        ending = f"was killed by signal {-status}" if status < 0 else f"exited with status {status}"
        # A failing program may exit before the recorder saves its last launches, so its
        # launch files wait for an import by hand.
        found = (
            f"lanescope import {args.out} turns the launch files it left into traces"
            if launches
            else "It launched no kernel that the recorder saw"
        )
        raise ValueError(f"the program {ending}. {found}")
    if not launches:
        raise ValueError(
            f"{args.out} holds no launch files. "
            "The program launched no kernel that the recorder saw"
        )
    for path in _import_traces(args.out):
        print(f"wrote {path}")
    return 0


def _import_traces(folder: pathlib.Path) -> list[pathlib.Path]:
    disassemblers = find().disassemblers
    if not disassemblers:
        print(f"lanescope: {capture.NO_DISASSEMBLER}", file=sys.stderr)
    return capture.import_recording(folder, disassemblers)


def _import(args: argparse.Namespace) -> int:
    paths = _import_traces(args.folder)
    if not paths:
        raise ValueError(f"{args.folder} holds no launch files")
    for path in paths:
        print(f"wrote {path}")
    return 0


def _bench_list(args: argparse.Namespace) -> int:
    for launch, case in enumerate(calibration.launches()):
        print(f"{launch:3d}  {case.name:40} {case.kernel}  mask {case.mask:#010x}")
    return 0


def _bench_build(args: argparse.Namespace) -> int:
    source = args.out.with_suffix(".cu")
    if source == args.out:
        raise ValueError(f"{source} names the source, so name the cubin such as bench.cubin")
    args.out.write_bytes(calibration.compile_kernels(args.arch, source_path=source))
    print(f"wrote {args.out}")
    print(f"wrote {source}")
    return 0


def _bench_run(args: argparse.Namespace) -> int:
    calibration.run(args.cubin, args.random, args.seed)
    return 0


def _bench_time(args: argparse.Namespace) -> int:
    times = calibration.time_transposes(args.cubin, args.size, args.repeats)
    print(f"{args.size} x {args.size} float matrix, {args.repeats} launches per kernel")
    print(f"{'kernel':20} {'ms':>8} {'GB/s':>8} {'speedup':>8}")
    for time in times:
        rate = time.gigabytes_per_second
        print(_timing_line(time.kernel, time.milliseconds, rate, times[0].milliseconds))
    matmuls = calibration.time_matmuls(args.cubin, args.matmul_size, args.repeats)
    size = args.matmul_size
    print()
    shape = f"{size} x {size} x {size}"
    print(f"{shape} half-precision matrix multiply, {args.repeats} launches per kernel")
    print(f"{'kernel':20} {'ms':>8} {'TFLOP/s':>8} {'speedup':>8}")
    for time in matmuls:
        print(_timing_line(time.kernel, time.milliseconds, time.teraflops, matmuls[0].milliseconds))
    return 0


def _timing_line(kernel: str, milliseconds: float, rate: float, baseline: float) -> str:
    speedup = f"{baseline / milliseconds:.2f}x"
    return f"{kernel:20} {milliseconds:8.3f} {rate:8.1f} {speedup:>8}"


def _validate(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if command and args.random:
        raise ValueError("random lane patterns apply only to the bench, not to a program")
    if command:
        summary = validate_program(args.out, command, args.kernel, args.arch)
    else:
        summary = validate(args.out, args.arch, random_count=args.random, seed=args.seed)
    print(summary.read_text())
    print(f"wrote {summary}")
    return 0


def _gpu_info(args: argparse.Namespace) -> int:
    driver = Driver()
    version = driver.version()
    if args.cuda_version:
        print(version_text(version))
        return 0
    print(f"GPU {driver.name()}, {driver.arch()}, driver CUDA version {version_text(version)}")
    tools = find()
    names = (
        "compiler",
        "nvcc",
        "nvrtc",
        "ptxas",
        "sanitizer_lib",
        "cuda_include",
        "ncu",
        "nvdisasm",
    )
    for name in names:
        print(f"{name:14} {getattr(tools, name) or 'not found'}")
    if tools.nvrtc is not None:
        major, minor = nvrtc.version(tools.nvrtc)
        print(f"CUDA tools     version {major}.{minor}")
    return 0


def _get_ncu(args: argparse.Namespace) -> int:
    found = find().ncu
    if found is None or args.force or args.cuda_version is not None:
        folder = cache_folder()
        if folder is None:
            raise ValueError("set HOME or LANESCOPE_CACHE to a folder for the download")
        cuda = args.cuda_version
        if cuda is None:
            driver = Driver()
            version = driver.version()
            driver.close()
            cuda = (version // 1000, version % 1000 // 10)
        print(f"downloading Nsight Compute for CUDA {cuda[0]}.{cuda[1]} from NVIDIA", flush=True)
        found = installer.install(cuda, folder, replace=args.force)
    print(f"Nsight Compute is at {found}")
    return 0


def _mcp(args: argparse.Namespace) -> int:
    mcp_server.serve()
    return 0


def _gui(args: argparse.Namespace) -> int:
    paths = list(args.paths)
    with tempfile.TemporaryDirectory(prefix="lanescope-demo-") as folder:
        if args.demo:
            demo = pathlib.Path(folder) / "demo"
            demo.mkdir()
            for name, trace in demos.demo_traces().items():
                trace.save(demo / name)
            paths.append(demo)
        server.serve(
            paths or [pathlib.Path.cwd()], args.host, args.port, args.allow_host, args.open
        )
    return 0


def _local_arch() -> str:
    driver = Driver()
    arch = driver.arch()
    driver.close()
    return arch


def _print_comparison(buffer: layouts.Buffer, data: dict) -> None:
    print(
        f"{data['change']}, buffer {buffer.start:#x}+{buffer.size} "
        f"grows by {data['extra_bytes']} bytes"
    )
    if data["later_shift"]:
        print(f"shared addresses above the buffer move up by {data['later_shift']} bytes")
    print(
        f"{'pc':8}{'line':>5}  {'space':7}{'kind':7}{'unit':11}{'before':>8}{'after':>8}"
        f"{'ideal':>7}{'excessive before':>18}{'after':>7}"
    )
    for item in data["instructions"]:
        old, new = item["before"], item["after"]
        print(
            f"{int(old['pc'], 16):#06x}  {old['line'] or '-':>5}  {old['space']:7}{old['kind']:7}"
            f"{old['unit']:11}{old['achieved']:8d}{new['achieved']:8d}{old['ideal']:7d}"
            f"{old['excessive']:18d}{new['excessive']:7d}"
        )


# argparse names the parser function in its message for a ValueError and drops the
# reason, so each parser raises ArgumentTypeError with the reason instead.
def _random_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--random",
        type=_count,
        default=0,
        help="random lane patterns to add after the fixed cases, as many for ldmatrix and cp.async",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the random lane patterns")


def _number(text: str) -> int:
    try:
        return int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number such as 0x120, got {text!r}") from None


def _cuda_version(text: str) -> tuple[int, int]:
    parts = text.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("expected a CUDA version such as 12.9")
    return int(parts[0]), int(parts[1])


def _port(text: str) -> int:
    return _integer(text, 0, 65535, "expected a port from 0 to 65535")


def _count(text: str) -> int:
    return _integer(text, 0, math.inf, "expected a count of zero or more")


def _size(text: str) -> int:
    return _integer(text, 1, math.inf, "expected a count of one or more")


def _matmul_size(text: str) -> int:
    size = _integer(text, 1, math.inf, "expected a positive multiple of 64")
    if size % calibration.MATMUL_TILE:
        raise argparse.ArgumentTypeError("expected a positive multiple of 64")
    return size


def _capacity(text: str) -> int:
    return _integer(text, 1, MAX_CAPACITY, f"expected a count from 1 to {MAX_CAPACITY} (2^26)")


def _integer(text: str, low: int, high: float, reason: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(reason) from None
    if not low <= value <= high:
        raise argparse.ArgumentTypeError(reason)
    return value


def _printable(value: object) -> str:
    # Trace files are untrusted, and control characters would reach the terminal.
    return "".join(char if char.isprintable() else "?" for char in str(value))


def _buffer(text: str) -> layouts.Buffer:
    start, size = _pair(text, ":")
    try:
        return layouts.Buffer(start, size)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def _ctas(text: str) -> tuple[int, int]:
    first, number = _pair(text, ":")
    if first < 0 or number <= 0:
        raise argparse.ArgumentTypeError("expected FIRST:COUNT with a positive count")
    return first, number


def _padding(text: str) -> layouts.Padding:
    row_bytes, pad_bytes = _pair(text, ":")
    return layouts.Padding(row_bytes, pad_bytes)


def _swizzle(text: str) -> layouts.XorSwizzle:
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("a swizzle needs BITS,BASE,SHIFT")
    bits, base, shift = (_number(part) for part in parts)
    return layouts.XorSwizzle(bits, base, shift)


def _pair(text: str, separator: str) -> tuple[int, int]:
    parts = text.split(separator)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected two numbers separated by {separator!r}")
    return _number(parts[0]), _number(parts[1])
