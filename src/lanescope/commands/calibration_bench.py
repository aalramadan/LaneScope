import argparse
import pathlib

from lanescope.commands import argument_types
from lanescope.nsight import calibration


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare bench with its list, build, run, and time steps."""
    bench = commands.add_parser("bench", help="calibration kernels with known lane patterns")
    steps = bench.add_subparsers(required=True, metavar="step")
    listing = steps.add_parser("list", help="print every case in launch order")
    listing.set_defaults(run=_list)
    compiling = steps.add_parser("build", help="compile the kernels with NVRTC, without a GPU")
    compiling.add_argument("--arch", required=True, help="such as sm_89")
    compiling.add_argument(
        "--out", type=pathlib.Path, required=True, help="cubin file, with its .cu source beside it"
    )
    compiling.set_defaults(run=_build)
    running = steps.add_parser("run", help="run every case once on the local GPU")
    running.add_argument(
        "--cubin", type=pathlib.Path, required=True, help="kernels for this GPU from bench build"
    )
    add_random_options(running)
    running.set_defaults(run=_run)
    timing = steps.add_parser(
        "time", help="time the transposes and the matmuls with their plain and changed layouts"
    )
    timing.add_argument(
        "--cubin", type=pathlib.Path, required=True, help="kernels for this GPU from bench build"
    )
    timing.add_argument(
        "--size",
        type=argument_types.size,
        default=8192,
        help="rows and columns of the matrix, a multiple of 32",
    )
    timing.add_argument(
        "--matmul-size",
        type=argument_types.matmul_size,
        default=4096,
        help="rows and columns of the matrices to multiply, a multiple of 64",
    )
    timing.add_argument(
        "--repeats", type=argument_types.size, default=50, help="launches timed per kernel"
    )
    timing.set_defaults(run=_time)


def add_random_options(parser: argparse.ArgumentParser) -> None:
    """Add the options that append random lane patterns to the calibration cases."""
    parser.add_argument(
        "--random",
        type=argument_types.count,
        default=0,
        help="random lane patterns to add after the fixed cases, as many for ldmatrix and cp.async",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed of the random lane patterns")


def _list(args: argparse.Namespace) -> int:
    for launch, case in enumerate(calibration.launches()):
        print(f"{launch:3d}  {case.name:40} {case.kernel}  mask {case.mask:#010x}")
    return 0


def _build(args: argparse.Namespace) -> int:
    source = args.out.with_suffix(".cu")
    if source == args.out:
        raise ValueError(f"{source} names the source, so name the cubin such as bench.cubin")
    args.out.write_bytes(calibration.compile_kernels(args.arch, source_path=source))
    print(f"wrote {args.out}")
    print(f"wrote {source}")
    return 0


def _run(args: argparse.Namespace) -> int:
    calibration.run(args.cubin, args.random, args.seed)
    return 0


def _time(args: argparse.Namespace) -> int:
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
