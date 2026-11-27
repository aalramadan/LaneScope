import argparse
import pathlib

from lanescope.commands import argument_types, calibration_bench
from lanescope.cuda import nvrtc
from lanescope.cuda.driver import Driver, version_text
from lanescope.cuda.toolchain import cache_folder, find
from lanescope.nsight import installer
from lanescope.nsight.validate import validate, validate_program


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare validate, gpu-info, and get-ncu."""
    check = commands.add_parser(
        "validate", help="compare LaneScope with Nsight Compute on the local GPU"
    )
    check.add_argument("--out", type=pathlib.Path, required=True, help="folder for every result")
    check.add_argument("--arch", help="such as sm_89, the local GPU by default")
    calibration_bench.add_random_options(check)
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
        type=argument_types.cuda_version,
        help="such as 12.9, the driver's version by default; a download even when ncu is found",
    )
    fetch.add_argument(
        "--force", action="store_true", help="download again, even when ncu is found"
    )
    fetch.set_defaults(run=_get_ncu)


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
        "cupti_lib",
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
