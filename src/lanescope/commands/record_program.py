import argparse
import pathlib
import sys

from lanescope import kernel_times
from lanescope.commands import argument_types
from lanescope.cuda.driver import Driver
from lanescope.cuda.toolchain import cache_folder, find_toolchain
from lanescope.recorder import capture
from lanescope.recorder.build import build_recorder
from lanescope.trace import read_metadata

# A recording with more traces than this prints their count in place of their paths.
LISTED_TRACES = 10


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare build, record, and import."""
    recorder = commands.add_parser("build", help="build the recorder for a GPU architecture")
    recorder.add_argument("--out", type=pathlib.Path, required=True, help="folder for the build")
    recorder.add_argument("--arch", help="such as sm_89, the local GPU by default")
    recorder.set_defaults(run=_build_recorder_files)

    run = commands.add_parser("record", help="run a CUDA program under the recorder")
    run.add_argument(
        "--out", type=pathlib.Path, required=True, help="folder for launches and traces"
    )
    run.add_argument("--arch", help="such as sm_89, the local GPU by default")
    run.add_argument(
        "--ctas",
        type=argument_types.ctas,
        default=(0, 64),
        help="FIRST:COUNT of the CTAs to record",
    )
    run.add_argument("--kernel", default="", help="record only kernels whose name holds this text")
    run.add_argument(
        "--capacity",
        type=argument_types.capacity,
        default=1 << 18,
        help="records kept per launch, up to 2^26",
    )
    run.add_argument("command", nargs=argparse.REMAINDER, help="-- program and its arguments")
    run.set_defaults(run=_record)

    imports = commands.add_parser("import", help="turn the launch files of a recording into traces")
    imports.add_argument("folder", type=pathlib.Path)
    imports.set_defaults(run=_import_launch_files)


def _build_recorder_files(args: argparse.Namespace) -> int:
    built = build_recorder(args.out, args.arch or _local_arch())
    print(f"library {built.library}")
    print(f"patches {built.patches}")
    return 0


def _record(args: argparse.Namespace) -> int:
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("give the program to record after --")
    built = build_recorder(args.out / "build", args.arch or _local_arch(), cache=cache_folder())
    kernel_times.write_stored_command(args.out, command)
    status = capture.record_command(
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
    paths, disassembled = _import_traces(args.out)
    _report_traces(args.out, paths, disassembled, args.capacity)
    return 0


def _import_traces(folder: pathlib.Path) -> tuple[list[pathlib.Path], bool]:
    disassemblers = find_toolchain().disassemblers
    if not disassemblers:
        print(f"lanescope: {capture.NO_DISASSEMBLER}", file=sys.stderr)
    return capture.import_recording(folder, disassemblers), bool(disassemblers)


def _import_launch_files(args: argparse.Namespace) -> int:
    paths, disassembled = _import_traces(args.folder)
    if not paths:
        raise ValueError(f"{args.folder} holds no launch files")
    _report_traces(args.folder, paths, disassembled)
    return 0


def _report_traces(
    folder: pathlib.Path, paths: list[pathlib.Path], disassembled: bool, capacity: int = 0
) -> None:
    """Print the traces of a recording, what limits their counts, and the next command."""
    if len(paths) <= LISTED_TRACES:
        for path in paths:
            print(f"wrote {path}")
    else:
        print(f"wrote {len(paths)} traces to {folder}")
    metadata = [read_metadata(path) for path in paths]
    launches = f"{len(paths)} launches" if len(paths) != 1 else "1 launch"
    without = sum(not item.get("lines") for item in metadata)
    if disassembled and without:
        print(
            f"lanescope: no source lines in {without} of {launches}, since their kernels were "
            "built without line tables. Build the program with -lineinfo, such as "
            "nvcc -lineinfo, to see the line of each instruction.",
            file=sys.stderr,
        )
    lost = [int(item.get("dropped", 0)) for item in metadata if item.get("dropped")]
    if lost:
        larger = (
            f", such as --capacity {min(2 * capacity, argument_types.MAX_CAPACITY)}"
            if capacity
            else ""
        )
        print(
            f"lanescope: the record buffer filled up in {len(lost)} of {launches} and left out "
            f"{sum(lost):,} accesses, therefore their counts are low. Record again with a larger "
            f"--capacity{larger}, or with fewer CTAs, such as --ctas 0:8.",
            file=sys.stderr,
        )
    timing = (
        f"lanescope time {folder} times its kernels without the recorder, "
        if (folder / kernel_times.RUN_FILE).is_file()
        else ""
    )
    if len(paths) > 1:
        print(
            f"Next, lanescope summary {folder} ranks the conflicts of all {len(paths)} launches, "
            f"{timing}and lanescope gui {folder} opens them in the browser."
        )
    elif paths:
        print(
            f"Next, lanescope explain {paths[0]} lists its memory instructions, worst first, "
            f"{timing}and lanescope gui {folder} opens it in the browser."
        )


def _local_arch() -> str:
    driver = Driver()
    arch = driver.arch()
    driver.close()
    return arch
