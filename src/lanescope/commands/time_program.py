import argparse
import pathlib
import sys
import tempfile

from lanescope import kernel_names, kernel_times
from lanescope.commands import argument_types
from lanescope.cuda.toolchain import cache_folder
from lanescope.recorder.build import build_kernel_timer
from lanescope.summary_text import duration, duration_range, percent
from lanescope.text_view import printable

USAGE = "lanescope time [-h] [--rounds N] FOLDER [FOLDER ...] [-- command ...]"
TABLE_HEADER = "{:>12}  {:>7}  {:>8}  {:>18}  kernel"


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare time."""
    timing = commands.add_parser(
        "time",
        usage=USAGE,
        help="time every kernel of the stored command of each folder, without the recorder",
    )
    _add_rounds(timing)
    # A list of folders before -- would take the command too, so the folders and the
    # command arrive as one list that _time splits.
    timing.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        metavar="FOLDER",
        help="recording folders, or one folder and the program to time after --",
    )
    timing.set_defaults(run=_time)


def _add_rounds(parser: argparse.ArgumentParser, default: int = 5) -> None:
    parser.add_argument(
        "--rounds",
        type=argument_types.positive_count,
        metavar="N",
        default=default,
        help="runs of each folder, taken in turns over the folders",
    )


def _time(args: argparse.Namespace) -> int:
    split = args.arguments.index("--") if "--" in args.arguments else len(args.arguments)
    folders_parser = argparse.ArgumentParser(prog="lanescope time", usage=USAGE)
    folders_parser.add_argument("folders", nargs="+", type=pathlib.Path)
    _add_rounds(folders_parser, args.rounds)
    parsed = folders_parser.parse_intermixed_args(args.arguments[:split])
    command = args.arguments[split + 1 :]
    folders: list[pathlib.Path] = parsed.folders
    if command and len(folders) != 1:
        raise ValueError("a program after -- takes exactly one folder, which stores it")
    if not command:
        for folder in folders:
            kernel_times.read_stored_command(folder)

    def report(round_index: int, folder: pathlib.Path, seconds: float) -> None:
        print(
            f"round {round_index + 1} of {parsed.rounds}: {folder} ran in {seconds:.2f} s",
            flush=True,
        )

    # The timer builds outside the folders, which a failed timing leaves as they were.
    with tempfile.TemporaryDirectory(prefix="lanescope-timer-") as build:
        timer = build_kernel_timer(pathlib.Path(build), cache=cache_folder())
        results = kernel_times.time_folders(
            folders, timer, parsed.rounds, report, [command] if command else None
        )
    for folder, result in zip(folders, results, strict=True):
        print()
        _print_kernel_times(folder, result)
        print(f"wrote {folder / kernel_times.TIMES_FILE}")
    return 0


def _print_kernel_times(folder: pathlib.Path, result: dict) -> None:
    kernels = kernel_times.ranked(result)
    names = kernel_names.readable(kernel.name for kernel in kernels)
    total = kernel_times.median_total(result)
    rounds = result["rounds"]
    print(
        f"{printable(folder)}: {len(kernels):,} kernel{'' if len(kernels) == 1 else 's'}, "
        f"{duration(total)} of kernel time, median of {rounds} round{'' if rounds == 1 else 's'}"
    )
    print(TABLE_HEADER.format("median", "of time", "launches", "range over rounds"))
    for kernel in kernels:
        print(
            f"{duration(kernel.median_ns):>12}  "
            f"{percent(round(kernel.median_ns), round(total)):>7}  "
            f"{kernel.launches:>8,}  "
            f"{_rounds_range(kernel):>18}  "
            f"{printable(kernel_names.short_name(names[kernel.name]))}"
        )
        if not kernel.even_launches:
            launches = ", ".join(str(n) for n in result["kernels"][kernel.name]["launches"])
            print(
                "lanescope: the launches of "
                f"{printable(kernel_names.short_name(names[kernel.name]))} "
                f"in {printable(folder)} "
                f"differ between rounds ({launches}). Its median compares rounds of unequal work.",
                file=sys.stderr,
            )
    if untimed := sum(result.get("untimed_launches", [])):
        print(
            f"lanescope: CUPTI gave no time for {untimed:,} launches in {printable(folder)}, "
            "which its times leave out.",
            file=sys.stderr,
        )
    if unfinished := sum(result.get("unfinished_processes", [])):
        print(
            f"lanescope: {unfinished:,} process{'' if unfinished == 1 else 'es'} in the "
            f"{rounds} round{'' if rounds == 1 else 's'} of {printable(folder)} ended through "
            "_exit or a signal, without the exit handler of the timer. Their kernel times stop "
            f"at the last flush of the timer, which runs every {kernel_times.FLUSH_SECONDS:g} s.",
            file=sys.stderr,
        )


def _rounds_range(kernel: kernel_times.KernelTime) -> str:
    """Return the fastest and the slowest round of a kernel in the unit of its median."""
    return duration_range(kernel.fastest_round_ns, kernel.slowest_round_ns, kernel.median_ns)
