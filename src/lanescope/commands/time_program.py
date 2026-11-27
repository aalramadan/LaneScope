import argparse
import pathlib
import sys

from lanescope import kernel_names, kernel_times
from lanescope.commands import argument_types
from lanescope.cuda.toolchain import cache_folder
from lanescope.recorder.build import build_kernel_timer
from lanescope.summary_text import percent
from lanescope.text_view import printable

USAGE = "lanescope time [-h] [--rounds N] FOLDER [FOLDER ...] [-- command ...]"
TABLE_HEADER = "{:>12}  {:>7}  {:>8}  {:>18}  kernel"
DURATION_UNITS = ((1e9, "s"), (1e6, "ms"), (1e3, "us"))


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
        type=argument_types.size,
        metavar="N",
        default=default,
        help="runs of each folder, taken in turns over the folders",
    )


def _time(args: argparse.Namespace) -> int:
    split = args.arguments.index("--") if "--" in args.arguments else len(args.arguments)
    folders_parser = argparse.ArgumentParser(prog="lanescope time", usage=USAGE)
    folders_parser.add_argument("folders", nargs="+", type=pathlib.Path)
    _add_rounds(folders_parser, args.rounds)
    parsed = folders_parser.parse_args(args.arguments[:split])
    command = args.arguments[split + 1 :]
    folders: list[pathlib.Path] = parsed.folders
    if command:
        if len(folders) != 1:
            raise ValueError("a program after -- takes exactly one folder, which stores it")
        kernel_times.write_run(folders[0], command)
    for folder in folders:
        kernel_times.read_run(folder)
    timer = build_kernel_timer(folders[0] / "build", cache=cache_folder())

    def report(round_index: int, folder: pathlib.Path, seconds: float) -> None:
        print(
            f"round {round_index + 1} of {parsed.rounds}: {folder} ran in {seconds:.2f} s",
            flush=True,
        )

    results = kernel_times.time_folders(folders, timer, parsed.rounds, report)
    for folder, result in zip(folders, results, strict=True):
        print()
        _print_table(folder, result)
        print(f"wrote {folder / kernel_times.TIMES_FILE}")
    return 0


def _print_table(folder: pathlib.Path, result: dict) -> None:
    kernels = kernel_times.ranked(result)
    names = kernel_names.readable(kernel.name for kernel in kernels)
    total = sum(kernel.median_ns for kernel in kernels)
    rounds = result["rounds"]
    print(
        f"{printable(folder)}: {len(kernels):,} kernels, {_duration(total)} of kernel time, "
        f"median of {rounds} round{'' if rounds == 1 else 's'}"
    )
    print(TABLE_HEADER.format("median", "of time", "launches", "range over rounds"))
    for kernel in kernels:
        print(
            f"{_duration(kernel.median_ns):>12}  "
            f"{percent(round(kernel.median_ns), round(total)):>7}  "
            f"{kernel.launches:>8,}  "
            f"{_range(kernel.fastest_round_ns, kernel.slowest_round_ns):>18}  "
            f"{printable(kernel_names.short(names[kernel.name]))}"
        )
        if not kernel.even_launches:
            launches = ", ".join(str(n) for n in result["kernels"][kernel.name]["launches"])
            print(
                f"lanescope: the launches of {printable(kernel_names.short(names[kernel.name]))} "
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


def _unit(nanoseconds: float) -> tuple[float, str]:
    return next(
        ((scale, name) for scale, name in DURATION_UNITS if nanoseconds >= scale), (1.0, "ns")
    )


def _duration(nanoseconds: float) -> str:
    scale, name = _unit(nanoseconds)
    return f"{nanoseconds / scale:,.1f} {name}" if scale > 1 else f"{nanoseconds:,.0f} ns"


def _range(fastest: float, slowest: float) -> str:
    scale, name = _unit(slowest)
    digits = 1 if scale > 1 else 0
    return f"{fastest / scale:,.{digits}f}-{slowest / scale:,.{digits}f} {name}"
