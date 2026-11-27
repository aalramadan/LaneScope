import argparse
import json
import pathlib
import sys

from lanescope import kernel_times, recording_comparison
from lanescope.commands import argument_types
from lanescope.summary_text import UNITS, duration, duration_range, unreadable_times_note
from lanescope.text_view import printable

COUNTS_HEADER = "{:>9}  {:>14}  {:>14}  {:>13}  {:>13}  kernel"
TIMES_HEADER = "{:>11}  {:>17}  {:>11}  {:>17}  {:>13}  kernel"
ALONE_HEADER = "{:>11}  {:>9}  {:>13}  {:>11}  kernel"
WITHIN_SPREAD = "within spread"
# How a row names the instances it adds up, by its matched_instances.
INSTANCE_MARKS = {"all": "", "some": " (shared instances)", "none": " (instances differ)"}
INSTANCE_NOTES = {
    "some": "shared instances: the row adds up only the instances of the template that both "
    "sides hold. The others are listed under the kernels of one side alone.",
    "none": "instances differ: the two sides hold different instances of the template, such as "
    "after a change of a template argument, and the row adds up all of them.",
}


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare compare."""
    comparing = commands.add_parser(
        "compare",
        help="extra passes and kernel times of two recordings, such as before and after a fix",
    )
    comparing.add_argument(
        "before", type=pathlib.Path, help="recording folder or trace file before the change"
    )
    comparing.add_argument(
        "after", type=pathlib.Path, help="recording folder or trace file after the change"
    )
    comparing.add_argument(
        "--global",
        dest="global_memory",
        action="store_true",
        help="compare the extra sectors of global memory in place of the passes of shared memory",
    )
    comparing.add_argument(
        "--limit", type=argument_types.positive_count, default=10, help="most kernels to list"
    )
    comparing.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    comparing.set_defaults(run=_compare)


def _compare(args: argparse.Namespace) -> int:
    before = recording_comparison.read_side(args.before, processes=True)
    after = recording_comparison.read_side(args.after, processes=True)
    data = recording_comparison.compare_summaries(before, after)
    if args.json:
        print(json.dumps({"before": str(args.before), "after": str(args.after)} | data, indent=2))
    else:
        unit = "sectors" if args.global_memory else "wavefronts"
        print(render_comparison(data, args.before, args.after, unit, args.limit))
    for unread in data["unreadable_times"]:
        print(f"lanescope: {unreadable_times_note(unread)}", file=sys.stderr)
    return 0


def render_comparison(
    data: dict,
    before: pathlib.Path,
    after: pathlib.Path,
    unit: str = "wavefronts",
    limit: int = 10,
) -> str:
    """Return a comparison as text: the totals, the counts and the times of each kernel."""
    memory, noun, cause = UNITS[unit]
    timed = data["timed"]
    out = [
        f"before: {printable(before)}{_rounds_suffix(timed['before'])}",
        f"after: {printable(after)}{_rounds_suffix(timed['after'])}",
    ]
    total = data["total"]
    first, second = total["before"][unit], total["after"][unit]
    out.append(
        f"{memory}: {first['achieved']:,} {noun} before and {second['achieved']:,} after. "
        f"{cause} add {first['excessive']:,} before and {second['excessive']:,} after."
    )
    if total["time"]["speedup"] is not None:
        out.append(f"Kernel time: {_time_change_words(total)}")
    counted = [
        row
        for row in data["kernels"]
        if row["before"][unit]["achieved"] or row["after"][unit]["achieved"]
    ]
    rows = sorted(
        counted,
        key=lambda row: (-row["before"][unit]["excessive"], -row["after"][unit]["excessive"]),
    )
    out += ["", f"Extra {noun} by kernel"]
    out.append(
        COUNTS_HEADER.format(
            "launches", f"{noun} before", f"{noun} after", "extra before", "extra after"
        )
    )
    for row in rows[:limit]:
        out.append(_counts_row(row, unit, _row_name(row)))
        out += [_counts_row(found, unit, f"  {found['name']}") for found in row["instances"]]
    out += _hidden_kernels_line(len(rows) - limit)
    out.append(_counts_row(total, unit, "all kernels"))
    out += _instance_notes(rows[:limit])
    if timed["before"] and timed["after"]:
        out += _times_table(data, limit)
    out += _one_sided_kernels(data, unit, noun)
    untimed = [
        (side, recording_comparison.times_folder(path))
        for side, path in (("before", before), ("after", after))
        if not timed[side]
    ]
    if untimed:
        out += ["", *_timing_steps(untimed, before, after)]
    return "\n".join(out)


def _timing_steps(
    untimed: list[tuple[str, pathlib.Path]], before: pathlib.Path, after: pathlib.Path
) -> list[str]:
    """Return the next step that gives each untimed side its kernel times.

    lanescope time runs the stored command of a folder, which a folder without run.json
    lacks, and times both sides in turns only when each of them has one.
    """
    folders = [recording_comparison.times_folder(path) for path in (before, after)]
    if len(untimed) == 2 and all(_has_command(folder) for folder in folders):
        shown = " ".join(printable(folder) for folder in folders)
        return [
            f"Next step: lanescope time {shown} times the kernels of both sides in turns, "
            "and lanescope compare then gives their times."
        ]
    steps = []
    for side, folder in untimed:
        shown = printable(folder)
        if _has_command(folder):
            steps.append(
                f"Next step for the {side} side: lanescope time {shown} times its kernels with "
                "the command of its recording, and lanescope compare then gives their times."
            )
        else:
            steps.append(
                f"Next step for the {side} side: {shown} holds no {kernel_times.RUN_FILE} with "
                f"the command of its recording. lanescope time {shown} -- ./my_app times the "
                "kernels of the program given after --, and a new recording with lanescope "
                f"record --out {shown} keeps its command for lanescope time."
            )
    return steps


def _has_command(folder: pathlib.Path) -> bool:
    return (folder / kernel_times.RUN_FILE).is_file()


def _row_name(row: dict) -> str:
    return row["name"] + INSTANCE_MARKS[row["matched_instances"]]


def _instance_notes(rows: list[dict]) -> list[str]:
    """Return the sentences that explain the marks of rows over some or differing instances."""
    found = {row["matched_instances"] for row in rows}
    notes = [INSTANCE_NOTES[match] for match in ("some", "none") if match in found]
    return ["", *notes] if notes else []


def _times_table(data: dict, limit: int) -> list[str]:
    rows = sorted(data["kernels"], key=lambda row: -(row["before"]["time_ns"] or 0))
    out = ["", "Kernel time by kernel"]
    out.append(
        TIMES_HEADER.format(
            "time before", "range over rounds", "time after", "range over rounds", "speedup"
        )
    )
    for row in rows[:limit]:
        out.append(_times_row(row, _row_name(row)))
        out += [_times_row(found, f"  {found['name']}") for found in row["instances"]]
    out += _hidden_kernels_line(len(rows) - limit)
    out.append(_times_row(data["total"], "all kernels"))
    if any(row["time"]["within_spread"] for row in [*rows[:limit], data["total"]]):
        out += [
            "",
            f"{WITHIN_SPREAD}: the rounds of the two sides lie apart by less than the larger "
            "of their spreads, the slowest round of a side minus its fastest. The difference "
            "can then come from the noise of the rounds, and it counts as neither a speedup "
            "nor a slowdown.",
        ]
    return out


def _one_sided_kernels(data: dict, unit: str, noun: str) -> list[str]:
    alone = [("before only", entry) for entry in data["only_before"]]
    alone += [("after only", entry) for entry in data["only_after"]]
    if not alone:
        return []
    out = [
        "",
        "Kernels of one side alone",
        ALONE_HEADER.format("side", "launches", f"extra {noun}", "time"),
    ]
    for side, entry in alone:
        spent = "-" if entry["time_ns"] is None else duration(entry["time_ns"])
        out.append(
            f"{side:>11}  {entry['launches']:>9,}  {entry[unit]['excessive']:>13,}  {spent:>11}  "
            f"{printable(entry['name'])}"
        )
    return out


def _counts_row(row: dict, unit: str, name: str) -> str:
    first, second = row["before"], row["after"]
    launches = (
        f"{first['launches']:,}"
        if first["launches"] == second["launches"]
        else f"{first['launches']:,}/{second['launches']:,}"
    )
    return (
        f"{launches:>9}  {first[unit]['achieved']:>14,}  {second[unit]['achieved']:>14,}  "
        f"{first[unit]['excessive']:>13,}  {second[unit]['excessive']:>13,}  {printable(name)}"
    )


def _times_row(row: dict, name: str) -> str:
    first, second = row["before"], row["after"]
    return (
        f"{_time_text(first):>11}  {_range_text(first):>17}  "
        f"{_time_text(second):>11}  {_range_text(second):>17}  "
        f"{_speedup(row['time']):>13}  {printable(name)}"
    )


def _time_change_words(row: dict) -> str:
    """Return the change of the time of a row in words."""
    first, second = row["before"]["time_ns"], row["after"]["time_ns"]
    times = f"{duration(first)} before and {duration(second)} after"
    if row["time"]["within_spread"]:
        return f"{times}. The difference lies within the spread of the rounds."
    return f"{times}, a speedup of {row['time']['speedup']:.2f}x."


def _speedup(change: dict) -> str:
    if change["speedup"] is None:
        return "-"
    return WITHIN_SPREAD if change["within_spread"] else f"{change['speedup']:.2f}x"


def _time_text(side: dict) -> str:
    return "-" if side["time_ns"] is None else duration(side["time_ns"])


def _range_text(side: dict) -> str:
    if side["time_range_ns"] is None:
        return "-"
    fastest, slowest = side["time_range_ns"]
    return duration_range(fastest, slowest, side["time_ns"])


def _rounds_suffix(rounds: int | None) -> str:
    if rounds is None:
        return ", no kernel times"
    return f", kernel times of {rounds} round{'' if rounds == 1 else 's'}"


def _hidden_kernels_line(count: int) -> list[str]:
    return [f"{'':>9}  and {count:,} more kernel{'' if count == 1 else 's'}"] if count > 0 else []
