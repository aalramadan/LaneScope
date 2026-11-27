import argparse
import json
import pathlib

from lanescope import recording_comparison
from lanescope.commands import argument_types
from lanescope.summary_text import UNITS, duration, duration_range
from lanescope.text_view import printable

COUNTS_HEADER = "{:>9}  {:>14}  {:>14}  {:>13}  {:>13}  kernel"
TIMES_HEADER = "{:>11}  {:>17}  {:>11}  {:>17}  {:>13}  kernel"
ALONE_HEADER = "{:>11}  {:>9}  {:>13}  {:>11}  kernel"
WITHIN_SPREAD = "within spread"


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
        "--limit", type=argument_types.size, default=10, help="most kernels to list"
    )
    comparing.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    comparing.set_defaults(run=_compare)


def _compare(args: argparse.Namespace) -> int:
    before = recording_comparison.read_side(args.before, processes=True)
    after = recording_comparison.read_side(args.after, processes=True)
    data = recording_comparison.compare(before, after)
    if args.json:
        print(json.dumps({"before": str(args.before), "after": str(args.after)} | data, indent=2))
        return 0
    unit = "sectors" if args.global_memory else "wavefronts"
    print(render(data, args.before, args.after, unit, args.limit))
    return 0


def render(
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
        f"before: {printable(before)}{_timing(timed['before'])}",
        f"after: {printable(after)}{_timing(timed['after'])}",
    ]
    total = data["total"]
    first, second = total["before"][unit], total["after"][unit]
    out.append(
        f"{memory}: {first['achieved']:,} {noun} before and {second['achieved']:,} after. "
        f"{cause} add {first['excessive']:,} before and {second['excessive']:,} after."
    )
    if total["time"]["speedup"] is not None:
        out.append(f"Kernel time: {_change(total)}")
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
        out.append(_counts_row(row, unit, row["name"]))
        out += [_counts_row(found, unit, f"  {found['name']}") for found in row["instances"]]
    out += _more(len(rows) - limit)
    out.append(_counts_row(total, unit, "all kernels"))
    if timed["before"] and timed["after"]:
        out += _times_table(data, limit)
    out += _alone(data, unit, noun)
    if not (timed["before"] and timed["after"]):
        folders = " ".join(
            printable(recording_comparison.times_folder(path)) for path in (before, after)
        )
        out += [
            "",
            f"Next step: lanescope time {folders} times the kernels of both sides in turns, "
            "and lanescope compare then gives their times.",
        ]
    return "\n".join(out)


def _times_table(data: dict, limit: int) -> list[str]:
    rows = sorted(data["kernels"], key=lambda row: -(row["before"]["time_ns"] or 0))
    out = ["", "Kernel time by kernel"]
    out.append(
        TIMES_HEADER.format(
            "time before", "range over rounds", "time after", "range over rounds", "speedup"
        )
    )
    for row in rows[:limit]:
        out.append(_times_row(row, row["name"]))
        out += [_times_row(found, f"  {found['name']}") for found in row["instances"]]
    out += _more(len(rows) - limit)
    out.append(_times_row(data["total"], "all kernels"))
    if any(row["time"]["within_spread"] for row in [*rows[:limit], data["total"]]):
        out += [
            "",
            f"{WITHIN_SPREAD}: the times of the rounds before and after overlap, therefore the "
            "difference lies within the spread of the rounds.",
        ]
    return out


def _alone(data: dict, unit: str, noun: str) -> list[str]:
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
        f"{_time(first):>11}  {_range(first):>17}  {_time(second):>11}  {_range(second):>17}  "
        f"{_speedup(row['time']):>13}  {printable(name)}"
    )


def _change(row: dict) -> str:
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


def _time(side: dict) -> str:
    return "-" if side["time_ns"] is None else duration(side["time_ns"])


def _range(side: dict) -> str:
    return "-" if side["time_range_ns"] is None else duration_range(*side["time_range_ns"])


def _timing(rounds: int | None) -> str:
    if rounds is None:
        return ", no kernel times"
    return f", kernel times of {rounds} round{'' if rounds == 1 else 's'}"


def _more(count: int) -> list[str]:
    return [f"{'':>9}  and {count:,} more kernel{'' if count == 1 else 's'}"] if count > 0 else []
