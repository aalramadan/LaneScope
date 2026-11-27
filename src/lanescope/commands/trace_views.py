import argparse
import json
import pathlib
import sys

from lanescope import summary, summary_text, text_view, trace_reports
from lanescope.commands import argument_types
from lanescope.trace import Trace

TABLE_HEADER = (
    f"{'pc':8}{'line':>5}  {'space':7}{'kind':6}{'width':>5}{'requests':>10}  {'unit':11}"
    f"{'achieved':>9}{'ideal':>7}{'excessive':>10}{'of kernel':>11}"
)


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare explain, summary, and lanes."""
    explain = commands.add_parser(
        "explain", help="counters of every memory instruction, worst first"
    )
    explain.add_argument("trace", type=pathlib.Path)
    explain.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    explain.set_defaults(run=print_explanation)

    summing = commands.add_parser(
        "summary", help="extra passes of every kernel and line of a recording, worst first"
    )
    summing.add_argument(
        "paths", nargs="+", type=pathlib.Path, help="recording folders or trace files"
    )
    summing.add_argument(
        "--global",
        dest="global_memory",
        action="store_true",
        help="rank the extra sectors of global memory in place of the passes of shared memory",
    )
    summing.add_argument(
        "--limit", type=argument_types.size, default=10, help="most kernels and lines to list"
    )
    summing.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    summing.set_defaults(run=_summary)

    lanes = commands.add_parser("lanes", help="lane-by-lane view of one access of an instruction")
    lanes.add_argument("trace", type=pathlib.Path)
    lanes.add_argument(
        "--pc",
        type=argument_types.number,
        required=True,
        help="PC of the instruction, such as 0x120",
    )
    lanes.add_argument("--row", type=int, help="trace row to show instead of the worst access")
    lanes.set_defaults(run=_lanes)


def print_explanation(args: argparse.Namespace) -> int:
    """Print the counters of every memory instruction of a trace, as explain does."""
    trace = Trace.load(args.trace)
    data = trace_reports.explain(trace)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    kernel = text_view.printable(data["name"] or "?")
    arch = text_view.printable(trace.metadata.get("arch", "?"))
    print(f"kernel {kernel} ({arch}), {len(trace)} warp-level accesses")
    print()
    print(TABLE_HEADER)
    achieved: dict[str, int] = {}
    for record in data["instructions"]:
        achieved[record["unit"]] = achieved.get(record["unit"], 0) + record["achieved"]
    for record in data["instructions"]:
        share = summary_text.percent(record["excessive"], achieved[record["unit"]])
        mark = f" {summary_text.MARK}" if record["data_dependent"] else ""
        print(
            f"{int(record['pc'], 16):#06x}  {record['line'] or '-':>5}  {record['space']:7}"
            f"{record['kind']:6}{record['width']:5d}{record['requests']:10d}  {record['unit']:11}"
            f"{record['achieved']:9d}{record['ideal']:7d}{record['excessive']:10d}{share:>11}{mark}"
        )
    if any(record["data_dependent"] for record in data["instructions"]):
        print()
        print(f"{summary_text.MARK} {text_view.DATA_DEPENDENT}")
    return 0


def _summary(args: argparse.Namespace) -> int:
    paths: list[pathlib.Path] = []
    for root in args.paths:
        if not root.exists():
            raise ValueError(f"no such file or folder: {root}")
        paths += summary.trace_files(root, summary.MAX_TRACES - len(paths))
    if not paths:
        raise ValueError(f"found no trace files in {', '.join(str(root) for root in args.paths)}")
    facts = _read_facts(paths)
    data = summary.summarize(
        (str(path), str(summary.run_of(path)), found)
        for path, found in zip(paths, facts, strict=True)
    )
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        unit = "sectors" if args.global_memory else "wavefronts"
        print(summary_text.render(data, unit, args.limit))
    for note in summary_text.notes(data):
        print(f"lanescope: {note}", file=sys.stderr)
    if len(paths) == summary.MAX_TRACES:
        limit = f"{summary.MAX_TRACES:,}"
        print(f"lanescope: summed only the first {limit} trace files", file=sys.stderr)
    return 0


def _read_facts(paths: list[pathlib.Path]) -> list[summary.TraceFacts | None]:
    """Return the facts of each trace, with the progress on a terminal for a large recording."""
    found = []
    shown = sys.stderr.isatty() and len(paths) > summary.PARALLEL_TRACES
    for index, facts in enumerate(summary.read_many(paths, processes=True), 1):
        found.append(facts)
        if shown:
            print(f"\rsumming launch {index:,} of {len(paths):,}", end="", file=sys.stderr)
    if shown:
        print("\r\033[K", end="", file=sys.stderr)
    return found


def _lanes(args: argparse.Namespace) -> int:
    print(trace_reports.lanes(Trace.load(args.trace), args.pc, args.row))
    return 0
