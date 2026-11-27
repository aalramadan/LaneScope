import argparse
import pathlib

from lanescope import html_report
from lanescope.commands import argument_types
from lanescope.trace import Trace


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare report."""
    report = commands.add_parser("report", help="write an HTML report with lane grids")
    report.add_argument("trace", type=pathlib.Path)
    report.add_argument("-o", "--out", type=pathlib.Path, required=True, help="HTML file")
    report.add_argument(
        "--limit",
        type=argument_types.nonnegative_count,
        default=20,
        help="most instructions to draw",
    )
    report.set_defaults(run=_report)


def _report(args: argparse.Namespace) -> int:
    args.out.write_text(
        html_report.render_trace_report(Trace.load(args.trace), args.limit), encoding="utf-8"
    )
    print(f"wrote {args.out}")
    return 0
