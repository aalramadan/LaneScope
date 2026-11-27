import argparse
import pathlib

from lanescope import demos, text_view, trace_reports
from lanescope.commands import layout_changes, trace_views
from lanescope.fixes import layouts
from lanescope.model.counters import count, worst_first


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare demo."""
    demo = commands.add_parser("demo", help="write and analyze the trace of a tiled transpose")
    demo.add_argument("--out", type=pathlib.Path, default=pathlib.Path("transpose.npz"))
    demo.set_defaults(run=_demo)


def _demo(args: argparse.Namespace) -> int:
    trace = demos.transpose_trace()
    trace.save(args.out)
    print(f"wrote {args.out}")
    print()
    trace_views.print_explanation(argparse.Namespace(trace=args.out, json=False))
    worst = worst_first(count(trace))[0]
    print()
    print(text_view.describe_row(trace, worst.worst_row))
    buffer = layouts.Buffer(0, demos.TILE * demos.TILE * demos.FLOAT_BYTES)
    row_bytes = demos.TILE * demos.FLOAT_BYTES
    for change in (layouts.Padding(row_bytes, demos.FLOAT_BYTES), layouts.XorSwizzle(5, 2, 5)):
        print()
        layout_changes.print_comparison(buffer, trace_reports.whatif(trace, buffer, change))
    return 0
