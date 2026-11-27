import argparse
import json
import pathlib

from lanescope import text_view, trace_reports
from lanescope.commands import argument_types
from lanescope.fixes import layouts
from lanescope.trace import Trace


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare whatif and suggest."""
    change = commands.add_parser(
        "whatif", help="counters after a layout change of one shared buffer"
    )
    change.add_argument("trace", type=pathlib.Path)
    change.add_argument(
        "--buffer", type=argument_types.buffer, required=True, help="START:SIZE in bytes"
    )
    kinds = change.add_mutually_exclusive_group(required=True)
    kinds.add_argument("--pad", type=argument_types.padding, help="ROW_BYTES:PAD_BYTES")
    kinds.add_argument(
        "--swizzle", type=argument_types.swizzle, help="BITS,BASE,SHIFT on byte offsets"
    )
    change.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    change.set_defaults(run=_whatif)

    search = commands.add_parser(
        "suggest", help="padding and swizzle changes ranked by predicted excess"
    )
    search.add_argument("trace", type=pathlib.Path)
    search.add_argument(
        "--pc",
        type=argument_types.number,
        help="instruction whose buffer to search, the worst by default",
    )
    search.add_argument(
        "--buffer", type=argument_types.buffer, help="START:SIZE in bytes, inferred by default"
    )
    search.add_argument(
        "--limit", type=argument_types.count, default=5, help="most changes to list"
    )
    search.add_argument("--json", action="store_true", help="print JSON for tools and agents")
    search.set_defaults(run=_suggest)


def print_comparison(buffer: layouts.Buffer, data: dict) -> None:
    """Print the counters before and after a layout change, as whatif does."""
    print(
        f"{data['change']}, buffer {buffer.start:#x}+{buffer.size} "
        f"grows by {data['extra_bytes']} bytes"
    )
    if data["later_shift"]:
        print(f"shared addresses above the buffer move up by {data['later_shift']} bytes")
    print(
        f"{'pc':8}{'line':>5}  {'space':7}{'kind':7}{'unit':11}{'before':>8}{'after':>8}"
        f"{'ideal':>7}{'excessive before':>18}{'after':>7}"
    )
    for item in data["instructions"]:
        old, new = item["before"], item["after"]
        print(
            f"{int(old['pc'], 16):#06x}  {old['line'] or '-':>5}  {old['space']:7}{old['kind']:7}"
            f"{old['unit']:11}{old['achieved']:8d}{new['achieved']:8d}{old['ideal']:7d}"
            f"{old['excessive']:18d}{new['excessive']:7d}"
        )


def _whatif(args: argparse.Namespace) -> int:
    data = trace_reports.whatif(Trace.load(args.trace), args.buffer, args.pad or args.swizzle)
    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print_comparison(args.buffer, data)
    return 0


def _suggest(args: argparse.Namespace) -> int:
    data = trace_reports.suggest(Trace.load(args.trace), args.buffer, args.pc, args.limit)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0
    print(f"buffer {data['buffer']['start']:#x}+{data['buffer']['size']}")
    if data["found"] == 0:
        print("no padding or swizzle lowers the excess wavefronts of this buffer")
        if data["data_dependent"]:
            print(text_view.DATA_DEPENDENT)
    elif data["data_dependent"]:
        print(text_view.DATA_DEPENDENT_FIX)
    for item in data["changes"]:
        before, after = item["excessive_sectors_before"], item["excessive_sectors_after"]
        sectors = f", excessive global sectors {before} -> {after}" if before != after else ""
        print(
            f"{item['change']}: excessive wavefronts {item['excessive_before']} -> "
            f"{item['excessive_after']}{sectors}, {item['extra_bytes']} extra bytes"
        )
    return 0
