import argparse
import sys
from collections.abc import Sequence

from lanescope.commands import (
    calibration_bench,
    layout_changes,
    nsight_check,
    record_program,
    servers,
    time_program,
    trace_views,
    transpose_demo,
    write_html_report,
)
from lanescope.cuda.driver import DriverError
from lanescope.cuda.nvrtc import CompileError
from lanescope.cuda.toolchain import ToolchainError
from lanescope.nsight.validate import ValidationError
from lanescope.recorder.build import BuildError

ERRORS = (
    ValueError, OSError, ImportError, BuildError, CompileError, DriverError, ToolchainError,
    ValidationError,
)  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return its exit status."""
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except ERRORS as error:
        print(f"lanescope: {error}", file=sys.stderr)
        return 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lanescope",
        description="Explain the shared and global memory counters of a CUDA kernel trace.",
    )
    commands = parser.add_subparsers(required=True, metavar="command")
    # The order of these calls is the order of the commands in --help.
    trace_views.add_parsers(commands)
    layout_changes.add_parsers(commands)
    write_html_report.add_parsers(commands)
    transpose_demo.add_parsers(commands)
    record_program.add_parsers(commands)
    time_program.add_parsers(commands)
    calibration_bench.add_parsers(commands)
    nsight_check.add_parsers(commands)
    servers.add_parsers(commands)
    return parser
