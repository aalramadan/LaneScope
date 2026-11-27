import functools
import pathlib
from collections.abc import Callable

from lanescope import __version__, recording_comparison, trace_reports
from lanescope.fixes.layouts import Buffer, Change, Padding, XorSwizzle
from lanescope.summary import MAX_TRACES, UNITS, read_many, summarize_files, trace_files
from lanescope.trace import Trace

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # The MCP SDK is an optional dependency.
    MCPServer = ToolError = None


def explain(trace_path: str) -> dict:
    """Return the shared wavefronts and global sectors of every instruction, worst first."""
    return trace_reports.explain(Trace.load(trace_path))


def lanes(trace_path: str, pc: str) -> str:
    """Return the lane-by-bank or lane-by-sector view of the worst access of an instruction."""
    return trace_reports.lanes(Trace.load(trace_path), int(pc, 0))


def whatif(
    trace_path: str,
    buffer_start: int,
    buffer_size: int,
    pad_row_bytes: int = 0,
    pad_bytes: int = 0,
    swizzle_bits: int = 0,
    swizzle_base: int = -1,
    swizzle_shift: int = 0,
) -> dict:
    """Predict the counters after padding each row of a shared buffer or applying an XOR swizzle.

    Give pad_row_bytes and pad_bytes for a padding, or swizzle_bits, swizzle_base, and
    swizzle_shift for the CuTe Swizzle<bits, base, shift> on byte offsets, never both.
    """
    change: Change
    if pad_row_bytes and swizzle_bits:
        raise ValueError("give a padding or a swizzle, not both")
    if pad_row_bytes:
        # A missing pad_bytes takes its default of zero, a padding that changes nothing.
        if pad_bytes <= 0:
            raise ValueError("a padding needs pad_bytes above zero, such as 4")
        change = Padding(pad_row_bytes, pad_bytes)
    elif swizzle_bits:
        if swizzle_base < 0:
            raise ValueError("a swizzle needs swizzle_base, such as 2 for 4-byte accesses")
        change = XorSwizzle(swizzle_bits, swizzle_base, swizzle_shift)
    else:
        raise ValueError("give pad_row_bytes for a padding or swizzle_bits for a swizzle")
    return trace_reports.whatif(Trace.load(trace_path), Buffer(buffer_start, buffer_size), change)


def suggest(
    trace_path: str, pc: str = "", buffer_start: int = 0, buffer_size: int = 0, limit: int = 5
) -> dict:
    """Rank paddings and swizzles of a shared buffer.

    Each change gives the excess wavefronts of the buffer and the excess global sectors of
    the cp.async copies that fill it, before and after the change. Give buffer_start and
    buffer_size for a known buffer. Otherwise the buffer is the
    range that the instruction at pc touches, the worst shared instruction by default.
    """
    if buffer_start and not buffer_size:
        raise ValueError("a buffer needs both buffer_start and buffer_size")
    buffer = Buffer(buffer_start, buffer_size) if buffer_size else None
    return trace_reports.suggest(Trace.load(trace_path), buffer, int(pc, 0) if pc else None, limit)


def summary(recording_path: str, limit: int = 20) -> dict:
    """Rank the kernel templates and source lines of a recording by extra wavefronts and sectors.

    Give the folder of a recording or a trace file. Each entry of kernels is one template,
    such as cub::DeviceMergeSortBlockSortKernel, and lists its kernel instances in
    instances. A source line adds up over the instances of its template. The answer keeps
    the worst lines of each unit up to the limit. Each line names the trace and the PC that
    the other tools take, and says whether its conflicts follow the input values over all
    launches, which no layout removes. A line inlined from another file, such as a header,
    keys on its origin there, with the call line in file, line, and source. After
    lanescope time, each kernel gives time_ns, its median time over the rounds, and times
    gives the kernel time of the run.
    """
    root = pathlib.Path(recording_path)
    if not root.exists():
        raise ValueError(f"no such file or folder: {recording_path}")
    paths = trace_files(root, MAX_TRACES)
    if not paths:
        raise ValueError(f"found no trace files in {recording_path}")
    data = summarize_files(paths, read_many(paths))
    kept = [
        line for unit in UNITS for line in [x for x in data["lines"] if x["unit"] == unit][:limit]
    ]
    return data | {"lines": kept}


def compare(before_path: str, after_path: str) -> dict:
    """Compare two recordings, such as before and after a fix, by kernel template.

    Give a recording folder or a trace file for each side. Each row of kernels gives the
    launches, the wavefronts and sectors (achieved, ideal, excessive), and the median
    time_ns over the rounds of lanescope time of each side, with time_range_ns over the
    rounds and the speedup, before over after. within_spread is true where the ranges of
    the rounds overlap, which makes no speedup claim. only_before and only_after list the
    templates of one side alone, and total adds up every kernel of each side. timed gives
    the rounds of each side, or null for a side that lanescope time has not timed.
    """
    before = recording_comparison.read_side(pathlib.Path(before_path))
    after = recording_comparison.read_side(pathlib.Path(after_path))
    return recording_comparison.compare_summaries(before, after)


TOOLS = (explain, lanes, whatif, suggest, summary, compare)


def build_mcp_server():
    """Return an MCP server with the LaneScope tools."""
    if MCPServer is None:
        raise ImportError("the MCP server needs pip install 'lanescope[mcp]'")
    mcp = MCPServer(
        "lanescope",
        version=__version__,
        instructions="Lane-level analyses of CUDA memory counters.",
    )
    for tool in TOOLS:
        mcp.tool()(_reported(tool))
    return mcp


def _reported(tool: Callable) -> Callable:
    # The SDK hides the text of unexpected exceptions, while an agent needs the reason
    # to correct its call. A ToolError reaches the agent with its message.
    @functools.wraps(tool)
    def call(*args, **kwargs):
        try:
            return tool(*args, **kwargs)
        except (ValueError, OSError) as error:
            assert ToolError is not None
            raise ToolError(str(error)) from error

    return call


def serve() -> None:
    """Serve the tools over stdio."""
    build_mcp_server().run("stdio")
