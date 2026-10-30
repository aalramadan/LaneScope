import functools
from collections.abc import Callable

from lanescope import __version__, analysis
from lanescope.fixes.layouts import Buffer, Change, Padding, XorSwizzle
from lanescope.trace import Trace

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError
except ImportError:  # The MCP SDK is an optional dependency.
    MCPServer = ToolError = None


def explain(trace_path: str) -> dict:
    """Return the shared wavefronts and global sectors of every instruction, worst first."""
    return analysis.explain(Trace.load(trace_path))


def lanes(trace_path: str, pc: str) -> str:
    """Return the lane-by-bank or lane-by-sector view of the worst access of an instruction."""
    return analysis.lanes(Trace.load(trace_path), int(pc, 0))


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
    return analysis.whatif(Trace.load(trace_path), Buffer(buffer_start, buffer_size), change)


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
    return analysis.suggest(Trace.load(trace_path), buffer, int(pc, 0) if pc else None, limit)


TOOLS = (explain, lanes, whatif, suggest)


def server():
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
    server().run("stdio")
