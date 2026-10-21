import asyncio
import json

import pytest

import lanescope
from lanescope import demos, mcp_server

try:
    from mcp.client.client import Client
    from mcp.types import TextContent
except ImportError:  # The MCP SDK is an optional dependency.
    Client = TextContent = None

pytestmark = pytest.mark.skipif(mcp_server.MCPServer is None, reason="the MCP SDK is needed")


def call(name, arguments):
    assert Client is not None
    client_type = Client

    async def run():
        async with client_type(mcp_server.server()) as client:
            tools = await client.list_tools()
            result = await client.call_tool(name, arguments)
            return [tool.name for tool in tools.tools], result

    return asyncio.run(run())


def test_server_lists_the_four_tools(tmp_path):
    path = tmp_path / "t.npz"
    demos.transpose_trace().save(path)
    names, _ = call("explain", {"trace_path": str(path)})
    assert sorted(names) == ["explain", "lanes", "suggest", "whatif"]


def test_explain_tool_returns_the_counters(tmp_path):
    path = tmp_path / "t.npz"
    demos.transpose_trace().save(path)
    _, result = call("explain", {"trace_path": str(path)})
    content = result.content[0]
    assert TextContent is not None and isinstance(content, TextContent)
    data = json.loads(content.text)
    assert data["instructions"][0]["excessive"] == 3968


def test_whatif_tool_needs_a_change():
    with pytest.raises(ValueError, match="pad_row_bytes"):
        mcp_server.whatif("unused.npz", 0, 4096)


def test_tool_errors_reach_the_agent_with_their_reason(tmp_path):
    path = tmp_path / "t.npz"
    demos.transpose_trace().save(path)
    _, result = call("lanes", {"trace_path": str(path), "pc": "0x999"})
    content = result.content[0]
    assert TextContent is not None and isinstance(content, TextContent)
    assert result.is_error
    assert "no access at pc 0x999" in content.text


def test_suggest_tool_takes_a_known_buffer(tmp_path):
    path = tmp_path / "t.npz"
    demos.transpose_trace().save(path)
    data = mcp_server.suggest(str(path), buffer_start=0, buffer_size=4096, limit=1)
    assert data["buffer"] == {"start": 0, "size": 4096}


def test_whatif_tool_needs_the_swizzle_base():
    with pytest.raises(ValueError, match="swizzle_base"):
        mcp_server.whatif("unused.npz", 0, 4096, swizzle_bits=5, swizzle_shift=5)


def test_whatif_tool_rejects_a_padding_and_a_swizzle_together():
    with pytest.raises(ValueError, match="not both"):
        mcp_server.whatif("unused.npz", 0, 4096, pad_row_bytes=128, pad_bytes=4, swizzle_bits=5)


def test_whatif_tool_needs_the_pad_of_a_padding():
    with pytest.raises(ValueError, match="pad_bytes above zero"):
        mcp_server.whatif("unused.npz", 0, 4096, pad_row_bytes=128)


def test_suggest_tool_needs_the_size_of_a_buffer_with_a_start():
    with pytest.raises(ValueError, match="buffer_size"):
        mcp_server.suggest("unused.npz", buffer_start=4096)


def test_suggest_tool_rejects_a_negative_limit(tmp_path):
    path = tmp_path / "t.npz"
    demos.transpose_trace().save(path)
    _, result = call("suggest", {"trace_path": str(path), "limit": -1})
    content = result.content[0]
    assert TextContent is not None and isinstance(content, TextContent)
    assert result.is_error
    assert "limit" in content.text


def test_server_reports_the_version_of_lanescope():
    assert Client is not None
    client_type = Client

    async def run():
        async with client_type(mcp_server.server()) as client:
            return client.server_info

    info = asyncio.run(run())
    assert info is not None and info.version == lanescope.__version__
