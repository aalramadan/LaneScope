import asyncio
import json
import os
import pathlib
import subprocess
import sys
import threading

import pytest

import lanescope
from lanescope import demos, mcp_server

try:
    from mcp.client.client import Client
    from mcp.types import TextContent
except ImportError:  # The MCP SDK is an optional dependency.
    Client = TextContent = None

SOURCE = pathlib.Path(__file__).parent.parent / "src"
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


def test_server_lists_the_six_tools(tmp_path):
    path = tmp_path / "t.npz"
    demos.transpose_trace().save(path)
    names, _ = call("explain", {"trace_path": str(path)})
    assert sorted(names) == ["compare", "explain", "lanes", "suggest", "summary", "whatif"]


def test_summary_tool_ranks_the_lines_of_a_recording_and_names_their_trace(tmp_path):
    demos.transpose_trace().save(tmp_path / "trace-0.npz")
    demos.row_sums_trace().save(tmp_path / "trace-1.npz")
    _, result = call("summary", {"recording_path": str(tmp_path), "limit": 1})
    content = result.content[0]
    assert TextContent is not None and isinstance(content, TextContent)
    data = json.loads(content.text)
    assert data["traces"] == 2
    shared = [line for line in data["lines"] if line["unit"] == "wavefronts"]
    assert len(shared) == 1
    assert shared[0]["worst"]["id"] == str(tmp_path / "trace-0.npz")
    assert shared[0]["excessive"] == 3968


def test_summary_tool_gives_one_entry_per_template_with_its_instances(tmp_path):
    for index, kernel in enumerate(["void ns::tile<4>(float*)", "void ns::tile<8>(float*)"]):
        trace = demos.transpose_trace()
        trace.metadata["kernel"] = kernel
        trace.save(tmp_path / f"trace-{index}.npz")
    (entry,) = mcp_server.summary(str(tmp_path))["kernels"]
    assert entry["template"] == "ns::tile"
    assert sorted(found["name"] for found in entry["instances"]) == ["tile<4>", "tile<8>"]


def test_summary_tool_gives_the_kernel_times_stored_with_the_recording(tmp_path):
    demos.transpose_trace().save(tmp_path / "trace-0.npz")
    times = {"launches": [1, 1], "nanoseconds": [800, 1_000], "shortest": 800, "longest": 1_000}
    result = {"format": 1, "rounds": 2, "kernels": {"transpose": times}}
    (tmp_path / "kernel-times.json").write_text(json.dumps(result))
    data = mcp_server.summary(str(tmp_path))
    assert data["kernels"][0]["time_ns"] == 900
    assert data["times"][0]["total_ns"] == 900


def test_compare_tool_gives_the_counts_of_each_template_before_and_after(tmp_path):
    for name in ("before", "after"):
        (tmp_path / name).mkdir()
        demos.transpose_trace().save(tmp_path / name / "trace-0.npz")
    demos.histogram_trace().save(tmp_path / "after" / "trace-1.npz")
    _, result = call(
        "compare",
        {"before_path": str(tmp_path / "before"), "after_path": str(tmp_path / "after")},
    )
    content = result.content[0]
    assert TextContent is not None and isinstance(content, TextContent)
    data = json.loads(content.text)
    (row,) = data["kernels"]
    assert row["before"]["wavefronts"] == row["after"]["wavefronts"]
    assert [entry["name"] for entry in data["only_after"]] == ["histogram"]
    assert data["timed"] == {"before": None, "after": None}


def test_summary_tool_names_a_missing_folder():
    with pytest.raises(ValueError, match="no such file or folder"):
        mcp_server.summary("/no/such/recording")


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


def test_lanescope_mcp_serves_the_tools_over_standard_input_and_output():
    client = {"name": "test", "version": "1"}
    hello = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": client}
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": hello},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    process = subprocess.Popen(
        [sys.executable, "-m", "lanescope", "mcp"],
        env=dict(os.environ, PYTHONPATH=str(SOURCE)),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None and process.stdout is not None
    # The server drops the requests still open when its input ends, so the input stays open
    # until the last reply arrives.
    watchdog = threading.Timer(120, process.kill)
    watchdog.start()
    replies = {}
    try:
        for message in messages:
            process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()
        while 2 not in replies and (line := process.stdout.readline()):
            reply = json.loads(line)
            replies[reply["id"]] = reply["result"]
    finally:
        process.stdin.close()
        process.wait()
        watchdog.cancel()
    assert process.returncode == 0
    assert replies[1]["serverInfo"]["name"] == "lanescope"
    assert sorted(tool["name"] for tool in replies[2]["tools"]) == [
        "compare",
        "explain",
        "lanes",
        "suggest",
        "summary",
        "whatif",
    ]
