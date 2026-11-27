import argparse
import pathlib
import tempfile

from lanescope import demos
from lanescope.commands import argument_types
from lanescope.gui import server


def add_parsers(commands: argparse._SubParsersAction) -> None:
    """Declare mcp and gui."""
    agents = commands.add_parser(
        "mcp", help="serve the analyses to kernel agents over MCP on stdio"
    )
    agents.set_defaults(run=_mcp)

    window = commands.add_parser("gui", help="explore traces and try fixes in the browser")
    window.add_argument(
        "paths", nargs="*", type=pathlib.Path, help="trace files or folders, this folder by default"
    )
    window.add_argument(
        "--host", default="127.0.0.1", help="address to listen on, 0.0.0.0 inside a container"
    )
    window.add_argument(
        "--port", type=argument_types.port, default=server.DEFAULT_PORT, help="port to listen on"
    )
    window.add_argument(
        "--allow-host",
        action="append",
        default=[],
        metavar="NAME",
        help="another host name that the page may use, such as a machine name",
    )
    window.add_argument("--open", action="store_true", help="open the page in the browser")
    window.add_argument(
        "--demo", action="store_true", help="add five demo kernels and two recordings of llama.cpp"
    )
    window.set_defaults(run=_gui)


def _mcp(args: argparse.Namespace) -> int:
    # Only this command needs the MCP SDK, whose import takes about 0.4 s, and the import
    # therefore waits until the command runs.
    from lanescope import mcp_server

    mcp_server.serve()
    return 0


def _gui(args: argparse.Namespace) -> int:
    paths = list(args.paths)
    with tempfile.TemporaryDirectory(prefix="lanescope-demo-") as folder:
        if args.demo:
            demo = pathlib.Path(folder) / "demo"
            demo.mkdir()
            for name, trace in demos.gallery_traces().items():
                trace.save(demo / name)
            paths.append(demo)
        server.serve(
            paths or [pathlib.Path.cwd()],
            args.host,
            args.port,
            args.allow_host,
            args.open,
            processes=True,
        )
    return 0
