import collections
import dataclasses
import http.server
import json
import os
import pathlib
import re
import signal
import socket
import sys
import threading
import traceback
import urllib.parse
import webbrowser
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from lanescope import __version__, analysis
from lanescope.fixes import codegen
from lanescope.fixes.layouts import Buffer, apply, infer_buffer, lanes_inside
from lanescope.fixes.ranking import divisors
from lanescope.nsight import validate
from lanescope.trace import Space, Trace, read_metadata

STATIC = pathlib.Path(__file__).parent
STATIC_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
}
STATIC_NAME = re.compile(r"[a-z0-9][a-z0-9_-]*\.(html|js|css|svg)")
DEFAULT_PORT = 8765
MAX_LISTED = 500
MAX_CACHED = 8
MAX_BODY = 1 << 16
SUGGESTIONS = 8
# The shared memory of current GPUs holds at most 228 KB per CTA, so a buffer that
# ends above 1 MiB is a typing error, and its search would run for hours.
MAX_BUFFER_END = 1 << 20
WILDCARD_HOSTS = frozenset({"", "0.0.0.0", "::"})
STOP_SIGNALS = tuple(
    getattr(signal, name) for name in ("SIGTERM", "SIGHUP") if hasattr(signal, name)
)
MEASURED_QUANTITY = {"wavefronts": "shared_wavefronts", "sectors": "global_sectors"}
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
SKIPPED = frozenset({"__pycache__", "node_modules"})
Field = TypeVar("Field", str, int, dict)
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data:; base-uri 'none'; "
        "form-action 'none'; frame-ancestors 'none'"
    ),
}


class NotFound(LookupError):
    """A request names a route or a trace that does not exist."""


@dataclasses.dataclass(frozen=True)
class Listed:
    """One trace file found under the paths of the server."""

    id: str
    path: pathlib.Path
    name: str
    folder: str


class Library:
    """The trace files under the given paths, with the most recently used traces in memory."""

    def __init__(self, paths: Sequence[pathlib.Path]) -> None:
        self.roots = [path.resolve() for path in paths]
        self._listed: dict[str, Listed] = {}
        self._cache: collections.OrderedDict[tuple, tuple[Trace, dict]] = collections.OrderedDict()
        self._lock = threading.Lock()

    def listing(self) -> dict:
        """Return every trace file under the roots, at most MAX_LISTED of them."""
        found: dict[str, tuple[Listed, dict]] = {}
        truncated = False
        for index, root in enumerate(self.roots):
            if root.is_file():
                candidates = [root] if root.suffix == ".npz" else []
                base = root.parent
            else:
                candidates = _trace_files(root) if root.is_dir() else []
                base = root
            for path in candidates:
                if len(found) == MAX_LISTED:
                    truncated = True
                    break
                facts = _file_facts(path)
                if facts is None:
                    continue
                relative = path.relative_to(base).as_posix()
                item = Listed(f"{index}/{relative}", path, path.name, _folder(base, path))
                found[item.id] = (item, facts)
        with self._lock:
            self._listed = {key: item for key, (item, _) in found.items()}
        return {
            "traces": [
                {"id": item.id, "name": item.name, "folder": item.folder} | facts
                for item, facts in found.values()
            ],
            "truncated": truncated,
        }

    def load(self, trace_id: str) -> tuple[Trace, dict]:
        """Return a listed trace with its overview, from memory while the file is unchanged."""
        with self._lock:
            item = self._listed.get(trace_id)
        if item is None:
            self.listing()
            with self._lock:
                item = self._listed.get(trace_id)
        if item is None:
            raise NotFound(f"no listed trace has the id {trace_id}")
        # The counts of a GPU check join the overview, so a new summary renews the entry too.
        summary = validate.summary_path(item.path)
        try:
            key = (item.path, _stamp(item.path), _stamp(summary))
            with self._lock:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
            trace = Trace.load(item.path)
        except FileNotFoundError:
            self.listing()
            raise NotFound(f"the trace {trace_id} no longer exists") from None
        counts = validate.measured_counts(item.path, str(trace.metadata.get("kernel", "")))
        entry = (trace, _with_measurements(analysis.overview(trace), counts))
        with self._lock:
            self._cache[key] = entry
            while len(self._cache) > MAX_CACHED:
                self._cache.popitem(last=False)
        return entry


class GuiServer(http.server.ThreadingHTTPServer):
    """The HTTP server of the GUI, with its library and the host names it answers to."""

    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        library: Library,
        allowed_hosts: Sequence[str] = (),
    ) -> None:
        bound = address[0].strip("[]").lower()
        if ":" in bound:
            self.address_family = socket.AF_INET6
        super().__init__((bound, address[1]), Handler)
        self.library = library
        hosts = set(LOOPBACK_HOSTS) | {host.strip("[]").lower() for host in allowed_hosts}
        if bound not in WILDCARD_HOSTS:
            # A host name or a short address such as 127.2 binds to an address that the
            # printed page address names, so both the name and that address are allowed.
            hosts |= {bound, str(self.server_address[0]).lower()}
        self.allowed_hosts = frozenset(hosts)

    @property
    def url(self) -> str:
        """Return the address of the page, on localhost for a server on every address."""
        host = str(self.server_address[0])
        if host in WILDCARD_HOSTS:
            host = "localhost"
        elif ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.server_address[1]}/"


class Handler(http.server.BaseHTTPRequestHandler):
    """Routes of the GUI: the page, its static files, and the JSON API."""

    server_version = f"LaneScope/{__version__}"

    @property
    def app(self) -> GuiServer:
        """Return the server with the library and the allowed hosts."""
        assert isinstance(self.server, GuiServer)
        return self.server

    def do_GET(self) -> None:
        """Answer the page, a static file, or a read-only API route."""
        self._respond(lambda: self._get(urllib.parse.urlsplit(self.path)))

    def do_POST(self) -> None:
        """Answer the what-if route, the one route with a JSON body."""
        self._respond(self._post)

    def log_message(self, format: str, *args) -> None:
        """Leave requests unlogged, since the page sends one per slider step."""

    def _get(self, url: urllib.parse.SplitResult) -> tuple[int, str, bytes]:
        if url.path in ("/", "/index.html"):
            return _static("index.html")
        if url.path.startswith("/static/"):
            return _static(url.path.removeprefix("/static/"))
        return _json(answer_get(self.app.library, url.path, _query(url.query)))

    def _post(self) -> tuple[int, str, bytes]:
        if urllib.parse.urlsplit(self.path).path != "/api/whatif":
            raise NotFound(f"no route {self.path}")
        if self.headers.get_content_type() != "application/json":
            raise ValueError("the what-if route takes a JSON body")
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            raise ValueError(f"the request body must hold 1 to {MAX_BODY} bytes")
        try:
            body = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"the request body is no JSON ({error})") from error
        return _json(answer_post(self.app.library, "/api/whatif", body))

    def _respond(self, route: Callable[[], tuple[int, str, bytes]]) -> None:
        if not self._host_allowed():
            self._send(403, *_error("the server answers only to its own host names"))
            return
        if self.path.startswith("/api/") and not self._same_origin():
            self._send(403, *_error("the API answers only its own page"))
            return
        try:
            status, kind, payload = route()
        except NotFound as error:
            status, kind, payload = 404, *_error(str(error))
        except ValueError as error:
            status, kind, payload = 400, *_error(str(error))
        except OSError as error:
            status, kind, payload = 400, *_error(f"cannot read the file ({error.strerror})")
        except Exception:  # noqa: BLE001 - the page needs an answer, the log keeps the cause
            traceback.print_exc(file=sys.stderr)
            status, kind, payload = 500, *_error("the server failed, its log has the cause")
        self._send(status, kind, payload)

    def _host_allowed(self) -> bool:
        host = self.headers.get("Host")
        if not host:
            return False
        try:
            name = urllib.parse.urlsplit(f"//{host}").hostname
        except ValueError:
            return False
        return name is not None and name.lower() in self.app.allowed_hosts

    def _same_origin(self) -> bool:
        # Browsers mark each request with the site of the page that sends it. Another
        # port of this machine counts as another site, as it does for the browser.
        if self.headers.get("Sec-Fetch-Site") in ("cross-site", "same-site"):
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin == f"http://{self.headers.get('Host')}"

    def _send(self, status: int, kind: str, payload: bytes) -> None:
        try:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            # The page cancels a request once a newer one replaces it, such as on a slider.
            self.close_connection = True


def answer_get(library: Library, path: str, query: dict[str, str]) -> dict:
    """Return the answer of a read-only API route of the page."""
    if path == "/api/info":
        return {"version": __version__, "roots": [str(root) for root in library.roots]}
    if path == "/api/traces":
        return library.listing()
    route = TRACE_ROUTES.get(path)
    if route is None:
        raise NotFound(f"no route {path}")
    trace, overview = library.load(_text(query, "id"))
    return route(trace, overview, query)


def answer_post(library: Library, path: str, body: object) -> dict:
    """Return the answer of the what-if route, the one route with a JSON body."""
    if path != "/api/whatif":
        raise NotFound(f"no route {path}")
    if not isinstance(body, dict):
        raise ValueError("the request body must be a JSON object")
    trace, overview = library.load(_field(body, "id", str))
    bounds = _field(body, "buffer", dict)
    buffer = _buffer(_field(bounds, "start", int), _field(bounds, "size", int))
    return _whatif(trace, overview, buffer, body)


def _trace_route(trace: Trace, overview: dict, query: dict[str, str]) -> dict:
    return overview


def _patterns_route(trace: Trace, overview: dict, query: dict[str, str]) -> dict:
    return analysis.patterns(trace, _integer(query, "pc"), _space(_text(query, "space")))


def _access_route(trace: Trace, overview: dict, query: dict[str, str]) -> dict:
    return analysis.access(trace, _integer(query, "row"))


def _buffer_route(trace: Trace, overview: dict, query: dict[str, str]) -> dict:
    if "size" in query:
        return _buffer_facts(trace, _buffer(_text(query, "start"), _text(query, "size")))
    return _buffer_facts(trace, infer_buffer(trace, _integer(query, "pc")))


def _suggest_route(trace: Trace, overview: dict, query: dict[str, str]) -> dict:
    buffer = None
    if "size" in query:
        buffer = _buffer(_text(query, "start"), _text(query, "size"))
    pc = _integer(query, "pc") if "pc" in query else None
    return analysis.suggest(trace, buffer, pc, SUGGESTIONS)


TRACE_ROUTES: dict[str, Callable[[Trace, dict, dict[str, str]], dict]] = {
    "/api/trace": _trace_route,
    "/api/patterns": _patterns_route,
    "/api/access": _access_route,
    "/api/buffer": _buffer_route,
    "/api/suggest": _suggest_route,
}


def serve(
    paths: Sequence[pathlib.Path],
    host: str = "127.0.0.1",
    port: int = DEFAULT_PORT,
    allowed_hosts: Sequence[str] = (),
    open_browser: bool = False,
) -> None:
    """Serve the GUI for the traces under `paths` until the process is stopped."""
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"no such file or folder: {', '.join(missing)}")
    server = GuiServer((host, port), Library(paths), allowed_hosts)
    print(f"LaneScope GUI on {server.url} (stop with Ctrl+C)", flush=True)
    if open_browser:
        webbrowser.open(server.url)
    # A stop by the process manager or a closed terminal ends the server as Ctrl+C
    # does, which lets the caller remove its temporary folders. A signal that the
    # caller ignores, such as SIGHUP under nohup, stays ignored.
    if threading.current_thread() is threading.main_thread():
        for number in STOP_SIGNALS:
            if signal.getsignal(number) is signal.SIG_DFL:
                signal.signal(number, _interrupt)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _interrupt(signum: int, frame: object) -> None:
    # A closed terminal sends SIGHUP again, which would cut the cleanup short, so the
    # first stop signal turns the rest off.
    for number in STOP_SIGNALS:
        signal.signal(number, signal.SIG_IGN)
    raise KeyboardInterrupt


def _whatif(trace: Trace, overview: dict, buffer: Buffer, body: dict) -> dict:
    change = analysis.change_from_params(body.get("change"))
    data = analysis.whatif(trace, buffer, change)
    element = _element_bytes(_widths(trace, buffer))
    data["element_bytes"] = element
    data["forms"] = codegen.forms(buffer, change, element)
    data["totals"] = _totals(overview, data["instructions"])
    row = body.get("row")
    if row is not None:
        if not isinstance(row, int) or isinstance(row, bool):
            raise ValueError("row must be an integer")
        data["access"] = analysis.access(apply(trace, buffer, change), row)
    return data


def _totals(overview: dict, changed: list[dict]) -> dict:
    """Return the achieved and excess counts of each unit before and after a change."""
    totals = {
        unit: {
            "achieved": [total["achieved"]] * 2,
            "excessive": [total["excessive"]] * 2,
        }
        for unit, total in overview["totals"].items()
    }
    for item in changed:
        unit = totals[item["before"]["unit"]]
        for name in ("achieved", "excessive"):
            unit[name][1] += item["after"][name] - item["before"][name]
    return {
        unit: {name: {"before": pair[0], "after": pair[1]} for name, pair in counts.items()}
        for unit, counts in totals.items()
    }


def _with_measurements(
    overview: dict, counts: dict[tuple[int, str], tuple[float | None, str]] | None
) -> dict:
    """Return the overview with the Nsight Compute count of each instruction that a check has."""
    if not counts:
        return overview
    records, matched, total = [], 0, 0
    for record in overview["instructions"]:
        found = counts.get((int(record["pc"], 16), MEASURED_QUANTITY[record["unit"]]))
        if found is not None:
            value, sass = found
            agrees = value is not None and round(value) == record["achieved"]
            record = record | {"measured": {"achieved": value, "sass": sass, "agrees": agrees}}
            total += 1
            matched += agrees
        records.append(record)
    return overview | {"instructions": records, "measured": {"matched": matched, "total": total}}


def _buffer_facts(trace: Trace, buffer: Buffer) -> dict:
    """Return a buffer with the size of its elements and the row sizes that fit it.

    The facts also give the highest end of a buffer, which the page checks before it
    sends a typed buffer.
    """
    widths = _widths(trace, buffer)
    widest = int(widths.max()) if widths.size else 4
    unit = max(widest, 4)
    return {
        "start": buffer.start,
        "size": buffer.size,
        "element_bytes": _element_bytes(widths),
        "widest": widest,
        "row_sizes": [size for size in divisors(buffer.size) if size % unit == 0],
        "max_end": MAX_BUFFER_END,
    }


def _element_bytes(widths: np.ndarray) -> int:
    # Vector loads of a float tile still hold floats, so the narrowest access names the element.
    return int(widths.min()) if widths.size else 4


def _widths(trace: Trace, buffer: Buffer) -> np.ndarray:
    inside = lanes_inside(trace, buffer).any(axis=1)
    return trace.width[inside]


def _buffer(start: object, size: object) -> Buffer:
    buffer = Buffer(_as_integer("start", start), _as_integer("size", size))
    if buffer.end > MAX_BUFFER_END:
        raise ValueError(f"a shared buffer must end at or below {MAX_BUFFER_END:#x}")
    return buffer


def _space(text: str) -> Space:
    try:
        return Space[text.upper()]
    except KeyError:
        raise ValueError("space must be shared or global") from None


def _query(text: str) -> dict[str, str]:
    pairs = urllib.parse.parse_qsl(text, keep_blank_values=True, max_num_fields=16)
    return dict(pairs)


def _text(query: dict[str, str], name: str) -> str:
    if name not in query:
        raise ValueError(f"the request needs the parameter {name}")
    return query[name]


def _integer(query: dict[str, str], name: str) -> int:
    return _as_integer(name, _text(query, name))


def _as_integer(name: str, value: object) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            pass
    raise ValueError(f"{name} must be an integer, such as 288 or 0x120")


def _field(body: dict, name: str, kind: type[Field]) -> Field:
    value = body.get(name)
    if not isinstance(value, kind) or isinstance(value, bool):
        raise ValueError(f"the request body needs {name} as {kind.__name__}")
    return value


def _trace_files(root: pathlib.Path) -> list[pathlib.Path]:
    """Return the .npz files under a folder, skipping hidden and cache folders."""
    found = []
    for folder, names, files in os.walk(root):
        names[:] = sorted(
            name for name in names if not name.startswith(".") and name not in SKIPPED
        )
        found.extend(pathlib.Path(folder) / name for name in sorted(files) if name.endswith(".npz"))
        if len(found) > MAX_LISTED:
            break
    return found


def _folder(base: pathlib.Path, path: pathlib.Path) -> str:
    parent = path.parent.relative_to(base).as_posix()
    return base.name if parent == "." else f"{base.name}/{parent}"


def _file_facts(path: pathlib.Path) -> dict | None:
    """Return the size, the age, and the kernel of a trace file, or None once it vanished."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    try:
        kernel = read_metadata(path).get("kernel")
    except (ValueError, OSError):
        kernel = None
    return {
        "bytes": stat.st_size,
        "modified": int(stat.st_mtime),
        "kernel": kernel if isinstance(kernel, str) else None,
    }


def _stamp(path: pathlib.Path | None) -> tuple[int, int] | None:
    """Return the change time and the size of a file, or None for no file."""
    if path is None:
        return None
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_size


def _static(name: str) -> tuple[int, str, bytes]:
    if not STATIC_NAME.fullmatch(name):
        raise NotFound(f"no static file {name}")
    path = STATIC / name
    if not path.is_file():
        raise NotFound(f"no static file {name}")
    return 200, STATIC_TYPES[path.suffix], path.read_bytes()


def _json(data: object) -> tuple[int, str, bytes]:
    payload = json.dumps(data, allow_nan=False, separators=(",", ":")).encode()
    return 200, "application/json", payload


def _error(message: str) -> tuple[str, bytes]:
    return "application/json", json.dumps({"error": message}).encode()
