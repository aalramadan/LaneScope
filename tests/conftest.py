import contextlib
import json
import threading
import time

import pytest

from lanescope import demos
from lanescope.gui import server as gui
from lanescope.trace import Access, Kind, Space, Trace


@contextlib.contextmanager
def served(paths, host="127.0.0.1", allowed_hosts=()):
    """Run a GUI server for the given paths on a free port until the block ends."""
    server = gui.GuiServer((host, 0), gui.Library(list(paths)), allowed_hosts)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(scope="session")
def serve():
    """Return the context manager that runs a GUI server for a list of paths."""
    return served


@pytest.fixture(scope="session")
def gui_folder(tmp_path_factory):
    """Return a folder of traces for the page: the transpose, wide and byte loads, and more."""
    folder = tmp_path_factory.mktemp("browser")
    demos.transpose_trace().save(folder / "transpose.npz")
    wide = Access(0x40, Space.SHARED, Kind.LOAD, 16, tuple(64 * lane for lane in range(32)))
    Trace.from_accesses([wide], {"kernel": "wide"}).save(folder / "wide.npz")
    byte = Access(0x40, Space.SHARED, Kind.LOAD, 1, tuple(0x40 + lane for lane in range(32)))
    Trace.from_accesses([byte], {"kernel": "bytes"}).save(folder / "bytes.npz")
    record = folder / "check" / "record"
    record.mkdir(parents=True)
    demos.transpose_trace(tiles_per_side=1).save(record / "trace-0.npz")
    rows = [
        {"launch": 0, "offset": 0x120, "quantity": "shared_wavefronts", "nsight": 1024.0},
        {"launch": 0, "offset": 0xA0, "quantity": "global_sectors", "nsight": 100.0},
        {"launch": 0, "offset": 0xB0, "quantity": "shared_wavefronts", "nsight": None},
    ]
    for row in rows:
        row |= {"kernel": "transpose", "sass": "?"}
    (folder / "check" / "validation.json").write_text(json.dumps({"rows": rows}))
    Trace.from_accesses([], {"kernel": "atomics_only"}).save(folder / "empty.npz")
    column = Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(32)))
    strided = Access(
        0x20, Space.GLOBAL, Kind.LOAD, 4, tuple(4096 + 128 * lane for lane in range(32))
    )
    source = {"source": ["x = tile[i][j] + in[k];"], "lines": {"0x10": 1, "0x20": 1}}
    Trace.from_accesses([column, strided], {"kernel": "mixed"} | source).save(folder / "mixed.npz")
    (folder / "broken.npz").write_bytes(b"not a trace")
    return folder


@pytest.fixture(scope="session")
def url(gui_folder):
    """Return the address of a GUI server for the folder of the page tests."""
    with served([gui_folder]) as server:
        yield f"http://localhost:{server.server_address[1]}/"


@pytest.fixture(scope="module")
def browser():
    """Return a headless Chromium, or skip when Playwright or its browser is missing.

    Playwright keeps an event loop running while it is open, which would break the
    asyncio tests of later modules, so each module of page tests opens its own.
    """
    driver = pytest.importorskip("playwright.sync_api")
    with driver.sync_playwright() as playwright:
        try:
            instance = playwright.chromium.launch()
        except driver.Error as error:
            pytest.skip(f"no browser for Playwright ({error.message.splitlines()[0]})")
        yield instance
        instance.close()


@pytest.fixture
def page(browser, url):
    """Return the page on the transpose, and fail the test on any error of its scripts."""
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    # The browser logs each answer of 400 or more, which some tests ask for on purpose.
    page.on(
        "console",
        lambda message: (
            message.type == "error"
            and not message.text.startswith("Failed to load resource")
            and errors.append(message.text)
        ),
    )
    page.goto(f"{url}#trace=0/transpose.npz")
    page.wait_for_selector("figure.grid svg")
    yield page
    context.close()
    assert errors == []


@pytest.fixture
def slow():
    """Return a function that wraps a server function to answer only after a delay."""

    def wrap(function, seconds):
        def call(*args, **kwargs):
            time.sleep(seconds)
            return function(*args, **kwargs)

        return call

    return wrap
