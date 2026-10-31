import importlib.util
import json
import pathlib

import pytest

from lanescope.gui.browser import BrowserApi

ROOT = pathlib.Path(__file__).parent.parent


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    return BrowserApi(str(tmp_path_factory.mktemp("browser") / "demo"))


def ask(api, method, path, payload):
    return json.loads(api.request(method, path, json.dumps(payload)))


def test_browser_api_lists_the_four_demo_traces(api):
    answer = ask(api, "GET", "/api/traces", {})
    assert answer["status"] == 200
    names = [trace["name"] for trace in answer["data"]["traces"]]
    assert names == ["1_transpose.npz", "2_row_sums.npz", "3_column_sums.npz", "4_matmul_tile.npz"]


def test_browser_api_answers_the_overview_and_the_whatif_of_a_trace(api):
    listing = ask(api, "GET", "/api/traces", {})["data"]["traces"]
    transpose = next(trace["id"] for trace in listing if trace["name"] == "1_transpose.npz")
    overview = ask(api, "GET", "/api/trace", {"id": transpose})["data"]
    assert overview["totals"]["wavefronts"]["excessive"] == 3968
    change = {"kind": "swizzle", "bits": 5, "base": 2, "shift": 5}
    body = {"id": transpose, "buffer": {"start": 0, "size": 4096}, "change": change}
    whatif = ask(api, "POST", "/api/whatif", body)
    assert whatif["status"] == 200
    assert whatif["data"]["totals"]["wavefronts"]["excessive"] == {"before": 3968, "after": 0}


@pytest.mark.parametrize(
    ("method", "path", "payload", "status"),
    [
        ("GET", "/api/none", {}, 404),
        ("GET", "/api/trace", {"id": "9/none.npz"}, 404),
        ("GET", "/api/trace", [], 400),
        ("POST", "/api/whatif", [], 400),
    ],
)
def test_browser_api_answers_a_bad_request_with_its_status(api, method, path, payload, status):
    answer = ask(api, method, path, payload)
    assert answer["status"] == status
    assert answer["error"]


def test_built_page_runs_the_api_in_the_browser_with_relative_paths(tmp_path):
    spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
    assert spec is not None and spec.loader is not None
    build_site = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_site)
    build_site.assemble(tmp_path, "lanescope-1.0.0-py3-none-any.whl")
    page = (tmp_path / "index.html").read_text()
    assert '<meta name="lanescope-backend" content="pyodide" />' in page
    assert 'content="./lanescope-1.0.0-py3-none-any.whl"' in page
    assert 'id="pyodide-status"' in page
    assert "/static/" not in page
    assert all((tmp_path / name).is_file() for name in build_site.ASSETS + (".nojekyll",))
