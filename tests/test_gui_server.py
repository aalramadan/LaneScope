import http.client
import io
import json
import socket
import time

import pytest

from lanescope import demos
from lanescope.gui import server as gui
from lanescope.trace import Access, Kind, Space, Trace


def make_folder(path):
    demos.transpose_trace().save(path / "transpose.npz")
    (path / "runs").mkdir()
    wide = Access(0x40, Space.SHARED, Kind.LOAD, 16, tuple(16 * lane for lane in range(32)))
    Trace.from_accesses([wide], {"kernel": "wide"}).save(path / "runs" / "wide.npz")
    (path / ".hidden").mkdir()
    demos.transpose_trace().save(path / ".hidden" / "skipped.npz")
    (path / "broken.npz").write_bytes(b"not a zip file")
    return path


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    return make_folder(tmp_path_factory.mktemp("traces"))


@pytest.fixture(scope="module")
def server(folder, serve):
    with serve([folder]) as instance:
        yield instance


def request(server, path, method="GET", body=None, headers=None):
    address = str(server.server_address[0])
    connection = http.client.HTTPConnection(address, server.server_address[1], timeout=30)
    sent = {"Host": f"localhost:{server.server_address[1]}"} | (headers or {})
    payload = None
    if body is not None:
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        sent.setdefault("Content-Type", "application/json")
    connection.request(method, path, body=payload, headers=sent)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response, data


def get_json(server, path):
    response, data = request(server, path)
    assert response.status == 200, data
    return json.loads(data)


def trace_id(server, name):
    listing = get_json(server, "/api/traces")
    return next(item["id"] for item in listing["traces"] if item["name"] == name)


def test_page_and_static_files_carry_their_types_and_security_headers(server):
    response, data = request(server, "/")
    assert response.status == 200
    assert b"<title>LaneScope</title>" in data
    assert "frame-ancestors 'none'" in (response.getheader("Content-Security-Policy") or "")
    assert response.getheader("X-Content-Type-Options") == "nosniff"
    response, _ = request(server, "/static/app.js")
    assert (response.getheader("Content-Type") or "").startswith("text/javascript")


def test_static_route_serves_every_module_that_the_page_imports(server, page_modules):
    assert len(page_modules) > 1
    for name in page_modules:
        response, _ = request(server, f"/static/{name}")
        assert response.status == 200, name


@pytest.mark.parametrize(
    "path",
    [
        "/static/server.py",
        "/static/__init__.py",
        "/static/../demos.py",
        "/static/%2e%2e/demos.py",
        "/static/.hidden.js",
        "/static/none.js",
    ],
)
def test_static_route_serves_only_files_of_the_gui_folder(server, path):
    response, _ = request(server, path)
    assert response.status == 404


def test_listing_skips_hidden_folders_and_names_each_folder(server, folder):
    listing = get_json(server, "/api/traces")
    names = {(item["name"], item["folder"]) for item in listing["traces"]}
    assert names == {
        ("broken.npz", folder.name),
        ("transpose.npz", folder.name),
        ("wide.npz", f"{folder.name}/runs"),
    }
    assert not listing["truncated"]


def test_trace_route_gives_the_overview_with_source_and_totals(server):
    data = get_json(server, f"/api/trace?id={trace_id(server, 'transpose.npz')}")
    assert data["instructions"][0]["pc"] == "0x120"
    assert data["totals"]["wavefronts"]["excessive"] == 3968
    assert len(data["source"]) == len(demos.TRANSPOSE_SOURCE)


def test_patterns_and_access_routes_show_the_worst_column_read(server):
    identifier = trace_id(server, "transpose.npz")
    patterns = get_json(server, f"/api/patterns?id={identifier}&pc=0x120&space=shared")
    row = patterns["patterns"][0]["row"]
    access = get_json(server, f"/api/access?id={identifier}&row={row}")
    assert (access["achieved"], access["phases"][0]["worst_bank"]) == (32, 0)


def test_buffer_route_gives_the_element_size_and_the_rows_that_fit(server):
    identifier = trace_id(server, "transpose.npz")
    data = get_json(server, f"/api/buffer?id={identifier}&pc=0x120")
    assert (data["start"], data["size"], data["element_bytes"], data["widest"]) == (0, 4096, 4, 4)
    assert 128 in data["row_sizes"]
    assert all(4096 % size == 0 for size in data["row_sizes"])


def test_suggest_route_ranks_the_swizzle_and_the_pad_first(server):
    identifier = trace_id(server, "transpose.npz")
    data = get_json(server, f"/api/suggest?id={identifier}&start=0&size=4096")
    kinds = [change["params"]["kind"] for change in data["changes"][:2]]
    assert kinds == ["swizzle", "padding"]
    assert len(data["changes"]) == gui.SUGGESTIONS


def test_whatif_route_predicts_the_swizzled_read_with_its_code(server):
    body = {
        "id": trace_id(server, "transpose.npz"),
        "buffer": {"start": 0, "size": 4096},
        "change": {"kind": "swizzle", "bits": 5, "base": 2, "shift": 5},
        "row": 8,
    }
    response, data = request(server, "/api/whatif", "POST", body)
    assert response.status == 200, data
    result = json.loads(data)
    assert result["totals"]["wavefronts"]["excessive"] == {"before": 3968, "after": 0}
    assert result["totals"]["sectors"]["excessive"] == {"before": 0, "after": 0}
    assert result["access"]["achieved"] == 1
    assert {"label": "CuTe", "code": "Swizzle<5, 0, 5>"} in result["forms"]


def test_whatif_route_counts_the_padding_of_wide_accesses_by_their_width(server):
    body = {
        "id": trace_id(server, "wide.npz"),
        "buffer": {"start": 0, "size": 512},
        "change": {"kind": "padding", "row_bytes": 128, "pad_bytes": 16},
    }
    response, data = request(server, "/api/whatif", "POST", body)
    assert response.status == 200, data
    assert json.loads(data)["element_bytes"] == 16


@pytest.mark.parametrize(
    ("body", "headers", "reason"),
    [
        (b"{}", {"Content-Type": "text/plain"}, "JSON body"),
        (b"not json", None, "no JSON"),
        ([1, 2], None, "JSON object"),
        ({"id": "0/transpose.npz", "buffer": {"start": 0}}, None, "size as int"),
        (
            {
                "id": "0/transpose.npz",
                "buffer": {"start": 0, "size": 4096},
                "change": {"kind": "x"},
            },
            None,
            "padding or swizzle",
        ),
        (
            {
                "id": "0/transpose.npz",
                "buffer": {"start": 2, "size": 96},
                "change": {"kind": "padding", "row_bytes": 32, "pad_bytes": 4},
            },
            None,
            "cross the edge",
        ),
        (b"x" * (gui.MAX_BODY + 1), None, "1 to"),
    ],
    ids=["text", "no-json", "list", "no-size", "no-kind", "crossing", "too-large"],
)
def test_whatif_route_explains_each_bad_request(server, body, headers, reason):
    response, data = request(server, "/api/whatif", "POST", body, headers)
    assert response.status == 400
    assert reason in json.loads(data)["error"]


def test_routes_report_unknown_traces_rows_and_paths(server):
    identifier = trace_id(server, "transpose.npz")
    assert request(server, "/api/trace?id=9/none.npz")[0].status == 404
    assert request(server, f"/api/access?id={identifier}&row=99999")[0].status == 400
    assert request(server, f"/api/access?id={identifier}&row=x")[0].status == 400
    assert request(server, f"/api/patterns?id={identifier}&pc=0x120&space=local")[0].status == 400
    assert request(server, "/api/nothing")[0].status == 404


def test_a_broken_trace_file_gives_its_reason(server):
    response, data = request(server, f"/api/trace?id={trace_id(server, 'broken.npz')}")
    assert response.status == 400
    assert "no .npz trace file" in json.loads(data)["error"]


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "", "[evil]", "[::1"])
def test_requests_that_name_another_host_are_refused(server, host):
    response, _ = request(server, "/api/traces", headers={"Host": host})
    assert response.status == 403


def test_server_bound_to_every_address_still_answers_only_to_known_hosts(folder, serve):
    with serve([folder], "0.0.0.0", ["workstation"]) as instance:
        assert instance.allowed_hosts == {"localhost", "127.0.0.1", "::1", "workstation"}
        assert instance.url == f"http://localhost:{instance.server_address[1]}/"
        foreign = request(instance, "/api/traces", headers={"Host": "evil.example"})[0]
        named = request(instance, "/api/traces", headers={"Host": "workstation:1"})[0]
        assert (foreign.status, named.status) == (403, 200)


def test_server_on_one_address_names_that_address(folder, serve):
    with serve([folder], "127.0.0.2") as instance:
        assert instance.url == f"http://127.0.0.2:{instance.server_address[1]}/"
        response, _ = request(instance, "/api/info", headers={"Host": "127.0.0.2"})
        assert response.status == 200


@pytest.mark.skipif(not socket.has_ipv6, reason="IPv6 is needed")
def test_server_listens_on_an_ipv6_address(folder, serve):
    try:
        context = serve([folder], "::1")
        instance = context.__enter__()
    except OSError as error:
        pytest.skip(f"no IPv6 loopback ({error})")
    try:
        assert instance.url == f"http://[::1]:{instance.server_address[1]}/"
        response, _ = request(instance, "/api/info", headers={"Host": "[::1]"})
        assert response.status == 200
    finally:
        context.__exit__(None, None, None)


def test_library_reloads_a_trace_file_that_changed(tmp_path):
    folder = make_folder(tmp_path)
    library = gui.Library([folder])
    identifier = next(
        item["id"] for item in library.listing()["traces"] if item["name"] == "wide.npz"
    )
    first, _ = library.load(identifier)
    narrow = Access(0x40, Space.SHARED, Kind.LOAD, 4, tuple(4 * lane for lane in range(32)))
    Trace.from_accesses([narrow, narrow]).save(folder / "runs" / "wide.npz")
    second, overview = library.load(identifier)
    assert (len(first), len(second)) == (1, 2)
    assert overview["rows"] == 2


def test_library_lists_a_single_trace_file(folder):
    listing = gui.Library([folder / "transpose.npz"]).listing()
    assert [item["name"] for item in listing["traces"]] == ["transpose.npz"]


def test_serve_rejects_a_missing_path(tmp_path):
    with pytest.raises(ValueError, match="no such file or folder"):
        gui.serve([tmp_path / "missing"], port=0)


class ClosedStream(io.BufferedIOBase):
    def write(self, buffer, /) -> int:
        raise BrokenPipeError


def test_a_request_that_the_page_cancelled_ends_without_an_error():
    handler = gui.GuiRequestHandler.__new__(gui.GuiRequestHandler)
    handler.wfile = ClosedStream()
    handler.request_version = "HTTP/1.1"
    handler.requestline = "GET /api/traces HTTP/1.1"
    handler.command = "GET"
    handler.close_connection = False
    handler._send(200, "application/json", b"{}")
    assert handler.close_connection


def test_trace_of_a_gpu_check_carries_the_nsight_compute_counts(tmp_path):
    record = tmp_path / "record"
    record.mkdir()
    demos.transpose_trace(tiles_per_side=1).save(record / "trace-0.npz")
    rows = [
        {"launch": 0, "offset": 0x120, "quantity": "shared_wavefronts", "nsight": 1024.0},
        {"launch": 0, "offset": 0xA0, "quantity": "global_sectors", "nsight": 100.0},
    ]
    for row, sass in zip(rows, ("LDS", "LDG"), strict=True):
        row |= {"sass": sass, "kernel": "transpose(float *, const float *, int)"}
    (tmp_path / "validation.json").write_text(json.dumps({"rows": rows}))
    library = gui.Library([tmp_path])
    (item,) = library.listing()["traces"]
    assert item["kernel"] == "transpose"
    _, overview = library.load(item["id"])
    by_pc = {record["pc"]: record for record in overview["instructions"]}
    assert by_pc["0x120"]["measured"] == {"achieved": 1024.0, "sass": "LDS", "agrees": True}
    assert by_pc["0xa0"]["measured"]["agrees"] is False
    assert "measured" not in by_pc["0xb0"]
    assert overview["measured"] == {"matched": 1, "total": 2}


@pytest.mark.parametrize(
    ("path", "headers", "status"),
    [
        ("/api/traces", {"Sec-Fetch-Site": "cross-site"}, 403),
        ("/api/traces", {"Sec-Fetch-Site": "same-site"}, 403),
        ("/api/traces", {"Origin": "http://evil.example"}, 403),
        ("/api/traces", {"Sec-Fetch-Site": "same-origin"}, 200),
        ("/", {"Sec-Fetch-Site": "cross-site"}, 200),
    ],
)
def test_api_answers_only_its_own_page(server, path, headers, status):
    assert request(server, path, headers=headers)[0].status == status


def test_api_accepts_the_origin_of_its_own_page(server):
    origin = f"http://localhost:{server.server_address[1]}"
    assert request(server, "/api/traces", headers={"Origin": origin})[0].status == 200


@pytest.mark.parametrize(
    "path",
    [
        "/api/suggest?id={id}&start=0&size=0x800000000000",
        "/api/buffer?id={id}&start=0x100000&size=16",
    ],
)
def test_buffers_beyond_the_shared_memory_window_are_refused(server, path):
    response, data = request(server, path.format(id=trace_id(server, "transpose.npz")))
    assert response.status == 400
    assert "must end at or below 0x100000" in json.loads(data)["error"]


def test_buffer_route_names_the_highest_end_of_a_typed_buffer(server):
    identifier = trace_id(server, "transpose.npz")
    data = get_json(server, f"/api/buffer?id={identifier}&pc=0x120")
    assert data["max_end"] == gui.MAX_BUFFER_END == 0x100000


def test_buffer_route_gives_the_rows_of_a_typed_buffer(server):
    identifier = trace_id(server, "transpose.npz")
    data = get_json(server, f"/api/buffer?id={identifier}&start=0&size=4096")
    assert data["row_sizes"] == [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    response, data = request(server, f"/api/buffer?id={identifier}&start=2&size=96")
    assert response.status == 400


def test_whatif_route_refuses_an_unhashable_change_kind(server):
    body = {
        "id": trace_id(server, "transpose.npz"),
        "buffer": {"start": 0, "size": 4096},
        "change": {"kind": []},
    }
    assert request(server, "/api/whatif", "POST", body)[0].status == 400


def test_overview_follows_a_check_summary_written_after_the_first_load(tmp_path):
    record = tmp_path / "record"
    record.mkdir()
    demos.transpose_trace(tiles_per_side=1).save(record / "trace-0.npz")
    library = gui.Library([tmp_path])
    (item,) = library.listing()["traces"]
    assert "measured" not in library.load(item["id"])[1]
    row = {"launch": 0, "offset": 0x120, "quantity": "shared_wavefronts", "nsight": 1024.0}
    row |= {"kernel": "transpose", "sass": "LDS"}
    (tmp_path / "validation.json").write_text(json.dumps({"rows": [row]}))
    assert library.load(item["id"])[1]["measured"] == {"matched": 1, "total": 1}


def test_a_listed_trace_that_vanished_answers_404(tmp_path, serve):
    demos.transpose_trace(tiles_per_side=1).save(tmp_path / "gone.npz")
    with serve([tmp_path]) as instance:
        identifier = trace_id(instance, "gone.npz")
        (tmp_path / "gone.npz").unlink()
        assert request(instance, f"/api/trace?id={identifier}")[0].status == 404
        assert get_json(instance, "/api/traces")["traces"] == []


def test_server_on_a_short_address_answers_at_the_address_it_prints(folder, serve):
    with serve([folder], "127.2") as instance:
        assert instance.url == f"http://127.0.0.2:{instance.server_address[1]}/"
        response, _ = request(instance, "/api/info", headers={"Host": "127.0.0.2"})
        assert response.status == 200


def test_listing_leaves_out_a_file_that_vanished_after_the_walk(tmp_path, monkeypatch):
    demos.transpose_trace(tiles_per_side=1).save(tmp_path / "kept.npz")
    walk = gui._trace_files
    monkeypatch.setattr(gui, "_trace_files", lambda root: [*walk(root), root / "gone.npz"])
    names = [item["name"] for item in gui.Library([tmp_path]).listing()["traces"]]
    assert names == ["kept.npz"]


def finished_summary(library, timeout=60.0):
    """Return the summary of a library once its thread has read every trace."""
    deadline = time.monotonic() + timeout
    while True:
        answer = library.summary()
        if answer["state"] == "done":
            return answer
        assert answer["state"] == "summing" and 0 <= answer["done"] < answer["total"]
        assert time.monotonic() < deadline, "the summary did not finish"
        time.sleep(0.02)


def test_summary_route_ranks_every_trace_once_its_thread_has_read_them(server):
    deadline = time.monotonic() + 60
    while (answer := get_json(server, "/api/summary"))["state"] != "done":
        assert time.monotonic() < deadline
        time.sleep(0.02)
    assert answer["traces"] == 2
    assert answer["unreadable"] == 1
    lines = [line for line in answer["lines"] if line["unit"] == "wavefronts"]
    assert lines[0]["worst"] == {"id": "0/transpose.npz", "pc": "0x120", "space": "shared"}
    assert answer["runs"] == [
        f"{server.library.roots[0].name}/transpose.npz",
        f"{server.library.roots[0].name}/runs/wide.npz",
    ]


def test_a_library_without_threads_sums_every_trace_during_the_call(tmp_path):
    demos.transpose_trace().save(tmp_path / "trace-0.npz")
    demos.histogram_trace().save(tmp_path / "trace-1.npz")
    answer = gui.Library([tmp_path], background=False).summary()
    assert answer["state"] == "done"
    assert answer["runs"] == [tmp_path.name]
    assert {kernel["name"] for kernel in answer["kernels"]} == {"transpose", "histogram"}


def test_a_summary_gives_the_kernel_times_of_each_run_folder_that_has_them(tmp_path):
    for name in ("timed", "untimed"):
        (tmp_path / name).mkdir()
        demos.transpose_trace().save(tmp_path / name / "trace-0.npz")
    times = {"launches": [1], "nanoseconds": [2_000], "shortest": 2_000, "longest": 2_000}
    result = {"format": 1, "rounds": 1, "kernels": {"transpose": times}}
    (tmp_path / "timed" / "kernel-times.json").write_text(json.dumps(result))
    answer = gui.Library([tmp_path], background=False).summary()
    assert answer["times"] == [
        {"run": f"{tmp_path.name}/timed", "rounds": 1, "total_ns": 2_000, "round_ns": [2_000]}
    ]
    assert {kernel["run"]: kernel["time_ns"] for kernel in answer["kernels"]} == {
        f"{tmp_path.name}/timed": 2_000,
        f"{tmp_path.name}/untimed": None,
    }


def test_a_summary_skips_a_broken_file_of_kernel_times_and_keeps_the_counts_of_its_run(tmp_path):
    for name in ("timed", "broken"):
        (tmp_path / name).mkdir()
        demos.transpose_trace().save(tmp_path / name / "trace-0.npz")
    times = {"launches": [1], "nanoseconds": [2_000], "shortest": 2_000, "longest": 2_000}
    result = {"format": 1, "rounds": 1, "kernels": {"transpose": times}}
    (tmp_path / "timed" / "kernel-times.json").write_text(json.dumps(result))
    (tmp_path / "broken" / "kernel-times.json").write_text('{"format": 1, "rounds": 2, "kern')
    answer = gui.Library([tmp_path], background=False).summary()
    assert answer["state"] == "done"
    assert [timed["run"] for timed in answer["times"]] == [f"{tmp_path.name}/timed"]
    assert [kernel["launches"] for kernel in answer["kernels"]] == [1, 1]
    (unread,) = answer["unreadable_times"]
    assert unread["run"] == f"{tmp_path.name}/broken"
    assert unread["file"] == str(tmp_path / "broken" / "kernel-times.json")


def test_a_summary_reads_a_changed_trace_again_and_forgets_a_deleted_one(tmp_path):
    demos.transpose_trace().save(tmp_path / "trace-0.npz")
    demos.transpose_trace().save(tmp_path / "trace-1.npz")
    library = gui.Library([tmp_path], background=False)
    assert library.summary()["kernels"][0]["launches"] == 2
    (tmp_path / "trace-1.npz").unlink()
    demos.histogram_trace().save(tmp_path / "trace-0.npz")
    answer = library.summary()
    assert [kernel["name"] for kernel in answer["kernels"]] == ["histogram"]
    assert len(library._facts) == 1


def test_a_summary_thread_that_meets_an_odd_file_goes_on_with_the_rest(tmp_path, monkeypatch):
    demos.transpose_trace().save(tmp_path / "a.npz")
    demos.transpose_trace().save(tmp_path / "b.npz")
    real = gui.summary.read_facts

    def odd(path):
        if path.name == "a.npz":
            raise RuntimeError("an odd file")
        return real(path)

    monkeypatch.setattr(gui.summary, "read_facts", odd)
    answer = finished_summary(gui.Library([tmp_path]))
    assert (answer["traces"], answer["unreadable"]) == (1, 1)


def test_a_trace_beyond_the_listing_opens_by_its_id(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        demos.transpose_trace(tiles_per_side=1).save(tmp_path / f"{name}.npz")
    monkeypatch.setattr(gui, "MAX_LISTED", 1)
    library = gui.Library([tmp_path])
    assert [item["name"] for item in library.listing()["traces"]] == ["a.npz"]
    trace, overview = library.load("0/c.npz")
    assert overview["kernel"] == "transpose"


@pytest.mark.parametrize(
    "trace_id",
    ["0/../outside.npz", "0/.hidden/skipped.npz", "0//etc/passwd", "1/transpose.npz", "x/a.npz"],
)
def test_an_id_outside_the_folders_of_the_page_opens_nothing(folder, trace_id):
    (folder.parent / "outside.npz").write_bytes(b"")
    with pytest.raises(gui.NotFound):
        gui.Library([folder]).load(trace_id)


def test_the_listing_names_each_kernel_by_its_short_name(tmp_path):
    long_name = "ns::kernel<" + ", ".join(f"ns::T{index}<int>" for index in range(9)) + ">"
    Trace.from_accesses([], {"kernel": long_name}).save(tmp_path / "long.npz")
    (item,) = gui.Library([tmp_path]).listing()["traces"]
    assert item["kernel"] == long_name
    assert item["kernel_name"] == "kernel<…>"


def test_a_server_reads_many_traces_in_several_processes(tmp_path, monkeypatch):
    for index in range(3):
        demos.transpose_trace(tiles_per_side=1).save(tmp_path / f"trace-{index}.npz")
    monkeypatch.setattr(gui.summary, "PARALLEL_TRACES", 1)
    answer = finished_summary(gui.Library([tmp_path], processes=True))
    (kernel,) = answer["kernels"]
    assert kernel["launches"] == 3


def test_closing_a_library_stops_its_summary_thread(tmp_path, monkeypatch, slow):
    for index in range(20):
        demos.transpose_trace(tiles_per_side=1).save(tmp_path / f"trace-{index}.npz")
    monkeypatch.setattr(gui.summary, "read_facts", slow(gui.summary.read_facts, 0.1))
    library = gui.Library([tmp_path])
    assert library.summary()["state"] == "summing"
    library.close()
    assert library._summing is not None and not library._summing.is_alive()
    assert len(library._facts) < 20
    assert library.summary()["state"] == "summing"
    assert not library._summing.is_alive()


def test_two_folders_of_the_same_name_stay_two_runs(tmp_path):
    for parent in ("a", "b"):
        (tmp_path / parent / "out").mkdir(parents=True)
        demos.transpose_trace().save(tmp_path / parent / "out" / "trace-0.npz")
    roots = [tmp_path / "a" / "out", tmp_path / "b" / "out"]
    answer = gui.Library(roots, background=False).summary()
    assert len(answer["runs"]) == 2
    assert [kernel["launches"] for kernel in answer["kernels"]] == [1, 1]


def test_an_id_through_a_linked_folder_outside_the_page_opens_nothing(tmp_path):
    root, outside = tmp_path / "root", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    demos.transpose_trace().save(outside / "secret.npz")
    (root / "link").symlink_to(outside, target_is_directory=True)
    library = gui.Library([root])
    assert library.listing()["traces"] == []
    with pytest.raises(gui.NotFound):
        library.load("0/link/secret.npz")
