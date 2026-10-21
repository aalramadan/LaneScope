import re
import time

import pytest

from lanescope import analysis
from lanescope.gui import server as gui

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect


def title_matches_grid(page):
    """Return whether the drawn grid belongs to the memory space that the subtitle names."""
    subtitle = page.locator("#lanes-subtitle").inner_text()
    caption = page.locator("figure.grid figcaption").first.inner_text()
    return ("global" in subtitle) == ("sector" in caption)


def test_a_slow_access_of_a_left_instruction_never_lands(page, monkeypatch, slow):
    monkeypatch.setattr(analysis, "patterns", slow(analysis.patterns, 0.4))
    monkeypatch.setattr(analysis, "access", slow(analysis.access, 0.8))
    page.locator('#instruction-rows tr[data-key="0xb0/shared"]').click()
    page.wait_for_timeout(900)
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    # The access of 0xb0 lands now, while the patterns of 0xa0 still load.
    page.wait_for_timeout(600)
    assert title_matches_grid(page)
    expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5")
    assert title_matches_grid(page)


def test_pattern_buttons_of_a_left_instruction_do_nothing(page, monkeypatch, slow):
    monkeypatch.setattr(analysis, "access", slow(analysis.access, 0.6))
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    page.wait_for_timeout(150)
    # The buttons still show the patterns of 0x120 while the access of 0xa0 runs.
    page.locator("#patterns button").nth(1).click()
    page.wait_for_timeout(1500)
    expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5")
    assert title_matches_grid(page)
    expect(page.locator("#fix-form")).to_be_hidden()


def test_fix_controls_moved_while_an_instruction_loads_send_nothing(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    expect(page.locator("#result .hero .figure")).to_have_text("0")
    posts = []
    page.on("request", lambda request: request.method == "POST" and posts.append(request))
    monkeypatch.setattr(analysis, "patterns", slow(analysis.patterns, 0.6))
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    page.locator("#pad-bytes").fill("8")
    expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5")
    page.wait_for_timeout(500)
    assert posts == []
    expect(page.locator("#result")).to_be_hidden()
    expect(page.locator("#code")).to_be_hidden()


def test_a_prediction_from_before_a_trace_switch_never_lands(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    monkeypatch.setattr(analysis, "whatif", slow(analysis.whatif, 0.8))
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    page.wait_for_timeout(200)
    # A trace without accesses starts no prediction of its own that could cover the old one.
    page.locator("#trace-select").select_option("0/empty.npz")
    expect(page.locator("#lanes-title")).to_have_text("No memory accesses")
    page.wait_for_timeout(1200)
    expect(page.locator("#result")).to_be_hidden()


def test_an_abort_while_the_answer_arrives_shows_no_error(page, monkeypatch):
    send = gui.Handler._send

    def slow_body(handler, status, kind, payload):
        if not handler.path.startswith("/api/patterns"):
            send(handler, status, kind, payload)
            return
        handler.send_response(status)
        handler.send_header("Content-Type", kind)
        handler.send_header("Content-Length", str(len(payload)))
        handler.end_headers()
        handler.wfile.flush()
        time.sleep(0.8)
        try:
            handler.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            handler.close_connection = True

    monkeypatch.setattr(gui.Handler, "_send", slow_body)
    page.locator('#instruction-rows tr[data-key="0xb0/shared"]').click()
    page.wait_for_timeout(300)
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5", timeout=5000)
    page.wait_for_timeout(500)
    expect(page.locator("#banner")).to_be_hidden()


def test_the_grid_stays_busy_until_the_newer_of_two_accesses_answers(page, monkeypatch, slow):
    monkeypatch.setattr(analysis, "access", slow(analysis.access, 0.8))
    page.locator("#patterns button").nth(1).click()
    page.wait_for_timeout(100)
    page.locator("#patterns button").nth(2).click()
    page.wait_for_timeout(400)
    expect(page.locator("#grids")).to_have_class(re.compile("busy"))
    expect(page.locator("#pattern-detail")).to_have_text(re.compile(r"^Pattern 3:"))
    expect(page.locator("#grids")).not_to_have_class(re.compile("busy"))


def test_a_trace_that_loads_slowly_still_opens_after_a_click_on_the_old_one(page, monkeypatch):
    load = gui.Library.load

    def slow_mixed(library, trace_id):
        if trace_id == "0/mixed.npz":
            time.sleep(1.0)
        return load(library, trace_id)

    monkeypatch.setattr(gui.Library, "load", slow_mixed)
    page.locator("#trace-select").select_option("0/mixed.npz")
    page.locator('#instruction-rows tr[data-key="0xb0/shared"]').click()
    expect(page.locator("#kernel-info")).to_contain_text("mixed", timeout=5000)
    expect(page.locator("#lanes-title")).to_have_text(re.compile(r"^0x(10|20), line 1"))
    assert page.locator("#trace-select").input_value() == "0/mixed.npz"


def test_a_trace_that_fails_to_open_cancels_nothing_of_the_one_on_screen(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    expect(page.locator("#result .hero .figure")).to_have_text("0")
    monkeypatch.setattr(analysis, "whatif", slow(analysis.whatif, 0.8))
    page.locator("#pad-bytes").fill("8")
    page.wait_for_timeout(200)
    page.locator("#trace-select").select_option("0/broken.npz")
    expect(page.locator("#banner")).to_contain_text("no .npz trace file")
    expect(page.locator("#result")).to_contain_text("The buffer grows by 256 B.", timeout=5000)
    assert page.locator("#trace-select").input_value() == "0/transpose.npz"


def test_a_slow_search_never_fills_the_panel_of_another_instruction(page, monkeypatch, slow):
    monkeypatch.setattr(analysis, "suggest", slow(analysis.suggest, 1.0))
    page.locator("#buffer-size").fill("2048")
    page.locator("#buffer-size").press("Enter")
    expect(page.locator("#suggestions-note")).to_contain_text("Searching")
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    page.wait_for_timeout(1500)
    expect(page.locator("#fix-note")).to_contain_text("shared memory layouts only")
    expect(page.locator("#suggestions")).to_be_hidden()


def test_none_drops_a_prediction_that_is_still_running(page, monkeypatch, slow):
    monkeypatch.setattr(analysis, "whatif", slow(analysis.whatif, 0.8))
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    page.wait_for_timeout(200)
    page.locator('input[name="kind"][value="none"]').check(force=True)
    page.wait_for_timeout(1200)
    expect(page.locator("#result")).to_be_hidden()
    expect(page.locator("figure.grid")).to_have_count(1)


def test_fix_controls_moved_while_a_shared_instruction_loads_send_nothing(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    expect(page.locator("#result .hero .figure")).to_have_text("0")
    posts = []
    page.on("request", lambda request: request.method == "POST" and posts.append(request))
    monkeypatch.setattr(analysis, "patterns", slow(analysis.patterns, 0.8))
    page.locator('#instruction-rows tr[data-key="0xb0/shared"]').click()
    page.locator("#pad-bytes").fill("8")
    page.wait_for_timeout(500)
    assert posts == []
    expect(page.locator("#lanes-title")).to_have_text("0xb0, line 5")


def test_a_ranked_fix_clicked_while_an_instruction_loads_sends_nothing(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    posts = []
    page.on("request", lambda request: request.method == "POST" and posts.append(request))
    monkeypatch.setattr(analysis, "patterns", slow(analysis.patterns, 0.8))
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    page.locator("#suggestion-list button").first.click()
    expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5")
    page.wait_for_timeout(300)
    assert posts == []
    expect(page.locator("#result")).to_be_hidden()


def test_a_buffer_typed_while_an_instruction_loads_opens_nothing(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    monkeypatch.setattr(analysis, "patterns", slow(analysis.patterns, 0.4))
    monkeypatch.setattr(gui, "_buffer_facts", slow(gui._buffer_facts, 1.2))
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    page.locator("#buffer-size").fill("2048")
    page.locator("#buffer-size").press("Enter")
    # The check of the typed buffer answers once 0xa0 is on screen.
    expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5")
    page.wait_for_timeout(1500)
    expect(page.locator("#suggestions")).to_be_hidden()
    expect(page.locator("#result")).to_be_hidden()


def test_the_controls_keep_working_while_the_buffer_is_inferred_again(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    expect(page.locator("#result .hero .figure")).to_have_text("0")
    monkeypatch.setattr(gui, "_buffer_facts", slow(gui._buffer_facts, 0.8))
    page.locator("#buffer-infer").click()
    page.locator("#buffer-size").fill("4090")
    page.locator("#buffer-size").press("Enter")
    expect(page.locator("#result .error")).to_contain_text("cross the edge of the buffer")
    expect(page.locator("#banner")).to_be_hidden()
    posts = []
    page.on("request", lambda request: request.method == "POST" and posts.append(request))
    page.locator("#pad-bytes").fill("8")
    expect(page.locator("#result")).to_contain_text("The buffer grows by 256 B.")
    assert posts


def test_an_answer_for_the_start_leaves_the_bytes_the_user_is_typing(page, monkeypatch, slow):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    monkeypatch.setattr(gui, "_buffer_facts", slow(gui._buffer_facts, 0.4))
    page.locator("#buffer-start").fill("0x80")
    page.locator("#buffer-start").press("Tab")
    expect(page.locator("#buffer-size")).to_be_focused()
    page.keyboard.press("Control+a")
    page.keyboard.type("20")
    # The answer for the start lands while the size is half typed.
    page.wait_for_timeout(900)
    page.keyboard.type("48")
    page.keyboard.press("Enter")
    expect(page.locator("#buffer-note")).to_contain_text("2,048 B of 4-byte elements")
    assert page.locator("#buffer-start").input_value() == "0x80"
    assert page.locator("#buffer-size").input_value() == "2048"
