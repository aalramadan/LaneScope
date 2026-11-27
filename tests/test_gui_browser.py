import json
import re

import pytest

from lanescope import demos, trace_reports
from lanescope.fixes import layouts
from lanescope.gui import server as gui

TRANSPOSE = demos.transpose_trace()

playwright = pytest.importorskip("playwright.sync_api")
expect = playwright.expect

# Each visible element whose own text is set in the given color, with the background
# color of the element and of each of its ancestors, innermost first.
TEXT_IN_COLOR = """(color) => {
  const found = [];
  for (const element of document.querySelectorAll("body *")) {
    const style = getComputedStyle(element);
    const own = element instanceof SVGElement ? style.fill : style.color;
    const texts = [...element.childNodes].filter(
      (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim(),
    );
    if (own !== color || texts.length === 0 || !element.checkVisibility()) {
      continue;
    }
    const layers = [];
    for (let node = element; node; node = node.parentElement) {
      layers.push(getComputedStyle(node).backgroundColor);
    }
    found.push({ text: element.textContent.trim(), layers });
  }
  return found;
}"""


def test_page_opens_on_the_worst_instruction_with_its_bank_conflict(page):
    selected = page.locator('#instruction-rows tr[aria-selected="true"]')
    assert selected.get_attribute("data-key") == "0x120/shared"
    caption = page.locator("figure.grid figcaption").first.inner_text()
    assert "32 wavefronts, ideal 1, 31 bank conflicts" in caption
    assert page.locator("figure.grid rect.cell.d4").count() == 32


def test_best_ranked_fix_removes_the_conflict_and_shows_its_code(page):
    page.locator("#suggestion-list button").first.click()
    page.wait_for_selector("figure.grid:nth-of-type(2)")
    after = page.locator("figure.grid figcaption").nth(1).inner_text()
    assert "1 wavefront, ideal 1, 0 bank conflicts" in after
    assert page.locator("#result .hero .figure").inner_text() == "0"
    assert "Swizzle<5, 0, 5>" in page.locator("#code-forms").inner_text()


def test_the_card_explains_the_conflict_and_its_fix_in_plain_words(page):
    card = page.locator("#diagnosis")
    expect(card.locator(".diagnosis-title")).to_have_text("Bank conflicts")
    expect(card.locator(".diagnosis-badge")).to_have_text("32× the passes it needs")
    expect(card).to_contain_text("up to 32 threads hit different words in the same bank")
    expect(card).to_contain_text("This line causes all the extra passes of the kernel.")
    fix = card.locator(".fix-line")
    expect(fix).to_contain_text("This removes all 3,968 extra passes on this buffer.")


def test_the_card_shows_its_fix_beside_the_recorded_access_and_hides_it_again(page):
    page.locator("#diagnosis .primary-button").click()
    after = page.locator("figure.grid figcaption").nth(1)
    expect(after).to_contain_text("With the fix")
    expect(after).to_contain_text("1 wavefront, ideal 1, 0 bank conflicts")
    button = page.locator("#diagnosis .primary-button")
    expect(button).to_have_text("Hide the fix")
    expect(button).to_have_attribute("aria-pressed", "true")
    button.click()
    expect(page.locator("figure.grid")).to_have_count(1)
    expect(page.locator("#diagnosis .primary-button")).to_have_text("Show this fix")
    expect(page.locator('input[name="kind"][value="none"]')).to_be_checked()


def test_a_clean_global_load_says_that_it_wastes_no_sector(page):
    page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
    card = page.locator("#diagnosis")
    expect(card.locator(".diagnosis-title")).to_have_text("No wasted sectors")
    expect(card).to_contain_text("The instructions with extra passes or sectors come first")
    expect(card.locator(".fix-line")).to_have_count(0)


def test_column_headers_and_counts_explain_their_terms_on_hover(page):
    excess = page.locator('table.instructions th[data-term="excess"]').get_attribute("title")
    assert excess.startswith("Passes or sectors above the fewest")
    whole_run = page.locator("#access-stats div").first.get_attribute("title")
    assert whole_run.startswith("A wavefront is one pass of shared memory.")


def predicted_excess(change):
    """Return the excess wavefronts of the demo transpose after a change, as the page shows it."""
    data = trace_reports.whatif(TRANSPOSE, layouts.Buffer(0, 4096), change)
    total = trace_reports.overview(TRANSPOSE)["totals"]["wavefronts"]["excessive"]
    for item in data["instructions"]:
        total += item["after"]["excessive"] - item["before"]["excessive"]
    return f"{total:,}"


def test_padding_controls_predict_a_pad_of_two_floats(page):
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    page.locator("#pad-row").select_option("128")
    page.locator("#pad-bytes").fill("8")
    expected = predicted_excess(layouts.Padding(128, 8))
    expect(page.locator("#result .hero .figure")).to_have_text(expected)
    assert expected not in ("0", "3,968")
    assert "The buffer grows by 256 B." in page.locator("#result").inner_text()
    expect(page.locator("#code-forms")).to_contain_text("tile[32][32 + 2]")


def test_swizzle_sliders_change_the_prediction(page):
    # The first switch to a kind starts from the best ranked fix of that kind.
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    page.locator('input[name="kind"][value="swizzle"]').check(force=True)
    expect(page.locator("#result .hero .figure")).to_have_text("0")
    page.locator("#swizzle-bits").focus()
    page.keyboard.press("ArrowLeft")
    page.keyboard.press("ArrowLeft")
    expect(page.locator("#swizzle-bits-value")).to_have_text("3")
    expected = predicted_excess(layouts.XorSwizzle(3, 2, 5))
    expect(page.locator("#result .hero .figure")).to_have_text(expected)
    expect(page.locator("#code-forms")).to_contain_text("Swizzle<3, 0, 5>")


def test_arrow_keys_move_through_the_instruction_table(page):
    page.locator('#instruction-rows tr[aria-selected="true"]').focus()
    page.keyboard.press("ArrowDown")
    expect(page.locator("#lanes-title")).to_have_text(re.compile(r"^0xa0"))
    assert "global load" in page.locator("#lanes-subtitle").inner_text()
    assert page.locator("#fix-form").is_hidden()


def test_wide_loads_draw_one_grid_per_phase(page):
    page.locator("#trace-select").select_option("0/wide.npz")
    expect(page.locator(".phase-title")).to_have_count(4)
    titles = page.locator(".phase-title").all_inner_texts()
    assert titles[0].startswith("Phase 1, lanes 0-7")


def test_check_folder_marks_each_count_that_nsight_compute_confirms(page):
    page.locator("#trace-select").select_option("0/check/record/trace-0.npz")
    expect(page.locator("#totals")).to_contain_text("1 of 3")
    confirmed = page.locator('#instruction-rows tr[data-key="0x120/shared"] .measured')
    contradicted = page.locator('#instruction-rows tr[data-key="0xa0/global"] .measured')
    expect(confirmed).to_have_text("\u2713")
    expect(contradicted).to_have_text("\u2717")
    assert contradicted.get_attribute("title") == "Nsight Compute measures 100"
    missing = page.locator('#instruction-rows tr[data-key="0xb0/shared"] .measured')
    expect(missing).to_have_text("?")


def test_pattern_strip_and_source_lines_select_what_they_show(page):
    page.locator("#patterns button").nth(1).click()
    expect(page.locator("#pattern-detail")).to_have_text(re.compile(r"^Pattern 2:"))
    page.locator('#source li[data-line="5"] button').click()
    expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5")
    page.locator('#source li[data-line="5"] button').click()
    expect(page.locator("#lanes-title")).to_have_text("0xb0, line 5")


def test_hovering_a_cell_names_its_lane_and_bank(page):
    page.locator("figure.grid rect.cell").nth(3).hover()
    expect(page.locator("#tooltip")).to_be_visible()
    expect(page.locator("#tooltip")).to_contain_text("Lane 3")
    expect(page.locator("#tooltip")).to_contain_text("bank 0, byte offset 0x180")


def test_a_trace_without_accesses_clears_the_previous_instruction(page):
    page.locator("#trace-select").select_option("0/empty.npz")
    expect(page.locator("#lanes-title")).to_have_text("No memory accesses")
    expect(page.locator("figure.grid")).to_have_count(0)
    expect(page.locator("#fix-form")).to_be_hidden()
    expect(page.locator("#fix-note")).to_contain_text("no memory instruction")


def test_a_source_line_counts_each_unit_on_its_own(page):
    page.locator("#trace-select").select_option("0/mixed.npz")
    badge = page.locator('#source li[data-line="1"] .badge')
    expect(badge).to_have_text("31 excess wavefronts, 28 excess sectors")
    code = page.locator('#source li[data-line="1"] .text')
    badge_box, code_box = badge.bounding_box(), code.bounding_box()
    assert badge_box and code_box
    assert badge_box["y"] >= code_box["y"] + code_box["height"] - 1


def test_a_listing_cut_at_its_limit_says_so_in_the_picker(browser, tmp_path, serve, monkeypatch):
    for name in ("a", "b", "c"):
        demos.transpose_trace(tiles_per_side=1).save(tmp_path / f"{name}.npz")
    monkeypatch.setattr(gui, "MAX_LISTED", 2)
    with serve([tmp_path]) as server:
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"http://localhost:{server.server_address[1]}/")
        last = page.locator("#trace-select option").last
        expect(last).to_have_text("Only the first 2 trace files are listed")
        assert last.get_attribute("disabled") is not None
        context.close()


def test_a_slider_snapped_by_a_new_buffer_sends_its_new_value(page):
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    page.locator('input[name="kind"][value="padding"]').check(force=True)
    expect(page.locator("#result .hero .figure")).to_have_text("0")
    bodies = []
    page.on(
        "request",
        lambda request: request.method == "POST" and bodies.append(json.loads(request.post_data)),
    )
    page.locator("#trace-select").select_option("0/wide.npz")
    expect(page.locator("#lanes-title")).to_have_text("0x40")
    expect(page.locator("#result")).to_be_visible()
    page.wait_for_timeout(300)
    assert bodies
    sent = bodies[-1]["change"]["pad_bytes"]
    assert sent == int(page.locator("#pad-bytes").input_value())
    assert sent % 16 == 0


@pytest.mark.parametrize(
    ("size", "reason"),
    [("4090", "cross the edge of the buffer"), ("2048 B", "size in bytes, such as 4096")],
)
def test_a_refused_buffer_names_the_problem_and_puts_the_fields_back(page, size, reason):
    page.locator("#buffer-size").fill(size)
    page.locator("#buffer-size").press("Enter")
    expect(page.locator("#result .error")).to_contain_text(reason)
    assert page.locator("#buffer-size").input_value() == "4096"
    assert page.locator("#buffer-start").input_value() == "0x0"


def test_a_ranked_swizzle_after_one_with_more_bits_keeps_its_own_shift(page):
    fixes = page.locator("#suggestion-list button")
    fixes.filter(has_text="Swizzle<5,2,5>").click()
    expect(page.locator("#code-forms")).to_contain_text("Swizzle<5, 0, 5>")
    fixes.filter(has_text="Swizzle<4,3,4>").click()
    expect(page.locator("#code-forms")).to_contain_text("Swizzle<4, 1, 4>")
    sliders = [page.locator(f"#swizzle-{name}").input_value() for name in ("bits", "base", "shift")]
    assert sliders == ["4", "3", "4"]
    expected = predicted_excess(layouts.XorSwizzle(4, 3, 4))
    expect(page.locator("#result .hero .figure")).to_have_text(expected)


def test_enter_on_a_pattern_keeps_the_focus_on_that_pattern(page):
    patterns = page.locator("#patterns button")
    patterns.nth(2).focus()
    page.keyboard.press("Enter")
    expect(page.locator("#pattern-detail")).to_have_text(re.compile(r"^Pattern 3:"))
    expect(patterns.nth(2)).to_be_focused()
    page.keyboard.press("Tab")
    expect(patterns.nth(3)).to_be_focused()


def rgb(color):
    """Return the red, green, blue, and alpha of a computed CSS color, each from 0 to 1."""
    if color.startswith("#"):
        return *(int(color[index : index + 2], 16) / 255 for index in (1, 3, 5)), 1.0
    numbers = [float(number) for number in re.findall(r"[\d.]+", color)]
    scale = 1 if color.startswith("color(") else 255
    alpha = numbers[3] if len(numbers) > 3 else 1.0
    return *(number / scale for number in numbers[:3]), alpha


def luminance(color):
    """Return the relative luminance of an opaque color, as WCAG 2 defines it."""
    linear = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in color[:3]]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def backdrop(layers):
    """Return the color behind a text, from the background layers of its element and ancestors."""
    red, green, blue = 1.0, 1.0, 1.0
    for layer in reversed(layers):
        *channels, alpha = rgb(layer)
        red, green, blue = (
            alpha * new + (1 - alpha) * old
            for new, old in zip(channels, (red, green, blue), strict=True)
        )
    return red, green, blue


@pytest.mark.parametrize(
    ("scheme", "clicks"),
    [("light", 0), ("dark", 0), ("light", 2)],
    ids=["light", "dark", "dark-theme-button"],
)
def test_secondary_text_keeps_a_contrast_of_at_least_four_and_a_half(browser, url, scheme, clicks):
    context = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme=scheme)
    page = context.new_page()
    page.goto(f"{url}#trace=0/transpose.npz")
    for _ in range(clicks):
        page.locator("#theme-toggle").click()
    page.locator("#suggestion-list button").first.click()
    expect(page.locator("#result .hero .figure")).to_have_text("0")
    muted = page.evaluate(
        "() => getComputedStyle(document.documentElement).getPropertyValue('--muted').trim()"
    )
    color = "rgb({}, {}, {})".format(*(round(255 * channel) for channel in rgb(muted)[:3]))
    texts = page.evaluate(TEXT_IN_COLOR, color)
    context.close()
    assert len(texts) > 20
    ink = luminance(rgb(color))
    low = []
    for item in texts:
        ground = luminance(backdrop(item["layers"]))
        lighter, darker = max(ink, ground), min(ink, ground)
        ratio = (lighter + 0.05) / (darker + 0.05)
        if ratio < 4.5:
            low.append((item["text"], round(ratio, 2)))
    assert low == []


# Each visible text of the diagnosis card, with its color and the background layers behind it.
CARD_TEXTS = """() => [...document.querySelectorAll("#diagnosis *")]
  .filter((element) => [...element.childNodes].some(
    (node) => node.nodeType === Node.TEXT_NODE && node.textContent.trim()))
  .map((element) => {
    const layers = [];
    for (let node = element; node; node = node.parentElement) {
      layers.push(getComputedStyle(node).backgroundColor);
    }
    return { text: element.textContent.trim(), color: getComputedStyle(element).color, layers };
  })"""


@pytest.mark.parametrize("scheme", ["light", "dark"])
@pytest.mark.parametrize("key", ["0x120/shared", "0xa0/global"], ids=["conflict", "clean"])
def test_the_diagnosis_card_keeps_a_contrast_of_at_least_four_and_a_half(browser, url, scheme, key):
    context = browser.new_context(viewport={"width": 1440, "height": 900}, color_scheme=scheme)
    page = context.new_page()
    page.goto(f"{url}#trace=0/transpose.npz")
    page.locator(f'#instruction-rows tr[data-key="{key}"]').click()
    expect(page.locator("#diagnosis .diagnosis-title")).to_be_visible()
    if key.endswith("shared"):
        expect(page.locator("#diagnosis .primary-button")).to_be_visible()
    texts = page.evaluate(CARD_TEXTS)
    context.close()
    low = []
    for item in texts:
        ink = luminance(rgb(item["color"]))
        ground = luminance(backdrop(item["layers"]))
        ratio = (max(ink, ground) + 0.05) / (min(ink, ground) + 0.05)
        if ratio < 4.5:
            low.append((item["text"], round(ratio, 2)))
    assert texts
    assert low == []


def test_each_source_line_carries_its_full_text_for_hover(page):
    lines = page.locator("#source .text")
    assert lines.count() == len(demos.TRANSPOSE_SOURCE)
    for index in range(lines.count()):
        assert lines.nth(index).get_attribute("title") == demos.TRANSPOSE_SOURCE[index]


def test_access_labels_fit_their_column(page):
    labels = page.locator("#instruction-rows .access-cell .what")
    expect(labels).to_have_count(4)
    assert "shared store" in labels.all_inner_texts()
    for index in range(labels.count()):
        label = labels.nth(index)
        width, room = label.evaluate("(element) => [element.scrollWidth, element.clientWidth]")
        assert width <= room, label.inner_text()


def test_the_top_bar_never_covers_the_focused_control_on_a_narrow_window(browser, url):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.goto(f"{url}#trace=0/transpose.npz")
    fixes = page.locator("#suggestion-list button")
    expect(fixes).to_have_count(gui.SUGGESTIONS)
    fixes.last.focus()
    bar = page.locator(".topbar")
    covered = []
    for _ in range(16):
        page.keyboard.press("Shift+Tab")
        focused = page.locator(":focus")
        box, bar_box = focused.bounding_box(), bar.bounding_box()
        assert box and bar_box
        if box["y"] < bar_box["y"] + bar_box["height"]:
            covered.append(focused.evaluate("(element) => element.outerHTML.slice(0, 60)"))
    context.close()
    assert covered == []


def test_hovering_a_wide_load_names_the_offset_of_each_word(page):
    page.locator("#trace-select").select_option("0/wide.npz")
    expect(page.locator(".phase-title")).to_have_count(4)
    # Lane 1 reads the 16 bytes at 0x40, one word in each of banks 16 to 19.
    page.locator('figure.grid rect.cell[data-lane="1"][data-bank="17"]').hover()
    expect(page.locator("#tooltip")).to_contain_text("bank 17, byte offset 0x44")


def test_hovering_a_byte_load_names_the_byte_that_each_lane_reads(page):
    page.locator("#trace-select").select_option("0/bytes.npz")
    expect(page.locator("#lanes-subtitle")).to_have_text("shared load, 1 B")
    # Lane 1 reads the byte at 0x41, inside the word at 0x40 in bank 16.
    page.locator("figure.grid").first.locator('rect.cell[data-lane="1"]').hover()
    expect(page.locator("#tooltip")).to_contain_text("bank 16, byte offset 0x41")


def test_hovering_a_sector_names_its_offset_from_the_first_line(page):
    page.locator("#trace-select").select_option("0/mixed.npz")
    page.locator('#instruction-rows tr[data-key="0x20/global"]').click()
    expect(page.locator("figure.grid figcaption").first).to_contain_text("32 sectors")
    # Lane 31 reads 4096 + 128 * 31, which lies 0xf80 bytes past the first line at 4096.
    page.locator('figure.grid rect.cell[data-lane="31"]').hover()
    expect(page.locator("#tooltip")).to_contain_text("sector at +0xf80")
    page.locator("figure.grid path.bar").last.hover()
    expect(page.locator("#tooltip")).to_contain_text("Sector at +0xf80")


@pytest.mark.parametrize(
    ("field", "value"), [("#buffer-size", "1048577"), ("#buffer-start", "0x100000")]
)
def test_a_buffer_past_the_shared_window_is_refused_by_the_page_itself(page, field, value):
    messages, checks = [], []
    page.on("console", lambda message: messages.append(message.text))
    page.on("request", lambda request: "/api/buffer" in request.url and checks.append(request.url))
    page.locator(field).fill(value)
    page.locator(field).press("Enter")
    expect(page.locator("#result .error")).to_contain_text("must end at or below 0x100000")
    assert page.locator("#buffer-start").input_value() == "0x0"
    assert page.locator("#buffer-size").input_value() == "4096"
    page.wait_for_timeout(200)
    assert (checks, messages) == ([], [])


def test_grid_titles_stay_on_one_line_beside_a_long_count(browser, url):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    page.goto(f"{url}#trace=0/transpose.npz")
    expect(page.locator("#suggestion-list li").first).to_be_visible()
    page.locator('input[name="kind"][value="swizzle"]').check(force=True)
    page.locator("#swizzle-bits").focus()
    page.keyboard.press("Home")
    caption = page.locator("figure.grid figcaption").nth(1)
    expect(caption).to_contain_text("16 wavefronts, ideal 1, 15 bank conflicts")
    title = caption.locator("strong").bounding_box()
    assert title is not None
    assert title["height"] < 20
    page.close()


def test_a_ranked_padding_names_the_global_sectors_it_adds_to_a_copy(browser, tmp_path, serve):
    demos.matmul_tile_trace().save(tmp_path / "matmul_tile.npz")
    with serve([tmp_path]) as server:
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        page.goto(f"http://localhost:{server.server_address[1]}/")
        fixes = page.locator("#suggestion-list button")
        expect(fixes.first).to_be_visible()
        expect(fixes.first).not_to_contain_text("sectors")
        padding = fixes.filter(has_text="pad each 128-byte row by 16 bytes")
        expect(padding).to_contain_text("adds 1,024 B, 256 more excess sectors")
        context.close()


@pytest.fixture(scope="module")
def summary_url(tmp_path_factory, serve):
    """Return the address of a page over one recording of four launches."""
    folder = tmp_path_factory.mktemp("summary")
    run = folder / "run1"
    run.mkdir()
    demos.transpose_trace().save(run / "trace-0.npz")
    demos.transpose_trace(tiles_per_side=1).save(run / "trace-1.npz")
    demos.histogram_trace().save(run / "trace-2.npz")
    demos.column_sums_trace().save(run / "trace-3.npz")
    with serve([folder]) as server:
        yield f"http://localhost:{server.server_address[1]}/"


@pytest.fixture
def summary_page(browser, summary_url):
    """Return the page on the summary, and fail the test on any error of its scripts."""
    context = browser.new_context(viewport={"width": 1600, "height": 1000})
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(summary_url)
    page.wait_for_selector("#summary-lines tr")
    yield page
    context.close()
    assert errors == []


def test_a_page_of_many_launches_opens_on_their_summary_worst_line_first(summary_page):
    expect(summary_page.locator("#summary-view")).to_be_visible()
    expect(summary_page.locator("#layout")).to_be_hidden()
    expect(summary_page.locator("#summary-subtitle")).to_have_text(
        re.compile(r"/run1: 4 launches of 3 kernels$")
    )
    first = summary_page.locator("#summary-lines tr").first
    expect(first).to_contain_text("Line 10")
    expect(first).to_contain_text("shared load, 4 B")
    expect(first).to_contain_text("transpose")
    expect(first.locator("td").nth(4)).to_have_text("2")
    assert summary_page.locator("#trace-select").input_value() == ""


def test_a_line_of_the_summary_opens_its_worst_launch_and_back_returns(summary_page):
    summary_page.locator("#summary-lines tr").first.click()
    summary_page.wait_for_selector("figure.grid svg")
    expect(summary_page.locator("#summary-view")).to_be_hidden()
    selected = summary_page.locator('#instruction-rows tr[aria-selected="true"]')
    expect(selected).to_have_attribute("data-key", "0x120/shared")
    assert summary_page.locator("#trace-select").input_value() == "0/run1/trace-0.npz"
    summary_page.go_back()
    expect(summary_page.locator("#summary-view")).to_be_visible()
    expect(summary_page.locator("#layout")).to_be_hidden()


def test_the_summary_button_returns_from_a_launch_to_the_summary(summary_page):
    summary_page.locator("#summary-lines tr").first.press("Enter")
    summary_page.wait_for_selector("figure.grid svg")
    button = summary_page.locator("#summary-button")
    expect(button).to_have_attribute("aria-pressed", "false")
    button.click()
    expect(summary_page.locator("#summary-view")).to_be_visible()
    expect(button).to_have_attribute("aria-pressed", "true")
    assert summary_page.evaluate("location.hash") == ""


def test_the_summary_ranks_global_memory_on_request(summary_page):
    summary_page.locator('input[name="summary-unit"][value="sectors"]').check(force=True)
    expect(summary_page.locator("#summary-lines-title")).to_have_text(
        "Lines with the most extra sectors"
    )
    expect(summary_page.locator("#summary-lines tr").first).to_contain_text("column_sums")


def test_a_line_that_follows_the_input_values_says_so_and_offers_no_fix(summary_page):
    row = summary_page.locator("#summary-lines tr", has_text="follows the input")
    expect(row).to_contain_text("histogram")
    row.click()
    summary_page.wait_for_selector("figure.grid svg")
    card = summary_page.locator("#diagnosis")
    expect(card).to_contain_text("follow the values it reads, such as the bins of a histogram")
    expect(card.locator(".fix-line")).to_contain_text(
        "No layout of the buffer removes these conflicts."
    )
    expect(card.locator(".primary-button")).to_have_count(0)


def test_a_template_row_lists_its_instances_and_each_opens_its_worst_launch(
    browser, tmp_path, serve
):
    for index, (tiles, kernel) in enumerate(
        [(1, "void ns::tile<4>(float*)"), (2, "void ns::tile<8>(float*)")]
    ):
        trace = demos.transpose_trace(tiles_per_side=tiles)
        trace.metadata["kernel"] = kernel
        trace.save(tmp_path / f"trace-{index}.npz")
    with serve([tmp_path]) as server:
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(f"http://localhost:{server.server_address[1]}/")
        page.wait_for_selector("#summary-kernels tr")
        expect(page.locator("#summary-subtitle")).to_contain_text("2 kernels in 1 template")
        expect(page.locator("#summary-lines tr").first).to_contain_text("2 instances")
        rows = page.locator("#summary-kernels tr")
        expect(rows.first).to_contain_text("tile")
        instances = page.locator("#summary-kernels tr.instance")
        expect(instances).to_have_count(2)
        expect(instances.first).to_be_hidden()
        toggle = rows.first.locator(".instances-toggle")
        expect(toggle).to_have_text("2 instances")
        toggle.focus()
        toggle.press("Enter")
        expect(toggle).to_have_attribute("aria-expanded", "true")
        expect(page.locator("#summary-view")).to_be_visible()
        expect(instances.nth(0)).to_contain_text("tile<8>")
        expect(instances.nth(1)).to_contain_text("tile<4>")
        page.locator('input[name="summary-unit"][value="sectors"]').check(force=True)
        page.locator('input[name="summary-unit"][value="wavefronts"]').check(force=True)
        expect(page.locator("#summary-kernels tr.instance").first).to_be_visible()
        page.locator("#summary-kernels tr.instance").nth(1).press("Enter")
        page.wait_for_selector("figure.grid svg")
        assert page.locator("#trace-select").input_value() == "0/trace-0.npz"
        context.close()
    assert errors == []


def test_each_excess_names_its_part_of_all_the_passes_of_the_kernel(page):
    selected = page.locator('#instruction-rows tr[aria-selected="true"] td').nth(3)
    expect(selected).to_contain_text("% of all")
    expect(page.locator("#diagnosis")).to_contain_text(
        "Removing them would cut the shared-memory passes of the kernel by"
    )


def test_a_page_of_one_launch_opens_on_it_without_a_summary_button(browser, tmp_path, serve):
    demos.transpose_trace().save(tmp_path / "only.npz")
    with serve([tmp_path]) as server:
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"http://localhost:{server.server_address[1]}/")
        page.wait_for_selector("figure.grid svg")
        expect(page.locator("#summary-button")).to_be_hidden()
        expect(page.locator("#summary-view")).to_be_hidden()
        context.close()


def test_the_summary_shows_its_progress_while_the_server_reads_the_launches(
    browser, tmp_path, serve, monkeypatch, slow
):
    for index in range(3):
        demos.transpose_trace(tiles_per_side=1).save(tmp_path / f"trace-{index}.npz")
    monkeypatch.setattr(gui.summary, "read_facts", slow(gui.summary.read_facts, 0.4))
    with serve([tmp_path]) as server:
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"http://localhost:{server.server_address[1]}/")
        expect(page.locator("#summary-progress-text")).to_contain_text("of 3 launches done")
        page.wait_for_selector("#summary-lines tr", timeout=20000)
        expect(page.locator("#summary-progress")).to_be_hidden()
        context.close()


def test_a_summary_of_wasted_sectors_alone_opens_on_global_memory(browser, tmp_path, serve):
    demos.column_sums_trace().save(tmp_path / "trace-0.npz")
    demos.column_sums_trace(blocks=1).save(tmp_path / "trace-1.npz")
    with serve([tmp_path]) as server:
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"http://localhost:{server.server_address[1]}/")
        page.wait_for_selector("#summary-lines tr")
        expect(page.locator('input[name="summary-unit"][value="sectors"]')).to_be_checked()
        expect(page.locator("#summary-lines tr").first).to_contain_text("column_sums")
        context.close()


def test_back_from_a_launch_picked_on_the_summary_returns_to_the_summary(summary_page):
    summary_page.select_option("#trace-select", "0/run1/trace-2.npz")
    summary_page.wait_for_selector("figure.grid svg")
    expect(summary_page.locator("#summary-view")).to_be_hidden()
    summary_page.go_back()
    expect(summary_page.locator("#summary-view")).to_be_visible()


def test_back_from_the_summary_button_returns_to_the_launch(summary_page):
    summary_page.locator("#summary-lines tr").first.click()
    summary_page.wait_for_selector("figure.grid svg")
    summary_page.locator("#summary-button").click()
    expect(summary_page.locator("#summary-view")).to_be_visible()
    summary_page.go_back()
    expect(summary_page.locator("#layout")).to_be_visible()
    selected = summary_page.locator('#instruction-rows tr[aria-selected="true"]')
    expect(selected).to_have_attribute("data-key", "0x120/shared")


def test_a_launch_that_fails_to_open_leaves_the_picker_on_the_summary(browser, tmp_path, serve):
    for index in range(2):
        demos.transpose_trace(tiles_per_side=1).save(tmp_path / f"trace-{index}.npz")
    with serve([tmp_path]) as server:
        context = browser.new_context()
        page = context.new_page()
        page.goto(f"http://localhost:{server.server_address[1]}/")
        page.wait_for_selector("#summary-lines tr")
        (tmp_path / "trace-1.npz").unlink()
        page.select_option("#trace-select", "0/trace-1.npz")
        expect(page.locator("#banner")).to_be_visible()
        expect(page.locator("#summary-view")).to_be_visible()
        assert page.locator("#trace-select").input_value() == ""
        context.close()


def test_the_card_names_the_header_line_of_an_inlined_instruction(
    browser, tmp_path, serve, inlined_transpose
):
    inlined_transpose.save(tmp_path / "inlined.npz")
    with serve([tmp_path]) as server:
        context = browser.new_context(viewport={"width": 1600, "height": 1000})
        page = context.new_page()
        page.goto(f"http://localhost:{server.server_address[1]}/")
        origin = page.locator("#lanes-origin")
        expect(origin).to_have_text("inlined from tile_copy.cuh:3: return tile[c][r];")
        expect(origin).to_have_attribute("title", "/src/include/tile_copy.cuh")
        page.locator('#instruction-rows tr[data-key="0xa0/global"]').click()
        expect(page.locator("#lanes-title")).to_have_text("0xa0, line 5")
        expect(origin).to_be_hidden()
        context.close()
