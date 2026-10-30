import json
import re

import pytest

from lanescope import analysis, demos
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


def predicted_excess(change):
    """Return the excess wavefronts of the demo transpose after a change, as the page shows it."""
    data = analysis.whatif(TRANSPOSE, layouts.Buffer(0, 4096), change)
    total = analysis.overview(TRANSPOSE)["totals"]["wavefronts"]["excessive"]
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
