import re
import time
from xml.etree import ElementTree

import numpy as np
import pytest

from lanescope import demos, html_report
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace


def svgs(page):
    return re.findall(r"<svg xmlns=.*?</svg>", page, flags=re.S)


def test_table_lists_every_instruction():
    page = html_report.render(demos.transpose_trace())
    table = page.split("<table>")[1].split("</table>")[0]
    assert table.count("<tr><td>") == 4


def test_only_instructions_above_their_ideal_get_a_grid():
    page = html_report.render(demos.transpose_trace())
    assert re.findall(r'<section id="pc-([0-9a-f]+)-(\w+)"', page) == [("120", "shared")]


def test_column_conflict_fills_bank_0_with_the_darkest_step():
    page = html_report.render(demos.transpose_trace())
    (grid,) = svgs(page)
    assert grid.count('class="cell d4"') == 32
    assert grid.count('class="cell') == 32


def test_every_grid_is_well_formed_svg():
    trace = demos.transpose_trace()
    accesses = [trace.access(row) for row in range(len(trace))]
    accesses.append(
        Access(0x400, Space.GLOBAL, Kind.LOAD, 4, tuple(8 * lane for lane in range(WARP_SIZE)))
    )
    for grid in svgs(html_report.render(Trace.from_accesses(accesses, trace.metadata))):
        ElementTree.fromstring(grid)


def test_source_text_is_escaped():
    trace = Trace.from_accesses(
        [Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(WARP_SIZE)))],
        {"source": ["if (a < b && c > d) x = s[i];"], "lines": {"0x10": 1}},
    )
    page = html_report.render(trace)
    assert "a &lt; b &amp;&amp; c &gt; d" in page
    assert "a < b" not in page


def test_trace_without_excess_says_so():
    trace = Trace.from_accesses(
        [Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(4 * lane for lane in range(WARP_SIZE)))]
    )
    assert "No instruction exceeds its ideal count." in html_report.render(trace)


def test_instructions_past_the_limit_are_named():
    accesses = [
        Access(pc, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(WARP_SIZE)))
        for pc in (0x10, 0x20, 0x30)
    ]
    page = html_report.render(Trace.from_accesses(accesses), limit=1)
    assert len(re.findall(r'<section id="pc-', page)) == 1
    assert "2 more instructions exceed their ideal count." in page


@pytest.mark.parametrize("limit", [0, 1, 3])
def test_table_links_only_to_the_sections_it_draws(limit):
    accesses = [
        Access(pc, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(WARP_SIZE)))
        for pc in (0x10, 0x20, 0x30)
    ]
    page = html_report.render(Trace.from_accesses(accesses), limit=limit)
    links = re.findall(r'href="#([^"]+)"', page)
    sections = re.findall(r'<section id="([^"]+)"', page)
    assert sorted(links) == sorted(sections)
    assert len(sections) == limit


def test_report_of_a_recording_at_the_default_capacity_takes_seconds():
    trace = demos.transpose_trace()
    large = trace.select(np.resize(np.arange(len(trace)), 1 << 18))
    start = time.perf_counter()
    page = html_report.render(large)
    # Moving and counting every row once per candidate change takes over a minute here.
    assert time.perf_counter() - start < 30
    assert "pad each 128-byte row by 4 bytes" in page


@pytest.mark.parametrize(
    ("degree", "step"), [(1, 0), (2, 1), (3, 2), (4, 2), (5, 3), (8, 3), (9, 4), (32, 4)]
)
def test_degree_class_bins_the_distinct_words_of_a_bank(degree, step):
    assert html_report.degree_class(degree) == step


def test_degree_outside_one_to_32_is_rejected():
    with pytest.raises(ValueError, match="1 to 32"):
        html_report.degree_class(0)


def test_negative_limit_is_rejected():
    with pytest.raises(ValueError, match="zero or more"):
        html_report.render(demos.transpose_trace(), limit=-1)


def test_shared_sections_list_the_predicted_changes():
    page = html_report.render(demos.transpose_trace())
    assert "XOR swizzle Swizzle&lt;5,2,5&gt; on byte offsets" in page
    assert "pad each 128-byte row by 4 bytes" in page
