import html

from lanescope import __version__, analysis, text_view
from lanescope.model import global_memory, shared_memory
from lanescope.model.counters import InstructionCounters, count, worst_first
from lanescope.trace import WARP_SIZE, Space, Trace

CELL = 14
STEP = CELL + 2
LEFT = 56
BAR_AREA = 48
BAR_WIDTH = 10
TOP = 16
AXIS = 18
SVG_NS = "http://www.w3.org/2000/svg"

# Five ordinal blue steps for the distinct words of a bank (1, 2, 3-4, 5-8, 9 or more).
# Each step keeps at least 2:1 contrast against its surface in the light and dark themes.
DEGREE_BINS = ((1, 1), (2, 2), (3, 4), (5, 8), (9, 32))
LIGHT_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281")
DARK_RAMP = ("#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4")
LIGHT = {
    "surface": "#fcfcfb",
    "page": "#f9f9f7",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "empty": "#f0efec",
    "bar": "#2a78d6",
}
DARK = {
    "surface": "#1a1a19",
    "page": "#0d0d0d",
    "ink": "#f0efec",
    "ink2": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "empty": "#262624",
    "bar": "#3987e5",
}


def render(trace: Trace, limit: int = 20) -> str:
    """Return the HTML report of a trace, with grids for at most `limit` instructions."""
    if limit < 0:
        raise ValueError("the report needs a limit of zero or more instructions")
    counters = worst_first(count(trace))
    worst = [counter for counter in counters if counter.excessive > 0]
    drawn = {_anchor(counter) for counter in worst[:limit]}
    kernel = html.escape(str(trace.metadata.get("kernel", "kernel")))
    arch = html.escape(str(trace.metadata.get("arch", "unknown architecture")))
    parts = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>LaneScope report of {kernel}</title>",
        f"<style>{_css()}</style></head>",
        '<body><main class="viz-root">',
        f"<h1>{kernel}</h1>",
        f'<p class="sub">{arch}, {len(trace)} warp-level accesses, LaneScope {__version__}</p>',
        "<h2>Memory instructions</h2>",
        _instruction_table(trace, counters, drawn),
        "<h2>Worst accesses</h2>",
    ]
    if not worst:
        parts.append("<p>No instruction exceeds its ideal count.</p>")
    for counter in worst[:limit]:
        parts.append(_section(trace, counter))
    if len(worst) > limit:
        parts.append(
            f"<p>{len(worst) - limit} more instructions exceed their ideal count. "
            "The table above lists them.</p>"
        )
    parts.append(
        '<p class="sub">Counts follow the LaneScope model of Nsight Compute Source page '
        "counters. Compare them with Nsight Compute before relying on them.</p>"
    )
    parts.append("</main></body></html>")
    return "\n".join(parts)


def degree_class(degree: int) -> int:
    """Return the ordinal step, from 0 to 4, for the distinct words of a bank."""
    for step, (low, high) in enumerate(DEGREE_BINS):
        if low <= degree <= high:
            return step
    raise ValueError(f"a bank serves 1 to 32 distinct words, not {degree}")


def _instruction_table(trace: Trace, counters: list[InstructionCounters], drawn: set[str]) -> str:
    head = "".join(
        f'<th class="num">{name}</th>' if numeric else f"<th>{name}</th>"
        for name, numeric in (
            ("PC", False), ("Line", False), ("Source", False), ("Space", False),
            ("Kind", False), ("Width", True), ("Requests", True), ("Unit", False),
            ("Achieved", True), ("Ideal", True), ("Excessive", True), ("Bank conflicts", True),
        )
    )  # fmt: skip
    rows = []
    for counter in counters:
        line = trace.source_line(counter.pc)
        pc = f"{counter.pc:#06x}"
        # Only the worst instructions up to the limit get a section to link to.
        if _anchor(counter) in drawn:
            pc = f'<a href="#{_anchor(counter)}">{pc}</a>'
        conflicts = counter.conflicts if counter.space == Space.SHARED else ""
        cells = [
            f"<td>{pc}</td>",
            f"<td>{line[0] if line else ''}</td>",
            f"<td><code>{html.escape(line[1]) if line else ''}</code></td>",
            f"<td>{counter.space.name.lower()}</td>",
            f"<td>{counter.kind.name.lower()}</td>",
            f'<td class="num">{counter.width}</td>',
            f'<td class="num">{counter.requests}</td>',
            f"<td>{counter.unit}</td>",
            f'<td class="num">{counter.achieved}</td>',
            f'<td class="num">{counter.ideal}</td>',
            f'<td class="num">{counter.excessive}</td>',
            f'<td class="num">{conflicts}</td>',
        ]
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<div class="scroll"><table><tr>{head}</tr>{"".join(rows)}</table></div>'


def _section(trace: Trace, counter: InstructionCounters) -> str:
    access = trace.access(counter.worst_row)
    line = trace.source_line(counter.pc)
    title = (
        f"{counter.pc:#06x}{f', line {line[0]}' if line else ''}, "
        f"{counter.space.name.lower()} {counter.kind.name.lower()} of {counter.width} bytes"
    )
    parts = [f'<section id="{_anchor(counter)}"><h3>{title}</h3>']
    if line and line[1]:
        parts.append(f"<p><code>{html.escape(line[1])}</code></p>")
    if access.space == Space.SHARED:
        result = shared_memory.analyze_row(trace, counter.worst_row)
        parts.append(
            f"<p>Warp {access.warp} of CTA {access.cta} takes {result.wavefronts} wavefronts "
            f"against an ideal of {result.ideal}, with {result.conflicts} bank conflicts.</p>"
        )
        for number, phase in enumerate(result.phases, start=1):
            parts.append(
                f'<p class="phase">Phase {number}, lanes {text_view.lane_ranges(phase.lanes)}, '
                f"{phase.wavefronts} wavefronts</p>"
            )
            parts.append(_bank_svg(phase, access.addresses, access.width))
        parts.append(_legend())
        parts.append(_suggestions(trace, counter.pc))
        text = text_view.bank_grid(result, access.addresses)
    else:
        sectors = global_memory.analyze_row(trace, counter.worst_row)
        parts.append(
            f"<p>Warp {access.warp} of CTA {access.cta} touches {sectors.sectors} sectors in "
            f"{sectors.lines} lines against an ideal of {sectors.ideal}.</p>"
        )
        parts.append(_sector_svg(sectors))
        text = text_view.sector_table(sectors)
    parts.append(f"<details><summary>Text view</summary><pre>{html.escape(text)}</pre></details>")
    parts.append("</section>")
    return "\n".join(parts)


def _suggestions(trace: Trace, pc: int, limit: int = 3) -> str:
    data = analysis.suggest(trace, pc=pc, limit=limit)
    if not data["changes"]:
        return "<p>No padding or swizzle lowers the excess wavefronts of this buffer.</p>"
    rows = "".join(
        f"<tr><td>{html.escape(item['change'])}</td>"
        f'<td class="num">{item["excessive_before"]}</td>'
        f'<td class="num">{item["excessive_after"]}</td>'
        f'<td class="num">{item["extra_bytes"]}</td></tr>'
        for item in data["changes"]
    )
    buffer = data["buffer"]
    return (
        f"<p>Predicted changes for the buffer at {buffer['start']:#x} of {buffer['size']} bytes, "
        "from the recorded addresses of its accesses.</p>"
        '<div class="scroll"><table><tr><th>Change</th><th class="num">Excess before</th>'
        '<th class="num">Excess after</th><th class="num">Extra bytes</th></tr>'
        f"{rows}</table></div>"
    )


def _bank_svg(phase: shared_memory.Phase, addresses: tuple[int | None, ...], width: int) -> str:
    degrees = [len(words) for words in phase.bank_words]
    peak = max(degrees)
    grid_top = TOP + BAR_AREA + AXIS
    label = (
        f"Bank grid of {len(phase.lanes)} lanes. Bank {phase.worst_bank} serves "
        f"{peak} distinct words, which takes {peak} wavefronts."
    )
    parts = [
        _svg_open(label, shared_memory.BANKS, grid_top + len(phase.lanes) * STEP + 4),
        _label(LEFT + phase.worst_bank * STEP + CELL / 2, TOP - 4, str(peak), css="ink"),
        _label(LEFT - 6, TOP + BAR_AREA + 13, "bank", anchor="end"),
    ]
    for bank, degree in enumerate(degrees):
        if degree:
            x = LEFT + bank * STEP + (CELL - BAR_WIDTH) / 2
            path = _bar_path(x, TOP + BAR_AREA, max(2.0, BAR_AREA * degree / peak))
            parts.append(
                f'<path class="bar" fill="{LIGHT["bar"]}" d="{path}">'
                f"<title>bank {bank}, {degree} distinct words</title></path>"
            )
    for bank in (0, 8, 16, 24, shared_memory.BANKS - 1):
        parts.append(_label(LEFT + bank * STEP + CELL / 2, TOP + BAR_AREA + 13, str(bank)))
    for row, lane in enumerate(phase.lanes):
        y = grid_top + row * STEP
        if row % 4 == 0 or row == len(phase.lanes) - 1:
            parts.append(_label(LEFT - 6, y + 11, f"lane {lane}", anchor="end"))
        parts.extend(
            _cell(LEFT + bank * STEP, y, "empty", LIGHT["empty"])
            for bank in range(shared_memory.BANKS)
        )
        address = addresses[lane]
        assert address is not None
        for word in shared_memory.words(address, width):
            bank = word % shared_memory.BANKS
            step = degree_class(degrees[bank])
            tip = (
                f"lane {lane}, bank {bank}, byte offset {word * shared_memory.WORD_BYTES:#x}, "
                f"word {phase.bank_words[bank].index(word) + 1} of {degrees[bank]} distinct "
                "words in this bank"
            )
            parts.append(_cell(LEFT + bank * STEP, y, f"cell d{step}", LIGHT_RAMP[step], tip))
    parts.append("</svg></div>")
    return "".join(parts)


def _sector_svg(result: global_memory.GlobalAccess) -> str:
    columns = [sector for sector, _ in result.sector_lanes]
    grid_top = TOP + AXIS
    label = f"Sector grid of {WARP_SIZE} lanes and {len(columns)} sectors."
    parts = [
        _svg_open(label, len(columns), grid_top + WARP_SIZE * STEP + 4),
        _label(LEFT - 6, TOP + 13, "sector", anchor="end"),
    ]
    for column in range(0, len(columns), 4):
        parts.append(_label(LEFT + column * STEP + CELL / 2, TOP + 13, str(column)))
    touched = {(lane, sector) for sector, lanes in result.sector_lanes for lane in lanes}
    for lane in range(WARP_SIZE):
        y = grid_top + lane * STEP
        if lane % 4 == 0 or lane == WARP_SIZE - 1:
            parts.append(_label(LEFT - 6, y + 11, f"lane {lane}", anchor="end"))
        for column, sector in enumerate(columns):
            x = LEFT + column * STEP
            if (lane, sector) in touched:
                tip = f"lane {lane}, sector at {result.sector_address(sector):#x}"
                parts.append(_cell(x, y, "cell touched", LIGHT["bar"], tip))
            else:
                parts.append(_cell(x, y, "empty", LIGHT["empty"]))
    parts.append("</svg></div>")
    return "".join(parts)


def _anchor(counter: InstructionCounters) -> str:
    # One PC of a generic instruction may hold counters for two spaces.
    return f"pc-{counter.pc:x}-{counter.space.name.lower()}"


def _svg_open(label: str, columns: int, height: int) -> str:
    width = LEFT + columns * STEP + 8
    return (
        f'<div class="scroll"><svg xmlns="{SVG_NS}" role="img" aria-label="{html.escape(label)}" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        'font-family="system-ui, sans-serif" font-size="10">'
    )


def _label(x: float, y: float, text: str, anchor: str = "middle", css: str = "muted") -> str:
    return (
        f'<text class="{css}" fill="{LIGHT["ink" if css == "ink" else "muted"]}" '
        f'x="{x}" y="{y}" text-anchor="{anchor}">{html.escape(text)}</text>'
    )


def _cell(x: float, y: float, css: str, fill: str, tip: str | None = None) -> str:
    shape = (
        f'<rect class="{css}" fill="{fill}" x="{x}" y="{y}" width="{CELL}" height="{CELL}" rx="2"'
    )
    return f"{shape}><title>{html.escape(tip)}</title></rect>" if tip else f"{shape}/>"


def _legend() -> str:
    items = []
    for step, (low, high) in enumerate(DEGREE_BINS):
        text = str(low) if low == high else f"{low} to {high}"
        items.append(
            f'<span class="key"><svg width="12" height="12" aria-hidden="true">'
            f'<rect class="d{step}" fill="{LIGHT_RAMP[step]}" width="12" height="12" rx="2"/>'
            f"</svg>{text}</span>"
        )
    return f'<p class="legend">Distinct words in the bank of each cell: {"".join(items)}</p>'


def _bar_path(x: float, baseline: float, height: float) -> str:
    # Rounded data end at the top, square end at the baseline.
    radius = min(4.0, height / 2, BAR_WIDTH / 2)
    top = baseline - height
    right = x + BAR_WIDTH
    return (
        f"M{x},{baseline} V{top + radius} Q{x},{top} {x + radius},{top} "
        f"H{right - radius} Q{right},{top} {right},{top + radius} V{baseline} Z"
    )


def _css() -> str:
    ramp = "".join(f".d{step}{{fill:{color}}}" for step, color in enumerate(DARK_RAMP))
    dark = (
        f"body{{background:{DARK['page']};color:{DARK['ink']}}}"
        f".viz-root{{background:{DARK['surface']}}}"
        f"th,td{{border-color:{DARK['grid']}}}.sub,.phase,.legend{{color:{DARK['ink2']}}}"
        f".ink{{fill:{DARK['ink']}}}.muted{{fill:{DARK['muted']}}}"
        f".empty{{fill:{DARK['empty']}}}.bar,.touched{{fill:{DARK['bar']}}}{ramp}"
        f"a{{color:{DARK['bar']}}}"
    )
    return (
        ":root{color-scheme:light dark}"
        f"body{{margin:0;background:{LIGHT['page']};color:{LIGHT['ink']};"
        "font:14px/1.5 system-ui,-apple-system,'Segoe UI',sans-serif}"
        ".viz-root{max-width:1000px;margin:0 auto;padding:24px 16px;"
        f"background:{LIGHT['surface']}}}"
        "h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}"
        "h3{font-size:15px;margin:24px 0 4px}"
        f".sub,.phase,.legend{{color:{LIGHT['ink2']}}}.phase{{margin:12px 0 2px}}"
        ".scroll{overflow-x:auto}"
        "table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
        f"th,td{{border-bottom:1px solid {LIGHT['grid']};padding:4px 8px;text-align:left;"
        "white-space:nowrap}"
        ".num{text-align:right}code{font-size:12px}"
        ".key{display:inline-flex;align-items:center;gap:4px;margin-left:12px}"
        "rect.cell:hover,path.bar:hover{opacity:.75}"
        f"a{{color:{LIGHT['bar']}}}"
        f"@media (prefers-color-scheme:dark){{{dark}}}"
    )
