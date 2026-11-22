import pathlib

from lanescope.text_view import printable

# The memory, the counted unit, and the cause of the extra units of each unit.
UNITS = {
    "wavefronts": ("Shared memory", "passes", "Bank conflicts"),
    "sectors": ("Global memory", "sectors", "Scattered accesses"),
}
MARK = "*"
KERNEL_HEADER = "{:>14}  {:>9}  {:>8}  kernel"
LINE_HEADER = "{:>14}  {:>6}  {:>9}  {:>8}  {:32}  kernel"


def render(data: dict, unit: str = "wavefronts", limit: int = 10) -> str:
    """Return a summary as text: the totals, the worst kernels, the worst lines, and a next step."""
    memory, noun, cause = UNITS[unit]
    kernels = data["kernels"]
    runs = ", ".join(printable(run) for run in data["runs"]) or "no run"
    out = [f"{runs}: {_plural(data['traces'], 'launch')} of {_plural(len(kernels), 'kernel')}"]
    total = data["totals"][unit]
    if total["achieved"] == 0:
        out.append(f"{memory}: no accesses")
        return "\n".join(out)
    share = percent(total["excessive"], total["achieved"])
    out.append(
        f"{memory}: {total['achieved']:,} {noun} where {total['ideal']:,} would do. "
        f"{cause} add {total['excessive']:,} of them ({share})."
    )
    ranked = [kernel for kernel in kernels if kernel["totals"][unit]["achieved"]]
    header = KERNEL_HEADER.format(f"extra {noun}", "of kernel", "launches")
    out += ["", f"Kernels by extra {noun}", header]
    for kernel in ranked[:limit]:
        counts = kernel["totals"][unit]
        out.append(
            f"{counts['excessive']:>14,}  {percent(counts['excessive'], counts['achieved']):>9}  "
            f"{kernel['launches']:>8,}  {printable(kernel['name'])}"
        )
    if len(ranked) > limit:
        out.append(f"{'':>14}  and {_plural(len(ranked) - limit, 'more kernel')}")
    lines = [line for line in data["lines"] if line["unit"] == unit]
    if not lines:
        out += ["", f"No line adds extra {noun}."]
        return "\n".join(out)
    header = LINE_HEADER.format(f"extra {noun}", "ratio", "of kernel", "launches", "line")
    out += ["", f"Lines by extra {noun}", header]
    for line in lines[:limit]:
        place = _place(line) + (f" {MARK}" if line["data_dependent"] else "")
        ratio = f"{line['achieved'] / line['ideal']:.1f}x" if line["ideal"] else "-"
        out.append(
            f"{line['excessive']:>14,}  {ratio:>6}  "
            f"{percent(line['excessive'], line['kernel_achieved']):>9}  {line['launches']:>8,}  "
            f"{place:32}  {printable(line['name'])}"
        )
    if len(lines) > limit:
        out.append(f"{'':>14}  and {_plural(len(lines) - limit, 'more line')}")
    if any(line["data_dependent"] for line in lines[:limit]):
        out += [
            "",
            f"{MARK} The conflicts of this line follow the values that the kernel reads, such as "
            "the bins of a histogram. No padding or swizzle removes them.",
        ]
    worst = lines[0]
    trace = printable(worst["worst"]["id"])
    step = "lanes" if worst["data_dependent"] or unit == "sectors" else "suggest"
    action = "shows its threads" if step == "lanes" else "ranks its fixes"
    out += [
        "",
        f"Worst line: lanescope {step} {trace} --pc {worst['worst']['pc']} {action}, and "
        f"lanescope gui {printable(worst['run'])} opens it in the browser.",
    ]
    return "\n".join(out)


def notes(data: dict) -> list[str]:
    """Return the warnings of a summary: launches without source lines or with lost records."""
    found = []
    traces = _plural(data["traces"], "launch")
    if data["without_lines"]:
        found.append(
            f"No source lines in {data['without_lines']} of {traces}. Build the program with "
            "-lineinfo, such as nvcc -lineinfo, to see the line of each instruction."
        )
    dropped = data["dropped"]
    if dropped["traces"]:
        found.append(
            f"The record buffer filled up in {dropped['traces']} of {traces} and left out "
            f"{dropped['records']:,} accesses, therefore their counts are low. Record again "
            "with a larger --capacity or with fewer CTAs, such as --ctas 0:8."
        )
    if data["unreadable"]:
        found.append(f"skipped {_plural(data['unreadable'], 'file')} without a trace")
    return found


def _place(line: dict) -> str:
    """Return where a line is, such as kernel.cu:25 load, or its PC without a line table."""
    if line["line"] is None:
        return f"pc {line['pc']} {line['kind']}"
    name = pathlib.PurePath(printable(line["file"])).name
    where = f"{name}:{line['line']}" if name else f"line {line['line']}"
    return f"{where} {line['kind']}"


def percent(part: int, whole: int) -> str:
    """Return a part of a whole as a percentage with one decimal, or a dash for no whole."""
    if whole == 0:
        return "-"
    share = 100 * part / whole
    return "<0.1%" if 0 < share < 0.1 else f"{share:.1f}%"


def _plural(value: int, noun: str) -> str:
    ending = "es" if noun.endswith("ch") else "s"
    return f"{value:,} {noun}{'' if value == 1 else ending}"
