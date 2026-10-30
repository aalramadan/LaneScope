import string

from lanescope.model import global_memory, shared_memory
from lanescope.trace import Space, Trace

WORD_SYMBOLS = string.ascii_uppercase + string.ascii_lowercase + string.digits
EMPTY = "."


def describe_row(trace: Trace, row: int) -> str:
    """Return the lane-by-lane view of one row of the trace."""
    access = trace.access(row)
    opcode = trace.opcode(access.pc)
    header = (
        f"pc {access.pc:#06x}, {access.space.name.lower()} {access.kind.name.lower()} "
        f"of {access.width} bytes{f' ({opcode})' if opcode else ''}, "
        f"CTA {access.cta}, warp {access.warp}"
    )
    if access.space == Space.SHARED:
        return header + "\n" + bank_grid(shared_memory.analyze_row(trace, row), access.addresses)
    return header + "\n" + sector_table(global_memory.analyze_row(trace, row))


def bank_grid(result: shared_memory.SharedAccess, addresses: tuple[int | None, ...]) -> str:
    """Return the bank grid of every phase of a shared access."""
    lines = [
        f"{result.wavefronts} wavefronts against an ideal of {result.ideal}, "
        f"{result.conflicts} bank conflicts"
    ]
    if result.paired:
        lines.append("lane pairs read identical addresses, so the access runs in fewer groups")
    for number, phase in enumerate(result.phases, start=1):
        lines.append("")
        lines.append(
            f"phase {number}, lanes {lane_ranges(phase.lanes)}, {phase.wavefronts} wavefronts, "
            f"worst bank {phase.worst_bank}"
        )
        lines.append("bank    " + "".join(str(bank // 10) for bank in range(shared_memory.BANKS)))
        lines.append("        " + "".join(str(bank % 10) for bank in range(shared_memory.BANKS)))
        for lane in phase.lanes:
            cells = [EMPTY] * shared_memory.BANKS
            address = addresses[lane]
            assert address is not None
            for word in shared_memory.words(address, result.width):
                bank = word % shared_memory.BANKS
                cells[bank] = WORD_SYMBOLS[phase.bank_words[bank].index(word)]
            lines.append(f"lane {lane:2d} {''.join(cells)}  offset {address:#x}")
        lines.append("words   " + "".join(_count_symbol(len(words)) for words in phase.bank_words))
    return "\n".join(lines)


def sector_table(result: global_memory.GlobalAccess) -> str:
    """Return the sectors of a global access with the lanes that touch each one."""
    lines = [
        f"{result.sectors} sectors against an ideal of {result.ideal}, in {result.lines} lines",
        "sector  address             lanes",
    ]
    for sector, lanes in result.sector_lanes:
        lines.append(f"{sector:6d}  {result.sector_address(sector):#018x}  {lane_ranges(lanes)}")
    return "\n".join(lines)


def lane_ranges(lanes: tuple[int, ...]) -> str:
    """Return lanes as comma-separated ranges, such as 0-7,16."""
    parts = []
    start = previous = lanes[0]
    for lane in lanes[1:]:
        if lane != previous + 1:
            parts.append(_range_text(start, previous))
            start = lane
        previous = lane
    parts.append(_range_text(start, previous))
    return ",".join(parts)


def _range_text(start: int, end: int) -> str:
    return str(start) if start == end else f"{start}-{end}"


def _count_symbol(count: int) -> str:
    # One character per bank keeps the count row aligned with the grid.
    return string.digits[count] if count < 10 else string.ascii_uppercase[count - 10]
