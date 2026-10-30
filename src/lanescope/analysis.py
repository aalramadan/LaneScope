import numpy as np

from lanescope import text_view
from lanescope.fixes import ranking
from lanescope.fixes.layouts import (
    Buffer,
    Change,
    Padding,
    XorSwizzle,
    compare,
    infer_buffer,
    later_shift,
)
from lanescope.model import global_memory, shared_memory
from lanescope.model.counters import as_record, count, distinct_rows, measure, worst_first
from lanescope.trace import Space, Trace


def explain(trace: Trace) -> dict:
    """Return the counters of every instruction of a trace, worst first."""
    return {
        "kernel": trace.metadata.get("kernel"),
        "arch": trace.metadata.get("arch"),
        "instructions": [as_record(trace, counter) for counter in worst_first(count(trace))],
    }


def overview(trace: Trace) -> dict:
    """Return the counters of every instruction with the source lines and the totals per unit."""
    data = explain(trace)
    totals: dict[str, dict[str, int]] = {}
    for record in data["instructions"]:
        total = totals.setdefault(
            record["unit"], {"requests": 0, "achieved": 0, "ideal": 0, "excessive": 0}
        )
        for name in total:
            total[name] += record[name]
    return data | {
        "rows": len(trace),
        "source": list(trace.metadata.get("source", [])),
        "totals": totals,
    }


def lanes(trace: Trace, pc: int, row: int | None = None) -> str:
    """Return the lane view of the worst access of an instruction, or of one of its rows."""
    matches = [counter for counter in count(trace) if counter.pc == pc]
    if not matches:
        raise ValueError(f"the trace has no access at pc {pc:#x}")
    # A generic instruction may access two spaces. The worse one is the one to show.
    worst = max(matches, key=lambda counter: counter.excessive)
    chosen = worst.worst_row if row is None else row
    if chosen < 0 or chosen >= len(trace) or int(trace.pc[chosen]) != pc:
        raise ValueError(f"row {chosen} is no access of pc {pc:#x}")
    return text_view.describe_row(trace, chosen)


def patterns(trace: Trace, pc: int, space: Space, limit: int = 50) -> dict:
    """Return the distinct access patterns of one instruction, worst first, with their repeats.

    Rows with the same lane offsets have the same counts, so each pattern stands
    for every warp that accessed memory in that way.
    """
    rows = np.flatnonzero((trace.pc == pc) & (trace.space == space))
    if rows.size == 0:
        raise ValueError(f"the trace has no {space.name.lower()} access at pc {pc:#x}")
    first, repeats = distinct_rows(trace.select(rows))
    found = []
    for start, weight in zip(first.tolist(), repeats.tolist(), strict=True):
        row = int(rows[start])
        achieved, ideal, excessive, conflicts = measure(trace, row)
        found.append(
            {
                "row": row,
                "repeats": weight,
                "cta": int(trace.cta[row]),
                "warp": int(trace.warp[row]),
                "achieved": achieved,
                "ideal": ideal,
                "excessive": excessive,
                "conflicts": conflicts,
            }
        )
    found.sort(key=lambda item: (-item["excessive"], -item["repeats"], item["row"]))
    return {
        "pc": f"{pc:#x}",
        "space": space.name.lower(),
        "rows": int(rows.size),
        "count": len(found),
        "patterns": found[:limit],
    }


def access(trace: Trace, row: int) -> dict:
    """Return one warp-level access lane by lane, with its counts and its text view."""
    if not 0 <= row < len(trace):
        raise ValueError(f"the trace has no row {row}")
    item = trace.access(row)
    data = {
        "row": row,
        "pc": f"{item.pc:#x}",
        "space": item.space.name.lower(),
        "kind": item.kind.name.lower(),
        "width": item.width,
        "cta": item.cta,
        "warp": item.warp,
        "active": sum(address is not None for address in item.addresses),
        "text": text_view.describe_row(trace, row),
    }
    if item.space == Space.SHARED:
        return data | _shared_lanes(trace, row)
    return data | _global_lanes(trace, row)


def whatif(trace: Trace, buffer: Buffer, change: Change) -> dict:
    """Return the counters before and after a layout change of every instruction it affects."""
    comparisons = compare(trace, buffer, change)
    if not comparisons:
        raise ValueError("no shared access of the trace falls inside the buffer")
    return {
        "change": change.describe(),
        "params": change_params(change),
        "extra_bytes": change.extra_bytes(buffer),
        "later_shift": later_shift(buffer, change),
        "instructions": [
            {"before": as_record(trace, item.before), "after": as_record(trace, item.after)}
            for item in comparisons
        ],
    }


def suggest(
    trace: Trace, buffer: Buffer | None = None, pc: int | None = None, limit: int = 5
) -> dict:
    """Return the best layout changes for a buffer, inferred from an instruction by default."""
    # A negative limit would slice changes off the end of the ranking.
    if limit < 0:
        raise ValueError("limit must be zero or more")
    if buffer is None:
        buffer = infer_buffer(trace, worst_shared_pc(trace) if pc is None else pc)
    found = ranking.suggest(trace, buffer)
    return {
        "buffer": {"start": buffer.start, "size": buffer.size},
        "found": len(found),
        "changes": [
            {
                "change": item.change.describe(),
                "params": change_params(item.change),
                "excessive_before": item.excessive_before,
                "excessive_after": item.excessive_after,
                "wavefronts_before": item.wavefronts_before,
                "wavefronts_after": item.wavefronts_after,
                "excessive_sectors_before": item.excessive_sectors_before,
                "excessive_sectors_after": item.excessive_sectors_after,
                "extra_bytes": item.extra_bytes,
            }
            for item in found[:limit]
        ],
    }


def worst_shared_pc(trace: Trace) -> int:
    """Return the shared-memory instruction with the most excess wavefronts."""
    shared = [counter for counter in worst_first(count(trace)) if counter.unit == "wavefronts"]
    if not shared:
        raise ValueError("the trace has no shared access")
    return shared[0].pc


def change_params(change: Change) -> dict:
    """Return the parameters of a layout change as JSON-ready data."""
    if isinstance(change, Padding):
        return {"kind": "padding", "row_bytes": change.row_bytes, "pad_bytes": change.pad_bytes}
    return {"kind": "swizzle", "bits": change.bits, "base": change.base, "shift": change.shift}


def change_from_params(params: object) -> Change:
    """Return the layout change that `change_params` describes, raising ValueError otherwise."""
    if not isinstance(params, dict):
        raise ValueError("a change needs its parameters as an object")
    names = {"padding": ("row_bytes", "pad_bytes"), "swizzle": ("bits", "base", "shift")}
    kind = params.get("kind")
    if not isinstance(kind, str) or kind not in names:
        raise ValueError("a change needs the kind padding or swizzle")
    values: list[int] = []
    for name in names[kind]:
        value = params.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"a {kind} needs the integers {', '.join(names[kind])}")
        values.append(value)
    return Padding(*values) if kind == "padding" else XorSwizzle(*values)


def _shared_lanes(trace: Trace, row: int) -> dict:
    addresses = trace.lane_addresses(row)
    width = int(trace.width[row])
    result = shared_memory.analyze_row(trace, row)
    phases = []
    for phase in result.phases:
        cells = []
        for lane in phase.lanes:
            address = addresses[lane]
            assert address is not None
            for word in shared_memory.words(address, width):
                bank = word % shared_memory.BANKS
                cells.append([lane, bank, word, phase.bank_words[bank].index(word)])
        phases.append(
            {
                "lanes": list(phase.lanes),
                "wavefronts": phase.wavefronts,
                "worst_bank": phase.worst_bank,
                "degrees": [len(words) for words in phase.bank_words],
                "cells": cells,
            }
        )
    return {
        "unit": "wavefronts",
        "achieved": result.wavefronts,
        "ideal": result.ideal,
        "excessive": result.excessive,
        "conflicts": result.conflicts,
        "paired": result.paired,
        "lanes": [None if address is None else {"offset": address} for address in addresses],
        "phases": phases,
    }


def _global_lanes(trace: Trace, row: int) -> dict:
    addresses = trace.lane_addresses(row)
    result = global_memory.analyze_row(trace, row)
    # Global addresses can pass 2**53, the largest integer that JSON readers keep
    # exact, so addresses travel as text and offsets from the first line as numbers.
    return {
        "unit": "sectors",
        "achieved": result.sectors,
        "ideal": result.ideal,
        "excessive": result.excessive,
        "lines": result.lines,
        "base": f"{result.base:#x}",
        "lanes": [
            None
            if address is None
            else {"address": f"{address:#x}", "offset": address - result.base}
            for address in addresses
        ],
        "sectors": [
            {
                "sector": sector,
                "offset": sector * global_memory.SECTOR_BYTES,
                "lanes": list(lanes),
            }
            for sector, lanes in result.sector_lanes
        ],
    }
