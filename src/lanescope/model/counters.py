import dataclasses

import numpy as np

from lanescope.model import global_memory, shared_memory
from lanescope.model.global_memory import LINE_BYTES
from lanescope.trace import Kind, Space, Trace

InstructionKey = tuple[int, int, int, int]


@dataclasses.dataclass(frozen=True)
class InstructionCounters:
    """Counters of one instruction, summed over its warp-level accesses."""

    pc: int
    space: Space
    kind: Kind
    width: int
    requests: int
    achieved: int
    ideal: int
    excessive: int
    conflicts: int
    worst_row: int

    @property
    def unit(self) -> str:
        """Return the name of the counted unit."""
        return "wavefronts" if self.space == Space.SHARED else "sectors"


def instruction_key(trace: Trace, row: int) -> InstructionKey:
    """Return the PC, space, kind, and width that identify the instruction of a row."""
    return (int(trace.pc[row]), int(trace.space[row]), int(trace.kind[row]), int(trace.width[row]))


def count(trace: Trace) -> list[InstructionCounters]:
    """Return the counters of every instruction in the trace, ordered by PC and space."""
    if len(trace) == 0:
        return []
    # Recordings repeat the same access in many CTAs and loop iterations. Measuring each
    # distinct row once and weighting it by its repeats keeps large traces fast.
    first, repeats = distinct_rows(trace)
    totals: dict[InstructionKey, list[int]] = {}
    for row, weight in zip(first.tolist(), repeats.tolist(), strict=True):
        achieved, ideal, excessive, conflicts = measure(trace, row)
        total = totals.setdefault(instruction_key(trace, row), [0, 0, 0, 0, 0, row, excessive])
        total[0] += weight
        total[1] += weight * achieved
        total[2] += weight * ideal
        total[3] += weight * excessive
        total[4] += weight * conflicts
        # The earliest access with the largest excess is the one worth showing lane by lane.
        if excessive > total[6]:
            total[5], total[6] = row, excessive
    return [
        InstructionCounters(
            pc=pc,
            space=Space(space),
            kind=Kind(kind),
            width=width,
            requests=total[0],
            achieved=total[1],
            ideal=total[2],
            excessive=total[3],
            conflicts=total[4],
            worst_row=total[5],
        )
        for (pc, space, kind, width), total in sorted(totals.items())
    ]


def row_keys(trace: Trace) -> np.ndarray:
    """Return a key per row. Rows with equal keys belong to one instruction and count alike."""
    active = trace.active()
    addresses = np.where(active, trace.addr, np.uint64(0))
    # Sector counts do not change when a global access moves by whole lines, so global
    # rows count from their first line and match across CTAs.
    is_global = trace.space == Space.GLOBAL
    if is_global.any():
        lanes = active[is_global]
        highest = np.uint64(np.iinfo(np.uint64).max)
        first = np.where(lanes, addresses[is_global], highest).min(axis=1)
        base = first // np.uint64(LINE_BYTES) * np.uint64(LINE_BYTES)
        addresses[is_global] = np.where(lanes, addresses[is_global] - base[:, None], np.uint64(0))
    columns = [trace.pc, trace.space, trace.kind, trace.width, trace.mask]
    keys = np.column_stack([*(c.astype(np.uint64) for c in columns), addresses])
    partners = trace.copy_partners
    if not partners:
        return keys
    # The two rows of a cp.async count from both sides of the copy, so each row keys on the
    # addresses of its partner as well.
    other = np.zeros_like(addresses)
    other[list(partners)] = addresses[list(partners.values())]
    return np.column_stack([keys, other])


def distinct_rows(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    """Return the first row of each distinct row key in trace order and how many rows share it."""
    _, first, repeats = np.unique(row_keys(trace), axis=0, return_index=True, return_counts=True)
    order = np.argsort(first)
    return first[order], repeats[order]


def measure(trace: Trace, row: int) -> tuple[int, int, int, int]:
    """Return the achieved, ideal, and excess counts and the bank conflicts of one row."""
    if trace.space[row] == Space.SHARED:
        shared = shared_memory.analyze_row(trace, row)
        return shared.wavefronts, shared.ideal, shared.excessive, shared.conflicts
    sectors = global_memory.analyze_row(trace, row)
    return sectors.sectors, sectors.ideal, sectors.excessive, 0


def as_record(trace: Trace, counter: InstructionCounters) -> dict:
    """Return the counters of one instruction as JSON-ready data, with its source line."""
    line = trace.source_line(counter.pc)
    return {
        "pc": f"{counter.pc:#x}",
        "line": line[0] if line else None,
        "source": line[1] if line else None,
        "space": counter.space.name.lower(),
        "kind": counter.kind.name.lower(),
        "width": counter.width,
        "requests": counter.requests,
        "unit": counter.unit,
        "achieved": counter.achieved,
        "ideal": counter.ideal,
        "excessive": counter.excessive,
        "bank_conflicts": counter.conflicts,
        "worst_row": counter.worst_row,
    }


def worst_first(counters: list[InstructionCounters]) -> list[InstructionCounters]:
    """Return the counters ordered by excess, largest first, then by PC."""
    return sorted(counters, key=lambda counter: (-counter.excessive, counter.pc))
