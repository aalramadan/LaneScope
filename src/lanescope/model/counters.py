import dataclasses

import numpy as np

from lanescope.model import global_memory, shared_memory
from lanescope.model.global_memory import LINE_BYTES
from lanescope.trace import Kind, Space, Trace

InstructionKey = tuple[int, int, int, int]
_ZERO_LANE = np.iinfo(np.uint64).max
# An odd 64-bit constant of the hash of FNV-1a, which spreads the bits of each column.
_HASH_MULTIPLIER = np.uint64(0x100000001B3)


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
    destinations, sources = trace.copy_destinations, trace.copy_sources
    if not destinations and not sources:
        return keys
    # The two rows of a cp.async count from both sides of the copy, so each row keys on the
    # other side as well: a global row on its shared addresses, and a shared row on its
    # global addresses from their first line, with a mark for each lane that fills zeros.
    other = np.zeros_like(addresses)
    for row, lanes in destinations.items():
        other[row] = [0 if address is None else address for address in lanes]
    for row, lanes in sources.items():
        known = [address for address in lanes if address is not None]
        base = min(known) // LINE_BYTES * LINE_BYTES if known else 0
        other[row] = [_ZERO_LANE if address is None else address - base for address in lanes]
    return np.column_stack([keys, other])


def distinct_rows(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    """Return the first row of each distinct row key in trace order and how many rows share it."""
    keys = row_keys(trace)
    # Sorting one hash per row is far faster than sorting whole rows. Comparing every row
    # with the first row of its hash keeps the result exact.
    _, first, inverse, repeats = np.unique(
        _row_hashes(keys), return_index=True, return_inverse=True, return_counts=True
    )
    if not np.array_equal(keys, keys[first[inverse]]):
        _, first, repeats = np.unique(keys, axis=0, return_index=True, return_counts=True)
    order = np.argsort(first)
    return first[order], repeats[order]


def _row_hashes(keys: np.ndarray) -> np.ndarray:
    # Unsigned integers wrap, which makes this a hash modulo 2**64.
    hashes = np.zeros(len(keys), dtype=np.uint64)
    for column in keys.T:
        hashes = (hashes ^ column) * _HASH_MULTIPLIER
        hashes ^= hashes >> np.uint64(29)
    return hashes


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
