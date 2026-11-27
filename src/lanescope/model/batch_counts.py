import dataclasses
from collections.abc import Callable

import numpy as np

from lanescope.model import async_copy, global_memory, shared_memory
from lanescope.model.global_memory import SECTOR_BYTES
from lanescope.model.shared_memory import BANKS, WAVEFRONT_BYTES, WORD_BYTES
from lanescope.trace import WARP_SIZE, WIDTHS, Kind, Space, Trace

_NO_WORD = np.uint64(np.iinfo(np.uint64).max)
# Rows counted in one NumPy pass, which bounds the memory of the word arrays of 16-byte rows.
_CHUNK_ROWS = 4096
# The lanes that one bank phase serves: the whole warp, half-warps for 8 bytes, and quarters
# for 16 bytes, as the groups of a shared store or a narrow load in shared_memory.
_GROUP_LANES = {width: len(groups[0]) for width, groups in shared_memory._GROUPS.items()}
_ATOMIC_GROUP_LANES = {
    width: WARP_SIZE if width <= WORD_BYTES else len(shared_memory._HALF_WARPS[0])
    for width in WIDTHS
}
_BYPASS = "BYPASS"


@dataclasses.dataclass(frozen=True)
class RowCounts:
    """Achieved, ideal, and excess counts and bank conflicts of many rows, one array each."""

    achieved: np.ndarray
    ideal: np.ndarray
    excessive: np.ndarray
    conflicts: np.ndarray


def row_counts(trace: Trace, rows: np.ndarray | None = None) -> RowCounts:
    """Return what counters.measure gives for each of the rows, or for every row of the trace.

    Shared atomics, shared stores, shared loads of up to 4 bytes, and global rows count in
    NumPy. The other rows go through the scalar model once per distinct pattern.
    """
    picked = np.arange(len(trace)) if rows is None else np.asarray(rows, dtype=np.intp)
    found = np.zeros((4, picked.size), dtype=np.int64)
    if picked.size == 0:
        return RowCounts(*found)
    space, kind, width = trace.space[picked], trace.kind[picked], trace.width[picked]
    addresses, active = trace.addr[picked], trace.active()[picked]
    opcodes = _opcodes(trace, picked)
    matrix = _opcode_flags(opcodes, lambda code: _base(code) in shared_memory._MATRIX_OPCODES)
    atomic_code = _opcode_flags(opcodes, lambda code: _base(code) == shared_memory._ATOMIC_OPCODE)
    copy = _opcode_flags(opcodes, lambda code: _base(code) == shared_memory._ASYNC_COPY_OPCODE)
    shared = space == Space.SHARED
    atomic = shared & ~matrix & ((kind == Kind.ATOMIC) | atomic_code)
    distinct = (
        shared
        & ~matrix
        & ~atomic
        & ~(copy & (kind == Kind.STORE))
        & ((kind == Kind.STORE) | ~np.isin(width, list(shared_memory._MERGE_SPANS)))
    )
    # Only the global side of a 16-byte cp.async that bypasses L1 counts from its writes.
    bypass = _opcode_flags(opcodes, lambda code: _BYPASS in code)
    writes = (
        ~shared
        & bypass
        & (width == async_copy.ROW_BYTES)
        & np.isin(picked, list(trace.copy_destinations))
    )
    sectors = ~shared & ~writes
    swap = _opcode_flags(opcodes, lambda code: shared_memory._SWAP_SUFFIX in code)
    for positions, counted in (
        (np.flatnonzero(distinct), lambda at: _bank_counts(addresses[at], active[at], width[at])),
        (
            np.flatnonzero(atomic),
            lambda at: _atomic_counts(addresses[at], active[at], width[at], swap[at]),
        ),
        (np.flatnonzero(sectors), lambda at: _sector_counts(addresses[at], active[at], width[at])),
    ):
        for start in range(0, positions.size, _CHUNK_ROWS):
            at = positions[start : start + _CHUNK_ROWS]
            found[:, at] = counted(at)
    scalar = np.flatnonzero(~(distinct | atomic | sectors))
    if scalar.size:
        found[:, scalar] = _scalar_counts(trace, picked[scalar], [opcodes[i] for i in scalar])
    return RowCounts(*found)


def copy_read_excess(trace: Trace) -> np.ndarray:
    """Return the excess sectors of the global reads of each cp.async that writes a shared row.

    Every other row gets zero, as global_memory.copy_reads gives it no reads.
    """
    found = np.zeros(len(trace), dtype=np.int64)
    rows = np.array(sorted(trace.copy_sources), dtype=np.intp)
    if rows.size == 0:
        return found
    seen: dict[tuple, int] = {}
    for row, lanes in zip(rows.tolist(), _lane_tuples(trace, rows), strict=True):
        sources = trace.copy_sources[row]
        key = (
            trace.opcode(int(trace.pc[row])),
            int(trace.width[row]),
            _moved(lanes),
            _moved(sources) if _reads_all(lanes, sources) else sources,
        )
        if key not in seen:
            reads = global_memory.copy_reads(trace, row)
            seen[key] = reads.excessive if reads is not None else 0
        found[row] = seen[key]
    return found


def _bank_counts(
    addresses: np.ndarray, active: np.ndarray, width: np.ndarray, atomic: bool = False
) -> np.ndarray:
    # Phases serve each distinct word of a bank once, but a shared atomic serves every lane.
    found = np.zeros((4, width.size), dtype=np.int64)
    lanes = _ATOMIC_GROUP_LANES if atomic else _GROUP_LANES
    for size in np.unique(width).tolist():
        at = np.flatnonzero(width == size)
        passes = _bank_passes(addresses[at], active[at], size, lanes[size], not atomic)
        achieved = passes.sum(axis=1)
        needed = -(-active[at].sum(axis=1) * size // WAVEFRONT_BYTES)
        found[0, at] = achieved
        found[1, at] = needed if atomic else np.minimum(needed, achieved)
        found[3, at] = np.maximum(passes - 1, 0).sum(axis=1)
    found[2] = np.maximum(found[0] - found[1], 0)
    return found


def _atomic_counts(
    addresses: np.ndarray, active: np.ndarray, width: np.ndarray, swap: np.ndarray
) -> np.ndarray:
    # A compare-and-swap repeats every pass, its ideal included.
    found = _bank_counts(addresses, active, width, atomic=True)
    found[[0, 1, 3]] *= np.where(swap, 2, 1)
    found[2] = np.maximum(found[0] - found[1], 0)
    return found


def _bank_passes(
    addresses: np.ndarray, active: np.ndarray, width: int, group_lanes: int, distinct: bool
) -> np.ndarray:
    # Lay out the words of each lane group in a row of its own, with idle lanes as _NO_WORD,
    # then count the words of each bank and take the busiest bank of each group.
    count = len(addresses)
    groups = WARP_SIZE // group_lanes
    word_count = max(width // WORD_BYTES, 1)
    first = addresses // np.uint64(WORD_BYTES)
    lanes = first[:, :, None] + np.arange(word_count, dtype=np.uint64)
    live = np.broadcast_to(active[:, :, None], lanes.shape)
    found = np.where(live, lanes, _NO_WORD).reshape(count, groups, -1)
    if distinct:
        found.sort(axis=2)
        live = found != _NO_WORD
        live[:, :, 1:] &= found[:, :, 1:] != found[:, :, :-1]
    else:
        live = found != _NO_WORD
    banks = (found % np.uint64(BANKS)).astype(np.intp)
    cells = np.arange(count * groups, dtype=np.intp).reshape(count, groups, 1) * BANKS + banks
    degrees = np.bincount(cells[live], minlength=count * groups * BANKS)
    return degrees.reshape(count, groups, BANKS).max(axis=2)


def _sector_counts(addresses: np.ndarray, active: np.ndarray, width: np.ndarray) -> np.ndarray:
    # An aligned lane of up to 16 bytes falls inside one 32-byte sector.
    found = np.where(active, addresses // np.uint64(SECTOR_BYTES), _NO_WORD)
    found.sort(axis=1)
    live = found != _NO_WORD
    live[:, 1:] &= found[:, 1:] != found[:, :-1]
    sectors = live.sum(axis=1)
    needed = -(-active.sum(axis=1) * width.astype(np.int64) // SECTOR_BYTES)
    ideal = np.minimum(needed, sectors)
    return np.stack([sectors, ideal, np.maximum(sectors - ideal, 0), np.zeros_like(sectors)])


def _scalar_counts(trace: Trace, rows: np.ndarray, opcodes: list[str]) -> np.ndarray:
    # A move of every lane by whole 128-byte lines keeps the banks, the equal words, and the
    # offsets inside global lines. Rows that differ by such a move share one call of the model.
    arch = str(trace.metadata.get("arch", ""))
    found = np.zeros((4, rows.size), dtype=np.int64)
    seen: dict[tuple, tuple[int, int, int, int]] = {}
    for index, (row, lanes, opcode) in enumerate(
        zip(rows.tolist(), _lane_tuples(trace, rows), opcodes, strict=True)
    ):
        width, kind = int(trace.width[row]), int(trace.kind[row])
        if trace.space[row] == Space.SHARED:
            sources = trace.copy_sources.get(row)
            if sources is not None and _reads_all(lanes, sources):
                sources = _moved(sources)
            key = (opcode, width, kind, _moved(lanes), sources)
            if key not in seen:
                access = shared_memory.analyze(
                    key[3],
                    width,
                    store=kind == Kind.STORE,
                    atomic=kind == Kind.ATOMIC,
                    arch=arch,
                    opcode=opcode,
                    sources=sources,
                )
                seen[key] = (access.wavefronts, access.ideal, access.excessive, access.conflicts)
        else:
            key = (_moved(lanes), _moved(trace.copy_destinations[row]))
            if key not in seen:
                copied = global_memory.copy_sectors(*key)
                seen[key] = (copied.sectors, copied.ideal, copied.excessive, 0)
        found[:, index] = seen[key]
    return found


def _lane_tuples(trace: Trace, rows: np.ndarray) -> list[tuple[int | None, ...]]:
    return [
        tuple(address if on else None for address, on in zip(lanes, live, strict=True))
        for lanes, live in zip(
            trace.addr[rows].tolist(), trace.active()[rows].tolist(), strict=True
        )
    ]


def _moved(lanes: tuple[int | None, ...]) -> tuple[int | None, ...]:
    present = [address for address in lanes if address is not None]
    if not present:
        return lanes
    base = min(present) // WAVEFRONT_BYTES * WAVEFRONT_BYTES
    return tuple(None if address is None else address - base for address in lanes)


def _reads_all(rows: tuple[int | None, ...], sources: tuple[int | None, ...]) -> bool:
    # A lane without a global address reads at a fixed offset of its lane number or fills
    # zeros, neither of which moves with the other sources.
    return all(
        source is not None for row, source in zip(rows, sources, strict=True) if row is not None
    )


def _opcodes(trace: Trace, rows: np.ndarray) -> list[str]:
    pcs, which = np.unique(trace.pc[rows], return_inverse=True)
    codes = [trace.opcode(pc) for pc in pcs.tolist()]
    return [codes[index] for index in which.reshape(-1).tolist()]


def _opcode_flags(opcodes: list[str], test: Callable[[str], bool]) -> np.ndarray:
    flags = {code: test(code) for code in set(opcodes)}
    return np.array([flags[code] for code in opcodes], dtype=bool)


def _base(opcode: str) -> str:
    return opcode.split(".")[0]
