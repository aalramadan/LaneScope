from collections.abc import Sequence

import numpy as np

from lanescope.model.counters import InstructionCounters, InstructionKey, row_hashes
from lanescope.trace import Space, Trace

# A layout sets the addresses of an instruction, therefore its warps repeat a few
# patterns of lane offsets and conflict alike. Addresses that follow the values a kernel
# reads, such as the bins of a histogram, give nearly every warp a pattern of its own
# and conflicts that differ from warp to warp.
MIN_ACCESSES = 32
MIN_PATTERN_SHARE = 0.5
MIN_LEVELS = 3
_NO_LANE = np.iinfo(np.uint64).max


def data_dependent(trace: Trace, counters: Sequence[InstructionCounters]) -> set[InstructionKey]:
    """Return the shared instructions of a trace whose bank conflicts follow the input values."""
    candidates = [
        counter
        for counter in counters
        if counter.space == Space.SHARED
        and counter.excessive > 0
        and counter.levels >= MIN_LEVELS
        and counter.requests >= MIN_ACCESSES
    ]
    if not candidates:
        return set()
    active = trace.active()
    found = set()
    for counter in candidates:
        rows = np.flatnonzero(
            (trace.pc == counter.pc)
            & (trace.space == counter.space)
            & (trace.kind == counter.kind)
            & (trace.width == counter.width)
        )
        warps = np.unique(trace.cta[rows].astype(np.uint64) << np.uint64(32) | trace.warp[rows])
        if warps.size < 2:
            continue
        if _offset_patterns(trace, active, rows) >= MIN_PATTERN_SHARE * rows.size:
            found.add((counter.pc, int(counter.space), int(counter.kind), counter.width))
    return found


def _offset_patterns(trace: Trace, active: np.ndarray, rows: np.ndarray) -> int:
    """Return how many distinct patterns of lane offsets from their lowest lane the rows hold."""
    lanes = active[rows]
    addresses = trace.addr[rows]
    lowest = np.where(lanes, addresses, _NO_LANE).min(axis=1)
    offsets = np.where(lanes, addresses - lowest[:, None], _NO_LANE)
    keys = np.column_stack([trace.mask[rows].astype(np.uint64), offsets])
    return int(np.unique(row_hashes(keys)).size)
