import dataclasses
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
# A layout also repeats its patterns in every launch, while values differ between
# launches. The launches of a line together judge it once they number this many.
MIN_LAUNCHES = 2
# The evidence of one trace keeps at most this many patterns, enough for the few warps
# of a launch that alone tell too little.
MAX_PATTERNS = 64
_NO_LANE = np.iinfo(np.uint64).max


@dataclasses.dataclass(frozen=True)
class InputEvidence:
    """What the accesses of one shared instruction with excess tell of the input values.

    `patterns` holds the hashes of the distinct patterns of lane offsets, or None for
    more than MAX_PATTERNS of them.
    """

    rows: int
    patterns: frozenset[int] | None
    excess_counts: frozenset[int]
    data_dependent: bool


def input_evidence(
    trace: Trace, counters: Sequence[InstructionCounters]
) -> dict[InstructionKey, InputEvidence]:
    """Return the evidence of each shared instruction with excess, judged as for one trace."""
    shared = [
        counter for counter in counters if counter.space == Space.SHARED and counter.excessive > 0
    ]
    if not shared:
        return {}
    active = trace.active()
    found = {}
    for counter in shared:
        rows = np.flatnonzero(
            (trace.pc == counter.pc)
            & (trace.space == counter.space)
            & (trace.kind == counter.kind)
            & (trace.width == counter.width)
        )
        patterns = _offset_patterns(trace, active, rows)
        warps = np.unique(trace.cta[rows].astype(np.uint64) << np.uint64(32) | trace.warp[rows])
        dependent = (
            counter.levels >= MIN_LEVELS
            and counter.requests >= MIN_ACCESSES
            and warps.size >= 2
            and patterns.size >= MIN_PATTERN_SHARE * rows.size
        )
        key = (counter.pc, int(counter.space), int(counter.kind), counter.width)
        found[key] = InputEvidence(
            rows=int(rows.size),
            patterns=frozenset(patterns.tolist()) if patterns.size <= MAX_PATTERNS else None,
            excess_counts=frozenset(counter.excess_counts),
            data_dependent=dependent,
        )
    return found


def data_dependent(trace: Trace, counters: Sequence[InstructionCounters]) -> set[InstructionKey]:
    """Return the shared instructions of a trace whose bank conflicts follow the input values."""
    return {key for key, found in input_evidence(trace, counters).items() if found.data_dependent}


def follows_input_over_launches(
    rows: int, launches: int, patterns: frozenset[int] | None, excess_counts: frozenset[int]
) -> bool:
    """Return whether the accesses of a line over all its launches follow the input values."""
    return (
        patterns is not None
        and rows >= MIN_ACCESSES
        and launches >= MIN_LAUNCHES
        and len(excess_counts) >= MIN_LEVELS
        and len(patterns) >= MIN_PATTERN_SHARE * rows
    )


def _offset_patterns(trace: Trace, active: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return the hashes of the distinct patterns of lane offsets from their lowest lane."""
    lanes = active[rows]
    addresses = trace.addr[rows]
    lowest = np.where(lanes, addresses, _NO_LANE).min(axis=1)
    offsets = np.where(lanes, addresses - lowest[:, None], _NO_LANE)
    keys = np.column_stack([trace.mask[rows].astype(np.uint64), offsets])
    return np.unique(row_hashes(keys))
