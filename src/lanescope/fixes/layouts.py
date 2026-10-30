import dataclasses

import numpy as np

from lanescope.model.counters import InstructionCounters, count
from lanescope.trace import Space, Trace

# Shared memory windows are far smaller than this limit, which keeps every
# offset and size well inside the uint64 arithmetic of the traces.
MAX_OFFSET_BITS = 48
MAX_OFFSET = 1 << MAX_OFFSET_BITS
# Later buffers keep the alignment of the widest access, 16 bytes.
ALIGNMENT = 16


@dataclasses.dataclass(frozen=True)
class Buffer:
    """A byte range of the shared memory window."""

    start: int
    size: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.size <= 0:
            raise ValueError("a buffer needs a start of zero or more and a positive size")
        if self.start + self.size > MAX_OFFSET:
            raise ValueError(f"a buffer must end at or below {MAX_OFFSET:#x}")

    @property
    def end(self) -> int:
        """Return the first byte offset past the buffer."""
        return self.start + self.size

    def contains(self, addresses: np.ndarray) -> np.ndarray:
        """Return which addresses fall inside the buffer."""
        return (addresses >= self.start) & (addresses < self.end)


@dataclasses.dataclass(frozen=True)
class Padding:
    """Pad bytes appended to every row of a row-major buffer."""

    row_bytes: int
    pad_bytes: int

    def check(self, buffer: Buffer) -> None:
        """Raise ValueError when the padding does not fit the buffer."""
        if self.row_bytes <= 0 or self.pad_bytes < 0:
            raise ValueError("padding needs a positive row size and a pad of zero or more bytes")
        if self.row_bytes + self.pad_bytes > MAX_OFFSET:
            raise ValueError(f"a padded row must stay at or below {MAX_OFFSET:#x} bytes")
        if buffer.size % self.row_bytes:
            raise ValueError(
                f"a buffer of {buffer.size} bytes is no whole number of {self.row_bytes}-byte rows"
            )
        if buffer.end + self.extra_bytes(buffer) + ALIGNMENT > MAX_OFFSET:
            raise ValueError(f"the padded buffer must end below {MAX_OFFSET:#x}")

    def keeps_whole(self, width: int) -> bool:
        """Return whether every aligned access of this width stays inside one row."""
        return self.row_bytes % width == 0

    def move(self, offsets: np.ndarray) -> np.ndarray:
        """Return the new byte offsets inside the buffer."""
        row, column = np.divmod(offsets, np.uint64(self.row_bytes))
        return row * np.uint64(self.row_bytes + self.pad_bytes) + column

    def extra_bytes(self, buffer: Buffer) -> int:
        """Return the bytes the buffer gains."""
        return buffer.size // self.row_bytes * self.pad_bytes

    def describe(self) -> str:
        """Return a short description of the change."""
        return f"pad each {self.row_bytes}-byte row by {self.pad_bytes} bytes"


@dataclasses.dataclass(frozen=True)
class XorSwizzle:
    """The CuTe Swizzle<bits, base, shift> applied to byte offsets.

    A CuTe swizzle on the offsets of elements of 2**e bytes equals the byte
    swizzle with base + e in place of base.
    """

    bits: int
    base: int
    shift: int

    def check(self, buffer: Buffer) -> None:
        """Raise ValueError when the swizzle is no permutation of the buffer."""
        if self.bits <= 0 or self.base < 0:
            raise ValueError("a swizzle needs one or more bits and a base of zero or more")
        # As in CuTe, the XOR source bits must lie above the target bits.
        if self.shift < self.bits:
            raise ValueError("a swizzle needs a shift of at least its bit count")
        if self.base + self.bits + self.shift > MAX_OFFSET_BITS:
            raise ValueError(f"a swizzle must act on offsets below 2**{MAX_OFFSET_BITS}")
        block = 1 << (self.base + self.bits)
        if buffer.size % block:
            raise ValueError(f"the swizzle needs a buffer size that is a multiple of {block} bytes")

    def keeps_whole(self, width: int) -> bool:
        """Return whether every aligned access of this width moves as one piece."""
        return width <= 1 << self.base

    def move(self, offsets: np.ndarray) -> np.ndarray:
        """Return the new byte offsets inside the buffer."""
        mask = np.uint64(((1 << self.bits) - 1) << self.base)
        return offsets ^ ((offsets >> np.uint64(self.shift)) & mask)

    def extra_bytes(self, buffer: Buffer) -> int:
        """Return the bytes the buffer gains, which is none."""
        return 0

    def describe(self) -> str:
        """Return a short description of the change."""
        return f"XOR swizzle Swizzle<{self.bits},{self.base},{self.shift}> on byte offsets"


Change = Padding | XorSwizzle


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Counters of one instruction before and after a layout change."""

    before: InstructionCounters
    after: InstructionCounters


def infer_buffer(trace: Trace, pc: int | None = None) -> Buffer:
    """Return the 16-byte-aligned range of the shared accesses of one PC, or of all PCs.

    The range of one PC takes in the range of every other shared instruction that
    overlaps it, until none does. A load that reads part of a tile thus gets the whole
    tile that its stores fill. A wider rounding, such as to 128-byte lines, would pull in
    the first bytes of a neighboring array whenever an array size is no multiple of the
    rounding.
    """
    ranges = _shared_ranges(trace)
    if pc is None:
        if not ranges:
            raise ValueError("no shared access to infer a buffer from")
        start, end = (
            min(low for low, _ in ranges.values()),
            max(high for _, high in ranges.values()),
        )
    else:
        if pc not in ranges:
            raise ValueError("no shared access to infer a buffer from")
        start, end = ranges[pc]
        grown = True
        while grown:
            grown = False
            for low, high in ranges.values():
                if low < end and start < high and (low < start or high > end):
                    start, end, grown = min(start, low), max(end, high), True
    start = start // ALIGNMENT * ALIGNMENT
    end = -(-end // ALIGNMENT) * ALIGNMENT
    return Buffer(start, end - start)


def _shared_ranges(trace: Trace) -> dict[int, tuple[int, int]]:
    shared = (trace.space == Space.SHARED)[:, None] & trace.active()
    ends = trace.addr + trace.width[:, None].astype(np.uint64)
    ranges: dict[int, tuple[int, int]] = {}
    for pc in np.unique(trace.pc[shared.any(axis=1)]).tolist():
        lanes = shared & (trace.pc == pc)[:, None]
        ranges[int(pc)] = (int(trace.addr[lanes].min()), int(ends[lanes].max()))
    return ranges


def later_shift(buffer: Buffer, change: Change) -> int:
    """Return the bytes by which the shared addresses above the buffer move up."""
    extra = change.extra_bytes(buffer)
    return -(-extra // ALIGNMENT) * ALIGNMENT


def lanes_inside(trace: Trace, buffer: Buffer) -> np.ndarray:
    """Return which lanes access the buffer, raising ValueError for an access across its edge."""
    shared = (trace.space == Space.SHARED)[:, None] & trace.active()
    end = trace.addr + trace.width[:, None].astype(np.uint64)
    inside = shared & (trace.addr >= buffer.start) & (end <= buffer.end)
    crossing = shared & (trace.addr < buffer.end) & (end > buffer.start) & ~inside
    if crossing.any():
        row = _first_row(crossing)
        raise ValueError(
            f"the {int(trace.width[row])}-byte accesses of pc {int(trace.pc[row]):#x} "
            "cross the edge of the buffer"
        )
    return inside


def apply(trace: Trace, buffer: Buffer, change: Change) -> Trace:
    """Return the trace with every shared access inside the buffer moved by the change."""
    change.check(buffer)
    inside = lanes_inside(trace, buffer)
    shared = (trace.space == Space.SHARED)[:, None] & trace.active()
    width = trace.width[:, None].astype(np.uint64)
    for row in np.flatnonzero(inside.any(axis=1)):
        if not change.keeps_whole(int(trace.width[row])):
            raise ValueError(
                f"{change.describe()} splits the {int(trace.width[row])}-byte accesses "
                f"of pc {int(trace.pc[row]):#x}"
            )
    addresses = trace.addr.copy()
    start = np.uint64(buffer.start)
    addresses[inside] = start + change.move(addresses[inside] - start)
    shift = later_shift(buffer, change)
    if shift:
        addresses[shared & (trace.addr >= buffer.end)] += np.uint64(shift)
    misaligned = inside & (addresses % width != 0)
    if misaligned.any():
        row = _first_row(misaligned)
        raise ValueError(
            f"{change.describe()} misaligns the {int(trace.width[row])}-byte accesses "
            f"of pc {int(trace.pc[row]):#x}"
        )
    return trace.with_addresses(addresses)


def compare(trace: Trace, buffer: Buffer, change: Change) -> list[Comparison]:
    """Return the counters before and after the change of every instruction it affects."""
    changed = apply(trace, buffer, change)
    active = trace.active()
    shared = (trace.space == Space.SHARED)[:, None] & active
    affected = ((changed.addr != trace.addr) & active).any(axis=1) | (
        shared & buffer.contains(trace.addr)
    ).any(axis=1)
    rows = rows_of_instructions(trace, affected)
    # Counting a subset numbers its rows from zero, so worst rows map back to the trace.
    index = np.flatnonzero(rows)
    before = count(trace.select(rows))
    after = count(changed.select(rows))
    return [
        Comparison(_in_trace(old, index), _in_trace(new, index))
        for old, new in zip(before, after, strict=True)
    ]


def rows_of_instructions(trace: Trace, picked: np.ndarray) -> np.ndarray:
    """Return every row of each instruction that has a picked row."""
    keys = np.column_stack([trace.pc, trace.space, trace.kind, trace.width]).astype(np.uint64)
    _, group = np.unique(keys, axis=0, return_inverse=True)
    group = group.reshape(-1)
    return np.isin(group, np.unique(group[picked]))


def _in_trace(counters: InstructionCounters, index: np.ndarray) -> InstructionCounters:
    return dataclasses.replace(counters, worst_row=int(index[counters.worst_row]))


def _first_row(flags: np.ndarray) -> int:
    return int(np.argwhere(flags)[0][0])
