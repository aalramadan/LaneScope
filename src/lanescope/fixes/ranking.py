import dataclasses
import math

import numpy as np

from lanescope.fixes.layouts import (
    ALIGNMENT,
    Buffer,
    Change,
    Padding,
    XorSwizzle,
    apply,
    lanes_inside,
)
from lanescope.model.counters import distinct_rows, measure
from lanescope.trace import Trace

PADS = (4, 8, 16)
MAX_SWIZZLE_BITS = 5
MAX_SHIFT = 8
MIN_ROW = 16


@dataclasses.dataclass(frozen=True)
class Suggestion:
    """One layout change with the shared-memory counts it predicts for the buffer."""

    change: Change
    wavefronts_before: int
    wavefronts_after: int
    excessive_before: int
    excessive_after: int
    extra_bytes: int


def divisors(number: int) -> list[int]:
    """Return the divisors of a positive number in ascending order."""
    small = [d for d in range(1, math.isqrt(number) + 1) if number % d == 0]
    large = [number // d for d in reversed(small) if d * d != number]
    return small + large


def candidates(buffer: Buffer, widest: int) -> list[Change]:
    """Return the paddings and swizzles that keep accesses of `widest` bytes whole and aligned."""
    unit = max(widest, 4)
    found: list[Change] = [
        Padding(row, pad)
        for row in divisors(buffer.size)
        if row >= max(MIN_ROW, unit) and row % unit == 0
        for pad in PADS
        if pad % widest == 0
    ]
    for base in range(max(widest.bit_length() - 1, 2), ALIGNMENT.bit_length()):
        for bits in range(1, MAX_SWIZZLE_BITS + 1):
            for shift in range(bits, MAX_SHIFT + 1):
                if buffer.size % (1 << (base + bits)) == 0:
                    found.append(XorSwizzle(bits, base, shift))
    return found


def suggest(trace: Trace, buffer: Buffer) -> list[Suggestion]:
    """Return every candidate change that lowers the excess wavefronts of the buffer, best first."""
    on_buffer = lanes_inside(trace, buffer).any(axis=1)
    if not on_buffer.any():
        raise ValueError("no shared access of the trace falls inside the buffer")
    accesses = trace.select(on_buffer)
    # A change moves equal rows alike, so each distinct row is moved and counted once and
    # weighted by its repeats. Recordings repeat each access in many CTAs.
    rows, repeats = distinct_rows(accesses)
    accesses = accesses.select(rows)
    wavefronts_before, excessive_before = _totals(accesses, repeats)
    widest = int(accesses.width.max())
    found = []
    for change in candidates(buffer, widest):
        try:
            changed = apply(accesses, buffer, change)
        except ValueError:
            continue
        wavefronts_after, excessive_after = _totals(changed, repeats)
        found.append(
            Suggestion(
                change=change,
                wavefronts_before=wavefronts_before,
                wavefronts_after=wavefronts_after,
                excessive_before=excessive_before,
                excessive_after=excessive_after,
                extra_bytes=change.extra_bytes(buffer),
            )
        )
    better = [s for s in found if s.excessive_after < s.excessive_before]
    return sorted(better, key=_rank)


def _totals(accesses: Trace, repeats: np.ndarray) -> tuple[int, int]:
    # Every row of the accesses is a shared access, as only those fall inside a buffer.
    wavefronts = excessive = 0
    for row, weight in enumerate(repeats.tolist()):
        achieved, _, excess, _ = measure(accesses, row)
        wavefronts += weight * achieved
        excessive += weight * excess
    return wavefronts, excessive


def _rank(suggestion: Suggestion) -> tuple[int, int, int, int]:
    change = suggestion.change
    bits = change.bits if isinstance(change, XorSwizzle) else 0
    shift = change.shift if isinstance(change, XorSwizzle) else 0
    return (suggestion.excessive_after, suggestion.extra_bytes, bits, shift)
