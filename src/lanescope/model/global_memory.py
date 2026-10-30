import dataclasses
import functools
import math
from collections.abc import Sequence

from lanescope.model import async_copy
from lanescope.trace import Trace, check_lanes

SECTOR_BYTES = 32
LINE_BYTES = 128
SECTORS_PER_LINE = LINE_BYTES // SECTOR_BYTES


@dataclasses.dataclass(frozen=True)
class GlobalAccess:
    """Modeled sectors of one warp-level global memory access."""

    width: int
    active: int
    base: int
    sector_lanes: tuple[tuple[int, tuple[int, ...]], ...]
    # A 16-byte cp.async that bypasses L1 has an ideal that Nsight Compute takes from its
    # shared rows, two to a sector, in place of its active bytes.
    ideal_sectors: int | None = None

    @property
    def sectors(self) -> int:
        """Return the distinct sectors that the active lanes touch."""
        return len(self.sector_lanes)

    @property
    def lines(self) -> int:
        """Return the distinct 128-byte lines that the active lanes touch."""
        return len({sector // SECTORS_PER_LINE for sector, _ in self.sector_lanes})

    @property
    def ideal(self) -> int:
        """Return the sectors that the active bytes need, at most the sectors they touch."""
        if self.ideal_sectors is not None:
            return self.ideal_sectors
        return min(math.ceil(self.active * self.width / SECTOR_BYTES), self.sectors)

    @property
    def excessive(self) -> int:
        """Return the sectors above the ideal."""
        return max(0, self.sectors - self.ideal)

    def sector_address(self, sector: int) -> int:
        """Return the address of the first byte of a sector index of this access."""
        return self.base + sector * SECTOR_BYTES


def analyze(addresses: Sequence[int | None], width: int) -> GlobalAccess:
    """Model one access from the address of each lane, with None for inactive lanes."""
    check_lanes(addresses, width)
    active = [address for address in addresses if address is not None]
    # Sector and line counts do not change when all addresses move by whole lines.
    # Counting from the first line lets accesses of different CTAs share one cache entry.
    base = min(active) // LINE_BYTES * LINE_BYTES
    relative = tuple(None if address is None else address - base for address in addresses)
    return dataclasses.replace(_analyze(relative, width), base=base)


def analyze_row(trace: Trace, row: int) -> GlobalAccess:
    """Model one global row of a trace.

    A 16-byte cp.async that bypasses L1 reads one sector for each of its writes into shared
    memory, as Nsight Compute counts it, and so takes its count from both of its rows.
    """
    addresses, width = trace.lane_addresses(row), int(trace.width[row])
    destinations = trace.copy_destinations.get(row)
    if (
        destinations is None
        or width != async_copy.ROW_BYTES
        or "BYPASS" not in trace.opcode(int(trace.pc[row]))
    ):
        return analyze(addresses, width)
    return copy_sectors(addresses, destinations)


def copy_sectors(sources: Sequence[int | None], destinations: Sequence[int | None]) -> GlobalAccess:
    """Model the sectors of a 16-byte cp.async that bypasses L1, one for each write.

    Only the lanes with a global address read, as the others fill zeros.
    """
    reads = tuple(
        None if source is None else destination
        for source, destination in zip(sources, destinations, strict=True)
    )
    writes = async_copy.copies(sources, reads)
    if not writes:
        return GlobalAccess(async_copy.ROW_BYTES, 0, 0, (), 0)
    base = min(write.source for write in writes) // LINE_BYTES * LINE_BYTES
    sector_lanes = tuple(((write.source - base) // SECTOR_BYTES, write.lanes) for write in writes)
    rows = {address // async_copy.ROW_BYTES for address in reads if address is not None}
    active = sum(address is not None for address in reads)
    ideal = math.ceil(len(rows) * async_copy.ROW_BYTES / SECTOR_BYTES)
    return GlobalAccess(async_copy.ROW_BYTES, active, base, sector_lanes, ideal)


@functools.lru_cache(maxsize=1 << 16)
def _analyze(relative: tuple[int | None, ...], width: int) -> GlobalAccess:
    lanes_of: dict[int, list[int]] = {}
    for lane, offset in enumerate(relative):
        if offset is None:
            continue
        for sector in range(offset // SECTOR_BYTES, (offset + width - 1) // SECTOR_BYTES + 1):
            lanes_of.setdefault(sector, []).append(lane)
    active = sum(offset is not None for offset in relative)
    sector_lanes = tuple((sector, tuple(lanes)) for sector, lanes in sorted(lanes_of.items()))
    return GlobalAccess(width, active, 0, sector_lanes)
