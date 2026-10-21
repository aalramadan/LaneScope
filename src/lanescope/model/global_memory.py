import dataclasses
import functools
import math
from collections.abc import Sequence

from lanescope.trace import check_lanes

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
