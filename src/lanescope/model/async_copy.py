import dataclasses
from collections.abc import Sequence

ROW_BYTES = 16
SECTOR_BYTES = 32
LINE_BYTES = 128
SECTORS = LINE_BYTES // SECTOR_BYTES
ROWS_PER_LINE = LINE_BYTES // ROW_BYTES
# One wavefront of a 16-byte cp.async moves its writes from a global line into a shared
# line by one rotation or one reflection of the four 32-byte sectors, as measured on an
# RTX 4060 with Nsight Compute.
SECTOR_MAPS = tuple(
    tuple((shift + sign * sector) % SECTORS for sector in range(SECTORS))
    for sign in (1, -1)
    for shift in range(SECTORS)
)


@dataclasses.dataclass(frozen=True)
class Copy:
    """One write of a 16-byte cp.async: a sector from a lane pair or a row from one lane."""

    lanes: tuple[int, ...]
    source: int
    row: int
    swapped: bool

    @property
    def rows(self) -> tuple[int, ...]:
        """Return the 16-byte shared rows that the write fills."""
        return (self.row, self.row + 1) if len(self.lanes) == 2 else (self.row,)

    @property
    def group(self) -> tuple[int, bool, int, int, int]:
        """Return what writes must share to move in one wavefront.

        That is the size, whether the 16-byte halves change places, the shared half of a
        lone row, the global line, and the shared line.
        """
        half = self.row % 2 if len(self.lanes) == 1 else 0
        line = self.row // ROWS_PER_LINE
        return (len(self.lanes), self.swapped, half, self.source // LINE_BYTES, line)

    @property
    def sectors(self) -> tuple[int, int]:
        """Return the sector that the write comes from and the sector it goes to, in their lines."""
        return self.source % LINE_BYTES // SECTOR_BYTES, self.row % ROWS_PER_LINE // 2


def copies(sources: Sequence[int | None], destinations: Sequence[int | None]) -> tuple[Copy, ...]:
    """Return the writes of one 16-byte cp.async in lane order.

    Two lanes of a pair that copy the two halves of one global sector into the two halves
    of one aligned shared sector make one write. Every other lane writes its row alone,
    unless an earlier write already fills that row.
    """
    written: set[int] = set()
    found: list[Copy] = []
    for first in range(0, len(destinations), 2):
        lanes = [lane for lane in (first, first + 1) if destinations[lane] is not None]
        pair = _pair(lanes, sources, destinations) if len(lanes) == 2 else None
        if pair is not None and not written.intersection(pair.rows):
            found.append(pair)
            written.update(pair.rows)
            continue
        for lane in lanes:
            row = _row(destinations, lane)
            if row not in written:
                written.add(row)
                source = _source(sources, lane)
                found.append(Copy((lane,), source, row, source // ROW_BYTES % 2 != row % 2))
    return tuple(found)


def wavefronts(writes: Sequence[Copy]) -> list[list[Copy]]:
    """Return the writes of each wavefront.

    Each write joins the first wavefront of its group whose sector maps still allow it.
    """
    open_waves: list[tuple[tuple, list[tuple[int, ...]], list[Copy]]] = []
    for write in writes:
        source, target = write.sectors
        for group, maps, members in open_waves:
            fitting = [mapping for mapping in maps if mapping[source] == target]
            if group == write.group and fitting:
                maps[:] = fitting
                members.append(write)
                break
        else:
            fitting = [mapping for mapping in SECTOR_MAPS if mapping[source] == target]
            open_waves.append((write.group, fitting, [write]))
    return [members for _, _, members in open_waves]


def _pair(
    lanes: list[int], sources: Sequence[int | None], destinations: Sequence[int | None]
) -> Copy | None:
    a, b = lanes
    source_a, source_b = _source(sources, a), _source(sources, b)
    row_a, row_b = _row(destinations, a), _row(destinations, b)
    if source_a // SECTOR_BYTES != source_b // SECTOR_BYTES or source_a == source_b:
        return None
    if row_a // 2 != row_b // 2 or row_a == row_b:
        return None
    swapped = (source_a < source_b) != (row_a < row_b)
    return Copy((a, b), min(source_a, source_b), min(row_a, row_b), swapped)


def _row(destinations: Sequence[int | None], lane: int) -> int:
    address = destinations[lane]
    assert address is not None
    return address // ROW_BYTES


def _source(sources: Sequence[int | None], lane: int) -> int:
    # Without its global address, a lane reads the 16 bytes after those of the lane before
    # it, as when each group of eight lanes copies one 128-byte line.
    address = sources[lane]
    return ROW_BYTES * lane if address is None else address
