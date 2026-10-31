import dataclasses
import functools
import math
import re
from collections.abc import Sequence

from lanescope.model import async_copy
from lanescope.trace import WARP_SIZE, Kind, Trace, check_lanes

BANKS = 32
WORD_BYTES = 4
WAVEFRONT_BYTES = BANKS * WORD_BYTES

_ALL_LANES = tuple(range(WARP_SIZE))
_HALVES = (tuple(range(0, 16)), tuple(range(16, 32)))
_QUARTERS = tuple(tuple(range(start, start + 8)) for start in range(0, WARP_SIZE, 8))

_FOURS = tuple(tuple(range(start, start + 4)) for start in range(0, WARP_SIZE, 4))

_GROUPS = {1: (_ALL_LANES,), 2: (_ALL_LANES,), 4: (_ALL_LANES,), 8: _HALVES, 16: _QUARTERS}
# The lanes inside which a load merges its groups into one.
_MERGE_SPANS = {8: (_ALL_LANES,), 16: _HALVES}
# The slots that a block of four lanes may take in a merged wavefront of a 16-byte load.
_SLOT_LIMIT = 2
# On the RTX 4060, an active lane 31 makes the first half-warp of a 16-byte load merge
# only under strict pairing.
_LAST_LANE = WARP_SIZE - 1
# A 16-byte load whose active lanes read one address fits one wavefront for the
# whole warp up to this many slots.
_BROADCAST_SLOTS = 8

# Lane i pairs with lane i XOR 1 in the pattern a,a,b,b and with lane i XOR 2 in
# the pattern a,b,a,b. Two paired lanes take one slot of a merged wavefront.
_PAIR_DISTANCES = (1, 2)

# The first compute capability measured to merge only full warps (H100 is sm_90).
_FULL_WARP_MERGE_MAJOR = 9

# An ldmatrix or stmatrix moves one 8x8 matrix for each group of eight lanes. Its
# SASS opcode ends in .2 or .4 for two or four matrices.
_MATRIX_OPCODES = ("LDSM", "STSM")
_ASYNC_COPY_OPCODE = "LDGSTS"
# A shared atomic writes each lane on its own, and a compare-and-swap moves two words.
_ATOMIC_OPCODE = "ATOMS"
_SWAP_SUFFIX = ".CAS"


@dataclasses.dataclass(frozen=True)
class Phase:
    """One group of active lanes that the banks serve together."""

    lanes: tuple[int, ...]
    bank_words: tuple[tuple[int, ...], ...]

    @functools.cached_property
    def wavefronts(self) -> int:
        """Return the most words that one bank serves in this phase."""
        return max(len(words) for words in self.bank_words)

    @property
    def worst_bank(self) -> int:
        """Return the lowest bank that serves the most distinct words."""
        counts = [len(words) for words in self.bank_words]
        return counts.index(max(counts))


@dataclasses.dataclass(frozen=True)
class SharedAccess:
    """Modeled wavefronts of one warp-level shared memory access."""

    width: int
    active: int
    paired: bool
    ideal: int
    phases: tuple[Phase, ...]

    @functools.cached_property
    def wavefronts(self) -> int:
        """Return the wavefronts of all phases."""
        return sum(phase.wavefronts for phase in self.phases)

    @property
    def excessive(self) -> int:
        """Return the wavefronts above the ideal, never below zero."""
        return max(0, self.wavefronts - self.ideal)

    @property
    def conflicts(self) -> int:
        """Return the bank conflicts, one for each wavefront after the first of a phase."""
        return sum(phase.wavefronts - 1 for phase in self.phases)


def analyze(
    addresses: Sequence[int | None],
    width: int,
    *,
    store: bool = False,
    atomic: bool = False,
    arch: str = "",
    opcode: str = "",
    sources: Sequence[int | None] | None = None,
) -> SharedAccess:
    """Model one access from the byte offset of each lane, with None for inactive lanes.

    The architecture, such as sm_89, decides how loads merge their groups. An
    unknown one takes the rule measured on the RTX 4060. The SASS opcode, such as
    LDSM.16.M88.4, selects the rule of an ldmatrix, stmatrix, cp.async, or atomic access.
    The global address of each lane of a cp.async shapes its writes. Without them, the
    lanes read consecutive rows, and a lane with None among them fills zeros.
    """
    check_lanes(addresses, width)
    base = opcode.split(".")[0]
    if base in _MATRIX_OPCODES:
        matrices = 4 if opcode.endswith(".4") else 2 if opcode.endswith(".2") else 1
        return _matrix(tuple(addresses), width, matrices)
    if atomic or base == _ATOMIC_OPCODE:
        return _atomic(tuple(addresses), width, _SWAP_SUFFIX in opcode)
    if base == _ASYNC_COPY_OPCODE and store:
        lanes = sources if sources is not None else [width * lane for lane in range(len(addresses))]
        return _async_copy(tuple(addresses), width, tuple(lanes), "BYPASS" in opcode)
    return _analyze(tuple(addresses), width, store, _full_warp_merge(arch))


def analyze_row(trace: Trace, row: int) -> SharedAccess:
    """Model one shared row of a trace with the rules of the GPU that recorded it."""
    return analyze(
        trace.lane_addresses(row),
        int(trace.width[row]),
        store=trace.kind[row] == Kind.STORE,
        atomic=trace.kind[row] == Kind.ATOMIC,
        arch=str(trace.metadata.get("arch", "")),
        opcode=trace.opcode(int(trace.pc[row])),
        sources=trace.copy_sources.get(row),
    )


def words(address: int, width: int) -> range:
    """Return the indices of the 4-byte words that one lane's access covers."""
    return range(address // WORD_BYTES, (address + width - 1) // WORD_BYTES + 1)


@functools.lru_cache(maxsize=1 << 16)
def _analyze(
    addresses: tuple[int | None, ...], width: int, store: bool, full_warp: bool
) -> SharedAccess:
    groups = _GROUPS[width]
    paired = False
    if not store and width in _MERGE_SPANS:
        groups, paired = _merge(addresses, width, full_warp)
    phases = []
    for group in groups:
        lanes = tuple(lane for lane in group if addresses[lane] is not None)
        if lanes:
            phases.append(_phase(addresses, lanes, width))
    active = sum(address is not None for address in addresses)
    # Nsight Compute counts the bytes of the active lanes, but never more
    # wavefronts than the access takes.
    wavefronts = sum(phase.wavefronts for phase in phases)
    ideal = min(math.ceil(active * width / WAVEFRONT_BYTES), wavefronts)
    return SharedAccess(width, active, paired, ideal, tuple(phases))


@functools.lru_cache(maxsize=1 << 16)
def _matrix(addresses: tuple[int | None, ...], width: int, matrices: int) -> SharedAccess:
    # Each group of eight lanes gives the rows of one matrix, and the banks serve one
    # matrix at a time. Lanes past the last matrix give addresses that the GPU ignores.
    phases = []
    for group in _QUARTERS[:matrices]:
        if lanes := _active(addresses, group):
            phases.append(_phase(addresses, lanes, width))
    return _with_ideal(addresses, width, tuple(phases))


@functools.lru_cache(maxsize=1 << 16)
def _async_copy(
    addresses: tuple[int | None, ...], width: int, sources: tuple[int | None, ...], bypass: bool
) -> SharedAccess:
    if width == 16 and bypass:
        return _async_copy_rows(addresses, sources)
    # Through L1, a copy writes the lanes that read one global line together, each lane on
    # its own. A bank serves one lane per wavefront, also when lanes share an address, and
    # lanes that read the same bytes take turns.
    lines: dict[int, list[dict[int, int]]] = {}
    for lane in _active(addresses, _ALL_LANES):
        source = sources[lane]
        address = width * lane if source is None else source
        turns = lines.setdefault(address // WAVEFRONT_BYTES, [])
        free = next((turn for turn in turns if address not in turn), None)
        if free is None:
            free = {}
            turns.append(free)
        free[address] = lane
    phases = [
        _lane_phase(addresses, tuple(sorted(turn.values())), width)
        for turns in lines.values()
        for turn in turns
    ]
    return _with_ideal(addresses, width, tuple(phases))


def _async_copy_rows(
    addresses: tuple[int | None, ...], sources: tuple[int | None, ...]
) -> SharedAccess:
    active = _active(addresses, _ALL_LANES)
    zeros = {lane for lane in active if sources[lane] is None}
    first: dict[int | None, int] = {}
    for lane in active:
        first.setdefault(addresses[lane], lane)
    # A lane that fills zeros sends the whole copy through a pass that writes every active
    # lane before the lanes with data. Data for a row that an earlier lane fills with zeros
    # does not reach shared memory.
    data = tuple(
        address if lane in active and lane not in zeros and first[address] not in zeros else None
        for lane, address in enumerate(addresses)
    )
    phases = [_lane_phase(addresses, lanes, 16) for lanes in _zero_groups(active, zeros, sources)]
    phases += _copy_phases(data, sources)
    ideal = _rows(data) + (_rows(addresses) if zeros else 0)
    return SharedAccess(16, len(active), False, ideal, tuple(phases))


def _zero_groups(
    active: tuple[int, ...], zeros: set[int], sources: tuple[int | None, ...]
) -> list[tuple[int, ...]]:
    # The pass writes the lanes of one global line together. A lane that fills zeros has no
    # address of its own and joins the line of the nearest lane with data among its eight,
    # or the group of its eight lanes when none of them reads.
    if not zeros:
        return []
    groups: dict[tuple[str, int], list[int]] = {}
    for lane in active:
        reader = lane if lane not in zeros else _nearest_reader(lane, active, zeros, sources)
        source = sources[reader] if reader is not None else None
        key = ("quarter", lane // 8) if source is None else ("line", source // WAVEFRONT_BYTES)
        groups.setdefault(key, []).append(lane)
    return [tuple(lanes) for lanes in groups.values()]


def _nearest_reader(
    lane: int, active: tuple[int, ...], zeros: set[int], sources: tuple[int | None, ...]
) -> int | None:
    start = lane // 8 * 8
    readers = [
        other
        for other in range(start, start + 8)
        if other in active and other not in zeros and sources[other] is not None
    ]
    return min(readers, key=lambda other: (abs(other - lane), other < lane), default=None)


def _copy_phases(addresses: tuple[int | None, ...], sources: tuple[int | None, ...]) -> list[Phase]:
    # Each wavefront shows the lanes of its writes and the lanes whose rows those fill.
    if not _active(addresses, _ALL_LANES):
        return []
    writer: dict[int, int] = {}
    waves = async_copy.wavefronts(async_copy.copies(sources, addresses))
    for index, members in enumerate(waves):
        for write in members:
            writer.update((row, index) for row in write.rows)
    lanes: list[list[int]] = [[] for _ in waves]
    for lane in _active(addresses, _ALL_LANES):
        address = addresses[lane]
        assert address is not None
        lanes[writer[address // async_copy.ROW_BYTES]].append(lane)
    return [_phase(addresses, tuple(found), 16) for found in lanes]


def _rows(addresses: tuple[int | None, ...]) -> int:
    distinct = {address for address in addresses if address is not None}
    return math.ceil(len(distinct) * async_copy.ROW_BYTES / WAVEFRONT_BYTES)


@functools.lru_cache(maxsize=1 << 16)
def _atomic(addresses: tuple[int | None, ...], width: int, swap: bool) -> SharedAccess:
    # Lanes that hit one word still take a turn each. A compare-and-swap takes two passes.
    lanes = _active(addresses, _ALL_LANES)
    phase = _lane_phase(addresses, lanes, width)
    passes = 2 if swap else 1
    ideal = passes * math.ceil(len(lanes) * width / WAVEFRONT_BYTES)
    return SharedAccess(width, len(lanes), False, ideal, (phase,) * passes)


def _with_ideal(
    addresses: tuple[int | None, ...], width: int, phases: tuple[Phase, ...]
) -> SharedAccess:
    active = sum(address is not None for address in addresses)
    ideal = min(math.ceil(active * width / WAVEFRONT_BYTES), len(phases))
    return SharedAccess(width, active, False, ideal, phases)


def _lane_phase(addresses: tuple[int | None, ...], lanes: tuple[int, ...], width: int) -> Phase:
    bank_words: list[list[int]] = [[] for _ in range(BANKS)]
    for lane in lanes:
        address = addresses[lane]
        assert address is not None
        for word in words(address, width):
            bank_words[word % BANKS].append(word)
    return Phase(lanes, tuple(tuple(sorted(found)) for found in bank_words))


def _phase(addresses: tuple[int | None, ...], lanes: tuple[int, ...], width: int) -> Phase:
    bank_words: list[set[int]] = [set() for _ in range(BANKS)]
    for lane in lanes:
        address = addresses[lane]
        assert address is not None
        for word in words(address, width):
            bank_words[word % BANKS].add(word)
    return Phase(lanes, tuple(tuple(sorted(found)) for found in bank_words))


def _full_warp_merge(arch: str) -> bool:
    # The last digit of a name such as sm_89, sm_90a, or sm_120 is the minor version.
    digits = re.match(r"sm_(\d+)", arch)
    return digits is not None and int(digits[1]) // 10 >= _FULL_WARP_MERGE_MAJOR


def _merge(
    addresses: tuple[int | None, ...], width: int, full_warp: bool
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    if full_warp:
        return _merge_full_warp(addresses, width)
    active = {address for address in addresses if address is not None}
    if width == 16 and len(active) == 1 and _slots(addresses, _ALL_LANES) <= _BROADCAST_SLOTS:
        return (_ALL_LANES,), True
    groups: list[tuple[int, ...]] = []
    paired = False
    for span in _MERGE_SPANS[width]:
        if not _active(addresses, span):
            continue
        if _fits(addresses, span, width) and _span_merges(addresses, width, span):
            groups.append(span)
            paired = True
        else:
            groups.extend(group for group in _GROUPS[width] if group[0] in span)
    return tuple(groups), paired


def _span_merges(addresses: tuple[int | None, ...], width: int, span: tuple[int, ...]) -> bool:
    # The first half-warp of a 16-byte load merges freely while lane 31 is idle. The second
    # half-warp, and the whole warp of an 8-byte load, merge under one pairing of all lanes.
    if width == 16 and span == _HALVES[0]:
        if addresses[_LAST_LANE] is None:
            return True
        blocks = [block for block in _FOURS if block[0] in span]
        return all(_slots(addresses, block) <= _SLOT_LIMIT for block in blocks) and _strict(
            addresses, blocks
        )
    return any(_agree(addresses, span, distance) for distance in _PAIR_DISTANCES)


def _agree(addresses: tuple[int | None, ...], lanes: tuple[int, ...], distance: int) -> bool:
    # An idle lane agrees with any partner.
    return all(
        addresses[lane] is None
        or addresses[lane ^ distance] is None
        or addresses[lane] == addresses[lane ^ distance]
        for lane in lanes
    )


def _strict(addresses: tuple[int | None, ...], blocks: list[tuple[int, ...]]) -> bool:
    # Each block pairs all its active lanes but one at most, and the lanes left over in the
    # half-warp read one address.
    left: list[int | None] = []
    for block in blocks:
        rest = _left_over(addresses, block)
        if len(rest) > 1:
            return False
        left += rest
    return len(set(left)) <= 1


def _left_over(addresses: tuple[int | None, ...], block: tuple[int, ...]) -> list[int | None]:
    # The pairing that covers more lanes of the block leaves the fewest lanes over.
    found = []
    for distance in _PAIR_DISTANCES:
        found.append(
            [
                addresses[lane]
                for lane in block
                if addresses[lane] is not None and addresses[lane ^ distance] != addresses[lane]
            ]
        )
    return min(found, key=len)


def _merge_full_warp(
    addresses: tuple[int | None, ...], width: int
) -> tuple[tuple[tuple[int, ...], ...], bool]:
    if any(address is None for address in addresses) or not _pairs(addresses, _ALL_LANES):
        return _GROUPS[width], False
    groups: list[tuple[int, ...]] = []
    for span in _MERGE_SPANS[width]:
        if _fits(addresses, span, width):
            groups.append(span)
        else:
            groups.extend(group for group in _GROUPS[width] if group[0] in span)
    return tuple(groups), len(groups) < len(_GROUPS[width])


def _active(addresses: tuple[int | None, ...], lanes: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(lane for lane in lanes if addresses[lane] is not None)


def _fits(addresses: tuple[int | None, ...], lanes: tuple[int, ...], width: int) -> bool:
    return _phase(addresses, _active(addresses, lanes), width).wavefronts == 1


def _slots(addresses: tuple[int | None, ...], lanes: tuple[int, ...]) -> int:
    slots = 0
    for start in range(lanes[0], lanes[-1] + 1, 4):
        a, b, c, d = addresses[start : start + 4]
        # Within four lanes, the pairs at distance 1 and those at distance 2
        # exclude each other, and the larger set saves the more slots.
        pairs = max(_same(a, b) + _same(c, d), _same(a, c) + _same(b, d))
        slots += sum(address is not None for address in (a, b, c, d)) - pairs
    return slots


def _same(first: int | None, second: int | None) -> int:
    return int(first is not None and first == second)


def _pairs(addresses: tuple[int | None, ...], lanes: tuple[int, ...]) -> bool:
    active = [lane for lane in lanes if addresses[lane] is not None]
    return bool(active) and any(
        all(addresses[lane ^ distance] == addresses[lane] for lane in active)
        for distance in _PAIR_DISTANCES
    )
