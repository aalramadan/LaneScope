import dataclasses
import random

import numpy as np
import pytest

from lanescope import demos
from lanescope.model import batch_counts, global_memory
from lanescope.model.counters import count_row
from lanescope.trace import WARP_SIZE, WIDTHS, Access, Kind, Space, Trace

ARCHES = ("sm_89", "sm_90", "sm_120", "")
PLAIN_PC = 0x10
# Each instruction of the generated traces takes one SASS opcode, which selects its rule.
OPCODES = {
    0x20: "LDSM.16.M88",
    0x30: "LDSM.16.M88.2",
    0x40: "LDSM.16.MT88.4",
    0x50: "STSM.16.M88.4",
    0x60: "ATOMS.ADD",
    0x70: "ATOMS.CAS",
    0x80: "ATOMS.CAS.64",
    0x90: "ATOMS.POPC.INC.32",
    0xA0: "LDS.128",
}
COPY_OPCODES = {
    0xB0: ("LDGSTS.E.BYPASS.128", 16),
    0xC0: ("LDGSTS.E.128", 16),
    0xD0: ("LDGSTS.E.64", 8),
    0xE0: ("LDGSTS.E", 4),
    0xF0: ("LDGSTS.E.BYPASS.LTC128B.128", 16),
}
GLOBAL_BASE = 0x7F00_0000_0000


def lane_slots(rng):
    """Return the slot of each lane for one sort of pattern."""
    sort = rng.randrange(8)
    if sort == 0:
        window = rng.choice([8, 64, 512, 4096])
        return [rng.randrange(window) for _ in range(WARP_SIZE)]
    if sort == 1:
        stride = rng.choice([1, 2, 3, 4, 8, 16, 32, 33])
        return [stride * lane for lane in range(WARP_SIZE)]
    if sort == 2:
        stride = rng.choice([1, 2, 8, 32])
        return [stride * (lane // 2) for lane in range(WARP_SIZE)]
    if sort == 3:
        stride = rng.choice([1, 2, 8, 32])
        return [stride * (lane // 4 * 2 + lane % 2) for lane in range(WARP_SIZE)]
    if sort == 4:
        return [rng.randrange(64)] * WARP_SIZE
    if sort == 5:
        # Pairs at either distance in each block of four, with some lanes off the pattern.
        distance = rng.choice([1, 2])
        return [
            (lane & ~distance) if rng.random() < 0.85 else rng.randrange(256)
            for lane in range(WARP_SIZE)
        ]
    if sort == 6:
        # Matrix rows: eight lanes of each group read rows a fixed pitch apart.
        pitch = rng.choice([1, 2, 4, 8, 9])
        return [pitch * (lane % 8) + 64 * (lane // 8) for lane in range(WARP_SIZE)]
    return [lane ^ rng.randrange(32) for lane in range(WARP_SIZE)]


def lane_mask(rng):
    """Return which lanes are active, mostly the whole warp."""
    sort = rng.randrange(6)
    if sort < 3:
        lanes = [True] * WARP_SIZE
    elif sort == 3:
        lanes = [rng.random() < 0.6 for _ in range(WARP_SIZE)]
    elif sort == 4:
        start = rng.randrange(WARP_SIZE)
        lanes = [start <= lane < start + rng.randrange(1, WARP_SIZE + 1) for lane in range(32)]
    else:
        lanes = [lane != WARP_SIZE - 1 for lane in range(WARP_SIZE)]
    if not any(lanes):
        lanes[rng.randrange(WARP_SIZE)] = True
    return lanes


def addresses(rng, width, base):
    """Return the aligned lane addresses of one pattern with idle lanes as None."""
    slots = lane_slots(rng)
    return tuple(
        base + width * slot if on else None for slot, on in zip(slots, lane_mask(rng), strict=True)
    )


def shifted(lanes, offset):
    return tuple(None if address is None else address + offset for address in lanes)


def shared_base(rng, width):
    """Return a base that keeps the banks, moves by a whole wavefront, or moves by part of one."""
    return rng.choice([0, 128 * rng.randrange(1, 300), width * rng.randrange(1, 64)])


def generated_accesses(rng, rows):
    """Return accesses of every kind, width, and opcode, with some repeated at a shift."""
    accesses = []
    warp = 0
    while len(accesses) < rows:
        warp += 1
        sort = rng.randrange(10)
        if sort < 3:
            width = rng.choice(WIDTHS)
            space = rng.choice([Space.SHARED, Space.SHARED, Space.GLOBAL])
            base = shared_base(rng, width) if space == Space.SHARED else GLOBAL_BASE
            kind = rng.choice(list(Kind))
            lanes = addresses(rng, width, base)
            group = [Access(PLAIN_PC, space, kind, width, lanes, warp=warp)]
        elif sort < 7:
            pc = rng.choice(list(OPCODES))
            width = 16 if pc < 0x60 and rng.random() < 0.8 else rng.choice(WIDTHS)
            # Opcodes decide the rule over kinds, so every kind takes a turn.
            drawn = rng.choice(list(Kind))
            kind = Kind.STORE if pc == 0x50 else drawn
            if 0x60 <= pc < 0xA0 and rng.random() < 0.8:
                kind = Kind.ATOMIC
            lanes = addresses(rng, width, shared_base(rng, width))
            group = [Access(pc, Space.SHARED, kind, width, lanes, warp=warp)]
        else:
            group = copy_accesses(rng, warp)
        accesses += group
        if rng.random() < 0.3:
            # The same pattern again, moved as a whole, as in another CTA or another stage.
            offsets = {space: rng.choice([128, 4096, 64, 16]) for space in Space}
            accesses += [
                dataclasses.replace(
                    access, addresses=shifted(access.addresses, offsets[access.space]), cta=9
                )
                for access in group
            ]
    return accesses


def copy_accesses(rng, warp):
    """Return the global and shared rows of one cp.async, or a shared row without its reads."""
    pc = rng.choice(list(COPY_OPCODES))
    width = COPY_OPCODES[pc][1]
    rows = addresses(rng, width, shared_base(rng, width))
    if rng.random() < 0.1:
        return [Access(pc, Space.SHARED, Kind.STORE, width, rows, cta=2, warp=warp)]
    sort = rng.randrange(4)
    if sort == 0:
        slots = list(range(WARP_SIZE))
    elif sort == 1:
        slots = [lane ^ 1 for lane in range(WARP_SIZE)]
    elif sort == 2:
        slots = [rng.randrange(256) for _ in range(WARP_SIZE)]
    else:
        slots = lane_slots(rng)
    line = GLOBAL_BASE + 128 * rng.randrange(1 << 16)
    sources = [line + width * slot for slot in slots]
    # Lanes that fill zeros read nothing: their global lanes are idle.
    zero_share = rng.choice([0, 0, 0.2, 0.6, 1.0])
    reads = tuple(
        None if row is None or rng.random() < zero_share else source
        for row, source in zip(rows, sources, strict=True)
    )
    if all(read is None for read in reads):
        if rng.random() < 0.5:
            return [Access(pc, Space.SHARED, Kind.STORE, width, rows, cta=2, warp=warp)]
        first = next(lane for lane, row in enumerate(rows) if row is not None)
        reads = tuple(sources[first] if lane == first else None for lane in range(WARP_SIZE))
    return [
        Access(pc, Space.GLOBAL, Kind.LOAD, width, reads, cta=3, warp=warp),
        Access(pc, Space.SHARED, Kind.STORE, width, rows, cta=3, warp=warp),
    ]


def generated_trace(seed, arch, rows=2500):
    rng = random.Random(seed)
    opcodes = {f"{pc:#x}": opcode for pc, opcode in OPCODES.items()}
    opcodes |= {f"{pc:#x}": opcode for pc, (opcode, _) in COPY_OPCODES.items()}
    metadata = {"opcodes": opcodes} | ({"arch": arch} if arch else {})
    return Trace.from_accesses(generated_accesses(rng, rows), metadata)


@pytest.fixture(scope="module", params=list(enumerate(ARCHES)), ids=lambda item: item[1] or "none")
def trace(request):
    seed, arch = request.param
    return generated_trace(seed, arch)


def test_generated_traces_hold_every_rule_of_the_model(trace):
    opcodes = {trace.opcode(int(pc)) for pc in trace.pc.tolist()}
    assert set(OPCODES.values()) | {code for code, _ in COPY_OPCODES.values()} <= opcodes | {""}
    shared = trace.space == Space.SHARED
    plain = shared & (trace.pc == PLAIN_PC)
    for kind in Kind:
        assert set(trace.width[plain & (trace.kind == kind)].tolist()) == set(WIDTHS)
    zero_fills = [
        row
        for row, sources in trace.copy_sources.items()
        if any(
            s is None and a is not None
            for s, a in zip(sources, trace.lane_addresses(row), strict=True)
        )
    ]
    assert zero_fills
    assert (trace.mask != 0xFFFFFFFF).any()


def test_row_counts_equal_the_scalar_model_on_every_row(trace):
    found = batch_counts.row_counts(trace)
    expected = np.array([count_row(trace, row) for row in range(len(trace))]).reshape(-1, 4)
    assert len(trace) > 2000
    for column, name in enumerate(("achieved", "ideal", "excessive", "conflicts")):
        np.testing.assert_array_equal(getattr(found, name), expected[:, column], err_msg=name)


def test_copy_read_excess_equals_the_scalar_global_reads(trace):
    found = batch_counts.copy_read_excess(trace)
    expected = [
        reads.excessive if (reads := global_memory.copy_reads(trace, row)) is not None else 0
        for row in range(len(trace))
    ]
    np.testing.assert_array_equal(found, expected)
    assert found.any()


def test_row_counts_of_picked_rows_equal_those_of_every_row(trace):
    rows = np.random.default_rng(7).permutation(len(trace))[:500]
    every = batch_counts.row_counts(trace)
    picked = batch_counts.row_counts(trace, rows)
    for name in ("achieved", "ideal", "excessive", "conflicts"):
        np.testing.assert_array_equal(getattr(picked, name), getattr(every, name)[rows])


def test_row_counts_equal_the_scalar_model_on_the_gallery():
    for name, trace in demos.gallery_traces().items():
        found = batch_counts.row_counts(trace)
        expected = np.array([count_row(trace, row) for row in range(len(trace))]).reshape(-1, 4)
        assert (
            np.column_stack(
                [found.achieved, found.ideal, found.excessive, found.conflicts]
            ).tolist()
            == expected.tolist()
        ), name


def test_a_pattern_moved_by_part_of_a_wavefront_keeps_its_own_count():
    lanes = tuple(16 * (lane // 2) for lane in range(WARP_SIZE))
    accesses = [
        Access(PLAIN_PC, Space.SHARED, Kind.LOAD, 16, lanes),
        Access(PLAIN_PC, Space.SHARED, Kind.LOAD, 16, shifted(lanes, 64)),
        Access(PLAIN_PC, Space.SHARED, Kind.LOAD, 16, shifted(lanes, 128)),
    ]
    trace = Trace.from_accesses(accesses, {"arch": "sm_89"})
    found = batch_counts.row_counts(trace)
    assert found.achieved.tolist() == [count_row(trace, row)[0] for row in range(3)]


def test_row_counts_of_no_rows_are_empty():
    found = batch_counts.row_counts(demos.transpose_trace(), np.array([], dtype=np.intp))
    assert found.achieved.shape == (0,)
