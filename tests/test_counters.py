import random

from lanescope import demos
from lanescope.model.counters import count, instruction_key, measure, worst_first
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace


def by_pc(counters):
    return {counter.pc: counter for counter in counters}


def test_column_reads_of_the_transpose_sum_31_excess_wavefronts_per_request():
    counters = by_pc(count(demos.transpose_trace()))
    column_reads = counters[demos.PC_LOAD_TILE]
    assert column_reads.requests == 128
    assert (column_reads.achieved, column_reads.ideal) == (4096, 128)
    assert (column_reads.excessive, column_reads.conflicts) == (3968, 3968)
    assert column_reads.unit == "wavefronts"


def test_row_writes_of_the_transpose_have_no_excess():
    row_writes = by_pc(count(demos.transpose_trace()))[demos.PC_STORE_TILE]
    assert (row_writes.achieved, row_writes.excessive) == (128, 0)


def test_coalesced_global_accesses_count_four_sectors_per_request():
    loads = by_pc(count(demos.transpose_trace()))[demos.PC_LOAD_INPUT]
    assert (loads.unit, loads.achieved, loads.ideal, loads.excessive) == ("sectors", 512, 512, 0)


def test_worst_row_is_the_access_with_the_largest_excess():
    accesses = [
        Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(4 * lane for lane in range(WARP_SIZE))),
        Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(128 * lane for lane in range(WARP_SIZE))),
        Access(0x10, Space.SHARED, Kind.LOAD, 4, tuple(8 * lane for lane in range(WARP_SIZE))),
    ]
    (counter,) = count(Trace.from_accesses(accesses))
    assert counter.worst_row == 1
    assert counter.achieved == 1 + 32 + 2


def test_one_pc_in_two_spaces_gives_two_counters():
    addresses = tuple(4 * lane for lane in range(WARP_SIZE))
    accesses = [
        Access(0x40, Space.SHARED, Kind.STORE, 4, addresses),
        Access(0x40, Space.GLOBAL, Kind.STORE, 4, addresses),
    ]
    counters = count(Trace.from_accesses(accesses))
    assert [(counter.pc, counter.space) for counter in counters] == [
        (0x40, Space.SHARED),
        (0x40, Space.GLOBAL),
    ]


def test_empty_trace_has_no_counters():
    assert count(Trace.from_accesses([])) == []


def test_instruction_excess_adds_the_excess_of_each_access():
    paired = tuple(16 * (lane // 2) for lane in range(WARP_SIZE))
    conflicting = [16 * lane for lane in range(WARP_SIZE)]
    conflicting[1], conflicting[2] = 128, 256
    accesses = [
        Access(0x10, Space.SHARED, Kind.LOAD, 16, paired),
        Access(0x10, Space.SHARED, Kind.LOAD, 16, tuple(conflicting)),
    ]
    (counter,) = count(Trace.from_accesses(accesses))
    assert (counter.achieved, counter.ideal, counter.excessive) == (8, 6, 2)
    assert counter.worst_row == 1


def test_global_broadcast_does_not_hide_the_excess_of_a_column_access():
    accesses = [
        Access(0x20, Space.GLOBAL, Kind.LOAD, 4, (4096,) * WARP_SIZE),
        Access(0x20, Space.GLOBAL, Kind.LOAD, 4, tuple(4096 * lane for lane in range(WARP_SIZE))),
    ]
    (counter,) = count(Trace.from_accesses(accesses))
    assert counter.excessive == 28


def test_worst_first_orders_by_excess_then_pc():
    counters = worst_first(count(demos.transpose_trace()))
    assert [counter.pc for counter in counters] == [
        demos.PC_LOAD_TILE,
        demos.PC_LOAD_INPUT,
        demos.PC_STORE_TILE,
        demos.PC_STORE_OUTPUT,
    ]


def reference_count(trace):
    totals = {}
    for row in range(len(trace)):
        achieved, ideal, excessive, conflicts = measure(trace, row)
        total = totals.setdefault(instruction_key(trace, row), [0, 0, 0, 0, 0, row, excessive])
        total[0] += 1
        total[1] += achieved
        total[2] += ideal
        total[3] += excessive
        total[4] += conflicts
        if excessive > total[6]:
            total[5], total[6] = row, excessive
    return {key: tuple(total[:6]) for key, total in totals.items()}


def random_trace(seed):
    rng = random.Random(seed)
    accesses = []
    for _ in range(400):
        width = rng.choice([1, 2, 4, 8, 16])
        space = rng.choice([Space.SHARED, Space.GLOBAL])
        base = 0 if space == Space.SHARED else rng.choice([0x7F00_0000_0000, 0x7F00_0000_1080])
        stride = rng.choice([1, 2, 4, 32]) * width
        addresses = [
            base + (stride * lane if rng.random() < 0.8 else width * rng.randrange(64))
            if rng.random() < 0.9
            else None
            for lane in range(WARP_SIZE)
        ]
        if all(a is None for a in addresses):
            addresses[0] = base
        pc = rng.choice([0x10, 0x20, 0x30])
        accesses.append(
            Access(pc, space, rng.choice([Kind.LOAD, Kind.STORE]), width, tuple(addresses))
        )
    return Trace.from_accesses(accesses)


def test_counting_distinct_rows_once_gives_the_counts_of_a_row_by_row_sum():
    for seed in range(5):
        trace = random_trace(seed)
        found = {
            (c.pc, int(c.space), int(c.kind), c.width): (
                c.requests,
                c.achieved,
                c.ideal,
                c.excessive,
                c.conflicts,
                c.worst_row,
            )
            for c in count(trace)
        }
        assert found == reference_count(trace)


def test_count_keeps_two_copies_apart_when_only_their_global_addresses_differ():
    pc, rows = 0x40, tuple(16 * lane for lane in range(WARP_SIZE))
    together = tuple(16 * lane for lane in range(WARP_SIZE))
    apart = tuple(32 * lane for lane in range(WARP_SIZE))
    accesses = [
        Access(pc, Space.GLOBAL, Kind.LOAD, 16, together, warp=0),
        Access(pc, Space.SHARED, Kind.STORE, 16, rows, warp=0),
        Access(pc, Space.GLOBAL, Kind.LOAD, 16, apart, warp=1),
        Access(pc, Space.SHARED, Kind.STORE, 16, rows, warp=1),
    ]
    trace = Trace.from_accesses(accesses, {"opcodes": {"0x40": "LDGSTS.E.BYPASS.128"}})
    found = {counter.space: (counter.achieved, counter.ideal) for counter in count(trace)}
    assert found[Space.SHARED] == (4 + 32, 8)
    assert found[Space.GLOBAL] == (16 + 32, 32)
