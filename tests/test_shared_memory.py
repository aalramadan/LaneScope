import pytest

from lanescope.model import shared_memory
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

# LDS.128 lane patterns with the wavefronts that Nsight Compute reported on H100 and
# RTX 5090, from NVIDIA forum topic 374947. Each entry is an int4 index per lane.
FORUM_LDS128_CASES = [
    ([0] * 8 + [1] * 8 + [2] * 8 + [3] * 8, 2),
    (
        [
            0,
            1,
            0,
            1,
            2,
            3,
            2,
            3,
            4,
            5,
            4,
            5,
            6,
            7,
            6,
            7,
            8,
            9,
            8,
            9,
            10,
            11,
            10,
            11,
            12,
            13,
            12,
            13,
            14,
            15,
            14,
            15,
        ],
        2,
    ),
    (
        [
            0,
            1,
            0,
            1,
            2,
            3,
            2,
            2,
            4,
            5,
            4,
            5,
            6,
            7,
            6,
            7,
            0,
            1,
            0,
            1,
            2,
            3,
            2,
            3,
            4,
            5,
            4,
            5,
            6,
            7,
            6,
            7,
        ],
        4,
    ),
    ([0, 1, 2, 3] * 8, 4),
    (list(range(8)) * 4, 4),
    ([0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15] * 2, 4),
    ([0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15] * 2, 8),
    (
        [
            0,
            2,
            0,
            2,
            4,
            6,
            4,
            6,
            8,
            10,
            8,
            10,
            12,
            14,
            12,
            14,
            16,
            18,
            16,
            18,
            20,
            22,
            20,
            22,
            24,
            26,
            24,
            26,
            28,
            30,
            28,
            30,
        ],
        4,
    ),
    ([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] * 2, 8),
]


def lanes(offset_of):
    return [offset_of(lane) for lane in range(WARP_SIZE)]


def test_consecutive_words_take_one_wavefront():
    result = shared_memory.analyze(lanes(lambda lane: 4 * lane), 4)
    assert (result.wavefronts, result.ideal, result.conflicts, result.excessive) == (1, 1, 0, 0)


def test_stride_of_128_bytes_takes_32_wavefronts():
    result = shared_memory.analyze(lanes(lambda lane: 128 * lane), 4)
    assert (result.wavefronts, result.conflicts, result.excessive) == (32, 31, 31)
    assert result.phases[0].worst_bank == 0


def test_stride_of_8_bytes_is_a_two_way_conflict():
    result = shared_memory.analyze(lanes(lambda lane: 8 * lane), 4)
    assert (result.wavefronts, result.conflicts) == (2, 1)


def test_broadcast_of_one_word_takes_one_wavefront():
    result = shared_memory.analyze([256] * WARP_SIZE, 4)
    assert (result.wavefronts, result.conflicts) == (1, 0)


def test_multicast_with_distinct_banks_takes_one_wavefront():
    result = shared_memory.analyze(lanes(lambda lane: 4 * (lane // 4)), 4)
    assert result.wavefronts == 1


def test_lanes_in_one_word_share_it_for_two_byte_accesses():
    result = shared_memory.analyze(lanes(lambda lane: 2 * lane), 2)
    assert (result.wavefronts, result.ideal) == (1, 1)


def test_consecutive_64_bit_accesses_take_two_wavefronts():
    result = shared_memory.analyze(lanes(lambda lane: 8 * lane), 8)
    assert [phase.lanes for phase in result.phases] == [tuple(range(16)), tuple(range(16, 32))]
    assert (result.wavefronts, result.ideal, result.excessive) == (2, 2, 0)


def test_consecutive_128_bit_accesses_take_four_wavefronts():
    result = shared_memory.analyze(lanes(lambda lane: 16 * lane), 16)
    assert len(result.phases) == 4
    assert (result.wavefronts, result.ideal, result.excessive) == (4, 4, 0)


def test_64_bit_conflict_matches_the_nvidia_staff_example():
    result = shared_memory.analyze(lanes(lambda lane: 128 * lane), 8)
    assert (result.wavefronts, result.conflicts, result.ideal, result.excessive) == (32, 30, 2, 30)


def test_128_bit_conflict_matches_the_nvidia_staff_example():
    result = shared_memory.analyze(lanes(lambda lane: 128 * lane), 16)
    assert (result.wavefronts, result.conflicts, result.ideal) == (32, 28, 4)


@pytest.mark.parametrize(("indices", "measured"), FORUM_LDS128_CASES)
def test_128_bit_patterns_match_the_forum_measurements_on_h100(indices, measured):
    result = shared_memory.analyze([16 * index for index in indices], 16, arch="sm_90")
    assert result.wavefronts == measured


def test_rtx_4060_merges_the_paired_half_warp_of_a_128_bit_load_on_its_own():
    indices = [0, 1, 0, 1, 2, 3, 2, 2, 4, 5, 4, 5, 6, 7, 6, 7]
    indices += [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7]
    result = shared_memory.analyze([16 * index for index in indices], 16, arch="sm_89")
    assert [len(phase.lanes) for phase in result.phases] == [8, 8, 16]
    assert (result.wavefronts, result.ideal) == (3, 3)


def test_merge_that_would_cost_a_bank_conflict_keeps_the_quarters():
    indices = [0, 2, 0, 2, 4, 6, 4, 6, 8, 10, 8, 10, 12, 14, 12, 14]
    result = shared_memory.analyze([16 * index for index in indices * 2], 16)
    assert not result.paired
    assert (len(result.phases), result.wavefronts, result.ideal) == (4, 4, 4)


def test_paired_store_keeps_its_quarters():
    addresses = [16 * (lane // 8) for lane in range(WARP_SIZE)]
    load = shared_memory.analyze(addresses, 16)
    store = shared_memory.analyze(addresses, 16, store=True)
    assert (load.wavefronts, store.wavefronts) == (2, 4)


def test_pairs_of_identical_64_bit_addresses_run_as_one_group():
    result = shared_memory.analyze(lanes(lambda lane: 8 * (lane // 2)), 8)
    assert result.paired
    assert len(result.phases) == 1
    assert (result.wavefronts, result.ideal, result.excessive) == (1, 1, 0)


def test_inactive_lanes_are_left_out():
    addresses = lanes(lambda lane: 128 * lane)
    for lane in range(1, WARP_SIZE, 2):
        addresses[lane] = None
    result = shared_memory.analyze(addresses, 4)
    assert (result.active, result.wavefronts, result.ideal) == (16, 16, 1)


def test_group_without_active_lanes_is_skipped():
    addresses = [16 * lane if lane < 8 else None for lane in range(WARP_SIZE)]
    result = shared_memory.analyze(addresses, 16)
    assert [phase.lanes for phase in result.phases] == [tuple(range(8))]
    assert (result.wavefronts, result.ideal) == (1, 1)


def test_bank_words_list_the_distinct_words_of_each_bank():
    result = shared_memory.analyze(lanes(lambda lane: 128 * (lane % 2)), 4)
    assert result.phases[0].bank_words[0] == (0, 32)
    assert all(words == () for words in result.phases[0].bank_words[1:])


def test_wrong_lane_count_is_rejected():
    with pytest.raises(ValueError, match="32 lane addresses"):
        shared_memory.analyze([0] * 16, 4)


def test_access_without_active_lanes_is_rejected():
    with pytest.raises(ValueError, match="active lane"):
        shared_memory.analyze([None] * WARP_SIZE, 4)


def test_pairing_needs_a_full_warp_on_h100():
    addresses = [16 * (lane // 8) if lane % 8 < 2 else None for lane in range(WARP_SIZE)]
    result = shared_memory.analyze(addresses, 16, arch="sm_90")
    assert not result.paired
    assert (len(result.phases), result.wavefronts) == (4, 4)


def test_pairing_on_the_rtx_4060_allows_inactive_lanes():
    addresses = [16 * (lane // 8) if lane % 8 < 2 else None for lane in range(WARP_SIZE)]
    result = shared_memory.analyze(addresses, 16, arch="sm_89")
    assert result.paired
    assert (len(result.phases), result.wavefronts, result.ideal) == (2, 2, 1)


def test_one_lane_per_quarter_merges_each_half_warp():
    addresses = [16 * (lane // 8) if lane % 8 == 0 else None for lane in range(WARP_SIZE)]
    result = shared_memory.analyze(addresses, 16)
    assert (result.wavefronts, result.ideal, result.excessive, result.conflicts) == (2, 1, 1, 0)


def test_lone_lanes_in_one_bank_keep_their_quarters():
    addresses = [128 * (lane // 8) if lane % 8 == 0 else None for lane in range(WARP_SIZE)]
    result = shared_memory.analyze(addresses, 16)
    assert (result.wavefronts, result.ideal, result.excessive, result.conflicts) == (4, 1, 3, 0)


def test_bytes_with_a_stride_of_128_take_32_wavefronts_in_one_phase():
    result = shared_memory.analyze(lanes(lambda lane: 128 * lane), 1)
    assert (len(result.phases), result.wavefronts, result.ideal) == (1, 32, 1)


def _rows(stride: int) -> list[int]:
    return [stride * lane for lane in range(WARP_SIZE)]


@pytest.mark.parametrize(
    ("opcode", "stride", "wavefronts", "ideal"),
    [
        ("LDSM.16.M88.4", 16, 4, 4),
        ("LDSM.16.M88.4", 128, 32, 4),
        ("LDSM.16.M88.2", 64, 8, 2),
        ("LDSM.16.M88", 128, 8, 1),
    ],
)
def test_ldmatrix_counts_the_distinct_rows_on_one_bank_in_each_matrix(
    opcode, stride, wavefronts, ideal
):
    result = shared_memory.analyze(_rows(stride), 16, opcode=opcode)
    assert (result.wavefronts, result.ideal) == (wavefronts, ideal)


def test_ldmatrix_of_one_matrix_reads_only_the_rows_of_the_first_eight_lanes():
    rows = [16 * lane if lane < 8 else 128 * lane for lane in range(WARP_SIZE)]
    assert shared_memory.analyze(rows, 16, opcode="LDSM.16.M88").wavefronts == 1


def test_cp_async_writes_to_one_address_take_one_wavefront_each():
    result = shared_memory.analyze([0] * WARP_SIZE, 4, store=True, opcode="LDGSTS.E")
    assert (result.wavefronts, result.ideal) == (32, 1)


@pytest.mark.parametrize(
    ("rows", "wavefronts"),
    [
        (list(range(WARP_SIZE)), 4),
        ([lane // 8 * 8 + (lane % 8 ^ lane // 8) for lane in range(WARP_SIZE)], 4),
        ([lane + 1 for lane in range(WARP_SIZE)], 12),
        ([2 * lane for lane in range(WARP_SIZE)], 32),
        ([0] * WARP_SIZE, 1),
    ],
)
def test_16_byte_cp_async_gives_the_wavefronts_measured_on_the_rtx_4060(rows, wavefronts):
    destinations = [16 * row for row in rows]
    result = shared_memory.analyze(destinations, 16, store=True, opcode="LDGSTS.E.BYPASS.128")
    assert (result.wavefronts, result.ideal) == (wavefronts, -(-len(set(rows)) // 8))


def test_16_byte_cp_async_counts_from_the_global_addresses_of_its_lanes():
    destinations = _rows(16)
    apart = [32 * lane for lane in range(WARP_SIZE)]
    result = shared_memory.analyze(
        destinations, 16, store=True, opcode="LDGSTS.E.BYPASS.128", sources=apart
    )
    assert result.wavefronts == 32


def test_analyze_row_reads_the_global_addresses_of_the_same_cp_async():
    pc, opcode = 0x40, "LDGSTS.E.BYPASS.128"
    apart = Access(pc, Space.GLOBAL, Kind.LOAD, 16, tuple(32 * lane for lane in range(WARP_SIZE)))
    write = Access(pc, Space.SHARED, Kind.STORE, 16, tuple(_rows(16)))
    trace = Trace.from_accesses([apart, write], metadata={"opcodes": {f"{pc:#x}": opcode}})
    assert shared_memory.analyze_row(trace, 1).wavefronts == 32


def test_analyze_row_takes_the_rule_from_the_opcode_in_the_trace():
    access = Access(0x10, Space.SHARED, Kind.LOAD, 16, tuple(_rows(128)))
    trace = Trace.from_accesses([access], metadata={"opcodes": {"0x10": "LDSM.16.M88"}})
    assert shared_memory.analyze_row(trace, 0).wavefronts == 8
