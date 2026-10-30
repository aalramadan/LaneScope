from lanescope import demos, text_view
from lanescope.model import global_memory, shared_memory
from lanescope.model.counters import count
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace


def grid_lines(addresses, width=4):
    text = text_view.bank_grid(shared_memory.analyze(addresses, width), tuple(addresses))
    return [line for line in text.splitlines() if line.startswith("lane ")]


def test_column_conflict_puts_32_distinct_words_in_bank_0():
    lines = grid_lines([128 * lane for lane in range(WARP_SIZE)])
    column = [line[8] for line in lines]
    assert len(set(column)) == 32
    assert all(line[9:40] == "." * 31 for line in lines)


def test_conflict_free_access_fills_the_diagonal():
    lines = grid_lines([4 * lane for lane in range(WARP_SIZE)])
    assert all(line[8 + lane] == "A" for lane, line in enumerate(lines))


def test_broadcast_shows_one_word_for_every_lane():
    lines = grid_lines([64] * WARP_SIZE)
    assert {line[8 + 16] for line in lines} == {"A"}


def test_word_counts_row_names_the_worst_bank():
    text = text_view.bank_grid(
        shared_memory.analyze([128 * lane for lane in range(WARP_SIZE)], 4),
        tuple(128 * lane for lane in range(WARP_SIZE)),
    )
    counts = [line for line in text.splitlines() if line.startswith("words")][0]
    assert counts[8] == "W"
    assert counts[9:] == "0" * 31


def test_wide_access_marks_four_banks_per_lane():
    lines = grid_lines([16 * lane for lane in range(WARP_SIZE)], width=16)
    assert lines[0][8:12] == "AAAA"
    assert lines[1][12:16] == "AAAA"


def test_paired_access_is_noted():
    text = text_view.bank_grid(
        shared_memory.analyze([8 * (lane // 2) for lane in range(WARP_SIZE)], 8),
        tuple(8 * (lane // 2) for lane in range(WARP_SIZE)),
    )
    assert "lane pairs read identical addresses" in text


def test_sector_table_lists_the_lanes_of_each_sector():
    result = global_memory.analyze([4096 + 8 * lane for lane in range(WARP_SIZE)], 4)
    text = text_view.sector_table(result)
    assert text.splitlines()[0] == "8 sectors against an ideal of 4, in 2 lines"
    assert text.splitlines()[2].endswith("0-3")


def test_describe_row_uses_the_view_of_the_space():
    trace = demos.transpose_trace()
    worst = {counter.pc: counter.worst_row for counter in count(trace)}
    assert "bank conflicts" in text_view.describe_row(trace, worst[demos.PC_LOAD_TILE])
    assert "sectors against" in text_view.describe_row(trace, worst[demos.PC_LOAD_INPUT])


def test_lane_ranges_join_consecutive_lanes():
    assert text_view.lane_ranges((0, 1, 2, 5, 7, 8)) == "0-2,5,7-8"
    assert text_view.lane_ranges(tuple(range(WARP_SIZE))) == "0-31"


def test_describe_row_names_the_sass_opcode_that_the_trace_keeps():
    rows = tuple(16 * lane for lane in range(WARP_SIZE))
    access = Access(0x40, Space.SHARED, Kind.LOAD, 16, rows)
    trace = Trace.from_accesses([access], {"opcodes": {"0x40": "LDSM.16.M88.4"}})
    header = text_view.describe_row(trace, 0).splitlines()[0]
    assert header == "pc 0x0040, shared load of 16 bytes (LDSM.16.M88.4), CTA 0, warp 0"
    plain = text_view.describe_row(Trace.from_accesses([access]), 0).splitlines()[0]
    assert plain == "pc 0x0040, shared load of 16 bytes, CTA 0, warp 0"
