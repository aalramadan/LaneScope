import json

import pytest

from lanescope import analysis, demos
from lanescope.fixes import layouts
from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace

TRANSPOSE = demos.transpose_trace()


def test_explain_lists_the_worst_instruction_first():
    data = analysis.explain(demos.transpose_trace())
    assert data["instructions"][0]["pc"] == "0x120"
    assert data["kernel"] == "transpose"


def test_lanes_shows_the_bank_grid_of_the_worst_access():
    assert "32 wavefronts against an ideal of 1" in analysis.lanes(demos.transpose_trace(), 0x120)


def test_whatif_reports_both_counts_and_the_shift():
    data = analysis.whatif(
        demos.transpose_trace(), layouts.Buffer(0, 4096), layouts.Padding(128, 4)
    )
    assert (data["extra_bytes"], data["later_shift"]) == (128, 128)
    worst = [item for item in data["instructions"] if item["before"]["pc"] == "0x120"][0]
    assert worst["after"]["excessive"] == 0


def test_suggest_infers_the_buffer_of_the_worst_instruction():
    data = analysis.suggest(demos.transpose_trace(), limit=2)
    assert data["buffer"] == {"start": 0, "size": 4096}
    assert [change["excessive_after"] for change in data["changes"]] == [0, 0]


def test_suggest_rejects_a_negative_limit():
    with pytest.raises(ValueError, match="limit must be zero or more"):
        analysis.suggest(TRANSPOSE, limit=-1)


def test_lanes_rejects_an_unknown_pc():
    with pytest.raises(ValueError, match="no access at pc 0x999"):
        analysis.lanes(demos.transpose_trace(), 0x999)


def test_overview_adds_the_source_and_the_totals_of_each_unit():
    data = analysis.overview(TRANSPOSE)
    assert data["source"][1].strip() == "__shared__ float tile[32][32];"
    assert data["rows"] == len(TRANSPOSE)
    assert data["totals"]["wavefronts"]["excessive"] == 3968
    assert data["totals"]["sectors"]["excessive"] == 0


def test_patterns_groups_repeated_rows_and_puts_the_worst_first():
    data = analysis.patterns(TRANSPOSE, demos.PC_LOAD_TILE, Space.SHARED)
    assert data["rows"] == 128
    assert sum(item["repeats"] for item in data["patterns"]) == 128
    worst = data["patterns"][0]
    assert (worst["achieved"], worst["ideal"], worst["excessive"]) == (32, 1, 31)


def test_patterns_stops_at_the_limit_but_counts_every_pattern():
    data = analysis.patterns(TRANSPOSE, demos.PC_LOAD_TILE, Space.SHARED, limit=3)
    assert len(data["patterns"]) == 3
    assert data["count"] == 32


def test_patterns_of_a_copy_keep_apart_reads_that_write_different_rows():
    lanes = range(WARP_SIZE)
    paired = tuple(16 * lane for lane in lanes)
    split = tuple(512 + 16 * (lane % 2 * 16 + lane // 2) for lane in lanes)
    accesses = [
        Access(0x40, Space.GLOBAL, Kind.LOAD, 16, tuple(1 << 32 | 16 * lane for lane in lanes)),
        Access(0x40, Space.SHARED, Kind.STORE, 16, paired),
        Access(
            0x40,
            Space.GLOBAL,
            Kind.LOAD,
            16,
            tuple(1 << 32 | 512 + 16 * lane for lane in lanes),
            warp=1,
        ),
        Access(0x40, Space.SHARED, Kind.STORE, 16, split, warp=1),
    ]
    trace = Trace.from_accesses(accesses, {"opcodes": {"0x40": "LDGSTS.E.BYPASS.128"}})
    data = analysis.patterns(trace, 0x40, Space.GLOBAL)
    assert sorted(item["achieved"] for item in data["patterns"]) == [16, 32]


def test_patterns_rejects_an_instruction_of_another_space():
    with pytest.raises(ValueError, match="no global access at pc 0x120"):
        analysis.patterns(TRANSPOSE, demos.PC_LOAD_TILE, Space.GLOBAL)


def test_access_gives_each_word_of_a_column_read_its_own_slot_in_bank_zero():
    row = analysis.patterns(TRANSPOSE, demos.PC_LOAD_TILE, Space.SHARED)["patterns"][0]["row"]
    data = analysis.access(TRANSPOSE, row)
    phase = data["phases"][0]
    assert (data["achieved"], data["conflicts"], phase["worst_bank"]) == (32, 31, 0)
    assert phase["degrees"][0] == 32
    assert sorted(cell[3] for cell in phase["cells"]) == list(range(32))
    assert {cell[1] for cell in phase["cells"]} == {0}
    assert "32 wavefronts against an ideal of 1" in data["text"]


def test_access_splits_a_16_byte_access_into_four_phases():
    access = Access(0x10, Space.SHARED, Kind.LOAD, 16, tuple(16 * lane for lane in range(32)))
    data = analysis.access(Trace.from_accesses([access]), 0)
    assert [phase["lanes"][0] for phase in data["phases"]] == [0, 8, 16, 24]
    assert all(len(phase["cells"]) == 32 for phase in data["phases"])


def test_access_of_global_memory_sends_addresses_as_text():
    data = analysis.access(TRANSPOSE, 0)
    assert data["unit"] == "sectors"
    assert data["base"] == f"{demos.INPUT_BASE:#x}"
    assert data["lanes"][1] == {"address": f"{demos.INPUT_BASE + 4:#x}", "offset": 4}
    assert [sector["lanes"][0] for sector in data["sectors"]] == [0, 8, 16, 24]
    json.dumps(data)


def test_access_rejects_a_row_outside_the_trace():
    with pytest.raises(ValueError, match="no row 512"):
        analysis.access(TRANSPOSE, len(TRANSPOSE))


def test_change_params_round_trip_through_json():
    for change in (layouts.Padding(128, 4), layouts.XorSwizzle(5, 2, 5)):
        params = json.loads(json.dumps(analysis.change_params(change)))
        assert analysis.change_from_params(params) == change


@pytest.mark.parametrize(
    "params",
    [
        None,
        {"kind": "rotate"},
        {"kind": "padding", "row_bytes": 128},
        {"kind": "swizzle", "bits": 5, "base": 2, "shift": "5"},
        {"kind": "padding", "row_bytes": True, "pad_bytes": 4},
        {"kind": []},
    ],
)
def test_change_from_params_rejects_malformed_parameters(params):
    with pytest.raises(ValueError):
        analysis.change_from_params(params)


def test_suggest_and_whatif_give_the_parameters_of_each_change():
    changes = analysis.suggest(TRANSPOSE, limit=1)["changes"]
    assert changes[0]["params"] == {"kind": "swizzle", "bits": 5, "base": 2, "shift": 5}
    data = analysis.whatif(TRANSPOSE, layouts.Buffer(0, 4096), layouts.Padding(128, 4))
    assert data["params"] == {"kind": "padding", "row_bytes": 128, "pad_bytes": 4}
