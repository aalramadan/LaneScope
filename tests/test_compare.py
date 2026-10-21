import json
import pathlib
import types

from lanescope import demos
from lanescope.nsight import compare

KERNEL_START = 0x7FD4_A325_D200
TIMESTAMP = 0x18E0_86DA_7F2B_1060


class StandInMetric:
    """An instanced metric of an Nsight Compute report, with its value per correlation ID."""

    def __init__(self, instances):
        self.ids, self.values = list(instances), list(instances.values())

    def has_correlation_ids(self):
        return True

    def correlation_ids(self):
        return types.SimpleNamespace(as_uint64=lambda index: self.ids[index])

    def num_instances(self):
        return len(self.ids)

    def as_double(self, index):
        return float(self.values[index])

    def as_uint64(self, index):
        return int(self.values[index])


class StandInAction:
    """One kernel launch of a report, with its metrics and the SASS of its module."""

    def __init__(self, metrics, sass):
        self.metrics, self.sass = metrics, sass

    def name(self):
        return "transposeCoalesced"

    def metric_names(self):
        return list(self.metrics)

    def metric_by_name(self, name):
        return self.metrics.get(name)

    def sass_by_pc(self, pc):
        return self.sass.get(pc, "")


def read_stand_in(monkeypatch, action):
    report = types.SimpleNamespace(
        num_ranges=lambda: 1,
        range_by_idx=lambda index: types.SimpleNamespace(
            num_actions=lambda: 1, action_by_idx=lambda index: action
        ),
    )
    monkeypatch.setattr(compare, "ncu_report", types.SimpleNamespace(load_report=lambda p: report))
    (kernel,) = compare.read_report(pathlib.Path("transposeCoalesced.ncu-rep"))
    return kernel


def sample_metrics():
    """Return the metrics of a launch whose module holds another kernel at address 0."""
    return {
        "inst_executed": StandInMetric(
            {KERNEL_START + offset: 32 for offset in (0x0, 0x10, 0x1B0, 0x1C0)}
        ),
        "memory_l1_wavefronts_shared": StandInMetric(
            {KERNEL_START + 0x1B0: 32, KERNEL_START + 0x1C0: 1}
        ),
        # Occupancy tables and NUMA facts number their instances from 0.
        "launch__occupancy_per_register_count": StandInMetric({i: 2 for i in range(257)}),
        "numa__id_cpu": StandInMetric({0: 0}),
        "profiler__timestamp_workload_start_0": StandInMetric({TIMESTAMP: 1}),
    }


SAMPLE_SASS = {
    0x0: "MOV R1, c[0x0][0x28]",
    0x10: "S2R R6, SR_CTAID.X",
    KERNEL_START: "MOV R1, c[0x0][0x28]",
    KERNEL_START + 0x10: "S2R R10, SR_CTAID.X",
    KERNEL_START + 0x1B0: "STS [R5], R2",
    KERNEL_START + 0x1C0: "LDS R7, [R6]",
    TIMESTAMP: "N/A",
}


def test_report_offsets_count_from_the_kernel_start_that_the_launch_names(monkeypatch):
    metrics = sample_metrics() | {"launch__function_pcs": StandInMetric({0: KERNEL_START})}
    kernel = read_stand_in(monkeypatch, StandInAction(metrics, SAMPLE_SASS))
    assert kernel.quantity("shared_wavefronts") == (
        "memory_l1_wavefronts_shared",
        {0x1B0: 32.0, 0x1C0: 1.0},
    )
    assert set(kernel.metrics) == {"inst_executed", "memory_l1_wavefronts_shared"}
    assert kernel.sass == {
        0x0: "MOV R1, c[0x0][0x28]",
        0x10: "S2R R10, SR_CTAID.X",
        0x1B0: "STS [R5], R2",
        0x1C0: "LDS R7, [R6]",
    }


def test_report_without_a_kernel_start_counts_from_its_first_instruction(monkeypatch):
    metrics = {
        name: metric
        for name, metric in sample_metrics().items()
        if name in ("inst_executed", "memory_l1_wavefronts_shared")
    }
    kernel = read_stand_in(monkeypatch, StandInAction(metrics, SAMPLE_SASS))
    assert kernel.quantity("shared_wavefronts")[1] == {0x1B0: 32.0, 0x1C0: 1.0}


TILE_SASS = {
    demos.PC_LOAD_INPUT: "LDG.E R2, [R2.64]",
    demos.PC_STORE_TILE: "STS [R5], R2",
    demos.PC_LOAD_TILE: "@P0 LDS R7, [R6]",
    demos.PC_STORE_OUTPUT: "STG.E [R8.64], R7",
}


def counters_for_transpose(load_tile_wavefronts=4096):
    return compare.KernelCounters(
        kernel="transpose",
        metrics={
            "memory_l1_wavefronts_shared": {
                demos.PC_STORE_TILE: 128.0,
                demos.PC_LOAD_TILE: float(load_tile_wavefronts),
            },
            "memory_l1_wavefronts_shared_ideal": {
                demos.PC_STORE_TILE: 128.0,
                demos.PC_LOAD_TILE: 128.0,
            },
            "smsp__sass_sectors_mem_global_op_ld_or_st": {
                demos.PC_LOAD_INPUT: 512.0,
                demos.PC_STORE_OUTPUT: 512.0,
            },
        },
        sass=TILE_SASS,
    )


def test_matching_counts_give_matching_rows():
    rows = compare.compare(0, demos.transpose_trace(), counters_for_transpose())
    shared = [row for row in rows if row.quantity == "shared_wavefronts"]
    assert {row.offset: row.matches for row in shared} == {
        demos.PC_STORE_TILE: True,
        demos.PC_LOAD_TILE: True,
    }


def test_a_different_count_is_a_mismatch():
    rows = compare.compare(0, demos.transpose_trace(), counters_for_transpose(2048))
    (row,) = [
        r for r in rows if r.offset == demos.PC_LOAD_TILE and r.quantity == "shared_wavefronts"
    ]
    assert (row.lanescope, row.nsight, row.matches) == (4096, 2048.0, False)


def test_unknown_metric_names_resolve_by_their_words():
    name, values = counters_for_transpose().quantity("global_sectors")
    assert name == "smsp__sass_sectors_mem_global_op_ld_or_st"
    assert values[demos.PC_LOAD_INPUT] == 512.0


def test_missing_quantity_gives_no_metric():
    assert counters_for_transpose().quantity("global_sectors_ideal") == (None, {})


def test_opcode_drops_the_predicate():
    assert compare.opcode("@P0 LDS.U.128 R4, [R2]") == "LDS.U.128"
    assert compare.opcode("") == ""


def test_summary_counts_matches_per_opcode_and_quantity():
    rows = compare.compare(0, demos.transpose_trace(), counters_for_transpose(2048))
    summary = compare.summarize(rows)
    assert summary["LDS shared_wavefronts"] == {"matched": 0, "total": 1}
    assert summary["STS shared_wavefronts"] == {"matched": 1, "total": 1}


def test_dump_writes_every_metric_with_hex_offsets(tmp_path):
    path = tmp_path / "metrics.json"
    compare.dump([counters_for_transpose()], path)
    (kernel,) = json.loads(path.read_text())
    assert kernel["sass"]["0x120"] == "@P0 LDS R7, [R6]"
    assert kernel["metrics"]["memory_l1_wavefronts_shared"]["0x120"] == 4096.0


def test_offsets_on_another_base_pair_by_rank():
    shifted = counters_for_transpose()
    moved = compare.KernelCounters(
        kernel=shifted.kernel,
        metrics={
            name: {offset + 0x1000: value for offset, value in values.items()}
            for name, values in shifted.metrics.items()
        },
        sass={offset + 0x1000: text for offset, text in shifted.sass.items()},
    )
    rows = compare.compare(0, demos.transpose_trace(), moved)
    assert all(row.matches for row in rows if row.quantity != "global_sectors_ideal")
    assert {row.sass for row in rows if row.offset == demos.PC_LOAD_TILE} == {"@P0 LDS R7, [R6]"}


def test_counts_split_by_direction_are_added_up():
    counters = compare.KernelCounters(
        kernel="k",
        metrics={
            "smsp__sass_wavefronts_mem_shared_op_ld": {0x10: 32.0},
            "smsp__sass_wavefronts_mem_shared_op_st": {0x20: 1.0},
        },
        sass={0x10: "LDS R1, [R2]", 0x20: "STS [R3], R4"},
    )
    name, values = counters.quantity("shared_wavefronts")
    assert name == "smsp__sass_wavefronts_mem_shared_op_ld + smsp__sass_wavefronts_mem_shared_op_st"
    assert values == {0x10: 32.0, 0x20: 1.0}
