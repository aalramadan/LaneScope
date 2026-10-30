import json
import pathlib

import pytest

from lanescope.model import global_memory, shared_memory
from lanescope.nsight import calibration

# Achieved and ideal counts that Nsight Compute 2026.3.1 measured on an RTX 4060
# Laptop GPU (sm_89): the calibration cases, five runs of 300 random lane
# patterns, and patterns aimed at the rules that merge the groups of wide loads.
MEASURED = json.loads((pathlib.Path(__file__).parent / "data" / "rtx4060_counts.json").read_text())

# Wide shared loads whose counts the model does not yet reproduce. All have inactive lanes,
# except two full-warp loads of seed 5 that differ in their ideal alone.
KNOWN_MISSES = {
    "random-seed-1/random-22-shared-load-16",
    "random-seed-1/random-89-shared-load-16",
    "random-seed-1/random-105-shared-load-16",
    "random-seed-1/random-248-shared-load-16",
    "random-seed-1/random-273-shared-load-8",
    "random-seed-2/random-48-shared-load-16",
    "random-seed-2/random-86-shared-load-8",
    "random-seed-3/random-32-shared-load-16",
    "random-seed-3/random-36-shared-load-16",
    "random-seed-3/random-202-shared-load-16",
    "random-seed-3/random-271-shared-load-16",
    "random-seed-4/random-47-shared-load-16",
    "random-seed-5/random-16-shared-load-8",
    "random-seed-5/random-17-shared-load-16",
    "random-seed-5/random-36-shared-load-16",
    "random-seed-5/random-166-shared-load-16",
    "random-seed-5/random-173-shared-load-16",
    "random-seed-5/random-233-shared-load-16",
    "random-seed-5/random-274-shared-load-16",
}


def cases():
    for group, name, space, kind, width, units, achieved, ideal in MEASURED["cases"]:
        case_id = f"{group}/{name}"
        marks = [pytest.mark.xfail(strict=True)] if case_id in KNOWN_MISSES else []
        yield pytest.param(space, kind, width, units, (achieved, ideal), id=case_id, marks=marks)


def model_counts(space: str, kind: str, width: int, units: list[int | None]) -> tuple[int, int]:
    addresses = [None if unit is None else unit * width for unit in units]
    if space == "shared":
        store = kind == "store"
        result = shared_memory.analyze(addresses, width, store=store, arch=MEASURED["arch"])
        return result.wavefronts, result.ideal
    sectors = global_memory.analyze(addresses, width)
    return sectors.sectors, sectors.ideal


def test_measured_random_patterns_come_from_the_calibration_run():
    for seed in (1, 2, 3, 4, 5):
        measured = [case for case in MEASURED["cases"] if case[0] == f"random-seed-{seed}"]
        generated = calibration.random_cases(len(measured), seed)
        assert len(measured) == 300
        for (_, name, _, _, width, units, *_), case in zip(measured, generated, strict=True):
            lanes = case.addresses()
            assert name == case.name
            assert units == [None if offset is None else offset // width for offset in lanes]


def test_measurements_cover_every_calibration_case():
    measured = {name for group, name, *_ in MEASURED["cases"] if group == "calibration"}
    assert measured == {case.name for case in calibration.cases()}


@pytest.mark.parametrize(("space", "kind", "width", "units", "measured"), list(cases()))
def test_model_matches_nsight_compute_on_an_rtx_4060(space, kind, width, units, measured):
    assert model_counts(space, kind, width, units) == measured


# Nsight Compute 2026.3.1 counts of ldmatrix, cp.async, and atomics on the same GPU: the
# calibration cases, the random patterns of seeds 3 to 5, the random atomics of seed 5, the
# random patterns of an earlier probe, 16-byte copies that write two distinct rows, a probe of
# 16-byte copies with chosen rows, global addresses, and active lanes, a probe of copies whose
# lanes fill zeros with replays of the copies of a cuDNN convolution, a probe of shared and
# global atomics, and the 16-byte probe again through L1. A copy without global addresses
# reads consecutive rows of its width, lane by lane, and a None among its addresses marks a
# lane that fills zeros.
INSTRUCTIONS = json.loads(
    (pathlib.Path(__file__).parent / "data" / "rtx4060_instruction_counts.json").read_text()
)


def sources_of(width, sources):
    return sources if sources is not None else [width * lane for lane in range(32)]


def instruction_cases():
    for group, name, opcode, width, offsets, wavefronts, ideal, sources, *_ in INSTRUCTIONS[
        "cases"
    ]:
        if wavefronts is not None:
            yield pytest.param(
                opcode, width, offsets, sources, (wavefronts, ideal), id=f"{group}/{name}"
            )


def copy_sector_cases():
    for group, name, opcode, width, offsets, _, _, sources, *measured in INSTRUCTIONS["cases"]:
        if measured[0] is not None:
            yield pytest.param(
                opcode, width, offsets, sources, tuple(measured), id=f"{group}/{name}"
            )


def test_measurements_cover_every_ldmatrix_cp_async_and_atomic_case():
    measured = {name for group, name, *_ in INSTRUCTIONS["cases"] if group == "calibration"}
    cases = calibration.instruction_cases() + calibration.atomic_cases()
    assert measured == {case.name for case in cases}


def test_measured_random_ldmatrix_and_cp_async_patterns_come_from_the_calibration_run():
    for seed in (3, 4, 5):
        measured = [case for case in INSTRUCTIONS["cases"] if case[0] == f"random-seed-{seed}"]
        generated = calibration.random_instruction_cases(len(measured), seed)
        assert len(measured) == 300
        for (_, name, _, width, offsets, *_), case in zip(measured, generated, strict=True):
            assert (name, width, offsets) == (case.name, case.width, list(case.offsets))


def test_measured_random_atomic_patterns_come_from_the_calibration_run():
    measured = [case for case in INSTRUCTIONS["cases"] if case[0] == "random-atomics-seed-5"]
    generated = calibration.random_atomic_cases(len(measured), 5)
    assert len(measured) == 300
    for (_, name, _, width, offsets, *_), case in zip(measured, generated, strict=True):
        assert (name, width, offsets) == (case.name, case.width, list(case.addresses()))


@pytest.mark.parametrize(
    ("opcode", "width", "offsets", "sources", "measured"), list(instruction_cases())
)
def test_model_matches_nsight_compute_on_ldmatrix_and_cp_async(
    opcode, width, offsets, sources, measured
):
    store = opcode.startswith("LDGSTS")
    arch = INSTRUCTIONS["arch"]
    result = shared_memory.analyze(
        offsets, width, store=store, arch=arch, opcode=opcode, sources=sources
    )
    assert (result.wavefronts, result.ideal) == measured


@pytest.mark.parametrize(
    ("opcode", "width", "offsets", "sources", "measured"), list(copy_sector_cases())
)
def test_model_matches_nsight_compute_on_the_global_sectors_of_cp_async(
    opcode, width, offsets, sources, measured
):
    reads = [
        None if offset is None else address
        for offset, address in zip(offsets, sources_of(width, sources), strict=True)
    ]
    if opcode.startswith(("RED", "ATOMG")):
        result = global_memory.analyze(offsets, width)
    elif "BYPASS" in opcode:
        result = global_memory.copy_sectors(reads, offsets)
    else:
        result = global_memory.analyze(reads, width)
    assert (result.sectors, result.ideal) == measured
