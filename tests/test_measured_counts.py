import json
import pathlib

import pytest

from lanescope.model import global_memory, shared_memory
from lanescope.nsight import calibration

# Achieved and ideal counts that Nsight Compute 2026.3.1 measured on an RTX 4060
# Laptop GPU (sm_89): the calibration cases, three runs of 300 random lane
# patterns, and patterns aimed at the rules that merge the groups of wide loads.
MEASURED = json.loads((pathlib.Path(__file__).parent / "data" / "rtx4060_counts.json").read_text())

# Wide loads with inactive lanes whose wavefronts the model does not yet reproduce.
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
    for seed in (1, 2, 3):
        measured = [case for case in MEASURED["cases"] if case[0] == f"random-seed-{seed}"]
        generated = calibration.random_cases(len(measured), seed)
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


# Nsight Compute 2026.3.1 counts of ldmatrix and cp.async on the same GPU: the calibration
# cases, the random patterns of seed 3, the random patterns of an earlier probe, and
# 16-byte copies aimed at the rule that merges two distinct rows.
INSTRUCTIONS = json.loads(
    (pathlib.Path(__file__).parent / "data" / "rtx4060_instruction_counts.json").read_text()
)
# 16-byte cp.async copies whose lanes merge in ways the model does not yet reproduce.
INSTRUCTION_MISSES = {
    "targeted/cp_async_16-alternate-next-0",
    "targeted/cp_async_16-lane0-next-0",
    *(
        f"random-probe/cp_async_16-random{index}"
        for index in (0, 2, 4, 6, 10, 16, 17, 18, 19, 21, 22, 23, 28, 29)
    ),
    *(
        f"random-seed-3/random-{index}-cp-async-16"
        for index in (
            *(12, 20, 31, 34, 36, 37, 41, 44, 62, 91),
            *(140, 149, 159, 168, 169, 170, 226, 277, 294),
        )
    ),
}


def instruction_cases():
    for group, name, opcode, width, offsets, wavefronts, ideal in INSTRUCTIONS["cases"]:
        case_id = f"{group}/{name}"
        marks = [pytest.mark.xfail(strict=True)] if case_id in INSTRUCTION_MISSES else []
        yield pytest.param(opcode, width, offsets, (wavefronts, ideal), id=case_id, marks=marks)


def test_measurements_cover_every_ldmatrix_and_cp_async_case():
    measured = {name for group, name, *_ in INSTRUCTIONS["cases"] if group == "calibration"}
    assert measured == {case.name for case in calibration.instruction_cases()}


def test_measured_random_ldmatrix_and_cp_async_patterns_come_from_the_calibration_run():
    measured = [case for case in INSTRUCTIONS["cases"] if case[0] == "random-seed-3"]
    generated = calibration.random_instruction_cases(len(measured), 3)
    for (_, name, _, width, offsets, *_), case in zip(measured, generated, strict=True):
        assert (name, width, offsets) == (case.name, case.width, list(case.offsets))


@pytest.mark.parametrize(("opcode", "width", "offsets", "measured"), list(instruction_cases()))
def test_model_matches_nsight_compute_on_ldmatrix_and_cp_async(opcode, width, offsets, measured):
    store = opcode.startswith("LDGSTS")
    arch = INSTRUCTIONS["arch"]
    result = shared_memory.analyze(offsets, width, store=store, arch=arch, opcode=opcode)
    assert (result.wavefronts, result.ideal) == measured
