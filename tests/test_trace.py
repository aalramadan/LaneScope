import json
import re
import zipfile

import numpy as np
import pytest

from lanescope.trace import WARP_SIZE, Access, Kind, Space, Trace, read_metadata

COLUMNS = ("pc", "space", "kind", "width", "mask", "cta", "warp", "addr")


def make_access(addresses, width=4, pc=0x10, space=Space.SHARED, kind=Kind.LOAD):
    return Access(pc, space, kind, width, tuple(addresses))


def contiguous(width=4, base=0) -> list[int | None]:
    return [base + width * lane for lane in range(WARP_SIZE)]


def test_mask_marks_the_lanes_with_an_address():
    addresses = contiguous()
    addresses[3] = None
    addresses[31] = None
    trace = Trace.from_accesses([make_access(addresses)])
    assert int(trace.mask[0]) == 0xFFFFFFFF & ~(1 << 3) & ~(1 << 31)
    assert trace.lane_addresses(0) == tuple(addresses)


def test_active_matches_the_mask():
    addresses: list[int | None] = [None] * WARP_SIZE
    addresses[0] = 0
    addresses[5] = 20
    trace = Trace.from_accesses([make_access(addresses)])
    assert np.flatnonzero(trace.active()[0]).tolist() == [0, 5]


def test_access_returns_the_row_it_was_built_from():
    original = Access(0x2A0, Space.GLOBAL, Kind.STORE, 8, tuple(contiguous(8, 4096)), cta=3, warp=2)
    trace = Trace.from_accesses([original])
    assert trace.access(0) == original


def test_save_and_load_keep_columns_and_metadata(tmp_path):
    trace = Trace.from_accesses(
        [make_access(contiguous()), make_access(contiguous(16), width=16, pc=0x20)],
        metadata={"kernel": "k", "arch": "sm_120"},
    )
    path = tmp_path / "trace.npz"
    trace.save(path)
    loaded = Trace.load(path)
    assert len(loaded) == 2
    assert loaded.metadata["kernel"] == "k"
    assert loaded.metadata["arch"] == "sm_120"
    for row in range(2):
        assert loaded.access(row) == trace.access(row)


def test_load_rejects_another_format_version(tmp_path):
    trace = Trace.from_accesses([make_access(contiguous())])
    path = tmp_path / "trace.npz"
    trace.save(path)
    with np.load(path) as data:
        columns = {name: data[name] for name in data.files if name != "metadata"}
    with open(path, "wb") as handle:
        np.savez(handle, metadata=np.array('{"format": 99}'), **columns)
    with pytest.raises(ValueError, match="format 99"):
        Trace.load(path)


def test_with_addresses_leaves_the_original_unchanged():
    trace = Trace.from_accesses([make_access(contiguous())])
    moved = trace.with_addresses(trace.addr + 128)
    assert trace.lane_addresses(0)[0] == 0
    assert moved.lane_addresses(0)[0] == 128


def test_misaligned_address_is_rejected():
    addresses = contiguous(8)
    addresses[7] = 60
    with pytest.raises(ValueError, match="not aligned"):
        Trace.from_accesses([make_access(addresses, width=8)])


def test_unknown_width_is_rejected():
    with pytest.raises(ValueError, match="widths"):
        Trace.from_accesses([make_access(contiguous(), width=3)])


def test_row_without_active_lane_is_rejected():
    with pytest.raises(ValueError, match="active lane"):
        Trace.from_accesses([make_access([None] * WARP_SIZE)])


def test_access_with_wrong_lane_count_is_rejected():
    with pytest.raises(ValueError, match="32 lane addresses"):
        Trace.from_accesses([make_access(contiguous()[:16])])


def saved_columns(tmp_path):
    path = tmp_path / "trace.npz"
    Trace.from_accesses([make_access(contiguous())], {"kernel": "k"}).save(path)
    with np.load(path) as data:
        return path, {name: data[name] for name in data.files}


def test_load_names_a_missing_column(tmp_path):
    path, columns = saved_columns(tmp_path)
    del columns["mask"]
    with open(path, "wb") as handle:
        np.savez(handle, **columns)
    with pytest.raises(ValueError, match="lacks the columns mask"):
        Trace.load(path)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ("[1, 2]", "no JSON object"),
        ('{"format": 1, "lines": [1]}', "lines must map"),
        ('{"format": 1, "lines": {"0x10": "5"}}', "lines must map"),
        ('{"format": 1, "source": "one line"}', "list of lines"),
        ('{"format": 1, "kernel": 7}', "kernel must be text"),
        ('{"format": 1, "opcodes": {"0x10": 7}}', "opcodes must map"),
    ],
)
def test_load_rejects_malformed_metadata(tmp_path, metadata, message):
    path, columns = saved_columns(tmp_path)
    columns["metadata"] = np.array(metadata)
    with open(path, "wb") as handle:
        np.savez(handle, **columns)
    with pytest.raises(ValueError, match=message):
        Trace.load(path)


@pytest.mark.parametrize(
    "metadata",
    [
        np.array("not json {"),
        np.array([{"format": 1}], dtype=object),
        np.array('{"format": 1, "kernel": 7}'),
    ],
    ids=["text", "objects", "field"],
)
def test_load_names_the_file_of_malformed_metadata(tmp_path, metadata):
    path, columns = saved_columns(tmp_path)
    columns["metadata"] = metadata
    with open(path, "wb") as handle:
        np.savez(handle, **columns)
    with pytest.raises(ValueError, match=re.escape(str(path))):
        Trace.load(path)
    with pytest.raises(ValueError, match=re.escape(str(path))):
        read_metadata(path)


def test_load_names_a_column_of_objects_as_no_readable_trace(tmp_path):
    path, columns = saved_columns(tmp_path)
    columns["cta"] = np.array([0], dtype=object)
    with open(path, "wb") as handle:
        np.savez(handle, **columns)
    with pytest.raises(ValueError, match=f"{re.escape(str(path))} is no readable trace file"):
        Trace.load(path)


def test_load_rejects_a_file_that_is_no_trace(tmp_path):
    path = tmp_path / "array.npz"
    with open(path, "wb") as handle:
        np.save(handle, np.zeros(3))
    with pytest.raises(ValueError, match="no .npz trace file"):
        Trace.load(path)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("pc", np.array([-16]), "outside 0"),
        ("mask", np.array([1 << 32]), "outside 0"),
        ("cta", np.array([0.5]), "needs integers"),
        ("warp", np.array([True]), "needs integers"),
        ("pc", np.array(16), "one value per row"),
    ],
)
def test_columns_with_values_that_do_not_fit_are_rejected(column, values, message):
    trace = Trace.from_accesses([make_access(contiguous())])
    columns = {name: getattr(trace, name) for name in COLUMNS}
    columns[column] = values
    with pytest.raises(ValueError, match=message):
        Trace(**columns)


def test_negative_pc_of_an_access_is_rejected():
    with pytest.raises(ValueError, match="column pc"):
        Trace.from_accesses([make_access(contiguous(), pc=-4)])


def test_load_rejects_a_damaged_member(tmp_path):
    path = tmp_path / "trace.npz"
    Trace.from_accesses([make_access(contiguous())]).save(path)
    damage_member(path, b"addr.npy")
    with pytest.raises(ValueError, match="no readable trace file"):
        Trace.load(path)


def test_load_names_a_file_of_other_bytes_as_no_trace_without_pickle_advice(tmp_path):
    path = tmp_path / "junk.npz"
    path.write_bytes(b"not a zip file")
    with pytest.raises(ValueError, match="no .npz trace file") as error:
        Trace.load(path)
    assert "pickle" not in str(error.value)


def test_read_metadata_skips_the_columns_but_checks_the_format(tmp_path):
    path = tmp_path / "trace.npz"
    Trace.from_accesses([make_access(contiguous())], {"kernel": "k"}).save(path)
    damage_member(path, b"addr.npy")
    assert read_metadata(path)["kernel"] == "k"
    with pytest.raises(ValueError, match="no readable trace file"):
        Trace.load(path)
    newer = tmp_path / "newer.npz"
    trace = Trace.from_accesses([make_access(contiguous())])
    columns = {name: getattr(trace, name) for name in COLUMNS}
    np.savez_compressed(newer, metadata=np.array(json.dumps({"format": 2})), **columns)
    with pytest.raises(ValueError, match="trace format 2"):
        read_metadata(newer)
    junk = tmp_path / "junk.npz"
    junk.write_bytes(b"x")
    with pytest.raises(ValueError, match="no .npz trace file"):
        read_metadata(junk)


def test_load_names_a_member_with_an_unknown_compression_as_no_readable_trace(tmp_path):
    path = tmp_path / "trace.npz"
    Trace.from_accesses([make_access(contiguous())]).save(path)
    data = bytearray(path.read_bytes())
    for signature, field in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        start = data.find(signature)
        while start >= 0:
            data[start + field : start + field + 2] = (99).to_bytes(2, "little")
            start = data.find(signature, start + 4)
    path.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="no readable trace file"):
        Trace.load(path)


def damage_member(path, name):
    data = bytearray(path.read_bytes())
    marker = data.find(name)
    for offset in range(marker + 40, marker + 60):
        data[offset] ^= 0xFF
    path.write_bytes(bytes(data))


@pytest.mark.parametrize(
    "header",
    [
        "{'descr : '<U622', 'fortran_order': False, 'shape': (), }",
        "{'descr': '<U8', 'fortran_order': False, 'shape': (10**30,), }",
        "{'descr': '<U8', 'fortran_order': False, 'shape': (1000000000000000000000000000000,), }",
    ],
    ids=["unterminated", "expression", "huge-shape"],
)
def test_load_turns_an_array_header_that_numpy_cannot_read_into_a_value_error(tmp_path, header):
    path = tmp_path / "trace.npz"
    Trace.from_accesses([make_access(contiguous())]).save(path)
    header += " " * ((64 - (10 + len(header) + 1) % 64) % 64) + "\n"
    member = b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little") + header.encode()
    with zipfile.ZipFile(path) as source:
        members = {info.filename: source.read(info.filename) for info in source.infolist()}
    members["metadata.npy"] = member + bytes(16)
    with zipfile.ZipFile(path, "w") as target:
        for name, data in members.items():
            target.writestr(name, data)
    with pytest.raises(ValueError):
        read_metadata(path)


@pytest.mark.parametrize("error", [IndentationError, TabError, OverflowError, MemoryError])
def test_load_names_each_reader_failure_as_no_readable_trace(tmp_path, monkeypatch, error):
    path = tmp_path / "trace.npz"
    Trace.from_accesses([make_access(contiguous())]).save(path)

    def failing(archive, name):
        raise error("header")

    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", failing)
    with pytest.raises(ValueError, match="no readable trace file"):
        read_metadata(path)


def test_columns_of_time_deltas_are_rejected():
    trace = Trace.from_accesses([make_access(contiguous())])
    columns = {name: getattr(trace, name) for name in COLUMNS}
    columns["cta"] = np.array([5], dtype="timedelta64[s]")
    with pytest.raises(ValueError, match="needs integers"):
        Trace(**columns)


def test_opcode_gives_the_sass_opcode_kept_for_a_pc_and_nothing_for_others():
    rows = tuple(16 * lane for lane in range(WARP_SIZE))
    access = Access(0x40, Space.SHARED, Kind.LOAD, 16, rows)
    trace = Trace.from_accesses([access], {"opcodes": {"0x40": "LDSM.16.M88.4"}})
    assert trace.opcode(0x40) == "LDSM.16.M88.4"
    assert trace.opcode(0x50) == ""


def test_copy_partners_pair_the_two_sides_of_each_cp_async_by_cta_and_warp():
    pc, lanes = 0x40, tuple(16 * lane for lane in range(WARP_SIZE))
    accesses = [
        Access(pc, Space.GLOBAL, Kind.LOAD, 16, lanes, warp=0),
        Access(pc, Space.GLOBAL, Kind.LOAD, 16, lanes, warp=1),
        Access(pc, Space.SHARED, Kind.STORE, 16, lanes, warp=1),
        Access(pc, Space.SHARED, Kind.STORE, 16, lanes, warp=0),
        Access(0x80, Space.SHARED, Kind.LOAD, 16, lanes, warp=0),
    ]
    trace = Trace.from_accesses(accesses, {"opcodes": {"0x40": "LDGSTS.E.BYPASS.128"}})
    assert trace.copy_partners == {0: 3, 3: 0, 1: 2, 2: 1}
    assert Trace.from_accesses(accesses).copy_partners == {}


def test_copy_sources_mark_the_lanes_and_the_warps_that_fill_zeros():
    lanes = tuple(16 * lane for lane in range(WARP_SIZE))
    reads = tuple(None if lane < 4 else 16 * lane for lane in range(WARP_SIZE))
    accesses = [
        Access(0x40, Space.GLOBAL, Kind.LOAD, 16, reads, warp=0),
        Access(0x40, Space.SHARED, Kind.STORE, 16, lanes, warp=0),
        Access(0x40, Space.SHARED, Kind.STORE, 16, lanes, warp=1),
    ]
    trace = Trace.from_accesses(accesses, {"opcodes": {"0x40": "LDGSTS.E.BYPASS.128"}})
    assert trace.copy_sources[1] == reads
    assert trace.copy_sources[2] == (None,) * WARP_SIZE


def test_a_subset_of_a_trace_keeps_the_global_side_of_each_copy():
    lanes = tuple(16 * lane for lane in range(WARP_SIZE))
    apart = tuple(32 * lane for lane in range(WARP_SIZE))
    accesses = [
        Access(0x40, Space.GLOBAL, Kind.LOAD, 16, apart),
        Access(0x40, Space.SHARED, Kind.STORE, 16, lanes),
    ]
    trace = Trace.from_accesses(accesses, {"opcodes": {"0x40": "LDGSTS.E.BYPASS.128"}})
    shared = trace.select(trace.space == Space.SHARED)
    assert shared.copy_sources == {0: apart}
    assert shared.with_addresses(shared.addr).copy_sources == {0: apart}


def test_a_subset_of_a_trace_keeps_the_shared_side_of_each_copy():
    lanes = tuple(16 * lane for lane in range(WARP_SIZE))
    apart = tuple(32 * lane for lane in range(WARP_SIZE))
    accesses = [
        Access(0x40, Space.GLOBAL, Kind.LOAD, 16, lanes),
        Access(0x40, Space.SHARED, Kind.STORE, 16, apart),
    ]
    trace = Trace.from_accesses(accesses, {"opcodes": {"0x40": "LDGSTS.E.BYPASS.128"}})
    assert trace.copy_destinations == {0: apart}
    assert trace.select(trace.space == Space.GLOBAL).copy_destinations == {0: apart}
    moved = trace.with_addresses(
        trace.addr + np.uint64(16) * (trace.space == Space.SHARED)[:, None]
    )
    assert moved.copy_destinations == {0: tuple(address + 16 for address in apart)}
