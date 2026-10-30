import dataclasses
import enum
import functools
import json
import pathlib
import tokenize
import zipfile
import zlib
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

WARP_SIZE = 32
FORMAT_VERSION = 1
WIDTHS = (1, 2, 4, 8, 16)
ZIP_MAGIC = b"PK\x03\x04"
# zipfile raises these for an unknown compression method, an encrypted member, and a
# member header that claims a huge array. NumPy raises the tokenize and syntax errors
# for an array header that no Python reader can parse, and an overflow for a shape
# beyond 64 bits.
ARCHIVE_ERRORS = (
    NotImplementedError, RuntimeError, MemoryError, tokenize.TokenError, SyntaxError,
    OverflowError,
)  # fmt: skip

_LANE_BITS = np.uint64(1) << np.arange(WARP_SIZE, dtype=np.uint64)
_TEXT_FIELDS = ("kernel", "arch")


class Space(enum.IntEnum):
    """Memory space of an access."""

    SHARED = 0
    GLOBAL = 1


class Kind(enum.IntEnum):
    """Direction of an access."""

    LOAD = 0
    STORE = 1


@dataclasses.dataclass(frozen=True)
class Access:
    """One warp-level access, with None as the address of each inactive lane."""

    pc: int
    space: Space
    kind: Kind
    width: int
    addresses: tuple[int | None, ...]
    cta: int = 0
    warp: int = 0


@dataclasses.dataclass(frozen=True, eq=False)
class Trace:
    """Warp-level accesses of one kernel run, stored column by column."""

    pc: np.ndarray
    space: np.ndarray
    kind: np.ndarray
    width: np.ndarray
    mask: np.ndarray
    cta: np.ndarray
    warp: np.ndarray
    addr: np.ndarray
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        # Columns arrive from files and builders in varied dtypes. Fixing them here
        # lets every analysis rely on one representation.
        for name, dtype in _COLUMN_DTYPES.items():
            object.__setattr__(self, name, _column(name, getattr(self, name), dtype))
        _check_metadata(self.metadata)
        _validate(self)

    def __len__(self) -> int:
        return int(self.pc.shape[0])

    @classmethod
    def from_accesses(
        cls, accesses: Iterable[Access], metadata: dict[str, Any] | None = None
    ) -> "Trace":
        """Build a trace from warp-level accesses."""
        rows = list(accesses)
        for access in rows:
            if len(access.addresses) != WARP_SIZE:
                raise ValueError(f"an access needs {WARP_SIZE} lane addresses")
        addr = np.array(
            [[0 if a is None else a for a in access.addresses] for access in rows],
            dtype=object,
        ).reshape(len(rows), WARP_SIZE)
        mask = [
            sum(1 << lane for lane, a in enumerate(access.addresses) if a is not None)
            for access in rows
        ]
        return cls(
            pc=np.array([access.pc for access in rows], dtype=object),
            space=np.array([int(access.space) for access in rows], dtype=object),
            kind=np.array([int(access.kind) for access in rows], dtype=object),
            width=np.array([access.width for access in rows], dtype=object),
            mask=np.array(mask, dtype=object),
            cta=np.array([access.cta for access in rows], dtype=object),
            warp=np.array([access.warp for access in rows], dtype=object),
            addr=addr,
            metadata=dict(metadata or {}),
        )

    def active(self) -> np.ndarray:
        """Return a boolean array of shape (rows, 32) that marks the active lanes."""
        return (self.mask.astype(np.uint64)[:, None] & _LANE_BITS) != 0

    def lane_addresses(self, row: int) -> tuple[int | None, ...]:
        """Return the address of each lane of one row, with None for inactive lanes."""
        mask = int(self.mask[row])
        return tuple(
            int(self.addr[row, lane]) if mask >> lane & 1 else None for lane in range(WARP_SIZE)
        )

    def access(self, row: int) -> Access:
        """Return one row as an access."""
        return Access(
            pc=int(self.pc[row]),
            space=Space(int(self.space[row])),
            kind=Kind(int(self.kind[row])),
            width=int(self.width[row]),
            addresses=self.lane_addresses(row),
            cta=int(self.cta[row]),
            warp=int(self.warp[row]),
        )

    def with_addresses(self, addr: np.ndarray) -> "Trace":
        """Return a copy of the trace with new lane addresses."""
        return dataclasses.replace(self, addr=addr)

    def select(self, rows: np.ndarray) -> "Trace":
        """Return the rows picked by a boolean mask or an index array."""
        return dataclasses.replace(
            self, **{name: getattr(self, name)[rows] for name in _COLUMN_DTYPES}
        )

    def source_line(self, pc: int) -> tuple[int, str] | None:
        """Return the source line number and text of a PC, when the metadata maps it."""
        line = self.metadata.get("lines", {}).get(f"{pc:#x}")
        if line is None:
            return None
        source = self.metadata.get("source", [])
        return line, source[line - 1].strip() if 0 < line <= len(source) else ""

    def opcode(self, pc: int) -> str:
        """Return the SASS opcode that the metadata keeps for a PC, or an empty string."""
        return str(self.metadata.get("opcodes", {}).get(f"{pc:#x}", ""))

    @functools.cached_property
    def copy_partners(self) -> dict[int, int]:
        """Return, for each row of a cp.async, the row of the other side of the same copy.

        A cp.async records its global row before its shared row at one PC, and the two
        sides pair up within one warp of one CTA.
        """
        pcs = [
            int(pc, 16)
            for pc, code in self.metadata.get("opcodes", {}).items()
            if code.startswith("LDGSTS")
        ]
        partners: dict[int, int] = {}
        reads: dict[tuple[int, int, int], int] = {}
        for row in np.flatnonzero(np.isin(self.pc, pcs)).tolist() if pcs else []:
            key = (int(self.pc[row]), int(self.cta[row]), int(self.warp[row]))
            if self.space[row] == Space.GLOBAL:
                reads[key] = row
            elif key in reads:
                read = reads.pop(key)
                partners[row], partners[read] = read, row
        return partners

    def save(self, path: str | pathlib.Path) -> None:
        """Write the trace to an .npz file."""
        metadata = dict(self.metadata, format=FORMAT_VERSION)
        with open(path, "wb") as handle:
            np.savez_compressed(
                handle,
                metadata=np.array(json.dumps(metadata, sort_keys=True)),
                **{name: getattr(self, name) for name in _COLUMN_DTYPES},
            )

    @classmethod
    def load(cls, path: str | pathlib.Path) -> "Trace":
        """Read a trace from an .npz file, raising ValueError for a malformed file."""
        metadata, columns = _read(path, tuple(_COLUMN_DTYPES))
        return cls(metadata=metadata, **columns)


def read_metadata(path: str | pathlib.Path) -> dict[str, Any]:
    """Return the metadata of a trace file without reading its columns."""
    return _read(path, ())[0]


def check_lanes(addresses: Sequence[int | None], width: int) -> None:
    """Raise ValueError unless an access has 32 lanes, a valid width, and an active lane."""
    if len(addresses) != WARP_SIZE:
        raise ValueError(f"an access needs {WARP_SIZE} lane addresses")
    if width not in WIDTHS:
        raise ValueError(f"access widths must be among {WIDTHS} bytes")
    if all(address is None for address in addresses):
        raise ValueError("an access needs at least one active lane")


_COLUMN_DTYPES = {
    "pc": np.uint64,
    "space": np.uint8,
    "kind": np.uint8,
    "width": np.uint8,
    "mask": np.uint32,
    "cta": np.uint32,
    "warp": np.uint32,
    "addr": np.uint64,
}


def _column(name: str, values: Any, dtype: type[np.unsignedinteger]) -> np.ndarray:
    array = np.asarray(values)
    if array.size == 0:
        return array.astype(dtype)
    if array.dtype == object:
        integral = all(isinstance(v, int) and not isinstance(v, bool) for v in array.flat)
    else:
        # NumPy counts time deltas as integers, which no column of a trace holds.
        integral = array.dtype.kind in "iu"
    if not integral:
        raise ValueError(f"column {name} needs integers")
    # A plain cast wraps negative and oversized values, which would then pass the checks.
    low, high = int(array.min()), int(array.max())
    if low < 0 or high > int(np.iinfo(dtype).max):
        raise ValueError(f"column {name} holds a value outside 0 to {np.iinfo(dtype).max}")
    return array.astype(dtype)


def _read(
    path: str | pathlib.Path, names: tuple[str, ...]
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    # NumPy reads any other file as a pickle and suggests loading it unsafely. The file
    # stays open here, since NumPy leaves its own handle open when a zip fails to parse.
    with open(path, "rb") as handle:
        if handle.read(len(ZIP_MAGIC)) != ZIP_MAGIC:
            raise ValueError(f"{path} is no .npz trace file")
        handle.seek(0)
        try:
            data = np.load(handle, allow_pickle=False)
        except (zipfile.BadZipFile, EOFError, ValueError, *ARCHIVE_ERRORS) as error:
            raise ValueError(f"{path} is no readable trace file ({error})") from error
        if not isinstance(data, np.lib.npyio.NpzFile):
            raise ValueError(f"{path} is no .npz trace file")
        with data:
            missing = [name for name in (*_COLUMN_DTYPES, "metadata") if name not in data.files]
            if missing:
                raise ValueError(f"{path} lacks the columns {', '.join(missing)}")
            # Members decompress on access, so a damaged member fails only here. NumPy raises
            # ValueError for a member that holds Python objects.
            try:
                text = str(data["metadata"])
                columns = {name: data[name] for name in names}
            except (
                zipfile.BadZipFile, zlib.error, EOFError, OSError, ValueError, *ARCHIVE_ERRORS,
            ) as error:  # fmt: skip
                raise ValueError(f"{path} is no readable trace file ({error})") from error
    try:
        metadata = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"{path} has metadata that is no JSON text ({error})") from error
    if not isinstance(metadata, dict):
        raise ValueError(f"{path} has metadata that is no JSON object")
    if metadata.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"{path} has trace format {metadata.get('format')}, expected {FORMAT_VERSION}"
        )
    try:
        _check_metadata(metadata)
    except ValueError as error:
        raise ValueError(f"{path} has malformed metadata ({error})") from error
    return metadata, columns


def _check_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise ValueError("trace metadata must be a dictionary")
    for field in _TEXT_FIELDS:
        if field in metadata and not isinstance(metadata[field], str):
            raise ValueError(f"metadata field {field} must be text")
    source = metadata.get("source", [])
    if not isinstance(source, list) or not all(isinstance(line, str) for line in source):
        raise ValueError("metadata field source must be a list of lines")
    lines = metadata.get("lines", {})
    if not isinstance(lines, dict) or not all(
        isinstance(pc, str) and isinstance(line, int) and not isinstance(line, bool)
        for pc, line in lines.items()
    ):
        raise ValueError("metadata field lines must map PC text such as 0x120 to line numbers")
    opcodes = metadata.get("opcodes", {})
    if not isinstance(opcodes, dict) or not all(
        isinstance(pc, str) and isinstance(opcode, str) for pc, opcode in opcodes.items()
    ):
        raise ValueError("metadata field opcodes must map PC text such as 0x120 to SASS opcodes")


def _validate(trace: Trace) -> None:
    rows = trace.pc.shape[0] if trace.pc.ndim == 1 else -1
    for name in _COLUMN_DTYPES:
        if name != "addr" and getattr(trace, name).shape != (rows,):
            raise ValueError(f"column {name} needs one value per row")
    if trace.addr.shape != (rows, WARP_SIZE):
        raise ValueError(f"column addr needs {WARP_SIZE} addresses per row")
    if not np.isin(trace.width, WIDTHS).all():
        raise ValueError(f"access widths must be among {WIDTHS} bytes")
    if not np.isin(trace.space, [int(s) for s in Space]).all():
        raise ValueError("unknown memory space")
    if not np.isin(trace.kind, [int(k) for k in Kind]).all():
        raise ValueError("unknown access kind")
    if (trace.mask == 0).any():
        raise ValueError("every row needs at least one active lane")
    # CUDA faults on misaligned accesses, so a misaligned lane marks a broken trace.
    misaligned = trace.active() & (trace.addr % trace.width[:, None].astype(np.uint64) != 0)
    if misaligned.any():
        row = int(np.argwhere(misaligned)[0][0])
        raise ValueError(
            f"row {row} (pc {int(trace.pc[row]):#x}) has an address not aligned "
            f"to its {int(trace.width[row])}-byte width"
        )
