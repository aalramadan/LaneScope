import dataclasses
import json
import os
import pathlib
import re
import subprocess
import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from lanescope.cuda import line_info
from lanescope.recorder.build import RecorderBuild
from lanescope.trace import WARP_SIZE, WIDTHS, Trace

RECORD_DTYPE = np.dtype(
    [
        ("pc", "<u8"),
        ("addr", "<u8", (WARP_SIZE,)),
        ("cta", "<u4"),
        ("warp", "<u4"),
        ("mask", "<u4"),
        ("width", "<u2"),
        ("space", "u1"),
        ("kind", "u1"),
    ]
)
ATOMIC = 2
LAUNCH_FORMAT = 1
LAUNCH_NAME = re.compile(r"launch-(\d+)\.json")
# The files that a recording and its import write, which the next recording replaces.
RECORDING_FILE = re.compile(r"launch-\d+\.(json|bin)|module-\d+\.cubin|trace-\d+\.npz")


def environment(
    build: RecorderBuild,
    out_dir: pathlib.Path,
    capacity: int = 1 << 18,
    ctas: tuple[int, int] = (0, 64),
    kernel: str = "",
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the environment that loads the recorder into a child process.

    The loader splits LD_PRELOAD at spaces as well as colons, so the libraries go
    in by name, with their folders on LD_LIBRARY_PATH, which splits only at colons
    and semicolons. A folder whose path holds either raises ValueError.
    """
    env = dict(os.environ if base is None else base)
    libraries = (build.library, build.sanitizer_library)
    folders = list(dict.fromkeys(str(library.parent.resolve()) for library in libraries))
    for folder in folders:
        if ":" in folder or ";" in folder:
            raise ValueError(
                f"the recorder cannot load a library from {folder}, since LD_LIBRARY_PATH "
                "splits paths at colons and semicolons. Use a folder whose path has neither."
            )
    search = ":".join(folders)
    env["LD_LIBRARY_PATH"] = (
        f"{search}:{env['LD_LIBRARY_PATH']}" if env.get("LD_LIBRARY_PATH") else search
    )
    preload = ":".join(library.name for library in libraries)
    env["LD_PRELOAD"] = f"{preload}:{env['LD_PRELOAD']}" if env.get("LD_PRELOAD") else preload
    env["LANESCOPE_OUT"] = str(out_dir)
    env["LANESCOPE_PATCHES"] = str(build.patches)
    env["LANESCOPE_CAPACITY"] = str(capacity)
    env["LANESCOPE_CTAS"] = f"{ctas[0]}:{ctas[1]}"
    if kernel:
        env["LANESCOPE_KERNEL"] = kernel
    return env


def record(command: Sequence[str], build: RecorderBuild, out_dir: pathlib.Path, **settings) -> int:
    """Run a command under the recorder and return its exit status.

    The run first removes the launch files, modules, and traces of an earlier
    recording in the folder. The recorder numbers the launches of each run from 0,
    so a run with fewer launches would otherwise leave older ones behind.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    env = environment(build, out_dir, **settings)
    for path in out_dir.iterdir():
        if RECORDING_FILE.fullmatch(path.name) and path.is_file():
            path.unlink()
    return subprocess.run(list(command), env=env, check=False).returncode


def launch_files(out_dir: pathlib.Path) -> list[pathlib.Path]:
    """Return the launch-<n>.json files of a recording, in launch order.

    Other names, such as launch-1.orig.json for a copy, hold no launch of the recording.
    """
    numbered = [
        (int(match.group(1)), path)
        for path in out_dir.glob("launch-*.json")
        if (match := LAUNCH_NAME.fullmatch(path.name))
    ]
    return [path for _, path in sorted(numbered)]


def load_launch(info_path: pathlib.Path) -> tuple[dict, np.ndarray]:
    """Return the launch description and its records, raising ValueError for a bad file."""
    try:
        info = json.loads(info_path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{info_path} is no launch description ({error})") from error
    if not isinstance(info, dict):
        raise ValueError(f"{info_path} is no launch description")
    if info.get("format") != LAUNCH_FORMAT:
        raise ValueError(f"{info_path} has launch format {info.get('format')!r}, expected 1")
    if info.get("record_bytes") != RECORD_DTYPE.itemsize:
        raise ValueError(f"{info_path} has {info.get('record_bytes')}-byte records, expected 280")
    data = info_path.with_suffix(".bin").read_bytes()
    if len(data) % RECORD_DTYPE.itemsize:
        raise ValueError(f"{info_path.with_suffix('.bin')} ends inside a record")
    return info, np.frombuffer(data, dtype=RECORD_DTYPE)


def to_trace(info: dict, records: np.ndarray) -> Trace:
    """Return the trace of one launch, without atomics, with PCs relative to the kernel."""
    keep = (records["kind"] != ATOMIC) & (records["mask"] != 0) & np.isin(records["width"], WIDTHS)
    kept = records[keep]
    base = int(info.get("function_pc", 0))
    end = base + int(info.get("function_size", 0)) if info.get("function_size") else 2**64
    relative = base > 0 and bool(((kept["pc"] >= base) & (kept["pc"] < end)).all())
    lanes = (kept["mask"].astype(np.uint64)[:, None] >> np.arange(WARP_SIZE, dtype=np.uint64)) & 1
    metadata = {
        "kernel": str(info.get("kernel", "")),
        "arch": str(info.get("arch", "")),
        "launch": int(info.get("launch", 0)),
        "grid": [int(v) for v in info.get("grid", [])],
        "block": [int(v) for v in info.get("block", [])],
        "records": int(info.get("records", len(records))),
        "dropped": int(info.get("dropped", 0)),
        "skipped": int(len(records) - len(kept)),
        "pc_base": base if relative else 0,
    }
    return Trace(
        pc=kept["pc"] - np.uint64(base) if relative else kept["pc"],
        space=kept["space"],
        kind=kept["kind"],
        width=kept["width"],
        mask=kept["mask"],
        cta=kept["cta"],
        warp=kept["warp"],
        # Inactive lanes may hold any value, so they are cleared for a stable trace.
        addr=np.where(lanes == 1, kept["addr"], np.uint64(0)),
        metadata=metadata,
    )


def import_recording(
    out_dir: pathlib.Path, disassemblers: Sequence[pathlib.Path] = ()
) -> list[pathlib.Path]:
    """Write trace-<n>.npz next to each launch file of a recording and return their paths.

    With nvdisasm among the disassemblers, each trace also maps its PCs to the
    source lines of its kernel.
    """
    paths = []
    modules: dict[int, dict[str, line_info.FunctionLines]] = {}
    for info_path in launch_files(out_dir):
        info, records = load_launch(info_path)
        trace = to_trace(info, records)
        if disassemblers and trace.metadata["pc_base"]:
            lines = _kernel_lines(out_dir, info, disassemblers, modules)
            trace = dataclasses.replace(trace, metadata=trace.metadata | lines)
        path = out_dir / f"trace-{info_path.stem.split('-')[1]}.npz"
        trace.save(path)
        paths.append(path)
    return paths


def _kernel_lines(
    out_dir: pathlib.Path,
    info: dict,
    disassemblers: Sequence[pathlib.Path],
    modules: dict[int, dict[str, line_info.FunctionLines]],
) -> dict:
    module = info.get("module")
    cubin = out_dir / f"module-{module}.cubin"
    if not isinstance(module, int) or not cubin.is_file():
        return {}
    if module not in modules:
        # A module that nvdisasm cannot read leaves its traces without lines.
        try:
            modules[module] = line_info.module_lines(disassemblers, cubin)
        except ValueError as error:
            warnings.warn(f"{error}. Its traces carry no source lines.", stacklevel=2)
            modules[module] = {}
    return line_info.metadata(modules[module].get(str(info.get("kernel", "")), {}))
