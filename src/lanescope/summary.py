import collections
import concurrent.futures
import dataclasses
import gc
import multiprocessing
import os
import pathlib
import re
import sys
from collections.abc import Generator, Iterable, Sequence

from lanescope import kernel_names
from lanescope.model.counters import count
from lanescope.model.data_dependence import data_dependent
from lanescope.trace import Trace

# The recorder names the traces of one run trace-<n>.npz, and the launches of a run add
# up per kernel. Any other trace file stands for a run of its own, such as a demo.
RECORDED_NAME = re.compile(r"trace-\d+\.npz")
UNITS = ("wavefronts", "sectors")
SKIPPED_FOLDERS = frozenset({"__pycache__", "node_modules"})
# The lines of each unit that a summary keeps, worst first.
MAX_LINES = 500
# A summary reads at most this many trace files, which bounds a mistyped folder.
MAX_TRACES = 20_000
# Summaries of more traces than this read them in several processes at once.
PARALLEL_TRACES = 8
MAX_WORKERS = 8
# Each worker has at most this many traces queued, which bounds the work that a program
# still waits for once it stops.
WINDOW_PER_WORKER = 4


@dataclasses.dataclass(frozen=True)
class Instruction:
    """The counts of one memory instruction in one trace."""

    pc: int
    space: str
    unit: str
    kind: str
    width: int
    line: int | None
    source: str
    instruction: str
    requests: int
    achieved: int
    ideal: int
    excessive: int
    data_dependent: bool
    fewest: int
    most: int


@dataclasses.dataclass(frozen=True)
class TraceFacts:
    """What a summary keeps of one trace: its kernel, its recording limits, and its counts."""

    kernel: str
    source_file: str
    dropped: int
    has_lines: bool
    instructions: tuple[Instruction, ...]


def trace_files(root: pathlib.Path, limit: int) -> list[pathlib.Path]:
    """Return up to `limit` .npz files of a file or a folder, skipping hidden and cache folders."""
    if not root.is_dir():
        return [root] if root.suffix == ".npz" and limit > 0 else []
    found: list[pathlib.Path] = []
    for folder, names, files in os.walk(root):
        names[:] = sorted(
            name for name in names if not name.startswith(".") and name not in SKIPPED_FOLDERS
        )
        for name in sorted(files):
            if name.endswith(".npz"):
                if len(found) == limit:
                    return found
                found.append(pathlib.Path(folder) / name)
    return found


def run_of(path: pathlib.Path) -> pathlib.Path:
    """Return the run of a trace file: the folder of a recorded trace, or the file itself."""
    return path.parent if RECORDED_NAME.fullmatch(path.name) else path


def read_facts(path: pathlib.Path) -> TraceFacts | None:
    """Return the facts of a trace file, or None for a file that holds no trace."""
    try:
        return trace_facts(Trace.load(path))
    except (ValueError, OSError):
        return None


def read_many(
    paths: Sequence[pathlib.Path], processes: bool = False
) -> Generator[TraceFacts | None, None, None]:
    """Yield the facts of each trace file in order, read in several processes on request.

    Each worker imports the main module of the program again, therefore only a program
    whose main module guards its start, such as the lanescope command, asks for processes.
    """
    workers = min(len(paths), os.cpu_count() or 1, MAX_WORKERS)
    many = processes and len(paths) > PARALLEL_TRACES and workers > 1
    if not many or not _main_importable():
        yield from (read_facts(path) for path in paths)
        return
    # Fresh interpreters suit a caller with threads of its own, such as the server of the GUI.
    context = multiprocessing.get_context("spawn")
    pool = concurrent.futures.ProcessPoolExecutor(workers, mp_context=context)
    pending: collections.deque[concurrent.futures.Future] = collections.deque()
    try:
        for path in paths:
            pending.append(pool.submit(read_facts, path))
            if len(pending) > WINDOW_PER_WORKER * workers:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()
    finally:
        # A reader closed early, as by a stopping server, drops the traces still queued.
        pool.shutdown(wait=True, cancel_futures=True)
        # The queues of a closed pool hold semaphores until a collection frees them, and a
        # program stopped before that would report them as leaked.
        gc.collect()


def _main_importable() -> bool:
    """Return whether a new interpreter can import the main module, as each worker does."""
    path = getattr(sys.modules.get("__main__"), "__file__", None)
    return path is None or pathlib.Path(path).is_file()


def trace_facts(trace: Trace) -> TraceFacts:
    """Return the facts of one trace that a summary adds up."""
    counters = count(trace)
    dependent = data_dependent(trace, counters)
    instructions = []
    for counter in counters:
        line = trace.source_line(counter.pc)
        key = (counter.pc, int(counter.space), int(counter.kind), counter.width)
        instructions.append(
            Instruction(
                pc=counter.pc,
                space=counter.space.name.lower(),
                unit=counter.unit,
                kind=counter.kind.name.lower(),
                width=counter.width,
                line=line[0] if line else None,
                source=line[1] if line else "",
                instruction=trace.opcode(counter.pc)
                or f"{counter.kind.name.lower()} {counter.width} B",
                requests=counter.requests,
                achieved=counter.achieved,
                ideal=counter.ideal,
                excessive=counter.excessive,
                data_dependent=key in dependent,
                fewest=counter.fewest,
                most=counter.most,
            )
        )
    metadata = trace.metadata
    return TraceFacts(
        kernel=str(metadata.get("kernel", "")),
        source_file=str(metadata.get("source_file", "")),
        dropped=int(metadata.get("dropped", 0)),
        has_lines=bool(metadata.get("lines")),
        instructions=tuple(instructions),
    )


def summarize(items: Iterable[tuple[str, str, TraceFacts | None]]) -> dict:
    """Return the counts of many traces by kernel and by source line, worst first.

    Each item names a trace by the id that opens it and by its run, with its facts, or
    with None for a file that holds no trace. The lines keep only instructions with
    extra passes or sectors.
    """
    traces = unreadable = without_lines = dropped_traces = dropped = 0
    runs: dict[str, None] = {}
    kernels: dict[tuple[str, str], dict] = {}
    lines: dict[tuple, dict] = {}
    for trace_id, run, facts in items:
        if facts is None:
            unreadable += 1
            continue
        traces += 1
        runs[run] = None
        without_lines += not facts.has_lines
        if facts.dropped:
            dropped_traces += 1
            dropped += facts.dropped
        kernel = kernels.setdefault((run, facts.kernel), _kernel_entry(run, facts.kernel))
        kernel["launches"] += 1
        for item in facts.instructions:
            _add(kernel["totals"][item.unit], item)
            worst = kernel["worst"][item.unit]
            if item.excessive > 0 and (worst is None or item.excessive > worst["excessive"]):
                kernel["worst"][item.unit] = _place(trace_id, item)
            if item.excessive == 0:
                continue
            where = item.line if item.line is not None else f"{item.pc:#x}"
            key = (run, facts.kernel, facts.source_file, where, item.instruction, item.space)
            entry = lines.setdefault(key, _line_entry(run, facts, item))
            _add(entry, item)
            entry["traces"].add(trace_id)
            entry["data_dependent"] |= item.data_dependent
            if item.data_dependent:
                low, high = entry["excess_range"] or (item.fewest, item.most)
                entry["excess_range"] = (min(low, item.fewest), max(high, item.most))
            if item.excessive > entry["worst"]["excessive"]:
                entry["worst"] = _place(trace_id, item)
    names = kernel_names.readable(kernel for _, kernel in kernels)
    totals = {unit: _zero() for unit in UNITS}
    for kernel in kernels.values():
        full = names[kernel["kernel"]]
        kernel |= {"name": kernel_names.short(full), "full_name": full}
        for unit in UNITS:
            for name, value in kernel["totals"][unit].items():
                totals[unit][name] += value
            if kernel["worst"][unit]:
                kernel["worst"][unit].pop("excessive")
    ranked_lines = []
    for unit in UNITS:
        found = [entry for entry in lines.values() if entry["unit"] == unit]
        found.sort(
            key=lambda entry: (
                -entry["excessive"], entry["run"], entry["kernel"], entry["file"],
                entry["line"] or 0, entry["instruction"],
            )
        )  # fmt: skip
        for entry in found[:MAX_LINES]:
            kernel = kernels[(entry["run"], entry["kernel"])]
            entry["worst"].pop("excessive")
            entry |= {
                "pc": entry["worst"]["pc"],
                "name": kernel["name"],
                "full_name": kernel["full_name"],
                "launches": len(entry.pop("traces")),
                "kernel_achieved": kernel["totals"][unit]["achieved"],
                "excess_range": list(entry["excess_range"]) if entry["excess_range"] else None,
            }
            ranked_lines.append(entry)
    ordered = sorted(
        kernels.values(),
        key=lambda kernel: (
            -kernel["totals"]["wavefronts"]["excessive"],
            -kernel["totals"]["sectors"]["excessive"],
            kernel["run"],
            kernel["name"],
        ),
    )
    return {
        "traces": traces,
        "unreadable": unreadable,
        "runs": list(runs),
        "totals": totals,
        "without_lines": without_lines,
        "dropped": {"traces": dropped_traces, "records": dropped},
        "kernels": ordered,
        "lines": ranked_lines,
    }


def _kernel_entry(run: str, kernel: str) -> dict:
    return {
        "run": run,
        "kernel": kernel,
        "launches": 0,
        "totals": {unit: _zero() for unit in UNITS},
        "worst": {unit: None for unit in UNITS},
    }


def _line_entry(run: str, facts: TraceFacts, item: Instruction) -> dict:
    return {
        "run": run,
        "kernel": facts.kernel,
        "file": facts.source_file if item.line is not None else "",
        "line": item.line,
        "source": item.source,
        "space": item.space,
        "kind": item.kind,
        "width": item.width,
        "unit": item.unit,
        "instruction": item.instruction,
        "traces": set(),
        "data_dependent": False,
        "excess_range": None,
        "worst": {"excessive": -1},
        **_zero(),
    }


def _zero() -> dict[str, int]:
    return {"requests": 0, "achieved": 0, "ideal": 0, "excessive": 0}


def _add(total: dict, item: Instruction) -> None:
    for name in ("requests", "achieved", "ideal", "excessive"):
        total[name] += getattr(item, name)


def _place(trace_id: str, item: Instruction) -> dict:
    """Return where the page opens an instruction: its trace, its PC, and its space."""
    return {"id": trace_id, "pc": f"{item.pc:#x}", "space": item.space, "excessive": item.excessive}
