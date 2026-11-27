import collections
import concurrent.futures
import dataclasses
import gc
import multiprocessing
import os
import pathlib
import re
import statistics
import sys
from collections.abc import Generator, Iterable, Mapping, Sequence

from lanescope import kernel_names, kernel_times
from lanescope.model import data_dependence
from lanescope.model.counters import count_instructions
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
    # The hashes of the distinct patterns of lane offsets and the distinct excess counts of
    # a shared instruction with excess, which judge its line over all launches of its kernel.
    patterns: frozenset[int] | None = frozenset()
    excess_counts: frozenset[int] = frozenset()
    # The file, line, and text that an inlined instruction comes from, such as a line of a
    # header that the kernel calls.
    origin: tuple[str, int, str] | None = None


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


def run_times(runs: Mapping[str, pathlib.Path]) -> tuple[dict[str, dict], list[dict]]:
    """Return the kernel times of each run that has them by its name, and the unreadable ones.

    The times of a recording lie in its folder, and those of a trace file beside it. An
    unreadable file names its run, its path, and the reason.
    """
    found = {}
    unreadable = []
    for name, run in runs.items():
        folder = run if run.is_dir() else run.parent
        try:
            result = kernel_times.read_times(folder)
        except ValueError as error:
            path = str(folder / kernel_times.TIMES_FILE)
            unreadable.append({"run": name, "file": path, "reason": str(error)})
            continue
        if result is not None:
            found[name] = result
    return found, unreadable


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
    counters = count_instructions(trace)
    evidence = data_dependence.input_evidence(trace, counters)
    instructions = []
    for counter in counters:
        line = trace.source_line(counter.pc)
        found = evidence.get((counter.pc, int(counter.space), int(counter.kind), counter.width))
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
                data_dependent=found is not None and found.data_dependent,
                fewest=counter.fewest,
                most=counter.most,
                patterns=found.patterns if found else frozenset(),
                excess_counts=found.excess_counts if found else frozenset(),
                origin=trace.origin(counter.pc),
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


def summarize(
    items: Iterable[tuple[str, str, TraceFacts | None]],
    by_template: bool = True,
    times: Mapping[str, dict] | None = None,
    unreadable_times: Sequence[dict] = (),
) -> dict:
    """Return the counts of many traces by kernel template and by source line, worst first.

    Each item names a trace by the id that opens it and by its run, with its facts, or
    with None for a file that holds no trace. Each entry of a template lists its kernel
    instances, and without `by_template` each instance stands alone. The lines keep only
    instructions with extra passes or sectors. `times` holds the kernel times of the runs
    that lanescope time timed, by run, and `unreadable_times` the files of times that
    run_times could not read.
    """
    times = times or {}
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
                kernel["worst"][item.unit] = _page_opening(trace_id, item)
            if item.excessive == 0:
                continue
            key = (run, facts.kernel, *_line_place(facts, item), item.instruction, item.space)
            entry = lines.setdefault(key, _line_entry(run, facts, item))
            _add_to_line(entry, trace_id, item)
    names = kernel_names.readable(kernel for _, kernel in kernels)
    totals = {unit: _empty_totals() for unit in UNITS}
    for kernel in kernels.values():
        full = names[kernel["kernel"]]
        kernel |= {"name": kernel_names.short_name(full), "full_name": full}
        kernel |= _instance_time(times.get(kernel["run"]), kernel["kernel"])
        for unit in UNITS:
            for name, value in kernel["totals"][unit].items():
                totals[unit][name] += value
    templates, owners = _group_templates(kernels, by_template)
    ranked_lines = []
    merged = _merge_lines(lines, templates, owners)
    for unit in UNITS:
        found = [entry for entry in merged.values() if entry["unit"] == unit]
        found.sort(
            key=lambda entry: (
                -entry["excessive"], entry["run"], entry["template"], entry["kernel"],
                *_shown_place(entry), entry["instruction"],
            )
        )  # fmt: skip
        for entry in found[:MAX_LINES]:
            template = templates[owners[(entry["run"], entry["kernel"])]]
            ranked_lines.append(_finished_line(entry, template))
    for template in templates.values():
        for place in [template, *template["instances"]]:
            for worst in place["worst"].values():
                if worst:
                    worst.pop("excessive", None)
    return {
        "traces": traces,
        "unreadable": unreadable,
        "runs": list(runs),
        "totals": totals,
        "without_lines": without_lines,
        "dropped": {"traces": dropped_traces, "records": dropped},
        "times": [_run_time(run, times[run]) for run in runs if run in times],
        "unreadable_times": list(unreadable_times),
        "kernels": sorted(templates.values(), key=_kernel_order),
        "lines": ranked_lines,
    }


def summarize_files(
    paths: Sequence[pathlib.Path],
    facts: Iterable[TraceFacts | None],
    by_template: bool = True,
    run: pathlib.Path | None = None,
) -> dict:
    """Return the summary of trace files with the kernel times of their runs.

    Each recorded trace belongs to the run of its folder, unless `run` names one folder or
    trace file for all of them.
    """
    runs = [run or run_of(path) for path in paths]
    times, unreadable_times = run_times({str(found_run): found_run for found_run in runs})
    return summarize(
        (
            (str(path), str(found_run), found)
            for path, found_run, found in zip(paths, runs, facts, strict=True)
        ),
        by_template=by_template,
        times=times,
        unreadable_times=unreadable_times,
    )


def _group_templates(
    kernels: dict[tuple[str, str], dict], by_template: bool
) -> tuple[dict[tuple[str, str], dict], dict[tuple[str, str], tuple[str, str]]]:
    """Return the entries of the templates of each run and the template key of each kernel."""
    templates: dict[tuple[str, str], dict] = {}
    owners = {}
    for key, kernel in kernels.items():
        template = kernel_names.template_name(kernel["full_name"])
        owner = (kernel["run"], template if by_template else kernel["kernel"])
        entry = templates.setdefault(owner, _template_entry(kernel["run"], template))
        owners[key] = owner
        entry["instances"].append(kernel)
        entry["launches"] += kernel["launches"]
        for unit in UNITS:
            for name, value in kernel["totals"][unit].items():
                entry["totals"][unit][name] += value
            worst, found = entry["worst"][unit], kernel["worst"][unit]
            if found and (worst is None or found["excessive"] > worst["excessive"]):
                entry["worst"][unit] = dict(found)
    for entry in templates.values():
        instances = entry["instances"]
        instances.sort(key=_kernel_order)
        if len(instances) == 1:
            entry |= {"name": instances[0]["name"], "full_name": instances[0]["full_name"]}
        else:
            entry |= {
                "name": kernel_names.short_name(entry["template"]),
                "full_name": entry["template"],
            }
        entry |= template_time(instances)
    return templates, owners


def _instance_time(result: dict | None, kernel: str) -> dict:
    """Return the median time of a kernel over the rounds of its run, with each round."""
    found = result["kernels"].get(kernel) if result else None
    if found is None:
        return {"time_ns": None, "timed_launches": None, "round_ns": None}
    return {
        "time_ns": round(statistics.median(found["nanoseconds"])),
        "timed_launches": round(statistics.median(found["launches"])),
        "round_ns": list(found["nanoseconds"]),
    }


def template_time(instances: Sequence[dict]) -> dict:
    """Return the time of the timed instances of a template, round by round.

    The median of the rounds of the sum stays within the range of the rounds, which a sum
    of the medians of the instances can leave.
    """
    timed = [instance for instance in instances if instance["round_ns"] is not None]
    if not timed:
        return {"time_ns": None, "timed_launches": None, "round_ns": None}
    rounds = [sum(values) for values in zip(*(item["round_ns"] for item in timed), strict=True)]
    return {
        "time_ns": round(statistics.median(rounds)),
        "timed_launches": sum(instance["timed_launches"] for instance in timed),
        "round_ns": rounds,
    }


def _run_time(run: str, result: dict) -> dict:
    """Return the kernel time of a run over every kernel it launched, also those without a trace."""
    rounds = kernel_times.round_totals(result)
    return {
        "run": run,
        "rounds": result["rounds"],
        "total_ns": round(statistics.median(rounds)),
        "round_ns": rounds,
    }


def _merge_lines(
    lines: dict[tuple, dict],
    templates: dict[tuple[str, str], dict],
    owners: dict[tuple[str, str], tuple[str, str]],
) -> dict[tuple, dict]:
    """Return the lines of each kernel added up per template.

    Each kernel judges on its own launches whether its line follows the input, since two
    instances with fixed layouts of their own differ like input values. A line without a
    line table stays apart per kernel, since the PCs of two instances do not match.
    """
    merged: dict[tuple, dict] = {}
    for (run, kernel, file, where, instruction, space), line in lines.items():
        line["data_dependent"] |= data_dependence.follows_input_over_launches(
            line["requests"], len(line["traces"]), line["patterns"], line["excess_counts"]
        )
        for name in ("patterns", "excess_counts"):
            del line[name]
        owner = owners[(run, kernel)]
        place = where if line["line"] is not None else (kernel, where)
        key = (owner, file, place, instruction, space)
        entry = merged.get(key)
        if entry is None:
            merged[key] = line | {"template": templates[owner]["template"], "instances": 1}
            continue
        for name in _empty_totals():
            entry[name] += line[name]
        entry["instances"] += 1
        entry["traces"] |= line["traces"]
        entry["data_dependent"] |= line["data_dependent"]
        entry["fewest"] = min(entry["fewest"], line["fewest"])
        entry["most"] = max(entry["most"], line["most"])
        if line["worst"]["excessive"] > entry["worst"]["excessive"]:
            for name in ("worst", "kernel", "file", "line", "source"):
                entry[name] = line[name]
    return merged


def _line_place(facts: TraceFacts, item: Instruction) -> tuple[str, int | str]:
    """Return the file and the line that a line of a summary keys on, or its PC without a line.

    An inlined instruction keys on the line it comes from, therefore the instructions of a
    header that one call line inlines stay apart.
    """
    if item.origin is not None:
        return item.origin[0], item.origin[1]
    return facts.source_file, item.line if item.line is not None else f"{item.pc:#x}"


def _shown_place(entry: dict) -> tuple[str, int]:
    origin = entry["origin"]
    if origin is not None:
        return origin["file"], origin["line"]
    return entry["file"], entry["line"] or 0


def _finished_line(entry: dict, template: dict) -> dict:
    """Return a line as a summary gives it, with the launches of every kernel that adds to it."""
    launches = len(entry.pop("traces"))
    dependent = entry["data_dependent"]
    fewest, most = entry.pop("fewest"), entry.pop("most")
    entry["worst"].pop("excessive")
    return entry | {
        "data_dependent": dependent,
        "pc": entry["worst"]["pc"],
        "name": template["name"],
        "full_name": template["full_name"],
        "launches": launches,
        "kernel_achieved": template["totals"][entry["unit"]]["achieved"],
        "excess_range": [fewest, most] if dependent else None,
    }


def _kernel_order(entry: dict) -> tuple:
    return (
        -entry["totals"]["wavefronts"]["excessive"],
        -entry["totals"]["sectors"]["excessive"],
        entry["run"],
        entry["name"],
    )


def _kernel_entry(run: str, kernel: str) -> dict:
    return {
        "run": run,
        "kernel": kernel,
        "launches": 0,
        "totals": {unit: _empty_totals() for unit in UNITS},
        "worst": {unit: None for unit in UNITS},
    }


def _template_entry(run: str, template: str) -> dict:
    return {
        "run": run,
        "template": template,
        "launches": 0,
        "totals": {unit: _empty_totals() for unit in UNITS},
        "worst": {unit: None for unit in UNITS},
        "instances": [],
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
        "origin": (
            None
            if item.origin is None
            else dict(zip(("file", "line", "text"), item.origin, strict=True))
        ),
        "traces": set(),
        "data_dependent": False,
        "fewest": item.fewest,
        "most": item.most,
        "patterns": set(),
        "excess_counts": set(),
        "worst": {"excessive": -1},
        **_empty_totals(),
    }


def _add_to_line(entry: dict, trace_id: str, item: Instruction) -> None:
    _add(entry, item)
    entry["traces"].add(trace_id)
    entry["data_dependent"] |= item.data_dependent
    entry["fewest"] = min(entry["fewest"], item.fewest)
    entry["most"] = max(entry["most"], item.most)
    _add_evidence(entry, item.patterns, item.excess_counts)
    if item.excessive > entry["worst"]["excessive"]:
        entry["worst"] = _page_opening(trace_id, item)
        # The call lines that inline one origin can differ, and the line names the worst.
        entry["line"], entry["source"] = item.line, item.source


def _add_evidence(
    entry: dict, patterns: Iterable[int] | None, excess_counts: Iterable[int]
) -> None:
    # A launch with too many patterns to keep leaves the line to the rule for one trace.
    if patterns is None:
        entry["patterns"] = None
    elif entry["patterns"] is not None:
        entry["patterns"].update(patterns)
    entry["excess_counts"].update(excess_counts)


def _empty_totals() -> dict[str, int]:
    return {"requests": 0, "achieved": 0, "ideal": 0, "excessive": 0}


def _add(total: dict, item: Instruction) -> None:
    for name in ("requests", "achieved", "ideal", "excessive"):
        total[name] += getattr(item, name)


def _page_opening(trace_id: str, item: Instruction) -> dict:
    """Return where the page opens an instruction: its trace, its PC, and its space."""
    return {"id": trace_id, "pc": f"{item.pc:#x}", "space": item.space, "excessive": item.excessive}
