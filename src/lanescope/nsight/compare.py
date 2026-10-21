import dataclasses
import json
import pathlib
import re
from collections.abc import Iterable

from lanescope.model.counters import count
from lanescope.trace import Space, Trace

try:
    import ncu_report
except ImportError:  # The report reader is an optional dependency of the GPU checks.
    ncu_report = None

QUANTITIES = (
    "shared_wavefronts",
    "shared_wavefronts_ideal",
    "global_sectors",
    "global_sectors_ideal",
)
KNOWN_NAMES = {
    "shared_wavefronts": (
        "memory_l1_wavefronts_shared",
        "smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared",
    ),
    "shared_wavefronts_ideal": (
        "memory_l1_wavefronts_shared_ideal",
        "smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_ideal",
    ),
    "global_sectors": ("memory_l2_theoretical_sectors_global", "smsp__sass_sectors_mem_global"),
    "global_sectors_ideal": (
        "memory_l2_theoretical_sectors_global_ideal",
        "smsp__sass_sectors_mem_global_ideal",
    ),
}
# Words a metric name must hold and words it must not hold, for names not listed above.
PATTERNS = {
    "shared_wavefronts": (("wavefronts", "shared"), ("ideal", "excessive", "pct", "ratio", "per_")),
    "shared_wavefronts_ideal": (("wavefronts", "shared", "ideal"), ("excessive", "pct", "ratio")),
    "global_sectors": (
        ("sectors", "global"),
        ("ideal", "excessive", "pct", "ratio", "per_", "local"),
    ),
    "global_sectors_ideal": (
        ("sectors", "global", "ideal"),
        ("excessive", "pct", "ratio", "local"),
    ),
}


@dataclasses.dataclass(frozen=True)
class KernelCounters:
    """Source counters of one kernel launch, keyed by instruction offset."""

    kernel: str
    metrics: dict[str, dict[int, float]]
    sass: dict[int, str]

    def quantity(self, name: str) -> tuple[str | None, dict[int, float]]:
        """Return the metric used for a quantity and its value per instruction offset."""
        for metric in KNOWN_NAMES[name]:
            if metric in self.metrics:
                return metric, self.metrics[metric]
        required, forbidden = PATTERNS[name]
        matches = sorted(
            (metric for metric in self.metrics if _matches(metric, required, forbidden)), key=len
        )
        if not matches:
            return None, {}
        base = matches[0]
        # Names such as <name>_op_ld and <name>_op_st split one count by direction.
        # Without the aggregate name, their sum gives the count of every instruction.
        split = [metric for metric in matches if re.search(r"_op_(ld|st)$", metric)]
        if split and not all(metric.startswith(base) for metric in matches):
            total: dict[int, float] = {}
            for metric in split:
                for offset, value in self.metrics[metric].items():
                    total[offset] = total.get(offset, 0.0) + value
            return " + ".join(split), total
        return base, self.metrics[base]


@dataclasses.dataclass(frozen=True)
class Row:
    """One quantity of one instruction, from LaneScope and from Nsight Compute."""

    launch: int
    kernel: str
    offset: int
    sass: str
    quantity: str
    lanescope: int
    nsight: float | None

    @property
    def matches(self) -> bool:
        """Return whether Nsight Compute reports exactly the LaneScope count."""
        return self.nsight is not None and round(self.nsight) == self.lanescope

    @property
    def opcode(self) -> str:
        """Return the SASS opcode of the instruction, such as LDS.128."""
        return opcode(self.sass) or "?"


def opcode(sass: str) -> str:
    """Return the opcode of a SASS line without its predicate, or an empty string."""
    words = [word for word in sass.split() if not word.startswith("@")]
    return words[0] if words else ""


def read_report(path: pathlib.Path) -> list[KernelCounters]:
    """Return the source counters of every kernel launch in an Nsight Compute report."""
    if ncu_report is None:
        raise ImportError("reading Nsight Compute reports needs pip install ncu-report")
    # The package defines load_report twice, and the type checker reads the wrong one.
    report = ncu_report.load_report(str(path))  # pyright: ignore[reportCallIssue]
    kernels = []
    for range_index in range(report.num_ranges()):
        launches = report.range_by_idx(range_index)
        for action_index in range(launches.num_actions()):
            kernels.append(_kernel_counters(launches.action_by_idx(action_index)))
    return kernels


def lanescope_counts(trace: Trace) -> dict[int, dict[str, int]]:
    """Return the LaneScope quantities of each instruction offset of a trace."""
    found: dict[int, dict[str, int]] = {}
    for counter in count(trace):
        prefix = "shared_wavefronts" if counter.space == Space.SHARED else "global_sectors"
        entry = found.setdefault(counter.pc, {})
        entry[prefix] = entry.get(prefix, 0) + counter.achieved
        entry[f"{prefix}_ideal"] = entry.get(f"{prefix}_ideal", 0) + counter.ideal
    return found


def compare(launch: int, trace: Trace, counters: KernelCounters) -> list[Row]:
    """Return one row per quantity of each instruction that the trace holds."""
    ours = lanescope_counts(trace)
    aligned = align(sorted(ours), counters)
    rows = []
    for offset, quantities in sorted(ours.items()):
        theirs_offset = aligned[offset]
        for quantity, value in quantities.items():
            rows.append(
                Row(
                    launch,
                    counters.kernel,
                    offset,
                    counters.sass.get(theirs_offset, ""),
                    quantity,
                    value,
                    counters.quantity(quantity)[1].get(theirs_offset),
                )
            )
    return rows


def align(offsets: list[int], counters: KernelCounters) -> dict[int, int]:
    """Map LaneScope instruction offsets to Nsight Compute offsets.

    Equal offsets match directly. When the recorder reports PCs on another base,
    the memory instructions of the two sides still come in the same order, so equal
    counts pair them by rank.
    """
    theirs = sorted({o for quantity in QUANTITIES for o in counters.quantity(quantity)[1]})
    if all(offset in theirs for offset in offsets) or len(offsets) != len(theirs):
        return {offset: offset for offset in offsets}
    return dict(zip(offsets, theirs, strict=True))


def summarize(rows: Iterable[Row]) -> dict[str, dict[str, int]]:
    """Return the matched and total rows per opcode and quantity."""
    summary: dict[str, dict[str, int]] = {}
    for row in rows:
        entry = summary.setdefault(f"{row.opcode} {row.quantity}", {"matched": 0, "total": 0})
        entry["total"] += 1
        entry["matched"] += row.matches
    return dict(sorted(summary.items()))


def dump(kernels: list[KernelCounters], path: pathlib.Path) -> None:
    """Write every source counter of every launch as JSON."""
    payload = [
        {
            "kernel": kernel.kernel,
            "sass": {f"{offset:#x}": text for offset, text in kernel.sass.items()},
            "metrics": {
                name: {f"{offset:#x}": value for offset, value in values.items()}
                for name, values in kernel.metrics.items()
            },
        }
        for kernel in kernels
    ]
    path.write_text(json.dumps(payload, indent=1))


def _kernel_counters(action) -> KernelCounters:
    start = _kernel_start(action)
    metrics: dict[str, dict[int, float]] = {}
    for name in action.metric_names():
        metric = action.metric_by_name(name)
        if not metric.has_correlation_ids():
            continue
        ids = metric.correlation_ids()
        values = {ids.as_uint64(i): metric.as_double(i) for i in range(metric.num_instances())}
        # Occupancy tables and other facts of the launch number their instances from 0,
        # and some reports map these small IDs to the SASS of another kernel.
        if start is not None:
            values = {pc: value for pc, value in values.items() if pc >= start}
        # Some instanced metrics carry timestamps as IDs, and those map to no SASS.
        if values and action.sass_by_pc(next(iter(values))) not in ("", "N/A"):
            metrics[name] = values
    pcs = sorted({pc for values in metrics.values() for pc in values})
    if start is None:
        start = pcs[0] if pcs else 0
    return KernelCounters(
        kernel=action.name(),
        metrics={
            name: {pc - start: v for pc, v in values.items()} for name, values in metrics.items()
        },
        sass={pc - start: action.sass_by_pc(pc) for pc in pcs},
    )


def _kernel_start(action) -> int | None:
    # The address of the kernel, from which the recorder counts its offsets as well.
    metric = action.metric_by_name("launch__function_pcs")
    if metric is None or not metric.num_instances():
        return None
    return min(metric.as_uint64(i) for i in range(metric.num_instances()))


def _matches(metric: str, required: tuple[str, ...], forbidden: tuple[str, ...]) -> bool:
    words = set(re.split(r"[_.]+", metric.lower()))
    return all(word in words for word in required) and not any(
        word in metric.lower() for word in forbidden
    )
