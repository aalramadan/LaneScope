import pathlib

from lanescope import kernel_names, summary

# The counts of each unit that a row of a comparison gives for each side.
COUNTS = ("achieved", "ideal", "excessive")


def read_side(path: pathlib.Path, processes: bool = False) -> dict:
    """Return the summary of one side of a comparison, a recording folder or a trace file.

    Every trace of the side adds up as one run, which takes its kernel times from the
    folder, or from beside the trace file.
    """
    if not path.exists():
        raise ValueError(f"no such file or folder: {path}")
    paths = summary.trace_files(path, summary.MAX_TRACES)
    if not paths:
        raise ValueError(f"found no trace files in {path}")
    facts = summary.read_many(paths, processes=processes)
    return summary.summarize_files(paths, facts, run=path)


def times_folder(path: pathlib.Path) -> pathlib.Path:
    """Return the folder whose kernel times belong to a side: the folder, or that of the file."""
    return path if path.is_dir() else path.parent


def compare_summaries(before: dict, after: dict) -> dict:
    """Return two summaries of one run each, compared by kernel template.

    Each row gives the launches, the counts, and the time of each side, with the speedup
    of the time before over the time after. A template with several instances also
    compares each instance whose name both sides share. Templates of one side alone are
    listed apart, and a total row adds up every kernel of each side.
    """
    first = {entry["template"]: entry for entry in before["kernels"]}
    second = {entry["template"]: entry for entry in after["kernels"]}
    rows = []
    for template, entry in first.items():
        if template in second:
            row = {"template": template} | _kernel_row(entry, second[template])
            rows.append(row | {"instances": _instances(entry, second[template])})
    rows.sort(key=_row_order)
    return {
        "timed": {"before": _timed_rounds(before), "after": _timed_rounds(after)},
        "kernels": rows,
        "only_before": [
            _unmatched_kernel(entry) for name, entry in first.items() if name not in second
        ],
        "only_after": [
            _unmatched_kernel(entry) for name, entry in second.items() if name not in first
        ],
        "total": _total_row(before, after),
    }


def _instances(before: dict, after: dict) -> list[dict]:
    """Return the rows of the instances of a template that both sides share by name."""
    if len(before["instances"]) < 2 and len(after["instances"]) < 2:
        return []
    second = {instance["kernel"]: instance for instance in after["instances"]}
    return [
        {"kernel": instance["kernel"]} | _kernel_row(instance, second[instance["kernel"]])
        for instance in before["instances"]
        if instance["kernel"] in second
    ]


def _kernel_row(before: dict, after: dict) -> dict:
    """Return the row of one template or one instance, named as both sides name it."""
    first, second = _side_summary(before), _side_summary(after)
    same = before["full_name"] == after["full_name"]
    template = kernel_names.template_name(before["full_name"])
    return {
        "name": before["name"] if same else kernel_names.short_name(template),
        "full_name": before["full_name"] if same else template,
        "before": first,
        "after": second,
        "time": _time_change(first, second),
    }


def _side_summary(entry: dict) -> dict:
    """Return what a row gives of one side: the launches, the counts, and the time."""
    rounds = entry["round_ns"]
    return {
        "launches": entry["launches"],
        **{unit: {name: entry["totals"][unit][name] for name in COUNTS} for unit in summary.UNITS},
        "time_ns": entry["time_ns"],
        "time_range_ns": [min(rounds), max(rounds)] if rounds else None,
    }


def _unmatched_kernel(entry: dict) -> dict:
    return {
        "template": entry["template"],
        "name": entry["name"],
        "full_name": entry["full_name"],
    } | _side_summary(entry)


def _time_change(before: dict, after: dict) -> dict:
    """Return the speedup of a row, and whether its ranges of rounds overlap.

    Rounds whose ranges overlap leave the difference within their spread, which makes no
    speedup claim.
    """
    if before["time_ns"] is None or after["time_ns"] is None:
        return {"speedup": None, "within_spread": None}
    (low, high), (other_low, other_high) = before["time_range_ns"], after["time_range_ns"]
    return {
        "speedup": before["time_ns"] / after["time_ns"] if after["time_ns"] else None,
        "within_spread": low <= other_high and other_low <= high,
    }


def _total_row(before: dict, after: dict) -> dict:
    sides = {}
    for name, data in (("before", before), ("after", after)):
        timed = data["times"][0] if data["times"] else None
        rounds = timed["round_ns"] if timed else None
        sides[name] = {
            "launches": data["traces"],
            **{unit: {key: data["totals"][unit][key] for key in COUNTS} for unit in summary.UNITS},
            "time_ns": timed["total_ns"] if timed else None,
            "time_range_ns": [min(rounds), max(rounds)] if rounds else None,
        }
    return sides | {"time": _time_change(sides["before"], sides["after"])}


def _timed_rounds(data: dict) -> int | None:
    return data["times"][0]["rounds"] if data["times"] else None


def _row_order(row: dict) -> tuple:
    return (
        -row["before"]["wavefronts"]["excessive"],
        -row["before"]["sectors"]["excessive"],
        -(row["before"]["time_ns"] or 0),
        row["name"],
    )
