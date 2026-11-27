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
    of the time before over the time after. A row adds up only the instances of its
    template that both sides hold, and the others join the kernels of one side alone. A
    template whose instances differ on each side, such as after a change of a template
    argument, adds up all of them. A template with several shared instances also compares
    each of them. A total row adds up every kernel of each side.
    """
    first = {entry["template"]: entry for entry in before["kernels"]}
    second = {entry["template"]: entry for entry in after["kernels"]}
    rows = []
    only_before = [_unmatched(entry) for name, entry in first.items() if name not in second]
    only_after = [_unmatched(entry) for name, entry in second.items() if name not in first]
    for template, entry in first.items():
        if template not in second:
            continue
        other = second[template]
        names = {instance["kernel"] for instance in entry["instances"]}
        shared = names & {instance["kernel"] for instance in other["instances"]}
        if not shared or len(shared) == len(names) == len(other["instances"]):
            matched = "all" if shared else "none"
            row = _kernel_row(entry, other)
        else:
            matched = "some"
            row = _kernel_row(_shared_part(entry, shared), _shared_part(other, shared))
            only_before += _unshared_instances(entry, shared)
            only_after += _unshared_instances(other, shared)
        rows.append(
            {"template": template}
            | row
            | {"matched_instances": matched, "instances": _instances(entry, other, shared)}
        )
    rows.sort(key=_row_order)
    return {
        "timed": {"before": _timed_rounds(before), "after": _timed_rounds(after)},
        "kernels": rows,
        "only_before": only_before,
        "only_after": only_after,
        "total": _total_row(before, after),
        "unreadable_times": before["unreadable_times"] + after["unreadable_times"],
    }


def _instances(before: dict, after: dict, shared: set[str]) -> list[dict]:
    """Return the rows of the instances of a template that both sides share, when several."""
    if len(shared) < 2:
        return []
    second = {instance["kernel"]: instance for instance in after["instances"]}
    return [
        {"kernel": instance["kernel"]} | _kernel_row(instance, second[instance["kernel"]])
        for instance in before["instances"]
        if instance["kernel"] in shared
    ]


def _shared_part(entry: dict, shared: set[str]) -> dict:
    """Return a template entry that adds up only its instances of the shared names."""
    instances = [instance for instance in entry["instances"] if instance["kernel"] in shared]
    if len(instances) == 1:
        name, full_name = instances[0]["name"], instances[0]["full_name"]
    else:
        name, full_name = kernel_names.short_name(entry["template"]), entry["template"]
    return {
        "name": name,
        "full_name": full_name,
        "launches": sum(instance["launches"] for instance in instances),
        "totals": {
            unit: {name: sum(item["totals"][unit][name] for item in instances) for name in COUNTS}
            for unit in summary.UNITS
        },
        **summary.template_time(instances),
    }


def _unshared_instances(entry: dict, shared: set[str]) -> list[dict]:
    return [
        _unmatched(instance, entry["template"])
        for instance in entry["instances"]
        if instance["kernel"] not in shared
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


def _unmatched(entry: dict, template: str | None = None) -> dict:
    """Return a template of one side alone, or an instance of `template` that one side holds."""
    return {
        "template": entry["template"] if template is None else template,
        "kernel": None if template is None else entry["kernel"],
        "name": entry["name"],
        "full_name": entry["full_name"],
    } | _side_summary(entry)


def _time_change(before: dict, after: dict) -> dict:
    """Return the speedup of a row, and whether the difference lies within the spread.

    A difference counts as a change only when the ranges of the rounds lie apart by at
    least the larger spread of the two sides, its slowest round minus its fastest.
    """
    if before["time_ns"] is None or after["time_ns"] is None:
        return {"speedup": None, "within_spread": None}
    (low, high), (other_low, other_high) = before["time_range_ns"], after["time_range_ns"]
    gap = max(other_low - high, low - other_high)
    return {
        "speedup": before["time_ns"] / after["time_ns"] if after["time_ns"] else None,
        "within_spread": gap <= 0 or gap < max(high - low, other_high - other_low),
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
