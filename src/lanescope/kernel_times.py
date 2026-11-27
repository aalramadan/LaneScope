import dataclasses
import json
import os
import pathlib
import re
import statistics
import subprocess
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence

FORMAT = 1
RUN_FILE = "run.json"
TIMES_FILE = "kernel-times.json"
# The timer writes the times of each process to this path with its process id appended.
TIMES_VARIABLE = "LANESCOPE_KERNEL_TIMES"
INJECTION_VARIABLE = "CUDA_INJECTION64_PATH"
PROCESS_TIMES = "times"
PROCESS_FILE = re.compile(re.escape(PROCESS_TIMES) + r"\.\d+")
# A failed run quotes this many of the last lines of its error output.
QUOTED_LINES = 5


@dataclasses.dataclass(frozen=True)
class KernelTime:
    """The time of one kernel over the rounds of a timing."""

    name: str
    launches: int
    median_ns: float
    fastest_round_ns: int
    slowest_round_ns: int
    even_launches: bool


def write_run(
    folder: pathlib.Path, command: Sequence[str], cwd: pathlib.Path | None = None
) -> None:
    """Store the command of a folder and remove the kernel times of an earlier command."""
    folder.mkdir(parents=True, exist_ok=True)
    run = {"format": FORMAT, "command": list(command), "cwd": str(cwd or pathlib.Path.cwd())}
    (folder / RUN_FILE).write_text(json.dumps(run, indent=1) + "\n")
    (folder / TIMES_FILE).unlink(missing_ok=True)


def read_run(folder: pathlib.Path) -> tuple[list[str], pathlib.Path]:
    """Return the stored command of a folder and the folder it runs in."""
    path = folder / RUN_FILE
    if not path.is_file():
        raise ValueError(
            f"{folder} holds no {RUN_FILE}. Record the program again with lanescope record, "
            f"or give it after --, such as lanescope time {folder} -- ./my_app"
        )
    try:
        run = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is no stored command ({error})") from error
    command = run.get("command") if isinstance(run, dict) else None
    if (
        not isinstance(run, dict)
        or run.get("format") != FORMAT
        or not isinstance(command, list)
        or not command
        or not all(isinstance(part, str) for part in command)
        or not isinstance(run.get("cwd"), str)
    ):
        raise ValueError(f"{path} is no stored command of format {FORMAT}")
    return command, pathlib.Path(run["cwd"])


def time_folders(
    folders: Sequence[pathlib.Path],
    timer: pathlib.Path,
    rounds: int,
    report: Callable[[int, pathlib.Path, float], None] | None = None,
) -> list[dict]:
    """Run the stored command of each folder in turns, write the kernel times of each, and
    return them.

    Each round runs every folder once, in the given order. A drift of the GPU clock then
    hits every folder alike. A run that fails or leaves no times raises ValueError
    before any folder gets its times.
    """
    runs = [read_run(folder) for folder in folders]
    seconds: list[list[float]] = [[] for _ in folders]
    measured: list[list[dict[str, dict]]] = [[] for _ in folders]
    untimed: list[list[int]] = [[] for _ in folders]
    for round_index in range(rounds):
        for index, (folder, (command, cwd)) in enumerate(zip(folders, runs, strict=True)):
            wall, kernels, without = _run_once(folder, command, cwd, timer, round_index)
            seconds[index].append(wall)
            measured[index].append(kernels)
            untimed[index].append(without)
            if report is not None:
                report(round_index, folder, wall)
    results = []
    for folder, walls, per_round, without in zip(folders, seconds, measured, untimed, strict=True):
        result = {
            "format": FORMAT,
            "rounds": rounds,
            "wall_seconds": walls,
            "untimed_launches": without,
            "kernels": _by_kernel(per_round),
        }
        (folder / TIMES_FILE).write_text(json.dumps(result, indent=1) + "\n")
        results.append(result)
    return results


def read_times(folder: pathlib.Path) -> dict | None:
    """Return the kernel times that lanescope time wrote into a folder, or None without them."""
    path = folder / TIMES_FILE
    if not path.is_file():
        return None
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{path} holds no kernel times ({error})") from error
    if not _valid_times(result):
        raise ValueError(
            f"{path} holds no kernel times of format {FORMAT}. Time the folder again with "
            f"lanescope time {folder}"
        )
    return result


def round_totals(result: dict, names: Iterable[str] | None = None) -> list[int]:
    """Return the nanoseconds of each round of a timing, summed over the named kernels or all."""
    kernels = result["kernels"]
    picked = [kernels[name] for name in (kernels if names is None else names) if name in kernels]
    return [
        sum(kernel["nanoseconds"][index] for kernel in picked) for index in range(result["rounds"])
    ]


def ranked(result: dict) -> list[KernelTime]:
    """Return the kernels of a timing by their median time over the rounds, longest first."""
    medians = {
        name: statistics.median(kernel["nanoseconds"]) for name, kernel in result["kernels"].items()
    }
    found = [
        KernelTime(
            name=name,
            launches=round(statistics.median(kernel["launches"])),
            median_ns=medians[name],
            fastest_round_ns=min(kernel["nanoseconds"]),
            slowest_round_ns=max(kernel["nanoseconds"]),
            even_launches=len(set(kernel["launches"])) == 1,
        )
        for name, kernel in result["kernels"].items()
    ]
    return sorted(found, key=lambda kernel: (-kernel.median_ns, kernel.name))


def _run_once(
    folder: pathlib.Path,
    command: Sequence[str],
    cwd: pathlib.Path,
    timer: pathlib.Path,
    round_index: int,
) -> tuple[float, dict[str, dict], int]:
    with tempfile.TemporaryDirectory(prefix="lanescope-times-") as scratch:
        env = dict(os.environ)
        env[INJECTION_VARIABLE] = str(timer)
        env[TIMES_VARIABLE] = str(pathlib.Path(scratch) / PROCESS_TIMES)
        start = time.perf_counter()
        try:
            done = subprocess.run(
                list(command),
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                errors="replace",
                check=False,
            )
        except OSError as error:
            raise ValueError(f"the command of {folder} cannot start: {error}") from error
        wall = time.perf_counter() - start
        where = f"the command of {folder} in round {round_index + 1}"
        quoted = "\n".join(done.stderr.splitlines()[-QUOTED_LINES:])
        tail = f":\n{quoted}" if quoted else ""
        if done.returncode:
            # subprocess reports a program that a signal kills by the negative signal number.
            ending = (
                f"was killed by signal {-done.returncode}"
                if done.returncode < 0
                else f"exited with status {done.returncode}"
            )
            raise ValueError(f"{where} {ending}{tail}")
        kernels, untimed = _process_times(pathlib.Path(scratch))
        if not kernels:
            raise ValueError(
                f"{where} exited with status 0 and left no kernel times. It launched no CUDA "
                f"kernel, or CUPTI could not time its kernels{tail}"
            )
    return wall, kernels, untimed


def _process_times(scratch: pathlib.Path) -> tuple[dict[str, dict], int]:
    """Return the times of every process of one run by kernel name, and the launches that
    CUPTI gave no time."""
    kernels: dict[str, dict] = {}
    untimed = 0
    for path in sorted(scratch.iterdir()):
        if not PROCESS_FILE.fullmatch(path.name):
            continue
        try:
            written = json.loads(path.read_text())
            found = written["kernels"]
            untimed += int(written.get("skipped", 0)) + int(written.get("dropped", 0))
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(f"the kernel timer wrote an unreadable file ({error})") from error
        for name, times in found.items():
            if name not in kernels:
                kernels[name] = dict(times)
                continue
            kept = kernels[name]
            kept["launches"] += times["launches"]
            kept["nanoseconds"] += times["nanoseconds"]
            kept["shortest"] = min(kept["shortest"], times["shortest"])
            kept["longest"] = max(kept["longest"], times["longest"])
    return kernels, untimed


def _valid_times(result: object) -> bool:
    if not isinstance(result, dict) or result.get("format") != FORMAT:
        return False
    rounds, kernels = result.get("rounds"), result.get("kernels")
    if not isinstance(rounds, int) or rounds < 1 or not isinstance(kernels, dict):
        return False
    return all(
        isinstance(kernel, dict)
        and all(
            isinstance(kernel.get(name), list)
            and len(kernel[name]) == rounds
            and all(isinstance(value, int) and value >= 0 for value in kernel[name])
            for name in ("launches", "nanoseconds")
        )
        for kernel in kernels.values()
    )


def _by_kernel(per_round: Sequence[dict[str, dict]]) -> dict[str, dict]:
    names = sorted({name for kernels in per_round for name in kernels})
    result = {}
    for name in names:
        present = [kernels[name] for kernels in per_round if name in kernels]
        result[name] = {
            "launches": [kernels.get(name, {}).get("launches", 0) for kernels in per_round],
            "nanoseconds": [kernels.get(name, {}).get("nanoseconds", 0) for kernels in per_round],
            "shortest": min(times["shortest"] for times in present),
            "longest": max(times["longest"] for times in present),
        }
    return result
