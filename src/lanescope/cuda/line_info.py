import collections
import pathlib
import re
import subprocess
from collections.abc import Sequence

SECTION = re.compile(r"^\.text\.(?P<name>\S+):\s*$")
LOCATION = re.compile(r'^\s*//## File "(?P<file>[^"]+)", line (?P<line>\d+)')
INSTRUCTION = re.compile(r"^\s*/\*(?P<offset>[0-9a-f]{4,})\*/")
OPCODE = re.compile(
    r"^\s*/\*(?P<offset>[0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?(?P<opcode>[A-Z][\w.]*)"
)
# Instructions whose wavefronts follow rules of their own, beyond those of their width.
RULE_OPCODES = ("LDSM", "STSM", "LDGSTS")
# Sources larger than this are no kernel files worth carrying in a trace.
MAX_SOURCE_BYTES = 1 << 20
# Line tables name any file the compiler read. Only absolute paths of source files
# are read, so an import never copies another file of the machine into a trace.
SOURCE_SUFFIXES = frozenset({".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".c"})

FunctionLines = dict[int, tuple[str, int]]


def parse(listing: str) -> dict[str, FunctionLines]:
    """Return the file and line of each instruction offset, for each function of a listing.

    An inlined instruction comes with its own line, then the lines of the calls that
    inlined it, ending with the call in the kernel. It keeps the innermost line that lies
    in the file of that last call.
    """
    functions: dict[str, FunctionLines] = {}
    current: FunctionLines | None = None
    location: tuple[str, int] | None = None
    calls: list[tuple[str, int]] = []
    for text in listing.splitlines():
        if section := SECTION.match(text):
            current = functions.setdefault(section["name"], {})
            location, calls = None, []
        elif found := LOCATION.match(text):
            calls.append((found["file"], int(found["line"])))
        elif current is not None and (code := INSTRUCTION.match(text)):
            if calls:
                location = next(call for call in calls if call[0] == calls[-1][0])
                calls = []
            if location is not None:
                current[int(code["offset"], 16)] = location
    return functions


def opcodes(listing: str) -> dict[str, dict[int, str]]:
    """Return the opcode of each instruction with a counting rule of its own, for each function."""
    functions: dict[str, dict[int, str]] = {}
    current: dict[int, str] | None = None
    for text in listing.splitlines():
        if section := SECTION.match(text):
            current = functions.setdefault(section["name"], {})
        elif current is not None and (code := OPCODE.match(text)):
            if code["opcode"].split(".")[0] in RULE_OPCODES:
                current[int(code["offset"], 16)] = code["opcode"]
    return functions


def module_listing(disassemblers: Sequence[pathlib.Path], cubin: pathlib.Path) -> str:
    """Return the listing of a cubin, with line information, from the first nvdisasm that reads it.

    A newer nvdisasm drops older GPUs, so the next copy gets its turn when one fails.
    Raises ValueError when none reads the cubin.
    """
    errors = []
    for nvdisasm in disassemblers:
        try:
            result = subprocess.run(
                [str(nvdisasm), "--print-line-info-inline", "--print-code", str(cubin)],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as error:
            errors.append(f"{nvdisasm} cannot start ({error.strerror})")
            continue
        if result.returncode == 0:
            return result.stdout
        errors.append(result.stderr.strip() or f"{nvdisasm} failed")
    raise ValueError(f"nvdisasm cannot read {cubin}: {'; '.join(errors) or 'no nvdisasm'}")


def metadata(lines: FunctionLines) -> dict:
    """Return the trace metadata of one function: its main source file, its text, and its lines.

    The main file is the one with the most instructions. An instruction inlined from
    another file carries the line of its call, so only an instruction that the line
    table places in another file keeps no line.
    """
    if not lines:
        return {}
    files = collections.Counter(file for file, _ in lines.values())
    main = files.most_common(1)[0][0]
    data: dict = {
        "source_file": main,
        "lines": {f"{offset:#x}": line for offset, (file, line) in lines.items() if file == main},
    }
    path = pathlib.Path(main)
    if not path.is_absolute() or path.suffix not in SOURCE_SUFFIXES:
        return data
    try:
        if path.is_file() and path.stat().st_size <= MAX_SOURCE_BYTES:
            data["source"] = path.read_text(errors="replace").splitlines()
    except OSError:
        pass
    return data
