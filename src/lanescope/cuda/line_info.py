import collections
import pathlib
import re
import struct
import subprocess
from collections.abc import Collection, Sequence

SECTION = re.compile(r"^\.text\.(?P<name>\S+):\s*$")
LOCATION = re.compile(r'^\s*//## File "(?P<file>[^"]+)", line (?P<line>\d+)')
INSTRUCTION = re.compile(r"^\s*/\*(?P<offset>[0-9a-f]{4,})\*/")
OPCODE = re.compile(
    r"^\s*/\*(?P<offset>[0-9a-f]{4,})\*/\s+(?:@!?U?P\w+\s+)?(?P<opcode>[A-Z][\w.]*)"
)
# Instructions whose wavefronts follow rules of their own, beyond those of their width.
RULE_OPCODES = ("LDSM", "STSM", "LDGSTS", "ATOMS")
# Sources larger than this are no kernel files worth carrying in a trace.
MAX_SOURCE_BYTES = 1 << 20
# Line tables name any file the compiler read. Only absolute paths of source files
# are read, so an import never copies another file of the machine into a trace.
SOURCE_SUFFIXES = frozenset({".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".cxx", ".c"})
# The parts of a 64-bit little-endian ELF file that name the functions of a cubin.
ELF64_MAGIC = b"\x7fELF\x02"
ELF_SECTION = struct.Struct("<IIQQQQIIQQ")
ELF_SYMBOL = struct.Struct("<IBBHQQ")
SYMBOL_TABLE = 2
FUNCTION_SYMBOL = 2

FunctionLines = dict[int, tuple[str, int]]


def parse_line_listing(listing: str) -> dict[str, FunctionLines]:
    """Return the file and line of each instruction offset, for each function of a listing.

    An inlined instruction comes with its own line, then the lines of the calls that
    inlined it, ending with the call in the kernel. It keeps the innermost line that lies
    in the file of that last call.
    """
    return {
        name: {offset: _kept_line(chain) for offset, chain in chains.items()}
        for name, chains in _inline_chains(listing).items()
    }


def origins(listing: str) -> dict[str, FunctionLines]:
    """Return the innermost location of each inlined instruction, for each function of a listing.

    An instruction appears only where that location differs from the line that `parse_line_listing`
    keeps for it.
    """
    return {
        name: {
            offset: chain[0] for offset, chain in chains.items() if chain[0] != _kept_line(chain)
        }
        for name, chains in _inline_chains(listing).items()
    }


def _inline_chains(listing: str) -> dict[str, dict[int, tuple[tuple[str, int], ...]]]:
    functions: dict[str, dict[int, tuple[tuple[str, int], ...]]] = {}
    current: dict[int, tuple[tuple[str, int], ...]] | None = None
    chain: tuple[tuple[str, int], ...] = ()
    calls: list[tuple[str, int]] = []
    for text in listing.splitlines():
        if section := SECTION.match(text):
            current = functions.setdefault(section["name"], {})
            chain, calls = (), []
        elif found := LOCATION.match(text):
            calls.append((found["file"], int(found["line"])))
        elif current is not None and (code := INSTRUCTION.match(text)):
            if calls:
                chain, calls = tuple(calls), []
            if chain:
                current[int(code["offset"], 16)] = chain
    return functions


def _kept_line(chain: tuple[tuple[str, int], ...]) -> tuple[str, int]:
    return next(call for call in chain if call[0] == chain[-1][0])


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


def function_symbols(cubin: bytes) -> dict[str, int]:
    """Return the symbol index of each function of an ELF cubin, by name.

    A cubin that is no readable 64-bit ELF file gives no functions.
    """
    try:
        (table_offset,) = struct.unpack_from("<Q", cubin, 0x28)
        entry_size, count = struct.unpack_from("<HH", cubin, 0x3A)
        if cubin[:5] != ELF64_MAGIC or entry_size != ELF_SECTION.size:
            return {}
        sections = [
            ELF_SECTION.unpack_from(cubin, table_offset + index * entry_size)
            for index in range(count)
        ]
        found = {}
        for _, kind, _, _, offset, size, link, _, _, symbol_size in sections:
            if kind != SYMBOL_TABLE or symbol_size != ELF_SYMBOL.size:
                continue
            names = sections[link][4]
            for index in range(size // symbol_size):
                name, info, *_ = ELF_SYMBOL.unpack_from(cubin, offset + index * symbol_size)
                if info & 0xF == FUNCTION_SYMBOL:
                    start = names + name
                    found[cubin[start : cubin.index(b"\0", start)].decode()] = index
        return found
    except (struct.error, IndexError, ValueError, OverflowError):
        return {}


def module_listing(
    disassemblers: Sequence[pathlib.Path], cubin: pathlib.Path, kernels: Collection[str] = ()
) -> str:
    """Return the listing of a cubin, with line information, from the first nvdisasm that reads it.

    With the names of kernels, the listing holds these kernels alone, and nvdisasm reads a
    large module in a fraction of the time. A name that the cubin lacks brings back the
    whole listing. A newer nvdisasm drops older GPUs, so the next copy gets its turn when
    one fails. Raises ValueError when none reads the cubin.
    """
    symbols = function_symbols(cubin.read_bytes()) if kernels else {}
    only = []
    if kernels and all(name in symbols for name in kernels):
        only = ["--cuda-function-index", ",".join(str(symbols[name]) for name in sorted(kernels))]
    errors = []
    for nvdisasm in disassemblers:
        try:
            result = subprocess.run(
                [str(nvdisasm), "--print-line-info-inline", "--print-code", *only, str(cubin)],
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


def metadata(lines: FunctionLines, inlined: FunctionLines | None = None) -> dict:
    """Return the trace metadata of one function: its main source file, its text, and its lines.

    The main file is the one with the most instructions. An instruction inlined from
    another file carries the line of its call, so only an instruction that the line
    table places in another file keeps no line. The innermost locations of inlined
    instructions, from `origins`, come with their files and the text of their lines.
    """
    if not lines:
        return {}
    files = collections.Counter(file for file, _ in lines.values())
    main = files.most_common(1)[0][0]
    data: dict = {
        "source_file": main,
        "lines": {f"{offset:#x}": line for offset, (file, line) in lines.items() if file == main},
    }
    if (source := _read_source(main)) is not None:
        data["source"] = source
    if inlined:
        data |= _origin_metadata(inlined)
    return data


def _origin_metadata(inlined: FunctionLines) -> dict:
    names = sorted({file for file, _ in inlined.values()})
    index = {file: number for number, file in enumerate(names)}
    texts: list[dict[str, str]] = []
    for file in names:
        source = _read_source(file) or []
        wanted = {line for name, line in inlined.values() if name == file}
        texts.append({str(line): source[line - 1] for line in wanted if 0 < line <= len(source)})
    return {
        "origin_files": names,
        "origins": {
            f"{offset:#x}": [index[file], line] for offset, (file, line) in inlined.items()
        },
        "origin_text": texts,
    }


def _read_source(file: str) -> list[str] | None:
    path = pathlib.Path(file)
    if not path.is_absolute() or path.suffix not in SOURCE_SUFFIXES:
        return None
    try:
        if path.is_file() and path.stat().st_size <= MAX_SOURCE_BYTES:
            return path.read_text(errors="replace").splitlines()
    except OSError:
        pass
    return None
