import re
import shutil
import subprocess
from collections.abc import Iterable

# Names this long fit one row of a table beside their counts.
SHORT_LENGTH = 64
ELLIPSIS = "…"
# Template arguments such as 128, true, or GGML_TYPE_F16 stay in a short name.
_PLAIN_ARGUMENT = re.compile(r"[\w.+-]{1,24}")
_OPENING = "<({["
_CLOSING = ">)}]"
_readable: dict[str, str] = {}


def readable(names: Iterable[str]) -> dict[str, str]:
    """Return the C++ name of each mangled kernel name, and each other name unchanged."""
    names = list(dict.fromkeys(names))
    missing = [name for name in names if name.startswith("_Z") and name not in _readable]
    tool = shutil.which("c++filt")
    if missing and tool:
        try:
            done = subprocess.run(
                [tool], input="\n".join(missing), capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError):
            done = None
        lines = done.stdout.splitlines() if done and done.returncode == 0 else []
        if len(lines) == len(missing):
            _readable.update(zip(missing, lines, strict=True))
    return {name: _readable.get(name, name) for name in names}


def short(name: str, length: int = SHORT_LENGTH) -> str:
    """Return a kernel name without its namespaces, return type, and parameters.

    Template arguments that are no plain number or word become an ellipsis once the
    name passes the length, and so does the end of a name that still passes it.
    """
    text = _last_part(_without_return_type(_without_parameters(name.strip())))
    if len(text) <= length:
        return text
    opening = _depth_zero(text, "<")
    if opening and text.endswith(">"):
        start = opening[0]
        arguments: list[str] = []
        for argument in _split(text[start + 1 : -1], ","):
            kept = argument if _PLAIN_ARGUMENT.fullmatch(argument) else ELLIPSIS
            # A run of cut arguments shows as one ellipsis.
            if kept != ELLIPSIS or arguments[-1:] != [ELLIPSIS]:
                arguments.append(kept)
        text = f"{text[:start]}<{', '.join(arguments)}>"
        if len(text) > length:
            text = f"{text[:start]}<{ELLIPSIS}>"
    return text if len(text) <= length else text[: length - 1] + ELLIPSIS


def _without_parameters(text: str) -> str:
    """Return a demangled name without the parameter list at its end."""
    if not text.endswith(")"):
        return text
    depth = 0
    for index in range(len(text) - 1, -1, -1):
        if text[index] == ")":
            depth += 1
        elif text[index] == "(":
            depth -= 1
            if depth == 0:
                return text[:index] if index else text
    return text


def _without_return_type(text: str) -> str:
    """Return a demangled name without the return type that templates put before it."""
    spaces = _depth_zero(text, " ")
    return text[spaces[-1] + 1 :] if spaces else text


def _last_part(text: str) -> str:
    """Return the last part of a qualified name, after its namespaces and classes."""
    colons = [index for index in _depth_zero(text, ":") if text[index : index + 2] == "::"]
    return text[colons[-1] + 2 :] if colons else text


def _split(text: str, separator: str) -> list[str]:
    """Return the parts of a list of template arguments, split at its own separators."""
    cuts = _depth_zero(text, separator)
    starts = [0, *(cut + 1 for cut in cuts)]
    ends = [*cuts, len(text)]
    return [text[start:end].strip() for start, end in zip(starts, ends, strict=True)]


def _depth_zero(text: str, character: str) -> list[int]:
    """Return the indices of a character outside every pair of brackets of a name."""
    found = []
    depth = 0
    for index, current in enumerate(text):
        if current == character and depth == 0:
            found.append(index)
        if current in _OPENING:
            depth += 1
        elif current in _CLOSING:
            depth = max(depth - 1, 0)
    return found
