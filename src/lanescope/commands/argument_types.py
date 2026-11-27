import argparse
import math

from lanescope.fixes import layouts
from lanescope.nsight import calibration

# The recorder keeps from 1 to 2**26 records per launch and ignores any other capacity.
MAX_CAPACITY = 1 << 26


# argparse names the parser function in its message for a ValueError and drops the
# reason, so each parser raises ArgumentTypeError with the reason instead.
def prefixed_integer(text: str) -> int:
    """Parse an integer in decimal or with a prefix such as 0x."""
    try:
        return int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number such as 0x120, got {text!r}") from None


def cuda_version(text: str) -> tuple[int, int]:
    """Parse a CUDA version such as 12.9 into its major and minor parts."""
    parts = text.split(".")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError("expected a CUDA version such as 12.9")
    return int(parts[0]), int(parts[1])


def port(text: str) -> int:
    """Parse a TCP port from 0 to 65535."""
    return _bounded_integer(text, 0, 65535, "expected a port from 0 to 65535")


def nonnegative_count(text: str) -> int:
    """Parse a count of zero or more."""
    return _bounded_integer(text, 0, math.inf, "expected a count of zero or more")


def positive_count(text: str) -> int:
    """Parse a count of one or more."""
    return _bounded_integer(text, 1, math.inf, "expected a count of one or more")


def matmul_size(text: str) -> int:
    """Parse a positive multiple of the matmul tile."""
    value = _bounded_integer(text, 1, math.inf, "expected a positive multiple of 64")
    if value % calibration.MATMUL_TILE:
        raise argparse.ArgumentTypeError("expected a positive multiple of 64")
    return value


def capacity(text: str) -> int:
    """Parse the number of records that the recorder keeps per launch."""
    return _bounded_integer(
        text, 1, MAX_CAPACITY, f"expected a count from 1 to {MAX_CAPACITY} (2^26)"
    )


def buffer(text: str) -> layouts.Buffer:
    """Parse START:SIZE into a shared buffer."""
    start, length = _integer_pair(text, ":")
    try:
        return layouts.Buffer(start, length)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def ctas(text: str) -> tuple[int, int]:
    """Parse FIRST:COUNT into the CTAs to record."""
    first, amount = _integer_pair(text, ":")
    if first < 0 or amount <= 0:
        raise argparse.ArgumentTypeError("expected FIRST:COUNT with a positive count")
    return first, amount


def padding(text: str) -> layouts.Padding:
    """Parse ROW_BYTES:PAD_BYTES into a padding."""
    row_bytes, pad_bytes = _integer_pair(text, ":")
    return layouts.Padding(row_bytes, pad_bytes)


def swizzle(text: str) -> layouts.XorSwizzle:
    """Parse BITS,BASE,SHIFT into an XOR swizzle."""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("a swizzle needs BITS,BASE,SHIFT")
    bits, base, shift = (prefixed_integer(part) for part in parts)
    return layouts.XorSwizzle(bits, base, shift)


def _bounded_integer(text: str, low: int, high: float, reason: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(reason) from None
    if not low <= value <= high:
        raise argparse.ArgumentTypeError(reason)
    return value


def _integer_pair(text: str, separator: str) -> tuple[int, int]:
    parts = text.split(separator)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected two numbers separated by {separator!r}")
    return prefixed_integer(parts[0]), prefixed_integer(parts[1])
