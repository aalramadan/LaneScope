from lanescope.fixes.layouts import Buffer, Change, Padding, XorSwizzle

# Example types of each element size, named in the comments of the code.
ELEMENT_TYPES = {1: "unsigned char", 2: "__half", 4: "float", 8: "double", 16: "float4"}


def forms(buffer: Buffer, change: Change, element_bytes: int) -> list[dict]:
    """Return the labeled C++ forms of a change for a buffer of elements of a given size."""
    if element_bytes not in ELEMENT_TYPES:
        raise ValueError(f"elements must have one of {tuple(ELEMENT_TYPES)} bytes")
    change.check(buffer)
    if isinstance(change, Padding):
        return _padding_forms(buffer, change, element_bytes)
    return _swizzle_forms(buffer, change, element_bytes)


def _padding_forms(buffer: Buffer, change: Padding, element_bytes: int) -> list[dict]:
    row, pad = change.row_bytes, change.pad_bytes
    found = [_form("Byte offset o inside the buffer", f"o + (o / {row}) * {pad}")]
    if row % element_bytes or pad % element_bytes:
        return found
    columns, extra = row // element_bytes, pad // element_bytes
    found.append(
        _form(f"Index i of {element_bytes}-byte elements", f"i + (i / {columns}) * {extra}")
    )
    rows = buffer.size // row
    found.append(
        _form(
            "Declaration",
            f"// T is a {element_bytes}-byte type, such as {ELEMENT_TYPES[element_bytes]}.\n"
            f"__shared__ T tile[{rows}][{columns} + {extra}];",
        )
    )
    return found


def _swizzle_forms(buffer: Buffer, change: XorSwizzle, element_bytes: int) -> list[dict]:
    mask = ((1 << change.bits) - 1) << change.base
    found = [
        _form(
            "Byte offset o inside the buffer",
            f"o ^ ((o >> {change.shift}) & {mask:#x})",
        )
    ]
    # Elements of 2**e bytes see the same swizzle with a base lower by e. A lower
    # base than e would move part of an element, which no element form can express.
    exponent = element_bytes.bit_length() - 1
    base = change.base - exponent
    if base < 0:
        return found
    element_mask = ((1 << change.bits) - 1) << base
    found.append(
        _form(
            f"Index i of {element_bytes}-byte elements",
            f"i ^ ((i >> {change.shift}) & {element_mask:#x})",
        )
    )
    found.append(_form("CuTe", f"Swizzle<{change.bits}, {base}, {change.shift}>"))
    # With rows of 2**shift elements, the shifted index is the row index r, and the
    # XOR stays inside the column bits when base + bits fits in the shift.
    columns = 1 << change.shift
    row_bytes = columns * element_bytes
    if base + change.bits <= change.shift and buffer.size % row_bytes == 0:
        found.append(
            _form(
                f"Rows of {columns} elements",
                f"// T is a {element_bytes}-byte type, such as {ELEMENT_TYPES[element_bytes]}.\n"
                f"__shared__ T tile[{buffer.size // row_bytes}][{columns}];\n"
                f"tile[r][c ^ (r & {element_mask:#x})]",
            )
        )
    return found


def _form(label: str, code: str) -> dict:
    return {"label": label, "code": code}
