import re

import pytest

from lanescope.fixes import codegen
from lanescope.fixes.layouts import Buffer, Padding, XorSwizzle

TILE = Buffer(0, 4096)


def _codes(found):
    return {form["label"]: form["code"] for form in found}


def test_padding_of_a_float_tile_gives_one_extra_column():
    codes = _codes(codegen.forms(TILE, Padding(128, 4), 4))
    assert codes["Byte offset o inside the buffer"] == "o + (o / 128) * 4"
    assert codes["Index i of 4-byte elements"] == "i + (i / 32) * 1"
    assert codes["Declaration"].endswith("__shared__ T tile[32][32 + 1];")


def test_padding_that_splits_elements_keeps_only_the_byte_form():
    found = codegen.forms(TILE, Padding(128, 4), 8)
    assert [form["label"] for form in found] == ["Byte offset o inside the buffer"]


def test_swizzle_of_a_float_tile_matches_the_textbook_column_xor():
    codes = _codes(codegen.forms(TILE, XorSwizzle(5, 2, 5), 4))
    assert codes["Byte offset o inside the buffer"] == "o ^ ((o >> 5) & 0x7c)"
    assert codes["Index i of 4-byte elements"] == "i ^ ((i >> 5) & 0x1f)"
    assert codes["CuTe"] == "Swizzle<5, 0, 5>"
    assert codes["Rows of 32 elements"].endswith("tile[r][c ^ (r & 0x1f)]")


def test_element_and_byte_forms_of_a_swizzle_move_every_offset_alike():
    change = XorSwizzle(3, 4, 7)
    for element in (1, 2, 4, 8, 16):
        codes = _codes(codegen.forms(Buffer(0, 1 << 12), change, element))
        byte_shift, byte_mask = _shift_and_mask(codes["Byte offset o inside the buffer"])
        index_shift, index_mask = _shift_and_mask(codes[f"Index i of {element}-byte elements"])
        for index in range(0, (1 << 12) // element, 7):
            offset = index * element
            moved = index ^ ((index >> index_shift) & index_mask)
            assert moved * element == offset ^ ((offset >> byte_shift) & byte_mask)


def _shift_and_mask(code):
    found = re.search(r">> (\d+)\) & (0x[0-9a-f]+)\)", code)
    assert found, code
    return int(found.group(1)), int(found.group(2), 16)


def test_swizzle_below_the_element_size_has_no_element_form():
    found = codegen.forms(TILE, XorSwizzle(2, 2, 6), 16)
    assert [form["label"] for form in found] == ["Byte offset o inside the buffer"]


def test_forms_reject_an_unknown_element_size_and_a_change_that_does_not_fit():
    with pytest.raises(ValueError, match="elements must have"):
        codegen.forms(TILE, Padding(128, 4), 3)
    with pytest.raises(ValueError, match="no whole number"):
        codegen.forms(TILE, Padding(100, 4), 4)
