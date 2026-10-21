import pathlib
import re
import shutil
import subprocess

import pytest

NODE = shutil.which("node")
SCRIPT_TESTS = sorted((pathlib.Path(__file__).parent / "gui").glob("*.test.mjs"))
# node --test runs test files since Node 18.
MINIMUM_NODE = 18


def node_major() -> int:
    if NODE is None:
        return 0
    version = subprocess.run([NODE, "--version"], capture_output=True, text=True).stdout
    found = re.match(r"v(\d+)", version)
    return int(found.group(1)) if found else 0


@pytest.mark.skipif(node_major() < MINIMUM_NODE, reason=f"node {MINIMUM_NODE} or later is needed")
def test_page_modules_pass_their_node_tests():
    assert SCRIPT_TESTS
    result = subprocess.run(
        [str(NODE), "--test", *map(str, SCRIPT_TESTS)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
