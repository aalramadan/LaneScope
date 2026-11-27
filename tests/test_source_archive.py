import fnmatch
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _in_manifest(path: pathlib.Path, lines: list[list[str]]) -> bool:
    relative = path.relative_to(ROOT)
    for command, *arguments in lines:
        if command == "include" and relative.as_posix() in arguments:
            return True
        if command == "recursive-include":
            folder, *patterns = arguments
            if relative.is_relative_to(folder) and any(
                fnmatch.fnmatch(path.name, pattern) for pattern in patterns
            ):
                return True
    return False


def test_the_source_archive_takes_every_file_of_the_test_data():
    text = (ROOT / "MANIFEST.in").read_text()
    lines = [line.split() for line in text.splitlines() if line.strip()]
    data = [path for path in (ROOT / "tests" / "data").rglob("*") if path.is_file()]
    assert data
    assert [path for path in data if not _in_manifest(path, lines)] == []
