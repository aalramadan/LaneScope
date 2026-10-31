import argparse
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
GUI = ROOT / "src" / "lanescope" / "gui"
ASSETS = ("api.js", "app.js", "format.js", "grid.js", "style.css", "icon.svg")
STATUS = (
    '<p id="pyodide-status" class="banner">Loading Python and NumPy into this page. '
    "The first visit takes a few seconds.</p>"
)


def main() -> int:
    """Build the demo page that runs LaneScope in the browser into a folder."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("out", type=pathlib.Path, help="folder for the page, such as _site")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    assemble(args.out, build_wheel(args.out).name)
    print(f"wrote the page to {args.out}")
    return 0


def build_wheel(out: pathlib.Path) -> pathlib.Path:
    """Build the wheel of LaneScope into the folder and return its path."""
    for old in out.glob("lanescope-*.whl"):
        old.unlink()
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--wheel-dir",
        str(out),
        str(ROOT),
    ]
    subprocess.run(command, check=True)
    (wheel,) = out.glob("lanescope-*.whl")
    return wheel


def assemble(out: pathlib.Path, wheel: str) -> None:
    """Copy the files of the page into the folder and point the page at Pyodide and the wheel."""
    for name in ASSETS:
        shutil.copy2(GUI / name, out / name)
    page = (GUI / "index.html").read_text()
    # GitHub Pages serves the page below the name of the repository, so paths stay relative.
    page = page.replace('"/static/', '"./')
    meta = (
        '<meta name="lanescope-backend" content="pyodide" />\n'
        f'    <meta name="lanescope-wheel" content="./{wheel}" />'
    )
    page = page.replace('<meta charset="utf-8" />', f'<meta charset="utf-8" />\n    {meta}', 1)
    page = page.replace('<div id="banner"', f'{STATUS}\n\n    <div id="banner"', 1)
    (out / "index.html").write_text(page)
    # Without this file, GitHub Pages would run Jekyll over the folder.
    (out / ".nojekyll").write_text("")


if __name__ == "__main__":
    sys.exit(main())
