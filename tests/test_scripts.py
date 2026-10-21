import datetime
import os
import shutil
import subprocess

import pytest

SCRIPTS = os.path.join(os.path.dirname(__file__), os.pardir, "scripts")
# One stub stands for pip, python, and lanescope inside the environment. It logs each
# call, answers the questions the scripts ask, adds pip when ensurepip runs, and adds
# lanescope when pip installs. The check prints the wheel switch it runs with.
STUB = """#!/bin/sh
echo "$(basename "$0") $*" >> "{log}"
if [ "$1 $2" = "gpu-info --cuda-version" ]; then echo "{cuda}"; fi
if [ "$1" = "validate" ]; then echo "LANESCOPE_USE_WHEELS=${{LANESCOPE_USE_WHEELS:-}}"; fi
if [ "$1" = "get-ncu" ]; then exit {ncu_status}; fi
if [ "$1" = "-c" ]; then exit {python_status}; fi
if [ "$1 $2" = "-m ensurepip" ]; then cp "$0" "$(dirname "$0")/pip"; fi
if [ "$1" = "install" ]; then cp "$0" "$(dirname "$0")/lanescope"; fi
exit 0
"""
# The python3 on the PATH, which makes a stub environment or fails after it made the folder.
PYTHON3 = """#!/bin/sh
echo "python3 $*" >> "{log}"
if [ "$1 $2" = "-m venv" ]; then
  mkdir -p "$3"
  if [ {venv_status} != 0 ]; then exit {venv_status}; fi
  mkdir -p "$3/bin" && touch "$3/pyvenv.cfg"
  for name in pip python lanescope; do cp "{stub}" "$3/bin/$name"; done
fi
exit 0
"""
# The python of an editable install, which knows the package by its folder without links.
INSTALLED_PYTHON = """#!/bin/sh
echo "python $*" >> "{log}"
[ "$3" = "{installed}/" ]
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash is needed")


@pytest.fixture
def layout(tmp_path):
    """Copy the scripts into a project folder and write the stub programs beside it."""
    scripts = tmp_path / "lanescope" / "scripts"
    scripts.mkdir(parents=True)
    for name in ("venv.sh", "gui.sh", "gpu_validate.sh"):
        shutil.copy(os.path.join(SCRIPTS, name), scripts / name)
    return tmp_path


def make_venv(layout, names=("pip", "python", "lanescope"), config=True, **answers):
    """Make a stub environment that holds the named programs."""
    folder = layout / "venv" / "bin"
    folder.mkdir(parents=True, exist_ok=True)
    if config:
        (layout / "venv" / "pyvenv.cfg").write_text("")
    for name in names:
        write_stub(layout, folder / name, **answers)


def write_stub(layout, path, cuda="12.9", ncu_status=0, python_status=0):
    log = layout / "calls.log"
    path.write_text(
        STUB.format(log=log, cuda=cuda, ncu_status=ncu_status, python_status=python_status)
    )
    path.chmod(0o755)


def run(layout, script, *args, venv_status=0, root=None):
    """Run a script with the stub python3 first on the PATH and return its calls.

    The script lives under `root`, such as a link to the layout, and runs in the layout.
    """
    log = layout / "calls.log"
    tools = layout / "tools"
    tools.mkdir(exist_ok=True)
    write_stub(layout, tools / "stub")
    python3 = tools / "python3"
    python3.write_text(PYTHON3.format(log=log, venv_status=venv_status, stub=tools / "stub"))
    python3.chmod(0o755)
    environment = dict(
        os.environ,
        LANESCOPE_VENV=str(layout / "venv"),
        PATH=f"{tools}{os.pathsep}{os.environ['PATH']}",
    )
    environment.pop("LANESCOPE_USE_WHEELS", None)
    script_path = (root or layout) / "lanescope" / "scripts" / script
    result = subprocess.run(
        ["bash", str(script_path), *args],
        env=environment,
        cwd=layout,
        capture_output=True,
        text=True,
        check=False,
    )
    calls = log.read_text().splitlines() if log.exists() else []
    return result, calls


def installs(calls):
    return [call for call in calls if call.startswith("pip install")]


def test_gui_script_shows_the_demo_without_a_check(layout):
    make_venv(layout)
    result, calls = run(layout, "gui.sh")
    assert result.returncode == 0, result.stderr
    assert calls[-1] == "lanescope gui --demo"
    assert installs(calls) == []


def test_gui_script_installs_lanescope_when_it_is_missing(layout):
    make_venv(layout, names=("pip", "python"))
    _, calls = run(layout, "gui.sh")
    assert installs(calls) == [
        "pip install --quiet --upgrade pip",
        f"pip install --quiet -e {layout}/lanescope",
    ]


def test_gui_script_installs_again_for_the_copy_of_another_folder(layout):
    make_venv(layout, python_status=1)
    _, calls = run(layout, "gui.sh")
    assert f"pip install --quiet -e {layout}/lanescope" in installs(calls)
    assert calls[-1] == "lanescope gui --demo"


def test_gui_script_opens_no_folder_outside_the_repository(layout):
    make_venv(layout)
    (layout / "analysis" / "lanescope").mkdir(parents=True)
    _, calls = run(layout, "gui.sh")
    assert calls[-1] == "lanescope gui --demo"


def test_gui_script_passes_its_arguments_through(layout):
    make_venv(layout)
    _, calls = run(layout, "gui.sh", "run1", "--host", "0.0.0.0")
    assert calls[-1] == "lanescope gui run1 --host 0.0.0.0"


def test_gui_script_keeps_the_install_when_the_project_is_reached_through_a_link(layout):
    make_venv(layout)
    installed = (layout / "lanescope" / "src").resolve()
    python = layout / "venv" / "bin" / "python"
    python.write_text(INSTALLED_PYTHON.format(log=layout / "calls.log", installed=installed))
    (layout / "link").symlink_to(layout, target_is_directory=True)
    result, calls = run(layout, "gui.sh", root=layout / "link")
    assert result.returncode == 0, result.stderr
    assert installs(calls) == []
    assert calls[-1] == "lanescope gui --demo"


def test_an_empty_folder_becomes_the_environment(layout):
    (layout / "venv").mkdir()
    result, calls = run(layout, "gui.sh")
    assert result.returncode == 0, result.stderr
    assert f"python3 -m venv {layout}/venv" in calls
    assert calls[-1] == "lanescope gui --demo"


def test_an_environment_without_pip_gets_pip_in_place(layout):
    make_venv(layout, names=("python",))
    result, calls = run(layout, "gui.sh")
    assert result.returncode == 0, result.stderr
    assert "python -m ensurepip --upgrade" in calls
    assert not any(call.startswith("python3 -m venv") for call in calls)
    assert (layout / "venv" / "pyvenv.cfg").exists()


def test_scripts_never_remove_a_folder_that_holds_no_environment(layout):
    folder = layout / "venv" / "drafts"
    folder.mkdir(parents=True)
    (folder / "thesis.txt").write_text("keep")
    result, _ = run(layout, "gui.sh")
    assert result.returncode == 1
    assert "holds no virtual environment" in result.stderr
    assert (folder / "thesis.txt").read_text() == "keep"


def test_a_failed_environment_keeps_a_folder_that_existed(layout):
    (layout / "venv").mkdir()
    result, _ = run(layout, "gui.sh", venv_status=1)
    assert result.returncode == 1
    assert "cannot make a virtual environment" in result.stderr
    assert (layout / "venv").is_dir()


def test_a_failed_environment_removes_the_folder_it_made(layout):
    result, _ = run(layout, "gui.sh", venv_status=1)
    assert result.returncode == 1
    assert not (layout / "venv").exists()


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")
def test_gpu_script_runs_the_check_when_the_nsight_compute_download_fails(layout):
    make_venv(layout, ncu_status=2)
    result, calls = run(layout, "gpu_validate.sh", str(layout / "out"))
    assert result.returncode == 0, result.stderr
    assert "The Nsight Compute download failed" in result.stderr
    assert f"pip install --quiet -e {layout}/lanescope[gpu]" in calls
    assert calls[-1] == f"lanescope validate --out {layout}/out"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")
def test_gpu_script_writes_the_check_into_the_current_folder_by_default(layout):
    make_venv(layout)
    result, calls = run(layout, "gpu_validate.sh")
    assert result.returncode == 0, result.stderr
    assert calls[-1] == f"lanescope validate --out check-{datetime.date.today().isoformat()}"


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")
def test_gpu_script_says_how_to_get_nsight_compute_when_the_download_fails(layout):
    make_venv(layout, ncu_status=2)
    result, _ = run(layout, "gpu_validate.sh", str(layout / "out"))
    assert f"Run {layout}/venv/bin/lanescope get-ncu or install Nsight Compute" in result.stderr


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")
@pytest.mark.parametrize(("cuda", "switch"), [("12.6", "1"), ("12.9", "")])
def test_gpu_script_takes_the_cuda_wheels_over_a_toolkit_when_it_pins_them(layout, cuda, switch):
    make_venv(layout, cuda=cuda)
    result, _ = run(layout, "gpu_validate.sh", str(layout / "out"))
    assert result.returncode == 0, result.stderr
    assert f"LANESCOPE_USE_WHEELS={switch}" in result.stdout.splitlines()


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")
def test_gpu_script_keeps_the_cuda_wheels_at_the_version_of_an_older_driver(layout):
    make_venv(layout, cuda="12.6")
    _, calls = run(layout, "gpu_validate.sh", str(layout / "out"))
    pinned = [call for call in calls if "nvidia-cuda-nvrtc-cu12" in call]
    assert pinned and ">=12.4,<12.7" in pinned[0]


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ is needed")
def test_gpu_script_stops_on_a_driver_older_than_cuda_12_4(layout):
    make_venv(layout, cuda="12.2")
    result, calls = run(layout, "gpu_validate.sh", str(layout / "out"))
    assert result.returncode == 1
    assert "needs CUDA 12.4 or later" in result.stderr
    assert not any("validate" in call for call in calls)


def test_scripts_with_a_shebang_are_executable():
    for name in sorted(os.listdir(SCRIPTS)):
        path = os.path.join(SCRIPTS, name)
        with open(path) as script:
            if script.readline().startswith("#!"):
                assert os.access(path, os.X_OK), f"{name} is not executable"
