import hashlib
import io
import json
import tarfile

import pytest

import lanescope.cli as cli_module
from lanescope.cli import main
from lanescope.cuda import toolchain
from lanescope.nsight import installer

INDEX = """<a href="redistrib_12.8.1.json">x</a> <a href="redistrib_12.9.1.json">x</a>
<a href="redistrib_12.9.0.json">x</a> <a href="redistrib_13.0.0.json">x</a> redistrib_bad.json"""
VERSION = "2025.2.1.3"


def make_redist(folder, digest=None, release="12.9.1", version=VERSION):
    """Write a small copy of the NVIDIA redist folder with one Nsight Compute archive."""
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:xz") as archive:
        script = b"#!/bin/sh\necho ncu\n"
        info = tarfile.TarInfo(f"nsight_compute-linux-x86_64-{version}-archive/ncu")
        info.size = len(script)
        info.mode = 0o755
        archive.addfile(info, io.BytesIO(script))
    relative = f"nsight_compute/linux-x86_64/nsight_compute-{version}-archive.tar.xz"
    (folder / relative).parent.mkdir(parents=True, exist_ok=True)
    (folder / relative).write_bytes(payload.getvalue())
    manifest = {
        "nsight_compute": {
            "version": version,
            "linux-x86_64": {
                "relative_path": relative,
                "sha256": digest or hashlib.sha256(payload.getvalue()).hexdigest(),
            },
        }
    }
    (folder / f"redistrib_{release}.json").write_text(json.dumps(manifest))
    (folder / "index.html").write_text(INDEX + f' <a href="redistrib_{release}.json">x</a>')


def base_url(folder):
    # A file URL of a folder has no index page, so the index lives in index.html.
    return folder.as_uri() + "/"


def test_releases_reads_every_manifest_name_once():
    assert installer.releases(INDEX + INDEX) == [(12, 8, 1), (12, 9, 0), (12, 9, 1), (13, 0, 0)]


def test_pick_takes_the_newest_release_the_driver_supports():
    found = installer.releases(INDEX)
    assert installer.pick(found, (12, 9)) == (12, 9, 1)
    assert installer.pick(found, (13, 2)) == (13, 0, 0)
    with pytest.raises(ValueError, match="at or below 12.4"):
        installer.pick(found, (12, 4))


@pytest.mark.parametrize(
    "manifest",
    [
        [],
        {"nsight_compute": {"version": VERSION}},
        {"nsight_compute": {"version": VERSION, "linux-x86_64": {"relative_path": "a"}}},
        {
            "nsight_compute": {
                "version": VERSION,
                "linux-x86_64": {"relative_path": "../../etc/passwd", "sha256": "0"},
            }
        },
    ],
)
def test_archive_rejects_a_manifest_without_a_safe_entry(manifest):
    with pytest.raises(ValueError):
        installer.archive(manifest)


def test_install_unpacks_the_checked_archive_and_reuses_it(tmp_path, monkeypatch):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist)
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    cache = tmp_path / "cache"
    found = installer.install((12, 9), cache, base_url(redist))
    assert found.read_bytes().startswith(b"#!/bin/sh")
    assert found.parent.parent == cache / f"nsight-compute-{VERSION}"
    for archive in redist.rglob("*.tar.xz"):
        archive.unlink()
    assert installer.install((12, 9), cache, base_url(redist)) == found


def test_install_refuses_an_archive_with_another_checksum(tmp_path, monkeypatch):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist, digest="0" * 64)
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    cache = tmp_path / "cache"
    with pytest.raises(ValueError, match="does not match the SHA-256"):
        installer.install((12, 9), cache, base_url(redist))
    assert list(cache.iterdir()) == []


def test_install_refuses_a_python_without_the_archive_filter_before_downloading(
    tmp_path, monkeypatch
):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist)
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    monkeypatch.delattr(tarfile, "data_filter")
    downloads = []
    monkeypatch.setattr(installer, "_download", lambda *args: downloads.append(args))
    with pytest.raises(ValueError, match="3.10.12, 3.11.4, 3.12"):
        installer.install((12, 9), tmp_path / "cache", base_url(redist))
    assert downloads == []


def test_toolchain_finds_the_installed_ncu_in_the_cache_of_home(tmp_path, monkeypatch):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist)
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    home = tmp_path / "home"
    found = installer.install((12, 9), home / ".cache" / "lanescope", base_url(redist))
    assert toolchain.find({"HOME": str(home), "PATH": ""}).ncu == found


def test_toolchain_prefers_the_downloaded_ncu_to_one_on_the_path(tmp_path, monkeypatch):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist)
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    cache = tmp_path / "cache"
    found = installer.install((12, 9), cache, base_url(redist))
    other = tmp_path / "bin" / "ncu"
    other.parent.mkdir()
    other.write_text("#!/bin/sh\n")
    other.chmod(0o755)
    environ = {"LANESCOPE_CACHE": str(cache), "PATH": str(other.parent)}
    assert toolchain.find(environ).ncu == found


def test_install_replaces_a_damaged_copy_on_request(tmp_path, monkeypatch):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist)
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    cache = tmp_path / "cache"
    found = installer.install((12, 9), cache, base_url(redist))
    found.write_text("damaged")
    assert installer.install((12, 9), cache, base_url(redist)).read_text() == "damaged"
    repaired = installer.install((12, 9), cache, base_url(redist), replace=True)
    assert repaired.read_text().startswith("#!/bin/sh")


def test_get_ncu_reports_an_ncu_that_is_found_without_downloading(monkeypatch, capsys, tmp_path):
    ncu = tmp_path / "ncu"
    monkeypatch.setattr(cli_module, "find", lambda: _tools(ncu))
    monkeypatch.setattr(installer, "install", lambda *args: pytest.fail("no download expected"))
    assert main(["get-ncu"]) == 0
    assert str(ncu) in capsys.readouterr().out


def test_get_ncu_downloads_for_a_given_cuda_version_even_beside_another_ncu(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(cli_module, "find", lambda: _tools(tmp_path / "ncu"))
    monkeypatch.setattr(cli_module, "cache_folder", lambda: tmp_path / "cache")

    def install(cuda, folder, replace=False):
        calls.append((cuda, folder, replace))
        return tmp_path / "cache" / "ncu"

    monkeypatch.setattr(installer, "install", install)
    assert main(["get-ncu", "--cuda-version", "12.6"]) == 0
    assert main(["get-ncu", "--cuda-version", "12.6", "--force"]) == 0
    assert calls == [((12, 6), tmp_path / "cache", False), ((12, 6), tmp_path / "cache", True)]


def test_get_ncu_rejects_a_malformed_cuda_version(capsys):
    with pytest.raises(SystemExit):
        main(["get-ncu", "--cuda-version", "twelve"])
    assert "such as 12.9" in capsys.readouterr().err


def _local_text(redist):
    def read(url):
        name = url.removeprefix(redist.as_uri() + "/") or "index.html"
        return (redist / name).read_text()

    return read


def _tools(ncu):
    return toolchain.Toolchain(None, None, None, None, None, None, ncu, None, None)


def test_a_failed_download_keeps_the_copy_it_would_replace(tmp_path, monkeypatch):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist)
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    cache = tmp_path / "cache"
    found = installer.install((12, 9), cache, base_url(redist))

    def broken(url, path, digest):
        raise OSError("timed out")

    monkeypatch.setattr(installer, "_download", broken)
    with pytest.raises(OSError, match="timed out"):
        installer.install((12, 9), cache, base_url(redist), replace=True)
    assert found.is_file()
    assert installer.cached_ncu(cache) == found


def test_cached_ncu_takes_the_copy_of_the_last_install(tmp_path):
    for version in ("2025.2.1.3", "2024.1.1.4"):
        ncu = tmp_path / f"nsight-compute-{version}" / "ncu"
        ncu.parent.mkdir()
        ncu.write_text("")
    assert installer.cached_ncu(tmp_path) == tmp_path / "nsight-compute-2025.2.1.3" / "ncu"
    (tmp_path / "current").write_text("nsight-compute-2024.1.1.4")
    assert installer.cached_ncu(tmp_path) == tmp_path / "nsight-compute-2024.1.1.4" / "ncu"


def test_cached_ncu_without_a_record_compares_versions_as_numbers(tmp_path):
    for version in ("2025.2.0.0", "2025.10.0.0"):
        ncu = tmp_path / f"nsight-compute-{version}" / "ncu"
        ncu.parent.mkdir()
        ncu.write_text("")
    assert installer.cached_ncu(tmp_path) == tmp_path / "nsight-compute-2025.10.0.0" / "ncu"


def test_toolchain_takes_the_copy_of_the_last_install_over_a_newer_one(tmp_path, monkeypatch):
    redist = tmp_path / "redist"
    redist.mkdir()
    make_redist(redist)
    make_redist(redist, release="12.4.1", version="2024.1.1.4")
    monkeypatch.setattr(installer, "_text", _local_text(redist))
    cache = tmp_path / "cache"
    newer = installer.install((12, 9), cache, base_url(redist))
    older = installer.install((12, 4), cache, base_url(redist))
    assert newer != older
    assert toolchain.find({"LANESCOPE_CACHE": str(cache), "PATH": ""}).ncu == older
