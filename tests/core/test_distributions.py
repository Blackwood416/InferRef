"""Regression tests for the release artifact gate (stdlib-only archive checks)."""
from __future__ import annotations

import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from scripts.check_distributions import check


def _pair(root: Path, *, typed: bool = True, source: str = "0.10.1") -> None:
    files = {
        "inferref/ir/version.py": f"INFERREF_VERSION = {source!r}\n".encode(),
        "inferref/alignment/runner.py": b"",
        "inferref/cli/main.py": b"",
    }
    if typed:
        files["inferref/py.typed"] = b""
    metadata = b"Metadata-Version: 2.1\nName: inferref\nVersion: 0.10.1\n"
    with zipfile.ZipFile(root / "inferref-0.10.1-py3-none-any.whl", "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("inferref-0.10.1.dist-info/METADATA", metadata)
    with tarfile.open(root / "inferref-0.10.1.tar.gz", "w:gz") as archive:
        for name, content in {**files, "PKG-INFO": metadata, "README.md": b"",
                              "pyproject.toml": b""}.items():
            info = tarfile.TarInfo("inferref-0.10.1/" + name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_distribution_pair_and_expected_tag(tmp_path: Path) -> None:
    _pair(tmp_path)
    assert check(tmp_path, "0.10.1")[2] == "0.10.1"
    with pytest.raises(ValueError, match="expected version"):
        check(tmp_path, "0.10.0")


def test_distribution_marker_required(tmp_path: Path) -> None:
    _pair(tmp_path, typed=False)
    with pytest.raises(KeyError, match="py.typed"):
        check(tmp_path)


def test_distribution_versions_must_agree(tmp_path: Path) -> None:
    _pair(tmp_path, source="0.9.2")
    with pytest.raises(ValueError, match="versions disagree"):
        check(tmp_path)


def test_distribution_output_directory_must_be_unambiguous(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly one"):
        check(tmp_path)
    _pair(tmp_path)
    (tmp_path / "old.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        check(tmp_path)
