"""Unified artifact containment, including resolved link targets."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from inferref.ir.package import TracePackage, is_trace_package
from inferref.ir.paths import (
    PathBoundaryError,
    resolve_allowed_path,
    resolve_contained_path,
)


def test_contained_relative_path_resolves_below_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    assert (
        resolve_contained_path(root, "nested/value.irtensor")
        == (root / "nested" / "value.irtensor").resolve()
    )


@pytest.mark.parametrize(
    "path",
    [
        "../secret.irtensor",
        r"..\secret.irtensor",
        "/var/tmp/secret.irtensor",
        r"\secret.irtensor",
        r"C:\secret.irtensor",
        r"C:secret.irtensor",
        r"\\server\share\secret.irtensor",
    ],
)
def test_cross_platform_absolute_and_parent_paths_are_rejected(
    tmp_path: Path, path: str
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(PathBoundaryError):
        resolve_contained_path(root, path)


def test_symlink_target_outside_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    link = root / "linked"
    os.symlink(outside, link, target_is_directory=True)

    with pytest.raises(PathBoundaryError, match="escapes root"):
        resolve_contained_path(root, "linked/secret.irtensor")
    with pytest.raises(PathBoundaryError, match="outside configured allowed roots"):
        resolve_allowed_path(link / "secret.irtensor", [root])


@pytest.mark.skipif(os.name != "nt", reason="Windows junction semantics")
def test_windows_junction_target_outside_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    junction = root / "junction"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout

    with pytest.raises(PathBoundaryError, match="escapes root"):
        resolve_contained_path(root, "junction/secret.irtensor")


def test_trace_metadata_symlink_cannot_escape_on_read_or_write(tmp_path: Path) -> None:
    trace = tmp_path / "trace"
    trace.mkdir()
    outside = tmp_path / "outside-manifest.json"
    outside.write_text("sentinel", encoding="utf-8")
    try:
        (trace / "manifest.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")
    (trace / "graph.json").write_text("{}", encoding="utf-8")

    assert not is_trace_package(trace)
    with pytest.raises(PathBoundaryError, match="escapes root"):
        TracePackage.load(trace)
    with pytest.raises(PathBoundaryError, match="escapes root"):
        TracePackage().save(trace)
    assert outside.read_text(encoding="utf-8") == "sentinel"
