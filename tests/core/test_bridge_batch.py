"""Batch-manifest execution through the C++ runtime bridge (InferRef 0.12 section 7.3).

One ``--batch-manifest`` invocation must run every ordered item in one bridge
process, write each item's outputs and ``manifest.json``, keep going after a
failing item, and publish ``batch-results.json`` with per-item status, exit
code, and paths plus counts.  The final exit code is zero only when every item
passed.  The tests build the reference ``inferref-bridge`` binary and the
KV-stub ``inferref_bridge_selftest`` binary with CMake and skip only when no
C++ toolchain is available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from inferref.tensor import codec

REPO_ROOT = Path(__file__).resolve().parents[2]

KV_REGION = "kv-cache/append/v1"
UNSUPPORTED_REGION = "unsupported/region/v1"


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _toolchain_unavailable(output: str) -> bool:
    lowered = output.lower()
    return any(
        marker in lowered
        for marker in (
            "no cmake_cxx_compiler",
            "the c++ compiler",
            "is not able to compile a simple test program",
            "compiler not found",
            "c1083",
            "cannot open include file",
            "no such file or directory",
        )
    )


def _locate(build_dir: Path, name: str) -> Path | None:
    for candidate in (
        build_dir / f"{name}.exe",
        build_dir / name,
        build_dir / "Release" / f"{name}.exe",
        build_dir / "Release" / name,
        build_dir / "Debug" / f"{name}.exe",
        build_dir / "Debug" / name,
    ):
        if candidate.is_file():
            return candidate
    return None


@pytest.fixture(scope="module")
def bridge_binaries(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    cmake = shutil.which("cmake")
    if not cmake:
        pytest.skip("CMake is not available on PATH")

    build_dir = tmp_path_factory.mktemp("bridge-batch-build")
    configure = _run(
        [
            cmake,
            "-S",
            str(REPO_ROOT / "cpp"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ]
    )
    if configure.returncode != 0:
        combined = f"{configure.stdout}\n{configure.stderr}"
        if _toolchain_unavailable(combined):
            pytest.skip(f"No C++ compiler available for CMake:\n{combined}")
        pytest.fail(f"CMake configure failed:\n{combined}")

    build = _run(
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "inferref_bridge",
            "inferref_bridge_selftest",
        ]
    )
    if build.returncode != 0:
        combined = f"{build.stdout}\n{build.stderr}"
        if _toolchain_unavailable(combined):
            pytest.skip(f"C++ build environment not configured:\n{combined}")
        pytest.fail(f"CMake build failed:\n{combined}")

    bridge = _locate(build_dir, "inferref_bridge")
    selftest = _locate(build_dir, "inferref_bridge_selftest")
    if bridge is None or selftest is None:
        pytest.skip(f"built bridge binaries not found in {build_dir}")
    return bridge, selftest


def _write_testcase(root: Path, region: str = KV_REGION) -> Path:
    """One testcase the C++ selftest's KV stub engine can dispatch."""
    root.mkdir(parents=True, exist_ok=True)
    inputs = {
        "cache": np.zeros((1, 1, 2, 2), dtype=np.float32),
        "update": np.full((1, 1, 1, 2), 2.0, dtype=np.float32),
        "scale": np.array([1.5], dtype=np.float32),
    }
    entries = []
    for name, array in inputs.items():
        payload = codec.write_array(root / "inputs" / f"{name}.irtensor", array)
        entries.append(
            {"name": name, "payload": payload.relative_to(root).as_posix()}
        )
    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.2",
        "name": "bridge-batch-fixture",
        "origin": {"region": region},
        "inputs": entries,
        "outputs": [{"name": "cache_out"}, {"name": "logits"}],
    }
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_batch_manifest(
    path: Path, items: list[tuple[str, str, str]]
) -> Path:
    manifest = {
        "format": "inferref-batch",
        "format_version": "1.0",
        "items": [
            {"id": item_id, "testcase_dir": testcase_dir, "output_dir": output_dir}
            for item_id, testcase_dir, output_dir in items
        ],
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _results(work: Path) -> dict[str, object]:
    return json.loads((work / "batch-results.json").read_text(encoding="utf-8"))


def test_batch_manifest_runs_items_in_one_process_with_partial_failure(
    bridge_binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    _, selftest = bridge_binaries
    work = tmp_path / "batch"
    _write_testcase(work / "case-a")
    _write_testcase(work / "case-b", region=UNSUPPORTED_REGION)
    _write_testcase(work / "case-c")
    manifest = _write_batch_manifest(
        work / "batch.json",
        [
            ("step-a", "case-a", "out-a"),
            ("step-b", "case-b", "out-b"),
            ("step-c", "case-c", "out-c"),
        ],
    )

    result = _run([str(selftest), "--batch-manifest", str(manifest)])

    # One bridge invocation is the one-process proof: all three items were
    # loaded and dispatched before this process exited.
    assert result.returncode == 1, (
        f"expected a non-zero final exit code, got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "step-b" in result.stderr

    for output in ("out-a", "out-c"):
        output_manifest = work / output / "manifest.json"
        assert output_manifest.is_file(), f"missing {output_manifest}"
        written = json.loads(output_manifest.read_text(encoding="utf-8"))
        assert [entry["name"] for entry in written["outputs"]] == [
            "cache_out",
            "logits",
        ]
        assert (work / output / "cache_out.irtensor").is_file()
        assert (work / output / "logits.irtensor").is_file()
    assert not (work / "out-b" / "manifest.json").exists()

    results = _results(work)
    assert results["format"] == "inferref-batch-results"
    assert results["format_version"] == "1.0"
    items = results["items"]
    assert [item["id"] for item in items] == ["step-a", "step-b", "step-c"]
    assert [item["status"] for item in items] == ["passed", "failed", "passed"]
    assert [item["exit_code"] for item in items] == [0, 1, 0]
    assert items[0]["error"] is None
    assert "unknown region" in items[1]["error"]
    assert Path(items[0]["testcase_dir"]).resolve() == (work / "case-a").resolve()
    assert Path(items[0]["output_dir"]).resolve() == (work / "out-a").resolve()
    assert results["counts"] == {"total": 3, "passed": 2, "failed": 1}


def test_batch_manifest_all_items_pass_exits_zero(
    bridge_binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    _, selftest = bridge_binaries
    work = tmp_path / "batch-pass"
    _write_testcase(work / "case-1")
    _write_testcase(work / "case-2")
    manifest = _write_batch_manifest(
        work / "batch.json",
        [("step-1", "case-1", "out-1"), ("step-2", "case-2", "out-2")],
    )

    result = _run([str(selftest), "--batch-manifest", str(manifest)])

    assert result.returncode == 0, (
        f"expected exit 0, got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    for output in ("out-1", "out-2"):
        assert (work / output / "manifest.json").is_file()
    results = _results(work)
    assert [item["status"] for item in results["items"]] == ["passed", "passed"]
    assert all(item["exit_code"] == 0 for item in results["items"])
    assert results["counts"] == {"total": 2, "passed": 2, "failed": 0}


def test_reference_bridge_cli_batch_mode_records_item_failures(
    bridge_binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    bridge, _ = bridge_binaries
    work = tmp_path / "reference-batch"
    _write_testcase(work / "case-0")
    _write_testcase(work / "case-1")
    manifest = _write_batch_manifest(
        work / "batch.json",
        [("step-0", "case-0", "out-0"), ("step-1", "case-1", "out-1")],
    )

    result = _run([str(bridge), "--batch-manifest", str(manifest)])

    assert result.returncode == 1, (
        f"the unregistered reference callback must fail every item, got "
        f"{result.returncode}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "no DebugInvoke registered" in result.stderr
    results = _results(work)
    assert [item["status"] for item in results["items"]] == ["failed", "failed"]
    assert all(item["exit_code"] == 1 for item in results["items"])
    assert "no DebugInvoke registered" in results["items"][0]["error"]
    assert results["counts"] == {"total": 2, "passed": 0, "failed": 2}


def test_batch_manifest_rejects_mixed_single_testcase_flags(
    bridge_binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    bridge, _ = bridge_binaries
    result = _run(
        [
            str(bridge),
            "--batch-manifest",
            str(tmp_path / "batch.json"),
            "--testcase",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert result.returncode == 2
    assert "cannot be combined" in result.stderr


_INVALID_MANIFESTS = [
    (
        "not-json",
        "{",
        "not valid JSON",
    ),
    (
        "wrong-format",
        '{"format": "inferref-other", "format_version": "1.0", '
        '"items": [{"id": "a", "testcase_dir": "a", "output_dir": "b"}]}',
        "format must be 'inferref-batch'",
    ),
    (
        "wrong-version",
        '{"format": "inferref-batch", "format_version": "2.0", '
        '"items": [{"id": "a", "testcase_dir": "a", "output_dir": "b"}]}',
        "format_version",
    ),
    (
        "empty-items",
        '{"format": "inferref-batch", "format_version": "1.0", "items": []}',
        "must not be empty",
    ),
    (
        "missing-id",
        '{"format": "inferref-batch", "format_version": "1.0", '
        '"items": [{"testcase_dir": "a", "output_dir": "b"}]}',
        "missing 'id'",
    ),
    (
        "missing-testcase-dir",
        '{"format": "inferref-batch", "format_version": "1.0", '
        '"items": [{"id": "a", "output_dir": "b"}]}',
        "missing 'testcase_dir'",
    ),
    (
        "duplicate-id",
        '{"format": "inferref-batch", "format_version": "1.0", '
        '"items": [{"id": "a", "testcase_dir": "a", "output_dir": "b"}, '
        '{"id": "a", "testcase_dir": "c", "output_dir": "d"}]}',
        "duplicate batch item id",
    ),
    (
        "traversal",
        '{"format": "inferref-batch", "format_version": "1.0", '
        '"items": [{"id": "a", "testcase_dir": "../escape", '
        '"output_dir": "b"}]}',
        "'..' path segments",
    ),
]


@pytest.mark.parametrize(
    ("name", "text", "fragment"),
    _INVALID_MANIFESTS,
    ids=[case[0] for case in _INVALID_MANIFESTS],
)
def test_batch_manifest_rejects_invalid_manifests(
    bridge_binaries: tuple[Path, Path],
    tmp_path: Path,
    name: str,
    text: str,
    fragment: str,
) -> None:
    _, selftest = bridge_binaries
    manifest = tmp_path / f"{name}.json"
    manifest.write_text(text, encoding="utf-8")

    result = _run([str(selftest), "--batch-manifest", str(manifest)])

    assert result.returncode == 1, (
        f"expected kBridgeError (1) for {name}, got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert fragment in result.stderr
    assert not (tmp_path / "batch-results.json").exists()
