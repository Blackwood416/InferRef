"""Quantized dispatch through the C++ runtime bridge (InferRef 0.12 section 5.4).

The legacy two-argument ``DebugInvoke`` path must refuse a testcase 0.4 with
quantized inputs with ``kBridgeUnsupportedQuantization`` (exit code 4) instead
of silently dropping scales, and the Testcase-aware path must handle the same
fixture.  These tests build the reference bridge binaries with CMake and skip
only when no C++ toolchain is available.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cpp_quantization_goldens"
QUANTIZED_CASE = FIXTURE_ROOT / "cases" / "int8-per-group"


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

    build_dir = tmp_path_factory.mktemp("bridge-build")
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


def test_legacy_bridge_refuses_quantized_testcase(
    bridge_binaries: tuple[Path, Path], tmp_path: Path
) -> None:
    bridge, _ = bridge_binaries
    assert QUANTIZED_CASE.is_dir(), f"missing fixture: {QUANTIZED_CASE}"
    output_dir = tmp_path / "output"

    result = _run(
        [
            str(bridge),
            "--testcase",
            str(QUANTIZED_CASE),
            "--output",
            str(output_dir),
        ]
    )

    assert result.returncode == 4, (
        f"expected kBridgeUnsupportedQuantization (4), got {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    diagnostic = f"{result.stdout}\n{result.stderr}".lower()
    assert "unsupported_quantization" in diagnostic
    assert "quantizes input role(s) 'weight'" in diagnostic
    assert "debuginvokeex" in diagnostic
    assert not (output_dir / "manifest.json").exists()


def test_bridge_selftest_covers_quantized_and_unquantized_dispatch(
    bridge_binaries: tuple[Path, Path]
) -> None:
    _, selftest = bridge_binaries
    result = _run([str(selftest), str(FIXTURE_ROOT)])

    assert result.returncode == 0, (
        f"bridge self-test failed with {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    assert "kBridgeUnsupportedQuantization" in result.stdout
    assert "quantized 0.4 dispatch" in result.stdout
