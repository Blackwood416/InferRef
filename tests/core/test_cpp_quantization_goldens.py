"""Cross-language agreement for the C++ testcase 0.4 quantization ABI.

The Python reference is the only source of truth for these fixtures: the
generator writes float32 bit-pattern expectations, and the C++ self-test must
reproduce them exactly.  The committed fixtures under
``tests/fixtures/cpp_quantization_goldens`` make regeneration reviewable and
prove that ``scripts/generate_cpp_golden.py`` is deterministic.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from inferref.quantization import QuantizationSpec, dequantize
from inferref.tensor import codec
from inferref.testcase.validate import validate_testcase

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR = REPO_ROOT / "scripts" / "generate_cpp_golden.py"
FIXTURE_ROOT = REPO_ROOT / "tests" / "fixtures" / "cpp_quantization_goldens"


def _generate(destination: Path) -> Path:
    """Run the generator for quantization goldens only (tensor fixtures go elsewhere)."""

    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(REPO_ROOT)
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    subprocess.run(
        [
            sys.executable,
            str(GENERATOR),
            str(destination / "tensors"),
            "--quantization-goldens",
            str(destination / "goldens"),
        ],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return destination / "goldens"


def _relative_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_generator_is_deterministic(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")
    assert _relative_files(first) == _relative_files(second)


def test_committed_fixtures_match_generator(tmp_path: Path) -> None:
    generated = _generate(tmp_path / "committed")
    assert generated.is_dir(), "generator did not write quantization goldens"
    assert _relative_files(generated) == _relative_files(FIXTURE_ROOT), (
        "committed goldens are stale; regenerate them with "
        "scripts/generate_cpp_golden.py <tmp> --quantization-goldens "
        "tests/fixtures/cpp_quantization_goldens"
    )


def _golden_cases() -> list[dict]:
    return json.loads((FIXTURE_ROOT / "goldens.json").read_text(encoding="utf-8"))["cases"]


def test_goldens_match_python_reference() -> None:
    cases = _golden_cases()
    assert len(cases) >= 12
    for case in cases:
        case_dir = FIXTURE_ROOT / case["testcase"]
        manifest = json.loads((case_dir / "testcase.json").read_text(encoding="utf-8"))
        input_role = next(
            item for item in manifest["inputs"] if item["name"] == case["input"]
        )
        codes = codec.read(case_dir / input_role["payload"])
        spec = QuantizationSpec.from_dict(input_role["quantization"])
        values = {
            item["value_id"]: codec.read(case_dir / item["payload"])
            for item in manifest["quantization_parameters"]
        }
        expected = dequantize(codes, spec, values=values)
        actual_bits = [
            int(bits) for bits in np.ascontiguousarray(expected).reshape(-1).view(np.uint32)
        ]
        assert actual_bits == [int(bits, 16) for bits in case["bits"]], case["name"]

        reference = codec.read(case_dir / "reference" / "y.irtensor")
        assert reference.dtype == "float32"
        assert [
            int(bits)
            for bits in np.ascontiguousarray(reference.data).reshape(-1).view(np.uint32)
        ] == [int(bits, 16) for bits in case["bits"]], case["name"]


def test_generated_testcases_pass_standalone_validation() -> None:
    for case in _golden_cases():
        result = validate_testcase(FIXTURE_ROOT / case["testcase"])
        assert result.valid, (case["name"], [issue.to_dict() for issue in result.errors])
        assert result.reproducible, case["name"]


def _existing_selftest() -> Path | None:
    override = os.environ.get("INFERREF_CPP_QUANTIZATION_SELFTEST")
    if override:
        return Path(override)
    for build in sorted(REPO_ROOT.glob("cpp/build*")):
        for candidate in (
            build / "inferref_quantization_selftest.exe",
            build / "inferref_quantization_selftest",
            build / "Release" / "inferref_quantization_selftest.exe",
            build / "Release" / "inferref_quantization_selftest",
        ):
            if candidate.is_file():
                return candidate
    return None


def _build_selftest(tmp_path: Path) -> Path | None:
    """Build only the quantization self-test; returns None without a toolchain."""

    cmake = shutil.which("cmake")
    if not cmake:
        return None
    build_dir = tmp_path / "cpp-build"
    configure = subprocess.run(
        [
            cmake,
            "-S",
            str(REPO_ROOT / "cpp"),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if configure.returncode != 0:
        return None
    build = subprocess.run(
        [
            cmake,
            "--build",
            str(build_dir),
            "--config",
            "Release",
            "--target",
            "inferref_quantization_selftest",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if build.returncode != 0:
        return None
    for candidate in (
        build_dir / "inferref_quantization_selftest.exe",
        build_dir / "inferref_quantization_selftest",
        build_dir / "Release" / "inferref_quantization_selftest.exe",
        build_dir / "Release" / "inferref_quantization_selftest",
    ):
        if candidate.is_file():
            return candidate
    return None


def test_cpp_selftest_matches_python_goldens(tmp_path: Path) -> None:
    """Bit-for-bit float32 agreement between the Python goldens and C++."""

    selftest = _existing_selftest() or _build_selftest(tmp_path)
    if selftest is None:
        pytest.skip("no C++ toolchain or existing quantization self-test binary")

    result = subprocess.run(
        [str(selftest), str(FIXTURE_ROOT)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "C++ quantization self-test disagreed with the Python goldens\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
