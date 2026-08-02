"""Comparator tests (SPEC §26-§31, §35).

Covers the metric definitions, the per-dtype tolerance policy, the layout /
value distinction of SPEC §29, and first-divergence search.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.compare.compare import (
    STATUS_FAIL,
    STATUS_MISSING,
    STATUS_PASS,
    compare_tensors,
    compare_testcase,
)
from inferref.compare.metrics import compute_metrics
from inferref.compare.report import render_report, render_short
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy
from inferref.tensor import codec


def _view(values, dtype="float32", stride=None, storage_offset=0):
    array = np.asarray(values, dtype=np.float32 if dtype == "float32" else dtype)
    shape = array.shape
    return codec.decode(
        codec.encode(
            dtype=dtype,
            shape=shape,
            stride=stride or codec.contiguous_stride(shape),
            storage_offset=storage_offset,
            payload=array.tobytes(),
        )
    )


# -- metrics ---------------------------------------------------------------


def test_identical_tensors() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    m = compute_metrics(a, a)
    assert m.exact_match
    assert m.max_abs_error == 0.0
    assert m.mismatch_count == 0
    assert m.cosine_similarity == pytest.approx(1.0)
    assert m.first_mismatch_index is None


def test_metric_values() -> None:
    reference = np.array([1.0, 2.0, 4.0])
    actual = np.array([1.0, 2.5, 3.0])
    m = compute_metrics(reference, actual)

    assert not m.exact_match
    assert m.max_abs_error == pytest.approx(1.0)          # |3 - 4|
    assert m.max_rel_error == pytest.approx(0.25)         # 0.5/2 vs 1.0/4
    assert m.mean_abs_error == pytest.approx(1.5 / 3)
    assert m.rmse == pytest.approx(np.sqrt((0 + 0.25 + 1.0) / 3))
    assert m.mismatch_count == 2
    assert m.element_count == 3


def test_first_mismatch_is_located_by_flat_order() -> None:
    reference = np.zeros((2, 3, 4))
    actual = np.zeros((2, 3, 4))
    actual[1, 2, 3] = 5.0
    actual[0, 1, 2] = 7.0

    m = compute_metrics(reference, actual)
    assert m.mismatch_count == 2
    # Row-major flat order: [0,1,2] comes first.
    assert m.first_mismatch_index == (0, 1, 2)
    assert m.first_mismatch_reference == 0.0
    assert m.first_mismatch_actual == 7.0


def test_tolerance_is_respected() -> None:
    reference = np.array([1.0, 100.0])
    actual = np.array([1.0009, 100.09])

    strict = compute_metrics(reference, actual, atol=0.0, rtol=0.0)
    assert strict.mismatch_count == 2

    loose = compute_metrics(reference, actual, atol=0.0, rtol=1e-3)
    assert loose.mismatch_count == 0


def test_nan_handling() -> None:
    reference = np.array([1.0, np.nan, 3.0])
    matching = np.array([1.0, np.nan, 3.0])
    differing = np.array([1.0, 2.0, 3.0])

    same = compute_metrics(reference, matching)
    assert same.mismatch_count == 0
    assert same.reference_nan_count == 1
    assert same.nan_count == 1
    assert not same.nan_mismatch

    diff = compute_metrics(reference, differing)
    assert diff.mismatch_count == 1
    assert diff.nan_mismatch
    assert diff.first_mismatch_index == (1,)


def test_inf_sign_matters() -> None:
    reference = np.array([np.inf, -np.inf])
    same = compute_metrics(reference, np.array([np.inf, -np.inf]))
    assert same.mismatch_count == 0

    flipped = compute_metrics(reference, np.array([np.inf, np.inf]))
    assert flipped.mismatch_count == 1
    assert flipped.first_mismatch_index == (1,)


def test_nonfinite_values_do_not_poison_aggregates() -> None:
    """A single inf must not turn every metric into nan."""
    reference = np.array([1.0, 2.0, np.inf])
    actual = np.array([1.0, 2.5, np.inf])
    m = compute_metrics(reference, actual)
    assert np.isfinite(m.max_abs_error)
    assert m.max_abs_error == pytest.approx(0.5)
    assert np.isfinite(m.rmse)
    assert np.isfinite(m.cosine_similarity)


def test_cosine_similarity_of_opposite_vectors() -> None:
    a = np.array([1.0, 2.0, 3.0])
    m = compute_metrics(a, -a)
    assert m.cosine_similarity == pytest.approx(-1.0)


def test_zero_vectors_are_similar() -> None:
    zeros = np.zeros(4)
    assert compute_metrics(zeros, zeros).cosine_similarity == pytest.approx(1.0)


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compute_metrics(np.zeros(3), np.zeros(4))


def test_empty_tensors_match() -> None:
    m = compute_metrics(np.zeros(0), np.zeros(0))
    assert m.exact_match
    assert m.mismatch_count == 0


# -- tolerance policy ------------------------------------------------------


def test_default_tolerances_scale_with_precision() -> None:
    policy = TolerancePolicy()
    f32_atol, _ = policy.for_dtype("float32")
    f16_atol, _ = policy.for_dtype("float16")
    bf16_atol, _ = policy.for_dtype("bfloat16")
    # bfloat16 has 8 mantissa bits; a float32 tolerance would be meaningless.
    assert f32_atol < f16_atol < bf16_atol
    assert policy.for_dtype("int64") == (0.0, 0.0)   # integers must be exact


def test_tolerance_override() -> None:
    policy = TolerancePolicy(override_atol=0.5, override_rtol=0.0)
    assert policy.for_dtype("float32") == (0.5, 0.0)
    assert policy.for_dtype("int32") == (0.5, 0.0)


def test_tolerance_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "tol.json"
    path.write_text(json.dumps({"float32": {"atol": 1e-2, "rtol": 1e-3}}))
    policy = TolerancePolicy.load(path)
    assert policy.for_dtype("float32") == (1e-2, 1e-3)
    # Unlisted dtypes keep their defaults.
    assert policy.for_dtype("float16") == DEFAULT_TOLERANCES["float16"]


# -- layout vs value (SPEC §29) --------------------------------------------


def test_stride_difference_passes_but_is_reported() -> None:
    """A fused engine need not reproduce PyTorch's layout (SPEC §20)."""
    reference = _view([1.0, 2.0, 3.0, 4.0], stride=(2,))
    actual = _view([1.0, 2.0, 3.0, 4.0], stride=(1,))

    result = compare_tensors("t", reference, actual, policy=TolerancePolicy())
    assert result.status == STATUS_PASS
    assert result.layout.stride_mismatch
    assert "identical when interpreted logically" in result.message


def test_strict_layout_makes_stride_fatal() -> None:
    reference = _view([1.0, 2.0], stride=(2,))
    actual = _view([1.0, 2.0], stride=(1,))
    result = compare_tensors(
        "t", reference, actual, policy=TolerancePolicy(), strict_layout=True
    )
    assert result.status == STATUS_FAIL
    assert result.metrics.mismatch_count == 0   # values were fine


def test_ignore_stride_suppresses_the_report() -> None:
    reference = _view([1.0, 2.0], stride=(2,))
    actual = _view([1.0, 2.0], stride=(1,))
    result = compare_tensors(
        "t", reference, actual, policy=TolerancePolicy(), ignore_stride=True
    )
    assert result.status == STATUS_PASS
    assert not result.layout.stride_mismatch
    assert result.message == ""


def test_shape_mismatch_is_not_comparable() -> None:
    result = compare_tensors(
        "t", _view([1.0, 2.0]), _view([1.0, 2.0, 3.0]), policy=TolerancePolicy()
    )
    assert result.status == STATUS_FAIL
    assert result.layout.shape_mismatch
    assert result.metrics is None       # values were never compared
    assert "shape" in result.message


def test_dtype_mismatch_is_not_comparable() -> None:
    result = compare_tensors(
        "t",
        _view([1.0, 2.0], dtype="float32"),
        _view([1.0, 2.0], dtype="float64"),
        policy=TolerancePolicy(),
    )
    assert result.status == STATUS_FAIL
    assert result.layout.dtype_mismatch
    assert result.metrics is None


# -- testcase comparison ---------------------------------------------------


def _make_testcase(root: Path, outputs: dict[str, np.ndarray]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"format": "inferref-testcase", "format_version": "0.1",
                "name": "t", "inputs": [], "outputs": []}
    for name, array in outputs.items():
        relative = f"reference/{name}.irtensor"
        path = codec.write_array(root / relative, array)
        manifest["outputs"].append(
            {
                "name": name,
                "value_id": None,
                "payload": relative,
                **codec.read(path).to_metadata(),
            }
        )
    (root / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_testcase_comparison_passes(tmp_path: Path) -> None:
    values = np.arange(6, dtype=np.float32).reshape(2, 3)
    _make_testcase(tmp_path / "tc", {"out": values})
    codec.write_array(tmp_path / "engine" / "out.irtensor", values)

    report = compare_testcase(tmp_path / "tc", tmp_path / "engine")
    assert report.status == STATUS_PASS
    assert report.passed_count == 1
    assert report.first_failure is None
    assert "PASS" in render_short(report)


def test_testcase_comparison_detects_missing_output(tmp_path: Path) -> None:
    _make_testcase(tmp_path / "tc", {"out": np.zeros(3, dtype=np.float32)})
    (tmp_path / "engine").mkdir()

    report = compare_testcase(tmp_path / "tc", tmp_path / "engine")
    assert report.status == STATUS_FAIL
    assert report.missing_count == 1
    assert report.first_failure.status == STATUS_MISSING


def test_testcase_comparison_reports_missing_reference_payload(tmp_path: Path) -> None:
    testcase = tmp_path / "tc"
    testcase.mkdir()
    (testcase / "testcase.json").write_text(
        json.dumps(
            {
                "format": "inferref-testcase",
                "format_version": "0.1",
                "name": "hash-only",
                "inputs": [],
                "outputs": [
                    {
                        "name": "out",
                        "value_id": 7,
                        "payload": None,
                        "capture": {"mode": "hash"},
                    }
                ],
                "values": [{"id": 7}],
            }
        ),
        encoding="utf-8",
    )

    report = compare_testcase(testcase, tmp_path / "engine")

    assert report.status == STATUS_FAIL
    assert report.first_failure.status == STATUS_MISSING
    assert report.first_failure.value_id == 7
    assert "capture mode hash" in report.first_failure.message


def test_first_failure_stops_early(tmp_path: Path) -> None:
    good = np.zeros(3, dtype=np.float32)
    _make_testcase(tmp_path / "tc", {"a": good, "b": good, "c": good})
    # Break the first output only.
    codec.write_array(tmp_path / "engine" / "a.irtensor", good + 1.0)
    codec.write_array(tmp_path / "engine" / "b.irtensor", good)
    codec.write_array(tmp_path / "engine" / "c.irtensor", good)

    full = compare_testcase(tmp_path / "tc", tmp_path / "engine")
    assert len(full.comparisons) == 3
    assert not full.stopped_early

    early = compare_testcase(tmp_path / "tc", tmp_path / "engine", first_failure=True)
    assert len(early.comparisons) == 1
    assert early.stopped_early
    assert early.first_failure.name == "a"


def test_engine_manifest_is_honoured(tmp_path: Path) -> None:
    """SPEC §22: an engine may declare its outputs rather than match filenames."""
    values = np.ones(4, dtype=np.float32)
    _make_testcase(tmp_path / "tc", {"out": values})
    codec.write_array(tmp_path / "engine" / "weird_name.bin", values)
    (tmp_path / "engine" / "manifest.json").write_text(
        json.dumps({"outputs": [{"name": "out", "payload": "weird_name.bin"}]})
    )

    report = compare_testcase(tmp_path / "tc", tmp_path / "engine")
    assert report.status == STATUS_PASS


def test_testcase_reference_payload_cannot_escape_directory(tmp_path: Path) -> None:
    testcase = tmp_path / "tc"
    _make_testcase(testcase, {"out": np.ones(4, dtype=np.float32)})
    manifest_path = testcase / "testcase.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["payload"] = "../outside.irtensor"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes root"):
        compare_testcase(testcase, tmp_path / "engine")


def test_engine_manifest_payload_cannot_escape_directory(tmp_path: Path) -> None:
    values = np.ones(4, dtype=np.float32)
    _make_testcase(tmp_path / "tc", {"out": values})
    codec.write_array(tmp_path / "outside.irtensor", values)
    engine = tmp_path / "engine"
    engine.mkdir()
    (engine / "manifest.json").write_text(
        json.dumps({"outputs": [{"name": "out", "payload": "../outside.irtensor"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes root"):
        compare_testcase(tmp_path / "tc", engine)


def test_engine_manifest_symlink_cannot_escape_directory(tmp_path: Path) -> None:
    values = np.ones(4, dtype=np.float32)
    _make_testcase(tmp_path / "tc", {"out": values})
    engine = tmp_path / "engine"
    engine.mkdir()
    outside = tmp_path / "outside-manifest.json"
    outside.write_text(json.dumps({"outputs": []}), encoding="utf-8")
    try:
        (engine / "manifest.json").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="escapes root"):
        compare_testcase(tmp_path / "tc", engine)


def test_output_name_cannot_escape_engine_directory_fallback(tmp_path: Path) -> None:
    values = np.ones(4, dtype=np.float32)
    _make_testcase(tmp_path / "tc", {"out": values})
    manifest_path = tmp_path / "tc" / "testcase.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][0]["name"] = "../outside"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    codec.write_array(tmp_path / "outside.irtensor", values)

    with pytest.raises(ValueError, match="escapes root"):
        compare_testcase(tmp_path / "tc", tmp_path / "engine")


def test_report_contains_spec_appendix_b_fields(tmp_path: Path) -> None:
    reference = np.zeros((2, 2), dtype=np.float32)
    actual = reference.copy()
    actual[1, 1] = 0.25
    _make_testcase(tmp_path / "tc", {"out": reference})
    codec.write_array(tmp_path / "engine" / "out.irtensor", actual)

    report = compare_testcase(tmp_path / "tc", tmp_path / "engine")
    text = render_report(report, verbose=True)

    for expected in (
        "First divergence:",
        "max_abs_error:",
        "cosine_similarity:",
        "First mismatching element:",
        "index:     [1, 1]",
        "Result: FAIL",
    ):
        assert expected in text, text

    payload = report.to_dict()
    assert payload["status"] == "fail"
    assert payload["first_failure"]["metrics"]["max_abs_error"] == pytest.approx(0.25)
    assert payload["first_failure"]["metrics"]["first_mismatch"]["index"] == [1, 1]
    # The JSON form must be serialisable for agent/CI consumption (SPEC §42).
    json.dumps(payload)
