"""Tests for multi-output comparators and object detection fixture (SPEC §7.4, C4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.comparators.object_detection import (
    OBJECT_DETECTION_COMPARATOR_ID,
    ObjectDetectionComparator,
)
from inferref.comparators import (
    Artifact,
    ArtifactSet,
    get_comparator,
    register_builtin_comparator,
    run_comparator,
)
from inferref.comparators import registry as comparator_registry
from inferref.tensor import codec


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    comparator_registry._reset_registry()
    yield
    comparator_registry._reset_registry()


def _write_irtensor(path: Path, array: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec.write_array(path, array)
    return path


def _create_detection_artifacts(
    root: Path,
    boxes: list[list[float]],
    scores: list[float],
    classes: list[int],
) -> ArtifactSet:
    boxes_path = _write_irtensor(root / "boxes.irtensor", np.array(boxes, dtype=np.float32))
    scores_path = _write_irtensor(root / "scores.irtensor", np.array(scores, dtype=np.float32))
    classes_path = _write_irtensor(root / "classes.irtensor", np.array(classes, dtype=np.int32))
    return {
        "boxes": Artifact("boxes", boxes_path),
        "scores": Artifact("scores", scores_path),
        "classes": Artifact("classes", classes_path),
    }


# -- Config Validation ------------------------------------------------------


def test_object_detection_config_validation() -> None:
    comp = ObjectDetectionComparator()
    # Valid configs
    comp.validate_config(None)
    comp.validate_config({})
    comp.validate_config({
        "box_format": "xyxy",
        "matching": "iou",
        "min_iou": 0.95,
        "class_exact": True,
        "score_atol": 0.005,
        "ordering": "ignore",
    })

    # Invalid non-dict
    with pytest.raises(ValueError, match="must be a dictionary"):
        comp.validate_config("invalid")

    # Unknown key
    with pytest.raises(ValueError, match="unknown object detection comparator config key"):
        comp.validate_config({"threshold": 0.5})

    # Invalid box_format
    with pytest.raises(ValueError, match="box_format must be 'xyxy' or 'xywh'"):
        comp.validate_config({"box_format": "corners"})

    # Invalid min_iou
    with pytest.raises(ValueError, match="min_iou must be a float between 0.0 and 1.0"):
        comp.validate_config({"min_iou": 1.5})
    with pytest.raises(ValueError, match="min_iou must be a float between 0.0 and 1.0"):
        comp.validate_config({"min_iou": -0.1})

    # Invalid score_atol
    with pytest.raises(ValueError, match="score_atol must be a non-negative float"):
        comp.validate_config({"score_atol": -0.01})

    # Invalid ordering
    with pytest.raises(ValueError, match="ordering must be 'ignore' or 'exact'"):
        comp.validate_config({"ordering": "strict"})


# -- Comparison Evaluation --------------------------------------------------


def test_object_detection_perfect_match(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    boxes = [[10.0, 20.0, 100.0, 200.0], [50.0, 50.0, 150.0, 150.0]]
    scores = [0.95, 0.88]
    classes = [1, 2]

    ref_set = _create_detection_artifacts(tmp_path / "ref", boxes, scores, classes)
    act_set = _create_detection_artifacts(tmp_path / "act", boxes, scores, classes)

    result = comp.compare(ref_set, act_set, config={"min_iou": 0.99, "score_atol": 0.001})
    assert result.passed is True
    assert result.status == "pass"
    assert result.comparator == OBJECT_DETECTION_COMPARATOR_ID
    assert result.metrics["reference_count"] == 2
    assert result.metrics["actual_count"] == 2
    assert result.metrics["matched"] == 2
    assert result.metrics["min_iou"] == 1.0
    assert result.first_failure is None


def test_object_detection_within_tolerance(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    ref_boxes = [[10.0, 20.0, 100.0, 200.0]]
    act_boxes = [[10.5, 20.0, 100.0, 200.0]]  # Slight shift, IoU > 0.98

    ref_set = _create_detection_artifacts(tmp_path / "ref", ref_boxes, [0.95], [1])
    act_set = _create_detection_artifacts(tmp_path / "act", act_boxes, [0.951], [1])

    result = comp.compare(ref_set, act_set, config={"min_iou": 0.95, "score_atol": 0.005})
    assert result.passed is True
    assert result.status == "pass"
    assert result.metrics["matched"] == 1
    assert result.metrics["min_iou"] >= 0.95


def test_object_detection_iou_below_threshold(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    ref_boxes = [[10.0, 20.0, 100.0, 200.0]]
    act_boxes = [[50.0, 80.0, 140.0, 260.0]]  # Significant shift, IoU < 0.5

    ref_set = _create_detection_artifacts(tmp_path / "ref", ref_boxes, [0.95], [1])
    act_set = _create_detection_artifacts(tmp_path / "act", act_boxes, [0.95], [1])

    result = comp.compare(ref_set, act_set, config={"min_iou": 0.90})
    assert result.passed is False
    assert result.status == "fail"
    assert result.metrics["matched"] == 0
    assert result.first_failure is not None
    assert result.first_failure["output"] == "boxes"


def test_object_detection_class_mismatch(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    boxes = [[10.0, 20.0, 100.0, 200.0]]
    scores = [0.95]

    ref_set = _create_detection_artifacts(tmp_path / "ref", boxes, scores, [1])
    act_set = _create_detection_artifacts(tmp_path / "act", boxes, scores, [2])  # class 2 instead of 1

    # Exact class required: should fail
    res_exact = comp.compare(ref_set, act_set, config={"class_exact": True})
    assert res_exact.passed is False
    assert res_exact.metrics["matched"] == 0

    # Class exact disabled: should pass
    res_inexact = comp.compare(ref_set, act_set, config={"class_exact": False})
    assert res_inexact.passed is True
    assert res_inexact.metrics["matched"] == 1


def test_object_detection_score_tolerance(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    boxes = [[10.0, 20.0, 100.0, 200.0]]

    ref_set = _create_detection_artifacts(tmp_path / "ref", boxes, [0.95], [1])
    act_set = _create_detection_artifacts(tmp_path / "act", boxes, [0.85], [1])  # diff = 0.10

    # Strict score atol: fails
    res_strict = comp.compare(ref_set, act_set, config={"score_atol": 0.01})
    assert res_strict.passed is False
    assert res_strict.metrics["matched"] == 0

    # Generous score atol: passes
    res_loose = comp.compare(ref_set, act_set, config={"score_atol": 0.15})
    assert res_loose.passed is True
    assert res_loose.metrics["matched"] == 1


def test_object_detection_ordering_ignore_vs_exact(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    box_a = [10.0, 20.0, 100.0, 200.0]
    box_b = [300.0, 300.0, 400.0, 400.0]

    # Reference order: A, B
    ref_set = _create_detection_artifacts(tmp_path / "ref", [box_a, box_b], [0.9, 0.8], [1, 2])
    # Actual order: B, A (permuted)
    act_set = _create_detection_artifacts(tmp_path / "act", [box_b, box_a], [0.8, 0.9], [2, 1])

    # ordering="ignore" (default) -> should match out-of-order pairs and pass
    res_ignore = comp.compare(ref_set, act_set, config={"ordering": "ignore"})
    assert res_ignore.passed is True
    assert res_ignore.metrics["matched"] == 2

    # ordering="exact" -> pairwise compare fails due to permutation
    res_exact = comp.compare(ref_set, act_set, config={"ordering": "exact"})
    assert res_exact.passed is False
    assert res_exact.metrics["matched"] == 0


def test_object_detection_count_mismatch(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    box_a = [10.0, 20.0, 100.0, 200.0]
    box_b = [300.0, 300.0, 400.0, 400.0]

    ref_set = _create_detection_artifacts(tmp_path / "ref", [box_a, box_b], [0.9, 0.8], [1, 2])
    act_set = _create_detection_artifacts(tmp_path / "act", [box_a], [0.9], [1])  # Missing box B

    result = comp.compare(ref_set, act_set)
    assert result.passed is False
    assert result.metrics["reference_count"] == 2
    assert result.metrics["actual_count"] == 1
    assert result.metrics["matched"] == 1
    assert result.metrics["unmatched_reference"] == 1
    assert result.metrics["unmatched_actual"] == 0


def test_object_detection_missing_role_short_circuits(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    ref_set = _create_detection_artifacts(tmp_path / "ref", [[1.0, 2.0, 3.0, 4.0]], [0.9], [1])

    # act_set only contains boxes and scores, classes is missing
    act_set: ArtifactSet = {
        "boxes": Artifact("boxes", tmp_path / "ref" / "boxes.irtensor"),
        "scores": Artifact("scores", tmp_path / "ref" / "scores.irtensor"),
    }

    result = run_comparator(comp, ref_set, act_set, required_roles=["boxes", "scores", "classes"])
    assert result.passed is False
    assert result.status == "fail"
    assert result.first_failure["output"] == "classes"
    assert "engine produced no output for role 'classes'" in result.first_failure["message"]


def test_object_detection_registered_and_run_by_id(tmp_path: Path) -> None:
    comp = ObjectDetectionComparator()
    register_builtin_comparator(comp)

    boxes = [[10.0, 20.0, 100.0, 200.0]]
    ref_set = _create_detection_artifacts(tmp_path / "ref", boxes, [0.9], [1])
    act_set = _create_detection_artifacts(tmp_path / "act", boxes, [0.9], [1])

    result = run_comparator(OBJECT_DETECTION_COMPARATOR_ID, ref_set, act_set)
    assert result.passed is True
    assert result.status == "pass"
    assert result.comparator == OBJECT_DETECTION_COMPARATOR_ID


def test_e2e_compare_testcase_with_custom_comparator(tmp_path: Path) -> None:
    import json
    from inferref.cli.main import EXIT_OK, main
    from inferref.compare import compare_testcase
    from inferref.testcase.requirements import derive_requirements

    comp = ObjectDetectionComparator()
    register_builtin_comparator(comp)

    tc_dir = tmp_path / "tc"
    eng_dir = tmp_path / "eng"
    boxes = [[10.0, 20.0, 100.0, 200.0]]
    ref_set = _create_detection_artifacts(tc_dir, boxes, [0.9], [1])
    _create_detection_artifacts(eng_dir, boxes, [0.9], [1])

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.3",
        "inferref_version": "0.9.0",
        "name": "detection_tc",
        "region_name": "detection_region",
        "inputs": [],
        "outputs": [
            {"name": "boxes", "value_id": None, "payload": "boxes.irtensor", **codec.read(ref_set["boxes"].path).to_metadata()},
            {"name": "scores", "value_id": None, "payload": "scores.irtensor", **codec.read(ref_set["scores"].path).to_metadata()},
            {"name": "classes", "value_id": None, "payload": "classes.irtensor", **codec.read(ref_set["classes"].path).to_metadata()},
        ],
        "nodes": [],
        "values": [],
        "comparison": {
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": OBJECT_DETECTION_COMPARATOR_ID,
            "config": {"min_iou": 0.9},
        },
    }
    manifest["requirements"] = derive_requirements(manifest)
    (tc_dir / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")

    # 1. compare_testcase direct
    report = compare_testcase(tc_dir, eng_dir)
    assert report.status == "pass"
    assert report.comparator is not None
    assert report.comparator["comparator"] == OBJECT_DETECTION_COMPARATOR_ID
    assert report.comparator["metrics"]["matched"] == 1

    # 2. CLI compare
    assert main(["compare", str(tc_dir), str(eng_dir), "--json"]) == EXIT_OK

    # 3. CLI agent compare
    assert main(["agent", "compare", str(tc_dir), str(eng_dir), "--json"]) == EXIT_OK


def test_e2e_per_output_comparator_and_tolerance_dispatch(tmp_path: Path) -> None:
    import json
    from inferref.compare import compare_testcase
    from inferref.testcase.requirements import derive_requirements

    comp = ObjectDetectionComparator()
    register_builtin_comparator(comp)

    tc_dir = tmp_path / "tc_per_output"
    eng_dir = tmp_path / "eng_per_output"
    tc_dir.mkdir(parents=True, exist_ok=True)
    eng_dir.mkdir(parents=True, exist_ok=True)

    # Output 1: boxes (vision comparator)
    boxes = [[10.0, 20.0, 100.0, 200.0]]
    ref_set = _create_detection_artifacts(tc_dir, boxes, [0.9], [1])
    _create_detection_artifacts(eng_dir, boxes, [0.9], [1])

    # Output 2: y (numeric with tight tolerance 1e-5 override vs actual diff 1e-3)
    y_ref = np.array([1.0, 2.0], dtype=np.float32)
    y_act_mismatch = np.array([1.001, 2.001], dtype=np.float32)
    y_path = _write_irtensor(tc_dir / "y.irtensor", y_ref)
    _write_irtensor(eng_dir / "y.irtensor", y_act_mismatch)

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.3",
        "inferref_version": "0.9.0",
        "name": "mixed_tc",
        "region_name": "mixed_region",
        "inputs": [],
        "outputs": [
            {"name": "boxes", "value_id": None, "payload": "boxes.irtensor", **codec.read(ref_set["boxes"].path).to_metadata()},
            {"name": "scores", "value_id": None, "payload": "scores.irtensor", **codec.read(ref_set["scores"].path).to_metadata()},
            {"name": "classes", "value_id": None, "payload": "classes.irtensor", **codec.read(ref_set["classes"].path).to_metadata()},
            {"name": "y", "value_id": None, "payload": "y.irtensor", **codec.read(y_path).to_metadata()},
        ],
        "nodes": [],
        "values": [],
        "comparison": {
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": "tensor/numeric/v1",
            "outputs": {
                "boxes": {
                    "comparator": OBJECT_DETECTION_COMPARATOR_ID,
                    "config": {"min_iou": 0.9},
                },
                "y": {
                    "config": {"atol": 1e-5, "rtol": 1e-5},
                },
            },
        },
    }
    manifest["requirements"] = derive_requirements(manifest)
    (tc_dir / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")

    # Tight tolerance on y fails while boxes passes via custom comparator
    report = compare_testcase(tc_dir, eng_dir)
    assert report.status == "fail"
    y_comp = next(c for c in report.comparisons if c.name == "y")
    assert y_comp.status == "fail"
    boxes_comp = next(c for c in report.comparisons if c.name == "boxes")
    assert boxes_comp.status == "pass"

    # Now make y within tolerance -> all pass
    _write_irtensor(eng_dir / "y.irtensor", np.array([1.000001, 2.000001], dtype=np.float32))
    report_pass = compare_testcase(tc_dir, eng_dir)
    assert report_pass.status == "pass"


def test_e2e_comparator_exception_maps_to_error_status(tmp_path: Path) -> None:
    import json
    from inferref.comparators.protocol import ComparatorPlugin, ComparatorResult
    from inferref.compare.compare import STATUS_ERROR, compare_testcase
    from inferref.testcase.requirements import derive_requirements

    class CrashingComparator:
        @property
        def id(self) -> str:
            return "testing/crasher/v1"

        def validate_config(self, config: dict[str, Any] | None) -> None:
            pass

        def compare(self, reference: ArtifactSet, actual: ArtifactSet, config: dict[str, Any] | None = None) -> ComparatorResult:
            raise RuntimeError("unexpected computational failure inside comparator")

    register_builtin_comparator(CrashingComparator())

    tc_dir = tmp_path / "tc_crash"
    eng_dir = tmp_path / "eng_crash"
    boxes = [[10.0, 20.0, 100.0, 200.0]]
    ref_set = _create_detection_artifacts(tc_dir, boxes, [0.9], [1])
    _create_detection_artifacts(eng_dir, boxes, [0.9], [1])

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.3",
        "inferref_version": "0.9.0",
        "name": "crash_tc",
        "region_name": "crash_region",
        "inputs": [],
        "outputs": [
            {"name": "boxes", "value_id": None, "payload": "boxes.irtensor", **codec.read(ref_set["boxes"].path).to_metadata()},
        ],
        "nodes": [],
        "values": [],
        "comparison": {
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": "testing/crasher/v1",
        },
    }
    manifest["requirements"] = derive_requirements(manifest)
    (tc_dir / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_testcase(tc_dir, eng_dir)
    assert report.status == STATUS_ERROR
    assert report.comparator is not None
    assert report.comparator["status"] == STATUS_ERROR
    assert len(report.comparisons) == 1
    assert report.comparisons[0].status == STATUS_ERROR
    assert "unexpected computational failure inside comparator" in report.comparisons[0].message


def test_case_a_missing_output_reporting_matches_numeric_shape(tmp_path: Path) -> None:
    import json
    from inferref.compare.compare import STATUS_MISSING, STATUS_PASS, compare_testcase
    from inferref.testcase.requirements import derive_requirements

    comp = ObjectDetectionComparator()
    register_builtin_comparator(comp)

    tc_dir = tmp_path / "tc_missing_role"
    eng_dir = tmp_path / "eng_missing_role"
    boxes = [[10.0, 20.0, 100.0, 200.0]]
    ref_set = _create_detection_artifacts(tc_dir, boxes, [0.9], [1])

    # Engine produces boxes and scores, but misses classes
    _write_irtensor(eng_dir / "boxes.irtensor", np.array(boxes, dtype=np.float32))
    _write_irtensor(eng_dir / "scores.irtensor", np.array([0.9], dtype=np.float32))

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.3",
        "inferref_version": "0.9.0",
        "name": "missing_role_tc",
        "region_name": "det_region",
        "inputs": [],
        "outputs": [
            {"name": "boxes", "value_id": None, "payload": "boxes.irtensor", **codec.read(ref_set["boxes"].path).to_metadata()},
            {"name": "scores", "value_id": None, "payload": "scores.irtensor", **codec.read(ref_set["scores"].path).to_metadata()},
            {"name": "classes", "value_id": None, "payload": "classes.irtensor", **codec.read(ref_set["classes"].path).to_metadata()},
        ],
        "nodes": [],
        "values": [],
        "comparison": {
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": OBJECT_DETECTION_COMPARATOR_ID,
        },
    }
    manifest["requirements"] = derive_requirements(manifest)
    (tc_dir / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_testcase(tc_dir, eng_dir)
    assert report.status == "fail"
    assert report.missing_count == 1
    assert report.passed_count == 2
    assert report.failed_count == 0

    classes_comp = next(c for c in report.comparisons if c.name == "classes")
    assert classes_comp.status == STATUS_MISSING
    assert "not found" in classes_comp.message

    boxes_comp = next(c for c in report.comparisons if c.name == "boxes")
    assert boxes_comp.status == STATUS_PASS


def test_case_a_all_roles_present_semantic_failure_reports_fail(tmp_path: Path) -> None:
    """N1 regression test: all output roles exist, but comparator detects semantic failure (IoU=0)."""
    import json
    from inferref.compare.compare import STATUS_FAIL, STATUS_PASS, compare_testcase
    from inferref.testcase.requirements import derive_requirements

    comp = ObjectDetectionComparator()
    register_builtin_comparator(comp)

    tc_dir = tmp_path / "tc_semantic_fail"
    eng_dir = tmp_path / "eng_semantic_fail"
    ref_boxes = [[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    ref_set = _create_detection_artifacts(tc_dir, ref_boxes, [0.9, 0.8], [1, 2])

    # Engine produces completely disjoint boxes
    eng_boxes = [[500.0, 500.0, 510.0, 510.0], [600.0, 600.0, 610.0, 610.0]]
    _write_irtensor(eng_dir / "boxes.irtensor", np.array(eng_boxes, dtype=np.float32))
    _write_irtensor(eng_dir / "scores.irtensor", np.array([0.9, 0.8], dtype=np.float32))
    _write_irtensor(eng_dir / "classes.irtensor", np.array([1, 2], dtype=np.int64))

    manifest = {
        "format": "inferref-testcase",
        "format_version": "0.3",
        "inferref_version": "0.9.0",
        "name": "semantic_fail_tc",
        "region_name": "det_region",
        "inputs": [],
        "outputs": [
            {"name": "boxes", "value_id": None, "payload": "boxes.irtensor", **codec.read(ref_set["boxes"].path).to_metadata()},
            {"name": "scores", "value_id": None, "payload": "scores.irtensor", **codec.read(ref_set["scores"].path).to_metadata()},
            {"name": "classes", "value_id": None, "payload": "classes.irtensor", **codec.read(ref_set["classes"].path).to_metadata()},
        ],
        "nodes": [],
        "values": [],
        "comparison": {
            "format": "inferref-comparison",
            "format_version": "0.1",
            "comparator": OBJECT_DETECTION_COMPARATOR_ID,
        },
    }
    manifest["requirements"] = derive_requirements(manifest)
    (tc_dir / "testcase.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = compare_testcase(tc_dir, eng_dir)
    assert report.status == "fail"
    assert report.failed_count >= 1
    assert report.first_failure is not None
    assert report.first_failure.status == "fail"
    assert report.first_failure.name == "boxes"

    rep_dict = report.to_dict()
    assert rep_dict["status"] == "fail"
    assert rep_dict["summary"]["failed"] >= 1
    assert rep_dict["first_failure"] is not None
    assert rep_dict["first_failure"]["status"] == "fail"




