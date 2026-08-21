"""Object detection multi-output comparator example (SPEC §7.4).

Demonstrates a custom ComparatorPlugin that receives a multi-output ArtifactSet
containing (boxes, scores, classes) and evaluates task-level semantic equivalence.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from inferref.comparators.protocol import ArtifactSet, ComparatorResult
from inferref.tensor import codec

OBJECT_DETECTION_COMPARATOR_ID = "vision/object-detection/v1"

_ALLOWED_CONFIG_KEYS = {
    "box_format",
    "matching",
    "min_iou",
    "class_exact",
    "score_atol",
    "ordering",
}


def _compute_iou(box1: np.ndarray, box2: np.ndarray, box_format: str = "xyxy") -> float:
    """Compute Intersection-over-Union (IoU) between two 4-element bounding boxes."""
    if box_format == "xywh":
        # convert [x, y, w, h] to [x1, y1, x2, y2]
        b1 = np.array([box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]], dtype=float)
        b2 = np.array([box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]], dtype=float)
    else:
        b1 = np.array(box1, dtype=float)
        b2 = np.array(box2, dtype=float)

    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = max(0.0, b1[2] - b1[0]) * max(0.0, b1[3] - b1[1])
    area2 = max(0.0, b2[2] - b2[0]) * max(0.0, b2[3] - b2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


class ObjectDetectionComparator:
    """Multi-output comparator for object detection predictions."""

    id: str = OBJECT_DETECTION_COMPARATOR_ID

    def validate_config(self, config: dict[str, Any] | None = None) -> None:
        """Validate object detection comparator configuration statically."""
        if config is None:
            return
        if not isinstance(config, dict):
            raise ValueError(f"config must be a dictionary, got {type(config).__name__}")

        unknown = set(config) - _ALLOWED_CONFIG_KEYS
        if unknown:
            raise ValueError(
                f"unknown object detection comparator config key(s): {sorted(unknown)}; "
                f"allowed keys: {sorted(_ALLOWED_CONFIG_KEYS)}"
            )

        if "box_format" in config:
            val = config["box_format"]
            if val not in ("xyxy", "xywh"):
                raise ValueError(f"box_format must be 'xyxy' or 'xywh', got {val!r}")

        if "matching" in config:
            val = config["matching"]
            if val not in ("iou", "greedy", "hungarian"):
                raise ValueError(f"matching must be 'iou', 'greedy', or 'hungarian', got {val!r}")

        if "min_iou" in config:
            val = config["min_iou"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                raise ValueError(f"min_iou must be a float between 0.0 and 1.0, got {val!r}")

        if "class_exact" in config:
            val = config["class_exact"]
            if not isinstance(val, bool):
                raise ValueError(f"class_exact must be a boolean, got {val!r}")

        if "score_atol" in config:
            val = config["score_atol"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or val < 0:
                raise ValueError(f"score_atol must be a non-negative float, got {val!r}")

        if "ordering" in config:
            val = config["ordering"]
            if val not in ("ignore", "exact"):
                raise ValueError(f"ordering must be 'ignore' or 'exact', got {val!r}")

    def compare(
        self,
        reference: ArtifactSet,
        actual: ArtifactSet,
        config: dict[str, Any] | None = None,
    ) -> ComparatorResult:
        """Jointly evaluate (boxes, scores, classes) artifact sets."""
        self.validate_config(config)
        cfg = config or {}

        box_format = str(cfg.get("box_format", "xyxy"))
        min_iou = float(cfg.get("min_iou", 0.99))
        class_exact = bool(cfg.get("class_exact", True))
        score_atol = float(cfg.get("score_atol", 0.002))
        ordering = str(cfg.get("ordering", "ignore"))

        required_roles = ("boxes", "scores", "classes")
        for role in required_roles:
            if role not in reference:
                raise ValueError(f"reference artifact set missing required role {role!r}")
            if role not in actual:
                return ComparatorResult(
                    status="fail",
                    comparator=self.id,
                    metrics={},
                    diagnostics=[
                        {
                            "output": role,
                            "code": "missing_output_role",
                            "message": f"candidate missing required role {role!r}",
                        }
                    ],
                    first_failure={
                        "output": role,
                        "message": f"candidate missing required role {role!r}",
                    },
                )

        ref_boxes_view = codec.read(reference["boxes"].path)
        ref_scores_view = codec.read(reference["scores"].path)
        ref_classes_view = codec.read(reference["classes"].path)

        act_boxes_view = codec.read(actual["boxes"].path)
        act_scores_view = codec.read(actual["scores"].path)
        act_classes_view = codec.read(actual["classes"].path)

        ref_boxes = ref_boxes_view.as_float32()
        ref_scores = ref_scores_view.as_float32()
        ref_classes = ref_classes_view.data.reshape(ref_classes_view.shape)

        act_boxes = act_boxes_view.as_float32()
        act_scores = act_scores_view.as_float32()
        act_classes = act_classes_view.data.reshape(act_classes_view.shape)

        ref_n = len(ref_boxes)
        act_n = len(act_boxes)

        if ref_n != len(ref_scores) or ref_n != len(ref_classes):
            raise ValueError("reference boxes, scores, and classes must have matching record counts")
        if act_n != len(act_scores) or act_n != len(act_classes):
            raise ValueError("candidate boxes, scores, and classes must have matching record counts")

        matched_pairs: list[tuple[int, int, float]] = []
        diagnostics: list[dict[str, Any]] = []

        if ordering == "exact":
            if ref_n != act_n:
                diagnostics.append(
                    {
                        "code": "count_mismatch",
                        "message": f"exact ordering count mismatch: reference has {ref_n}, candidate has {act_n}",
                    }
                )
            limit = min(ref_n, act_n)
            for i in range(limit):
                iou = _compute_iou(ref_boxes[i], act_boxes[i], box_format=box_format)
                cls_match = bool(ref_classes[i] == act_classes[i]) if class_exact else True
                score_diff = abs(float(ref_scores[i]) - float(act_scores[i]))
                score_match = score_diff <= score_atol

                if iou >= min_iou and cls_match and score_match:
                    matched_pairs.append((i, i, iou))
                else:
                    diagnostics.append(
                        {
                            "index": i,
                            "code": "detection_mismatch",
                            "iou": iou,
                            "class_match": cls_match,
                            "score_diff": score_diff,
                            "message": f"detection at index {i} failed criteria (iou={iou:.4f}, class={cls_match}, score_diff={score_diff:.4f})",
                        }
                    )
        else:
            # Bipartite / greedy matching on IoU
            matched_ref: set[int] = set()
            matched_act: set[int] = set()

            # Build all potential candidate matches
            potential_matches: list[tuple[float, float, int, int]] = []
            for r_idx in range(ref_n):
                for a_idx in range(act_n):
                    cls_match = bool(ref_classes[r_idx] == act_classes[a_idx]) if class_exact else True
                    if not cls_match:
                        continue
                    score_diff = abs(float(ref_scores[r_idx]) - float(act_scores[a_idx]))
                    if score_diff > score_atol:
                        continue
                    iou = _compute_iou(ref_boxes[r_idx], act_boxes[a_idx], box_format=box_format)
                    if iou >= min_iou:
                        # sort key: highest iou first, then lowest score_diff
                        potential_matches.append((iou, -score_diff, r_idx, a_idx))

            potential_matches.sort(key=lambda item: (item[0], item[1]), reverse=True)

            for iou, _neg_score_diff, r_idx, a_idx in potential_matches:
                if r_idx not in matched_ref and a_idx not in matched_act:
                    matched_ref.add(r_idx)
                    matched_act.add(a_idx)
                    matched_pairs.append((r_idx, a_idx, iou))

            unmatched_ref = set(range(ref_n)) - matched_ref
            unmatched_act = set(range(act_n)) - matched_act

            for r_idx in sorted(unmatched_ref):
                diagnostics.append(
                    {
                        "reference_index": r_idx,
                        "code": "unmatched_reference_detection",
                        "message": f"unmatched reference detection at index {r_idx}",
                    }
                )
            for a_idx in sorted(unmatched_act):
                diagnostics.append(
                    {
                        "candidate_index": a_idx,
                        "code": "unmatched_candidate_detection",
                        "message": f"unmatched candidate detection at index {a_idx}",
                    }
                )

        matched_count = len(matched_pairs)
        min_iou_found = min((p[2] for p in matched_pairs), default=0.0) if matched_pairs else 0.0

        metrics = {
            "reference_count": ref_n,
            "actual_count": act_n,
            "matched": matched_count,
            "min_iou": round(float(min_iou_found), 4),
            "unmatched_reference": ref_n - matched_count,
            "unmatched_actual": act_n - matched_count,
        }

        passed = (matched_count == ref_n == act_n)
        first_failure = None
        if not passed:
            first_failure = {
                "output": "boxes",
                "message": (
                    f"unmatched detection ({matched_count}/{ref_n} matched, candidate count {act_n})"
                    if diagnostics
                    else "object detection mismatch"
                ),
                "metrics": metrics,
            }

        return ComparatorResult(
            status="pass" if passed else "fail",
            comparator=self.id,
            metrics=metrics,
            diagnostics=diagnostics,
            first_failure=first_failure,
        )
