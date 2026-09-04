"""Tests for Causal Narrowing and Failure-Domain Classification (0.10 L5)."""

from __future__ import annotations

from inferref.alignment.boundaries import inventory_boundaries
from inferref.alignment.classification import classify_failure_domain
from inferref.alignment.evidence import (
    build_comparison_evidence,
    build_manual_note_evidence,
)
from inferref.alignment.narrowing import compute_narrowing
from inferref.alignment.schema import AlignmentEvent, Evidence
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _mock_provenance(engine_rev: str = "rev-1") -> dict:
    return {
        "trace_metadata_sha256": DIGEST_A,
        "testcase": {"path": "testcases/tc1", "tree_sha256": DIGEST_C},
        "run_record": {"path": "runs/run1/inferref-run.json", "sha256": DIGEST_C},
        "adapter_normalized_sha256": DIGEST_B,
        "engine_identity": {"kind": "git", "revision": engine_rev, "dirty": False},
        "effective_comparison_sha256": DIGEST_A,
    }


def _package() -> TracePackage:
    graph = Graph()
    graph.values = [
        TensorValueRecord(
            id=value_id,
            shape=(2,),
            producer=producer,
            capture=CaptureInfo(mode="full", payload=f"v{value_id}.irtensor"),
        )
        for value_id, producer in ((1, None), (2, 10), (3, 11), (4, 12))
    ]
    graph.operators = [
        OperatorRecord(
            id=10,
            execution_index=0,
            op="one",
            result=TensorRef(2),
            positional_args=(TensorRef(1),),
            module_stack=(100,),
        ),
        OperatorRecord(
            id=11,
            execution_index=1,
            op="two",
            result=TensorRef(3),
            positional_args=(TensorRef(2),),
            module_stack=(100, 101),
        ),
        OperatorRecord(
            id=12,
            execution_index=2,
            op="three",
            result=TensorRef(4),
            positional_args=(TensorRef(3),),
            module_stack=(100, 102),
        ),
    ]
    graph.reindex()
    return TracePackage(
        manifest=Manifest(),
        graph=graph,
        modules=[
            ModuleRecord(id=100, path="model"),
            ModuleRecord(id=101, path="model.left"),
            ModuleRecord(id=102, path="model.right"),
        ],
        regions=[
            RegionRecord(id=1, name="parent", node_ids=(10, 11, 12)),
            RegionRecord(id=2, name="left", node_ids=(10, 11)),
            RegionRecord(id=3, name="right", node_ids=(12,)),
        ],
    )


def test_narrowing_identifies_smallest_confirmed_failure_and_candidates() -> None:
    pkg = _package()
    inv = inventory_boundaries(pkg)

    # 1. Only parent region:1 has failing evidence
    ev_parent_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={"status": "fail", "first_failure": {"output": "out", "value_id": 3}},
    )
    state = compute_narrowing(inv, {"region:1": [ev_parent_fail]}, package_or_graph=pkg)
    assert state.smallest_confirmed_failure == "region:1"
    # region:2 produces value 3, so it is in next candidates
    assert "region:2" in state.next_candidates
    assert state.interaction_gap is False

    # 2. Child region:2 also fails -> smallest confirmed failure narrows to region:2
    ev_child_fail = build_comparison_evidence(
        subject={"boundary": "region:2", "name": "left"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={"status": "fail"},
    )
    state2 = compute_narrowing(
        inv,
        {"region:1": [ev_parent_fail], "region:2": [ev_child_fail]},
        package_or_graph=pkg,
    )
    assert state2.smallest_confirmed_failure == "region:2"
    assert "region:2" in state2.failing_descendants


def test_narrowing_reports_interaction_gap_when_children_pass() -> None:
    pkg = _package()
    inv = inventory_boundaries(pkg)

    ev_parent_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={"status": "fail"},
    )
    ev_left_pass = build_comparison_evidence(
        subject={"boundary": "region:2", "name": "left"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={"status": "pass"},
    )
    ev_right_pass = build_comparison_evidence(
        subject={"boundary": "region:3", "name": "right"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={"status": "pass"},
    )

    state = compute_narrowing(
        inv,
        {
            "region:1": [ev_parent_fail],
            "region:2": [ev_left_pass],
            "region:3": [ev_right_pass],
        },
        package_or_graph=pkg,
    )
    # Both direct children of region:1 passed, yet region:1 failed!
    assert state.smallest_confirmed_failure == "region:1"
    assert state.interaction_gap is True
    assert "interaction" in state.reason


def test_classification_layout_rule() -> None:
    ev_layout = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={
            "status": "fail",
            "first_failure": {"output": "out", "max_abs_error": 0.0},
            "diagnostics": {"layout_mismatch": True},
        },
    )
    candidates, _decisions = classify_failure_domain("region:1", [ev_layout])
    assert len(candidates) == 1
    assert candidates[0].domain == "layout"
    assert candidates[0].confidence == 0.95
    assert candidates[0].rule == "logical_pass_strict_layout_fails/v1"


def test_classification_implementation_rule() -> None:
    ev_synth_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={"status": "fail", "first_failure": {"max_abs_error": 0.2}},
    )
    candidates, _ = classify_failure_domain("region:1", [ev_synth_fail])
    assert any(c.domain == "implementation" and c.confidence == 0.60 for c in candidates)


def test_classification_model_artifact_and_quantization_rules() -> None:
    prov = _mock_provenance()
    ev_synth = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov,
        comparison_report={"status": "pass", "output_roles": ["y"]},
    )
    ev_ref = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov,
        comparison_report={"status": "pass", "output_roles": ["y"]},
    )
    ev_real_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "real_weight", "tags": []},
        provenance=prov,
        comparison_report={"status": "fail", "output_roles": ["y"]},
    )

    # Real weight fails while synthetic and reference pass -> model_artifact
    candidates, _ = classify_failure_domain("region:1", [ev_synth, ev_ref, ev_real_fail])
    assert any(c.domain == "model_artifact" and c.confidence == 0.70 for c in candidates)

    # If real_weight has quantized tag -> quantization
    ev_quant_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "real_weight", "tags": ["quantized"]},
        provenance=prov,
        comparison_report={"status": "fail", "output_roles": ["y"]},
    )
    candidates_quant, _ = classify_failure_domain("region:1", [ev_synth, ev_ref, ev_quant_fail])
    assert any(c.domain == "quantization" and c.confidence == 0.75 for c in candidates_quant)


def test_classification_unknown_fallback_and_decisions() -> None:
    # A single reference_derived failure with no other fixtures
    ev_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report={"status": "fail", "first_failure": {"max_abs_error": 0.5}},
    )
    candidates, _ = classify_failure_domain("region:1", [ev_fail])
    assert len(candidates) == 1
    assert candidates[0].domain == "unknown"
    assert candidates[0].confidence == 0.0

    # User rejection note rejects the candidate using explicit target
    note_reject = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        category="classification-reject",
        reason="rejecting candidate",
        target=candidates[0].id,
    )
    # User override note targeting quantization using explicit target
    note_override = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        category="classification-override",
        reason="quantization verified by expert",
        target="quantization",
        duration_minutes=20.0,
    )
    active_cands, decisions = classify_failure_domain("region:1", [ev_fail], notes=[note_reject, note_override])
    # The unknown candidate was rejected
    assert len(active_cands) == 0
    assert len(decisions) == 2
    assert decisions[0].kind == "reject"
    assert decisions[0].target == candidates[0].id
    assert decisions[1].kind == "override"
    assert decisions[1].target == "quantization"
    assert decisions[1].duration_minutes == 20.0


def test_cross_fixture_comparability_mismatch_prevents_rule() -> None:
    prov1 = _mock_provenance()
    prov2 = dict(_mock_provenance())
    prov2["effective_comparison_sha256"] = "b" * 64

    ev_synth = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov1,
        comparison_report={"status": "pass"},
    )
    ev_ref = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov1,
        comparison_report={"status": "pass"},
    )
    # Real weight fails, but was compared with a different effective comparison policy
    ev_real_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "real_weight", "tags": []},
        provenance=prov2,
        comparison_report={"status": "fail"},
    )

    # Because comparison policies differ, they are not cross-fixture comparable
    candidates, _ = classify_failure_domain("region:1", [ev_synth, ev_ref, ev_real_fail])
    assert not any(c.domain == "model_artifact" for c in candidates)
    assert any(c.domain == "unknown" for c in candidates)


def test_layout_classification_tolerance_within_pass() -> None:
    prov = _mock_provenance()
    # Comparison report with non-zero max_abs_error but 0 mismatch_count, and stride mismatch
    report = {
        "format": "inferref-comparison-report",
        "format_version": "0.2",
        "status": "fail",
        "first_failure": {
            "name": "hidden_states",
            "metrics": {"max_abs_error": 1e-7, "max_rel_error": 1e-6, "mismatch_count": 0},
            "layout": {
                "dtype_mismatch": False,
                "shape_mismatch": False,
                "stride_mismatch": True,
                "storage_offset_mismatch": False,
            },
        },
    }
    ev = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov,
        comparison_report=report,
    )
    candidates, _ = classify_failure_domain("region:1", [ev])
    assert any(c.domain == "layout" and c.rule == "logical_pass_strict_layout_fails/v1" for c in candidates)


def test_cross_fixture_comparability_newer_unrelated_synthetic_does_not_shadow() -> None:
    prov_a = _mock_provenance()
    prov_b = dict(_mock_provenance())
    prov_b["effective_comparison_sha256"] = "b" * 64

    # Policy A complete triplet
    ev_synth_a = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov_a,
        comparison_report={"status": "pass", "output_roles": ["y"]},
        created_at="2026-09-03T01:00:00Z",
    )
    ev_ref_a = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_a,
        comparison_report={"status": "pass", "output_roles": ["y"]},
        created_at="2026-09-03T01:05:00Z",
    )
    ev_real_a = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "real_weight", "tags": []},
        provenance=prov_a,
        comparison_report={"status": "fail", "output_roles": ["y"]},
        created_at="2026-09-03T01:10:00Z",
    )

    # Newer Policy B synthetic run (which should not shadow Policy A triplet)
    ev_synth_b = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov_b,
        comparison_report={"status": "pass", "output_roles": ["y"]},
        created_at="2026-09-03T01:15:00Z",
    )

    candidates, _ = classify_failure_domain(
        "region:1", [ev_synth_a, ev_ref_a, ev_real_a, ev_synth_b]
    )
    # Policy A group forms a complete triplet, identifying model_artifact
    assert any(c.domain == "model_artifact" and c.confidence == 0.70 for c in candidates)


def test_empty_evidence_does_not_raise_unbound_local_error() -> None:
    candidates, decisions = classify_failure_domain("region:1", [])
    assert len(candidates) == 1
    assert candidates[0].domain == "unknown"
    assert candidates[0].confidence == 0.0
    assert len(decisions) == 0


def test_policy_a_synthetic_fail_with_policy_b_reference_pass() -> None:
    prov_a = _mock_provenance()
    prov_b = dict(_mock_provenance())
    prov_b["effective_comparison_sha256"] = "b" * 64

    # Policy A synthetic failure
    ev_synth_a = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov_a,
        comparison_report={"status": "fail"},
        created_at="2026-09-03T01:00:00Z",
    )
    # Policy B reference pass
    ev_ref_b = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_b,
        comparison_report={"status": "pass"},
        created_at="2026-09-03T01:05:00Z",
    )

    candidates, _ = classify_failure_domain("region:1", [ev_synth_a, ev_ref_b])
    assert any(c.domain == "implementation" and c.rule == "synthetic_boundary_fails_execution_succeeds/v1" for c in candidates)


def test_layout_missing_numerical_facts_fails_closed() -> None:
    prov = _mock_provenance()
    # Evidence with layout mismatch, but NO numerical metrics, mismatch_count, or max_abs_error
    ev = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov,
        comparison_report={
            "format": "inferref-comparison-report",
            "format_version": "0.2",
            "status": "fail",
            "first_failure": {
                "name": "y",
                "message": "layout mismatch with unrecorded numerical errors",
                "layout": {"stride_mismatch": True},
            },
        },
    )
    # Missing numerical facts must fail closed (not default to 0.0 max_abs_error)
    candidates, _ = classify_failure_domain("region:1", [ev])
    assert not any(c.domain == "layout" for c in candidates)
    assert any(c.domain == "unknown" for c in candidates)


def test_empty_output_roles_contract_not_cross_fixture_comparable() -> None:
    prov = _mock_provenance()
    # Three evidences missing output-role contract (empty output_roles)
    ev_synth = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov,
        comparison_report={"status": "pass"},
        created_at="2026-09-03T01:00:00Z",
    )
    ev_ref = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov,
        comparison_report={"status": "pass"},
        created_at="2026-09-03T01:05:00Z",
    )
    ev_real = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "parent"},
        fixture={"kind": "real_weight", "tags": []},
        provenance=prov,
        comparison_report={"status": "fail"},
        created_at="2026-09-03T01:10:00Z",
    )
    # Ensure none of them have output_roles
    assert "output_roles" not in ev_synth.observations
    assert "output_roles" not in ev_ref.observations
    assert "output_roles" not in ev_real.observations

    candidates, _ = classify_failure_domain("region:1", [ev_synth, ev_ref, ev_real])
    # Empty output-role contract must not trigger model_artifact
    assert not any(c.domain == "model_artifact" for c in candidates)
    assert any(c.domain == "unknown" for c in candidates)


def test_layout_not_shadowed_by_different_comparison_policy_pass() -> None:
    prov_a = _mock_provenance()
    prov_b = dict(_mock_provenance())
    prov_b["effective_comparison_sha256"] = "b" * 64

    # Policy A: layout failure with logical_pass (mismatch_count=0)
    ev_policy_a = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "r1"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov_a,
        comparison_report={
            "status": "fail",
            "first_failure": {
                "name": "out",
                "metrics": {"max_abs_error": 1e-7, "mismatch_count": 0},
                "layout": {
                    "dtype_mismatch": False,
                    "shape_mismatch": False,
                    "stride_mismatch": True,
                    "storage_offset_mismatch": False,
                },
            },
        },
        created_at="2026-09-03T01:00:00Z",
    )

    # Policy B: passes, created later
    ev_policy_b = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "r1"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov_b,
        comparison_report={"status": "pass", "layout_match": True, "mismatch_count": 0},
        created_at="2026-09-03T01:05:00Z",
    )

    candidates, _ = classify_failure_domain("region:1", [ev_policy_a, ev_policy_b])
    # Policy B pass must NOT shadow Policy A's valid layout failure!
    layout_cand = next((c for c in candidates if c.domain == "layout"), None)
    assert layout_cand is not None
    assert ev_policy_a.id in layout_cand.supporting_evidence


def test_fallback_unknown_collects_uncovered_current_failure_ids_across_policies() -> None:
    prov_a = _mock_provenance()
    prov_b = dict(_mock_provenance())
    prov_b["effective_comparison_sha256"] = "b" * 64

    # Policy A: generic numerical failure on reference_derived (no layout, no implementation rule)
    ev_policy_a = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "r1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_a,
        comparison_report={"status": "fail", "mismatch_count": 42, "max_abs_error": 0.5},
        created_at="2026-09-03T01:00:00Z",
    )

    # Policy B: passes, created later
    ev_policy_b = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "r1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_b,
        comparison_report={"status": "pass", "mismatch_count": 0},
        created_at="2026-09-03T01:05:00Z",
    )

    candidates, _ = classify_failure_domain("region:1", [ev_policy_a, ev_policy_b])
    # Fallback unknown must collect Policy A's failure id, NOT return empty supporting_evidence!
    unknown_cand = next((c for c in candidates if c.domain == "unknown"), None)
    assert unknown_cand is not None
    assert ev_policy_a.id in unknown_cand.supporting_evidence


def test_classification_with_events_prioritizes_newer_event_at_same_timestamp() -> None:
    same_time = "2026-09-03T01:00:00Z"
    prov = _mock_provenance()

    # Ensure ev_layout_fail has an id that sorts higher than ev_pass lexicographically
    # so that without events it would win, but with events ev_pass (seq=3) wins.
    ev_layout_fail = Evidence(
        id="ev-zzz-layout-fail",
        created_at=same_time,
        type="numerical_comparison",
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        result="fail",
        observations={
            "layout_match": False,
            "first_failure": {"mismatch_count": 0, "max_abs_error": 0.0},
            "output_roles": ["logits"],
        },
        provenance=prov,
    )
    ev_pass = Evidence(
        id="ev-aaa-pass",
        created_at=same_time,
        type="numerical_comparison",
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        result="pass",
        observations={
            "layout_match": True,
            "mismatch_count": 0,
            "output_roles": ["logits"],
        },
        provenance=prov,
    )

    events = [
        AlignmentEvent(seq=1, timestamp=same_time, event="session_initialized", data={}),
        AlignmentEvent(
            seq=2,
            timestamp=same_time,
            event="boundary_failed",
            data={"boundary": "region:1"},
            evidence_ids=[ev_layout_fail.id],
        ),
        AlignmentEvent(
            seq=3,
            timestamp=same_time,
            event="boundary_verified",
            data={"boundary": "region:1"},
            evidence_ids=[ev_pass.id],
        ),
    ]

    # Without events: file loading order (ev-aaa then ev-zzz) causes ev-zzz layout failure to win
    candidates_no_events, _ = classify_failure_domain("region:1", [ev_pass, ev_layout_fail])
    assert candidates_no_events[0].domain == "layout"

    # With events: event seq 3 > event seq 2 -> pass wins, cell is clean, domain is unknown
    candidates_with_events, _ = classify_failure_domain(
        "region:1",
        [ev_pass, ev_layout_fail],
        events=events,
    )
    assert candidates_with_events[0].domain == "unknown"
