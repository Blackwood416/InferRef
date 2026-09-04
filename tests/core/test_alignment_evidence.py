"""Tests for Evidence v0.1 normalization and Fixture Matrix (0.10 L3)."""

from __future__ import annotations

from inferref.alignment.evidence import (
    build_comparison_evidence,
    build_manual_note_evidence,
    is_comparable,
)
from inferref.alignment.matrix import build_fixture_matrix
from inferref.alignment.schema import AlignmentSnapshot

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
DIGEST_C = "c" * 64


def _mock_session(engine_rev: str = "rev-1") -> AlignmentSnapshot:
    return AlignmentSnapshot(
        session_id="al-1234567812345678",
        created_at="2026-09-03T00:00:00Z",
        updated_at="2026-09-03T00:00:00Z",
        last_event_seq=1,
        source={"trace": {"path": "trace.irtrace", "metadata_sha256": DIGEST_A}},
        target={
            "model": "model",
            "engine": "engine",
            "adapter": {"path": "adapter.json", "normalized_sha256": DIGEST_B},
        },
        required_fixture_kinds=("reference_derived",),
    )


def _mock_provenance(engine_rev: str = "rev-1") -> dict:
    return {
        "trace_metadata_sha256": DIGEST_A,
        "testcase": {"path": "testcases/tc1", "tree_sha256": DIGEST_C},
        "run_record": {"path": "runs/run1/inferref-run.json", "sha256": DIGEST_C},
        "adapter_normalized_sha256": DIGEST_B,
        "engine_identity": {"kind": "git", "revision": engine_rev, "dirty": False},
        "effective_comparison_sha256": DIGEST_A,
    }


def test_build_comparison_evidence_observations() -> None:
    # Pass report
    pass_report = {"status": "pass", "results": {"out": {"status": "pass"}}}
    ev_pass = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report=pass_report,
    )
    assert ev_pass.result == "pass"
    assert ev_pass.observations["outputs_match"] is True
    assert ev_pass.observations["layout_match"] is True
    assert ev_pass.observations["provided_inputs_valid"] is True
    # Crucial 0.10 invariant: engine inputs never claimed to be observed
    assert ev_pass.observations["engine_inputs_observed"] is False
    assert ev_pass.observations["first_failure"] is None

    # Fail report with first_failure
    fail_report = {
        "status": "fail",
        "first_failure": {"output": "out", "max_abs_error": 0.123},
        "diagnostics": {"layout_mismatch": False},
    }
    ev_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report=fail_report,
    )
    assert ev_fail.result == "fail"
    assert ev_fail.observations["outputs_match"] is False
    assert ev_fail.observations["first_failure"]["max_abs_error"] == 0.123

    # Layout failure report
    layout_fail_report = {
        "status": "fail",
        "first_failure": {"output": "out", "max_abs_error": 0.0},
        "diagnostics": {"layout_mismatch": True},
    }
    ev_layout = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report=layout_fail_report,
    )
    assert ev_layout.observations["layout_match"] is False


def test_comparability_equivalence_class() -> None:
    ev1 = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": ["fp16"]},
        provenance=_mock_provenance("rev-1"),
        comparison_report={"status": "pass"},
    )
    ev2 = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": ["fp16"]},
        provenance=_mock_provenance("rev-1"),
        comparison_report={"status": "fail", "first_failure": {"max_abs_error": 0.5}},
    )
    # Same boundary, fixture, testcase, adapter, engine, policy
    assert is_comparable(ev1, ev2)

    # Different engine revision -> not comparable
    ev3 = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": ["fp16"]},
        provenance=_mock_provenance("rev-2"),
        comparison_report={"status": "pass"},
    )
    assert not is_comparable(ev1, ev3)


def test_fixture_matrix_state_aggregation_and_stale_detection() -> None:
    session = _mock_session()
    prov_rev1 = _mock_provenance("rev-1")

    # 1. Synthetic pass alone does not verify a boundary requiring reference_derived
    ev_synth = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=prov_rev1,
        comparison_report={"status": "pass"},
        created_at="2026-09-03T00:10:00Z",
    )
    matrix = build_fixture_matrix([ev_synth], session, known_boundaries=["region:1", "region:2"])
    assert matrix.boundary_states["region:1"] == "untested"
    assert matrix.boundary_states["region:2"] == "untested"

    # 2. Reference-derived pass verifies the boundary
    ev_ref = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_rev1,
        comparison_report={"status": "pass"},
        created_at="2026-09-03T00:20:00Z",
    )
    matrix = build_fixture_matrix([ev_synth, ev_ref], session, known_boundaries=["region:1"])
    assert matrix.boundary_states["region:1"] == "verified"
    assert matrix.verified_boundaries == ("region:1",)

    # 3. Newer failing evidence makes it failing
    ev_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_rev1,
        comparison_report={"status": "fail"},
        created_at="2026-09-03T00:30:00Z",
    )
    matrix = build_fixture_matrix([ev_synth, ev_ref, ev_fail], session, known_boundaries=["region:1"])
    assert matrix.boundary_states["region:1"] == "failing"

    # 4. Changing engine revision marks previous verification as stale
    matrix_rev2 = build_fixture_matrix(
        [ev_ref],
        session,
        known_boundaries=["region:1"],
        active_engine_identity={"kind": "git", "revision": "rev-2", "dirty": False},
    )
    assert matrix_rev2.boundary_states["region:1"] == "stale"
    assert matrix_rev2.stale_boundaries == ("region:1",)


def test_workflow_metrics_and_notes() -> None:
    session = _mock_session()
    ev1 = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "synthetic", "tags": []},
        provenance=_mock_provenance("rev-1"),
        comparison_report={"status": "pass"},
    )
    ev2 = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance("rev-1"),
        comparison_report={"status": "pass"},
    )
    note1 = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        category="custom-tool",
        reason="inspected memory dump",
        duration_minutes=15.0,
    )
    note2 = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        category="milestone",
        reason="model matches reference",
        milestone="first-correct-model",
        created_at="2026-09-03T01:00:00Z",
    )

    matrix = build_fixture_matrix([ev1, ev2, note1, note2], session, known_boundaries=["region:1"])
    metrics = matrix.metrics
    assert metrics["inferref_validation_steps"] == 2
    assert metrics["ad_hoc_debugging_steps"] == 1
    assert metrics["ad_hoc_debugging_minutes"] == 15.0
    # 2 validation steps / (2 + 1 ad-hoc) = 2/3 = 0.6667
    assert metrics["adoption_ratio"] == 0.6667
    assert metrics["tfcm_seconds"] == 3600.0

    # Note without duration reports ad_hoc_debugging_minutes as None (unknown)
    note_no_dur = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        category="custom-tool",
        reason="quick inspect without duration",
    )
    matrix_no_dur = build_fixture_matrix([ev1, ev2, note_no_dur], session, known_boundaries=["region:1"])
    assert matrix_no_dur.metrics["ad_hoc_debugging_steps"] == 1
    assert matrix_no_dur.metrics["ad_hoc_debugging_minutes"] is None

    # If partial notes lack duration, ad_hoc_debugging_minutes must also be None (unknown)
    note_with_dur = build_manual_note_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        category="custom-tool",
        reason="note with duration",
        duration_minutes=5.0,
    )
    matrix_partial_dur = build_fixture_matrix(
        [ev1, ev2, note_with_dur, note_no_dur],
        session,
        known_boundaries=["region:1"],
    )
    assert matrix_partial_dur.metrics["ad_hoc_debugging_steps"] == 2
    assert matrix_partial_dur.metrics["ad_hoc_debugging_minutes"] is None

    # Test numerical_alignment_time_seconds
    ev_fail_early = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance("rev-1"),
        comparison_report={"status": "fail"},
        created_at="2026-09-03T00:30:00Z",
    )
    ev_pass_late = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance("rev-1"),
        comparison_report={"status": "pass"},
        created_at="2026-09-03T01:00:00Z",
    )
    # When session.status is "aligning", numerical_alignment_time_seconds must be None
    matrix_nat_aligning = build_fixture_matrix(
        [ev_fail_early, ev_pass_late],
        session,
        known_boundaries=["region:1"],
    )
    assert matrix_nat_aligning.metrics["numerical_alignment_time_seconds"] is None

    # Even with verified + unsupported boundaries, if session is still aligning, must be None
    ev_unsupported = build_comparison_evidence(
        subject={"boundary": "region:2", "name": "layer2"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance("rev-1"),
        comparison_report={"status": "unsupported"},
        created_at="2026-09-03T01:05:00Z",
    )
    matrix_nat_unsupp = build_fixture_matrix(
        [ev_fail_early, ev_pass_late, ev_unsupported],
        session,
        known_boundaries=["region:1", "region:2"],
    )
    assert matrix_nat_unsupp.metrics["numerical_alignment_time_seconds"] is None

    # When session.status is "verified", numerical_alignment_time_seconds is computed
    verified_session = AlignmentSnapshot(
        session_id=session.session_id,
        created_at=session.created_at,
        updated_at="2026-09-03T01:00:00Z",
        status="verified",
        last_event_seq=2,
        source=session.source,
        target=session.target,
        required_fixture_kinds=session.required_fixture_kinds,
    )
    from inferref.alignment.schema import AlignmentEvent
    fail_event = AlignmentEvent(
        seq=1,
        timestamp="2026-09-03T00:30:00Z",
        event="boundary_failed",
        data={"boundary": "region:1"},
        evidence_ids=[ev_fail_early.id],
    )
    pass_event = AlignmentEvent(
        seq=2,
        timestamp="2026-09-03T01:00:00Z",
        event="boundary_verified",
        data={"boundary": "region:1"},
        evidence_ids=[ev_pass_late.id],
    )
    ver_event = AlignmentEvent(
        seq=3,
        timestamp="2026-09-03T01:00:00Z",
        event="session_verified",
        data={"verified_boundaries": ["region:1"]},
    )
    matrix_nat_verified = build_fixture_matrix(
        [ev_fail_early, ev_pass_late],
        verified_session,
        known_boundaries=["region:1"],
        events=[fail_event, pass_event, ver_event],
    )
    assert matrix_nat_verified.metrics["numerical_alignment_time_seconds"] == 1800.0


def test_build_comparison_evidence_with_real_comparison_report() -> None:
    comp_report = {
        "format": "inferref-comparison-report",
        "format_version": "0.2",
        "status": "fail",
        "first_failure": {
            "name": "hidden_states",
            "metrics": {"max_abs_error": 0.042, "max_rel_error": 0.001},
            "layout": {
                "dtype_mismatch": False,
                "shape_mismatch": False,
                "stride_mismatch": True,
                "storage_offset_mismatch": False,
            },
        },
        "comparisons": [
            {
                "name": "hidden_states",
                "status": "fail",
                "metrics": {"max_abs_error": 0.042, "max_rel_error": 0.001},
                "layout": {
                    "dtype_mismatch": False,
                    "shape_mismatch": False,
                    "stride_mismatch": True,
                    "storage_offset_mismatch": False,
                },
            }
        ],
    }
    ev = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=_mock_provenance(),
        comparison_report=comp_report,
    )
    assert ev.result == "fail"
    assert ev.observations["outputs_match"] is False
    assert ev.observations["layout_match"] is False
    assert ev.observations["first_failure"]["output"] == "hidden_states"
    assert ev.observations["first_failure"]["max_abs_error"] == 0.042
    assert ev.observations["first_failure"]["max_rel_error"] == 0.001


def test_matrix_deterministic_evidence_ordering_same_timestamp_reverse_files() -> None:
    session = _mock_session()
    prov_old = _mock_provenance(engine_rev="rev-old")
    prov_new = _mock_provenance(engine_rev="rev-new")

    # Identical ISO timestamp string (same second)
    same_time = "2026-09-04T12:00:00Z"

    ev_pass = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_old,
        comparison_report={"status": "pass"},
        created_at=same_time,
    )
    ev_fail = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_new,
        comparison_report={"status": "fail", "mismatch_count": 10},
        created_at=same_time,
    )

    from inferref.alignment.schema import AlignmentEvent

    events = [
        AlignmentEvent(seq=1, timestamp=same_time, event="session_initialized", data={}),
        AlignmentEvent(seq=2, timestamp=same_time, event="boundary_verified", data={"boundary": "region:1"}, evidence_ids=[ev_pass.id]),
        AlignmentEvent(seq=3, timestamp=same_time, event="boundary_failed", data={"boundary": "region:1"}, evidence_ids=[ev_fail.id]),
    ]

    # Order 1: Reverse file order (ev_fail first, ev_pass second)
    matrix1 = build_fixture_matrix(
        [ev_fail, ev_pass],
        session,
        known_boundaries=["region:1"],
        events=events,
    )
    assert matrix1.active_engine["revision"] == "rev-new"
    assert matrix1.boundary_states["region:1"] == "failing"
    assert any(ev.id == ev_fail.id for ev in matrix1.current_evidence.values())

    # Order 2: Forward order (ev_pass first, ev_fail second)
    matrix2 = build_fixture_matrix(
        [ev_pass, ev_fail],
        session,
        known_boundaries=["region:1"],
        events=events,
    )
    assert matrix2.active_engine["revision"] == "rev-new"
    assert matrix2.boundary_states["region:1"] == "failing"
    assert any(ev.id == ev_fail.id for ev in matrix2.current_evidence.values())


def test_matrix_excludes_uncommitted_crash_orphans_and_accepts_prospective_evidence() -> None:
    session = _mock_session()
    prov_old = _mock_provenance("rev-old")
    prov_new = _mock_provenance("rev-new")

    # Committed evidence: rev-old, verified
    ev_committed = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_old,
        comparison_report={"status": "pass"},
        created_at="2026-09-03T00:10:00Z",
    )

    # Crash orphan evidence: written to disk during an interrupted run, rev-new, failing
    # NOT referenced by any event in events.jsonl
    ev_orphan = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_new,
        comparison_report={"status": "fail", "mismatch_count": 5},
        created_at="2026-09-03T00:20:00Z",
    )

    from inferref.alignment.schema import AlignmentEvent

    events = [
        AlignmentEvent(seq=1, timestamp="2026-09-03T00:00:00Z", event="session_initialized", data={}),
        AlignmentEvent(
            seq=2,
            timestamp="2026-09-03T00:10:00Z",
            event="boundary_verified",
            data={"boundary": "region:1"},
            evidence_ids=[ev_committed.id],
        ),
    ]

    # When building matrix from loaded evidence (which includes the orphan on disk):
    matrix_reloaded = build_fixture_matrix(
        [ev_committed, ev_orphan],
        session,
        known_boundaries=["region:1"],
        events=events,
    )
    # The orphan MUST be excluded: active_engine remains rev-old, boundary state remains verified
    assert matrix_reloaded.active_engine["revision"] == "rev-old"
    assert matrix_reloaded.boundary_states["region:1"] == "verified"
    assert any(ev.id == ev_committed.id for ev in matrix_reloaded.current_evidence.values())
    assert not any(ev.id == ev_orphan.id for ev in matrix_reloaded.current_evidence.values())

    # Now runner executes a new prospective run with prospective_evidence:
    ev_prospective = build_comparison_evidence(
        subject={"boundary": "region:1", "name": "layer1"},
        fixture={"kind": "reference_derived", "tags": []},
        provenance=prov_new,
        comparison_report={"status": "fail", "mismatch_count": 2},
        created_at="2026-09-03T00:30:00Z",
    )
    matrix_prospective = build_fixture_matrix(
        [ev_committed, ev_orphan],
        session,
        known_boundaries=["region:1"],
        events=events,
        prospective_evidence=ev_prospective,
    )
    # The prospective candidate is evaluated as newest, while the crash orphan remains excluded
    assert matrix_prospective.active_engine["revision"] == "rev-new"
    assert matrix_prospective.boundary_states["region:1"] == "failing"
    assert any(ev.id == ev_prospective.id for ev in matrix_prospective.current_evidence.values())
    assert not any(ev.id == ev_orphan.id for ev in matrix_prospective.current_evidence.values())
