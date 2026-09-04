"""Evidence matrix projection and workflow metrics for Alignment Session v0.1 (L3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from inferref.alignment.evidence import (
    engine_identities_equal,
    normalize_engine_identity,
)
from inferref.alignment.schema import AlignmentSnapshot, Evidence


@dataclass
class FixtureMatrix:
    """Matrix of current evidence across boundaries and fixture kinds."""

    current_evidence: dict[tuple[Any, ...], Evidence] = field(default_factory=dict)
    historical_evidence: dict[tuple[Any, ...], list[Evidence]] = field(default_factory=dict)
    boundary_states: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    active_engine_identity: dict[str, Any] | None = None

    @property
    def active_engine(self) -> dict[str, Any] | None:
        return self.active_engine_identity

    @property
    def verified_boundaries(self) -> tuple[str, ...]:
        return tuple(sorted(b for b, s in self.boundary_states.items() if s == "verified"))

    @property
    def failing_boundaries(self) -> tuple[str, ...]:
        return tuple(sorted(b for b, s in self.boundary_states.items() if s == "failing"))

    @property
    def untested_boundaries(self) -> tuple[str, ...]:
        return tuple(sorted(b for b, s in self.boundary_states.items() if s == "untested"))

    @property
    def stale_boundaries(self) -> tuple[str, ...]:
        return tuple(sorted(b for b, s in self.boundary_states.items() if s == "stale"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "boundary_states": dict(sorted(self.boundary_states.items())),
            "metrics": dict(self.metrics),
            "active_engine_identity": self.active_engine_identity,
            "counts": {
                "verified": len(self.verified_boundaries),
                "failing": len(self.failing_boundaries),
                "untested": len(self.untested_boundaries),
                "stale": len(self.stale_boundaries),
            },
        }


def _parse_timestamp(iso_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def build_fixture_matrix(
    evidence_list: Sequence[Evidence],
    session: AlignmentSnapshot,
    known_boundaries: Sequence[str] = (),
    *,
    active_engine_identity: Mapping[str, Any] | None = None,
    events: Sequence[Any] = (),
    prospective_evidence: Evidence | Sequence[Evidence] | None = None,
) -> FixtureMatrix:
    """Build the fixture matrix and evaluate boundary states and metrics."""
    # Build mapping from evidence_id to event sequence number
    evidence_seq: dict[str, int] = {}
    max_event_seq = 0
    for ev_item in events:
        s = getattr(ev_item, "seq", None)
        if s is None and isinstance(ev_item, Mapping):
            s = ev_item.get("seq")
        if isinstance(s, int):
            max_event_seq = max(max_event_seq, s)
            e_ids = getattr(ev_item, "evidence_ids", None)
            if e_ids is None and isinstance(ev_item, Mapping):
                e_ids = ev_item.get("evidence_ids", ())
            for eid in (e_ids or ()):
                if isinstance(eid, str):
                    evidence_seq[eid] = max(evidence_seq.get(eid, 0), s)

    # Normalize explicit prospective evidence
    prospective_list: list[Evidence] = []
    if prospective_evidence is not None:
        if isinstance(prospective_evidence, Evidence):
            prospective_list = [prospective_evidence]
        else:
            prospective_list = list(prospective_evidence)
    prospective_ids = {getattr(ev, "id", None) for ev in prospective_list if getattr(ev, "id", None)}

    # Combine input evidence list with prospective evidence
    combined_evidence: list[Evidence] = list(evidence_list)
    for p_ev in prospective_list:
        if not any(getattr(e, "id", None) == getattr(p_ev, "id", None) for e in combined_evidence):
            combined_evidence.append(p_ev)

    # When event log is provided, project only committed evidence (referenced by events)
    # and explicit prospective evidence. Uncommitted / crash-orphaned evidence files are excluded.
    if events:
        filtered_evidence = [
            ev
            for ev in combined_evidence
            if getattr(ev, "id", None) in evidence_seq or getattr(ev, "id", None) in prospective_ids
        ]
    else:
        filtered_evidence = combined_evidence

    evidence_index = {getattr(ev, "id", str(i)): i for i, ev in enumerate(filtered_evidence)}

    def _ev_order_key(ev: Evidence) -> tuple[int, float, int, str]:
        ev_id = getattr(ev, "id", "")
        if ev_id in prospective_ids:
            s = (max_event_seq + 1) if max_event_seq > 0 else 1
        else:
            s = evidence_seq.get(ev_id, 0)
        t = _parse_timestamp(ev.created_at) if hasattr(ev, "created_at") else None
        ts = t.timestamp() if t is not None else 0.0
        idx = evidence_index.get(ev_id, 0)
        return (s, ts, idx, ev_id)

    # Determine active engine identity from parameter, or latest numerical evidence
    active_engine: dict[str, Any] | None = None
    if active_engine_identity is not None:
        active_engine = normalize_engine_identity(active_engine_identity)
    else:
        # Find the latest numerical evidence's engine identity
        latest_key = None
        for ev in filtered_evidence:
            if ev.type == "numerical_comparison":
                k = _ev_order_key(ev)
                if latest_key is None or k > latest_key:
                    latest_key = k
                    active_engine = normalize_engine_identity(ev.provenance.get("engine_identity"))

    session_trace_sha = session.source.get("trace", {}).get("metadata_sha256")
    session_adapter_sha = session.target.get("adapter", {}).get("normalized_sha256")
    required_kinds = set(session.required_fixture_kinds or ["reference_derived"])

    current_evidence: dict[tuple[Any, ...], Evidence] = {}
    historical_evidence: dict[tuple[Any, ...], list[Evidence]] = {}

    # Separate current vs historical evidence
    for ev in filtered_evidence:
        if ev.type != "numerical_comparison":
            continue
        boundary = ev.subject["boundary"]
        kind = ev.fixture["kind"]
        tags_tuple = tuple(sorted(ev.fixture.get("tags", ())))
        prov = ev.provenance
        tc_sha = prov.get("testcase", {}).get("tree_sha256")
        comp_sha = prov.get("effective_comparison_sha256")
        cell_key = (boundary, kind, tags_tuple, tc_sha, comp_sha)

        # Check comparability with session definition
        trace_match = prov.get("trace_metadata_sha256") == session_trace_sha
        adapter_match = prov.get("adapter_normalized_sha256") == session_adapter_sha
        engine_match = active_engine is None or engine_identities_equal(
            prov.get("engine_identity"), active_engine
        )

        is_current = trace_match and adapter_match and engine_match

        if is_current:
            existing = current_evidence.get(cell_key)
            if existing is None or _ev_order_key(ev) > _ev_order_key(existing):
                if existing is not None:
                    historical_evidence.setdefault(cell_key, []).append(existing)
                current_evidence[cell_key] = ev
            else:
                historical_evidence.setdefault(cell_key, []).append(ev)
        else:
            historical_evidence.setdefault(cell_key, []).append(ev)

    all_boundary_ids = set(known_boundaries)
    for k in current_evidence:
        all_boundary_ids.add(k[0])
    for k in historical_evidence:
        all_boundary_ids.add(k[0])

    boundary_states: dict[str, str] = {}
    for boundary in sorted(all_boundary_ids):
        # Current evidence for this boundary across all comparability cells
        current_for_b = [ev for k, ev in current_evidence.items() if k[0] == boundary]
        hist_for_b = [ev for k, evs in historical_evidence.items() if k[0] == boundary for ev in evs]

        if not current_for_b and not hist_for_b:
            boundary_states[boundary] = "untested"
            continue

        if not current_for_b:
            # Only historical runs exist (e.g. engine identity changed)
            boundary_states[boundary] = "stale"
            continue

        # Check for failures or errors in current evidence
        has_fail = any(ev.result == "fail" for ev in current_for_b)
        has_error = any(ev.result == "error" for ev in current_for_b)
        has_unsupported = any(ev.result == "unsupported" for ev in current_for_b)

        if has_fail:
            boundary_states[boundary] = "failing"
        elif has_error:
            boundary_states[boundary] = "blocked"
        elif has_unsupported:
            boundary_states[boundary] = "unsupported"
        else:
            # Check if all required fixture kinds pass
            passed_kinds = {ev.fixture["kind"] for ev in current_for_b if ev.result == "pass"}
            if required_kinds.issubset(passed_kinds):
                boundary_states[boundary] = "verified"
            else:
                # Some passed, but not all required kinds have been tested yet
                boundary_states[boundary] = "untested"

    # Compute workflow metrics (Spec §12)
    num_validation_steps = sum(1 for ev in filtered_evidence if ev.type == "numerical_comparison")
    distinct_engines = {
        (eng.get("kind"), eng.get("revision"), eng.get("dirty"))
        for ev in filtered_evidence
        if ev.type == "numerical_comparison"
        for eng in [normalize_engine_identity(ev.provenance.get("engine_identity"))]
    }
    distinct_runs = {
        ev.provenance.get("run_record", {}).get("path")
        for ev in filtered_evidence
        if ev.type == "numerical_comparison" and "run_record" in ev.provenance
    }

    ad_hoc_time = 0.0
    ad_hoc_steps = 0
    ad_hoc_durations_count = 0
    tfcm_seconds = None
    session_start = _parse_timestamp(session.created_at)

    for ev in filtered_evidence:
        if ev.type == "manual_note" and ev.note:
            cat = ev.note.get("category")
            if cat in ("custom-tool", "missing-observation"):
                ad_hoc_steps += 1
                dur = ev.note.get("duration_minutes")
                if isinstance(dur, (int, float)):
                    ad_hoc_time += float(dur)
                    ad_hoc_durations_count += 1
            if ev.note.get("milestone") == "first-correct-model":
                note_time = _parse_timestamp(ev.created_at)
                if session_start and note_time:
                    tfcm_seconds = max(0.0, (note_time - session_start).total_seconds())

    # If ad_hoc steps exist but some or all durations are missing, report as unknown (None)
    ad_hoc_debugging_minutes: float | None
    if ad_hoc_steps == 0:
        ad_hoc_debugging_minutes = 0.0
    elif ad_hoc_durations_count < ad_hoc_steps:
        ad_hoc_debugging_minutes = None
    else:
        ad_hoc_debugging_minutes = ad_hoc_time

    # Compute numerical alignment time (from first mismatch to session verified)
    numerical_alignment_time_seconds: float | None = None
    failing_numerical_times = [
        _parse_timestamp(ev.created_at)
        for ev in filtered_evidence
        if ev.type == "numerical_comparison" and ev.result in ("fail", "error")
    ]
    valid_failing_times = [t for t in failing_numerical_times if t is not None]
    if valid_failing_times and session.status == "verified":
        first_mismatch_time = min(valid_failing_times)
        terminal_time: datetime | None = None
        for e in reversed(events):
            ev_name = getattr(e, "event", None) if not isinstance(e, Mapping) else e.get("event")
            if ev_name == "session_verified":
                raw_ts = getattr(e, "timestamp", None) if not isinstance(e, Mapping) else e.get("timestamp")
                terminal_time = _parse_timestamp(raw_ts)
                break
        if terminal_time is None:
            terminal_time = _parse_timestamp(session.updated_at)
        if terminal_time is None:
            verify_times = [
                _parse_timestamp(ev.created_at)
                for ev in filtered_evidence
                if ev.type == "numerical_comparison" and ev.result == "pass"
            ]
            valid_verify_times = [t for t in verify_times if t is not None]
            if valid_verify_times:
                terminal_time = max(valid_verify_times)

        if terminal_time is not None and terminal_time >= first_mismatch_time:
            numerical_alignment_time_seconds = max(
                0.0, (terminal_time - first_mismatch_time).total_seconds()
            )

    denom = num_validation_steps + ad_hoc_steps
    adoption_ratio = (num_validation_steps / denom) if denom > 0 else None

    metrics = {
        "inferref_validation_steps": num_validation_steps,
        "engine_builds_count": len(distinct_engines),
        "run_records_count": len(distinct_runs),
        "ad_hoc_debugging_steps": ad_hoc_steps,
        "ad_hoc_debugging_minutes": ad_hoc_debugging_minutes,
        "numerical_alignment_time_seconds": numerical_alignment_time_seconds,
        "tfcm_seconds": tfcm_seconds,
        "adoption_ratio": round(adoption_ratio, 4) if adoption_ratio is not None else None,
        "adoption_complete": False,
        "boundary_counts": {
            "verified": sum(1 for s in boundary_states.values() if s == "verified"),
            "failing": sum(1 for s in boundary_states.values() if s == "failing"),
            "untested": sum(1 for s in boundary_states.values() if s == "untested"),
            "stale": sum(1 for s in boundary_states.values() if s == "stale"),
            "unsupported": sum(1 for s in boundary_states.values() if s == "unsupported"),
            "blocked": sum(1 for s in boundary_states.values() if s == "blocked"),
        },
    }

    return FixtureMatrix(
        current_evidence=current_evidence,
        historical_evidence=historical_evidence,
        boundary_states=boundary_states,
        metrics=metrics,
        active_engine_identity=active_engine,
    )
