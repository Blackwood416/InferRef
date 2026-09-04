"""Rule-based failure-domain classification for Alignment Session v0.1 (L5)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from inferref.alignment.evidence import (
    engine_identities_equal,
    normalize_engine_identity,
)
from inferref.alignment.schema import Evidence

STABLE_DOMAINS = (
    "implementation",
    "layout",
    "input_preparation",
    "model_artifact",
    "quantization",
    "reference",
    "environment",
    "unknown",
)


@dataclass
class ClassificationCandidate:
    """Deterministic failure domain hypothesis with supporting evidence."""

    id: str
    domain: str
    confidence: float
    rule: str
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        if self.limitations:
            return f"{self.rule} ({'; '.join(self.limitations)})"
        return self.rule

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "confidence": self.confidence,
            "rule": self.rule,
            "supporting_evidence": list(self.supporting_evidence),
            "contradicting_evidence": list(self.contradicting_evidence),
            "limitations": list(self.limitations),
        }


@dataclass
class UserDecision:
    """User-supplied classification rejection or override."""

    kind: str  # "reject" or "override"
    target: str  # candidate ID or domain name
    reason: str
    duration_minutes: float | None = None

    def to_dict(self) -> dict[str, Any]:
        res: dict[str, Any] = {
            "kind": self.kind,
            "target": self.target,
            "reason": self.reason,
        }
        if self.duration_minutes is not None:
            res["duration_minutes"] = self.duration_minutes
        return res


def make_candidate_id(rule: str, supporting: Sequence[str], contradicting: Sequence[str] = ()) -> str:
    sup_str = ",".join(sorted(supporting))
    con_str = ",".join(sorted(contradicting))
    token = f"{rule}:{sup_str}:{con_str}"
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]
    return f"fc-{digest}"


def _parse_timestamp(iso_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def classify_failure_domain(
    subject: str,
    evidence_list: Sequence[Evidence],
    *,
    notes: Sequence[Evidence] = (),
    active_engine_identity: Mapping[str, Any] | None = None,
    events: Sequence[Any] = (),
    prospective_evidence: Evidence | Sequence[Evidence] | None = None,
) -> tuple[tuple[ClassificationCandidate, ...], tuple[UserDecision, ...]]:
    """Evaluate deterministic 0.10 classification rules for a failing subject."""
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

    combined_evidence: list[Evidence] = list(evidence_list)
    for p_ev in prospective_list:
        if not any(getattr(e, "id", None) == getattr(p_ev, "id", None) for e in combined_evidence):
            combined_evidence.append(p_ev)

    if events:
        filtered_evidence = [
            ev
            for ev in combined_evidence
            if getattr(ev, "id", None) in evidence_seq or getattr(ev, "id", None) in prospective_ids
        ]
        filtered_notes = [
            n
            for n in notes
            if getattr(n, "id", None) in evidence_seq or getattr(n, "id", None) in prospective_ids
        ]
    else:
        filtered_evidence = combined_evidence
        filtered_notes = list(notes)

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

    # Filter numerical evidence for this subject and active engine
    active_engine = (
        normalize_engine_identity(active_engine_identity)
        if active_engine_identity is not None
        else None
    )

    subject_evidence = [
        ev
        for ev in filtered_evidence
        if ev.type == "numerical_comparison" and ev.subject.get("boundary") == subject
    ]
    if active_engine is not None:
        subject_evidence = [
            ev
            for ev in subject_evidence
            if engine_identities_equal(ev.provenance.get("engine_identity"), active_engine)
        ]

    # Map current evidence by comparability cell:
    # (fixture_kind, tags, engine_identity, effective_comparison_sha256, output_roles)
    cell_latest: dict[tuple[Any, ...], Evidence] = {}
    for ev in sorted(subject_evidence, key=_ev_order_key):
        kind = ev.fixture.get("kind")
        tags = tuple(sorted(ev.fixture.get("tags", ())))
        eng = normalize_engine_identity(ev.provenance.get("engine_identity"))
        eng_key = (eng.get("kind"), eng.get("revision"), eng.get("dirty"))
        comp_digest = ev.provenance.get("effective_comparison_sha256") or ""
        roles = tuple(ev.observations.get("output_roles") or ())
        cell_key = (kind, tags, eng_key, comp_digest, roles)
        cell_latest[cell_key] = ev

    current_evidence_list = sorted(cell_latest.values(), key=_ev_order_key, reverse=True)

    candidates: list[ClassificationCandidate] = []

    # Rule 1: Layout mismatch while logical values match
    for ev in current_evidence_list:
        if ev.result == "fail":
            obs = ev.observations
            if obs.get("layout_match") is False:
                ff = obs.get("first_failure") or {}
                mismatch_cnt = ff.get("mismatch_count")
                if mismatch_cnt is None and isinstance(obs.get("metrics"), Mapping):
                    mismatch_cnt = obs["metrics"].get("mismatch_count")
                has_explicit_zero_abs = (
                    "max_abs_error" in ff
                    and ff["max_abs_error"] is not None
                    and ff["max_abs_error"] == 0.0
                )
                logical_pass = (
                    mismatch_cnt == 0
                    or has_explicit_zero_abs
                    or obs.get("comparison_status") == "pass"
                )
                if logical_pass:
                    rule = "logical_pass_strict_layout_fails/v1"
                    cand_id = make_candidate_id(rule, [ev.id])
                    candidates.append(
                        ClassificationCandidate(
                            id=cand_id,
                            domain="layout",
                            confidence=0.95,
                            rule=rule,
                            supporting_evidence=(ev.id,),
                            limitations=("layout mismatch observed; numerical values match within tolerance",),
                        )
                    )
                    break

    # Rule 3 & 4: real_weight fails while synthetic and reference_derived pass
    # Group evidence by cross-fixture comparability equivalence class:
    # (subject, engine identity, effective comparison digest, output roles)
    compat_groups: dict[tuple[Any, ...], list[Evidence]] = {}
    for ev in subject_evidence:
        subj = ev.subject.get("boundary")
        eng = normalize_engine_identity(ev.provenance.get("engine_identity"))
        eng_key = (eng.get("kind"), eng.get("revision"), eng.get("dirty"))
        comp_digest = ev.provenance.get("effective_comparison_sha256") or ""
        # Semantic ordering of output roles must be preserved
        roles = tuple(ev.observations.get("output_roles") or ())
        compat_groups.setdefault((subj, eng_key, comp_digest, roles), []).append(ev)

    for (grp_subj, _grp_eng, grp_comp, grp_roles), ev_list in compat_groups.items():
        # Missing effective comparison or missing output-role contract cannot prove comparability
        if not grp_comp or not grp_roles:
            continue
        grp_latest_by_kind: dict[str, Evidence] = {}
        for ev in sorted(ev_list, key=_ev_order_key):
            grp_latest_by_kind[ev.fixture["kind"]] = ev

        ev_real = grp_latest_by_kind.get("real_weight")
        ev_synth = grp_latest_by_kind.get("synthetic")
        ev_ref = grp_latest_by_kind.get("reference_derived")

        if (
            ev_real is not None
            and ev_real.result == "fail"
            and ev_synth is not None
            and ev_synth.result == "pass"
            and ev_ref is not None
            and ev_ref.result == "pass"
        ):
            sup_ids = (ev_real.id, ev_synth.id, ev_ref.id)
            if "quantized" in ev_real.fixture.get("tags", []):
                rule = "quantized_real_fails_after_synthetic_passes/v1"
                candidates.append(
                    ClassificationCandidate(
                        id=make_candidate_id(rule, sup_ids),
                        domain="quantization",
                        confidence=0.75,
                        rule=rule,
                        supporting_evidence=tuple(sorted(sup_ids)),
                        limitations=("engine inputs were not observed",),
                    )
                )
            else:
                rule = "fixture_real_fails_after_synthetic_passes/v1"
                candidates.append(
                    ClassificationCandidate(
                        id=make_candidate_id(rule, sup_ids),
                        domain="model_artifact",
                        confidence=0.70,
                        rule=rule,
                        supporting_evidence=tuple(sorted(sup_ids)),
                        limitations=("engine inputs were not observed",),
                    )
                )
            break

    # Rule 2: Synthetic boundary fails after successful execution
    # Evaluated independently for current synthetic failure(s) without leaking loop variables
    synthetic_failures: list[Evidence] = []
    for ev in current_evidence_list:
        if ev.fixture.get("kind") == "synthetic" and ev.result == "fail":
            synthetic_failures.append(ev)

    for synth_ev in synthetic_failures:
        rule = "synthetic_boundary_fails_execution_succeeds/v1"
        cand = ClassificationCandidate(
            id=make_candidate_id(rule, [synth_ev.id]),
            domain="implementation",
            confidence=0.60,
            rule=rule,
            supporting_evidence=(synth_ev.id,),
            limitations=("engine inputs were not observed",),
        )
        if not any(c.id == cand.id for c in candidates):
            candidates.append(cand)

    # Fallback: Unknown
    if not candidates:
        failing_ids = [ev.id for ev in current_evidence_list if ev.result in ("fail", "error")]
        rule = "no_rule_adequate_support/v1"
        candidates.append(
            ClassificationCandidate(
                id=make_candidate_id(rule, failing_ids),
                domain="unknown",
                confidence=0.0,
                rule=rule,
                supporting_evidence=tuple(sorted(failing_ids)),
                limitations=("available observations do not match specific failure domain rules",),
            )
        )

    # Process user decision notes (rejections & overrides)
    decisions: list[UserDecision] = []
    rejected_ids: set[str] = set()

    for note_ev in filtered_notes:
        if note_ev.type != "manual_note" or not note_ev.note:
            continue
        cat = note_ev.note.get("category")
        reason = note_ev.note.get("reason", "")
        dur = note_ev.note.get("duration_minutes")

        if cat == "classification-reject":
            target = (
                note_ev.note.get("target")
                or note_ev.note.get("target_candidate")
                or note_ev.note.get("candidate_id", "")
            )
            if not target and "fc-" in reason:
                # Extract candidate id from reason if embedded
                for part in reason.split():
                    if part.startswith("fc-"):
                        target = part
                        break
            if target:
                rejected_ids.add(target)
                decisions.append(UserDecision(kind="reject", target=target, reason=reason, duration_minutes=dur))
        elif cat == "classification-override":
            domain = (
                note_ev.note.get("target")
                or note_ev.note.get("override_domain")
                or note_ev.note.get("domain", "implementation")
            )
            decisions.append(UserDecision(kind="override", target=domain, reason=reason, duration_minutes=dur))

    active_candidates = tuple(c for c in candidates if c.id not in rejected_ids)
    return active_candidates, tuple(decisions)
