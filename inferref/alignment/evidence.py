"""Evidence normalization and comparability for Alignment Session v0.1 (L3).

Evidence v0.1 records immutable observations from engine execution and comparison.
It strictly observes:
- provided_inputs_valid: True when the testcase was validated
- engine_inputs_observed: False (0.10 has no engine input observation channel)
- outputs_match: True when comparison status is pass
- layout_match: True when layout checks passed
- first_failure: Details of the first numerical divergence, if any

0.10 NEVER infers or claims `inputs_match` without observed engine inputs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from inferref.alignment.schema import (
    Evidence,
    make_evidence_id,
    utc_now,
)


def normalize_engine_identity(identity: Any) -> dict[str, Any]:
    """Normalize engine identity dict for deterministic comparisons."""
    if not isinstance(identity, Mapping):
        return {"kind": "unknown", "revision": str(identity or "unknown"), "dirty": False}
    return {
        "kind": str(identity.get("kind", "unknown")),
        "revision": str(identity.get("revision", "unknown")),
        "dirty": bool(identity.get("dirty", False)),
    }


def engine_identities_equal(a: Any, b: Any) -> bool:
    norm_a = normalize_engine_identity(a)
    norm_b = normalize_engine_identity(b)
    return (
        norm_a["kind"] == norm_b["kind"]
        and norm_a["revision"] == norm_b["revision"]
        and norm_a["dirty"] == norm_b["dirty"]
    )


def extract_observations(
    *,
    comparison_report: Mapping[str, Any] | None = None,
    execution_result: Mapping[str, Any] | None = None,
    provided_inputs_valid: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Extract result and observations dictionary from run/comparison outputs."""
    # 0.10 rule: engine_inputs_observed is always False for native runs.
    engine_inputs_observed = False

    if execution_result is not None:
        exec_status = str(execution_result.get("status", ""))
        if exec_status == "unsupported":
            return "unsupported", {
                "provided_inputs_valid": provided_inputs_valid,
                "engine_inputs_observed": engine_inputs_observed,
                "outputs_match": False,
                "layout_match": False,
                "first_failure": None,
                "diagnostics": [{"code": "unsupported", "reasons": list(execution_result.get("unsupported", []))}],
            }
        if exec_status in ("error", "failed", "crash"):
            return "error", {
                "provided_inputs_valid": provided_inputs_valid,
                "engine_inputs_observed": engine_inputs_observed,
                "outputs_match": False,
                "layout_match": False,
                "first_failure": None,
            }

    if comparison_report is None:
        return "error", {
            "provided_inputs_valid": provided_inputs_valid,
            "engine_inputs_observed": engine_inputs_observed,
            "outputs_match": False,
            "layout_match": False,
            "first_failure": None,
        }

    status = str(comparison_report.get("status", "fail"))
    result = "pass" if status == "pass" else "fail"
    outputs_match = result == "pass"

    # Check layout match and collect diagnostics
    layout_match = True
    diag_list: list[dict[str, Any]] = []

    # 1. Check layout from first_failure
    raw_ff = comparison_report.get("first_failure")
    if isinstance(raw_ff, Mapping) and isinstance(raw_ff.get("layout"), Mapping):
        ld = raw_ff["layout"]
        if (
            ld.get("dtype_mismatch")
            or ld.get("shape_mismatch")
            or ld.get("stride_mismatch")
            or ld.get("storage_offset_mismatch")
        ):
            layout_match = False
            diag_list.append({"code": "layout_mismatch", "detail": "first_failure layout mismatch"})

    # 2. Check layout from comparisons list (ComparisonReport schema)
    comparisons = comparison_report.get("comparisons")
    if isinstance(comparisons, Sequence):
        for comp in comparisons:
            if isinstance(comp, Mapping) and isinstance(comp.get("layout"), Mapping):
                ld = comp["layout"]
                if (
                    ld.get("dtype_mismatch")
                    or ld.get("shape_mismatch")
                    or ld.get("stride_mismatch")
                    or ld.get("storage_offset_mismatch")
                ):
                    layout_match = False
                    diag_list.append(
                        {"code": "layout_mismatch", "detail": f"comparison {comp.get('name')} layout mismatch"}
                    )

    # 3. Check raw diagnostics if present
    raw_diag = comparison_report.get("diagnostics")
    if isinstance(raw_diag, Mapping):
        if raw_diag.get("layout_mismatch") or raw_diag.get("layout_issues"):
            layout_match = False
        for k, v in raw_diag.items():
            diag_list.append({"code": str(k), "detail": v})
    elif isinstance(raw_diag, Sequence):
        for diag in raw_diag:
            if isinstance(diag, Mapping):
                diag_list.append(dict(diag))
                if "layout" in str(diag.get("code", "")).lower():
                    layout_match = False

    # Extract first failure
    first_failure: dict[str, Any] | None = None
    if isinstance(raw_ff, Mapping):
        first_failure = {}
        out_name = raw_ff.get("name") or raw_ff.get("output") or raw_ff.get("tensor")
        if out_name is not None:
            first_failure["output"] = str(out_name)

        ff_metrics = raw_ff.get("metrics") if isinstance(raw_ff.get("metrics"), Mapping) else raw_ff
        if isinstance(ff_metrics, Mapping):
            if "max_abs_error" in ff_metrics and ff_metrics["max_abs_error"] is not None:
                first_failure["max_abs_error"] = float(ff_metrics["max_abs_error"])
            if "max_rel_error" in ff_metrics and ff_metrics["max_rel_error"] is not None:
                first_failure["max_rel_error"] = float(ff_metrics["max_rel_error"])
            if "mismatch_count" in ff_metrics and ff_metrics["mismatch_count"] is not None:
                first_failure["mismatch_count"] = int(ff_metrics["mismatch_count"])
            if "index" in ff_metrics and ff_metrics["index"] is not None:
                first_failure["index"] = int(ff_metrics["index"])

        if "message" in raw_ff and raw_ff["message"] is not None:
            first_failure["message"] = str(raw_ff["message"])
    elif status in ("fail", "error") and isinstance(comparisons, Sequence):
        for comp in comparisons:
            if isinstance(comp, Mapping) and (not comp.get("passed", True) or comp.get("status") in ("fail", "error")):
                first_failure = {}
                out_name = comp.get("name") or comp.get("output") or comp.get("tensor")
                if out_name is not None:
                    first_failure["output"] = str(out_name)
                comp_metrics = comp.get("metrics") if isinstance(comp.get("metrics"), Mapping) else comp
                if isinstance(comp_metrics, Mapping):
                    if "max_abs_error" in comp_metrics and comp_metrics["max_abs_error"] is not None:
                        first_failure["max_abs_error"] = float(comp_metrics["max_abs_error"])
                    if "max_rel_error" in comp_metrics and comp_metrics["max_rel_error"] is not None:
                        first_failure["max_rel_error"] = float(comp_metrics["max_rel_error"])
                    if "mismatch_count" in comp_metrics and comp_metrics["mismatch_count"] is not None:
                        first_failure["mismatch_count"] = int(comp_metrics["mismatch_count"])
                    if "index" in comp_metrics and comp_metrics["index"] is not None:
                        first_failure["index"] = int(comp_metrics["index"])
                if "message" in comp and comp["message"] is not None:
                    first_failure["message"] = str(comp["message"])
                break
    elif status == "fail":
        results = comparison_report.get("results")
        if isinstance(results, Mapping):
            for out_name, out_res in sorted(results.items()):
                if isinstance(out_res, Mapping) and out_res.get("status") == "fail":
                    first_failure = {
                        "output": str(out_name),
                        "max_abs_error": float(out_res.get("max_abs_error", 0.0)),
                    }
                    if "max_rel_error" in out_res:
                        first_failure["max_rel_error"] = float(out_res["max_rel_error"])
                    if "mismatch_count" in out_res and out_res["mismatch_count"] is not None:
                        first_failure["mismatch_count"] = int(out_res["mismatch_count"])
                    break

    output_roles: list[str] = []
    if isinstance(comparison_report.get("output_roles"), Sequence):
        output_roles = [str(r) for r in comparison_report["output_roles"]]
    elif isinstance(comparisons, Sequence):
        output_roles = [str(c["name"]) for c in comparisons if isinstance(c, Mapping) and "name" in c]
    elif isinstance(comparison_report.get("results"), Mapping):
        output_roles = sorted(str(k) for k in comparison_report["results"])

    obs: dict[str, Any] = {
        "provided_inputs_valid": bool(provided_inputs_valid),
        "engine_inputs_observed": engine_inputs_observed,
        "outputs_match": outputs_match,
        "layout_match": layout_match,
        "first_failure": first_failure,
    }
    if output_roles:
        obs["output_roles"] = output_roles
    if diag_list:
        obs["diagnostics"] = diag_list
    obs_metrics: dict[str, Any] = {}
    if isinstance(comparison_report.get("metrics"), Mapping):
        obs_metrics.update(comparison_report["metrics"])
    elif first_failure and "mismatch_count" in first_failure:
        obs_metrics["mismatch_count"] = first_failure["mismatch_count"]
    if obs_metrics:
        obs["metrics"] = obs_metrics
    return result, obs


def build_comparison_evidence(
    *,
    subject: Mapping[str, str],
    fixture: Mapping[str, Any],
    provenance: Mapping[str, Any],
    comparison_report: Mapping[str, Any] | None = None,
    execution_result: Mapping[str, Any] | None = None,
    provided_inputs_valid: bool = True,
    evidence_id: str | None = None,
    created_at: str | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> Evidence:
    """Build and validate a numerical comparison Evidence record."""
    result, observations = extract_observations(
        comparison_report=comparison_report,
        execution_result=execution_result,
        provided_inputs_valid=provided_inputs_valid,
    )

    ev_id = evidence_id or make_evidence_id()
    timestamp = created_at or utc_now()

    clean_provenance = {
        "trace_metadata_sha256": str(provenance["trace_metadata_sha256"]),
        "testcase": {
            "path": str(provenance["testcase"]["path"]),
            "tree_sha256": str(provenance["testcase"]["tree_sha256"]),
        },
        "run_record": {
            "path": str(provenance["run_record"]["path"]),
            "sha256": str(provenance["run_record"]["sha256"]),
        },
        "adapter_normalized_sha256": str(provenance["adapter_normalized_sha256"]),
        "engine_identity": normalize_engine_identity(provenance["engine_identity"]),
        "effective_comparison_sha256": str(provenance["effective_comparison_sha256"]),
    }

    clean_subject = {
        "boundary": str(subject["boundary"]),
        "name": str(subject.get("name", subject["boundary"])),
    }
    clean_fixture = {
        "kind": str(fixture["kind"]),
        "tags": [str(t) for t in fixture.get("tags", [])],
    }

    evidence = Evidence(
        id=ev_id,
        created_at=timestamp,
        type="numerical_comparison",
        subject=clean_subject,
        fixture=clean_fixture,
        result=result,
        observations=observations,
        provenance=clean_provenance,
        extensions=dict(extensions or {}),
    )
    evidence.validate()
    return evidence


def build_manual_note_evidence(
    *,
    subject: Mapping[str, str],
    category: str,
    reason: str,
    duration_minutes: float | None = None,
    milestone: str | None = None,
    target: str | None = None,
    evidence_id: str | None = None,
    created_at: str | None = None,
    extensions: Mapping[str, Any] | None = None,
) -> Evidence:
    """Build and validate a manual note Evidence record."""
    ev_id = evidence_id or make_evidence_id()
    timestamp = created_at or utc_now()
    note_payload: dict[str, Any] = {
        "category": str(category),
        "reason": str(reason),
    }
    if duration_minutes is not None:
        note_payload["duration_minutes"] = float(duration_minutes)
    if milestone is not None:
        note_payload["milestone"] = str(milestone)
    if target is not None:
        note_payload["target"] = str(target)

    clean_subject = {
        "boundary": str(subject["boundary"]),
        "name": str(subject.get("name", subject["boundary"])),
    }
    evidence = Evidence(
        id=ev_id,
        created_at=timestamp,
        type="manual_note",
        subject=clean_subject,
        fixture={"kind": "other", "tags": []},
        result="recorded",
        observations={},
        provenance={},
        note=note_payload,
        extensions=dict(extensions or {}),
    )
    evidence.validate()
    return evidence


def comparability_key(evidence: Evidence) -> tuple[Any, ...]:
    """Return the equivalence class key for comparable evidence.

    Comparable evidence has the same subject, fixture kind/tags,
    testcase tree digest, adapter digest, engine identity, and
    effective comparison policy digest.
    """
    if evidence.type != "numerical_comparison":
        return (evidence.type, evidence.id)

    prov = evidence.provenance
    eng = normalize_engine_identity(prov.get("engine_identity"))
    return (
        evidence.subject["boundary"],
        evidence.fixture["kind"],
        tuple(sorted(evidence.fixture.get("tags", ()))),
        prov.get("testcase", {}).get("tree_sha256"),
        prov.get("adapter_normalized_sha256"),
        (eng["kind"], eng["revision"], eng["dirty"]),
        prov.get("effective_comparison_sha256"),
    )


def is_comparable(a: Evidence, b: Evidence) -> bool:
    """Determine whether two evidence records belong to the same comparability class."""
    if a.type != "numerical_comparison" or b.type != "numerical_comparison":
        return False
    return comparability_key(a) == comparability_key(b)
