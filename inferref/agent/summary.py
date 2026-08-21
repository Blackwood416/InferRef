"""Agent JSON summary generation for inferref-agent-summary v0.1 (SPEC §9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

AGENT_SUMMARY_FORMAT = "inferref-agent-summary"
AGENT_SUMMARY_FORMAT_VERSION = "0.1"


def build_agent_summary(
    operation: str,
    status: str,
    *,
    first_failure: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    next_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Construct an inferref-agent-summary v0.1 payload."""
    payload: dict[str, Any] = {
        "format": AGENT_SUMMARY_FORMAT,
        "format_version": AGENT_SUMMARY_FORMAT_VERSION,
        "status": status,
        "operation": operation,
        "first_failure": first_failure,
        "metrics": metrics if metrics is not None else {},
        "artifacts": artifacts if artifacts is not None else {},
        "next_actions": next_actions if next_actions is not None else [],
    }
    return payload


def summarize_report(
    operation: str,
    report: dict[str, Any] | Any,
    *,
    run_record: str | None = None,
    artifacts: dict[str, Any] | None = None,
    next_actions: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Derive an agent summary from a compare report, agent response, or run record."""
    data = report if isinstance(report, dict) else (report.to_dict() if hasattr(report, "to_dict") else {})
    status = data.get("status", "error")
    # Normalize status to pass | fail | error
    if status in {"ok", "pass"}:
        norm_status = "pass"
    elif status in {"mismatch", "fail"}:
        norm_status = "fail"
    else:
        norm_status = "error"

    first_failure: dict[str, Any] | None = None
    metrics: dict[str, Any] = {}

    # Check if this run used a custom comparator
    comparator_info = data.get("comparator")
    comparison_info = data.get("comparison")

    if isinstance(comparator_info, dict) and "metrics" in comparator_info:
        metrics = dict(comparator_info.get("metrics") or {})
        first_failure = comparator_info.get("first_failure")
    elif isinstance(comparison_info, dict) and comparison_info.get("comparator") and "metrics" in comparison_info:
        metrics = dict(comparison_info.get("metrics") or {})
        first_failure = comparison_info.get("first_failure")
    else:
        # Numeric run metrics
        summary = data.get("summary") or (comparison_info.get("summary") if isinstance(comparison_info, dict) else None)
        if isinstance(summary, dict):
            metrics = {
                "total_tensors": summary.get("total_tensors", 0),
                "matched_tensors": summary.get("passed_tensors", 0),
                "total_elements": summary.get("total_elements", 0),
                "mismatched_elements": summary.get("mismatched_elements", 0),
                "max_observed_atol": summary.get("max_observed_atol", 0.0),
                "max_observed_rtol": summary.get("max_observed_rtol", 0.0),
            }
        # First failure in numeric comparison
        ff = data.get("first_divergence") or data.get("first_failure") or (comparison_info.get("first_divergence") if isinstance(comparison_info, dict) else None)
        if isinstance(ff, dict):
            first_failure = {
                "output": ff.get("output") or ff.get("tensor_name") or ff.get("name") or "unknown",
                "message": ff.get("message") or ff.get("reason") or "numerical tolerance exceeded",
            }
        elif norm_status == "fail" and not first_failure:
            first_failure = {
                "output": "output",
                "message": "output comparison failed",
            }

    art_dict = dict(artifacts or {})
    if run_record:
        art_dict["run_record"] = run_record
    elif "output" in data and (Path(data["output"]) / "inferref-run.json").is_file():
        art_dict["run_record"] = str(Path(data["output"]) / "inferref-run.json")

    actions = next_actions
    if actions is None:
        raw_actions = data.get("next_actions")
        if isinstance(raw_actions, (list, tuple)):
            actions = list(raw_actions)
        elif norm_status == "pass":
            actions = []
        elif norm_status == "fail":
            actions = [
                {
                    "operation": "inspect_first_divergence",
                    "reason": "Fix the first comparison failure and rerun.",
                }
            ]
        else:
            actions = [
                {
                    "operation": "inspect_execution",
                    "reason": "Inspect adapter stderr, exit code, and execution environment.",
                }
            ]

    return build_agent_summary(
        operation=operation,
        status=norm_status,
        first_failure=first_failure,
        metrics=metrics,
        artifacts=art_dict,
        next_actions=actions,
    )
