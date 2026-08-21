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
        comparisons_list = data.get("comparisons") or (comparison_info.get("comparisons") if isinstance(comparison_info, dict) else [])
        if isinstance(summary, dict):
            total_tensors = summary.get("compared") if "compared" in summary else summary.get("total_tensors", 0)
            matched_tensors = summary.get("passed") if "passed" in summary else summary.get("passed_tensors", 0)
            total_elements = int(summary.get("total_elements", 0))
            mismatched_elements = int(summary.get("mismatched_elements", 0))
            max_observed_atol = float(summary.get("max_observed_atol", 0.0))
            max_observed_rtol = float(summary.get("max_observed_rtol", 0.0))

            if isinstance(comparisons_list, list) and comparisons_list:
                comp_total = 0
                comp_mismatched = 0
                comp_max_atol = 0.0
                comp_max_rtol = 0.0
                has_comp_metrics = False
                for comp in comparisons_list:
                    if isinstance(comp, dict):
                        comp_metrics = comp.get("metrics")
                        if isinstance(comp_metrics, dict):
                            has_comp_metrics = True
                            comp_total += int(comp_metrics.get("element_count", 0))
                            comp_mismatched += int(comp_metrics.get("mismatch_count", 0))
                            abs_err = comp_metrics.get("max_abs_error") if "max_abs_error" in comp_metrics else comp_metrics.get("max_abs_diff", 0.0)
                            rel_err = comp_metrics.get("max_rel_error") if "max_rel_error" in comp_metrics else comp_metrics.get("max_rel_diff", 0.0)
                            comp_max_atol = max(comp_max_atol, float(abs_err or 0.0))
                            comp_max_rtol = max(comp_max_rtol, float(rel_err or 0.0))
                if has_comp_metrics:
                    total_elements = comp_total
                    mismatched_elements = comp_mismatched
                    max_observed_atol = comp_max_atol
                    max_observed_rtol = comp_max_rtol

            metrics = {
                "total_tensors": total_tensors,
                "matched_tensors": matched_tensors,
                "total_elements": total_elements,
                "mismatched_elements": mismatched_elements,
                "max_observed_atol": max_observed_atol,
                "max_observed_rtol": max_observed_rtol,
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
