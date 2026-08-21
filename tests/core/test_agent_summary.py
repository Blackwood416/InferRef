"""Tests for inferref-agent-summary v0.1 format and --json-summary CLI option (SPEC §9)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from inferref.agent.summary import (
    AGENT_SUMMARY_FORMAT,
    AGENT_SUMMARY_FORMAT_VERSION,
    build_agent_summary,
    summarize_report,
)
from inferref.cli.main import build_parser, main


def test_build_agent_summary_structure():
    summary = build_agent_summary(
        operation="run_engine",
        status="fail",
        first_failure={"output": "boxes", "message": "unmatched detection"},
        metrics={"reference_count": 10, "matched": 8},
        artifacts={"run_record": "runs/abc/inferref-run.json"},
        next_actions=[{"operation": "modify_engine", "reason": "Fix mismatch"}],
    )
    assert summary["format"] == AGENT_SUMMARY_FORMAT
    assert summary["format_version"] == AGENT_SUMMARY_FORMAT_VERSION
    assert summary["status"] == "fail"
    assert summary["operation"] == "run_engine"
    assert summary["first_failure"]["output"] == "boxes"
    assert summary["metrics"]["matched"] == 8
    assert summary["artifacts"]["run_record"] == "runs/abc/inferref-run.json"
    assert len(summary["next_actions"]) == 1


def test_summarize_report_numeric():
    numeric_report = {
        "status": "fail",
        "summary": {
            "total_tensors": 2,
            "passed_tensors": 1,
            "total_elements": 100,
            "mismatched_elements": 5,
            "max_observed_atol": 0.05,
            "max_observed_rtol": 0.1,
        },
        "first_divergence": {
            "tensor_name": "output_0",
            "reason": "tolerance exceeded",
        },
    }
    summary = summarize_report("compare", numeric_report)
    assert summary["format"] == AGENT_SUMMARY_FORMAT
    assert summary["status"] == "fail"
    assert summary["first_failure"]["output"] == "output_0"
    assert summary["metrics"]["total_elements"] == 100
    assert summary["metrics"]["max_observed_atol"] == 0.05


def test_cli_json_and_json_summary_mutually_exclusive():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["compare", "ref", "act", "--json", "--json-summary"])
