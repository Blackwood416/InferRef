"""Schema/runnable split for scenario manifests (SPEC §5)."""

from __future__ import annotations

import json
from typing import Any

from inferref.scenario.schema import (
    SCENARIO_FORMAT_VERSION,
    ScenarioError,
    load_scenario,
)
from inferref.testcase.validate import validate_testcase


def validate_scenario(
    path: str, *, allow_nonreproducible: bool = False
) -> dict[str, Any]:
    """Validate one scenario, separating structure from runnability.

    ``schema_valid`` means the manifest and every referenced testcase pass
    structural validation. ``runnable`` additionally requires every referenced
    testcase to be reproducible.
    """

    try:
        scenario = load_scenario(path)
    except (ScenarioError, OSError, json.JSONDecodeError) as exc:
        issues = list(exc.issues) if isinstance(exc, ScenarioError) else []
        if not issues:
            issues = [
                {
                    "code": "scenario_invalid_manifest",
                    "message": str(exc),
                    "where": "",
                }
            ]
        issues = [{"severity": "error", **issue} for issue in issues]
        return {
            "format": "inferref-scenario-validation",
            "format_version": SCENARIO_FORMAT_VERSION,
            "status": "fail",
            "schema_valid": False,
            "runnable": False,
            "non_runnable_steps": [],
            "issues": issues,
            "error": str(exc),
        }

    non_runnable: list[str] = []
    issues: list[dict[str, Any]] = []
    for step in scenario.steps:
        validation = validate_testcase(step.testcase)
        if not validation.reproducible:
            non_runnable.append(step.id)
            issues.append(
                {
                    "severity": "warning",
                    "code": "scenario_testcase_nonreproducible",
                    "message": (
                        f"step {step.id!r} testcase {step.testcase_ref!r} is not "
                        "reproducible"
                    ),
                    "where": f"steps[{step.id}].testcase",
                    "blocks_reproduction": True,
                }
            )
    runnable = not non_runnable
    return {
        "format": "inferref-scenario-validation",
        "format_version": SCENARIO_FORMAT_VERSION,
        "status": "pass" if runnable else "fail",
        "schema_valid": True,
        "runnable": runnable,
        "non_runnable_steps": non_runnable,
        "allow_nonreproducible": allow_nonreproducible,
        "issues": issues,
        "scenario_id": scenario.id,
        "steps": len(scenario.steps),
    }


__all__ = ["validate_scenario"]
