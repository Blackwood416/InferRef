"""Framework-neutral operations exposed identically through Python, CLI, and MCP."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from inferref.agent.adapter import execute_adapter
from inferref.agent.protocol import (
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    AgentProtocolError,
    AgentResponse,
    EngineAdapter,
)
from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import TolerancePolicy
from inferref.inspect.analyze import analyze
from inferref.ir.package import TracePackage, is_trace_package
from inferref.ir.validate import validate_package
from inferref.ir.version import (
    FORMAT,
    FORMAT_VERSION,
    INFERREF_VERSION,
    TENSOR_FORMAT_VERSION,
    TESTCASE_FORMAT,
    TESTCASE_FORMAT_VERSION,
)
from inferref.scenario import run_scenario as run_scenario_artifact
from inferref.testcase.extract import ExtractionError, extract_operator, extract_region
from inferref.testcase.validate import validate_testcase


def capabilities() -> AgentResponse:
    """Describe stable operations before an Agent decides which tool to call."""

    operations = [
        {
            "name": "context",
            "mutates": False,
            "description": "Summarise and validate a trace or standalone testcase.",
        },
        {
            "name": "extract_testcase",
            "mutates": True,
            "description": "Project one operator or region into a portable testcase.",
        },
        {
            "name": "compare_outputs",
            "mutates": False,
            "description": "Compare engine outputs with a testcase reference.",
        },
        {
            "name": "run_engine",
            "mutates": True,
            "description": (
                "Execute a trusted shell-free engine adapter in a fresh run directory "
                "and compare its outputs."
            ),
        },
        {
            "name": "run_scenario",
            "mutates": True,
            "description": (
                "Execute an ordered chain of testcases with explicit state binding "
                "through one trusted engine adapter."
            ),
        },
    ]
    return AgentResponse(
        operation="capabilities",
        status="ok",
        data={
            "inferref_version": INFERREF_VERSION,
            "formats": {
                "trace": {"format": FORMAT, "version": FORMAT_VERSION},
                "testcase": {
                    "format": TESTCASE_FORMAT,
                    "version": TESTCASE_FORMAT_VERSION,
                },
                "scenario": {"format": "inferref-scenario", "version": "0.1"},
                "tensor": {"format": "irtensor", "version": TENSOR_FORMAT_VERSION},
                "engine_adapter": {
                    "format": ENGINE_ADAPTER_FORMAT,
                    "version": ENGINE_ADAPTER_VERSION,
                },
            },
            "operations": operations,
            "mcp_tools": [f"inferref_{item['name']}" for item in operations]
            + ["inferref_capabilities"],
        },
        next_actions=(
            {
                "operation": "context",
                "reason": "Inspect a trace or testcase before extracting or executing it.",
            },
        ),
    )


def context(path: str | Path) -> AgentResponse:
    """Return compact, actionable context for one InferRef artifact."""

    operation = "context"
    try:
        root = Path(path).resolve()
        if is_trace_package(root):
            return _trace_context(root)
        testcase_path = root / "testcase.json"
        if testcase_path.is_file():
            return _testcase_context(root)
        return AgentResponse.error(
            operation,
            f"{root} is neither a trace package nor an InferRef testcase",
            code="artifact_not_found",
        )
    except (OSError, ValueError) as exc:
        return AgentResponse.error(operation, str(exc), code="artifact_invalid")


def extract_testcase(
    trace: str | Path,
    output: str | Path,
    *,
    region: str | int | None = None,
    op_id: int | None = None,
    name: str | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    contracts: Sequence[str] | None = None,
) -> AgentResponse:
    """Extract one operator/region and return an Agent protocol envelope."""

    operation = "extract_testcase"
    try:
        if (region is None) == (op_id is None):
            raise AgentProtocolError("pass exactly one of region or op_id")
        package = TracePackage.load(trace)
        if region is not None:
            selected = package.region(region)
            if selected is None:
                raise AgentProtocolError(f"no region named or identified by {region!r}")
            result = extract_region(
                package,
                selected,
                output,
                name=name,
                input_names=input_names,
                output_names=output_names,
                contracts=contracts,
            )
        else:
            result = extract_operator(
                package,
                int(op_id),
                output,
                name=name,
                input_names=input_names,
                output_names=output_names,
                contracts=contracts,
            )
        status = "pass" if result.reproducible else "fail"
        actions: tuple[dict[str, Any], ...]
        if result.reproducible:
            actions = (
                {
                    "operation": "run_engine",
                    "testcase": str(result.path.resolve()),
                    "reason": "The testcase is independently reproducible.",
                },
            )
        else:
            actions = (
                {
                    "operation": "trace",
                    "reason": "Re-trace with full capture for missing boundary payloads.",
                },
            )
        return AgentResponse(
            operation=operation,
            status=status,
            data=result.to_dict(),
            diagnostics=tuple(
                {
                    "severity": "error",
                    "code": detail.get("reason", "missing_payload"),
                    "message": f"Boundary value {detail.get('value_id')} has no runnable payload.",
                    "detail": detail,
                }
                for detail in result.missing_payload_details
            ),
            next_actions=actions,
        )
    except (ExtractionError, OSError, ValueError, KeyError) as exc:
        return AgentResponse.error(operation, str(exc), code="extraction_failed")


def compare_outputs(
    testcase: str | Path,
    engine_output: str | Path,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> AgentResponse:
    """Compare one engine output directory and return the first actionable failure."""

    operation = "compare_outputs"
    try:
        policy = _policy(atol=atol, rtol=rtol)
        report = compare_testcase(
            testcase,
            engine_output,
            policy=policy,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
        )
        passed = report.status == "pass"
        actions = (
            ()
            if passed
            else (
                {
                    "operation": "modify_engine",
                    "reason": "Fix the reported first divergence, rebuild, and run again.",
                },
                {
                    "operation": "compare_outputs",
                    "reason": "Repeat until the structured comparison status is pass.",
                },
            )
        )
        return AgentResponse(
            operation=operation,
            status="pass" if passed else "fail",
            data=report.to_dict(),
            next_actions=actions,
        )
    except (OSError, ValueError) as exc:
        return AgentResponse.error(operation, str(exc), code="comparison_failed")


def run_engine(
    testcase: str | Path,
    adapter_path: str | Path,
    runs_root: str | Path,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> AgentResponse:
    """Execute a trusted adapter with timeout/output controls, then compare."""

    operation = "run_engine"
    try:
        adapter = EngineAdapter.load(adapter_path)
        result = execute_adapter(
            testcase,
            adapter,
            runs_root,
            policy=_policy(atol=atol, rtol=rtol),
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
        )
        status = result["status"]
        if status == "pass":
            response_status = "pass"
            actions: tuple[dict[str, Any], ...] = ()
        elif status == "mismatch":
            response_status = "fail"
            actions = (
                {
                    "operation": "modify_engine",
                    "reason": "Fix the first comparison divergence and rerun this adapter.",
                },
            )
        else:
            response_status = "error"
            actions = (
                {
                    "operation": "inspect_execution",
                    "reason": "Inspect adapter stderr, exit code, command, and cwd.",
                },
            )
        diagnostics = ()
        if response_status == "error":
            detail = result.get("comparison") or result.get("execution") or {}
            diagnostics = (
                {
                    "severity": "error",
                    "code": status,
                    "message": detail.get(
                        "message", "Engine adapter did not complete successfully."
                    ),
                },
            )
        return AgentResponse(
            operation=operation,
            status=response_status,
            data=result,
            diagnostics=diagnostics,
            next_actions=actions,
        )
    except (OSError, ValueError) as exc:
        return AgentResponse.error(operation, str(exc), code="adapter_failed")


def run_scenario(
    scenario: str | Path,
    adapter_path: str | Path,
    runs_root: str | Path,
    *,
    state_mode: str = "reference",
    compare_state: bool = False,
    atol: float | None = None,
    rtol: float | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> AgentResponse:
    """Execute a scenario chain and return the standard Agent envelope."""

    operation = "run_scenario"
    try:
        report = run_scenario_artifact(
            scenario,
            adapter_path,
            runs_root,
            state_mode=state_mode,
            compare_state=compare_state,
            atol=atol,
            rtol=rtol,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
        )
    except (OSError, ValueError) as exc:
        return AgentResponse.error(operation, str(exc), code="scenario_failed")
    status = report["status"]
    if status == "pass":
        response_status = "pass"
    elif status == "fail":
        response_status = "fail"
    else:
        response_status = "error"
    failed_steps = [step for step in report["steps"] if step["status"] != "pass"]
    actions: tuple[dict[str, Any], ...]
    if not failed_steps:
        actions = ()
    elif response_status == "fail":
        actions = (
            {
                "operation": "run_scenario",
                "step": failed_steps[0]["id"],
                "reason": "Fix the first failing scenario step and rerun this adapter.",
            },
        )
    else:
        actions = (
            {
                "operation": "run_scenario",
                "step": failed_steps[0]["id"],
                "reason": "Inspect the first errored step's run record and adapter stderr.",
            },
        )
    diagnostics = ()
    if response_status != "pass":
        step = failed_steps[0]
        detail = step.get("run") or {}
        diagnostics = (
            {
                "severity": "error" if response_status == "error" else "warning",
                "code": step.get("state_status", detail.get("status", status)),
                "message": (
                    f"Scenario step {step['id']!r} finished with status "
                    f"{step['status']!r}."
                ),
            },
        )
    return AgentResponse(
        operation=operation,
        status=response_status,
        data=report,
        diagnostics=diagnostics,
        next_actions=actions,
    )


def _trace_context(root: Path) -> AgentResponse:
    package = TracePackage.load(root)
    analysis = analyze(package).to_dict()
    issues = validate_package(package)
    errors = [issue for issue in issues if issue.severity == "error"]
    diagnostics = tuple(
        {
            "severity": issue.severity,
            "code": f"trace_invariant_{issue.invariant}",
            "message": issue.message,
            "where": issue.where,
            "invariant": issue.invariant,
        }
        for issue in issues
    )
    actions: list[dict[str, Any]] = []
    if errors:
        actions.append(
            {
                "operation": "validate",
                "reason": "Repair Trace IR errors before extracting testcases.",
            }
        )
    elif not package.regions:
        actions.append(
            {
                "operation": "region_detect",
                "reason": "Discover semantic/fused boundaries before extraction.",
            }
        )
    else:
        actions.append(
            {
                "operation": "extract_testcase",
                "reason": "Select a reproducible region to hand to an engine adapter.",
            }
        )
    if analysis["coverage"]["payload"] < 1.0:
        actions.append(
            {
                "operation": "trace",
                "reason": "Re-trace with full capture if the selected boundary lacks payloads.",
            }
        )
    return AgentResponse(
        operation="context",
        status="ok" if not errors else "fail",
        data={
            "artifact": "trace",
            "path": str(root),
            "analysis": analysis,
            "validation": {
                "status": "pass" if not errors else "fail",
                "errors": len(errors),
                "warnings": len(issues) - len(errors),
            },
            "regions": [region.to_dict() for region in package.regions],
        },
        diagnostics=diagnostics,
        next_actions=tuple(actions),
    )


def _testcase_context(root: Path) -> AgentResponse:
    validation = validate_testcase(root)
    manifest = validation.manifest
    reproducible = validation.reproducible
    diagnostics = tuple(issue.to_dict() for issue in validation.issues)
    action = (
        {
            "operation": "run_engine",
            "reason": "Execute a trusted adapter against this isolated testcase.",
        }
        if reproducible
        else {
            "operation": "trace",
            "reason": "Recreate this testcase from a trace with complete boundary capture.",
        }
    )
    return AgentResponse(
        operation="context",
        status="error" if not validation.valid else ("ok" if reproducible else "fail"),
        data={
            "artifact": "testcase",
            "path": str(root),
            "name": manifest.get("name", ""),
            "format_version": manifest.get("format_version"),
            "reproducible": reproducible,
            "validation": validation.to_dict(),
            "origin": manifest.get("origin") or {},
            "inputs": manifest.get("inputs") or [],
            "outputs": manifest.get("outputs") or [],
            "nodes": manifest.get("nodes") or [],
            "values": manifest.get("values") or [],
        },
        diagnostics=diagnostics,
        next_actions=(action,) if validation.valid else (),
    )


def _policy(*, atol: float | None, rtol: float | None) -> TolerancePolicy:
    policy = TolerancePolicy()
    policy.override_atol = atol
    policy.override_rtol = rtol
    return policy
