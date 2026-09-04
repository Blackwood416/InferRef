"""Framework-neutral operations exposed identically through Python, CLI, and MCP."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
        {
            "name": "alignment_init",
            "mutates": True,
            "description": "Initialize an InferRef alignment session.",
        },
        {
            "name": "alignment_plan",
            "mutates": False,
            "description": "Inspect alignment candidates and recommendation.",
        },
        {
            "name": "alignment_run",
            "mutates": True,
            "description": "Execute alignment boundaries against an engine adapter.",
        },
        {
            "name": "alignment_status",
            "mutates": False,
            "description": "Inspect alignment session progress and active failure.",
        },
        {
            "name": "alignment_summary",
            "mutates": False,
            "description": "Summarize alignment metrics, TFCM, and adoption ratio.",
        },
        {
            "name": "alignment_narrow",
            "mutates": False,
            "description": "Compute smallest confirmed failing boundary and diagnosis.",
        },
        {
            "name": "alignment_note",
            "mutates": True,
            "description": "Record human/agent manual note evidence.",
        },
        {
            "name": "alignment_report",
            "mutates": True,
            "description": "Generate markdown and JSON sidecar alignment reports.",
        },
    ]
    return AgentResponse(
        operation="capabilities",
        status="ok",
        data={
            "inferref_version": INFERREF_VERSION,
            "formats": {
                "trace": {"format": FORMAT, "version": FORMAT_VERSION},
                "alignment": {"format": "inferref-alignment", "version": "0.1"},
                "evidence": {"format": "inferref-evidence", "version": "0.1"},
                "testcase": {
                    "format": TESTCASE_FORMAT,
                    "version": TESTCASE_FORMAT_VERSION,
                    "writes": ["0.2", "0.3"],
                    "reads": ["0.1", "0.2", "0.3"],
                },
                "suite": {
                    "format": "inferref-suite",
                    "version": "0.2",
                    "writes": ["0.2", "0.3"],
                    "reads": ["0.1", "0.2", "0.3"],
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
    comparator: str | None = None,
    comparison_config: dict[str, Any] | None = None,
    tolerance: str | Path | dict[str, Any] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> AgentResponse:
    """Compare one engine output directory and return the first actionable failure."""

    operation = "compare_outputs"
    try:
        policy = _policy(atol=atol, rtol=rtol, tolerance=tolerance)
        report = compare_testcase(
            testcase,
            engine_output,
            policy=policy,
            ignore_stride=ignore_stride,
            strict_layout=strict_layout,
            first_failure=first_failure,
            comparator=comparator,
            comparison_config=comparison_config,
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
    comparator: str | None = None,
    comparison_config: dict[str, Any] | None = None,
    tolerance: str | Path | dict[str, Any] | None = None,
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
            comparator=comparator,
            comparison_config=comparison_config,
            tolerance=tolerance,
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
    comparator: str | None = None,
    comparison_config: dict[str, Any] | None = None,
    tolerance: str | Path | dict[str, Any] | None = None,
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
            comparator=comparator,
            comparison_config=comparison_config,
            tolerance=tolerance,
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


def _policy(
    *,
    atol: float | None = None,
    rtol: float | None = None,
    tolerance: str | Path | dict[str, Any] | None = None,
) -> TolerancePolicy:
    if tolerance is not None:
        if isinstance(tolerance, (str, Path)):
            policy = TolerancePolicy.load(tolerance)
        elif isinstance(tolerance, dict):
            policy = TolerancePolicy.from_dict(tolerance)
        else:
            policy = TolerancePolicy()
    else:
        policy = TolerancePolicy()
    if atol is not None:
        policy.override_atol = atol
    if rtol is not None:
        policy.override_rtol = rtol
    return policy


# -- Alignment Service Functions (0.10 L7) ----------------------------------


def _is_path_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "outside configured allowed roots" in msg or "not under an allowed" in msg or "allowed root" in msg


def alignment_init(
    trace_path: str | Path,
    adapter_path: str | Path,
    *,
    session_dir: str | Path | None = None,
    name: str | None = None,
    force: bool = False,
    access_policy: Any = None,
) -> AgentResponse:
    """Initialize an InferRef alignment session."""
    import shutil

    from inferref.alignment import (
        AlignmentStore,
        normalized_adapter_sha256,
        trace_metadata_sha256,
    )

    operation = "alignment_init"
    try:
        t_path = Path(trace_path).resolve()
        a_path = Path(adapter_path).resolve()
        s_dir = Path(session_dir).resolve() if session_dir else Path.cwd() / "alignment"

        if access_policy is not None:
            t_path = Path(access_policy(t_path, "trace input")).resolve()
            a_path = Path(access_policy(a_path, "adapter")).resolve()
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()

        pkg = TracePackage.load(t_path)
        adapter = EngineAdapter.load(a_path)
        trace_sha = trace_metadata_sha256(pkg.root)
        adapter_sha = normalized_adapter_sha256(adapter.to_dict())

        if force and s_dir.exists():
            shutil.rmtree(s_dir, ignore_errors=True)

        session_id = None
        if name:
            session_id = name if name.startswith("al-") else f"al-{name}"

        store = AlignmentStore.initialize(
            s_dir,
            trace_path=t_path,
            adapter_path=a_path,
            trace_metadata_digest=trace_sha,
            adapter_digest=adapter_sha,
            session_id=session_id,
        )
        return AgentResponse(
            operation=operation,
            status="pass",
            data=store.snapshot.to_dict(),
            next_actions=(
                {
                    "operation": "alignment_plan",
                    "reason": "Inspect boundary candidates and recommendation.",
                },
            ),
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


align_init = alignment_init


def alignment_plan(
    session_dir: str | Path | None = None,
    *,
    trace_path: str | Path | None = None,
    adapter_path: str | Path | None = None,
    access_policy: Any = None,
) -> AgentResponse:
    """Inspect alignment candidates and recommended boundary."""
    from inferref.alignment import AlignmentStore, plan_alignment

    operation = "alignment_plan"
    try:
        s_dir = Path(session_dir).resolve() if session_dir else Path.cwd() / "alignment"
        if access_policy is not None and (session_dir or (s_dir / "alignment.json").is_file()):
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()

        store = AlignmentStore.load(s_dir) if (s_dir / "alignment.json").is_file() else None

        t_path = (
            Path(trace_path).resolve()
            if trace_path
            else (Path(store.snapshot.source["trace"]["path"]).resolve() if store else None)
        )
        if not t_path or not t_path.exists():
            return AgentResponse.error(
                operation, "trace path required or session must contain valid trace"
            )

        a_path = (
            Path(adapter_path).resolve()
            if adapter_path
            else (Path(store.snapshot.target["adapter"]["path"]).resolve() if store else None)
        )

        if access_policy is not None:
            t_path = Path(access_policy(t_path, "trace input")).resolve()
            if a_path:
                a_path = Path(access_policy(a_path, "adapter")).resolve()

        adapter = EngineAdapter.load(a_path) if a_path and a_path.is_file() else None
        pkg = TracePackage.load(t_path)
        plan = plan_alignment(pkg, session=store.snapshot if store else None, adapter=adapter)
        recommended = plan.candidates[0].boundary if plan.candidates else None
        data = plan.to_dict()
        data["recommended_boundary"] = recommended
        actions = (
            (
                {
                    "operation": "alignment_run",
                    "reason": f"Run recommended boundary {recommended}",
                },
            )
            if recommended
            else ()
        )
        return AgentResponse(
            operation=operation,
            status="ok",
            data=data,
            next_actions=actions,
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


align_plan = alignment_plan


def alignment_run(
    session_dir: str | Path,
    adapter: str | Path,
    *,
    limit: int = 1,
    boundary_ids: Sequence[str] = (),
    continue_on_fail: bool = False,
    engine_revision: str | None = None,
    comparator_overrides: Mapping[str, Any] | None = None,
    fixture_kind: str | None = None,
    fixture_tags: Sequence[str] = (),
    external_testcase: str | Path | None = None,
    comparator: str | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    access_policy: Any = None,
) -> AgentResponse:
    """Execute alignment boundary runs against an explicit engine adapter."""
    from inferref.alignment import AlignmentRunner, AlignmentStore
    from inferref.alignment.hashing import normalized_adapter_sha256

    operation = "alignment_run"
    try:
        s_dir = Path(session_dir).resolve()
        a_path = Path(adapter).resolve()
        if access_policy is not None:
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()
            a_path = Path(access_policy(a_path, "adapter")).resolve()

        ext_tc_path = None
        if external_testcase:
            if not fixture_kind:
                return AgentResponse.error(
                    operation,
                    "external_testcase requires an explicit fixture_kind",
                    code="invalid_request",
                )
            ext_tc_path = Path(external_testcase).resolve()
            if access_policy is not None:
                ext_tc_path = Path(access_policy(ext_tc_path, "external testcase")).resolve()
            if len(boundary_ids) != 1:
                return AgentResponse.error(
                    operation,
                    f"external_testcase requires exactly one boundary, got {len(boundary_ids)}",
                    code="invalid_request",
                )

        effective_fixture_kind = fixture_kind or "reference_derived"

        if not (s_dir / "alignment.json").is_file():
            return AgentResponse.error(operation, f"no alignment session found at {s_dir}")

        store = AlignmentStore.load(s_dir)

        # Transitive policy check on trace
        t_path = Path(store.snapshot.source["trace"]["path"]).resolve()
        if access_policy is not None:
            t_path = Path(access_policy(t_path, "trace input")).resolve()

        if not a_path.is_file():
            return AgentResponse.error(operation, f"adapter file not found: {a_path}")

        engine_adapter = EngineAdapter.load(a_path)
        actual_adapter_sha = normalized_adapter_sha256(engine_adapter.to_dict())
        pinned_adapter_sha = store.snapshot.target["adapter"]["normalized_sha256"]
        if actual_adapter_sha != pinned_adapter_sha:
            return AgentResponse.error(
                operation,
                f"adapter digest mismatch: expected {pinned_adapter_sha}, got {actual_adapter_sha}",
                code="adapter_digest_mismatch",
            )

        pkg = TracePackage.load(t_path)
        runner = AlignmentRunner(store, pkg, engine_adapter)

        merged_comparator_overrides = dict(comparator_overrides or {})
        if comparator:
            merged_comparator_overrides["comparator"] = comparator
        if atol is not None:
            merged_comparator_overrides["atol"] = atol
        if rtol is not None:
            merged_comparator_overrides["rtol"] = rtol
        if ignore_stride:
            merged_comparator_overrides["ignore_stride"] = True
        if strict_layout:
            merged_comparator_overrides["strict_layout"] = True

        results = runner.run(
            limit=limit,
            boundary_ids=boundary_ids,
            continue_on_fail=continue_on_fail,
            engine_revision=engine_revision,
            comparator_overrides=merged_comparator_overrides or None,
            fixture_kind=effective_fixture_kind,
            fixture_tags=fixture_tags,
            external_testcase=ext_tc_path,
        )
        has_failure = any(r.status in ("fail", "error") for r in results)
        has_error = any(r.status == "error" for r in results)
        if has_error:
            run_status = "error"
        elif has_failure:
            run_status = "fail"
        else:
            run_status = "pass"

        actions = (
            (
                {
                    "operation": "alignment_narrow",
                    "reason": "Diagnose earliest failing boundary.",
                },
            )
            if has_failure
            else ()
        )
        return AgentResponse(
            operation=operation,
            status=run_status,
            data={
                "results": [
                    {
                        "boundary": r.boundary,
                        "status": r.status,
                        "evidence_id": r.evidence.id if r.evidence else None,
                        "error_message": r.error_message,
                    }
                    for r in results
                ],
                "snapshot": store.snapshot.to_dict(),
            },
            next_actions=actions,
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


def align_run(
    session_dir: str | Path,
    adapter: str | Path,
    **kwargs: Any,
) -> AgentResponse:
    """Backward-compatible alias for alignment_run; requires explicit adapter."""
    return alignment_run(session_dir, adapter, **kwargs)


def alignment_status(
    session_dir: str | Path,
    *,
    access_policy: Any = None,
) -> AgentResponse:
    """Inspect alignment session progress and active failure."""
    from inferref.alignment import (
        AlignmentStore,
        build_fixture_matrix,
        classify_failure_domain,
        compute_narrowing,
        inventory_boundaries,
    )

    operation = "alignment_status"
    try:
        s_dir = Path(session_dir).resolve()
        if access_policy is not None:
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()

        if not (s_dir / "alignment.json").is_file():
            return AgentResponse.error(operation, f"no alignment session found at {s_dir}")

        store = AlignmentStore.load(s_dir)
        snap = store.snapshot

        if access_policy is not None:
            access_policy(snap.source["trace"]["path"], "trace input")
            access_policy(snap.target["adapter"]["path"], "adapter")

        matrix = build_fixture_matrix(
            list(store.evidence.values()),
            snap,
            known_boundaries=list(snap.boundaries.keys()),
            events=store.events,
        )

        evidence_by_b: dict[str, list[Any]] = {}
        for ev in matrix.current_evidence.values():
            if ev.type == "numerical_comparison":
                evidence_by_b.setdefault(ev.subject["boundary"], []).append(ev)

        t_path = Path(store.snapshot.source["trace"]["path"]).resolve()
        pkg = TracePackage.load(t_path)
        inventory = inventory_boundaries(pkg)

        narrowing = compute_narrowing(inventory, evidence_by_b)
        candidates: list[Any] = []
        if narrowing.smallest_confirmed_failure:
            notes = [ev for ev in store.evidence.values() if ev.type == "manual_note"]
            candidates, _ = classify_failure_domain(
                narrowing.smallest_confirmed_failure,
                list(matrix.current_evidence.values()),
                notes=notes,
                active_engine_identity=matrix.active_engine_identity,
                events=store.events,
            )

        verified = sum(1 for s in matrix.boundary_states.values() if s == "verified")
        failing = sum(1 for s in matrix.boundary_states.values() if s == "failing")
        untested = sum(1 for s in matrix.boundary_states.values() if s == "untested")
        stale = sum(1 for s in matrix.boundary_states.values() if s == "stale")

        return AgentResponse(
            operation=operation,
            status="ok",
            data={
                "session_id": snap.session_id,
                "status": snap.status,
                "counts": {
                    "verified": verified,
                    "failing": failing,
                    "untested": untested,
                    "stale": stale,
                    "total": len(snap.boundaries),
                },
                "active_failure": snap.narrowing.get("current_failure"),
                "smallest_confirmed_failure": narrowing.smallest_confirmed_failure,
                "classification": [c.to_dict() for c in candidates],
            },
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


align_status = alignment_status


def alignment_summary(
    session_dir: str | Path,
    *,
    access_policy: Any = None,
) -> AgentResponse:
    """Summarize alignment metrics, TFCM, and adoption ratio (compact Spec §11.3)."""
    from inferref.alignment import (
        AlignmentStore,
        build_fixture_matrix,
        classify_failure_domain,
    )

    operation = "alignment_summary"
    try:
        s_dir = Path(session_dir).resolve()
        if access_policy is not None:
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()

        if not (s_dir / "alignment.json").is_file():
            return AgentResponse.error(operation, f"no alignment session found at {s_dir}")

        store = AlignmentStore.load(s_dir)
        if access_policy is not None:
            access_policy(store.snapshot.source["trace"]["path"], "trace input")
            access_policy(store.snapshot.target["adapter"]["path"], "adapter")

        matrix = build_fixture_matrix(
            list(store.evidence.values()),
            store.snapshot,
            known_boundaries=list(store.snapshot.boundaries.keys()),
            events=store.events,
        )

        counts = {
            "verified": len(matrix.verified_boundaries),
            "failing": len(matrix.failing_boundaries),
            "untested": len(matrix.untested_boundaries),
            "stale": len(matrix.stale_boundaries),
        }

        current_failure = None
        candidates_out: list[dict[str, Any]] = []
        if matrix.failing_boundaries:
            first_fail = matrix.failing_boundaries[0]
            fail_name = store.snapshot.boundaries.get(first_fail, {}).get("name", first_fail)
            b_fixtures: dict[str, str] = {}
            for k, ev in matrix.current_evidence.items():
                if k[0] == first_fail:
                    b_fixtures[ev.fixture["kind"]] = ev.result

            current_failure = {
                "boundary": first_fail,
                "name": fail_name,
                "engine_inputs": "not_observed",
                "fixtures": b_fixtures,
            }

            notes = [ev for ev in store.evidence.values() if ev.type == "manual_note"]
            candidates, _ = classify_failure_domain(
                first_fail,
                list(matrix.current_evidence.values()),
                notes=notes,
                active_engine_identity=matrix.active_engine_identity,
                events=store.events,
            )
            candidates_out = [
                {
                    "id": c.id,
                    "domain": c.domain,
                    "confidence": c.confidence,
                    "supporting_evidence": list(c.supporting_evidence),
                }
                for c in candidates[:3]
            ]

        actions: list[dict[str, Any]] = []
        if current_failure:
            actions.append({
                "operation": "alignment_note",
                "reason": f"Record observation or failure hypothesis for {current_failure['boundary']}.",
                "boundary": current_failure["boundary"],
            })
            actions.append({
                "operation": "alignment_narrow",
                "reason": f"Inspect narrowing path for {current_failure['boundary']}.",
                "boundary": current_failure["boundary"],
            })
        elif matrix.untested_boundaries:
            actions.append({
                "operation": "alignment_run",
                "reason": f"Run next untested boundary {matrix.untested_boundaries[0]}.",
                "boundary": matrix.untested_boundaries[0],
            })

        summary_data: dict[str, Any] = {
            "status": store.snapshot.status,
            "counts": counts,
            "current_failure": current_failure,
            "classification_candidates": candidates_out,
            "next_actions": actions[:3],
            "artifacts": {"alignment": "alignment.json"},
            "metrics": {
                "tfcm_seconds": matrix.metrics.get("tfcm_seconds"),
                "adoption_ratio": matrix.metrics.get("adoption_ratio"),
                "validation_steps": matrix.metrics.get("inferref_validation_steps", 0),
            },
        }

        return AgentResponse(
            operation=operation,
            status="ok",
            data=summary_data,
            next_actions=tuple(actions[:3]),
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


align_summary = alignment_summary


def alignment_narrow(
    session_dir: str | Path,
    *,
    boundary_id: str | None = None,
    access_policy: Any = None,
) -> AgentResponse:
    """Compute smallest confirmed failing boundary and failure domain classification."""
    from inferref.alignment import (
        AlignmentStore,
        build_fixture_matrix,
        classify_failure_domain,
        compute_narrowing,
        inventory_boundaries,
    )

    operation = "alignment_narrow"
    try:
        s_dir = Path(session_dir).resolve()
        if access_policy is not None:
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()

        if not (s_dir / "alignment.json").is_file():
            return AgentResponse.error(operation, f"no alignment session found at {s_dir}")

        store = AlignmentStore.load(s_dir)
        if access_policy is not None:
            access_policy(store.snapshot.source["trace"]["path"], "trace input")
            access_policy(store.snapshot.target["adapter"]["path"], "adapter")

        matrix = build_fixture_matrix(
            list(store.evidence.values()),
            store.snapshot,
            known_boundaries=list(store.snapshot.boundaries.keys()),
            events=store.events,
        )

        evidence_by_b: dict[str, list[Any]] = {}
        for ev in matrix.current_evidence.values():
            if ev.type == "numerical_comparison":
                evidence_by_b.setdefault(ev.subject["boundary"], []).append(ev)

        t_path = Path(store.snapshot.source["trace"]["path"]).resolve()
        pkg = TracePackage.load(t_path)
        inventory = inventory_boundaries(pkg)

        narrowing = compute_narrowing(
            inventory,
            evidence_by_b,
            current_failure=boundary_id,
        )
        candidates: list[Any] = []
        if narrowing.smallest_confirmed_failure:
            notes = [ev for ev in store.evidence.values() if ev.type == "manual_note"]
            candidates, _ = classify_failure_domain(
                narrowing.smallest_confirmed_failure,
                list(matrix.current_evidence.values()),
                notes=notes,
                active_engine_identity=matrix.active_engine_identity,
                events=store.events,
            )

        return AgentResponse(
            operation=operation,
            status="ok",
            data={
                "narrowing": narrowing.to_dict(),
                "classification": [c.to_dict() for c in candidates],
            },
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


align_narrow = alignment_narrow


def alignment_note(
    session_dir: str | Path,
    boundary_id: str,
    category: str,
    text: str,
    *,
    duration_minutes: float | None = None,
    milestone: str | None = None,
    target: str | None = None,
    author: str = "agent",
    access_policy: Any = None,
) -> AgentResponse:
    """Record human/agent manual note evidence."""
    from inferref.alignment import (
        AlignmentStore,
        build_manual_note_evidence,
        inventory_boundaries,
    )

    operation = "alignment_note"
    try:
        s_dir = Path(session_dir).resolve()
        if access_policy is not None:
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()

        if not (s_dir / "alignment.json").is_file():
            return AgentResponse.error(operation, f"no alignment session found at {s_dir}")

        store = AlignmentStore.load(s_dir)
        if access_policy is not None:
            access_policy(store.snapshot.source["trace"]["path"], "trace input")
            access_policy(store.snapshot.target["adapter"]["path"], "adapter")

        t_path = Path(store.snapshot.source["trace"]["path"]).resolve()
        pkg = TracePackage.load(t_path)
        inventory = inventory_boundaries(pkg)

        boundary = inventory.resolve(boundary_id)
        if boundary is None:
            return AgentResponse.error(operation, f"boundary not in inventory: {boundary_id}")

        evidence = build_manual_note_evidence(
            subject={"boundary": boundary.id, "name": boundary.name},
            category=category,
            reason=text,
            duration_minutes=duration_minutes,
            milestone=milestone,
            target=target,
            extensions={"author": author},
        )
        with store.lock():
            event_data: dict[str, Any] = {
                "boundary": boundary.id,
                "category": category,
                "text": text,
            }
            if duration_minutes is not None:
                event_data["duration_minutes"] = duration_minutes
            if milestone is not None:
                event_data["milestone"] = milestone
            if target is not None:
                event_data["target"] = target
            store.commit_evidence(
                evidence,
                event_name="note_recorded",
                event_data=event_data,
            )
        return AgentResponse(
            operation=operation,
            status="pass",
            data=evidence.to_dict(),
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


align_note = alignment_note


def alignment_report(
    session_dir: str | Path,
    *,
    output: str | Path | None = None,
    json_sidecar: str | Path | None = None,
    access_policy: Any = None,
) -> AgentResponse:
    """Generate markdown and JSON sidecar alignment reports."""
    from inferref.alignment import AlignmentStore
    from inferref.alignment.report import generate_alignment_report

    operation = "alignment_report"
    try:
        s_dir = Path(session_dir).resolve()
        if access_policy is not None:
            s_dir = Path(access_policy(s_dir, "alignment session")).resolve()

        if not (s_dir / "alignment.json").is_file():
            return AgentResponse.error(operation, f"no alignment session found at {s_dir}")

        store = AlignmentStore.load(s_dir)
        if access_policy is not None:
            access_policy(store.snapshot.source["trace"]["path"], "trace input")
            access_policy(store.snapshot.target["adapter"]["path"], "adapter")

        md_p = Path(output).resolve() if output else s_dir / "reports" / "alignment-report.md"
        js_p = Path(json_sidecar).resolve() if json_sidecar else s_dir / "reports" / "alignment-report.json"
        if access_policy is not None:
            md_p = Path(access_policy(md_p, "report output")).resolve()
            js_p = Path(access_policy(js_p, "sidecar output")).resolve()

        _md_text, sidecar = generate_alignment_report(
            store,
            markdown_path=md_p,
            json_sidecar_path=js_p,
        )
        return AgentResponse(
            operation=operation,
            status="ok",
            data=sidecar,
        )
    except Exception as exc:
        code = "path_not_allowed" if _is_path_error(exc) else "invalid_request"
        return AgentResponse.error(operation, str(exc), code=code)


align_report = alignment_report
