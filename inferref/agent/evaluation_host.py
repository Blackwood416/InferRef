"""Host orchestration and external-Agent drivers for evaluation v0.2."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from inferref.agent.adapter import (
    _assign_windows_kill_job,
    _close_windows_job,
    _terminate_process_tree,
)
from inferref.agent.evaluation import (
    CANDIDATE_URI,
    RUNS_URI,
    VISIBLE_URI,
    AuditLog,
    EvaluationBenchmark,
    EvaluationCase,
    _canonical_json_sha256,
    _is_sha256,
    _read_regular_file,
    engine_patch,
    execute_case,
    load_audit,
    prepare_workspace,
    workspace_hashes,
)
from inferref.ir.version import INFERREF_VERSION


@dataclass(frozen=True)
class AgentProcessResult:
    command: tuple[str, ...]
    exit_code: int | None
    timed_out: bool
    duration_ms: float
    stdout_path: Path
    stderr_path: Path
    stdout: str
    stderr: str
    usage: dict[str, Any]
    cli_version: str | None = None
    transcript_error: str | None = None
    runner_evidence: dict[str, Any] = field(default_factory=dict)
    model_evidence: dict[str, Any] = field(default_factory=dict)


Driver = Callable[[str, str, EvaluationBenchmark, Path, Path, Path], AgentProcessResult]


@dataclass(frozen=True)
class ResolvedAgentCommand:
    """Frozen Agent launch prefix split into hashable files and full argv."""

    components: tuple[str, ...]
    prefix: tuple[str, ...]


def _run_formal_worker(
    benchmark_path: str | Path,
    *,
    agents: Sequence[str],
    report_dir: str | Path,
    claude_settings: str | Path | None,
    claude_model: str | None,
    public_attestation: str | Path,
) -> dict[str, Any]:
    """Execute formal evaluation in a fresh isolated Python interpreter."""

    benchmark = EvaluationBenchmark.load(benchmark_path)
    request = {
        "benchmark": str(benchmark.source),
        "agents": list(agents),
        "report_dir": str(Path(report_dir).resolve()),
        "public_attestation": str(Path(public_attestation).resolve()),
        "claude_settings": (
            str(Path(claude_settings).resolve())
            if claude_settings is not None
            else None
        ),
        "claude_model": claude_model,
    }
    timeout = benchmark.max_wall_seconds * max(1, len(set(agents))) + 180
    with tempfile.TemporaryDirectory(prefix="inferref-formal-worker-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_bytes = json.dumps(request, ensure_ascii=False).encode("utf-8") + b"\n"
        request_sha256 = hashlib.sha256(request_bytes).hexdigest()
        request_path.write_bytes(request_bytes)
        try:
            completed = subprocess.run(
                [
                    str(Path(sys.executable).resolve()),
                    "-I",
                    "-m",
                    "inferref.agent.evaluation_worker",
                    "--request",
                    str(request_path),
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                check=False,
                shell=False,
                timeout=timeout,
                text=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("formal evaluation worker timed out") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()[:2048]
        raise RuntimeError(
            f"formal evaluation worker failed with code {completed.returncode}: {detail}"
        )
    report_path = Path(request["report_dir"]) / "report.json"
    try:
        report = json.loads(_read_regular_file(report_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "formal evaluation worker did not produce a valid report"
        ) from exc
    if not isinstance(report, dict):
        raise TypeError("formal evaluation worker report root is not an object")
    reported_request_sha = (
        (report.get("worker_evidence") or {})
        .get("launch_policy", {})
        .get("request_sha256")
    )
    if reported_request_sha != request_sha256:
        raise RuntimeError(
            "formal worker evidence does not match the parent request bytes"
        )
    return report


def evaluate_benchmark(
    benchmark_path: str | Path,
    *,
    agents: Sequence[str],
    report_dir: str | Path,
    driver: Driver | None = None,
    claude_settings: str | Path | None = None,
    claude_model: str | None = None,
    public_attestation: str | Path | None = None,
) -> dict[str, Any]:
    """Run a development evaluation or delegate formal publication to a worker."""

    if public_attestation is not None:
        if driver is not None:
            raise ValueError(
                "formal public attestation requires the isolated built-in Agent worker"
            )
        return _run_formal_worker(
            benchmark_path,
            agents=agents,
            report_dir=report_dir,
            claude_settings=claude_settings,
            claude_model=claude_model,
            public_attestation=public_attestation,
        )
    return _evaluate_benchmark_core(
        benchmark_path,
        agents=agents,
        report_dir=report_dir,
        driver=driver,
        claude_settings=claude_settings,
        claude_model=claude_model,
    )


def _evaluate_benchmark_core(
    benchmark_path: str | Path,
    *,
    agents: Sequence[str],
    report_dir: str | Path,
    driver: Driver | None = None,
    claude_settings: str | Path | None = None,
    claude_model: str | None = None,
    public_attestation: str | Path | None = None,
    formal_worker_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    benchmark = EvaluationBenchmark.load(benchmark_path)
    selected = tuple(dict.fromkeys(agents))
    if not selected or any(name not in benchmark.drivers for name in selected):
        raise ValueError(
            "agents must select one or more configured drivers: "
            + ", ".join(sorted(benchmark.drivers))
        )
    public_attestation_path = (
        Path(public_attestation).resolve() if public_attestation is not None else None
    )
    if formal_worker_evidence is not None and (
        public_attestation_path is None or driver is not None
    ):
        raise ValueError("formal worker requires built-in drivers and a public output")
    runner_mode = (
        "isolated_builtin_cli_worker"
        if formal_worker_evidence is not None
        else ("builtin_driver_path" if driver is None else "custom_driver")
    )
    repository_before = _repository_evidence(benchmark.directory)
    evaluator_source_before = _evaluator_source_manifest()
    runtime_before = _runtime_evidence(benchmark.directory)
    capture_claude_settings = driver is None and "claude" in selected
    settings_before = (
        _capture_external_file(claude_settings) if capture_claude_settings else None
    )
    if public_attestation_path is not None and (
        repository_before["commit"] is None or repository_before["dirty"]
    ):
        raise RuntimeError(
            "formal public attestation requires a clean Git worktree at a commit"
        )
    report_root = Path(report_dir).resolve()
    if report_root.exists() and (
        not report_root.is_dir() or any(report_root.iterdir())
    ):
        raise FileExistsError(
            f"evaluation report directory is not empty: {report_root}"
        )
    report_root.mkdir(parents=True, exist_ok=True)
    if public_attestation_path is not None and public_attestation_path.exists():
        raise FileExistsError(
            f"public attestation already exists: {public_attestation_path}"
        )
    if public_attestation_path is not None:
        if public_attestation_path.is_relative_to(report_root):
            raise ValueError(
                "public attestation must be outside the private report directory"
            )
        public_attestation_path.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for name in selected:
        model = (
            claude_model
            if name == "claude" and claude_model is not None
            else benchmark.drivers[name].model
        )
        agent_root = report_root / name
        agent_root.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix=f"inferref-eval-{name}-") as temporary:
            workspace = prepare_workspace(benchmark, Path(temporary) / "workspace")
            session_root = agent_root / "session"
            audit_path = agent_root / "audit.jsonl"
            initial_hashes = workspace_hashes(workspace)
            host_paths = _protected_host_paths(benchmark)
            initial_host_hashes = _hash_named_files(host_paths)
            prompt = _candidate_prompt(benchmark)
            if driver is None:
                process = run_agent_cli(
                    name,
                    model,
                    benchmark,
                    workspace,
                    session_root,
                    audit_path,
                    claude_settings=claude_settings,
                    claude_model_override=claude_model,
                )
            else:
                process = driver(
                    name,
                    model,
                    benchmark,
                    workspace,
                    session_root,
                    audit_path,
                )
            final_hashes = workspace_hashes(workspace)
            final_host_hashes = _hash_named_files(host_paths)
            audit = load_audit(audit_path)
            result = assess_candidate(
                benchmark,
                agent=name,
                model=model,
                workspace=workspace,
                agent_root=agent_root,
                process=process,
                initial_hashes=initial_hashes,
                final_hashes=final_hashes,
                audit=audit,
                initial_host_hashes=initial_host_hashes,
                final_host_hashes=final_host_hashes,
            )
            shutil.copytree(workspace, agent_root / "final-workspace")
            result["prompt"] = prompt
        results.append(result)
        (agent_root / "result.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    repository_after = _repository_evidence(benchmark.directory)
    evaluator_source_after = _evaluator_source_manifest()
    runtime_after = _runtime_evidence(benchmark.directory)
    settings_after = (
        _capture_external_file(claude_settings) if capture_claude_settings else None
    )
    settings_evidence = (
        _external_file_pair(settings_before, settings_after)
        if capture_claude_settings
        else None
    )
    settings_evidence_valid = (
        settings_evidence is None or _external_file_evidence_valid(settings_evidence)
    )
    repository_unchanged = repository_after == repository_before
    evaluator_unchanged = evaluator_source_after == evaluator_source_before
    runtime_unchanged = runtime_after == runtime_before
    benchmark_after_sha256 = _sha256_or_none(benchmark.source)
    benchmark_unchanged = benchmark_after_sha256 == benchmark.source_sha256
    runner_integrity = (
        None
        if driver is not None
        else all(
            _runner_evidence_valid(item["process"].get("runner")) for item in results
        )
    )
    model_evidence_valid = (
        None
        if driver is not None
        else all(
            _model_evidence_valid(item["process"].get("model_evidence"), item["agent"])
            for item in results
        )
    )
    model_request_satisfied = _aggregate_optional_bools(
        _model_request_satisfied(item["process"].get("model_evidence"), item["agent"])
        for item in results
    )
    host_unchanged = (
        repository_unchanged
        and evaluator_unchanged
        and benchmark_unchanged
        and runtime_unchanged
        and settings_evidence_valid
        and runner_integrity is not False
        and model_evidence_valid is not False
        and model_request_satisfied is not False
    )
    passed_count = sum(item["status"] == "pass" for item in results)
    passed = passed_count >= benchmark.success.required_agent_passes and host_unchanged
    external_files = (
        {"claude_settings": settings_evidence}
        if settings_evidence is not None
        else None
    )
    report = {
        "format": "inferref-agent-evaluation-report",
        "format_version": "0.2",
        "benchmark": benchmark.id,
        "runner_mode": runner_mode,
        "status": "pass" if passed else "fail",
        "acceptance": {
            "configured_agents": list(benchmark.drivers),
            "selected_agents": list(selected),
            "policy": benchmark.success.to_dict(),
            "required_passes": benchmark.success.required_agent_passes,
            "passed": passed_count,
            "agent_model_request_satisfied": model_request_satisfied,
            "host_integrity_passed": host_unchanged,
        },
        "host_integrity": {
            "repository_unchanged": repository_unchanged,
            "evaluator_source_unchanged": evaluator_unchanged,
            "benchmark_source_unchanged": benchmark_unchanged,
            "runtime_unchanged": runtime_unchanged,
            "claude_settings_unchanged": settings_evidence_valid,
            "agent_command_chains_unchanged": runner_integrity,
            "agent_model_evidence_valid": model_evidence_valid,
            "agent_model_request_satisfied": model_request_satisfied,
            "external_files": external_files,
        },
        "agents": results,
        "worker_evidence": formal_worker_evidence,
    }
    report_path = report_root / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    formal_repository_valid = (
        repository_unchanged
        and repository_after["commit"] is not None
        and not repository_before["dirty"]
        and not repository_after["dirty"]
    )
    if public_attestation_path is not None and not (
        _formal_worker_evidence_valid(formal_worker_evidence)
        and formal_repository_valid
        and evaluator_unchanged
        and benchmark_unchanged
        and runtime_unchanged
        and runner_integrity is True
        and model_evidence_valid is True
        and model_request_satisfied is not False
        and settings_evidence_valid
    ):
        raise RuntimeError(
            "formal public attestation refused because repository, evaluator, "
            "benchmark, runtime, Agent command/model-identity, or external-file "
            "evidence changed during the run"
        )
    attestation = build_public_attestation(
        benchmark,
        report,
        report_root=report_root,
        report_path=report_path,
        runner_mode=runner_mode,
        repository_before=repository_before,
        repository_after=repository_after,
        repository_unchanged=repository_unchanged,
        evaluator_source=evaluator_source_before,
        evaluator_unchanged=evaluator_unchanged,
        benchmark_unchanged=benchmark_unchanged,
        runtime_before=runtime_before,
        runtime_after=runtime_after,
        runtime_unchanged=runtime_unchanged,
        external_files=external_files,
        formal_worker_evidence=formal_worker_evidence,
    )
    attestation_path = report_root / "attestation.json"
    _write_new_json(attestation_path, attestation)
    if public_attestation_path is not None:
        _write_new_json(public_attestation_path, attestation)
    return report


def build_public_attestation(
    benchmark: EvaluationBenchmark,
    report: dict[str, Any],
    *,
    report_root: Path,
    report_path: Path,
    runner_mode: str = "custom_driver",
    repository_before: dict[str, Any] | None = None,
    repository_after: dict[str, Any] | None = None,
    repository_unchanged: bool | None = None,
    evaluator_source: dict[str, Any] | None = None,
    evaluator_unchanged: bool = True,
    benchmark_unchanged: bool = True,
    runtime_before: dict[str, Any] | None = None,
    runtime_after: dict[str, Any] | None = None,
    runtime_unchanged: bool | None = None,
    external_files: dict[str, Any] | None = None,
    formal_worker_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    agents = []
    for result in report["agents"]:
        agent = result["agent"]
        agent_root = report_root / agent
        integrity = result["integrity"]
        agents.append(
            {
                "agent": agent,
                "model": result["process"]["model_evidence"],
                "model_request_satisfied": _model_request_satisfied(
                    result["process"].get("model_evidence"), agent
                ),
                "runner": result["process"]["runner"],
                "cli_version": result["process"]["cli_version"],
                "status": result["status"],
                "classification": result["classification"],
                "duration_ms": result["process"]["duration_ms"],
                "engine_runs": result["visible"]["runs"],
                "final_engine_sha256": result["final_engine_sha256"],
                "historical_visible_passed": result["visible"]["historical_passed"],
                "final_visible": result["visible"]["final"],
                "holdouts": result["holdouts"],
                "tool_audit": [
                    _public_audit_record(item) for item in result["tool_audit"]
                ],
                "audit_integrity": result["audit_integrity"],
                "engine_patch": result["patch"],
                "protected_file_hashes": integrity["protected_file_hashes"],
                "host_protected_file_hashes": integrity["host_protected_file_hashes"],
                "raw_transcript_sha256": _sha256_or_none(
                    agent_root / Path(result["process"]["stdout_path"]).name
                ),
                "raw_stderr_sha256": _sha256_or_none(
                    agent_root / Path(result["process"]["stderr_path"]).name
                ),
                "usage": result["process"]["usage"],
            }
        )
    repository_before = repository_before or _repository_evidence(benchmark.directory)
    repository_after = repository_after or _repository_evidence(benchmark.directory)
    if repository_unchanged is None:
        repository_unchanged = repository_before == repository_after
    evaluator_source = evaluator_source or _evaluator_source_manifest()
    runtime_before = runtime_before or _runtime_evidence(benchmark.directory)
    runtime_after = runtime_after or _runtime_evidence(benchmark.directory)
    if runtime_unchanged is None:
        runtime_unchanged = runtime_before == runtime_after
    formal = (
        _formal_worker_evidence_valid(formal_worker_evidence)
        and runner_mode == "isolated_builtin_cli_worker"
        and repository_unchanged
        and repository_before["commit"] is not None
        and repository_after["commit"] is not None
        and not repository_before["dirty"]
        and not repository_after["dirty"]
        and evaluator_unchanged
        and benchmark_unchanged
        and runtime_unchanged
        and all(_runner_evidence_valid(agent["runner"]) for agent in agents)
        and all(
            _model_evidence_valid(agent["model"], agent["agent"]) for agent in agents
        )
        and all(agent["model_request_satisfied"] is not False for agent in agents)
        and (
            external_files is None
            or all(
                _external_file_evidence_valid(item) for item in external_files.values()
            )
        )
    )
    return {
        "format": "inferref-agent-evaluation-attestation",
        "format_version": "0.5",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "attestation_level": "formal" if formal else "development",
        "runner_mode": runner_mode,
        "worker": formal_worker_evidence,
        "repository": {
            "before": repository_before,
            "after": repository_after,
            "unchanged": repository_unchanged,
        },
        "benchmark": {
            "id": benchmark.id,
            "format": "inferref-agent-evaluation",
            "format_version": "0.2",
            "loaded_source_sha256": benchmark.source_sha256,
            "source_unchanged": benchmark_unchanged,
            "success_policy": benchmark.success.to_dict(),
        },
        "evaluator": {
            "source_tree_sha256": evaluator_source["source_tree_sha256"],
            "source_tree_unchanged": evaluator_unchanged,
            "files": evaluator_source["files"],
        },
        "runtime": {
            "before": runtime_before,
            "after": runtime_after,
            "unchanged": runtime_unchanged,
        },
        "external_files": external_files,
        "report_json_sha256": _sha256_or_none(report_path),
        "status": report["status"],
        "acceptance": report["acceptance"],
        "agents": agents,
    }


def assess_candidate(
    benchmark: EvaluationBenchmark,
    *,
    agent: str,
    model: str,
    workspace: Path,
    agent_root: Path,
    process: AgentProcessResult,
    initial_hashes: dict[str, Any],
    final_hashes: dict[str, Any],
    audit: AuditLog,
    initial_host_hashes: dict[str, str | None] | None = None,
    final_host_hashes: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    audit_records = list(audit.records)
    editable = set(benchmark.editable_paths)
    changed = sorted(
        name
        for name in set(initial_hashes) | set(final_hashes)
        if initial_hashes.get(name) != final_hashes.get(name)
    )
    protected_changes = [name for name in changed if name not in editable]
    observed_tools = [
        item.get("tool") for item in audit_records if isinstance(item.get("tool"), str)
    ]
    missing_tools = sorted(set(benchmark.required_tools) - set(observed_tools))
    budget_attempts = [
        item
        for item in audit_records
        if "budget_exhausted" in (item.get("diagnostic_codes") or [])
    ]
    integrity_attempts = [
        item
        for item in audit_records
        if "integrity_failure" in (item.get("diagnostic_codes") or [])
    ]
    sequence_attempts = [
        item
        for item in audit_records
        if "required_tool_sequence" in (item.get("diagnostic_codes") or [])
    ]
    initial_host_hashes = initial_host_hashes or {}
    final_host_hashes = final_host_hashes or {}
    host_changes = sorted(
        name
        for name in set(initial_host_hashes) | set(final_host_hashes)
        if initial_host_hashes.get(name) != final_host_hashes.get(name)
    )
    visible_runs = [
        item for item in audit_records if item.get("tool") == "inferref_run_engine"
    ]
    historical_visible_pass = any(item.get("status") == "pass" for item in visible_runs)

    classification = "pass"
    message = "candidate satisfied the benchmark success policy"
    if process.timed_out:
        classification = "infrastructure_failure"
        message = "Agent process exceeded the wall-clock limit"
    elif process.runner_evidence and not _runner_evidence_valid(
        process.runner_evidence
    ):
        classification = "integrity_failure"
        message = "Agent executable command chain changed during execution"
    elif process.transcript_error is not None:
        classification = "infrastructure_failure"
        message = process.transcript_error
    elif process.exit_code not in (0, None):
        classification = "infrastructure_failure"
        message = _agent_api_failure(process.stdout_path) or (
            f"Agent CLI exited with code {process.exit_code}"
        )
    elif not audit.valid:
        classification = "infrastructure_failure"
        message = f"evaluation MCP audit is invalid: {audit.error}"
    elif (
        (
            benchmark.success.protected_paths_unchanged
            and (protected_changes or host_changes)
        )
        or integrity_attempts
        or sequence_attempts
        or budget_attempts
        or missing_tools
    ):
        classification = "integrity_failure"
        message = "candidate violated editable-path, tool-use, or run-budget policy"
    elif (
        process.model_evidence
        and _model_request_satisfied(process.model_evidence, agent) is False
    ):
        classification = "identity_policy_failure"
        message = "Agent reported a model that does not match the requested model"
    elif not historical_visible_pass:
        classification = "agent_failure"
        message = "candidate did not pass the visible case within the run budget"

    candidate_engine = workspace / benchmark.editable_paths[0]
    candidate_entry = final_hashes.get(benchmark.editable_paths[0])
    if classification == "pass" and (
        candidate_entry is None or candidate_entry.kind != "file"
    ):
        classification = "integrity_failure"
        message = "editable engine must remain a regular no-follow file"
    final_engine_sha256: str | None = None
    final_visible: dict[str, Any] | None = None
    holdouts: list[dict[str, Any]] = []
    if classification == "pass":
        try:
            final_engine = _read_regular_file(candidate_engine)
        except OSError as exc:
            classification = "agent_failure"
            message = f"final candidate engine is unavailable: {exc}"
        else:
            final_engine_sha256 = hashlib.sha256(final_engine).hexdigest()
            final_visible_outcome = _execute_captured_case(
                benchmark.visible_case,
                engine_bytes=final_engine,
                validation_root=agent_root / "final-validation",
                slot="visible",
            )
            final_visible = _public_case_result(final_visible_outcome)
            for index, case in enumerate(benchmark.holdout_cases):
                outcome = _execute_captured_case(
                    case,
                    engine_bytes=final_engine,
                    validation_root=agent_root / "final-validation",
                    slot=f"holdout-{index}",
                )
                holdouts.append(_public_case_result(outcome))
            final_results = [final_visible, *holdouts]
            if any(item["status"] == "integrity_failure" for item in final_results):
                classification = "integrity_failure"
                message = "final candidate modified an input-only validation testcase"
            elif final_visible["status"] != benchmark.success.visible_status:
                classification = "agent_failure"
                message = "final candidate failed silent visible revalidation"
            elif benchmark.success.all_holdouts_pass and any(
                item["status"] != "pass" for item in holdouts
            ):
                classification = "overfit_failure"
                message = "visible case passed but one or more hidden holdouts failed"

    template_engine = benchmark.directory / benchmark.editable_paths[0]
    try:
        patch = engine_patch(template_engine, candidate_engine)
    except (OSError, UnicodeError):
        patch = ""
    return {
        "agent": agent,
        "model": model,
        "status": "pass" if classification == "pass" else "fail",
        "classification": classification,
        "message": message,
        "process": {
            "cli_version": process.cli_version,
            "model_evidence": process.model_evidence
            or _model_evidence(agent, model, None, "unavailable"),
            "runner": process.runner_evidence,
            "transcript_error": process.transcript_error,
            "exit_code": process.exit_code,
            "timed_out": process.timed_out,
            "duration_ms": process.duration_ms,
            "stdout": process.stdout,
            "stderr": process.stderr,
            "stdout_path": process.stdout_path.name,
            "stderr_path": process.stderr_path.name,
            "usage": process.usage,
        },
        "integrity": {
            "changed_paths": changed,
            "protected_changes": protected_changes,
            "protected_file_hashes": {
                name: {
                    "before": _workspace_entry_dict(initial_hashes.get(name)),
                    "after": _workspace_entry_dict(final_hashes.get(name)),
                }
                for name in sorted(set(initial_hashes) | set(final_hashes))
                if name not in editable
            },
            "host_protected_changes": host_changes,
            "host_protected_file_hashes": {
                name: {
                    "before": initial_host_hashes.get(name),
                    "after": final_host_hashes.get(name),
                }
                for name in sorted(set(initial_host_hashes) | set(final_host_hashes))
            },
            "candidate_input_modification_attempts": len(integrity_attempts),
            "invalid_tool_sequence_attempts": len(sequence_attempts),
            "missing_required_tools": missing_tools,
            "budget_exhausted_attempts": len(budget_attempts),
        },
        "tool_audit": audit_records,
        "audit_integrity": audit.to_integrity_dict(),
        "final_engine_sha256": final_engine_sha256,
        "visible": {
            "runs": len(visible_runs),
            "historical_passed": historical_visible_pass,
            "final": final_visible,
            "passed": (
                final_visible is not None
                and final_visible["status"] == benchmark.success.visible_status
            ),
        },
        "holdouts": holdouts,
        "patch": patch,
    }


def run_agent_cli(
    agent: str,
    model: str,
    benchmark: EvaluationBenchmark,
    workspace: Path,
    session_root: Path,
    audit_path: Path,
    *,
    claude_settings: str | Path | None = None,
    claude_model_override: str | None = None,
) -> AgentProcessResult:
    session_root.mkdir(parents=True, exist_ok=True)
    resolved = _normalize_agent_command(_agent_executable(agent))
    resolved_executable = list(resolved.prefix)
    components_before = _command_component_evidence(resolved.components)
    cli_version = _agent_version(resolved_executable)
    components_after_version = _command_component_evidence(resolved.components)
    command = _agent_command(
        agent,
        model,
        benchmark,
        workspace,
        session_root,
        audit_path,
        executable=resolved_executable,
        claude_settings=claude_settings,
        claude_model_override=claude_model_override,
    )
    settings_before = (
        _capture_external_file(claude_settings) if agent == "claude" else None
    )
    result = _run_process(
        command,
        cwd=workspace,
        timeout_seconds=benchmark.max_wall_seconds,
        stdout_path=session_root.parent / "agent.stdout.jsonl",
        stderr_path=session_root.parent / "agent.stderr.log",
    )
    components_after = _command_component_evidence(resolved.components)
    components_unchanged = (
        components_before == components_after_version == components_after
        and all(item.get("sha256") for item in components_after)
    )
    settings_after = (
        _capture_external_file(claude_settings) if agent == "claude" else None
    )
    external_files = (
        {"claude_settings": _external_file_pair(settings_before, settings_after)}
        if agent == "claude"
        else None
    )
    argv_policy = _agent_argv_policy(agent, command)
    return replace(
        result,
        cli_version=cli_version,
        transcript_error=_validate_event_stream(result.stdout_path),
        runner_evidence={
            "kind": agent,
            "command_components": {
                "before": components_before,
                "after_version": components_after_version,
                "after": components_after,
            },
            "components_unchanged": components_unchanged,
            "argv_instance_sha256": hashlib.sha256(
                json.dumps(command, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
            "argv_policy": argv_policy,
            "argv_policy_sha256": _canonical_json_sha256(argv_policy),
            "version_output": cli_version,
            "external_files": external_files,
        },
        model_evidence=_agent_model_evidence(agent, model, result.stdout_path),
    )


def parse_agent_usage(path: Path) -> dict[str, Any]:
    aggregate: dict[str, float] = {}
    if not path.is_file():
        return {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        for key, value in _walk_scalars(item):
            lowered = key.lower()
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and ("token" in lowered or "cost" in lowered)
            ):
                aggregate[lowered] = max(float(value), aggregate.get(lowered, 0.0))
    return {
        key: int(value) if value.is_integer() else value
        for key, value in aggregate.items()
    }


def _agent_command(
    agent: str,
    model: str,
    benchmark: EvaluationBenchmark,
    workspace: Path,
    session_root: Path,
    audit_path: Path,
    *,
    executable: Sequence[str] | None = None,
    claude_settings: str | Path | None = None,
    claude_model_override: str | None = None,
) -> list[str]:
    resolved = _normalize_agent_command(executable or _agent_executable(agent))
    executable = tuple(resolved.prefix)
    server_args = [
        "-m",
        "inferref.agent.evaluation_mcp",
        "--benchmark",
        str(benchmark.source),
        "--workspace",
        str(workspace),
        "--session-root",
        str(session_root),
        "--audit",
        str(audit_path),
    ]
    prompt = _candidate_prompt(benchmark)
    if agent == "codex":
        return [
            *executable,
            "exec",
            "--ignore-user-config",
            "--ephemeral",
            "--json",
            "--skip-git-repo-check",
            "--cd",
            str(workspace),
            "--sandbox",
            "workspace-write",
            "--model",
            model,
            "-c",
            f"mcp_servers.inferref.command={json.dumps(str(Path(sys.executable).absolute()))}",
            "-c",
            f"mcp_servers.inferref.args={json.dumps(server_args)}",
            "-c",
            'mcp_servers.inferref.default_tools_approval_mode="approve"',
            "-c",
            'approval_policy="never"',
            "-c",
            'windows.sandbox="elevated"',
            "-c",
            'web_search="disabled"',
            prompt,
        ]
    if agent == "claude":
        mcp_config = session_root / "claude-mcp.json"
        mcp_config.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "inferref": {
                            "command": str(Path(sys.executable).absolute()),
                            "args": server_args,
                        }
                    }
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        tools = (
            "Read,Edit,mcp__inferref__inferref_capabilities,"
            "mcp__inferref__inferref_context,mcp__inferref__inferref_run_engine"
        )
        command = [
            *executable,
            prompt,
            "--print",
            "--output-format",
            "stream-json",
            "--verbose",
            "--effort",
            "high",
            "--no-session-persistence",
            "--strict-mcp-config",
            "--mcp-config",
            str(mcp_config),
            "--disable-slash-commands",
            "--no-chrome",
            "--permission-mode",
            "acceptEdits",
            "--tools",
            tools,
            "--allowedTools",
            tools,
        ]
        if claude_model_override is not None:
            command.extend(("--model", claude_model_override))
        elif claude_settings is None:
            command.extend(("--model", model))
        if claude_settings is not None:
            settings_path = Path(claude_settings).resolve()
            if not settings_path.is_file():
                raise FileNotFoundError(
                    f"Claude settings file does not exist: {settings_path}"
                )
            command.extend(("--settings", str(settings_path)))
        return command
    raise ValueError(f"unsupported Agent driver {agent!r}")


def _normalize_agent_command(
    value: ResolvedAgentCommand | Sequence[str],
) -> ResolvedAgentCommand:
    """Accept either a resolved command or a plain argv prefix (test shims)."""

    if isinstance(value, ResolvedAgentCommand):
        return value
    prefix = tuple(value)
    return ResolvedAgentCommand(components=prefix, prefix=prefix)


def _agent_executable(agent: str) -> ResolvedAgentCommand:
    """Resolve Windows npm shims or POSIX shebangs without a command shell."""

    if os.name != "nt":
        return _resolve_posix_agent_command(agent)
    return _resolve_windows_agent_command(agent)


def _resolve_posix_agent_command(agent: str) -> ResolvedAgentCommand:
    executable = shutil.which(agent)
    if executable is None:
        raise FileNotFoundError(f"Agent CLI is not installed: {agent}")
    entry = Path(executable).resolve(strict=True)
    try:
        first_line = (
            _read_regular_file(entry).splitlines()[0].decode("utf-8", errors="strict")
        )
    except (OSError, UnicodeError, IndexError):
        return ResolvedAgentCommand((str(entry),), (str(entry),))
    if not first_line.startswith("#!"):
        return ResolvedAgentCommand((str(entry),), (str(entry),))
    tokens = shlex.split(first_line[2:].strip())
    if not tokens:
        return ResolvedAgentCommand((str(entry),), (str(entry),))
    interpreter = tokens[0]
    arguments: list[str] = []
    if Path(interpreter).name == "env":
        rest = tokens[1:]
        has_s = "-S" in rest
        index = 0
        value_options = {"-u", "--unset", "-C", "--chdir"}
        while index < len(rest):
            token = rest[index]
            if not token.startswith("-"):
                break
            if token in value_options and index + 1 < len(rest):
                index += 2
            else:
                index += 1
        if index >= len(rest):
            raise FileNotFoundError(
                f"cannot resolve interpreter for Agent CLI: {agent}"
            )
        interpreter = rest[index]
        arguments = rest[index + 1 :]
        if has_s and shlex.split(interpreter) != [interpreter]:
            parts = shlex.split(interpreter)
            interpreter = parts[0]
            arguments = [*parts[1:], *arguments]
    else:
        arguments = tokens[1:]
    if "/" not in interpreter:
        resolved = shutil.which(interpreter)
        if resolved is None:
            raise FileNotFoundError(
                f"cannot resolve interpreter for Agent CLI: {agent}"
            )
        interpreter = resolved
    interpreter_path = str(Path(interpreter).resolve(strict=True))
    components = (interpreter_path, str(entry))
    prefix = (interpreter_path, *arguments, str(entry))
    return ResolvedAgentCommand(components, prefix)


def _resolve_windows_agent_command(agent: str) -> ResolvedAgentCommand:
    shim = shutil.which(agent)
    if shim is None:
        raise FileNotFoundError(f"Agent CLI is not installed: {agent}")
    npm_root = Path(shim).parent
    if agent == "claude":
        native = (
            npm_root
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "bin"
            / "claude.exe"
        )
        if native.is_file():
            path = str(native.resolve(strict=True))
            return ResolvedAgentCommand((path,), (path,))
    if agent == "codex":
        script = npm_root / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        node = shutil.which("node.exe") or shutil.which("node")
        if script.is_file() and node is not None:
            node_path = str(Path(node).resolve(strict=True))
            script_path = str(script.resolve(strict=True))
            return ResolvedAgentCommand(
                (node_path, script_path), (node_path, script_path)
            )
    raise FileNotFoundError(f"cannot resolve native executable behind {shim}")


def _command_component_evidence(components: Sequence[str]) -> list[dict[str, Any]]:
    roles = (
        ["cli_executable"]
        if len(components) == 1
        else ["runtime", "cli_entry", *("component" for _ in components[2:])]
    )
    evidence = []
    for role, raw_path in zip(roles, components, strict=True):
        path = Path(raw_path)
        try:
            payload = _read_regular_file(path)
        except OSError as exc:
            evidence.append(
                {
                    "role": role,
                    "name": path.name,
                    "size": None,
                    "sha256": None,
                    "error": str(exc)[:256],
                }
            )
        else:
            evidence.append(
                {
                    "role": role,
                    "name": path.name,
                    "size": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "error": None,
                }
            )
    return evidence


def _runner_evidence_valid(evidence: Any) -> bool:
    if (
        not isinstance(evidence, dict)
        or evidence.get("components_unchanged") is not True
    ):
        return False
    snapshots = evidence.get("command_components")
    if not isinstance(snapshots, dict):
        return False
    if evidence.get("kind") not in {"codex", "claude"}:
        return False
    before = snapshots.get("before")
    after_version = snapshots.get("after_version")
    after = snapshots.get("after")
    if not isinstance(before, list) or before != after_version or before != after:
        return False
    if not before:
        return False
    expected_roles = (
        ["cli_executable"]
        if len(before) == 1
        else ["runtime", "cli_entry", *("component" for _ in before[2:])]
    )
    if [item.get("role") for item in before] != expected_roles:
        return False
    for item in before:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not item["name"]
            or not isinstance(item.get("size"), int)
            or item["size"] < 0
            or not _is_sha256(item.get("sha256"))
            or item.get("error") is not None
        ):
            return False
    if (
        not isinstance(evidence.get("version_output"), str)
        or not evidence["version_output"]
    ):
        return False
    if not _is_sha256(evidence.get("argv_instance_sha256")):
        return False
    policy = evidence.get("argv_policy")
    if not isinstance(policy, dict):
        return False
    policy_digest = evidence.get("argv_policy_sha256")
    if not _is_sha256(policy_digest) or policy_digest != _canonical_json_sha256(policy):
        return False
    external_files = evidence.get("external_files")
    if external_files is not None:
        if not isinstance(external_files, dict) or not external_files:
            return False
        if not all(
            _external_file_evidence_valid(item) for item in external_files.values()
        ):
            return False
    return True


def _formal_worker_evidence_valid(evidence: Any) -> bool:
    if (
        not isinstance(evidence, dict)
        or evidence.get("python_isolated") is not True
        or evidence.get("entry_module") != "inferref.agent.evaluation_worker"
    ):
        return False
    policy = evidence.get("launch_policy")
    if not isinstance(policy, dict):
        return False
    if policy.get("flags") != ["-I", "-m"]:
        return False
    if policy.get("entry_module") != "inferref.agent.evaluation_worker":
        return False
    if policy.get("request_transport") != "regular-file-json":
        return False
    if policy.get("request_schema_version") != "0.1":
        return False
    if not _is_sha256(policy.get("request_sha256")):
        return False
    digest = evidence.get("launch_policy_sha256")
    if not _is_sha256(digest) or digest != _canonical_json_sha256(policy):
        return False
    executable = evidence.get("python_executable")
    return executable is None or (
        isinstance(executable, dict)
        and isinstance(executable.get("name"), str)
        and bool(executable["name"])
        and isinstance(executable.get("size"), int)
        and executable["size"] >= 0
        and _is_sha256(executable.get("sha256"))
        and executable.get("error") is None
    )


def _argv_flag_pairs(command: Sequence[str]) -> list[tuple[str, str | None]]:
    pairs: list[tuple[str, str | None]] = []
    index = 0
    while index < len(command):
        token = command[index]
        if not token.startswith("-"):
            index += 1
            continue
        if "=" in token:
            key, value = token.split("=", 1)
            pairs.append((key, value))
            index += 1
            continue
        if index + 1 < len(command) and not command[index + 1].startswith("-"):
            pairs.append((token, command[index + 1]))
            index += 2
            continue
        pairs.append((token, None))
        index += 1
    return pairs


def _agent_argv_policy(agent: str, command: Sequence[str]) -> dict[str, Any]:
    """Path- and prompt-free normalized argv policy for public attestation."""

    pairs = _argv_flag_pairs(command)
    values = dict(pairs)
    if agent == "codex":
        config: dict[str, str] = {}
        for key, value in pairs:
            if key != "-c" or not isinstance(value, str) or "=" not in value:
                continue
            config_key, config_value = value.split("=", 1)
            if config_key in {
                "approval_policy",
                "windows.sandbox",
                "web_search",
                "mcp_servers.inferref.default_tools_approval_mode",
            }:
                try:
                    config[config_key] = json.loads(config_value)
                except json.JSONDecodeError:
                    config[config_key] = config_value
        return {
            "kind": "codex",
            "ephemeral": "--ephemeral" in values,
            "json_output": "--json" in values,
            "ignore_user_config": "--ignore-user-config" in values,
            "skip_git_repo_check": "--skip-git-repo-check" in values,
            "sandbox": values.get("--sandbox"),
            "model": values.get("--model"),
            "config": config,
        }
    if agent == "claude":
        return {
            "kind": "claude",
            "print": "--print" in values,
            "output_format": values.get("--output-format"),
            "verbose": "--verbose" in values,
            "effort": values.get("--effort"),
            "no_session_persistence": "--no-session-persistence" in values,
            "strict_mcp_config": "--strict-mcp-config" in values,
            "disable_slash_commands": "--disable-slash-commands" in values,
            "no_chrome": "--no-chrome" in values,
            "permission_mode": values.get("--permission-mode"),
            "model": values.get("--model"),
            "settings_present": "--settings" in values,
            "tools": values.get("--tools"),
            "allowed_tools": values.get("--allowedTools"),
        }
    return {"kind": agent}


def _capture_external_file(path: str | Path | None) -> dict[str, Any]:
    """No-follow capture of one external regular file, without publishing paths."""

    if path is None:
        return {
            "present": False,
            "kind": None,
            "size": None,
            "sha256": None,
            "file_id": None,
            "mtime_ns": None,
            "ctime_ns": None,
        }
    try:
        payload = _read_regular_file(Path(path))
    except OSError as exc:
        return {
            "present": True,
            "kind": None,
            "size": None,
            "sha256": None,
            "file_id": None,
            "mtime_ns": None,
            "ctime_ns": None,
            "error": str(exc)[:256],
        }
    status = os.lstat(Path(path))
    return {
        "present": True,
        "kind": "regular_file",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "file_id": f"{status.st_dev}:{status.st_ino}",
        "mtime_ns": status.st_mtime_ns,
        "ctime_ns": status.st_ctime_ns,
    }


def _external_file_pair(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, Any]:
    """Merge a before/after capture into a single unchanged verdict."""

    record = dict(before)
    record["after_sha256"] = after.get("sha256")
    record["after_kind"] = after.get("kind")
    record["after_size"] = after.get("size")
    record["after_file_id"] = after.get("file_id")
    record["after_mtime_ns"] = after.get("mtime_ns")
    record["after_ctime_ns"] = after.get("ctime_ns")
    if before.get("present") and after.get("present"):
        record["unchanged"] = (
            before.get("kind") == "regular_file"
            and after.get("kind") == "regular_file"
            and before.get("sha256") == after.get("sha256")
            and before.get("file_id") == after.get("file_id")
            and before.get("mtime_ns") == after.get("mtime_ns")
            and before.get("ctime_ns") == after.get("ctime_ns")
        )
    else:
        record["unchanged"] = before.get("present") == after.get("present")
    return record


def _external_file_evidence_valid(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if record.get("present") is False:
        return (
            record.get("kind") is None
            and record.get("size") is None
            and record.get("sha256") is None
            and record.get("after_sha256") is None
            and record.get("unchanged") is True
        )
    if record.get("present") is not True or record.get("kind") != "regular_file":
        return False
    if not isinstance(record.get("size"), int) or record["size"] < 0:
        return False
    if not _is_sha256(record.get("sha256")) or not _is_sha256(
        record.get("after_sha256")
    ):
        return False
    return record.get("unchanged") is True


def _model_request_satisfied(evidence: Any, agent: str) -> bool | None:
    """Whether the reported model satisfies the requested model (None = unknown)."""

    if not isinstance(evidence, dict) or evidence.get("agent") != agent:
        return None
    level = evidence.get("identity_level")
    if level == "requested_only":
        return None
    if level != "cli_self_reported":
        return None
    return evidence.get("matches_request") is True


def _aggregate_optional_bools(values: Iterable[bool | None]) -> bool | None:
    present = [value for value in values if value is not None]
    if any(value is False for value in present):
        return False
    if present and all(value is True for value in present):
        return True
    return None


def _model_evidence_valid(evidence: Any, agent: str) -> bool:
    if not isinstance(evidence, dict) or evidence.get("agent") != agent:
        return False
    requested = evidence.get("requested")
    reported = evidence.get("reported")
    level = evidence.get("identity_level")
    if not isinstance(requested, str) or not requested:
        return False
    if reported is None:
        return (
            level == "requested_only"
            and evidence.get("matches_request") is None
            and evidence.get("evidence_source") == "unavailable"
            and evidence.get("provider_verified") is False
        )
    return (
        isinstance(reported, str)
        and bool(reported)
        and level == "cli_self_reported"
        and evidence.get("matches_request") == (reported == requested)
        and isinstance(evidence.get("evidence_source"), str)
        and evidence.get("provider_verified") is False
    )


def _agent_version(executable: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            [*executable, "--version"],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=10,
            text=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    version = (completed.stdout or completed.stderr).strip()
    return version[:256] or None


def _claude_model_from_events(path: Path) -> tuple[str | None, str]:
    if not path.is_file():
        return None, "unavailable"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            isinstance(event, dict)
            and event.get("type") == "system"
            and event.get("subtype") == "init"
            and isinstance(event.get("model"), str)
        ):
            return event["model"][:256], "cli_system_init_event"
    return None, "unavailable"


def _codex_model_from_events(path: Path) -> tuple[str | None, str]:
    if not path.is_file():
        return None, "unavailable"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        model = event.get("model")
        if event_type in {
            "thread.started",
            "turn.started",
            "session.configured",
        } and isinstance(model, str):
            return model[:256], f"codex_{event_type.replace('.', '_')}_event"
    return None, "unavailable"


def _agent_reported_model(agent: str, path: Path) -> tuple[str | None, str]:
    if agent == "codex":
        return _codex_model_from_events(path)
    if agent == "claude":
        return _claude_model_from_events(path)
    return None, "unavailable"


def _agent_model_evidence(agent: str, requested: str, path: Path) -> dict[str, Any]:
    reported, source = _agent_reported_model(agent, path)
    return _model_evidence(agent, requested, reported, source)


def _model_evidence(
    agent: str, requested: str, reported: str | None, source: str
) -> dict[str, Any]:
    return {
        "requested": requested,
        "reported": reported,
        "evidence_source": source,
        "matches_request": reported == requested if reported is not None else None,
        "identity_level": (
            "cli_self_reported" if reported is not None else "requested_only"
        ),
        "provider_verified": False,
        "agent": agent,
    }


def _agent_api_failure(path: Path) -> str | None:
    if not path.is_file():
        return None
    for line in reversed(
        path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict) or event.get("type") != "result":
            continue
        status = event.get("api_error_status")
        detail = event.get("result")
        if isinstance(status, int):
            message = f"Agent API failed with HTTP {status}"
            if isinstance(detail, str) and detail:
                message += f": {detail[:384]}"
            return message
    return None


def _validate_event_stream(path: Path) -> str | None:
    if not path.is_file():
        return "Agent JSON event stream is missing"
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        return "Agent JSON event stream is empty"
    for line_number, line in enumerate(lines, start=1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            return f"Agent JSON event stream is malformed at line {line_number}"
        if not isinstance(event, dict):
            return f"Agent JSON event at line {line_number} is not an object"
    return None


def _run_process(
    command: list[str],
    *,
    cwd: Path,
    timeout_seconds: float,
    stdout_path: Path,
    stderr_path: Path,
) -> AgentProcessResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    process: subprocess.Popen[bytes] | None = None
    windows_job: int | None = None
    timed_out = False
    exit_code: int | None = None
    try:
        child_env = os.environ.copy()
        # A Codex process started by Codex Desktop must not inherit the parent
        # task's permission profile or thread identity.  The evaluation driver
        # supplies its own explicit sandbox and approval policy.
        for name in (
            "CODEX_INTERNAL_ORIGINATOR_OVERRIDE",
            "CODEX_PERMISSION_PROFILE",
            "CODEX_PERMISSION_PROFILE_TYPE",
            "CODEX_THREAD_ID",
        ):
            child_env.pop(name, None)
        options: dict[str, Any] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=child_env,
                shell=False,
                **options,
            )
            if os.name == "nt":
                windows_job = _assign_windows_kill_job(process)
            try:
                exit_code = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                _terminate_process_tree(process)
                exit_code = process.returncode
    except OSError as exc:
        stderr_path.write_text(str(exc), encoding="utf-8")
    finally:
        if windows_job is not None:
            _close_windows_job(windows_job)
        elif process is not None and os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    duration = round((time.perf_counter() - started) * 1000, 3)
    return AgentProcessResult(
        command=tuple(command),
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=duration,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout=_bounded_text(stdout_path),
        stderr=_bounded_text(stderr_path),
        usage=parse_agent_usage(stdout_path),
    )


def _candidate_prompt(benchmark: EvaluationBenchmark) -> str:
    task = (benchmark.directory / benchmark.task).read_text(encoding="utf-8")
    engine = (benchmark.directory / benchmark.editable_paths[0]).read_text(
        encoding="utf-8"
    )
    return (
        "Repair this isolated candidate. The complete task and initial engine are "
        "included below, so do not use shell commands to rediscover them. Use your "
        "file editing tool only for engine.py. "
        "You may edit only engine.py. You must call inferref_capabilities, then "
        f"inferref_context with path {VISIBLE_URI}, and use inferref_run_engine with "
        f"testcase={VISIBLE_URI}, adapter={CANDIDATE_URI}, runs_root={RUNS_URI}. "
        f"Use no more than {benchmark.max_engine_runs} engine runs. Finish only after "
        "the tool returns status pass. Do not create or modify any other file.\n\n"
        f"--- TASK.md ---\n{task}\n--- engine.py ---\n{engine}"
    )


def _execute_captured_case(
    case: EvaluationCase,
    *,
    engine_bytes: bytes,
    validation_root: Path,
    slot: str,
) -> dict[str, Any]:
    candidate_root = validation_root / "candidates" / slot
    candidate_root.mkdir(parents=True, exist_ok=False)
    engine = candidate_root / "engine.py"
    engine.write_bytes(engine_bytes)
    return execute_case(
        case,
        engine=engine,
        runs_root=validation_root / "runs" / slot,
    )


def _public_case_result(outcome: dict[str, Any]) -> dict[str, Any]:
    comparison = outcome.get("comparison") or {}
    return {
        "case": outcome.get("case"),
        "status": outcome.get("status"),
        "duration_ms": outcome.get("execution", {}).get("duration_ms"),
        "first_failure": comparison.get("first_failure"),
        "input_changes": outcome.get("integrity", {}).get("input_changes", []),
    }


def _protected_host_paths(benchmark: EvaluationBenchmark) -> dict[str, Path]:
    package_root = Path(__file__).resolve().parents[1]
    paths = {
        "benchmark/benchmark.json": benchmark.source,
        "benchmark/TASK.md": benchmark.directory / benchmark.task,
        "benchmark/engine.py": benchmark.directory / benchmark.editable_paths[0],
    }
    for name, entry in workspace_hashes(package_root).items():
        if entry.kind != "file" or _excluded_source_path(name):
            continue
        paths[f"evaluator/inferref/{name}"] = package_root / name
    return paths


def _hash_named_files(paths: dict[str, Path]) -> dict[str, str | None]:
    hashes: dict[str, str | None] = {}
    for name, path in paths.items():
        try:
            hashes[name] = hashlib.sha256(_read_regular_file(path)).hexdigest()
        except OSError:
            hashes[name] = None
    return hashes


def _sha256_or_none(path: Path) -> str | None:
    try:
        return hashlib.sha256(_read_regular_file(path)).hexdigest()
    except OSError:
        return None


def _workspace_entry_dict(entry: Any) -> dict[str, Any] | None:
    return entry.to_dict() if entry is not None else None


def _excluded_source_path(name: str) -> bool:
    parts = Path(name).parts
    return "__pycache__" in parts or name.endswith((".pyc", ".pyo"))


def _evaluator_source_manifest() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[1]
    files = {
        f"inferref/{name}": entry.to_dict()
        for name, entry in workspace_hashes(package_root).items()
        if entry.kind != "directory" and not _excluded_source_path(name)
    }
    encoded = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "source_tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": files,
    }


def _repository_evidence(cwd: Path) -> dict[str, Any]:
    root = _git_output(cwd, ["rev-parse", "--show-toplevel"])
    commit = _git_output(cwd, ["rev-parse", "HEAD"])
    status = _git_bytes(cwd, ["status", "--porcelain=v1", "--untracked-files=all"])
    diff = _git_bytes(cwd, ["diff", "--binary", "HEAD", "--", "."])
    valid_commit = commit if commit is not None and len(commit) == 40 else None
    return {
        "commit": valid_commit,
        "dirty": status is None or bool(status.strip()),
        "status_sha256": hashlib.sha256(status or b"").hexdigest(),
        "git_diff_sha256": hashlib.sha256(diff or b"").hexdigest(),
        "git_available": root is not None,
    }


def _runtime_evidence(cwd: Path) -> dict[str, Any]:
    """Describe the interpreter and installed distributions without local paths."""

    package_root = Path(__file__).resolve().parents[1]
    repository_root_text = _git_output(cwd, ["rev-parse", "--show-toplevel"])
    repository_relative_import = None
    if repository_root_text is not None:
        repository_root = Path(repository_root_text).resolve()
        if package_root.is_relative_to(repository_root):
            repository_relative_import = package_root.relative_to(
                repository_root
            ).as_posix()
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "byteorder": sys.byteorder,
        "numpy": np.__version__,
        "inferref": INFERREF_VERSION,
        "mcp_sdk": _distribution_version("mcp"),
        "import": {
            "mode": (
                "repository_source"
                if repository_relative_import is not None
                else "installed_distribution"
            ),
            "repository_relative_root": repository_relative_import,
        },
        "distributions": {
            name: _distribution_evidence(name) for name in ("inferref", "numpy", "mcp")
        },
    }


def _distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _distribution_evidence(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": None, "record_sha256": None}
    record_sha256 = None
    for item in distribution.files or ():
        if item.as_posix().endswith(".dist-info/RECORD"):
            record_sha256 = _sha256_or_none(Path(distribution.locate_file(item)))
            break
    return {
        "version": distribution.version,
        "record_sha256": record_sha256,
    }


def _git_output(cwd: Path, arguments: list[str]) -> str | None:
    payload = _git_bytes(cwd, arguments)
    if payload is None:
        return None
    return payload.decode("utf-8", errors="replace").strip()


def _git_bytes(cwd: Path, arguments: list[str]) -> bytes | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            shell=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _public_audit_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: record.get(key)
        for key in (
            "tool",
            "call_index",
            "engine_runs",
            "status",
            "operation",
            "diagnostic_codes",
        )
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _bounded_text(path: Path, limit: int = 65_536) -> str:
    if not path.is_file():
        return ""
    data = path.read_bytes()
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace")
    half = limit // 2
    return (
        data[:half].decode("utf-8", errors="replace")
        + "\n... output truncated ...\n"
        + data[-half:].decode("utf-8", errors="replace")
    )


def _walk_scalars(value: Any, prefix: str = ""):
    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_scalars(item, child)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_scalars(item, prefix)
    else:
        yield prefix, value
