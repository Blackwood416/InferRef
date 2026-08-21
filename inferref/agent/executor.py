"""Shell-free engine execution followed by InferRef comparison."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inferref.agent.process_policy import (
    StreamCapture,
    assign_windows_kill_job,
    close_windows_job,
    terminate_process_tree,
    wait_with_limits,
)
from inferref.agent.protocol import AgentProtocolError, EngineAdapter
from inferref.agent.run_record import write_run_record
from inferref.comparators.numeric import NUMERIC_COMPARATOR_ID
from inferref.comparators.protocol import Artifact
from inferref.comparators.runner import run_comparator
from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy
from inferref.comparison.resolution import resolve_comparison_policy
from inferref.comparison.schema import ComparisonSpec
from inferref.contracts import contract_requirements
from inferref.testcase.requirements import testcase_requirements
from inferref.testcase.validate import require_valid_testcase


def execute_adapter(
    testcase: str | Path,
    adapter: EngineAdapter,
    runs_root: str | Path,
    *,
    suite_spec: ComparisonSpec | dict[str, Any] | None = None,
    comparator: str | None = None,
    comparison_config: dict[str, Any] | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    tolerance: str | Path | dict[str, Any] | None = None,
    policy: TolerancePolicy | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> dict[str, Any]:
    testcase_path = Path(testcase).resolve()
    validation = require_valid_testcase(testcase_path)
    if not validation.reproducible:
        blockers = ", ".join(issue.code for issue in validation.issues) or "unknown"
        raise AgentProtocolError(
            f"testcase is not independently reproducible after validation (blockers: {blockers})"
        )

    tc_spec = validation.manifest.get("comparison")
    cli_atol = policy.override_atol if policy and policy.override_atol is not None else atol
    cli_rtol = policy.override_rtol if policy and policy.override_rtol is not None else rtol
    cli_tolerance = policy.per_dtype if (policy and policy.per_dtype != DEFAULT_TOLERANCES) else tolerance

    effective_comp = resolve_comparison_policy(
        testcase_spec=tc_spec,
        suite_spec=suite_spec,
        cli_comparator=comparator,
        cli_atol=cli_atol,
        cli_rtol=cli_rtol,
        cli_strict_layout=strict_layout if strict_layout is not False else None,
        cli_ignore_stride=ignore_stride if ignore_stride is not False else None,
        cli_tolerance=cli_tolerance,
        cli_config=comparison_config,
    )

    # Pre-flight check: validate comparator plugin exists and config is valid before launching engine process
    try:
        effective_comp.validate()
    except Exception as exc:
        raise AgentProtocolError(f"invalid comparison policy: {exc}") from exc

    requirements = testcase_requirements(validation.manifest)
    capability_status = "unchecked"
    if adapter.capabilities is not None:
        capability_status = adapter.capabilities.assessment(requirements)
        incompatible = list(adapter.capabilities.incompatibilities(requirements))
        for contract in requirements.get("contracts", []):
            per_contract = contract_requirements(validation.manifest, contract)
            incompatible.extend(
                adapter.capabilities.contract_incompatibilities(contract, per_contract)
            )
        if incompatible:
            return {
                "run_id": None,
                "adapter": adapter.to_dict(),
                "testcase": str(testcase_path),
                "requirements": requirements,
                "capability_status": "unsupported",
                "status": "unsupported",
                "execution": None,
                "comparison": None,
                "effective_comparison": effective_comp.to_dict(),
                "unsupported": incompatible,
            }

    cwd = adapter.working_directory()
    if not cwd.is_dir():
        raise AgentProtocolError(f"adapter working directory does not exist: {cwd}")
    output_root = Path(runs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
    output_path = output_root / run_id
    output_path.mkdir()
    command, configured_env = adapter.expand(
        testcase=testcase_path,
        output=output_path,
        python=Path(sys.executable).absolute(),
    )
    environment = os.environ.copy()
    environment.update(configured_env)
    environment.update(
        {"INFERREF_TESTCASE": str(testcase_path), "INFERREF_OUTPUT": str(output_path)}
    )
    started = time.perf_counter()
    execution: dict[str, Any] = {
        "command": list(command),
        "cwd": str(cwd),
        "timeout_seconds": adapter.timeout_seconds,
    }
    stdout = StreamCapture(
        output_path / "inferref-stdout.log", adapter.max_output_chars
    )
    stderr = StreamCapture(
        output_path / "inferref-stderr.log", adapter.max_output_chars
    )
    process: subprocess.Popen[bytes] | None = None
    windows_job: int | None = None
    try:
        options: dict[str, Any] = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt"
            else {"start_new_session": True}
        )
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **options,
        )
        if os.name == "nt":
            windows_job = assign_windows_kill_job(process)
        stdout.start(process.stdout)
        stderr.start(process.stderr)
        process_status, artifact = wait_with_limits(
            process,
            deadline=time.monotonic() + adapter.timeout_seconds,
            output_path=output_path,
            max_artifact_bytes=adapter.max_artifact_bytes,
            max_artifact_files=adapter.max_artifact_files,
            captures=(stdout, stderr),
            windows_job=windows_job,
        )
        windows_job = None
        stdout.join()
        stderr.join()
        execution.update(
            {
                "status": process_status,
                "exit_code": process.returncode,
                "stdout": stdout.text(),
                "stderr": stderr.text(),
                "stdout_path": stdout.path.name,
                "stderr_path": stderr.path.name,
                "stdout_bytes": stdout.observed_bytes,
                "stderr_bytes": stderr.observed_bytes,
                "max_output_bytes_per_stream": adapter.max_output_chars,
                "artifact_bytes": artifact.total_bytes,
                "artifact_files": artifact.files,
                "artifact_scan_entries": artifact.entries,
                "max_artifact_bytes": adapter.max_artifact_bytes,
                "max_artifact_files": adapter.max_artifact_files,
                "process_tree_strategy": "windows_job_object"
                if os.name == "nt"
                else "posix_process_group",
            }
        )
    except OSError as exc:
        if windows_job is not None:
            close_windows_job(windows_job)
        if process is not None and process.poll() is None:
            terminate_process_tree(process)
        execution.update(
            {"status": "error", "exit_code": None, "stdout": "", "stderr": str(exc)}
        )
    execution["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    result: dict[str, Any] = {
        "run_id": run_id,
        "adapter": adapter.to_dict(),
        "testcase": str(testcase_path),
        "output": str(output_path),
        "execution": execution,
        "comparison": None,
        "effective_comparison": effective_comp.to_dict(),
        "requirements": requirements,
        "capability_status": capability_status,
    }
    if execution["status"] != "completed":
        result["status"] = execution["status"]
    elif execution["exit_code"] != 0:
        result["status"] = "execution_error"
    else:
        try:
            report = compare_testcase(
                testcase_path,
                output_path,
                effective_comparison=effective_comp,
                first_failure=first_failure,
            )
            rep_dict = report.to_dict()
            result["comparison"] = rep_dict
            if "comparator" in rep_dict and rep_dict["comparator"]:
                result["comparator"] = rep_dict["comparator"]
            result["status"] = "pass" if report.status == "pass" else ("error" if report.status == "error" else "mismatch")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result.update(
                status="comparison_error",
                comparison={"status": "error", "message": str(exc)},
            )
    write_run_record(output_path, result)
    return result
