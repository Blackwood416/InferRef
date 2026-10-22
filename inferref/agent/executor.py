"""Shell-free engine execution followed by InferRef comparison."""

from __future__ import annotations

import json
import os
import re
import string
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
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
from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import DEFAULT_TOLERANCES, TolerancePolicy
from inferref.comparison.resolution import resolve_comparison_policy
from inferref.comparison.schema import ComparisonSpec
from inferref.contracts import contract_requirements
from inferref.testcase.requirements import testcase_requirements
from inferref.testcase.validate import require_valid_testcase

# Batch wire format written for, and read back from, one adapter process
# (InferRef 0.12 section 7.3, mirrored from cpp/include/inferref/bridge.hpp).
_BATCH_FORMAT = "inferref-batch"
_BATCH_FORMAT_VERSION = "1.0"
_BATCH_RESULTS_FORMAT = "inferref-batch-results"
_BATCH_MANIFEST_FILE_NAME = "batch-manifest.json"
_BATCH_RESULTS_FILE_NAME = "batch-results.json"
_BATCH_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _run_command(
    command: tuple[str, ...],
    environment: dict[str, str],
    *,
    cwd: Path,
    output_path: Path,
    timeout_seconds: float,
    max_output_chars: int,
    max_artifact_bytes: int,
    max_artifact_files: int,
) -> dict[str, Any]:
    """Launch one engine process and record its bounded outcome."""
    started = time.perf_counter()
    execution: dict[str, Any] = {
        "command": list(command),
        "cwd": str(cwd),
        "timeout_seconds": timeout_seconds,
    }
    stdout = StreamCapture(output_path / "inferref-stdout.log", max_output_chars)
    stderr = StreamCapture(output_path / "inferref-stderr.log", max_output_chars)
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
            deadline=time.monotonic() + timeout_seconds,
            output_path=output_path,
            max_artifact_bytes=max_artifact_bytes,
            max_artifact_files=max_artifact_files,
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
                "max_output_bytes_per_stream": max_output_chars,
                "artifact_bytes": artifact.total_bytes,
                "artifact_files": artifact.files,
                "artifact_scan_entries": artifact.entries,
                "max_artifact_bytes": max_artifact_bytes,
                "max_artifact_files": max_artifact_files,
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
    return execution


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
    from inferref.quantization.artifacts import needs_quantization, validate_comparison

    validate_comparison(validation.manifest, effective_comp)
    capability_status = "unchecked"
    if adapter.capabilities is None and needs_quantization(requirements):
        return {
            "run_id": None, "adapter": adapter.to_dict(), "testcase": str(testcase_path),
            "requirements": requirements, "capability_status": "unsupported",
            "status": "unsupported", "execution": None, "comparison": None,
            "effective_comparison": effective_comp.to_dict(),
            "unsupported": ["quantized testcases require explicit adapter capabilities"],
        }
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
    execution = _run_command(
        command,
        environment,
        cwd=cwd,
        output_path=output_path,
        timeout_seconds=adapter.timeout_seconds,
        max_output_chars=adapter.max_output_chars,
        max_artifact_bytes=adapter.max_artifact_bytes,
        max_artifact_files=adapter.max_artifact_files,
    )
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
            if rep_dict.get("comparator"):
                result["comparator"] = rep_dict["comparator"]
            result["status"] = "pass" if report.status == "pass" else ("error" if report.status == "error" else "mismatch")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result.update(
                status="comparison_error",
                comparison={"status": "error", "message": str(exc)},
            )
    write_run_record(output_path, result)
    return result


def _batch_requests(
    items: Sequence[tuple[str, str | Path]],
) -> tuple[tuple[str, Path], ...]:
    """Validate the requested batch order and resolve every testcase directory."""

    if not items:
        raise AgentProtocolError("batch execution requires at least one item")
    requests: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        try:
            item_id, testcase = item
        except (TypeError, ValueError) as exc:
            raise AgentProtocolError(
                f"batch item {index} must be an (id, testcase_dir) pair"
            ) from exc
        if not isinstance(item_id, str) or not _BATCH_ID_PATTERN.fullmatch(item_id):
            raise AgentProtocolError(
                f"batch item {index} id must be an identifier such as 'step-0': "
                f"{item_id!r}"
            )
        if item_id in seen:
            raise AgentProtocolError(f"duplicate batch item id {item_id!r}")
        seen.add(item_id)
        testcase_dir = Path(testcase).resolve()
        if not testcase_dir.is_dir():
            raise AgentProtocolError(
                f"batch item {item_id!r} testcase directory does not exist: "
                f"{testcase_dir}"
            )
        requests.append((item_id, testcase_dir))
    return tuple(requests)


def _per_testcase_command_placeholders(command: Sequence[str]) -> list[str]:
    formatter = string.Formatter()
    found: set[str] = set()
    for value in command:
        found.update(
            field
            for _, field, _, _ in formatter.parse(value)
            if field in {"testcase", "output"}
        )
    return sorted(found)


def _batch_results(
    results_path: Path, requests: Sequence[tuple[str, Path, Path]]
) -> list[dict[str, Any]]:
    """Map ``batch-results.json`` entries back onto the requested order."""

    try:
        data = json.loads(results_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AgentProtocolError(
            f"cannot read batch results {results_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise AgentProtocolError(f"batch results are not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise AgentProtocolError("batch results root must be a JSON object")
    if data.get("format") != _BATCH_RESULTS_FORMAT:
        raise AgentProtocolError(
            f"batch results format must be {_BATCH_RESULTS_FORMAT!r}"
        )
    if data.get("format_version") != _BATCH_FORMAT_VERSION:
        raise AgentProtocolError(
            "unsupported batch results format_version; this build reads "
            f"{_BATCH_FORMAT_VERSION!r}"
        )
    entries = data.get("items")
    if not isinstance(entries, list):
        raise AgentProtocolError("batch results 'items' must be an array")
    output_dirs = {item_id: output_dir for item_id, _, output_dir in requests}
    mapped: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(entries):
        context = f"batch results item {index}"
        if not isinstance(entry, dict):
            raise AgentProtocolError(f"{context} must be a JSON object")
        item_id = entry.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise AgentProtocolError(f"{context} is missing a non-empty 'id'")
        if item_id in mapped:
            raise AgentProtocolError(f"duplicate batch result id {item_id!r}")
        if item_id not in output_dirs:
            raise AgentProtocolError(
                f"batch results contain an id that was not requested: {item_id!r}"
            )
        status = entry.get("status")
        if status not in {"passed", "failed"}:
            raise AgentProtocolError(f"{context} status must be 'passed' or 'failed'")
        exit_code = entry.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise AgentProtocolError(f"{context} requires an integer 'exit_code'")
        if (status == "passed") != (exit_code == 0):
            raise AgentProtocolError(
                f"{context} status {status!r} contradicts exit_code {exit_code}"
            )
        reported_output = entry.get("output_dir")
        if not isinstance(reported_output, str) or not reported_output:
            raise AgentProtocolError(f"{context} requires a non-empty 'output_dir'")
        error = entry.get("error")
        if error is not None and not isinstance(error, str):
            raise AgentProtocolError(f"{context} 'error' must be a string or null")
        mapped[item_id] = {
            "id": item_id,
            "status": status,
            "exit_code": exit_code,
            # The manifest named the output directory, so keep that value rather
            # than trusting an adapter echo.
            "output_dir": str(output_dirs[item_id]),
            "error": error,
        }
    missing = sorted(set(output_dirs) - set(mapped))
    if missing:
        raise AgentProtocolError(
            "batch results are missing item id(s): " + ", ".join(missing)
        )
    return [mapped[item_id] for item_id in output_dirs]


def execute_adapter_batch(
    items: Sequence[tuple[str, str | Path]],
    adapter: EngineAdapter,
    runs_root: str | Path,
    *,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Execute ordered testcase directories in one adapter process.

    The helper owns the batch transport only: it writes a versioned
    ``inferref-batch`` manifest, launches the adapter once with
    ``--batch-manifest``, and maps ``batch-results.json`` back onto the
    requested order. Per-item comparison, run records, and the decision to
    batch at all stay with the caller, so an adapter without the
    ``batch_execution`` feature is the caller's business.

    A command that still expands ``{testcase}``/``{output}`` describes one
    testcase per launch, so no batch can be built from it without guessing:
    the result is ``unsupported`` and the caller falls back to sequential
    execution. Adapters that declare ``batch_execution`` may keep those
    placeholders in their environment; they expand to this batch's manifest and
    output root, and ``INFERREF_BATCH_MANIFEST``/``INFERREF_BATCH_OUTPUT`` name
    the same paths. Per-item paths come from the manifest, never from those
    variables.

    One process emits every item's artifacts, so the adapter's per-testcase
    timeout and artifact budgets are multiplied by the item count; pass
    ``timeout_seconds`` to replace the timeout for one batch.
    """

    requests = _batch_requests(items)
    per_testcase = _per_testcase_command_placeholders(adapter.command)
    if per_testcase:
        diagnostic = (
            f"adapter {adapter.name!r} command references per-testcase "
            "placeholder(s) "
            + ", ".join(f"{{{name}}}" for name in per_testcase)
            + "; a batch launch needs a command that accepts --batch-manifest, "
            "so the caller must fall back to sequential execution"
        )
        return {
            "status": "unsupported",
            "adapter": adapter.to_dict(),
            "manifest": None,
            "batch_output": None,
            "execution": None,
            "results": [],
            "counts": {"total": len(requests), "passed": 0, "failed": 0},
            "unsupported": [diagnostic],
            "error": diagnostic,
        }

    cwd = adapter.working_directory()
    if not cwd.is_dir():
        raise AgentProtocolError(f"adapter working directory does not exist: {cwd}")
    output_root = Path(runs_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    batch_dir = output_root / f"{stamp}-{uuid.uuid4().hex[:8]}"
    batch_dir.mkdir()
    manifest_path = batch_dir / _BATCH_MANIFEST_FILE_NAME
    ordered: list[tuple[str, Path, Path]] = []
    manifest_items: list[dict[str, str]] = []
    for item_id, testcase_dir in requests:
        item_output = batch_dir / item_id
        item_output.mkdir()
        ordered.append((item_id, testcase_dir, item_output))
        manifest_items.append(
            {
                "id": item_id,
                "testcase_dir": testcase_dir.as_posix(),
                "output_dir": item_output.as_posix(),
            }
        )
    manifest_path.write_text(
        json.dumps(
            {
                "format": _BATCH_FORMAT,
                "format_version": _BATCH_FORMAT_VERSION,
                "items": manifest_items,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    replacements = {
        "testcase": str(manifest_path),
        "output": str(batch_dir),
        "adapter_dir": str(adapter.directory),
        "python": str(Path(sys.executable).absolute()),
    }
    command = (
        *(item.format_map(replacements) for item in adapter.command),
        "--batch-manifest",
        str(manifest_path),
    )
    configured_env = {
        key: value.format_map(replacements)
        for key, value in adapter.environment.items()
    }
    environment = os.environ.copy()
    environment.update(configured_env)
    environment.update(
        {
            "INFERREF_BATCH_MANIFEST": str(manifest_path),
            "INFERREF_BATCH_OUTPUT": str(batch_dir),
        }
    )
    item_count = len(ordered)
    execution = _run_command(
        command,
        environment,
        cwd=cwd,
        output_path=batch_dir,
        timeout_seconds=(
            timeout_seconds
            if timeout_seconds is not None
            else adapter.timeout_seconds * item_count
        ),
        max_output_chars=adapter.max_output_chars,
        max_artifact_bytes=adapter.max_artifact_bytes * item_count,
        max_artifact_files=adapter.max_artifact_files * item_count,
    )

    results: list[dict[str, Any]] = []
    results_error: str | None = None
    try:
        results = _batch_results(batch_dir / _BATCH_RESULTS_FILE_NAME, ordered)
    except AgentProtocolError as exc:
        results_error = str(exc)
    failed = sum(1 for item in results if item["status"] != "passed")
    if execution["status"] != "completed":
        status = execution["status"]
    elif results_error is not None:
        status = "results_error"
    elif failed:
        status = "failed"
    elif execution["exit_code"] != 0:
        # A clean per-item report and a non-zero batch exit contradict each
        # other; surface the process code instead of passing the batch.
        status = "execution_error"
    else:
        status = "pass"
    error = results_error
    if error is None and execution["status"] == "error":
        error = execution["stderr"]
    return {
        "status": status,
        "adapter": adapter.to_dict(),
        "manifest": str(manifest_path),
        "batch_output": str(batch_dir),
        "execution": execution,
        "results": results,
        "counts": {
            "total": item_count,
            "passed": len(results) - failed,
            "failed": failed,
        },
        "unsupported": [],
        "error": error,
    }
