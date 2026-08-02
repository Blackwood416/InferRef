"""Shell-free engine execution followed by an InferRef comparison."""

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

from inferref.agent.protocol import AgentProtocolError, EngineAdapter
from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import TolerancePolicy
from inferref.ir.version import TESTCASE_FORMAT, TESTCASE_FORMAT_VERSION


def execute_adapter(
    testcase: str | Path,
    adapter: EngineAdapter,
    runs_root: str | Path,
    *,
    policy: TolerancePolicy | None = None,
    ignore_stride: bool = False,
    strict_layout: bool = False,
    first_failure: bool = True,
) -> dict[str, Any]:
    """Execute one trusted adapter in a fresh output directory and compare it."""

    testcase_path = Path(testcase).resolve()
    manifest_path = testcase_path / "testcase.json"
    if not manifest_path.is_file():
        raise AgentProtocolError(f"testcase manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise AgentProtocolError("testcase manifest root must be a JSON object")
    if manifest.get("format") != TESTCASE_FORMAT:
        raise AgentProtocolError(f"not an InferRef testcase: {testcase_path}")
    if manifest.get("format_version") != TESTCASE_FORMAT_VERSION:
        raise AgentProtocolError(
            f"unsupported testcase format_version {manifest.get('format_version')!r}"
        )
    if not manifest.get("reproducible", True):
        raise AgentProtocolError(
            "testcase is not independently reproducible; inspect its missing payload diagnostics"
        )

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
        python=Path(sys.executable).resolve(),
    )
    environment = os.environ.copy()
    environment.update(configured_env)
    environment["INFERREF_TESTCASE"] = str(testcase_path)
    environment["INFERREF_OUTPUT"] = str(output_path)

    started = time.perf_counter()
    execution: dict[str, Any] = {
        "command": list(command),
        "cwd": str(cwd),
        "timeout_seconds": adapter.timeout_seconds,
    }
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=adapter.timeout_seconds,
            shell=False,
            check=False,
        )
        execution.update(
            {
                "status": "completed",
                "exit_code": completed.returncode,
                "stdout": _bounded(completed.stdout, adapter.max_output_chars),
                "stderr": _bounded(completed.stderr, adapter.max_output_chars),
            }
        )
    except subprocess.TimeoutExpired as exc:
        execution.update(
            {
                "status": "timeout",
                "exit_code": None,
                "stdout": _bounded(_as_text(exc.stdout), adapter.max_output_chars),
                "stderr": _bounded(_as_text(exc.stderr), adapter.max_output_chars),
            }
        )
    except OSError as exc:
        execution.update(
            {
                "status": "error",
                "exit_code": None,
                "stdout": "",
                "stderr": str(exc),
            }
        )
    execution["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)

    result: dict[str, Any] = {
        "run_id": run_id,
        "adapter": adapter.to_dict(),
        "testcase": str(testcase_path),
        "output": str(output_path),
        "execution": execution,
        "comparison": None,
    }
    if execution["status"] != "completed" or execution["exit_code"] != 0:
        result["status"] = "execution_error"
    else:
        try:
            report = compare_testcase(
                testcase_path,
                output_path,
                policy=policy or TolerancePolicy(),
                ignore_stride=ignore_stride,
                strict_layout=strict_layout,
                first_failure=first_failure,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            result["status"] = "comparison_error"
            result["comparison"] = {
                "status": "error",
                "message": str(exc),
            }
        else:
            result["comparison"] = report.to_dict()
            result["status"] = "pass" if report.status == "pass" else "mismatch"

    (output_path / "inferref-run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def _bounded(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    half = max(1, (limit - 80) // 2)
    omitted = len(value) - 2 * half
    return value[:half] + f"\n... {omitted} character(s) omitted ...\n" + value[-half:]


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return (
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
    )
