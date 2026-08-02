"""Host orchestration and external-Agent drivers for evaluation v0.2."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from inferref.agent.adapter import (
    _assign_windows_kill_job,
    _close_windows_job,
    _terminate_process_tree,
)
from inferref.agent.evaluation import (
    CANDIDATE_URI,
    RUNS_URI,
    VISIBLE_URI,
    EvaluationBenchmark,
    engine_patch,
    execute_case,
    load_audit,
    prepare_workspace,
    workspace_hashes,
)


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
    resolved_model: str | None = None
    transcript_error: str | None = None


Driver = Callable[[str, str, EvaluationBenchmark, Path, Path, Path], AgentProcessResult]


def evaluate_benchmark(
    benchmark_path: str | Path,
    *,
    agents: Sequence[str],
    report_dir: str | Path,
    driver: Driver | None = None,
    claude_settings: str | Path | None = None,
    claude_model: str | None = None,
) -> dict[str, Any]:
    benchmark = EvaluationBenchmark.load(benchmark_path)
    selected = tuple(dict.fromkeys(agents))
    if not selected or any(name not in benchmark.drivers for name in selected):
        raise ValueError(
            "agents must select one or more configured drivers: "
            + ", ".join(sorted(benchmark.drivers))
        )
    report_root = Path(report_dir).resolve()
    if report_root.exists() and (
        not report_root.is_dir() or any(report_root.iterdir())
    ):
        raise FileExistsError(
            f"evaluation report directory is not empty: {report_root}"
        )
    report_root.mkdir(parents=True, exist_ok=True)
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
    passed = all(item["status"] == "pass" for item in results) and len(results) == 2
    report = {
        "format": "inferref-agent-evaluation-report",
        "format_version": "0.2",
        "benchmark": benchmark.id,
        "status": "pass" if passed else "fail",
        "acceptance": {
            "required_agents": ["codex", "claude"],
            "selected_agents": list(selected),
            "required_passes": 2,
            "passed": sum(item["status"] == "pass" for item in results),
        },
        "agents": results,
    }
    (report_root / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def assess_candidate(
    benchmark: EvaluationBenchmark,
    *,
    agent: str,
    model: str,
    workspace: Path,
    agent_root: Path,
    process: AgentProcessResult,
    initial_hashes: dict[str, str],
    final_hashes: dict[str, str],
    audit: list[dict[str, Any]],
    initial_host_hashes: dict[str, str | None] | None = None,
    final_host_hashes: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    editable = set(benchmark.editable_paths)
    changed = sorted(
        name
        for name in set(initial_hashes) | set(final_hashes)
        if initial_hashes.get(name) != final_hashes.get(name)
    )
    protected_changes = [name for name in changed if name not in editable]
    observed_tools = [
        item.get("tool") for item in audit if isinstance(item.get("tool"), str)
    ]
    missing_tools = sorted(set(benchmark.required_tools) - set(observed_tools))
    budget_attempts = [
        item
        for item in audit
        if "budget_exhausted" in (item.get("diagnostic_codes") or [])
    ]
    integrity_attempts = [
        item
        for item in audit
        if "integrity_failure" in (item.get("diagnostic_codes") or [])
    ]
    sequence_attempts = [
        item
        for item in audit
        if "required_tool_sequence" in (item.get("diagnostic_codes") or [])
    ]
    initial_host_hashes = initial_host_hashes or {}
    final_host_hashes = final_host_hashes or {}
    host_changes = sorted(
        name
        for name in set(initial_host_hashes) | set(final_host_hashes)
        if initial_host_hashes.get(name) != final_host_hashes.get(name)
    )
    visible_runs = [item for item in audit if item.get("tool") == "inferref_run_engine"]
    visible_pass = any(item.get("status") == "pass" for item in visible_runs)

    classification = "pass"
    message = "candidate passed visible and hidden evaluation cases"
    if process.timed_out:
        classification = "infrastructure_failure"
        message = "Agent process exceeded the wall-clock limit"
    elif process.transcript_error is not None:
        classification = "infrastructure_failure"
        message = process.transcript_error
    elif process.exit_code not in (0, None):
        classification = "infrastructure_failure"
        message = _agent_api_failure(process.stdout_path) or (
            f"Agent CLI exited with code {process.exit_code}"
        )
    elif not audit:
        classification = "infrastructure_failure"
        message = "evaluation MCP server produced no audit records"
    elif (
        protected_changes
        or host_changes
        or integrity_attempts
        or sequence_attempts
        or budget_attempts
        or missing_tools
    ):
        classification = "integrity_failure"
        message = "candidate violated editable-path, tool-use, or run-budget policy"
    elif not visible_pass:
        classification = "agent_failure"
        message = "candidate did not pass the visible case within the run budget"

    holdouts: list[dict[str, Any]] = []
    if classification == "pass":
        engine = workspace / benchmark.editable_paths[0]
        for case in benchmark.holdout_cases:
            outcome = execute_case(
                case,
                engine=engine,
                runs_root=agent_root / "holdout-runs",
            )
            holdouts.append(_public_case_result(outcome))
        if any(item["status"] != "pass" for item in holdouts):
            classification = "overfit_failure"
            message = "visible case passed but one or more hidden holdouts failed"

    template_engine = benchmark.directory / benchmark.editable_paths[0]
    candidate_engine = workspace / benchmark.editable_paths[0]
    patch = (
        engine_patch(template_engine, candidate_engine)
        if template_engine.is_file() and candidate_engine.is_file()
        else ""
    )
    return {
        "agent": agent,
        "model": model,
        "status": "pass" if classification == "pass" else "fail",
        "classification": classification,
        "message": message,
        "process": {
            "cli_version": process.cli_version,
            "resolved_model": process.resolved_model,
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
                    "before": initial_hashes.get(name),
                    "after": final_hashes.get(name),
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
        "tool_audit": audit,
        "visible": {
            "runs": len(visible_runs),
            "passed": visible_pass,
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
    command = _agent_command(
        agent,
        model,
        benchmark,
        workspace,
        session_root,
        audit_path,
        claude_settings=claude_settings,
        claude_model_override=claude_model_override,
    )
    result = _run_process(
        command,
        cwd=workspace,
        timeout_seconds=benchmark.max_wall_seconds,
        stdout_path=session_root.parent / "agent.stdout.jsonl",
        stderr_path=session_root.parent / "agent.stderr.log",
    )
    return replace(
        result,
        cli_version=_agent_version(agent),
        resolved_model=_agent_model_from_events(result.stdout_path),
        transcript_error=_validate_event_stream(result.stdout_path),
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
    claude_settings: str | Path | None = None,
    claude_model_override: str | None = None,
) -> list[str]:
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
            *_agent_executable("codex"),
            "exec",
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
            "mcp_servers.node_repl.enabled=false",
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
            *_agent_executable("claude"),
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


def _agent_executable(agent: str) -> list[str]:
    """Resolve Windows npm shims without invoking a command shell."""

    if os.name != "nt":
        executable = shutil.which(agent)
        if executable is None:
            raise FileNotFoundError(f"Agent CLI is not installed: {agent}")
        return [executable]
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
            return [str(native)]
    if agent == "codex":
        script = npm_root / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        node = shutil.which("node.exe") or shutil.which("node")
        if script.is_file() and node is not None:
            return [node, str(script)]
    raise FileNotFoundError(f"cannot resolve native executable behind {shim}")


def _agent_version(agent: str) -> str | None:
    try:
        completed = subprocess.run(
            [*_agent_executable(agent), "--version"],
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


def _agent_model_from_events(path: Path) -> str | None:
    if not path.is_file():
        return None
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
            return event["model"][:256]
    return None


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


def _public_case_result(outcome: dict[str, Any]) -> dict[str, Any]:
    comparison = outcome.get("comparison") or {}
    return {
        "case": outcome.get("case"),
        "status": outcome.get("status"),
        "duration_ms": outcome.get("execution", {}).get("duration_ms"),
        "first_failure": comparison.get("first_failure"),
    }


def _protected_host_paths(benchmark: EvaluationBenchmark) -> dict[str, Path]:
    module_root = Path(__file__).resolve().parent
    return {
        "benchmark/benchmark.json": benchmark.source,
        "benchmark/TASK.md": benchmark.directory / benchmark.task,
        "benchmark/engine.py": benchmark.directory / benchmark.editable_paths[0],
        "host/adapter.py": module_root / "adapter.py",
        "host/evaluation.py": module_root / "evaluation.py",
        "host/evaluation_host.py": Path(__file__).resolve(),
        "host/evaluation_mcp.py": module_root / "evaluation_mcp.py",
    }


def _hash_named_files(paths: dict[str, Path]) -> dict[str, str | None]:
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        for name, path in paths.items()
    }


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
