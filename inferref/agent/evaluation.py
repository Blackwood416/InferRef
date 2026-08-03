"""Blind, host-side evaluation of coding Agents using InferRef diagnostics."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from inferref.agent.adapter import (
    _assign_windows_kill_job,
    _close_windows_job,
    _StreamCapture,
    _terminate_process_tree,
    _wait_with_limits,
)
from inferref.agent.protocol import AgentProtocolError, AgentResponse
from inferref.compare.compare import (
    STATUS_ERROR,
    STATUS_MISSING,
    ComparisonReport,
    TensorComparison,
    compare_tensors,
)
from inferref.compare.tolerance import TolerancePolicy
from inferref.ir.paths import resolve_contained_path
from inferref.tensor import codec
from inferref.tensor.codec import TensorView

EVALUATION_FORMAT = "inferref-agent-evaluation"
EVALUATION_VERSION = "0.2"
VISIBLE_URI = "eval://visible"
CANDIDATE_URI = "eval://candidate"
RUNS_URI = "eval://runs"
REQUIRED_TOOLS = (
    "inferref_capabilities",
    "inferref_context",
    "inferref_run_engine",
)


@dataclass(frozen=True)
class WorkspaceEntry:
    """A no-follow filesystem entry used by evaluation integrity checks."""

    kind: str
    size: int | None
    mode: int
    sha256: str | None = None
    link_target: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "size": self.size,
            "mode": self.mode,
            "sha256": self.sha256,
            "link_target": self.link_target,
        }


@dataclass(frozen=True)
class AuditLog(Sequence[dict[str, Any]]):
    """Strictly parsed evaluation evidence, excluding its session footer."""

    records: tuple[dict[str, Any], ...] = ()
    valid: bool = False
    error: str | None = None
    footer: dict[str, Any] | None = None

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)

    def to_integrity_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "error": self.error,
            "record_count": len(self.records),
            "records_sha256": (
                self.footer.get("records_sha256") if self.footer is not None else None
            ),
        }


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    seed: int
    shape: tuple[int, int, int, int]

    @classmethod
    def from_dict(cls, data: Any, *, where: str) -> EvaluationCase:
        if not isinstance(data, dict):
            raise AgentProtocolError(f"{where} must be an object")
        identifier = data.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise AgentProtocolError(f"{where}.id must be a non-empty string")
        seed = data.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise AgentProtocolError(f"{where}.seed must be an integer")
        raw_shape = data.get("shape")
        if (
            not isinstance(raw_shape, list)
            or len(raw_shape) != 4
            or not all(
                isinstance(item, int) and not isinstance(item, bool) and item > 0
                for item in raw_shape
            )
        ):
            raise AgentProtocolError(
                f"{where}.shape must contain four positive integers"
            )
        shape = tuple(raw_shape)
        if shape[-1] % 2:
            raise AgentProtocolError(f"{where}.shape[-1] must be even")
        return cls(id=identifier, seed=seed, shape=shape)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "seed": self.seed, "shape": list(self.shape)}


@dataclass(frozen=True)
class AgentDriverSpec:
    name: str
    model: str


@dataclass(frozen=True)
class EvaluationSuccessPolicy:
    required_agent_passes: int
    visible_status: str
    all_holdouts_pass: bool
    protected_paths_unchanged: bool

    @classmethod
    def from_dict(cls, data: Any, *, configured_agents: int) -> EvaluationSuccessPolicy:
        if not isinstance(data, dict):
            raise AgentProtocolError("success must be an object")
        required = _bounded_integer(
            data,
            "required_agent_passes",
            minimum=1,
            maximum=configured_agents,
        )
        visible_status = _required_string(data, "visible_status")
        if visible_status != "pass":
            raise AgentProtocolError("success.visible_status must be 'pass' in v0.2")
        return cls(
            required_agent_passes=required,
            visible_status=visible_status,
            all_holdouts_pass=_required_bool(
                data, "all_holdouts_pass", where="success"
            ),
            protected_paths_unchanged=_required_bool(
                data, "protected_paths_unchanged", where="success"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_agent_passes": self.required_agent_passes,
            "visible_status": self.visible_status,
            "all_holdouts_pass": self.all_holdouts_pass,
            "protected_paths_unchanged": self.protected_paths_unchanged,
        }


@dataclass(frozen=True)
class EvaluationBenchmark:
    source: Path
    source_sha256: str
    id: str
    title: str
    task: str
    workspace_template: tuple[str, ...]
    editable_paths: tuple[str, ...]
    required_tools: tuple[str, ...]
    max_engine_runs: int
    max_wall_seconds: int
    visible_case: EvaluationCase
    holdout_cases: tuple[EvaluationCase, ...]
    drivers: dict[str, AgentDriverSpec]
    success: EvaluationSuccessPolicy

    @classmethod
    def load(cls, path: str | Path) -> EvaluationBenchmark:
        source = Path(path).resolve()
        source_bytes = _read_regular_file(source)
        data = json.loads(source_bytes.decode("utf-8", errors="strict"))
        if not isinstance(data, dict):
            raise AgentProtocolError("evaluation manifest root must be an object")
        if data.get("format") != EVALUATION_FORMAT:
            raise AgentProtocolError("not an InferRef Agent evaluation manifest")
        if data.get("format_version") != EVALUATION_VERSION:
            raise AgentProtocolError(
                f"unsupported evaluation format_version {data.get('format_version')!r}; "
                f"expected {EVALUATION_VERSION!r}"
            )
        identifier = _required_string(data, "id")
        title = _required_string(data, "title")
        task = _required_string(data, "task")
        templates = _string_array(data.get("workspace_template"), "workspace_template")
        candidate = data.get("candidate")
        if not isinstance(candidate, dict):
            raise AgentProtocolError("candidate must be an object")
        editable = _string_array(
            candidate.get("editable_paths"), "candidate.editable_paths"
        )
        if task not in templates:
            raise AgentProtocolError("task must name a file in workspace_template")
        if len(editable) != 1 or any(path not in templates for path in editable):
            raise AgentProtocolError(
                "candidate.editable_paths must name exactly one workspace template file"
            )
        required = _string_array(
            candidate.get("required_tools"), "candidate.required_tools"
        )
        if set(required) != set(REQUIRED_TOOLS):
            raise AgentProtocolError(
                "candidate.required_tools must contain exactly the three InferRef evaluation tools"
            )
        max_runs = _bounded_integer(candidate, "max_engine_runs", minimum=1, maximum=32)
        max_wall = _bounded_integer(
            candidate, "max_wall_seconds", minimum=30, maximum=3600
        )
        cases = data.get("cases")
        if not isinstance(cases, dict):
            raise AgentProtocolError("cases must be an object")
        visible = EvaluationCase.from_dict(cases.get("visible"), where="cases.visible")
        raw_holdouts = cases.get("holdouts")
        if not isinstance(raw_holdouts, list) or not raw_holdouts:
            raise AgentProtocolError("cases.holdouts must be a non-empty array")
        holdouts = tuple(
            EvaluationCase.from_dict(item, where=f"cases.holdouts[{index}]")
            for index, item in enumerate(raw_holdouts)
        )
        all_ids = [visible.id, *(item.id for item in holdouts)]
        if len(set(all_ids)) != len(all_ids):
            raise AgentProtocolError("evaluation case ids must be unique")
        raw_drivers = data.get("drivers")
        if not isinstance(raw_drivers, dict):
            raise AgentProtocolError("drivers must be an object")
        drivers: dict[str, AgentDriverSpec] = {}
        for name in ("codex", "claude"):
            entry = raw_drivers.get(name)
            if not isinstance(entry, dict):
                raise AgentProtocolError(f"drivers.{name} must be an object")
            drivers[name] = AgentDriverSpec(
                name=name, model=_required_string(entry, "model")
            )
        success = EvaluationSuccessPolicy.from_dict(
            data.get("success"), configured_agents=len(drivers)
        )
        return cls(
            source=source,
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            id=identifier,
            title=title,
            task=task,
            workspace_template=templates,
            editable_paths=editable,
            required_tools=required,
            max_engine_runs=max_runs,
            max_wall_seconds=max_wall,
            visible_case=visible,
            holdout_cases=holdouts,
            drivers=drivers,
            success=success,
        )

    @property
    def directory(self) -> Path:
        return self.source.parent


@dataclass
class EvaluationSession:
    benchmark: EvaluationBenchmark
    workspace: Path
    root: Path
    audit_path: Path
    engine_runs: int = 0
    call_counts: dict[str, int] = field(default_factory=dict)
    successful_tools: set[str] = field(default_factory=set)
    _audit_records: int = field(default=0, init=False, repr=False)
    _audit_digest: Any = field(default_factory=hashlib.sha256, init=False, repr=False)
    _audit_lines: list[bytes] = field(default_factory=list, init=False, repr=False)
    _audit_lock_acquired: bool = field(default=False, init=False, repr=False)
    _audit_finalized: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self.workspace = self.workspace.resolve()
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.audit_path = self.audit_path.resolve()
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def audit(self, tool: str, response: AgentResponse) -> None:
        if self._audit_finalized:
            raise RuntimeError("evaluation audit session is already finalized")
        self.call_counts[tool] = self.call_counts.get(tool, 0) + 1
        if response.status in {"ok", "pass"}:
            self.successful_tools.add(tool)
        self._audit_records += 1
        record = {
            "type": "tool_call",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "call_index": self._audit_records,
            "engine_runs": self.engine_runs,
            "status": response.status,
            "operation": response.operation,
            "diagnostic_codes": [
                code
                for item in response.diagnostics
                if isinstance((code := item.get("code")), str)
            ],
        }
        encoded = _audit_line(record)
        self._audit_digest.update(encoded)
        self._audit_lines.append(encoded)
        self._write_audit_checkpoint()

    def finalize_audit(self) -> None:
        """Seal the evidence stream with a count and cumulative digest."""

        if self._audit_finalized:
            return
        if self._audit_records == 0:
            self._audit_finalized = True
            return
        self._write_audit_checkpoint()
        self._audit_finalized = True

    def _write_audit_checkpoint(self) -> None:
        if not self._audit_lock_acquired:
            lock_path = self.audit_path.with_name(self.audit_path.name + ".lock")
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            self._audit_lock_acquired = True
        footer = {
            "type": "session_footer",
            "record_count": self._audit_records,
            "records_sha256": self._audit_digest.hexdigest(),
        }
        payload = b"".join(self._audit_lines) + _audit_line(footer)
        temporary = self.audit_path.with_name(
            f".{self.audit_path.name}.{os.getpid()}.tmp"
        )
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.audit_path)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def capabilities(self) -> AgentResponse:
        from inferref.agent.service import capabilities

        base = capabilities()
        data = dict(base.data)
        data["mcp_tools"] = list(REQUIRED_TOOLS)
        data["operations"] = [
            item
            for item in data.get("operations", [])
            if item.get("name") in {"context", "run_engine"}
        ]
        data["evaluation"] = {
            "format": EVALUATION_FORMAT,
            "version": EVALUATION_VERSION,
            "benchmark": self.benchmark.id,
            "visible_testcase": VISIBLE_URI,
            "candidate_adapter": CANDIDATE_URI,
            "runs_root": RUNS_URI,
            "max_engine_runs": self.benchmark.max_engine_runs,
            "oracle_isolation": "reference_payloads_not_materialized",
        }
        return AgentResponse(
            operation=base.operation,
            status=base.status,
            data=data,
            diagnostics=base.diagnostics,
            next_actions=base.next_actions,
        )

    def context(self, path: str) -> AgentResponse:
        if "inferref_capabilities" not in self.successful_tools:
            return AgentResponse.error(
                "context",
                "inferref_capabilities must succeed before inferref_context",
                code="required_tool_sequence",
            )
        if path != VISIBLE_URI:
            return AgentResponse.error(
                "context",
                f"evaluation artifact must be {VISIBLE_URI!r}",
                code="path_not_allowed",
            )
        inputs, references = rope_arrays(self.benchmark.visible_case)
        return AgentResponse(
            operation="context",
            status="ok",
            data={
                "artifact": "testcase",
                "path": VISIBLE_URI,
                "name": self.benchmark.id,
                "reproducible": True,
                "oracle": "host_memory",
                "inputs": [
                    _array_summary(name, value) for name, value in inputs.items()
                ],
                "outputs": [
                    _array_summary(name, value) for name, value in references.items()
                ],
                "origin": {
                    "kind": "agent_evaluation",
                    "benchmark": self.benchmark.id,
                    "region": "RoPE@agent-eval",
                },
            },
            next_actions=(
                {
                    "operation": "run_engine",
                    "testcase": VISIBLE_URI,
                    "adapter": CANDIDATE_URI,
                    "runs_root": RUNS_URI,
                    "reason": "Run the isolated candidate and inspect the first divergence.",
                },
            ),
        )

    def run_visible(self, testcase: str, adapter: str, runs_root: str) -> AgentResponse:
        if not {
            "inferref_capabilities",
            "inferref_context",
        }.issubset(self.successful_tools):
            return AgentResponse.error(
                "run_engine",
                "capabilities and visible context must succeed before engine execution",
                code="required_tool_sequence",
            )
        if (testcase, adapter, runs_root) != (VISIBLE_URI, CANDIDATE_URI, RUNS_URI):
            return AgentResponse.error(
                "run_engine",
                "evaluation run requires the eval:// URIs reported by capabilities",
                code="path_not_allowed",
            )
        if self.engine_runs >= self.benchmark.max_engine_runs:
            return AgentResponse.error(
                "run_engine",
                f"engine run budget exhausted at {self.benchmark.max_engine_runs}",
                code="budget_exhausted",
                data={
                    "engine_runs": self.engine_runs,
                    "max_engine_runs": self.benchmark.max_engine_runs,
                },
            )
        self.engine_runs += 1
        result = execute_case(
            self.benchmark.visible_case,
            engine=self.workspace / self.benchmark.editable_paths[0],
            runs_root=self.root / "visible-runs",
        )
        result["testcase"] = VISIBLE_URI
        result["adapter"] = CANDIDATE_URI
        result["output"] = f"{RUNS_URI}/{result['run_id']}"
        result["input_view"] = f"{VISIBLE_URI}/input-view"
        if isinstance(result.get("comparison"), dict):
            result["comparison"]["reference"] = VISIBLE_URI
            result["comparison"]["actual"] = result["output"]
        status = result["status"]
        if status == "pass":
            return AgentResponse(operation="run_engine", status="pass", data=result)
        if status == "mismatch":
            return AgentResponse(
                operation="run_engine",
                status="fail",
                data=result,
                next_actions=(
                    {
                        "operation": "modify_engine",
                        "reason": "Fix the reported first divergence and rerun.",
                    },
                ),
            )
        return AgentResponse.error(
            "run_engine",
            result.get("execution", {}).get("stderr")
            or "candidate engine did not complete successfully",
            code=status,
            data=result,
        )


def rope_arrays(
    case: EvaluationCase,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(case.seed)
    batch, heads, sequence, dimension = case.shape
    query = rng.normal(size=(batch, heads, sequence, dimension)).astype(np.float32)
    key = rng.normal(size=(batch, heads, sequence, dimension)).astype(np.float32)
    angles = rng.uniform(0.0, 1.25, size=(sequence, dimension // 2)).astype(np.float32)
    cos = np.concatenate((np.cos(angles), np.cos(angles)), axis=-1).astype(np.float32)
    sin = np.concatenate((np.sin(angles), np.sin(angles)), axis=-1).astype(np.float32)
    inputs = {"query": query, "key": key, "cos": cos, "sin": sin}
    references = {
        "q_embed": _apply_rope(query, cos, sin),
        "k_embed": _apply_rope(key, cos, sin),
    }
    return inputs, references


def execute_case(
    case: EvaluationCase,
    *,
    engine: Path,
    runs_root: Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Execute one candidate against input-only files and in-memory references."""

    engine = engine.resolve()
    runs_root = runs_root.resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{case.id}-{uuid.uuid4().hex[:10]}"
    run_root = runs_root / run_id
    input_root = run_root / "testcase"
    output_root = run_root / "output"
    input_root.mkdir(parents=True)
    output_root.mkdir()
    inputs, references = rope_arrays(case)
    _write_input_view(input_root, case, inputs, references)
    initial_input_hashes = workspace_hashes(input_root)

    command = [
        str(Path(sys.executable).absolute()),
        str(engine),
        str(input_root),
        "--output",
        str(output_root),
    ]
    stdout_capture = _StreamCapture(run_root / "stdout.log", 16_384)
    stderr_capture = _StreamCapture(run_root / "stderr.log", 16_384)
    process: subprocess.Popen[bytes] | None = None
    windows_job: int | None = None
    started = time.perf_counter()
    execution: dict[str, Any] = {"timeout_seconds": timeout_seconds}
    try:
        options: dict[str, Any] = {}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        process = subprocess.Popen(
            command,
            cwd=engine.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            **options,
        )
        if os.name == "nt":
            windows_job = _assign_windows_kill_job(process)
        stdout_capture.start(process.stdout)
        stderr_capture.start(process.stderr)
        process_status, scan = _wait_with_limits(
            process,
            deadline=time.monotonic() + timeout_seconds,
            output_path=output_root,
            max_artifact_bytes=64 * 1024 * 1024,
            max_artifact_files=256,
            captures=(stdout_capture, stderr_capture),
            windows_job=windows_job,
        )
        windows_job = None
        stdout_capture.join()
        stderr_capture.join()
        execution.update(
            {
                "status": process_status,
                "exit_code": process.returncode,
                "stdout": stdout_capture.text(),
                "stderr": stderr_capture.text(),
                "stdout_bytes": stdout_capture.observed_bytes,
                "stderr_bytes": stderr_capture.observed_bytes,
                "artifact_bytes": scan.total_bytes,
                "artifact_files": scan.files,
            }
        )
    except OSError as exc:
        if windows_job is not None:
            _close_windows_job(windows_job)
        if process is not None and process.poll() is None:
            _terminate_process_tree(process)
        execution.update({"status": "error", "exit_code": None, "stderr": str(exc)})
    execution["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
    final_input_hashes = workspace_hashes(input_root)
    input_changes = sorted(
        name
        for name in set(initial_input_hashes) | set(final_input_hashes)
        if initial_input_hashes.get(name) != final_input_hashes.get(name)
    )
    result: dict[str, Any] = {
        "run_id": run_id,
        "case": case.to_dict(),
        "execution": execution,
        "comparison": None,
        "input_view": str(input_root),
        "output": str(output_root),
        "integrity": {
            "input_changes": input_changes,
            "input_hashes": {
                name: {
                    "before": _entry_dict(initial_input_hashes.get(name)),
                    "after": _entry_dict(final_input_hashes.get(name)),
                }
                for name in sorted(set(initial_input_hashes) | set(final_input_hashes))
            },
        },
    }
    if input_changes:
        result["status"] = "integrity_failure"
    elif execution["status"] != "completed":
        result["status"] = execution["status"]
    elif execution["exit_code"] != 0:
        result["status"] = "execution_error"
    else:
        report = _compare_memory_references(output_root, references)
        result["comparison"] = report.to_dict()
        result["status"] = "pass" if report.status == "pass" else "mismatch"
    (run_root / "evaluation-run.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def prepare_workspace(benchmark: EvaluationBenchmark, destination: Path) -> Path:
    destination = destination.resolve()
    if destination.exists() and (
        not destination.is_dir() or any(destination.iterdir())
    ):
        raise FileExistsError(f"evaluation workspace is not empty: {destination}")
    destination.mkdir(parents=True)
    for relative in benchmark.workspace_template:
        source = resolve_contained_path(
            benchmark.directory, relative, kind="workspace template"
        )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def workspace_hashes(root: Path) -> dict[str, WorkspaceEntry]:
    """Snapshot every entry below *root* without following links/reparse points."""

    root = root.resolve(strict=True)
    entries: dict[str, WorkspaceEntry] = {}

    def walk(directory: Path, prefix: Path) -> None:
        with os.scandir(directory) as stream:
            children = sorted(stream, key=lambda item: item.name)
        for child in children:
            relative = prefix / child.name
            name = relative.as_posix()
            status = child.stat(follow_symlinks=False)
            mode = stat.S_IMODE(status.st_mode)
            reparse = _is_reparse_status(status)
            if child.is_symlink() or reparse:
                try:
                    target = os.readlink(child.path)
                except OSError:
                    target = None
                entries[name] = WorkspaceEntry(
                    kind="symlink",
                    size=status.st_size,
                    mode=mode,
                    link_target=target,
                )
            elif stat.S_ISREG(status.st_mode):
                payload = _read_regular_file(Path(child.path))
                entries[name] = WorkspaceEntry(
                    kind="file",
                    size=len(payload),
                    mode=mode,
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            elif stat.S_ISDIR(status.st_mode):
                entries[name] = WorkspaceEntry(kind="directory", size=None, mode=mode)
                walk(Path(child.path), relative)
            else:
                entries[name] = WorkspaceEntry(
                    kind="other", size=status.st_size, mode=mode
                )

    walk(root, Path())
    return entries


def engine_patch(template: Path, candidate: Path) -> str:
    before = _read_regular_file(template).decode("utf-8").splitlines(keepends=True)
    after = _read_regular_file(candidate).decode("utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before, after, fromfile="engine.py (baseline)", tofile="engine.py"
        )
    )


def load_audit(path: Path) -> AuditLog:
    """Load a sealed audit JSONL stream; any malformed evidence fails closed."""

    try:
        payload = _read_regular_file(path)
    except OSError as exc:
        return AuditLog(error=f"audit is unavailable: {exc}")
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        return AuditLog(error=f"audit is not UTF-8: {exc}")

    raw_lines = payload.splitlines(keepends=True)
    parsed: list[tuple[dict[str, Any], bytes]] = []
    for line_number, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return AuditLog(error=f"audit line {line_number} is invalid JSON: {exc}")
        if not isinstance(item, dict):
            return AuditLog(error=f"audit line {line_number} must be an object")
        parsed.append((item, raw_line))
    if not parsed:
        return AuditLog(error="audit contains no records")

    footer, _ = parsed[-1]
    if footer.get("type") != "session_footer":
        return AuditLog(error="audit is missing its terminal session footer")
    records_with_bytes = parsed[:-1]
    records = tuple(item for item, _ in records_with_bytes)
    expected_count = footer.get("record_count")
    if (
        not isinstance(expected_count, int)
        or isinstance(expected_count, bool)
        or expected_count != len(records)
    ):
        return AuditLog(records=records, error="audit footer record count mismatch")
    expected_digest = footer.get("records_sha256")
    digest = hashlib.sha256()
    for _, raw_line in records_with_bytes:
        digest.update(raw_line)
    if not isinstance(expected_digest, str) or expected_digest != digest.hexdigest():
        return AuditLog(records=records, error="audit footer digest mismatch")

    previous_runs = 0
    capabilities_succeeded = False
    context_succeeded = False
    for index, record in enumerate(records, start=1):
        error = _validate_audit_record(
            record,
            index=index,
            previous_runs=previous_runs,
            capabilities_succeeded=capabilities_succeeded,
            context_succeeded=context_succeeded,
        )
        if error is not None:
            return AuditLog(records=records, error=error, footer=footer)
        previous_runs = record["engine_runs"]
        if record["tool"] == "inferref_capabilities" and record["status"] == "ok":
            capabilities_succeeded = True
        elif record["tool"] == "inferref_context" and record["status"] == "ok":
            context_succeeded = True
    return AuditLog(records=records, valid=True, footer=footer)


def _audit_line(record: dict[str, Any]) -> bytes:
    return (
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _validate_audit_record(
    record: dict[str, Any],
    *,
    index: int,
    previous_runs: int,
    capabilities_succeeded: bool,
    context_succeeded: bool,
) -> str | None:
    if record.get("type") != "tool_call":
        return f"audit record {index} has an invalid type"
    if record.get("call_index") != index:
        return f"audit call_index is not continuous at record {index}"
    engine_runs = record.get("engine_runs")
    if (
        not isinstance(engine_runs, int)
        or isinstance(engine_runs, bool)
        or engine_runs < previous_runs
        or engine_runs > previous_runs + 1
    ):
        return f"audit engine_runs is not monotonic at record {index}"
    for key in ("timestamp", "tool", "status", "operation"):
        if not isinstance(record.get(key), str) or not record[key]:
            return f"audit record {index}.{key} must be a non-empty string"
    codes = record.get("diagnostic_codes")
    if not isinstance(codes, list) or not all(isinstance(code, str) for code in codes):
        return f"audit record {index}.diagnostic_codes must be a string list"

    tool = record["tool"]
    operation = record["operation"]
    status = record["status"]
    run_delta = engine_runs - previous_runs
    operation_by_tool = {
        "inferref_capabilities": "capabilities",
        "inferref_context": "context",
        "inferref_run_engine": "run_engine",
    }
    if tool not in operation_by_tool:
        return f"audit record {index}.tool is not an evaluation tool"
    if operation != operation_by_tool[tool]:
        return f"audit record {index} has an invalid tool/operation pairing"

    blocked_codes = {"required_tool_sequence", "path_not_allowed", "budget_exhausted"}
    blocked = bool(blocked_codes.intersection(codes))
    if tool == "inferref_capabilities":
        if status != "ok" or codes or run_delta != 0:
            return f"audit record {index} has invalid capabilities semantics"
    elif tool == "inferref_context":
        if run_delta != 0:
            return f"audit record {index} changes engine_runs outside run_engine"
        if status == "ok":
            if not capabilities_succeeded or codes:
                return f"audit record {index} has invalid successful context semantics"
        elif status == "error":
            expected_codes = (
                ["path_not_allowed"]
                if capabilities_succeeded
                else ["required_tool_sequence"]
            )
            if codes != expected_codes:
                return f"audit record {index} has invalid rejected context semantics"
        else:
            return f"audit record {index} has invalid context status"
    else:
        prerequisites_met = capabilities_succeeded and context_succeeded
        if run_delta == 1:
            if (
                not prerequisites_met
                or blocked
                or status not in {"pass", "fail", "error"}
            ):
                return f"audit record {index} has invalid executed run semantics"
        elif run_delta == 0:
            expected_codes = (
                {"path_not_allowed", "budget_exhausted"}
                if prerequisites_met
                else {"required_tool_sequence"}
            )
            if status != "error" or len(codes) != 1 or codes[0] not in expected_codes:
                return f"audit record {index} has invalid rejected run semantics"
        else:  # guarded by the monotonicity check, kept explicit for the FSM.
            return f"audit record {index} has an invalid engine run transition"
    return None


def _is_reparse_status(status: os.stat_result) -> bool:
    attributes = getattr(status, "st_file_attributes", 0)
    return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _read_regular_file(path: Path) -> bytes:
    """Read one regular file while rejecting a final symlink/reparse point."""

    before = os.lstat(path)
    if not stat.S_ISREG(before.st_mode) or _is_reparse_status(before):
        raise OSError(f"not a regular no-follow file: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _is_reparse_status(opened):
            raise OSError(f"not a regular no-follow file: {path.name}")
        if (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino):
            raise OSError(f"file changed while opening: {path.name}")
        chunks = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if (opened.st_dev, opened.st_ino) != (after.st_dev, after.st_ino):
        raise OSError(f"file changed while reading: {path.name}")
    return b"".join(chunks)


def _entry_dict(entry: WorkspaceEntry | None) -> dict[str, Any] | None:
    return entry.to_dict() if entry is not None else None


def _write_input_view(
    root: Path,
    case: EvaluationCase,
    inputs: dict[str, np.ndarray],
    references: dict[str, np.ndarray],
) -> None:
    manifest_inputs = []
    for value_id, (name, value) in enumerate(inputs.items(), start=1):
        relative = f"inputs/{name}.irtensor"
        metadata = codec.read(codec.write_array(root / relative, value)).to_metadata()
        manifest_inputs.append(
            {"name": name, "value_id": value_id, "payload": relative, **metadata}
        )
    manifest = {
        "format": "inferref-evaluation-input-view",
        "format_version": "0.1",
        "name": case.id,
        "inputs": manifest_inputs,
        "outputs": [
            {"name": name, **_array_metadata(value)}
            for name, value in references.items()
        ],
        "oracle": "not_materialized",
    }
    (root / "testcase.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def _compare_memory_references(
    output_root: Path, references: dict[str, np.ndarray]
) -> ComparisonReport:
    report = ComparisonReport(reference="host-memory", actual=str(output_root))
    output_map: dict[str, str] = {}
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            raw_outputs = (
                manifest.get("outputs", []) if isinstance(manifest, dict) else []
            )
            if isinstance(raw_outputs, list):
                for entry in raw_outputs:
                    if (
                        isinstance(entry, dict)
                        and isinstance(entry.get("name"), str)
                        and isinstance(entry.get("payload"), str)
                    ):
                        output_map[entry["name"]] = entry["payload"]
        except (OSError, json.JSONDecodeError):
            pass
    policy = TolerancePolicy()
    for name, reference in references.items():
        relative = output_map.get(name, f"{name}.irtensor")
        try:
            actual_path = resolve_contained_path(
                output_root, relative, kind="engine payload"
            )
        except ValueError as exc:
            comparison = TensorComparison(
                name=name, status=STATUS_ERROR, message=str(exc)
            )
        else:
            if not actual_path.is_file():
                comparison = TensorComparison(
                    name=name,
                    status=STATUS_MISSING,
                    message=f"engine output is missing {name!r}",
                )
            else:
                try:
                    actual = codec.read(actual_path)
                    expected = TensorView(
                        dtype="float32",
                        shape=tuple(reference.shape),
                        stride=tuple(codec.contiguous_stride(reference.shape)),
                        storage_offset=0,
                        data=np.ascontiguousarray(reference),
                    )
                    comparison = compare_tensors(
                        name, expected, actual, policy=policy, strict_layout=False
                    )
                except (OSError, ValueError) as exc:
                    comparison = TensorComparison(
                        name=name, status=STATUS_ERROR, message=str(exc)
                    )
        comparison.operator = "aten.add.Tensor"
        comparison.module_path = "layers.0.self_attn"
        comparison.source = "reference_model.py:42 in apply_rotary_pos_emb"
        comparison.region = "RoPE@agent-eval"
        report.comparisons.append(comparison)
        if not comparison.passed:
            report.stopped_early = True
            break
    return report


def _rotate_half(value: np.ndarray) -> np.ndarray:
    half = value.shape[-1] // 2
    return np.concatenate((-value[..., half:], value[..., :half]), axis=-1)


def _apply_rope(value: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    return value * cos[None, None, :, :] + _rotate_half(value) * sin[None, None, :, :]


def _array_metadata(value: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": list(value.shape),
        "stride": list(codec.contiguous_stride(value.shape)),
        "storage_offset": 0,
        "numel": int(value.size),
        "nbytes": int(value.nbytes),
    }


def _array_summary(name: str, value: np.ndarray) -> dict[str, Any]:
    return {"name": name, **_array_metadata(value), "payload": None}


def _required_string(data: dict[str, Any], name: str) -> str:
    value = data.get(name)
    if not isinstance(value, str) or not value.strip():
        raise AgentProtocolError(f"{name} must be a non-empty string")
    return value.strip()


def _string_array(value: Any, where: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise AgentProtocolError(f"{where} must be a non-empty string array")
    if len(set(value)) != len(value):
        raise AgentProtocolError(f"{where} entries must be unique")
    return tuple(value)


def _bounded_integer(
    data: dict[str, Any], name: str, *, minimum: int, maximum: int
) -> int:
    value = data.get(name)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise AgentProtocolError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _required_bool(data: dict[str, Any], name: str, *, where: str) -> bool:
    value = data.get(name)
    if not isinstance(value, bool):
        raise AgentProtocolError(f"{where}.{name} must be a boolean")
    return value
