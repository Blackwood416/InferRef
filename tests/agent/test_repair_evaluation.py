"""Blind Agent evaluation host, oracle-isolation and MCP proxy tests."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

mcp = pytest.importorskip("mcp")

from mcp import Client

from inferref.agent import evaluation_host, evaluation_worker
from inferref.agent.evaluation import (
    CANDIDATE_URI,
    RUNS_URI,
    VISIBLE_URI,
    EvaluationBenchmark,
    EvaluationSession,
    _audit_line,
    execute_case,
    load_audit,
    prepare_workspace,
    workspace_hashes,
)
from inferref.agent.evaluation_host import (
    AgentProcessResult,
    assess_candidate,
    evaluate_benchmark,
    parse_agent_usage,
)
from inferref.agent.evaluation_mcp import create_evaluation_server
from inferref.agent.protocol import AgentProtocolError
from inferref.ir.version import INFERREF_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_PATH = REPO_ROOT / "examples" / "agent_eval" / "rope_sign" / "benchmark.json"
WRONG = "return np.concatenate((second, -first), axis=-1)"
FIXED = "return np.concatenate((-second, first), axis=-1)"


def _benchmark() -> EvaluationBenchmark:
    return EvaluationBenchmark.load(BENCHMARK_PATH)


def _process(
    root: Path,
    *,
    exit_code: int = 0,
    timed_out: bool = False,
    transcript_error: str | None = None,
) -> AgentProcessResult:
    stdout = root / "stdout.jsonl"
    stderr = root / "stderr.log"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return AgentProcessResult(
        command=("fake-agent",),
        exit_code=exit_code,
        timed_out=timed_out,
        duration_ms=1.0,
        stdout_path=stdout,
        stderr_path=stderr,
        stdout="",
        stderr="",
        usage={},
        transcript_error=transcript_error,
    )


def _exercise_required_tools(session: EvaluationSession, *, repair: bool) -> None:
    response = session.capabilities()
    session.audit("inferref_capabilities", response)
    response = session.context(VISIBLE_URI)
    session.audit("inferref_context", response)
    first = session.run_visible(VISIBLE_URI, CANDIDATE_URI, RUNS_URI)
    session.audit("inferref_run_engine", first)
    if repair:
        engine = session.workspace / "engine.py"
        source = engine.read_text(encoding="utf-8")
        engine.write_text(source.replace(WRONG, FIXED), encoding="utf-8")
        second = session.run_visible(VISIBLE_URI, CANDIDATE_URI, RUNS_URI)
        session.audit("inferref_run_engine", second)


def _fake_repair_driver(
    agent: str,
    model: str,
    benchmark: EvaluationBenchmark,
    workspace: Path,
    session_root: Path,
    audit_path: Path,
) -> AgentProcessResult:
    del agent, model
    session = EvaluationSession(benchmark, workspace, session_root, audit_path)
    _exercise_required_tools(session, repair=True)
    session.finalize_audit()
    return _process(session_root.parent)


def _valid_runner_evidence(agent: str) -> dict[str, object]:
    components = [
        {
            "role": "cli_executable",
            "name": f"{agent}.exe",
            "size": 123,
            "sha256": "a" * 64,
            "error": None,
        }
    ]
    policy = {"kind": agent, "policy_marker": "fixture"}
    return {
        "kind": agent,
        "command_components": {
            "before": components,
            "after_version": components,
            "after": components,
        },
        "components_unchanged": True,
        "argv_instance_sha256": "b" * 64,
        "argv_policy": policy,
        "argv_policy_sha256": evaluation_host._canonical_json_sha256(policy),
        "version_output": f"{agent} test-version",
        "external_files": None,
    }


def _valid_worker_evidence() -> dict[str, object]:
    policy = {
        "flags": ["-I", "-m"],
        "entry_module": "inferref.agent.evaluation_worker",
        "request_transport": "regular-file-json",
        "request_schema_version": "0.1",
        "request_sha256": "c" * 64,
    }
    return {
        "python_isolated": True,
        "entry_module": "inferref.agent.evaluation_worker",
        "launch_policy": policy,
        "launch_policy_sha256": evaluation_host._canonical_json_sha256(policy),
        "python_executable": {
            "name": "python.exe",
            "size": 103192,
            "sha256": "d" * 64,
        },
    }


def _runner_with_component(**changes: object) -> dict[str, object]:
    evidence = _valid_runner_evidence("codex")
    component = {**evidence["command_components"]["before"][0], **changes}
    snapshot = {
        "before": [component],
        "after_version": [component],
        "after": [component],
    }
    return {**evidence, "command_components": snapshot}


def _mismatched_model_driver(
    agent: str,
    model: str,
    benchmark: EvaluationBenchmark,
    workspace: Path,
    session_root: Path,
    audit_path: Path,
    **kwargs: object,
) -> AgentProcessResult:
    del kwargs
    process = _fake_builtin_driver(
        agent, model, benchmark, workspace, session_root, audit_path
    )
    return replace(
        process,
        model_evidence={
            "requested": model,
            "reported": "different-model",
            "evidence_source": "cli_system_init_event",
            "matches_request": False,
            "identity_level": "cli_self_reported",
            "provider_verified": False,
            "agent": agent,
        },
    )


def _fake_builtin_driver(
    agent: str,
    model: str,
    benchmark: EvaluationBenchmark,
    workspace: Path,
    session_root: Path,
    audit_path: Path,
    **kwargs: object,
) -> AgentProcessResult:
    del kwargs
    process = _fake_repair_driver(
        agent, model, benchmark, workspace, session_root, audit_path
    )
    return replace(
        process,
        cli_version=f"{agent} test-version",
        runner_evidence=_valid_runner_evidence(agent),
        model_evidence={
            "requested": model,
            "reported": None,
            "evidence_source": "unavailable",
            "matches_request": None,
            "identity_level": "requested_only",
            "provider_verified": False,
            "agent": agent,
        },
    )


def test_v02_contract_and_workspace_hide_host_assets(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")

    assert benchmark.max_engine_runs == 4
    assert benchmark.drivers["codex"].model == "gpt-5.6-sol"
    assert benchmark.drivers["claude"].model == "opus"
    assert benchmark.success.required_agent_passes == 2
    assert benchmark.success.visible_status == "pass"
    assert benchmark.success.all_holdouts_pass is True
    assert benchmark.success.protected_paths_unchanged is True
    assert {path.name for path in workspace.iterdir()} == {"engine.py", "TASK.md"}
    assert "reference" not in " ".join(path.as_posix() for path in workspace.rglob("*"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_agent_passes", 0),
        ("visible_status", "fail"),
        ("all_holdouts_pass", "yes"),
        ("protected_paths_unchanged", None),
    ],
)
def test_success_policy_rejects_malformed_values(
    tmp_path: Path, field: str, value: object
) -> None:
    manifest = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    manifest["success"][field] = value
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(AgentProtocolError):
        EvaluationBenchmark.load(path)


def test_candidate_input_view_never_materializes_reference_payloads(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    outcome = execute_case(
        benchmark.visible_case,
        engine=workspace / "engine.py",
        runs_root=tmp_path / "runs",
    )

    input_view = Path(outcome["input_view"])
    manifest = json.loads((input_view / "testcase.json").read_text(encoding="utf-8"))
    assert manifest["oracle"] == "not_materialized"
    assert not (input_view / "reference").exists()
    assert all("payload" not in entry for entry in manifest["outputs"])
    assert outcome["status"] == "mismatch"


def test_candidate_cannot_modify_input_only_staging(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    engine = workspace / "engine.py"
    source = engine.read_text(encoding="utf-8").replace(WRONG, FIXED)
    source = source.replace(
        "results = {",
        '(testcase / "testcase.json").write_text("tampered", encoding="utf-8")\n'
        "    results = {",
    )
    engine.write_text(source, encoding="utf-8")

    outcome = execute_case(
        benchmark.visible_case,
        engine=engine,
        runs_root=tmp_path / "runs",
    )

    assert outcome["status"] == "integrity_failure"
    assert outcome["integrity"]["input_changes"] == ["testcase.json"]


def test_evaluation_mcp_enforces_opaque_uris_and_run_budget(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    session = EvaluationSession(
        benchmark, workspace, tmp_path / "session", tmp_path / "audit.jsonl"
    )

    async def exercise() -> None:
        async with Client(create_evaluation_server(session)) as client:
            capabilities = await client.call_tool("inferref_capabilities", {})
            assert capabilities.structured_content["status"] == "ok"
            assert (
                capabilities.structured_content["data"]["evaluation"][
                    "visible_testcase"
                ]
                == VISIBLE_URI
            )
            out_of_order = await client.call_tool(
                "inferref_run_engine",
                {
                    "testcase": VISIBLE_URI,
                    "adapter": CANDIDATE_URI,
                    "runs_root": RUNS_URI,
                },
            )
            assert (
                out_of_order.structured_content["diagnostics"][0]["code"]
                == "required_tool_sequence"
            )
            denied = await client.call_tool("inferref_context", {"path": "C:/oracle"})
            assert (
                denied.structured_content["diagnostics"][0]["code"]
                == "path_not_allowed"
            )
            context = await client.call_tool("inferref_context", {"path": VISIBLE_URI})
            assert context.structured_content["data"]["oracle"] == "host_memory"
            for _ in range(benchmark.max_engine_runs):
                await client.call_tool(
                    "inferref_run_engine",
                    {
                        "testcase": VISIBLE_URI,
                        "adapter": CANDIDATE_URI,
                        "runs_root": RUNS_URI,
                    },
                )
            exhausted = await client.call_tool(
                "inferref_run_engine",
                {
                    "testcase": VISIBLE_URI,
                    "adapter": CANDIDATE_URI,
                    "runs_root": RUNS_URI,
                },
            )
            assert exhausted.structured_content["status"] == "error"
            assert (
                exhausted.structured_content["diagnostics"][0]["code"]
                == "budget_exhausted"
            )

    asyncio.run(exercise())
    session.finalize_audit()
    records = load_audit(tmp_path / "audit.jsonl")
    assert records.valid
    assert records[-1]["diagnostic_codes"] == ["budget_exhausted"]


def test_zero_call_probe_does_not_reserve_the_audit_path(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    audit_path = tmp_path / "audit.jsonl"

    probe = EvaluationSession(benchmark, workspace, tmp_path / "probe", audit_path)
    probe.finalize_audit()
    assert not audit_path.exists()

    real = EvaluationSession(benchmark, workspace, tmp_path / "real", audit_path)
    response = real.capabilities()
    real.audit("inferref_capabilities", response)
    assert load_audit(audit_path).valid
    real.finalize_audit()

    assert load_audit(audit_path).valid


def _write_sealed_audit(path: Path, records: list[dict[str, object]]) -> None:
    encoded = b"".join(_audit_line(record) for record in records)
    footer = {
        "type": "session_footer",
        "record_count": len(records),
        "records_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    path.write_bytes(encoded + _audit_line(footer))


def _audit_record(index: int, engine_runs: int = 0) -> dict[str, object]:
    return {
        "type": "tool_call",
        "timestamp": "2026-08-03T00:00:00+00:00",
        "tool": "inferref_capabilities",
        "call_index": index,
        "engine_runs": engine_runs,
        "status": "ok",
        "operation": "capabilities",
        "diagnostic_codes": [],
    }


@pytest.mark.parametrize(
    "corruption",
    ["invalid_json", "non_object", "missing_footer", "wrong_count", "wrong_digest"],
)
def test_audit_corruption_fails_closed(tmp_path: Path, corruption: str) -> None:
    path = tmp_path / "audit.jsonl"
    _write_sealed_audit(path, [_audit_record(1)])
    lines = path.read_bytes().splitlines(keepends=True)
    if corruption == "invalid_json":
        path.write_bytes(path.read_bytes() + b"{broken\n")
    elif corruption == "non_object":
        path.write_bytes(b"[]\n" + b"".join(lines[1:]))
    elif corruption == "missing_footer":
        path.write_bytes(lines[0])
    else:
        footer = json.loads(lines[-1])
        if corruption == "wrong_count":
            footer["record_count"] = 2
        else:
            footer["records_sha256"] = "0" * 64
        path.write_bytes(b"".join(lines[:-1]) + _audit_line(footer))

    audit = load_audit(path)

    assert not audit.valid
    assert audit.error


@pytest.mark.parametrize(
    "records",
    [
        [_audit_record(2)],
        [_audit_record(1, 1), _audit_record(2, 0)],
        [{**_audit_record(1), "diagnostic_codes": "not-a-list"}],
    ],
)
def test_audit_schema_and_monotonicity_fail_closed(
    tmp_path: Path, records: list[dict[str, object]]
) -> None:
    path = tmp_path / "audit.jsonl"
    _write_sealed_audit(path, records)

    audit = load_audit(path)

    assert not audit.valid
    assert audit.error


@pytest.mark.parametrize(
    "records",
    [
        [{**_audit_record(1), "operation": "context"}],
        [{**_audit_record(1), "tool": "unknown_tool"}],
        [{**_audit_record(1), "status": "pass"}],
        [
            _audit_record(1),
            {**_audit_record(2, 1), "tool": "inferref_context", "operation": "context"},
        ],
        [
            _audit_record(1),
            {
                **_audit_record(2),
                "tool": "inferref_run_engine",
                "operation": "run_engine",
                "status": "pass",
            },
        ],
        [
            {
                **_audit_record(1),
                "tool": "inferref_context",
                "operation": "context",
                "status": "error",
                "diagnostic_codes": ["budget_exhausted"],
            }
        ],
        [
            {
                **_audit_record(1),
                "tool": "inferref_run_engine",
                "operation": "run_engine",
                "status": "error",
                "diagnostic_codes": ["budget_exhausted"],
            }
        ],
    ],
)
def test_audit_semantic_state_machine_fails_closed(
    tmp_path: Path, records: list[dict[str, object]]
) -> None:
    path = tmp_path / "audit.jsonl"
    _write_sealed_audit(path, records)

    audit = load_audit(path)

    assert not audit.valid
    assert audit.error


def test_rejected_out_of_order_run_is_valid_audit_evidence(tmp_path: Path) -> None:
    records = [
        {
            **_audit_record(1),
            "tool": "inferref_run_engine",
            "operation": "run_engine",
            "status": "error",
            "diagnostic_codes": ["required_tool_sequence"],
        }
    ]
    path = tmp_path / "audit.jsonl"
    _write_sealed_audit(path, records)

    audit = load_audit(path)

    assert audit.valid
    assert audit.records[0]["engine_runs"] == 0


def test_torn_audit_after_required_tools_is_infrastructure_failure(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    initial, audit_path = _passing_session(benchmark, workspace, tmp_path)
    with audit_path.open("ab") as stream:
        stream.write(b"{torn\n")

    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(audit_path),
    )

    assert result["classification"] == "infrastructure_failure"
    assert result["audit_integrity"]["valid"] is False


def test_fake_dual_agent_evaluation_passes_visible_and_holdouts(tmp_path: Path) -> None:
    report = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex", "claude"),
        report_dir=tmp_path / "report",
        driver=_fake_repair_driver,
    )

    assert report["status"] == "pass"
    assert report["acceptance"]["passed"] == 2
    for agent in report["agents"]:
        assert agent["classification"] == "pass"
        assert agent["visible"]["runs"] == 2
        assert agent["visible"]["historical_passed"] is True
        assert agent["visible"]["final"]["status"] == "pass"
        assert agent["visible"]["passed"] is True
        assert agent["final_engine_sha256"]
        assert len(agent["holdouts"]) == 3
        assert all(case["status"] == "pass" for case in agent["holdouts"])


def test_final_engine_is_revalidated_after_historical_visible_pass(
    tmp_path: Path,
) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    initial = workspace_hashes(workspace)
    audit_path = tmp_path / "audit.jsonl"
    session = EvaluationSession(benchmark, workspace, tmp_path / "session", audit_path)
    _exercise_required_tools(session, repair=True)
    session.finalize_audit()

    engine = workspace / "engine.py"
    source = engine.read_text(encoding="utf-8")
    source = source.replace(
        FIXED,
        "if value.shape == (1, 2, 4, 8):\n"
        "        return np.concatenate((second, -first), axis=-1)\n"
        "    " + FIXED,
    )
    engine.write_text(source, encoding="utf-8")

    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(audit_path),
    )

    assert result["visible"]["historical_passed"] is True
    assert result["visible"]["final"]["status"] == "mismatch"
    assert result["visible"]["passed"] is False
    assert all(case["status"] == "pass" for case in result["holdouts"])
    assert result["classification"] == "agent_failure"


def _passing_session(
    benchmark: EvaluationBenchmark, workspace: Path, root: Path
) -> tuple[dict[str, object], Path]:
    initial = workspace_hashes(workspace)
    audit_path = root / "audit.jsonl"
    session = EvaluationSession(benchmark, workspace, root / "session", audit_path)
    _exercise_required_tools(session, repair=True)
    session.finalize_audit()
    return initial, audit_path


def _symlink_or_skip(target: Path, link: Path, *, directory: bool = False) -> None:
    try:
        os.symlink(target, link, target_is_directory=directory)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")


@pytest.mark.parametrize("protected_name", ["TASK.md"])
def test_same_content_symlink_is_an_integrity_change(
    tmp_path: Path, protected_name: str
) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    initial, audit_path = _passing_session(benchmark, workspace, tmp_path)
    protected = workspace / protected_name
    external = tmp_path / "same-content.txt"
    external.write_bytes(protected.read_bytes())
    protected.unlink()
    _symlink_or_skip(external, protected)

    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(audit_path),
    )

    assert result["classification"] == "integrity_failure"
    assert protected_name in result["integrity"]["protected_changes"]
    assert (
        result["integrity"]["protected_file_hashes"][protected_name]["after"]["kind"]
        == "symlink"
    )


def test_editable_engine_symlink_is_rejected(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    initial, audit_path = _passing_session(benchmark, workspace, tmp_path)
    engine = workspace / "engine.py"
    external = tmp_path / "engine.py"
    external.write_bytes(engine.read_bytes())
    engine.unlink()
    _symlink_or_skip(external, engine)

    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(audit_path),
    )

    assert result["classification"] == "integrity_failure"
    assert "regular no-follow file" in result["message"]


def test_new_empty_directory_is_an_integrity_change(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    initial, audit_path = _passing_session(benchmark, workspace, tmp_path)
    (workspace / "extra-empty-directory").mkdir()

    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(audit_path),
    )

    assert result["classification"] == "integrity_failure"
    assert "extra-empty-directory" in result["integrity"]["protected_changes"]


@pytest.mark.skipif(os.name != "nt", reason="Windows junction coverage")
def test_workspace_manifest_rejects_windows_junction(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    target = tmp_path / "target"
    workspace.mkdir()
    target.mkdir()
    junction = workspace / "junction"
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(target)],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"junction creation is unavailable: {completed.stderr}")
    try:
        assert workspace_hashes(workspace)["junction"].kind == "symlink"
    finally:
        os.rmdir(junction)


def test_success_policy_controls_required_agent_passes(tmp_path: Path) -> None:
    manifest = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    manifest["success"]["required_agent_passes"] = 1
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(manifest), encoding="utf-8")
    for name in ("engine.py", "TASK.md"):
        (tmp_path / name).write_bytes((BENCHMARK_PATH.parent / name).read_bytes())

    report = evaluate_benchmark(
        benchmark_path,
        agents=("codex",),
        report_dir=tmp_path / "report",
        driver=_fake_repair_driver,
    )

    assert report["status"] == "pass"
    assert report["acceptance"]["required_passes"] == 1
    assert report["acceptance"]["policy"] == manifest["success"]


def test_development_attestation_omits_private_transcript_and_paths(
    tmp_path: Path,
) -> None:
    report_root = tmp_path / "private-report"
    report = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex", "claude"),
        report_dir=report_root,
        driver=_fake_repair_driver,
    )

    attestation = json.loads(
        (report_root / "attestation.json").read_text(encoding="utf-8")
    )
    assert attestation["status"] == report["status"] == "pass"
    assert attestation["benchmark"]["success_policy"] == _benchmark().success.to_dict()
    assert all(agent["engine_patch"] for agent in attestation["agents"])
    assert all(agent["final_engine_sha256"] for agent in attestation["agents"])
    assert all(agent["raw_transcript_sha256"] for agent in attestation["agents"])
    assert attestation["format_version"] == "0.5"
    assert attestation["attestation_level"] == "development"
    assert attestation["runner_mode"] == "custom_driver"
    assert attestation["repository"]["unchanged"] is True
    assert (
        attestation["benchmark"]["loaded_source_sha256"] == _benchmark().source_sha256
    )
    assert attestation["benchmark"]["source_unchanged"] is True
    assert attestation["evaluator"]["source_tree_sha256"]
    assert attestation["evaluator"]["files"]
    assert {
        "inferref/compare/compare.py",
        "inferref/compare/tolerance.py",
        "inferref/tensor/codec.py",
        "inferref/ir/paths.py",
        "inferref/agent/protocol.py",
        "inferref/agent/adapter.py",
    } <= set(attestation["evaluator"]["files"])
    assert attestation["runtime"]["unchanged"] is True
    assert attestation["runtime"]["before"]["python"]
    assert attestation["runtime"]["before"]["numpy"]
    assert attestation["runtime"]["before"]["inferref"] == INFERREF_VERSION
    assert attestation["runtime"]["before"]["byteorder"] in {"little", "big"}
    assert attestation["runtime"]["before"]["import"]["mode"] == "repository_source"
    assert attestation["report_json_sha256"]
    assert attestation["external_files"] is None
    assert attestation["acceptance"]["agent_model_request_satisfied"] is None
    assert attestation["agents"][0]["model_request_satisfied"] is None
    rendered = json.dumps(attestation)
    assert str(tmp_path) not in rendered
    assert '"prompt"' not in rendered
    assert '"stdout"' not in rendered
    assert "thinking" not in rendered


def test_formal_public_attestation_rejects_custom_driver(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="isolated built-in Agent worker"):
        evaluate_benchmark(
            BENCHMARK_PATH,
            agents=("codex",),
            report_dir=tmp_path / "report",
            driver=_fake_repair_driver,
            public_attestation=tmp_path / "public.json",
        )

    assert not (tmp_path / "report").exists()


def test_public_attestation_delegates_to_isolated_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = {"status": "pass", "source": "worker"}
    captured: dict[str, object] = {}

    def fake_worker(benchmark_path: str | Path, **kwargs: object) -> dict[str, object]:
        captured["benchmark"] = benchmark_path
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(evaluation_host, "_run_formal_worker", fake_worker)
    monkeypatch.setattr(
        evaluation_host,
        "run_agent_cli",
        lambda *args, **kwargs: pytest.fail("parent run_agent_cli must not execute"),
    )

    result = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex", "claude"),
        report_dir=tmp_path / "report",
        public_attestation=tmp_path / "public.json",
    )

    assert result is sentinel
    assert captured["agents"] == ("codex", "claude")
    assert captured["public_attestation"] == tmp_path / "public.json"


def test_formal_worker_launches_isolated_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        captured["kwargs"] = kwargs
        report_root = tmp_path / "report"
        report_root.mkdir()
        request_path = Path(command[command.index("--request") + 1])
        request_sha256 = hashlib.sha256(request_path.read_bytes()).hexdigest()
        (report_root / "report.json").write_text(
            json.dumps(
                {
                    "status": "pass",
                    "worker_evidence": evaluation_worker._worker_evidence(
                        request_sha256
                    ),
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(evaluation_host.subprocess, "run", fake_run)

    report = evaluation_host._run_formal_worker(
        BENCHMARK_PATH,
        agents=("codex",),
        report_dir=tmp_path / "report",
        claude_settings=None,
        claude_model=None,
        public_attestation=tmp_path / "public.json",
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[1:4] == ["-I", "-m", "inferref.agent.evaluation_worker"]
    assert report["status"] == "pass"
    assert report["worker_evidence"]["launch_policy"]["request_sha256"]
    assert report["worker_evidence"]["python_executable"]["sha256"]


def test_formal_worker_entry_refuses_nonisolated_python(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires Python isolated mode"):
        evaluation_worker.run_request(tmp_path / "missing-request.json")


def test_custom_driver_attestation_is_development_evidence(tmp_path: Path) -> None:
    report_root = tmp_path / "report"
    report = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex",),
        report_dir=report_root,
        driver=_fake_repair_driver,
    )

    attestation = json.loads(
        (report_root / "attestation.json").read_text(encoding="utf-8")
    )
    assert report["runner_mode"] == "custom_driver"
    assert attestation["runner_mode"] == "custom_driver"
    assert attestation["attestation_level"] == "development"


def test_formal_attestation_level_requires_worker_and_bound_runner(
    tmp_path: Path,
) -> None:
    report_root = tmp_path / "report"
    report = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex",),
        report_dir=report_root,
        driver=_fake_repair_driver,
    )
    report["agents"][0]["process"]["runner"] = _valid_runner_evidence("codex")
    report["agents"][0]["process"]["model_evidence"] = {
        "requested": "gpt-5.6-sol",
        "reported": None,
        "evidence_source": "unavailable",
        "matches_request": None,
        "identity_level": "requested_only",
        "provider_verified": False,
        "agent": "codex",
    }
    clean = {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "git_available": True,
    }
    runtime = {"python": "test"}
    arguments = {
        "report_root": report_root,
        "report_path": report_root / "report.json",
        "runner_mode": "isolated_builtin_cli_worker",
        "repository_before": clean,
        "repository_after": clean,
        "repository_unchanged": True,
        "evaluator_unchanged": True,
        "benchmark_unchanged": True,
        "runtime_before": runtime,
        "runtime_after": runtime,
        "runtime_unchanged": True,
    }

    development = evaluation_host.build_public_attestation(
        _benchmark(), report, formal_worker_evidence=None, **arguments
    )
    formal = evaluation_host.build_public_attestation(
        _benchmark(),
        report,
        formal_worker_evidence=_valid_worker_evidence(),
        **arguments,
    )

    assert development["attestation_level"] == "development"
    assert formal["attestation_level"] == "formal"
    assert formal["format_version"] == "0.5"
    assert formal["worker"] == _valid_worker_evidence()
    assert formal["agents"][0]["model"]["identity_level"] == "requested_only"
    assert formal["agents"][0]["runner"]["components_unchanged"] is True


def test_runtime_is_rechecked_after_agent_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = iter(({"python": "before"}, {"python": "after"}))
    monkeypatch.setattr(
        evaluation_host, "_runtime_evidence", lambda path: next(evidence)
    )

    report_root = tmp_path / "report"
    report = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex",),
        report_dir=report_root,
        driver=_fake_repair_driver,
    )
    attestation = json.loads(
        (report_root / "attestation.json").read_text(encoding="utf-8")
    )

    assert report["status"] == "fail"
    assert report["host_integrity"]["runtime_unchanged"] is False
    assert attestation["runtime"] == {
        "before": {"python": "before"},
        "after": {"python": "after"},
        "unchanged": False,
    }


def test_formal_public_attestation_rejects_dirty_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluation_host,
        "_repository_evidence",
        lambda path: {
            "commit": "a" * 40,
            "dirty": True,
            "status_sha256": "status",
            "git_diff_sha256": "diff",
            "git_available": True,
        },
    )

    with pytest.raises(RuntimeError, match="clean Git worktree"):
        evaluation_host._evaluate_benchmark_core(
            BENCHMARK_PATH,
            agents=("codex",),
            report_dir=tmp_path / "report",
            public_attestation=tmp_path / "public.json",
            formal_worker_evidence=_valid_worker_evidence(),
        )

    assert not (tmp_path / "report").exists()


def test_formal_attestation_rechecks_repository_after_agents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "git_available": True,
    }
    changed = {**clean, "commit": "b" * 40}
    evidence = iter((clean, changed))
    monkeypatch.setattr(
        evaluation_host, "_repository_evidence", lambda path: next(evidence)
    )
    monkeypatch.setattr(
        evaluation_host,
        "run_agent_cli",
        _fake_builtin_driver,
    )

    with pytest.raises(RuntimeError, match="changed during the run"):
        evaluation_host._evaluate_benchmark_core(
            BENCHMARK_PATH,
            agents=("codex",),
            report_dir=tmp_path / "report",
            public_attestation=tmp_path / "public.json",
            formal_worker_evidence=_valid_worker_evidence(),
        )

    report = json.loads((tmp_path / "report" / "report.json").read_text())
    assert report["host_integrity"]["repository_unchanged"] is False
    assert report["status"] == "fail"
    assert not (tmp_path / "public.json").exists()


def test_loaded_benchmark_hash_is_frozen_and_change_fails_host_integrity(
    tmp_path: Path,
) -> None:
    manifest = BENCHMARK_PATH.read_bytes()
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_bytes(manifest)
    for name in ("engine.py", "TASK.md"):
        (tmp_path / name).write_bytes((BENCHMARK_PATH.parent / name).read_bytes())
    loaded_sha256 = hashlib.sha256(manifest).hexdigest()

    def mutating_driver(
        agent: str,
        model: str,
        benchmark: EvaluationBenchmark,
        workspace: Path,
        session_root: Path,
        audit_path: Path,
    ) -> AgentProcessResult:
        benchmark.source.write_bytes(manifest + b"\n")
        return _fake_repair_driver(
            agent, model, benchmark, workspace, session_root, audit_path
        )

    report_root = tmp_path / "report"
    report = evaluate_benchmark(
        benchmark_path,
        agents=("codex",),
        report_dir=report_root,
        driver=mutating_driver,
    )
    attestation = json.loads(
        (report_root / "attestation.json").read_text(encoding="utf-8")
    )

    assert report["host_integrity"]["benchmark_source_unchanged"] is False
    assert report["status"] == "fail"
    assert attestation["benchmark"]["loaded_source_sha256"] == loaded_sha256
    assert attestation["benchmark"]["source_unchanged"] is False


def test_visible_only_patch_is_classified_as_overfit(tmp_path: Path) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    engine = workspace / "engine.py"
    source = engine.read_text(encoding="utf-8")
    replacement = (
        "if value.shape[-1] == 8:\n"
        "        return np.concatenate((-second, first), axis=-1)\n"
        "    return np.concatenate((second, -first), axis=-1)"
    )
    engine.write_text(source.replace(WRONG, replacement), encoding="utf-8")
    initial = workspace_hashes(workspace)
    # The modified engine is the permitted baseline for this focused classifier test.
    initial["engine.py"] = "baseline-placeholder"
    session = EvaluationSession(
        benchmark, workspace, tmp_path / "session", tmp_path / "audit.jsonl"
    )
    _exercise_required_tools(session, repair=False)
    session.finalize_audit()
    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(tmp_path / "audit.jsonl"),
    )

    assert result["visible"]["passed"] is True
    assert result["classification"] == "overfit_failure"
    assert any(case["status"] != "pass" for case in result["holdouts"])

    relaxed = replace(
        benchmark,
        success=replace(benchmark.success, all_holdouts_pass=False),
    )
    relaxed_result = assess_candidate(
        relaxed,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "relaxed-agent",
        process=_process(tmp_path),
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(tmp_path / "audit.jsonl"),
    )
    assert relaxed_result["classification"] == "pass"


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("missing_tools", "integrity_failure"),
        ("protected_change", "integrity_failure"),
        ("adapter_change", "integrity_failure"),
        ("timeout", "infrastructure_failure"),
        ("malformed_output", "infrastructure_failure"),
        ("malformed_audit", "infrastructure_failure"),
    ],
)
def test_failure_classification_is_structured(
    tmp_path: Path, mode: str, expected: str
) -> None:
    benchmark = _benchmark()
    workspace = prepare_workspace(benchmark, tmp_path / "workspace")
    initial = workspace_hashes(workspace)
    audit_path = tmp_path / "audit.jsonl"
    if mode == "missing_tools":
        session = EvaluationSession(
            benchmark, workspace, tmp_path / "session", audit_path
        )
        response = session.capabilities()
        session.audit("inferref_capabilities", response)
        session.finalize_audit()
    elif mode in {"protected_change", "adapter_change"}:
        session = EvaluationSession(
            benchmark, workspace, tmp_path / "session", audit_path
        )
        _exercise_required_tools(session, repair=True)
        session.finalize_audit()
        if mode == "protected_change":
            (workspace / "TASK.md").write_text("tampered", encoding="utf-8")
    elif mode == "malformed_audit":
        audit_path.write_text("not-json\n", encoding="utf-8")
    process = _process(
        tmp_path,
        timed_out=mode == "timeout",
        transcript_error=(
            "Agent JSON event stream is malformed at line 1"
            if mode == "malformed_output"
            else None
        ),
    )
    result = assess_candidate(
        benchmark,
        agent="fake",
        model="fake",
        workspace=workspace,
        agent_root=tmp_path / "agent",
        process=process,
        initial_hashes=initial,
        final_hashes=workspace_hashes(workspace),
        audit=load_audit(audit_path),
        initial_host_hashes={"host/adapter.py": "before"},
        final_host_hashes={
            "host/adapter.py": "after" if mode == "adapter_change" else "before"
        },
    )

    assert result["classification"] == expected
    if mode == "protected_change":
        assert result["integrity"]["protected_file_hashes"]["TASK.md"]["before"]
        assert result["integrity"]["protected_file_hashes"]["TASK.md"]["after"]
    if mode == "adapter_change":
        assert result["integrity"]["host_protected_changes"] == ["host/adapter.py"]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "turn.completed", "usage": {"input_tokens": 12, "output_tokens": 7}},
        {
            "type": "result",
            "total_cost_usd": 0.125,
            "usage": {"input_tokens": 20, "output_tokens": 9},
        },
    ],
)
def test_codex_and_claude_json_usage_fixtures(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    usage = parse_agent_usage(path)

    assert usage["usage.input_tokens"] >= 12
    assert usage["usage.output_tokens"] >= 7


def test_codex_driver_ignores_user_plugins_and_mcp_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    monkeypatch.setattr(evaluation_host, "_agent_executable", lambda agent: [agent])

    command = evaluation_host._agent_command(
        "codex",
        "gpt-5.6-sol",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
    )

    assert command[:3] == ["codex", "exec", "--ignore-user-config"]
    assert any(item.startswith("mcp_servers.inferref.command=") for item in command)
    assert any(item.startswith("mcp_servers.inferref.args=") for item in command)
    assert not any("node_repl" in item for item in command)


def test_agent_runner_resolves_once_and_binds_executable_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    runtime = tmp_path / "node.exe"
    entry = tmp_path / "codex.js"
    runtime.write_bytes(b"node-runtime")
    entry.write_bytes(b"codex-entry")
    resolutions = 0

    def resolve(agent: str) -> list[str]:
        nonlocal resolutions
        resolutions += 1
        assert agent == "codex"
        return [str(runtime), str(entry)]

    def version(executable: list[str]) -> str:
        assert executable == [str(runtime), str(entry)]
        return "codex-cli test"

    def run_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> AgentProcessResult:
        del cwd, timeout_seconds
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            json.dumps({"type": "thread.started", "thread_id": "test"}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return AgentProcessResult(
            command=tuple(command),
            exit_code=0,
            timed_out=False,
            duration_ms=1.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout="",
            stderr="",
            usage={},
        )

    monkeypatch.setattr(evaluation_host, "_agent_executable", resolve)
    monkeypatch.setattr(evaluation_host, "_agent_version", version)
    monkeypatch.setattr(evaluation_host, "_run_process", run_process)

    result = evaluation_host.run_agent_cli(
        "codex",
        "gpt-5.6-sol",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
    )

    assert resolutions == 1
    assert result.command[:2] == (str(runtime), str(entry))
    assert result.cli_version == "codex-cli test"
    assert result.runner_evidence["components_unchanged"] is True
    assert evaluation_host._runner_evidence_valid(result.runner_evidence)
    assert result.runner_evidence["argv_policy"]["kind"] == "codex"
    assert result.runner_evidence["argv_policy"]["model"] == "gpt-5.6-sol"
    assert result.runner_evidence["argv_policy"]["config"]["approval_policy"] == "never"
    assert result.runner_evidence["argv_policy"]["config"]["web_search"] == "disabled"
    assert (
        evaluation_host._canonical_json_sha256(result.runner_evidence["argv_policy"])
        == result.runner_evidence["argv_policy_sha256"]
    )
    components = result.runner_evidence["command_components"]["before"]
    assert [item["role"] for item in components] == ["runtime", "cli_entry"]
    assert [item["name"] for item in components] == ["node.exe", "codex.js"]
    assert all(item["sha256"] for item in components)
    assert not any("path" in item for item in components)
    assert result.model_evidence["identity_level"] == "requested_only"


def test_agent_runner_detects_component_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    executable = tmp_path / "claude.exe"
    executable.write_bytes(b"before")
    monkeypatch.setattr(
        evaluation_host, "_agent_executable", lambda agent: [str(executable)]
    )
    monkeypatch.setattr(
        evaluation_host, "_agent_version", lambda command: "claude test"
    )

    def replacing_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> AgentProcessResult:
        del cwd, timeout_seconds
        executable.write_bytes(b"after")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(
            json.dumps({"type": "system", "subtype": "init", "model": "opus"}) + "\n",
            encoding="utf-8",
        )
        stderr_path.write_text("", encoding="utf-8")
        return AgentProcessResult(
            command=tuple(command),
            exit_code=0,
            timed_out=False,
            duration_ms=1.0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            stdout="",
            stderr="",
            usage={},
        )

    monkeypatch.setattr(evaluation_host, "_run_process", replacing_process)

    result = evaluation_host.run_agent_cli(
        "claude",
        "opus",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
    )

    assert result.runner_evidence["components_unchanged"] is False
    assert not evaluation_host._runner_evidence_valid(result.runner_evidence)


def test_claude_driver_exposes_only_permitted_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    monkeypatch.setattr(evaluation_host, "_agent_executable", lambda agent: [agent])
    (tmp_path / "session").mkdir()
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    command = evaluation_host._agent_command(
        "claude",
        "opus",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
        claude_settings=settings,
        claude_model_override="deepseek-v4-flash",
    )

    tools = command[command.index("--tools") + 1].split(",")
    assert tools == [
        "Read",
        "Edit",
        "mcp__inferref__inferref_capabilities",
        "mcp__inferref__inferref_context",
        "mcp__inferref__inferref_run_engine",
    ]
    assert command[command.index("--allowedTools") + 1].split(",") == tools
    assert command[command.index("--settings") + 1] == str(settings.resolve())
    assert command[command.index("--model") + 1] == "deepseek-v4-flash"


def test_claude_model_evidence_is_read_from_init_event(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "deepseek-v4-flash[1m]",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    evidence = evaluation_host._agent_model_evidence(
        "claude", "deepseek-v4-flash[1m]", events
    )

    assert evidence == {
        "requested": "deepseek-v4-flash[1m]",
        "reported": "deepseek-v4-flash[1m]",
        "evidence_source": "cli_system_init_event",
        "matches_request": True,
        "identity_level": "cli_self_reported",
        "provider_verified": False,
        "agent": "claude",
    }


def test_codex_model_evidence_is_requested_only_without_reported_event(
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"type": "thread.started", "thread_id": "test"}) + "\n",
        encoding="utf-8",
    )

    evidence = evaluation_host._agent_model_evidence("codex", "gpt-5.6-sol", events)

    assert evidence["requested"] == "gpt-5.6-sol"
    assert evidence["reported"] is None
    assert evidence["evidence_source"] == "unavailable"
    assert evidence["matches_request"] is None
    assert evidence["identity_level"] == "requested_only"


def test_codex_model_evidence_accepts_explicit_thread_model(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps({"type": "thread.started", "model": "gpt-5.6-sol"}) + "\n",
        encoding="utf-8",
    )

    evidence = evaluation_host._agent_model_evidence("codex", "gpt-5.6-sol", events)

    assert evidence["reported"] == "gpt-5.6-sol"
    assert evidence["evidence_source"] == "codex_thread_started_event"
    assert evidence["identity_level"] == "cli_self_reported"


def test_agent_api_failure_is_summarized_from_result_event(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text(
        json.dumps(
            {
                "type": "result",
                "api_error_status": 429,
                "result": "Request rejected: rate limit",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert evaluation_host._agent_api_failure(events) == (
        "Agent API failed with HTTP 429: Request rejected: rate limit"
    )


def test_malformed_agent_json_event_is_rejected(tmp_path: Path) -> None:
    events = tmp_path / "events.jsonl"
    events.write_text('{"type":"turn.started"}\nnot-json\n', encoding="utf-8")

    assert evaluation_host._validate_event_stream(events) == (
        "Agent JSON event stream is malformed at line 2"
    )


# --- attestation v0.5 hardening: settings binding, model identity, worker argv ---


def _settings_process(
    stdout_path: Path, stderr_path: Path, command: tuple[str, ...]
) -> AgentProcessResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.write_text(
        json.dumps({"type": "system", "subtype": "init", "model": "deepseek-v4-flash"})
        + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")
    return AgentProcessResult(
        command=command,
        exit_code=0,
        timed_out=False,
        duration_ms=1.0,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        stdout="",
        stderr="",
        usage={},
    )


def test_claude_settings_binding_unchanged_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    settings = tmp_path / "settings.json"
    settings.write_text('{"provider": "default"}', encoding="utf-8")
    executable = tmp_path / "claude.exe"
    executable.write_bytes(b"claude-binary")
    monkeypatch.setattr(
        evaluation_host, "_agent_executable", lambda agent: [str(executable)]
    )
    monkeypatch.setattr(
        evaluation_host, "_agent_version", lambda command: "claude test"
    )

    def run_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> AgentProcessResult:
        del cwd, timeout_seconds
        return _settings_process(stdout_path, stderr_path, tuple(command))

    monkeypatch.setattr(evaluation_host, "_run_process", run_process)

    result = evaluation_host.run_agent_cli(
        "claude",
        "deepseek-v4-flash",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
        claude_settings=settings,
        claude_model_override="deepseek-v4-flash",
    )

    evidence = result.runner_evidence["external_files"]["claude_settings"]
    assert evidence["present"] is True
    assert evidence["kind"] == "regular_file"
    assert evidence["sha256"] == evidence["after_sha256"]
    assert evidence["unchanged"] is True
    assert evaluation_host._external_file_evidence_valid(evidence)
    assert evaluation_host._runner_evidence_valid(result.runner_evidence)
    rendered = json.dumps(result.runner_evidence)
    assert str(settings) not in rendered
    assert str(tmp_path) not in rendered


def test_claude_settings_change_fails_runner_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    settings = tmp_path / "settings.json"
    settings.write_text('{"provider": "default"}', encoding="utf-8")
    executable = tmp_path / "claude.exe"
    executable.write_bytes(b"claude-binary")
    monkeypatch.setattr(
        evaluation_host, "_agent_executable", lambda agent: [str(executable)]
    )
    monkeypatch.setattr(
        evaluation_host, "_agent_version", lambda command: "claude test"
    )

    def mutating_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> AgentProcessResult:
        del cwd, timeout_seconds
        settings.write_text('{"provider": "changed"}', encoding="utf-8")
        return _settings_process(stdout_path, stderr_path, tuple(command))

    monkeypatch.setattr(evaluation_host, "_run_process", mutating_process)

    result = evaluation_host.run_agent_cli(
        "claude",
        "deepseek-v4-flash",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
        claude_settings=settings,
        claude_model_override="deepseek-v4-flash",
    )

    evidence = result.runner_evidence["external_files"]["claude_settings"]
    assert evidence["sha256"] != evidence["after_sha256"]
    assert evidence["unchanged"] is False
    assert not evaluation_host._external_file_evidence_valid(evidence)
    assert not evaluation_host._runner_evidence_valid(result.runner_evidence)


def test_claude_without_settings_reports_present_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    executable = tmp_path / "claude.exe"
    executable.write_bytes(b"claude-binary")
    monkeypatch.setattr(
        evaluation_host, "_agent_executable", lambda agent: [str(executable)]
    )
    monkeypatch.setattr(
        evaluation_host, "_agent_version", lambda command: "claude test"
    )

    def run_process(
        command: list[str],
        *,
        cwd: Path,
        timeout_seconds: float,
        stdout_path: Path,
        stderr_path: Path,
    ) -> AgentProcessResult:
        del cwd, timeout_seconds
        return _settings_process(stdout_path, stderr_path, tuple(command))

    monkeypatch.setattr(evaluation_host, "_run_process", run_process)

    result = evaluation_host.run_agent_cli(
        "claude",
        "deepseek-v4-flash",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
        claude_model_override="deepseek-v4-flash",
    )

    evidence = result.runner_evidence["external_files"]["claude_settings"]
    assert evidence["present"] is False
    assert evidence["unchanged"] is True
    assert evaluation_host._external_file_evidence_valid(evidence)
    assert evaluation_host._runner_evidence_valid(result.runner_evidence)


def test_claude_settings_symlink_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"provider": "default"}', encoding="utf-8")
    link = tmp_path / "settings.json"
    _symlink_or_skip(target, link)

    before = evaluation_host._capture_external_file(link)
    after = evaluation_host._capture_external_file(link)
    pair = evaluation_host._external_file_pair(before, after)

    assert before["present"] is True
    assert before["kind"] is None
    assert before["error"]
    assert pair["unchanged"] is False
    assert not evaluation_host._external_file_evidence_valid(pair)


def test_claude_settings_delete_and_recreate_is_detected(tmp_path: Path) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"provider": "default"}', encoding="utf-8")
    before = evaluation_host._capture_external_file(settings)

    settings.unlink()
    settings.write_text('{"provider": "default"}', encoding="utf-8")
    after = evaluation_host._capture_external_file(settings)
    pair = evaluation_host._external_file_pair(before, after)

    assert pair["sha256"] == pair["after_sha256"]
    assert pair["unchanged"] is False
    assert not evaluation_host._external_file_evidence_valid(pair)


def test_claude_settings_inode_change_with_same_content_is_detected(
    tmp_path: Path,
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"provider": "default"}', encoding="utf-8")
    before = evaluation_host._capture_external_file(settings)

    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"provider": "default"}', encoding="utf-8")
    os.replace(replacement, settings)
    after = evaluation_host._capture_external_file(settings)
    pair = evaluation_host._external_file_pair(before, after)

    assert pair["sha256"] == pair["after_sha256"]
    assert pair["unchanged"] is False
    assert not evaluation_host._external_file_evidence_valid(pair)


def test_benchmark_level_settings_change_fails_host_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"provider": "default"}', encoding="utf-8")

    def mutating_builtin_driver(
        agent: str,
        model: str,
        benchmark: EvaluationBenchmark,
        workspace: Path,
        session_root: Path,
        audit_path: Path,
        **kwargs: object,
    ) -> AgentProcessResult:
        del kwargs
        settings.write_text('{"provider": "changed"}', encoding="utf-8")
        return _fake_builtin_driver(
            agent, model, benchmark, workspace, session_root, audit_path
        )

    monkeypatch.setattr(evaluation_host, "run_agent_cli", mutating_builtin_driver)
    report_root = tmp_path / "report"
    report = evaluation_host._evaluate_benchmark_core(
        BENCHMARK_PATH,
        agents=("claude",),
        report_dir=report_root,
        claude_settings=settings,
        claude_model="deepseek-v4-flash",
    )

    assert report["status"] == "fail"
    assert report["host_integrity"]["claude_settings_unchanged"] is False
    external = report["host_integrity"]["external_files"]["claude_settings"]
    assert external["sha256"] != external["after_sha256"]
    assert external["unchanged"] is False


def test_formal_attestation_refuses_changed_claude_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = tmp_path / "settings.json"
    settings.write_text('{"provider": "default"}', encoding="utf-8")
    clean = {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "git_available": True,
    }
    monkeypatch.setattr(evaluation_host, "_repository_evidence", lambda path: clean)

    def mutating_builtin_driver(
        agent: str,
        model: str,
        benchmark: EvaluationBenchmark,
        workspace: Path,
        session_root: Path,
        audit_path: Path,
        **kwargs: object,
    ) -> AgentProcessResult:
        del kwargs
        settings.write_text('{"provider": "changed"}', encoding="utf-8")
        return _fake_builtin_driver(
            agent, model, benchmark, workspace, session_root, audit_path
        )

    monkeypatch.setattr(evaluation_host, "run_agent_cli", mutating_builtin_driver)

    with pytest.raises(RuntimeError, match="changed during the run"):
        evaluation_host._evaluate_benchmark_core(
            BENCHMARK_PATH,
            agents=("claude",),
            report_dir=tmp_path / "report",
            claude_settings=settings,
            claude_model="deepseek-v4-flash",
            public_attestation=tmp_path / "public.json",
            formal_worker_evidence=_valid_worker_evidence(),
        )

    assert not (tmp_path / "public.json").exists()


def test_model_request_satisfied_semantics() -> None:
    base = {"agent": "claude"}
    assert (
        evaluation_host._model_request_satisfied(
            {**base, "identity_level": "requested_only", "matches_request": None},
            "claude",
        )
        is None
    )
    assert (
        evaluation_host._model_request_satisfied(
            {
                **base,
                "identity_level": "cli_self_reported",
                "matches_request": True,
            },
            "claude",
        )
        is True
    )
    assert (
        evaluation_host._model_request_satisfied(
            {
                **base,
                "identity_level": "cli_self_reported",
                "matches_request": False,
            },
            "claude",
        )
        is False
    )
    assert evaluation_host._model_request_satisfied({}, "claude") is None
    assert (
        evaluation_host._model_request_satisfied(
            {**base, "identity_level": "provider_verified"}, "claude"
        )
        is None
    )


def test_reported_model_mismatch_fails_benchmark_and_classifies_identity_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluation_host, "run_agent_cli", _mismatched_model_driver)
    report_root = tmp_path / "report"
    report = evaluation_host._evaluate_benchmark_core(
        BENCHMARK_PATH,
        agents=("claude",),
        report_dir=report_root,
        claude_model="deepseek-v4-flash",
    )

    assert report["status"] == "fail"
    assert report["acceptance"]["agent_model_request_satisfied"] is False
    assert report["host_integrity"]["agent_model_request_satisfied"] is False
    assert report["agents"][0]["classification"] == "identity_policy_failure"
    attestation = json.loads(
        (report_root / "attestation.json").read_text(encoding="utf-8")
    )
    assert attestation["agents"][0]["model_request_satisfied"] is False
    assert attestation["attestation_level"] == "development"


def test_formal_attestation_refuses_model_identity_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(evaluation_host, "run_agent_cli", _mismatched_model_driver)
    clean = {
        "commit": "a" * 40,
        "dirty": False,
        "status_sha256": hashlib.sha256(b"").hexdigest(),
        "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
        "git_available": True,
    }
    monkeypatch.setattr(evaluation_host, "_repository_evidence", lambda path: clean)

    with pytest.raises(RuntimeError, match="changed during the run"):
        evaluation_host._evaluate_benchmark_core(
            BENCHMARK_PATH,
            agents=("claude",),
            report_dir=tmp_path / "report",
            claude_model="deepseek-v4-flash",
            public_attestation=tmp_path / "public.json",
            formal_worker_evidence=_valid_worker_evidence(),
        )

    assert not (tmp_path / "public.json").exists()


def test_worker_launch_policy_is_canonical_and_validated() -> None:
    evidence = {
        **evaluation_worker._worker_evidence("c" * 64),
        "python_isolated": True,
    }

    assert evaluation_host._formal_worker_evidence_valid(evidence) is True
    assert evidence["launch_policy"]["request_sha256"] == "c" * 64
    assert evidence["launch_policy"]["flags"] == ["-I", "-m"]
    assert evidence["python_executable"]["sha256"]
    assert "orig_argv" not in json.dumps(evidence)
    assert (
        evaluation_host._canonical_json_sha256(evidence["launch_policy"])
        == evidence["launch_policy_sha256"]
    )

    tampered = {
        **evidence,
        "launch_policy": {
            **evidence["launch_policy"],
            "request_transport": "network",
        },
    }
    assert not evaluation_host._formal_worker_evidence_valid(tampered)

    wrong_digest = {**evidence, "launch_policy_sha256": "e" * 64}
    assert not evaluation_host._formal_worker_evidence_valid(wrong_digest)

    bad_executable = {
        **evidence,
        "python_executable": {
            "name": "python.exe",
            "size": -1,
            "sha256": "f" * 64,
        },
    }
    assert not evaluation_host._formal_worker_evidence_valid(bad_executable)


def test_formal_worker_rejects_report_without_matching_request_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(
        command: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        del kwargs
        report_root = tmp_path / "report"
        report_root.mkdir()
        (report_root / "report.json").write_text(
            json.dumps({"status": "pass"}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(evaluation_host.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="does not match the parent request"):
        evaluation_host._run_formal_worker(
            BENCHMARK_PATH,
            agents=("codex",),
            report_dir=tmp_path / "report",
            claude_settings=None,
            claude_model=None,
            public_attestation=tmp_path / "public.json",
        )


def _posix_which_map(entries: dict[str, str]):
    def which(name: str) -> str | None:
        return entries.get(name)

    return which


def test_posix_env_shebang_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_text("#!/usr/bin/env my-node\n", encoding="utf-8")
    node = tmp_path / "node"
    node.write_bytes(b"node-binary")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry), "my-node": str(node)}),
    )

    resolved = evaluation_host._resolve_posix_agent_command("codex")

    assert resolved.components == (str(node), str(entry))
    assert resolved.prefix == (str(node), str(entry))


def test_posix_env_s_shebang_preserves_interpreter_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_text("#!/usr/bin/env -S my-node --no-warnings\n", encoding="utf-8")
    node = tmp_path / "node"
    node.write_bytes(b"node-binary")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry), "my-node": str(node)}),
    )

    resolved = evaluation_host._resolve_posix_agent_command("codex")

    assert resolved.components == (str(node), str(entry))
    assert resolved.prefix == (str(node), "--no-warnings", str(entry))


def test_posix_env_s_shebang_preserves_quoted_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_text('#!/usr/bin/env -S my-node --flag "a b"\n', encoding="utf-8")
    node = tmp_path / "node"
    node.write_bytes(b"node-binary")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry), "my-node": str(node)}),
    )

    resolved = evaluation_host._resolve_posix_agent_command("codex")

    assert resolved.prefix == (str(node), "--flag", "a b", str(entry))


def test_posix_env_s_shebang_quoted_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_text('#!/usr/bin/env -S "my-node --no-warnings"\n', encoding="utf-8")
    node = tmp_path / "node"
    node.write_bytes(b"node-binary")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry), "my-node": str(node)}),
    )

    resolved = evaluation_host._resolve_posix_agent_command("codex")

    assert resolved.components == (str(node), str(entry))
    assert resolved.prefix == (str(node), "--no-warnings", str(entry))


def test_posix_env_value_option_does_not_become_interpreter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_text("#!/usr/bin/env -u FOO my-node\n", encoding="utf-8")
    node = tmp_path / "node"
    node.write_bytes(b"node-binary")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry), "my-node": str(node)}),
    )

    resolved = evaluation_host._resolve_posix_agent_command("codex")

    assert resolved.components == (str(node), str(entry))
    assert resolved.prefix == (str(node), str(entry))


@pytest.mark.skipif(os.name == "nt", reason="absolute POSIX interpreter")
def test_posix_direct_shebang_preserves_interpreter_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_text("#!/usr/bin/python3 -s\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry)}),
    )

    resolved = evaluation_host._resolve_posix_agent_command("codex")

    interpreter = str(Path("/usr/bin/python3").resolve(strict=True))
    assert resolved.components == (interpreter, str(entry))
    assert resolved.prefix == (interpreter, "-s", str(entry))


def test_posix_shebang_missing_interpreter_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_text("#!/usr/bin/env missing-interpreter\n", encoding="utf-8")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry)}),
    )

    with pytest.raises(FileNotFoundError, match="interpreter"):
        evaluation_host._resolve_posix_agent_command("codex")


def test_posix_native_executable_without_shebang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = tmp_path / "codex"
    entry.write_bytes(b"\x7fELF\x02\x01\x01\x00binary")
    monkeypatch.setattr(
        evaluation_host.shutil,
        "which",
        _posix_which_map({"codex": str(entry)}),
    )

    resolved = evaluation_host._resolve_posix_agent_command("codex")

    assert resolved.components == (str(entry),)
    assert resolved.prefix == (str(entry),)


def test_claude_argv_policy_records_tools_and_settings_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    benchmark = _benchmark()
    monkeypatch.setattr(evaluation_host, "_agent_executable", lambda agent: [agent])
    (tmp_path / "session").mkdir()
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")

    command = evaluation_host._agent_command(
        "claude",
        "opus",
        benchmark,
        tmp_path / "workspace",
        tmp_path / "session",
        tmp_path / "audit.jsonl",
        claude_settings=settings,
        claude_model_override="deepseek-v4-flash",
    )
    policy = evaluation_host._agent_argv_policy("claude", command)

    assert policy["settings_present"] is True
    assert policy["model"] == "deepseek-v4-flash"
    assert policy["permission_mode"] == "acceptEdits"
    assert "Read,Edit" in policy["tools"]
    assert policy["strict_mcp_config"] is True
    rendered = json.dumps(policy)
    assert str(settings) not in rendered
    assert str(tmp_path) not in rendered


def test_runner_evidence_rejects_malformed_component_shape() -> None:
    assert evaluation_host._runner_evidence_valid(_valid_runner_evidence("codex"))
    assert not evaluation_host._runner_evidence_valid(
        _runner_with_component(sha256="A" * 64)
    )
    assert not evaluation_host._runner_evidence_valid(_runner_with_component(size=-1))
    assert not evaluation_host._runner_evidence_valid(
        _runner_with_component(error="boom")
    )
    assert not evaluation_host._runner_evidence_valid(_runner_with_component(name=""))


def test_runner_evidence_rejects_wrong_role_sequence() -> None:
    evidence = _valid_runner_evidence("codex")
    component = evidence["command_components"]["before"][0]
    duplicate = [component, component]
    snapshot = {"before": duplicate, "after_version": duplicate, "after": duplicate}

    assert not evaluation_host._runner_evidence_valid(
        {**evidence, "command_components": snapshot}
    )


def test_runner_evidence_rejects_tampered_argv_policy() -> None:
    evidence = _valid_runner_evidence("codex")
    tampered = {**evidence, "argv_policy": {"kind": "codex", "policy_marker": "evil"}}

    assert not evaluation_host._runner_evidence_valid(tampered)


def test_runner_evidence_rejects_invalid_external_file_evidence() -> None:
    evidence = _valid_runner_evidence("claude")
    evidence["external_files"] = {
        "claude_settings": {
            "present": True,
            "kind": None,
            "size": None,
            "sha256": None,
            "after_sha256": None,
            "unchanged": False,
        }
    }

    assert not evaluation_host._runner_evidence_valid(evidence)
