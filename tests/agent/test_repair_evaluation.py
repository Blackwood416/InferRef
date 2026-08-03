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

from inferref.agent import evaluation_host
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


def test_public_attestation_omits_private_transcript_and_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        evaluation_host,
        "_repository_evidence",
        lambda path: {
            "commit": "a" * 40,
            "dirty": False,
            "status_sha256": hashlib.sha256(b"").hexdigest(),
            "git_diff_sha256": hashlib.sha256(b"").hexdigest(),
            "git_available": True,
        },
    )
    public_path = tmp_path / "public" / "attestation.json"
    report_root = tmp_path / "private-report"
    report = evaluate_benchmark(
        BENCHMARK_PATH,
        agents=("codex", "claude"),
        report_dir=report_root,
        driver=_fake_repair_driver,
        public_attestation=public_path,
    )

    attestation = json.loads(public_path.read_text(encoding="utf-8"))
    assert attestation == json.loads(
        (report_root / "attestation.json").read_text(encoding="utf-8")
    )
    assert attestation["status"] == report["status"] == "pass"
    assert attestation["benchmark"]["success_policy"] == _benchmark().success.to_dict()
    assert all(agent["engine_patch"] for agent in attestation["agents"])
    assert all(agent["final_engine_sha256"] for agent in attestation["agents"])
    assert all(agent["raw_transcript_sha256"] for agent in attestation["agents"])
    assert attestation["repository"]["dirty"] is False
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
    assert attestation["report_json_sha256"]
    rendered = json.dumps(attestation)
    assert str(tmp_path) not in rendered
    assert '"prompt"' not in rendered
    assert '"stdout"' not in rendered
    assert "thinking" not in rendered


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
        evaluate_benchmark(
            BENCHMARK_PATH,
            agents=("codex",),
            report_dir=tmp_path / "report",
            driver=_fake_repair_driver,
            public_attestation=tmp_path / "public.json",
        )

    assert not (tmp_path / "report").exists()


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


def test_claude_resolved_model_is_read_from_init_event(tmp_path: Path) -> None:
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

    assert evaluation_host._agent_model_from_events(events) == "deepseek-v4-flash[1m]"


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
