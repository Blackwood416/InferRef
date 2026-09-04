"""CLI integration tests for `inferref align` commands (0.10 L6)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.cli.main import main
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.tensor.codec import write_array


def _setup_trace_and_adapter(tmp_path: Path) -> tuple[Path, Path]:
    trace_dir = tmp_path / "trace"
    payloads = trace_dir / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)

    write_array(payloads / "v1.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    write_array(payloads / "v2.irtensor", np.array([2.0, 4.0], dtype=np.float32))

    graph = Graph()
    graph.values = [
        TensorValueRecord(
            id=1,
            shape=(2,),
            stride=(1,),
            dtype="float32",
            capture=CaptureInfo(mode="full", payload="payloads/v1.irtensor"),
        ),
        TensorValueRecord(
            id=2,
            shape=(2,),
            stride=(1,),
            dtype="float32",
            producer=10,
            capture=CaptureInfo(mode="full", payload="payloads/v2.irtensor"),
        ),
    ]
    graph.operators = [
        OperatorRecord(
            id=10,
            execution_index=0,
            op="double",
            result=TensorRef(2),
            positional_args=(TensorRef(1),),
            module_stack=(100,),
        ),
    ]
    graph.reindex()
    pkg = TracePackage(
        manifest=Manifest(),
        graph=graph,
        modules=[ModuleRecord(id=100, path="model")],
        regions=[RegionRecord(id=1, name="op_double", node_ids=(10,), inputs=(1,), outputs=(2,))],
        root=trace_dir,
    )
    pkg.save(trace_dir)

    adapter_path = tmp_path / "adapter.json"
    adapter_data = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "test-engine",
        "target_device": "cpu",
        "capabilities": {"device_types": ["cpu"], "dtypes": ["float32"], "max_rank": 4},
        "command": [
            "{python}",
            "-c",
            "import sys, shutil; from pathlib import Path; tc, out = Path(sys.argv[1]), Path(sys.argv[2]); shutil.copytree(tc / 'reference', out, dirs_exist_ok=True)",
            "{testcase}",
            "{output}",
        ],
    }
    adapter_path.write_text(json.dumps(adapter_data, indent=2), encoding="utf-8")
    return trace_dir, adapter_path


def test_align_cli_workflow(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    trace_dir, adapter_path = _setup_trace_and_adapter(tmp_path)
    session_dir = tmp_path / "alignment_cli"

    # 1. align init
    ret = main([
        "align", "init", str(trace_dir), str(adapter_path),
        "--dir", str(session_dir), "--name", "cli-test", "--json"
    ])
    assert ret == 0
    captured = capsys.readouterr()
    init_data = json.loads(captured.out)
    assert init_data["format"] == "inferref-alignment"
    assert (session_dir / "alignment.json").is_file()

    # 2. align plan
    ret = main(["align", "plan", str(session_dir), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    plan_data = json.loads(captured.out)
    assert plan_data["recommended_boundary"] == "region:1"

    # 3. align run
    ret = main(["align", "run", str(session_dir), "--limit", "1", "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    run_data = json.loads(captured.out)
    assert run_data["results"][0]["status"] == "pass"

    # 4. align status
    ret = main(["align", "status", str(session_dir), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    status_data = json.loads(captured.out)
    assert status_data["counts"]["verified"] == 1

    # 5. align summary
    ret = main(["align", "summary", str(session_dir), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    summary_data = json.loads(captured.out)
    assert summary_data["counts"]["verified"] == 1

    # 6. align narrow
    ret = main(["align", "narrow", str(session_dir), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    narrow_data = json.loads(captured.out)
    assert "narrowing" in narrow_data

    # 7. align note with full options
    ret = main([
        "align", "note", str(session_dir), "region:1", "milestone", "First boundary passed",
        "--duration-minutes", "4.5",
        "--milestone", "first-correct-model",
        "--target", "compiler",
        "--json"
    ])
    assert ret == 0
    captured = capsys.readouterr()
    note_data = json.loads(captured.out)
    assert note_data["note"]["category"] == "milestone"
    assert note_data["note"]["duration_minutes"] == 4.5
    assert note_data["note"]["target"] == "compiler"

    # 8. align report
    rep_md = session_dir / "reports" / "custom-report.md"
    rep_json = session_dir / "reports" / "custom-report.json"
    ret = main([
        "align", "report", str(session_dir),
        "--output", str(rep_md),
        "--json-sidecar", str(rep_json),
    ])
    assert ret == 0
    assert rep_md.is_file()
    assert rep_json.is_file()
    sidecar = json.loads(rep_json.read_text(encoding="utf-8"))
    assert sidecar["format"] == "inferref-alignment-report"

    # 9. Test alignment alias and agent subcommands
    ret = main(["alignment", "status", str(session_dir), "--json"])
    assert ret == 0

    ret = main(["agent", "alignment_status", str(session_dir), "--json"])
    assert ret == 0

    ret = main(["agent", "alignment", "status", str(session_dir), "--json"])
    assert ret == 0


def test_align_cli_invalid_session_exit_code_2(tmp_path: Path) -> None:
    nonexistent = tmp_path / "nonexistent"
    assert main(["align", "status", str(nonexistent)]) == 2
    assert main(["align", "run", str(nonexistent)]) == 2
    assert main(["align", "summary", str(nonexistent)]) == 2
    assert main(["align", "narrow", str(nonexistent)]) == 2
    assert main(["align", "note", str(nonexistent), "b1", "cat", "txt"]) == 2
    assert main(["align", "report", str(nonexistent)]) == 2


def test_align_cli_continue_on_fail_returns_exit_1(tmp_path: Path) -> None:
    trace_dir, _ = _setup_trace_and_adapter(tmp_path)
    # Adapter that produces mismatched output
    bad_adapter = tmp_path / "bad_adapter.json"
    bad_adapter.write_text(
        json.dumps({
            "format": "inferref-engine-adapter",
            "format_version": "0.2",
            "name": "bad-engine",
            "target_device": "cpu",
            "capabilities": {"device_types": ["cpu"], "dtypes": ["float32"], "max_rank": 4},
            "command": [
                "{python}",
                "-c",
                "import sys, shutil; from pathlib import Path; tc, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True); (out / 'v2.irtensor').write_bytes(b'bad data')",
                "{testcase}",
                "{output}",
            ],
        }),
        encoding="utf-8",
    )
    session_dir = tmp_path / "session_fail"
    ret = main(["align", "init", str(trace_dir), str(bad_adapter), "--dir", str(session_dir)])
    assert ret == 0

    # Even with --continue-on-fail, a failing run must return EXIT_FAIL (1)
    ret = main(["align", "run", str(session_dir), "--limit", "1", "--continue-on-fail"])
    assert ret == 1


def test_align_init_invalid_paths_returns_exit_2(tmp_path: Path) -> None:
    trace_dir, adapter_path = _setup_trace_and_adapter(tmp_path)
    # Missing trace returns 2
    assert main(["align", "init", str(tmp_path / "nonexistent_trace"), str(adapter_path)]) == 2
    # Missing adapter returns 2
    assert main(["align", "init", str(trace_dir), str(tmp_path / "nonexistent_adapter.json")]) == 2


def test_align_run_testcase_constraints_returns_exit_2(tmp_path: Path) -> None:
    trace_dir, adapter_path = _setup_trace_and_adapter(tmp_path)
    session_dir = tmp_path / "session_tc_constraints"
    assert main(["align", "init", str(trace_dir), str(adapter_path), "--dir", str(session_dir)]) == 0

    dummy_tc = tmp_path / "dummy_tc"
    dummy_tc.mkdir()

    # 1. --testcase without subject returns 2
    assert main(["align", "run", str(session_dir), "--testcase", str(dummy_tc), "--fixture-kind", "synthetic"]) == 2

    # 2. --testcase with multiple boundaries returns 2
    assert (
        main([
            "align",
            "run",
            str(session_dir),
            "--testcase",
            str(dummy_tc),
            "--boundary",
            "region:1",
            "--boundary",
            "region:2",
            "--fixture-kind",
            "synthetic",
        ])
        == 2
    )

    # 3. --testcase without explicit --fixture-kind returns 2
    assert main(["align", "run", str(session_dir), "--testcase", str(dummy_tc), "--subject", "region:1"]) == 2


def test_cli_agent_report_consistency_with_events_same_timestamp(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from inferref.agent.service import alignment_status
    from inferref.alignment.report import generate_alignment_report
    from inferref.alignment.schema import AlignmentEvent, Evidence
    from inferref.alignment.store import AlignmentStore

    trace_dir, adapter_path = _setup_trace_and_adapter(tmp_path)
    session_dir = tmp_path / "session_same_ts"
    assert main(["align", "init", str(trace_dir), str(adapter_path), "--dir", str(session_dir)]) == 0

    store = AlignmentStore.load(session_dir)
    same_time = "2026-09-03T01:00:00Z"
    dummy_tc = session_dir / "testcases" / "dummy"
    dummy_tc.mkdir(parents=True, exist_ok=True)
    (dummy_tc / "testcase.json").write_text("{}", encoding="utf-8")
    from inferref.alignment.hashing import sha256_file, tree_sha256

    tc_sha = tree_sha256(dummy_tc)

    dummy_run = session_dir / "runs" / "dummy"
    dummy_run.mkdir(parents=True, exist_ok=True)
    run_file = dummy_run / "inferref-run.json"
    run_file.write_text("{}", encoding="utf-8")
    run_sha = sha256_file(run_file)

    prov = {
        "trace_metadata_sha256": store.snapshot.source["trace"]["metadata_sha256"],
        "adapter_normalized_sha256": store.snapshot.target["adapter"]["normalized_sha256"],
        "engine_identity": {"kind": "git", "revision": "rev-1", "dirty": False},
        "effective_comparison_sha256": "0" * 64,
        "testcase": {"path": "testcases/dummy", "tree_sha256": tc_sha},
        "run_record": {"path": "runs/dummy/inferref-run.json", "sha256": run_sha},
    }

    # ev-zzz (layout fail, seq 2) vs ev-aaa (pass, seq 3)
    ev_fail = Evidence(
        id="ev-zzz-layout-fail",
        created_at=same_time,
        type="numerical_comparison",
        subject={"boundary": "region:1", "name": "root"},
        fixture={"kind": "reference_derived", "tags": []},
        result="fail",
        observations={
            "layout_match": False,
            "first_failure": {"mismatch_count": 0, "max_abs_error": 0.0},
            "output_roles": ["logits"],
        },
        provenance=prov,
    )
    ev_pass = Evidence(
        id="ev-aaa-pass",
        created_at=same_time,
        type="numerical_comparison",
        subject={"boundary": "region:1", "name": "root"},
        fixture={"kind": "reference_derived", "tags": []},
        result="pass",
        observations={
            "layout_match": True,
            "outputs_match": True,
            "output_roles": ["logits"],
            "comparison_status": "pass",
        },
        provenance=prov,
    )

    store.evidence[ev_fail.id] = ev_fail
    store.evidence[ev_pass.id] = ev_pass

    e1 = AlignmentEvent(seq=1, timestamp=same_time, event="session_initialized", data={})
    e2 = AlignmentEvent(seq=2, timestamp=same_time, event="boundary_failed", data={"boundary": "region:1"}, evidence_ids=[ev_fail.id])
    e3 = AlignmentEvent(seq=3, timestamp=same_time, event="boundary_verified", data={"boundary": "region:1"}, evidence_ids=[ev_pass.id])

    # Save to disk
    with open(store.events_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(ev_obj.to_dict()) + "\n" for ev_obj in (e1, e2, e3))

    ev_dir = session_dir / "evidence"
    ev_dir.mkdir(parents=True, exist_ok=True)
    (ev_dir / f"{ev_fail.id}.json").write_text(json.dumps(ev_fail.to_dict()), encoding="utf-8")
    (ev_dir / f"{ev_pass.id}.json").write_text(json.dumps(ev_pass.to_dict()), encoding="utf-8")

    # 1. Test align report
    store_reloaded = AlignmentStore.load(session_dir, validate_artifacts=False)
    _, sidecar = generate_alignment_report(store_reloaded)
    rep_cands = sidecar.get("classification", [])
    assert not any(c.get("domain") == "layout" for c in rep_cands)

    # 2. Test align status via CLI
    capsys.readouterr()
    ret = main(["align", "status", str(session_dir), "--json"])
    assert ret == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    cli_cands = payload.get("classification", [])
    assert not any(c.get("domain") == "layout" for c in cli_cands)

    # 3. Test AgentService alignment_status
    resp = alignment_status(str(session_dir))
    assert resp.status == "ok"
    agent_cands = resp.data.get("classification", [])
    assert not any(c.get("domain") == "layout" for c in agent_cands)
