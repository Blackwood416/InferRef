"""Tests for Alignment Runner and Orchestration (0.10 L4)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.protocol import EngineAdapter
from inferref.alignment.hashing import (
    normalized_adapter_sha256,
    sha256_file,
    trace_metadata_sha256,
)
from inferref.alignment.runner import (
    AlignmentRunner,
    PreparedCandidate,
    resolve_engine_identity,
)
from inferref.alignment.store import AlignmentStore
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.tensor.codec import write_array


def _make_trace(tmp_path: Path) -> TracePackage:
    payloads = tmp_path / "payloads"
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
        root=tmp_path,
    )
    pkg.save(tmp_path)
    return pkg


def _make_adapter_file(
    path: Path,
    *,
    script: str = "import sys, shutil; from pathlib import Path; tc, out = Path(sys.argv[1]), Path(sys.argv[2]); shutil.copytree(tc / 'reference', out, dirs_exist_ok=True)",
    capabilities: dict | None = None,
) -> Path:
    if capabilities is None:
        capabilities = {"device_types": ["cpu"], "dtypes": ["float32"], "max_rank": 4}
    data = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "test-engine",
        "target_device": "cpu",
        "capabilities": capabilities,
        "command": [
            "{python}",
            "-c",
            script,
            "{testcase}",
            "{output}",
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def test_engine_identity_resolution(tmp_path: Path) -> None:
    adapter_path = _make_adapter_file(tmp_path / "adapter.json")
    adapter = EngineAdapter.load(adapter_path)

    # 1. Explicit revision
    ident = resolve_engine_identity(adapter, explicit_revision="rev-123")
    assert ident["kind"] == "manual"
    assert ident["revision"] == "rev-123"

    # 2. Local script/executable file resolution
    ident_auto = resolve_engine_identity(adapter)
    assert ident_auto["kind"] in ("git", "file")
    assert ident_auto["revision"]


def test_engine_identity_resolution_follows_trusted_python_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_python = tmp_path / "python-real"
    real_python.write_bytes(b"test interpreter")
    linked_python = tmp_path / "python-link"
    try:
        linked_python.symlink_to(real_python)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    import inferref.alignment.runner as runner_module

    monkeypatch.setattr(runner_module.sys, "executable", str(linked_python))
    adapter = EngineAdapter.load(_make_adapter_file(tmp_path / "adapter.json"))

    identity = resolve_engine_identity(adapter)

    assert identity["kind"] == "file"
    assert identity["revision"].startswith("sha256:")
    assert identity["dirty"] is False


def test_runner_initialization_digest_check(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    adapter_path = _make_adapter_file(tmp_path / "adapter.json")
    adapter = EngineAdapter.load(adapter_path)

    trace_sha = trace_metadata_sha256(pkg.root)
    adapter_sha = normalized_adapter_sha256(adapter.to_dict())

    # Correct store initialization
    store = AlignmentStore.initialize(
        tmp_path / "alignment",
        trace_path=tmp_path / "trace",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_sha,
        adapter_digest=adapter_sha,
    )
    runner = AlignmentRunner(store, pkg, adapter)
    assert runner.inventory.resolve("region:1") is not None

    # Mismatched adapter sha fails closed
    store_bad = AlignmentStore.initialize(
        tmp_path / "alignment_bad",
        trace_path=tmp_path / "trace",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_sha,
        adapter_digest="0" * 64,
    )
    with pytest.raises(ValueError, match="adapter digest mismatch"):
        AlignmentRunner(store_bad, pkg, adapter)


def test_runner_preflight_capability_check(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    # Adapter only supports float64, while trace uses float32
    adapter_path = _make_adapter_file(
        tmp_path / "adapter_caps.json",
        capabilities={"device_types": ["cpu"], "dtypes": ["float64"], "max_rank": 4},
    )
    adapter = EngineAdapter.load(adapter_path)
    store = AlignmentStore.initialize(
        tmp_path / "alignment_caps",
        trace_path=tmp_path / "trace",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    results = runner.run(limit=1, boundary_ids=["region:1"])
    assert len(results) == 1
    assert results[0].status == "unsupported"
    assert store.snapshot.boundaries["region:1"]["state"] == "unsupported"


def test_cli_plan_and_run_choose_same_adapter_supported_boundary(tmp_path: Path, capsys) -> None:
    from inferref.alignment.planner import plan_alignment
    from inferref.cli.main import main

    pkg = _make_trace(tmp_path / "trace")
    # The first region requires float32. The adapter only supports the second,
    # otherwise-equivalent float64 region, so omission of adapter changes plan.
    for vid, producer in [(3, None), (4, 11)]:
        write_array(pkg.root / f"payloads/v{vid}.irtensor", np.array([2., 4.], dtype=np.float64))
        pkg.graph.values.append(TensorValueRecord(
            id=vid, shape=(2,), stride=(1,), dtype="float64", producer=producer,
            capture=CaptureInfo(mode="full", payload=f"payloads/v{vid}.irtensor"),
        ))
    pkg.graph.operators.append(OperatorRecord(
        id=11, execution_index=1, op="double", result=TensorRef(4),
        positional_args=(TensorRef(3),),
    ))
    pkg.regions.append(RegionRecord(id=2, name="op_double_64", node_ids=(11,), inputs=(3,), outputs=(4,)))
    pkg.graph.reindex()
    pkg.save(pkg.root)
    adapter_path = _make_adapter_file(tmp_path / "adapter.json", capabilities={
        "device_types": ["cpu"], "dtypes": ["float64"], "max_rank": 4,
    })
    adapter = EngineAdapter.load(adapter_path)
    session = tmp_path / "alignment"
    store = AlignmentStore.initialize(
        session, trace_path=pkg.root, adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    expected = plan_alignment(pkg, session=store.snapshot, adapter=adapter).candidates[0].boundary
    assert expected == "region:2"
    assert plan_alignment(pkg, session=store.snapshot).candidates[0].boundary != expected
    assert main(["align", "plan", str(session), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["recommended_boundary"] == expected
    assert main(["align", "run", str(session), "--limit", "1", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["results"][0]["boundary"] == expected
    assert result["results"][0]["status"] == "pass"


def test_runner_does_not_auto_select_known_unsupported_boundaries(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    adapter_path = _make_adapter_file(tmp_path / "adapter.json", capabilities={
        "device_types": ["cpu"], "dtypes": ["int8"], "max_rank": 4,
    })
    adapter = EngineAdapter.load(adapter_path)
    store = AlignmentStore.initialize(
        tmp_path / "session", trace_path=pkg.root, adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    assert AlignmentRunner(store, pkg, adapter).run(limit=1) == []


def test_runner_execution_reuse_and_narrowing(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    adapter_path = _make_adapter_file(tmp_path / "adapter.json")
    adapter = EngineAdapter.load(adapter_path)

    store = AlignmentStore.initialize(
        tmp_path / "alignment_run",
        trace_path=tmp_path / "trace",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    # 1. First run extracts and verifies
    results1 = runner.run(limit=1, boundary_ids=["region:1"])
    assert len(results1) == 1
    assert results1[0].passed
    assert store.snapshot.boundaries["region:1"]["state"] == "verified"

    # Verify event log has testcase_extracted, boundary_verified, and session_verified
    event_names = [e.event for e in store.events]
    assert "testcase_extracted" in event_names
    assert "boundary_verified" in event_names
    assert "session_verified" in event_names
    assert store.snapshot.status == "verified"

    # Reloaded store also has verified status
    reloaded = AlignmentStore.load(store.root)
    assert reloaded.snapshot.status == "verified"

    # 2. Second run reuses the cached testcase
    results2 = runner.run(limit=1, boundary_ids=["region:1"])
    assert len(results2) == 1
    event_names2 = [e.event for e in store.events]
    assert "testcase_reused" in event_names2


def _make_2region_trace(tmp_path: Path) -> TracePackage:
    payloads = tmp_path / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)

    write_array(payloads / "v1.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    write_array(payloads / "v2.irtensor", np.array([2.0, 4.0], dtype=np.float32))
    write_array(payloads / "v3.irtensor", np.array([4.0, 8.0], dtype=np.float32))

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
        TensorValueRecord(
            id=3,
            shape=(2,),
            stride=(1,),
            dtype="float32",
            producer=20,
            capture=CaptureInfo(mode="full", payload="payloads/v3.irtensor"),
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
        OperatorRecord(
            id=20,
            execution_index=1,
            op="double",
            result=TensorRef(3),
            positional_args=(TensorRef(2),),
            module_stack=(100,),
        ),
    ]
    graph.reindex()
    pkg = TracePackage(
        manifest=Manifest(),
        graph=graph,
        modules=[ModuleRecord(id=100, path="model")],
        regions=[
            RegionRecord(id=1, name="first_step", node_ids=(10,), inputs=(1,), outputs=(2,)),
            RegionRecord(id=2, name="second_step", node_ids=(20,), inputs=(2,), outputs=(3,)),
        ],
        root=tmp_path,
    )
    pkg.save(tmp_path)
    return pkg


def test_runner_multi_boundary_no_premature_session_verified(tmp_path: Path) -> None:
    pkg = _make_2region_trace(tmp_path / "trace_2b")
    adapter_path = _make_adapter_file(tmp_path / "adapter.json")
    adapter = EngineAdapter.load(adapter_path)

    store = AlignmentStore.initialize(
        tmp_path / "alignment_2b",
        trace_path=tmp_path / "trace_2b",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    # 1. Run only region:1
    results1 = runner.run(limit=1, boundary_ids=["region:1"])
    assert len(results1) == 1
    assert results1[0].passed
    assert store.snapshot.boundaries["region:1"]["state"] == "verified"
    # Session must NOT be verified because region:2 is not verified!
    assert store.snapshot.status == "aligning"
    event_names = [e.event for e in store.events]
    assert "session_verified" not in event_names
    assert AlignmentStore.load(store.root).snapshot.status == "aligning"

    # 2. Run region:2
    results2 = runner.run(limit=1, boundary_ids=["region:2"])
    assert len(results2) == 1
    assert results2[0].passed
    assert store.snapshot.boundaries["region:2"]["state"] == "verified"
    # Now that both required boundaries are verified, session transitions to verified!
    assert store.snapshot.status == "verified"
    event_names2 = [e.event for e in store.events]
    assert "session_verified" in event_names2
    assert AlignmentStore.load(store.root).snapshot.status == "verified"


def test_runner_external_testcase_imported_to_session(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace_ext")
    adapter_path = _make_adapter_file(tmp_path / "adapter.json")
    adapter = EngineAdapter.load(adapter_path)

    # Extract standalone testcase outside session
    ext_tc_dir = tmp_path / "external_standalone_tc"
    from inferref.testcase.extract import extract_region

    extract_region(pkg, pkg.regions[0], ext_tc_dir, force=True)

    store = AlignmentStore.initialize(
        tmp_path / "alignment_ext",
        trace_path=tmp_path / "trace_ext",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    results = runner.run(
        limit=1,
        boundary_ids=["region:1"],
        fixture_kind="reference_derived",
        external_testcase=ext_tc_dir,
    )
    assert len(results) == 1
    assert results[0].passed
    assert results[0].evidence is not None

    # Verify evidence provenance testcase path is a relative POSIX path under testcases/
    ev = store.evidence[results[0].evidence.id]
    tc_path_in_provenance = ev.provenance["testcase"]["path"]
    assert not Path(tc_path_in_provenance).is_absolute()
    assert tc_path_in_provenance.startswith("testcases/")
    assert "\\" not in tc_path_in_provenance

    # Verify testcase was actually imported into session cache
    cached_path = store.root / tc_path_in_provenance
    assert cached_path.is_dir()
    assert (cached_path / "testcase.json").is_file()


def test_runner_unsupported_boundary_does_not_verify_session(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace_unsup")
    adapter_path = _make_adapter_file(
        tmp_path / "adapter_caps.json",
        capabilities={"device_types": ["cpu"], "dtypes": ["float64"], "max_rank": 4},
    )
    adapter = EngineAdapter.load(adapter_path)
    store = AlignmentStore.initialize(
        tmp_path / "alignment_unsup",
        trace_path=tmp_path / "trace_unsup",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    results = runner.run(limit=1, boundary_ids=["region:1"])
    assert results[0].status == "unsupported"
    assert store.snapshot.boundaries["region:1"]["state"] == "unsupported"
    assert store.snapshot.status != "verified"
    event_names = [e.event for e in store.events]
    assert "session_verified" not in event_names


def test_runner_external_testcase_cache_revalidates_and_recovers_if_corrupted(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace_ext2")
    adapter_path = _make_adapter_file(tmp_path / "adapter2.json")
    adapter = EngineAdapter.load(adapter_path)

    ext_tc_dir = tmp_path / "external_standalone_tc2"
    from inferref.testcase.extract import extract_region

    extract_region(pkg, pkg.regions[0], ext_tc_dir, force=True)

    store = AlignmentStore.initialize(
        tmp_path / "alignment_ext2",
        trace_path=tmp_path / "trace_ext2",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    # First run imports to cache
    results1 = runner.run(
        limit=1,
        boundary_ids=["region:1"],
        fixture_kind="reference_derived",
        external_testcase=ext_tc_dir,
    )
    assert len(results1) == 1
    assert results1[0].passed
    tc_rel = store.evidence[results1[0].evidence.id].provenance["testcase"]["path"]
    cached_tc = store.root / tc_rel

    # Corrupt the cached testcase
    (cached_tc / "testcase.json").write_text("corrupted json content", encoding="utf-8")

    # Second run with the valid external testcase must detect corrupted cache, re-import cleanly, and succeed!
    results2 = runner.run(
        limit=1,
        boundary_ids=["region:1"],
        fixture_kind="reference_derived",
        external_testcase=ext_tc_dir,
    )
    assert len(results2) == 1
    assert results2[0].passed
    assert (cached_tc / "testcase.json").is_file()
    # Cache content is now restored and valid JSON
    assert "corrupted" not in (cached_tc / "testcase.json").read_text(encoding="utf-8")


def test_prepare_then_finalize_from_output_matches_sequential_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch seam must reproduce a sequential run without launching an adapter."""
    import inferref.alignment.runner as runner_module
    from inferref.agent.executor import execute_adapter as execute_adapter_impl

    pkg = _make_trace(tmp_path / "trace_prepare")
    adapter_path = _make_adapter_file(tmp_path / "adapter_prepare.json")
    adapter = EngineAdapter.load(adapter_path)

    def _open_session(name: str) -> AlignmentStore:
        return AlignmentStore.initialize(
            tmp_path / name,
            trace_path=tmp_path / "trace_prepare",
            adapter_path=adapter_path,
            trace_metadata_digest=trace_metadata_sha256(pkg.root),
            adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
        )

    sequential_store = _open_session("alignment_sequential")
    sequential = AlignmentRunner(sequential_store, pkg, adapter).run(
        limit=1, boundary_ids=["region:1"], engine_revision="rev-prepare"
    )
    assert len(sequential) == 1
    sequential_result = sequential[0]
    assert sequential_result.status == "pass"
    assert sequential_result.evidence is not None

    finalize_store = _open_session("alignment_finalize")
    runner = AlignmentRunner(finalize_store, pkg, adapter)
    prepared = runner._prepare_candidate("region:1", engine_revision="rev-prepare")
    assert isinstance(prepared, PreparedCandidate)
    assert prepared.unsupported == ()

    # Produce the output outside the runner, the way a batch process would.
    execution_result = execute_adapter_impl(
        testcase=prepared.testcase_path,
        adapter=adapter,
        runs_root=prepared.output_dir.parent,
    )

    def _must_not_launch(*args: object, **kwargs: object) -> None:
        raise AssertionError("run_candidate_from_output must not launch an adapter")

    monkeypatch.setattr(runner_module, "execute_adapter", _must_not_launch)
    finalized = runner.run_candidate_from_output(
        prepared, execution_result, execution_result["output"]
    )

    assert finalized.status == sequential_result.status
    assert finalized.boundary == sequential_result.boundary
    assert finalized.evidence is not None
    assert finalized.evidence.subject == sequential_result.evidence.subject
    assert finalized.evidence.fixture == sequential_result.evidence.fixture
    assert finalized.evidence.observations == sequential_result.evidence.observations

    # The evidence is committed and its run record is a hash-linked artifact.
    committed = finalize_store.evidence[finalized.evidence.id]
    assert committed.result == "pass"
    run_record_path = finalize_store.root / committed.provenance["run_record"]["path"]
    assert run_record_path.is_file()
    assert sha256_file(run_record_path) == committed.provenance["run_record"]["sha256"]
    record = json.loads(run_record_path.read_text(encoding="utf-8"))
    assert record["status"] == "pass"
    assert Path(record["testcase"]) == prepared.testcase_path
    assert "boundary_verified" in [event.event for event in finalize_store.events]

    reloaded = AlignmentStore.load(finalize_store.root)
    assert reloaded.snapshot.boundaries["region:1"]["state"] == "verified"


def test_runner_engine_revision_stale_boundary_persisted_and_reloaded_and_planned(tmp_path: Path) -> None:
    pkg = _make_2region_trace(tmp_path / "trace_2b_stale")
    adapter_path = _make_adapter_file(tmp_path / "adapter_stale.json")
    adapter = EngineAdapter.load(adapter_path)

    store = AlignmentStore.initialize(
        tmp_path / "alignment_2b_stale",
        trace_path=tmp_path / "trace_2b_stale",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    # 1. Run both boundaries under engine rev-1
    res1 = runner.run(limit=1, boundary_ids=["region:1"], engine_revision="rev-1")
    assert res1[0].passed
    res2 = runner.run(limit=1, boundary_ids=["region:2"], engine_revision="rev-1")
    assert res2[0].passed
    assert store.snapshot.status == "verified"
    assert store.snapshot.boundaries["region:1"]["state"] == "verified"
    assert store.snapshot.boundaries["region:2"]["state"] == "verified"

    # 2. Run only region:1 under engine rev-2
    res3 = runner.run(limit=1, boundary_ids=["region:1"], engine_revision="rev-2")
    assert res3[0].passed

    # In memory: region:1 is verified, region:2 is stale, session is aligning
    assert store.snapshot.boundaries["region:1"]["state"] == "verified"
    assert store.snapshot.boundaries["region:2"]["state"] == "stale"
    assert store.snapshot.status == "aligning"

    # 3. Reload from disk: reloaded snapshot must match memory and reflect stale region:2!
    reloaded = AlignmentStore.load(store.root, replay=False)
    assert reloaded.snapshot.boundaries["region:1"]["state"] == "verified"
    assert reloaded.snapshot.boundaries["region:2"]["state"] == "stale"
    assert reloaded.snapshot.status == "aligning"

    # 4. Replay from event log: must also reach identical state!
    replayed = AlignmentStore.load(store.root, replay=True)
    assert replayed.snapshot.boundaries["region:1"]["state"] == "verified"
    assert replayed.snapshot.boundaries["region:2"]["state"] == "stale"
    assert replayed.snapshot.status == "aligning"

    # 5. Planner on reloaded session: region:2 must be planned as candidate (not skipped)!
    from inferref.alignment.planner import AlignmentPlanner

    planner = AlignmentPlanner(pkg)
    plan = planner.plan(session=reloaded.snapshot)
    planned_boundary_ids = [c.boundary for c in plan.candidates]
    assert "region:2" in planned_boundary_ids


def _make_3region_trace(tmp_path: Path) -> TracePackage:
    payloads = tmp_path / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)

    write_array(payloads / "v1.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    write_array(payloads / "v2.irtensor", np.array([2.0, 4.0], dtype=np.float32))
    write_array(payloads / "v3.irtensor", np.array([4.0, 8.0], dtype=np.float32))
    write_array(payloads / "v4.irtensor", np.array([8.0, 16.0], dtype=np.float64))

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
        TensorValueRecord(
            id=3,
            shape=(2,),
            stride=(1,),
            dtype="float32",
            producer=20,
            capture=CaptureInfo(mode="full", payload="payloads/v3.irtensor"),
        ),
        TensorValueRecord(
            id=4,
            shape=(2,),
            stride=(1,),
            dtype="float64",
            producer=30,
            capture=CaptureInfo(mode="full", payload="payloads/v4.irtensor"),
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
        OperatorRecord(
            id=20,
            execution_index=1,
            op="double",
            result=TensorRef(3),
            positional_args=(TensorRef(2),),
            module_stack=(100,),
        ),
        OperatorRecord(
            id=30,
            execution_index=2,
            op="cast_double",
            result=TensorRef(4),
            positional_args=(TensorRef(3),),
            module_stack=(100,),
        ),
    ]
    graph.reindex()
    pkg = TracePackage(
        manifest=Manifest(),
        graph=graph,
        modules=[ModuleRecord(id=100, path="model")],
        regions=[
            RegionRecord(id=1, name="step_1", node_ids=(10,), inputs=(1,), outputs=(2,)),
            RegionRecord(id=2, name="step_2", node_ids=(20,), inputs=(2,), outputs=(3,)),
            RegionRecord(id=3, name="step_3", node_ids=(30,), inputs=(3,), outputs=(4,)),
        ],
        root=tmp_path,
    )
    pkg.save(tmp_path)
    return pkg


def test_runner_unsupported_path_propagates_engine_revision_stale(tmp_path: Path) -> None:
    pkg = _make_3region_trace(tmp_path / "trace_3b")
    # Adapter only supports float32; region:3 has float64 output so it is unsupported!
    adapter_path = _make_adapter_file(
        tmp_path / "adapter_caps.json",
        capabilities={"device_types": ["cpu"], "dtypes": ["float32"], "max_rank": 4},
    )
    adapter = EngineAdapter.load(adapter_path)

    store = AlignmentStore.initialize(
        tmp_path / "alignment_3b",
        trace_path=tmp_path / "trace_3b",
        adapter_path=adapter_path,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        adapter_digest=normalized_adapter_sha256(adapter.to_dict()),
    )
    runner = AlignmentRunner(store, pkg, adapter)

    # 1. Run region:1 and region:2 under engine rev-1
    res1 = runner.run(limit=1, boundary_ids=["region:1"], engine_revision="rev-1")
    assert res1[0].passed
    res2 = runner.run(limit=1, boundary_ids=["region:2"], engine_revision="rev-1")
    assert res2[0].passed
    assert store.snapshot.boundaries["region:1"]["state"] == "verified"
    assert store.snapshot.boundaries["region:2"]["state"] == "verified"

    # 2. Run region:3 under engine rev-2 -> hits unsupported path
    res3 = runner.run(limit=1, boundary_ids=["region:3"], engine_revision="rev-2")
    assert len(res3) == 1
    assert res3[0].status == "unsupported"

    # Region 3 is unsupported, and Region 1 & Region 2 must become STALE!
    assert store.snapshot.boundaries["region:3"]["state"] == "unsupported"
    assert store.snapshot.boundaries["region:1"]["state"] == "stale"
    assert store.snapshot.boundaries["region:2"]["state"] == "stale"
    assert store.snapshot.status == "aligning"

    # 3. Reload from disk: stale states must be persisted on disk!
    reloaded = AlignmentStore.load(store.root, replay=False)
    assert reloaded.snapshot.boundaries["region:1"]["state"] == "stale"
    assert reloaded.snapshot.boundaries["region:2"]["state"] == "stale"
    assert reloaded.snapshot.boundaries["region:3"]["state"] == "unsupported"
    assert reloaded.snapshot.status == "aligning"

    # 4. Planner must not skip region:1 and region:2
    from inferref.alignment.planner import AlignmentPlanner

    planner = AlignmentPlanner(pkg)
    plan = planner.plan(session=reloaded.snapshot)
    candidate_boundaries = [c.boundary for c in plan.candidates]
    assert "region:1" in candidate_boundaries
    assert "region:2" in candidate_boundaries
