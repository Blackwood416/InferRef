"""AlignmentRunner batch execution: one adapter process for K boundaries (0.12)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.protocol import AdapterCapabilities, EngineAdapter
from inferref.alignment.hashing import normalized_adapter_sha256, trace_metadata_sha256
from inferref.alignment.runner import AlignmentRunner, batch_item_id
from inferref.alignment.schema import AlignmentEvent, Evidence
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

# Stand-in runtime bridge: it honours ``--batch-manifest`` the way
# cpp/include/inferref/bridge.hpp does, appends one line per launch so a test can
# count processes, and falls back to the executor's INFERREF_TESTCASE/
# INFERREF_OUTPUT environment for a sequential launch. Behaviour knobs arrive as
# environment variables so no test has to edit the script.
FAKE_BRIDGE = """\
import json
import os
import shutil
import sys
from pathlib import Path

LOG = Path(os.environ["FAKE_LAUNCH_LOG"])
FAIL_NAMES = {value for value in os.environ.get("FAKE_FAIL_NAMES", "").split(",") if value}
MISMATCH_NAMES = {
    value for value in os.environ.get("FAKE_MISMATCH_NAMES", "").split(",") if value
}


def note(line):
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\\n")


def run_item(testcase, output):
    name = json.loads((testcase / "testcase.json").read_text(encoding="utf-8")).get("name", "")
    output.mkdir(parents=True, exist_ok=True)
    if name in FAIL_NAMES:
        return name, False
    if name in MISMATCH_NAMES:
        return name, True
    shutil.copytree(testcase / "reference", output, dirs_exist_ok=True)
    return name, True


def run_batch(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    note("batch:%d" % len(manifest["items"]))
    items = []
    failed = 0
    for item in manifest["items"]:
        name, passed = run_item(Path(item["testcase_dir"]), Path(item["output_dir"]))
        failed += 0 if passed else 1
        items.append(
            {
                "id": item["id"],
                "status": "passed" if passed else "failed",
                "exit_code": 0 if passed else 4,
                "testcase_dir": item["testcase_dir"],
                "output_dir": item["output_dir"],
                "error": None if passed else "fake engine rejected " + name,
            }
        )
    if os.environ.get("FAKE_RESULTS_MODE") != "missing":
        document = {
            "format": "inferref-batch-results",
            "format_version": "1.0",
            "items": items,
            "counts": {
                "total": len(items),
                "passed": len(items) - failed,
                "failed": failed,
            },
        }
        (manifest_path.parent / "batch-results.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
    return int(os.environ.get("FAKE_BATCH_EXIT_CODE", "1" if failed else "0"))


def main(argv):
    if "--batch-manifest" in argv:
        raise SystemExit(run_batch(Path(argv[argv.index("--batch-manifest") + 1])))
    note("sequential")
    _, passed = run_item(
        Path(os.environ["INFERREF_TESTCASE"]), Path(os.environ["INFERREF_OUTPUT"])
    )
    raise SystemExit(0 if passed else 4)


main(sys.argv[1:])
"""


def _make_trace(tmp_path: Path, *, tail_dtype: str = "float32") -> TracePackage:
    payloads = tmp_path / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)
    write_array(payloads / "v1.irtensor", np.array([1.0, 2.0], dtype=np.float32))
    write_array(payloads / "v2.irtensor", np.array([2.0, 4.0], dtype=np.float32))
    write_array(payloads / "v3.irtensor", np.array([4.0, 8.0], dtype=np.float32))
    tail = np.float64 if tail_dtype == "float64" else np.float32
    write_array(payloads / "v4.irtensor", np.array([8.0, 16.0], dtype=tail))

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
            dtype=tail_dtype,
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
            op="cast",
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
            RegionRecord(
                id=1, name="first_step", node_ids=(10,), inputs=(1,), outputs=(2,)
            ),
            RegionRecord(
                id=2, name="second_step", node_ids=(20,), inputs=(2,), outputs=(3,)
            ),
            RegionRecord(
                id=3, name="third_step", node_ids=(30,), inputs=(3,), outputs=(4,)
            ),
        ],
        root=tmp_path,
    )
    pkg.save(tmp_path)
    return pkg


def _bridge_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_bridge.py"
    script.write_text(FAKE_BRIDGE, encoding="utf-8")
    return script


def _capabilities(
    *, features: Sequence[str], dtypes: Sequence[str] = ("float32",)
) -> dict[str, object]:
    return {
        "device_types": ["cpu"],
        "dtypes": list(dtypes),
        "max_rank": 4,
        "features": list(features),
    }


def _batch_adapter(
    tmp_path: Path,
    *,
    log: Path,
    dtypes: Sequence[str] = ("float32",),
) -> EngineAdapter:
    document = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "fake-batch-engine",
        "target_device": "cpu",
        "capabilities": _capabilities(features=("batch_execution",), dtypes=dtypes),
        # A batch command receives --batch-manifest, never a single testcase.
        "command": ["{python}", str(_bridge_script(tmp_path))],
        "environment": {"FAKE_LAUNCH_LOG": str(log)},
    }
    path = tmp_path / "adapter_batch.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return EngineAdapter.load(path)


def _sequential_adapter(tmp_path: Path, *, log: Path) -> EngineAdapter:
    document = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "fake-sequential-engine",
        "target_device": "cpu",
        "capabilities": _capabilities(features=()),
        "command": [
            "{python}",
            str(_bridge_script(tmp_path)),
            "{testcase}",
            "{output}",
        ],
        "environment": {"FAKE_LAUNCH_LOG": str(log)},
    }
    path = tmp_path / "adapter_sequential.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return EngineAdapter.load(path)


def _session(
    tmp_path: Path,
    pkg: TracePackage,
    adapter: EngineAdapter,
    *,
    name: str,
) -> AlignmentStore:
    return AlignmentStore.initialize(
        tmp_path / name,
        trace_path=pkg.root,
        trace_metadata_digest=trace_metadata_sha256(pkg.root),
        target={
            "model": "cpu",
            "engine": "fake-engine",
            "adapter": {
                "path": str(tmp_path / "adapter.json"),
                "normalized_sha256": normalized_adapter_sha256(adapter.to_dict()),
            },
        },
    )


def _launches(log: Path) -> list[str]:
    if not log.is_file():
        return []
    return [line for line in log.read_text(encoding="utf-8").splitlines() if line]


def _evidence_by_boundary(store: AlignmentStore) -> dict[str, list[Evidence]]:
    grouped: dict[str, list[Evidence]] = {}
    for evidence in store.evidence.values():
        if evidence.type == "numerical_comparison":
            grouped.setdefault(evidence.subject["boundary"], []).append(evidence)
    return grouped


def _events(store: AlignmentStore, name: str) -> list[AlignmentEvent]:
    return [event for event in store.events if event.event == name]


def test_limit_three_batches_every_boundary_in_one_launch(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3, boundary_ids=["region:1", "region:2", "region:3"]
    )

    assert [result.boundary for result in results] == [
        "region:1",
        "region:2",
        "region:3",
    ]
    assert [result.status for result in results] == ["pass", "pass", "pass"]
    assert _launches(log) == ["batch:3"]
    grouped = _evidence_by_boundary(store)
    assert sorted(grouped) == ["region:1", "region:2", "region:3"]
    assert all(len(items) == 1 for items in grouped.values())
    assert store.snapshot.status == "verified"
    assert _events(store, "note_recorded") == []

    record_path = store.root / grouped["region:2"][0].provenance["run_record"]["path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["status"] == "pass"
    assert Path(record["output"]).is_dir()
    manifest = json.loads(
        Path(record["execution"]["batch_manifest"]).read_text(encoding="utf-8")
    )
    assert [item["id"] for item in manifest["items"]] == [
        batch_item_id(0, "region:1"),
        batch_item_id(1, "region:2"),
        batch_item_id(2, "region:3"),
    ]


def test_limit_one_stays_sequential(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=1, boundary_ids=["region:1", "region:2"]
    )

    assert [result.status for result in results] == ["pass"]
    assert _launches(log) == ["sequential"]
    assert _events(store, "note_recorded") == []


def test_adapter_without_batch_feature_stays_sequential(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _sequential_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3, boundary_ids=["region:1", "region:2", "region:3"]
    )

    assert [result.status for result in results] == ["pass", "pass", "pass"]
    assert _launches(log) == ["sequential", "sequential", "sequential"]
    assert _events(store, "note_recorded") == []


def test_batch_item_ids_are_pattern_safe_and_collision_free() -> None:
    pattern = r"[A-Za-z0-9][A-Za-z0-9._-]*"
    first = batch_item_id(0, "region:1")
    colliding_slug = batch_item_id(1, "region-1")

    assert re.fullmatch(pattern, first)
    assert first == "0-region-1"
    # Both boundary ids slug to "region-1"; the request index keeps them apart.
    assert colliding_slug != first
    assert re.fullmatch(pattern, colliding_slug)


def test_batch_helper_unsupported_falls_back_to_sequential(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    # A hand-built adapter bypasses load-time validation: it declares the batch
    # feature but keeps the per-testcase placeholders a batch manifest cannot
    # carry, so the helper refuses the batch before any launch.
    adapter = EngineAdapter(
        name="hand-built",
        command=(
            "{python}",
            str(_bridge_script(tmp_path)),
            "{testcase}",
            "{output}",
        ),
        capabilities=AdapterCapabilities(
            device_types=("cpu",),
            dtypes=("float32",),
            max_rank=4,
            features=("batch_execution",),
        ),
        environment={"FAKE_LAUNCH_LOG": str(log)},
    )
    store = _session(tmp_path, pkg, adapter, name="session")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3, boundary_ids=["region:1", "region:2", "region:3"]
    )

    assert [result.status for result in results] == ["pass", "pass", "pass"]
    assert _launches(log) == ["sequential", "sequential", "sequential"]
    notes = _events(store, "note_recorded")
    assert len(notes) == 1
    assert notes[0].data["reason"].startswith("batch fallback: unsupported")
    assert notes[0].data["boundaries"] == ["region:1", "region:2", "region:3"]
    grouped = _evidence_by_boundary(store)
    assert all(len(items) == 1 for items in grouped.values())


def test_batch_results_error_falls_back_to_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")
    monkeypatch.setenv("FAKE_RESULTS_MODE", "missing")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3, boundary_ids=["region:1", "region:2", "region:3"]
    )

    assert [result.status for result in results] == ["pass", "pass", "pass"]
    assert _launches(log) == [
        "batch:3",
        "sequential",
        "sequential",
        "sequential",
    ]
    notes = _events(store, "note_recorded")
    assert len(notes) == 1
    assert notes[0].data["reason"].startswith("batch fallback: results_error")
    grouped = _evidence_by_boundary(store)
    assert all(len(items) == 1 for items in grouped.values())
    run_record = store.root / grouped["region:1"][0].provenance["run_record"]["path"]
    record = json.loads(run_record.read_text(encoding="utf-8"))
    # The committed evidence comes from the sequential launch, not the batch.
    assert "batch_manifest" not in record["execution"]


def test_batch_execution_error_falls_back_to_sequential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")
    # Every item reports passed but the process exits non-zero, so the batch has
    # no trustworthy per-item report.
    monkeypatch.setenv("FAKE_BATCH_EXIT_CODE", "1")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3, boundary_ids=["region:1", "region:2", "region:3"]
    )

    assert [result.status for result in results] == ["pass", "pass", "pass"]
    assert _launches(log) == [
        "batch:3",
        "sequential",
        "sequential",
        "sequential",
    ]
    notes = _events(store, "note_recorded")
    assert len(notes) == 1
    assert notes[0].data["reason"].startswith("batch fallback: execution_error")
    grouped = _evidence_by_boundary(store)
    assert all(len(items) == 1 for items in grouped.values())


def test_partial_failure_continues_when_continue_on_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")
    monkeypatch.setenv("FAKE_MISMATCH_NAMES", "second_step")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3,
        boundary_ids=["region:1", "region:2", "region:3"],
        continue_on_fail=True,
    )

    assert [result.status for result in results] == ["pass", "fail", "pass"]
    assert _launches(log) == ["batch:3"]
    grouped = _evidence_by_boundary(store)
    assert sorted(grouped) == ["region:1", "region:2", "region:3"]
    assert all(len(items) == 1 for items in grouped.values())


def test_partial_failure_stops_finalizing_without_continue_on_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")
    monkeypatch.setenv("FAKE_MISMATCH_NAMES", "second_step")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3,
        boundary_ids=["region:1", "region:2", "region:3"],
        continue_on_fail=False,
    )

    # The bridge already executed all three items, but the sequential contract
    # stops at the first failure and never finalizes the later item.
    assert [result.boundary for result in results] == ["region:1", "region:2"]
    assert [result.status for result in results] == ["pass", "fail"]
    assert _launches(log) == ["batch:3"]
    assert sorted(_evidence_by_boundary(store)) == ["region:1", "region:2"]


def test_engine_rejected_item_is_recorded_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = _make_trace(tmp_path / "trace")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")
    monkeypatch.setenv("FAKE_FAIL_NAMES", "second_step")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3,
        boundary_ids=["region:1", "region:2", "region:3"],
        continue_on_fail=True,
    )

    # A failed item is a trustworthy per-item report: no sequential fallback.
    assert [result.status for result in results] == ["pass", "error", "pass"]
    assert _launches(log) == ["batch:3"]
    assert _events(store, "note_recorded") == []
    failed = _evidence_by_boundary(store)["region:2"][0]
    record = store.root / failed.provenance["run_record"]["path"]
    execution = json.loads(record.read_text(encoding="utf-8"))["execution"]
    assert execution["exit_code"] == 4
    assert execution["error"].startswith("fake engine rejected")


def test_mixed_capability_batches_runnable_boundaries_only(tmp_path: Path) -> None:
    pkg = _make_trace(tmp_path / "trace", tail_dtype="float64")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3,
        boundary_ids=["region:1", "region:2", "region:3"],
        continue_on_fail=True,
    )

    assert [result.boundary for result in results] == [
        "region:1",
        "region:2",
        "region:3",
    ]
    assert [result.status for result in results] == ["pass", "pass", "unsupported"]
    # The unsupported boundary never reaches the batch manifest.
    assert _launches(log) == ["batch:2"]
    grouped = _evidence_by_boundary(store)
    assert sorted(grouped) == ["region:1", "region:2", "region:3"]
    assert all(len(items) == 1 for items in grouped.values())
    assert results[2].evidence is not None
    assert results[2].evidence.result == "unsupported"


def test_mixed_capability_without_continue_on_fail_stays_sequential(
    tmp_path: Path,
) -> None:
    pkg = _make_trace(tmp_path / "trace", tail_dtype="float64")
    log = tmp_path / "launches.txt"
    adapter = _batch_adapter(tmp_path, log=log)
    store = _session(tmp_path, pkg, adapter, name="session")

    results = AlignmentRunner(store, pkg, adapter).run(
        limit=3,
        boundary_ids=["region:1", "region:2", "region:3"],
        continue_on_fail=False,
    )

    # An unsupported boundary finalizes during preparation, so mixing it with a
    # batch that may stop earlier is ambiguous and stays on the sequential path.
    assert [result.status for result in results] == ["pass", "pass", "unsupported"]
    assert _launches(log) == ["sequential", "sequential"]
    assert _events(store, "note_recorded") == []
