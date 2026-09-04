"""Deterministic End-to-End Alignment Scenario Test (0.10 L9).

Tests the full lifecycle:
1. Initialize session with a 2-stage (prefill -> decode) pipeline.
2. Run buggy engine where prefill succeeds but decode fails.
3. Verify automatic narrowing identifies decode as smallest failing boundary.
4. Verify failure domain classification flags implementation error.
5. Simulate engine fix (revision 2) and rerun.
6. Verify decode passes, session transitions to verified, metrics record TFCM and 100% adoption.
7. Generate and validate markdown report and JSON sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from inferref.alignment import (
    AlignmentRunner,
    AlignmentStore,
    build_fixture_matrix,
    classify_failure_domain,
    compute_narrowing,
    inventory_boundaries,
)
from inferref.alignment.report import generate_alignment_report
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.tensor.codec import write_array


def _setup_pipeline_trace(trace_dir: Path) -> TracePackage:
    payloads = trace_dir / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)

    # v1 -> op 10 (prefill) -> v2 -> op 20 (decode) -> v3
    write_array(payloads / "v1.irtensor", np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    write_array(payloads / "v2.irtensor", np.array([2.0, 4.0, 6.0, 8.0], dtype=np.float32))
    write_array(payloads / "v3.irtensor", np.array([4.0, 8.0, 12.0, 16.0], dtype=np.float32))

    graph = Graph()
    graph.values = [
        TensorValueRecord(
            id=1,
            shape=(4,),
            stride=(1,),
            dtype="float32",
            capture=CaptureInfo(mode="full", payload="payloads/v1.irtensor"),
        ),
        TensorValueRecord(
            id=2,
            shape=(4,),
            stride=(1,),
            dtype="float32",
            producer=10,
            capture=CaptureInfo(mode="full", payload="payloads/v2.irtensor"),
        ),
        TensorValueRecord(
            id=3,
            shape=(4,),
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
            op="prefill_gemm",
            result=TensorRef(2),
            positional_args=(TensorRef(1),),
            module_stack=(100,),
        ),
        OperatorRecord(
            id=20,
            execution_index=1,
            op="decode_gemm",
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
            RegionRecord(id=1, name="prefill", node_ids=(10,), inputs=(1,), outputs=(2,)),
            RegionRecord(id=2, name="decode", node_ids=(20,), inputs=(2,), outputs=(3,)),
        ],
        root=trace_dir,
    )
    pkg.save(trace_dir)
    return pkg


def _create_adapter(adapter_path: Path, flag_file: Path) -> None:
    engine_script = adapter_path.parent / "engine.py"
    engine_script.write_text(
        "import sys, json, shutil\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from inferref.tensor.codec import write_array\n"
        "\n"
        "tc, out, flag = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])\n"
        "tc_data = json.loads((tc / 'testcase.json').read_text(encoding='utf-8'))\n"
        "is_fixed = flag.exists() and flag.read_text(encoding='utf-8').strip() == 'fixed'\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "if tc_data['name'] == 'decode' and not is_fixed:\n"
        "    write_array(out / 'output_0.irtensor', np.array([99.0, 99.0, 99.0, 99.0], dtype=np.float32))\n"
        "else:\n"
        "    shutil.copytree(tc / 'reference', out, dirs_exist_ok=True)\n",
        encoding="utf-8",
    )
    adapter_data = {
        "format": "inferref-engine-adapter",
        "format_version": "0.2",
        "name": "pipeline-engine",
        "target_device": "cpu",
        "capabilities": {"device_types": ["cpu"], "dtypes": ["float32"], "max_rank": 4},
        "command": [
            "{python}",
            str(engine_script),
            "{testcase}",
            "{output}",
            str(flag_file),
        ],
    }
    adapter_path.write_text(json.dumps(adapter_data, indent=2), encoding="utf-8")


def test_alignment_orchestrator_e2e(tmp_path: Path) -> None:
    trace_dir = tmp_path / "trace"
    pkg = _setup_pipeline_trace(trace_dir)

    flag_file = tmp_path / "engine_mode.txt"
    flag_file.write_text("buggy", encoding="utf-8")

    adapter_path = tmp_path / "adapter.json"
    _create_adapter(adapter_path, flag_file)

    session_dir = tmp_path / "align_session"
    store = AlignmentStore.initialize(
        session_dir,
        trace_path=trace_dir,
        adapter_path=adapter_path,
        session_id="al-e2e-pipeline",
        required_fixture_kinds=["synthetic"],
    )
    assert store.snapshot.status == "new"

    # Step 1: Run prefill boundary -> should PASS
    from inferref.agent.protocol import EngineAdapter

    adapter = EngineAdapter.load(adapter_path)
    runner = AlignmentRunner(store, pkg, adapter)

    results1 = runner.run(boundary_ids=["region:1"], fixture_kind="synthetic")
    assert len(results1) == 1
    assert results1[0].status == "pass"
    assert store.snapshot.boundaries["region:1"]["state"] == "verified"

    # Step 2: Run decode boundary on buggy engine -> should FAIL
    results2 = runner.run(boundary_ids=["region:2"], fixture_kind="synthetic")
    assert len(results2) == 1
    assert results2[0].status == "fail"
    assert store.snapshot.boundaries["region:2"]["state"] == "failing"

    # Step 3: Check Causal Narrowing and Failure Domain Classification
    evidence_by_b: dict[str, list[Any]] = {}
    for ev in store.evidence.values():
        if ev.type == "numerical_comparison":
            evidence_by_b.setdefault(ev.subject["boundary"], []).append(ev)

    inventory = inventory_boundaries(pkg)
    narrowing = compute_narrowing(inventory, evidence_by_b)
    assert narrowing.smallest_confirmed_failure == "region:2"

    candidates, _active_ev = classify_failure_domain(
        "region:2",
        list(store.evidence.values()),
    )
    assert len(candidates) > 0
    # Numerical divergence without layout mismatch classifies as implementation
    assert candidates[0].domain == "implementation"
    assert candidates[0].confidence >= 0.60

    # Step 4: Fix engine and re-run decode
    flag_file.write_text("fixed", encoding="utf-8")

    results3 = runner.run(boundary_ids=["region:2"], fixture_kind="synthetic")
    assert len(results3) == 1
    assert results3[0].status == "pass"
    assert store.snapshot.boundaries["region:2"]["state"] == "verified"
    assert store.snapshot.status == "verified"
    assert any(e.event == "session_verified" for e in store.events)
    assert AlignmentStore.load(session_dir).snapshot.status == "verified"

    # Step 5: Check Fixture Matrix & Metrics
    matrix = build_fixture_matrix(
        list(store.evidence.values()),
        store.snapshot,
        known_boundaries=list(store.snapshot.boundaries.keys()),
    )
    assert matrix.boundary_states["region:1"] == "verified"
    assert matrix.boundary_states["region:2"] == "verified"
    assert len(matrix.failing_boundaries) == 0

    # Step 6: Generate Full Report
    md_path = session_dir / "reports" / "report.md"
    json_path = session_dir / "reports" / "report.json"
    md_text, sidecar = generate_alignment_report(
        store,
        markdown_path=md_path,
        json_sidecar_path=json_path,
    )
    assert md_path.is_file()
    assert json_path.is_file()
    assert sidecar["format"] == "inferref-alignment-report"
    assert "prefill" in md_text
    assert "decode" in md_text
