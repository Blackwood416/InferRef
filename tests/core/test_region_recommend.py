"""Tests for region preview (--details) and recommendation ranking (SPEC §10)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferref.cli.main import EXIT_OK, main
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest, ModelInfo
from inferref.ir.operator import Annotation, OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.region.analysis import analyze_region
from inferref.region.recommend import recommend_regions


@pytest.fixture
def sample_package(tmp_path: Path) -> TracePackage:
    graph = Graph()
    # Boundary input without producer -> estimated parameter
    v_param = TensorValueRecord(
        id=1,
        dtype="float32",
        shape=(10, 10),
        producer=None,
        capture=CaptureInfo(mode="full", payload="payload_param.irtensor"),
    )
    # Boundary input with producer outside (external graph input)
    v_act = TensorValueRecord(
        id=2,
        dtype="float32",
        shape=(10, 10),
        producer=100,
        capture=CaptureInfo(mode="full", payload="payload_act.irtensor"),
    )
    # Output produced inside
    v_out = TensorValueRecord(
        id=3,
        dtype="float32",
        shape=(10, 10),
        producer=10,
        capture=CaptureInfo(mode="full", payload="payload_out.irtensor"),
    )
    op = OperatorRecord(
        id=10,
        namespace="aten",
        op="linear",
        overload="default",
        execution_index=0,
        result=TensorRef(v_out.id),
        positional_args=(TensorRef(v_act.id), TensorRef(v_param.id)),
    )
    graph.values = [v_param, v_act, v_out]
    graph.operators = [op]
    graph.reindex()

    reg = RegionRecord(
        id=0,
        name="LinearBlock",
        node_ids=(10,),
        inputs=(1, 2),
        outputs=(3,),
        semantic=Annotation(type="semantic", name="Linear"),
    )
    manifest = Manifest(model=ModelInfo(name="test_model"))
    package = TracePackage(
        root=tmp_path,
        manifest=manifest,
        graph=graph,
        regions=[reg],
    )
    return package


def test_analyze_region_details(sample_package: TracePackage):
    region = sample_package.regions[0]
    details = analyze_region(sample_package, region)
    assert details.operators == 1
    assert details.inputs == 2
    assert details.outputs == 1
    # 10*10 float32 = 400 bytes each
    assert details.parameter_bytes == 400
    assert details.activation_bytes == 800  # 1 input + 1 output
    assert details.largest_tensor == 400
    assert details.payload_coverage == 1.0
    assert details.reproducible is True
    assert details.mutation is False

    d_dict = details.to_dict()
    assert d_dict["parameter_bytes"] == {"estimated": 400}


def test_recommend_regions(sample_package: TracePackage):
    recommendations = recommend_regions(sample_package)
    assert len(recommendations) == 1
    rec = recommendations[0]
    assert rec.region_id == 0
    assert rec.score > 0
    assert any("+30: complete payload coverage" in r for r in rec.reasons)


def test_cli_region_list_details_and_recommend(sample_package: TracePackage, capsys):
    sample_package.save(sample_package.root)
    # Test region list --details --json
    assert main(["region", "list", str(sample_package.root), "--details", "--json"]) == EXIT_OK
    data = json.loads(capsys.readouterr().out)
    assert "regions" in data
    assert len(data["regions"]) == 1
    assert data["regions"][0]["parameter_bytes"] == {"estimated": 400}

    # Test region recommend --json
    assert main(["region", "recommend", str(sample_package.root), "--json"]) == EXIT_OK
    rec_data = json.loads(capsys.readouterr().out)
    assert rec_data["format"] == "inferref-region-recommendations"
    assert len(rec_data["recommendations"]) == 1
