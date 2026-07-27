"""Trace IR v0.1 acceptance criteria 8 and 9 (IR §57).

8. a standalone operator testcase
9. a multi-op RoPE reference region
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import inferref
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.region.boundary import derive_boundary, nodes_between
from inferref.region.manager import (
    RegionError,
    create_region,
    create_region_from_ops,
    create_region_from_source_function,
)
from inferref.tensor import codec
from inferref.testcase.extract import ExtractionError, extract_operator, extract_region


@pytest.fixture
def traced(trace_dir: Path, mini_llama, mini_llama_input) -> TracePackage:
    with torch.no_grad(), inferref.trace(
        output=trace_dir, capture_tensors="all", model_name="MiniLlama"
    ) as session:
        session.mark_input("hidden_states", mini_llama_input)
        session.mark_output("out", mini_llama(mini_llama_input))
    return TracePackage.load(trace_dir)


# -- criterion 8: a standalone operator testcase ---------------------------


def test_extract_single_operator(traced: TracePackage, tmp_path: Path) -> None:
    # Llama-style Linears are bias-free, so they dispatch as t + mm.
    mm = next(
        op for op in traced.graph.ops_in_execution_order()
        if op.canonical_name == "aten.mm.default"
    )
    out = tmp_path / "op"
    result = extract_operator(traced, mm.id, out)

    assert result.reproducible
    assert (out / "testcase.json").is_file()
    assert (out / "README.md").is_file()

    manifest = json.loads((out / "testcase.json").read_text(encoding="utf-8"))
    assert manifest["format"] == "inferref-testcase"
    assert manifest["format_version"] == "0.1"
    assert manifest["origin"]["operator_id"] == mm.id
    assert len(manifest["inputs"]) == 2         # activation, weight^T
    assert len(manifest["outputs"]) == 1

    # Every declared payload exists and decodes to the declared metadata.
    for entry in manifest["inputs"] + manifest["outputs"]:
        view = codec.read(out / entry["payload"])
        assert list(view.shape) == entry["shape"]
        assert view.dtype == entry["dtype"]
        assert list(view.stride) == entry["stride"]


def test_extracted_operator_reproduces_reference(
    traced: TracePackage, tmp_path: Path
) -> None:
    """The point of a testcase: recompute it standalone and match."""
    mm = next(
        op for op in traced.graph.ops_in_execution_order()
        if op.canonical_name == "aten.mm.default"
    )
    out = tmp_path / "op"
    extract_operator(traced, mm.id, out, input_names=["mat1", "mat2"])
    manifest = json.loads((out / "testcase.json").read_text(encoding="utf-8"))

    loaded = {
        entry["name"]: codec.read(out / entry["payload"]).as_comparable()
        for entry in manifest["inputs"]
    }
    reference = codec.read(out / manifest["outputs"][0]["payload"]).as_comparable()

    # mm(mat1, mat2) == mat1 @ mat2, recomputed with numpy alone.
    actual = loaded["mat1"] @ loaded["mat2"]
    assert np.allclose(actual, reference, atol=1e-5, rtol=1e-5)


def test_extract_rejects_unknown_operator(traced: TracePackage, tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="no operator with id"):
        extract_operator(traced, 99999, tmp_path / "nope")


def test_testcase_without_payloads_is_flagged(
    trace_dir: Path, mini_llama, mini_llama_input, tmp_path: Path
) -> None:
    """A metadata-only trace cannot yield a runnable testcase, and says so."""
    with torch.no_grad(), inferref.trace(
        output=trace_dir, capture_tensors="metadata"
    ) as session:
        session.mark_output("out", mini_llama(mini_llama_input))
    package = TracePackage.load(trace_dir)

    op = package.graph.ops_in_execution_order()[0]
    result = extract_operator(package, op.id, tmp_path / "tc")

    assert not result.reproducible
    assert result.missing_payloads
    manifest = json.loads((tmp_path / "tc" / "testcase.json").read_text(encoding="utf-8"))
    assert manifest["reproducible"] is False
    assert "missing_payloads" in manifest
    assert "not independently reproducible" in (tmp_path / "tc" / "README.md").read_text(
        encoding="utf-8"
    )


# -- criterion 9: a multi-op RoPE reference region --------------------------


def test_rope_region_boundary(traced: TracePackage) -> None:
    region = create_region_from_source_function(
        traced, "RoPE_all", "apply_rotary_pos_emb", semantic="RoPE"
    )
    # SPEC §51: the reference region is slice/neg/cat/mul/mul/add per call.
    names = {traced.graph.op(n).canonical_name for n in region.node_ids}
    assert {"aten.mul.Tensor", "aten.neg.default", "aten.cat.default"} <= names

    assert region.semantic is not None
    assert region.semantic.name == "RoPE"
    assert region.creation_method == "source_function"


def test_single_layer_rope_region_has_four_inputs_two_outputs(
    traced: TracePackage,
) -> None:
    """One RoPE call: (query, key, cos, sin) -> (q_embed, k_embed) (SPEC §4.3)."""
    rope_ops = [
        op
        for op in traced.graph.ops_in_execution_order()
        if (src := traced.source(op.source_id)) is not None
        and src.primary is not None
        and src.primary.function == "apply_rotary_pos_emb"
    ]
    assert rope_ops

    # Layer 0's invocation is the first contiguous execution run. Take the whole
    # execution range so the helper calls inlined inside it (rotate_half) are
    # included — a region must be a contiguous slice of execution to have a
    # clean boundary.
    start = rope_ops[0].execution_index
    last = next(
        op for op in reversed(rope_ops) if op.execution_index < start + 30
    )
    nodes = nodes_between(traced.graph, rope_ops[0].id, last.id)
    region = create_region(traced, "RoPE_layer0", nodes, semantic="RoPE")

    assert len(region.inputs) == 4, [
        traced.graph.value(v).shape for v in region.inputs
    ]
    assert len(region.outputs) == 2

    inputs = [traced.graph.value(v) for v in region.inputs]
    outputs = [traced.graph.value(v) for v in region.outputs]
    # cos/sin are rank-2 tables; query/key are rank-4 (batch, heads, seq, dim).
    assert sorted(v.rank for v in inputs) == [2, 2, 4, 4]
    assert all(v.rank == 4 for v in outputs)

    # SPEC §51: the reference region really is slice/neg/cat/mul/mul/add.
    names = {traced.graph.op(n).canonical_name for n in region.node_ids}
    assert {
        "aten.slice.Tensor",
        "aten.neg.default",
        "aten.cat.default",
        "aten.mul.Tensor",
        "aten.add.Tensor",
    } <= names


def test_source_function_region_can_be_non_contiguous(traced: TracePackage) -> None:
    """A source-function region skips ops issued by inlined helpers.

    ``apply_rotary_pos_emb`` calls ``rotate_half``; operators attributed to the
    helper carry the helper's source location, so selecting purely by function
    name yields a node set with holes and therefore extra boundary inputs.
    Documented because it is the reason ``--from-op/--to-op`` exists.
    """
    region = create_region_from_source_function(
        traced, "RoPE_fn", "apply_rotary_pos_emb", semantic="RoPE"
    )
    indices = sorted(traced.graph.op(n).execution_index for n in region.node_ids)
    has_gap = any(b - a > 1 for a, b in zip(indices, indices[1:]))
    assert has_gap

    # The boundary is still derived correctly and self-consistently (IR §34).
    inputs, outputs = derive_boundary(traced.graph, region.node_ids)
    assert list(region.inputs) == inputs
    assert list(region.outputs) == outputs


def test_region_boundary_matches_derivation(traced: TracePackage) -> None:
    """IR §34: recorded boundary must equal the derived one (invariant 9)."""
    region = create_region_from_ops(traced, "Head", 1, 12)
    inputs, outputs = derive_boundary(traced.graph, region.node_ids)
    assert list(region.inputs) == inputs
    assert list(region.outputs) == outputs

    traced.save_regions()
    reloaded = TracePackage.load(traced.root)
    errors = [i for i in validate_package(reloaded) if i.severity == "error"]
    assert not errors, "\n".join(str(i) for i in errors)


def test_region_testcase_roundtrip(traced: TracePackage, tmp_path: Path) -> None:
    region = create_region_from_ops(traced, "Slice", 1, 8)
    out = tmp_path / "region"
    result = extract_region(traced, region, out)

    assert result.reproducible
    manifest = json.loads((out / "testcase.json").read_text(encoding="utf-8"))
    assert manifest["origin"]["region_id"] == region.id
    assert len(manifest["nodes"]) == len(region.node_ids)
    assert [e["name"] for e in manifest["inputs"]] == result.inputs
    assert [e["name"] for e in manifest["outputs"]] == result.outputs


def test_explicit_boundary_names(traced: TracePackage, tmp_path: Path) -> None:
    region = create_region_from_ops(traced, "Named", 1, 8)
    inputs, outputs = derive_boundary(traced.graph, region.node_ids)
    names = [f"in{i}" for i in range(len(inputs))]
    out_names = [f"out{i}" for i in range(len(outputs))]

    result = extract_region(
        traced, region, tmp_path / "named", input_names=names, output_names=out_names
    )
    assert result.inputs == names
    assert result.outputs == out_names
    for name in names:
        assert (tmp_path / "named" / "inputs" / f"{name}.irtensor").is_file()


def test_wrong_name_count_is_rejected(traced: TracePackage, tmp_path: Path) -> None:
    region = create_region_from_ops(traced, "Bad", 1, 8)
    with pytest.raises(ExtractionError, match="name"):
        extract_region(traced, region, tmp_path / "bad", input_names=["only_one"])


def test_duplicate_region_name_is_rejected(traced: TracePackage) -> None:
    create_region_from_ops(traced, "Dup", 1, 4)
    with pytest.raises(RegionError, match="already exists"):
        create_region_from_ops(traced, "Dup", 5, 8)


def test_empty_region_is_rejected(traced: TracePackage) -> None:
    with pytest.raises(RegionError, match="no known operators"):
        create_region(traced, "Empty", [99999])
