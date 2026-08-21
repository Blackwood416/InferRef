"""Static KV-cache correctness, mutation and extraction tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import inferref
from examples.mini_llama.kv_cache import (
    CachedAttention,
    StaticKVCache,
    copy_attention_weights,
)
from examples.mini_llama.model import Attention, MiniLlamaConfig, RotaryEmbedding
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.semantic import apply_detections, detect
from inferref.tensor import codec
from inferref.testcase.extract import ExtractionError, extract_region


@pytest.fixture
def cached_scenario():
    config = MiniLlamaConfig(num_layers=1)
    torch.manual_seed(0)
    reference = Attention(config).eval()
    cached = CachedAttention(config, max_seq_len=8).eval()
    copy_attention_weights(reference, cached)
    torch.manual_seed(1)
    hidden = torch.randn(1, 4, config.hidden_size)
    cos, sin = RotaryEmbedding(config.head_dim)(4)
    return reference, cached, hidden, cos, sin


def _trace_cached(path: Path, cached_scenario) -> TracePackage:
    _reference, cached, hidden, cos, sin = cached_scenario
    with torch.no_grad(), inferref.trace(
        output=path,
        capture_tensors="all",
        model_name="CachedAttention",
    ) as session:
        session.mark_input("prefill", hidden[:, :3])
        cached(hidden[:, :3], cos[:3], sin[:3], 0)
        session.mark_input("decode", hidden[:, 3:])
        decode = cached(hidden[:, 3:], cos[3:], sin[3:], 3)
        session.mark_output("decode_output", decode)
    return TracePackage.load(path)


def test_cached_prefill_and_decode_match_uncached_attention(cached_scenario) -> None:
    reference, cached, hidden, cos, sin = cached_scenario
    with torch.no_grad():
        expected = reference(hidden, cos, sin)
        prefill = cached(hidden[:, :3], cos[:3], sin[:3], 0)
        decode = cached(hidden[:, 3:], cos[3:], sin[3:], 3)
    torch.testing.assert_close(prefill, expected[:, :3])
    torch.testing.assert_close(decode, expected[:, 3:])


def test_cache_capacity_is_checked() -> None:
    config = MiniLlamaConfig(num_layers=1)
    cache = StaticKVCache(config, max_seq_len=2)
    states = torch.zeros(1, config.num_kv_heads, 2, config.head_dim)
    with pytest.raises(ValueError, match="exceeds capacity"):
        cache(states, states, 1)


def test_cache_writes_advance_each_storage_once_per_call(
    tmp_path: Path, cached_scenario
) -> None:
    package = _trace_cached(tmp_path / "trace", cached_scenario)
    writes = [
        op for op in package.graph.ops_in_execution_order()
        if op.canonical_name == "aten.copy_.default"
        and package.module_path(op.module_stack) == "cache"
    ]
    assert len(writes) == 4
    transitions = [
        (mutation.storage_id, mutation.version_before, mutation.version_after)
        for op in writes
        for mutation in op.effects.mutated_storages
    ]
    assert len(transitions) == 4
    by_storage: dict[int, list[tuple[int, int]]] = {}
    for storage_id, before, after in transitions:
        by_storage.setdefault(storage_id, []).append((before, after))
    assert len(by_storage) == 2
    assert all(steps == [(0, 1), (1, 2)] for steps in by_storage.values())


def test_post_mutation_base_values_have_effect_producers(
    tmp_path: Path, cached_scenario
) -> None:
    package = _trace_cached(tmp_path / "trace", cached_scenario)
    writes = [
        op for op in package.graph.ops_in_execution_order()
        if op.canonical_name == "aten.copy_.default"
        and package.module_path(op.module_stack) == "cache"
    ]
    assert writes

    for write in writes:
        explicit_results = set(package.graph.op_output_value_ids(write))
        # The crucial edge is not copy_'s returned target view.  It is another
        # tensor object over the newly written storage generation (usually the
        # base cache) that a later operator reads.  Shape is deliberately not
        # part of this test: torch 2.1 and current lower slices differently.
        effect_values = [
            value
            for value in package.graph.values
            if value.producer == write.id
            and value.id not in explicit_results
            and value.consumers
            and any(
                mutation.storage_id == value.storage_id
                and mutation.version_after == value.storage_version
                for mutation in write.effects.mutated_storages
            )
        ]
        assert effect_values, (
            f"copy_ op:{write.id} did not produce a causal edge for its "
            "post-mutation storage generation"
        )


def test_cache_trace_satisfies_all_ir_invariants(
    tmp_path: Path, cached_scenario
) -> None:
    package = _trace_cached(tmp_path / "trace", cached_scenario)
    errors = [issue for issue in validate_package(package) if issue.severity == "error"]
    assert not errors, "\n".join(str(issue) for issue in errors)


def test_detector_splits_prefill_and_decode_updates(
    tmp_path: Path, cached_scenario
) -> None:
    package = _trace_cached(tmp_path / "trace", cached_scenario)
    detections = [d for d in detect(package) if d.name == "KVCacheUpdate"]
    assert [d.region_name for d in detections] == [
        "KVCacheUpdate@cache#0",
        "KVCacheUpdate@cache#1",
    ]
    assert all(d.method == "source_function" for d in detections)
    assert all(d.confidence == pytest.approx(0.95) for d in detections)


def test_mutation_regions_have_truthful_inputs_and_reproducible_payloads(
    tmp_path: Path, cached_scenario
) -> None:
    package = _trace_cached(tmp_path / "trace", cached_scenario)
    result = apply_detections(package, detect(package))
    regions = [
        region for region in result.regions
        if region.semantic is not None and region.semantic.name == "KVCacheUpdate"
    ]
    assert len(regions) == 2
    assert all(len(region.inputs) == 4 for region in regions)

    for index, region in enumerate(regions):
        testcase = extract_region(package, region, tmp_path / f"case-{index}")
        assert testcase.reproducible
        assert len(testcase.inputs) == len(set(testcase.inputs))
        assert len(testcase.outputs) == len(set(testcase.outputs))

        manifest = json.loads(
            (testcase.path / "testcase.json").read_text(encoding="utf-8")
        )
        for entry in manifest["inputs"] + manifest["outputs"]:
            tensor = codec.read(testcase.path / entry["payload"])
            assert list(tensor.shape) == entry["shape"]


def test_explicit_duplicate_boundary_names_are_rejected(
    tmp_path: Path, cached_scenario
) -> None:
    package = _trace_cached(tmp_path / "trace", cached_scenario)
    result = apply_detections(package, detect(package))
    region = next(
        region for region in result.regions
        if region.semantic is not None and region.semantic.name == "KVCacheUpdate"
    )
    with pytest.raises(ExtractionError, match="names must be unique"):
        extract_region(
            package,
            region,
            tmp_path / "bad-case",
            input_names=["same"] * len(region.inputs),
        )
