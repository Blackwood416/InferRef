"""Hermetic Hugging Face hybrid-cache regression coverage.

This uses Transformers' real Qwen3.5 implementation with a tiny random config.
It exercises the same DynamicCache layer classes and pure-PyTorch Gated
DeltaNet fallback as the official 0.8B checkpoint, without downloading model
weights in normal CI.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import pytest
import torch

import inferref
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.semantic import apply_detections, detect

transformers = pytest.importorskip(
    "transformers", reason="hybrid-cache tests need the hf extra"
)
Qwen3_5TextConfig = getattr(transformers, "Qwen3_5TextConfig", None)
Qwen3_5ForCausalLM = getattr(transformers, "Qwen3_5ForCausalLM", None)
if Qwen3_5TextConfig is None or Qwen3_5ForCausalLM is None:
    pytest.skip(
        "installed transformers does not expose Qwen3.5",
        allow_module_level=True,
    )


@dataclass(frozen=True)
class HybridCacheRun:
    package: TracePackage
    max_error: float
    cache_seq_length: int
    layer_types: tuple[str, ...]


def _tiny_qwen35():
    config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=32,
        pad_token_id=0,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(config).eval()


@pytest.fixture(scope="module")
def hybrid_cache_run(tmp_path_factory) -> HybridCacheRun:
    model = _tiny_qwen35()
    input_ids = torch.tensor([[1, 7, 11, 19]])
    prefill_ids = input_ids[:, :3]
    decode_ids = input_ids[:, 3:]

    with torch.no_grad():
        reference = model(input_ids=input_ids, use_cache=False).logits[:, -1:]

    trace_dir = tmp_path_factory.mktemp("hf-hybrid-cache") / "trace"
    with torch.no_grad(), inferref.trace(
        output=trace_dir,
        capture_tensors="metadata",
        model_name="tiny-hf-qwen3.5-hybrid-cache",
    ) as session:
        session.mark_input("prefill_ids", prefill_ids)
        prefill = model(input_ids=prefill_ids, use_cache=True)
        session.mark_output("prefill_logits", prefill.logits)

        session.mark_input("decode_ids", decode_ids)
        decode = model(
            input_ids=decode_ids,
            past_key_values=prefill.past_key_values,
            use_cache=True,
        )
        session.mark_output("decode_logits", decode.logits)

    cache = decode.past_key_values
    return HybridCacheRun(
        package=TracePackage.load(trace_dir),
        max_error=float((decode.logits - reference).abs().max()),
        cache_seq_length=int(cache.get_seq_length()),
        layer_types=tuple(type(layer).__name__ for layer in cache.layers),
    )


def _source_functions(package: TracePackage, op_id: int) -> set[str]:
    source = package.source(package.graph.op(op_id).source_id)
    return set() if source is None else {frame.function for frame in source.stack}


def test_qwen35_hybrid_prefill_decode_matches_uncached(
    hybrid_cache_run: HybridCacheRun,
) -> None:
    assert hybrid_cache_run.max_error < 1e-5
    assert hybrid_cache_run.cache_seq_length == 4
    assert hybrid_cache_run.layer_types == (
        "LinearAttentionLayer",
        "DynamicLayer",
    )


def test_qwen35_state_cache_versions_are_truthful(
    hybrid_cache_run: HybridCacheRun,
) -> None:
    package = hybrid_cache_run.package
    state_mutations = []
    for op in package.graph.ops_in_execution_order():
        functions = _source_functions(package, op.id)
        if (
            functions
            & {
                "update_conv_state",
                "update_recurrent_state",
                "torch_causal_conv1d_update",
            }
            and op.effects.mutated_storages
        ):
            state_mutations.append(op)

    # Transformers 5.15 can avoid an explicit decode-time conv copy while the
    # recurrent state still advances through prefill and decode. Assert the
    # storage-version contract instead of pinning the exact copy count.
    assert len(state_mutations) >= 3
    transitions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for op in state_mutations:
        assert op.canonical_name == "aten.copy_.default"
        assert len(op.effects.mutated_storages) == 1
        mutation = op.effects.mutated_storages[0]
        transitions[mutation.storage_id].append(
            (mutation.version_before, mutation.version_after)
        )

    # Both state storages are traced, each starting at version 0, and every
    # transition is a consecutive version advance.
    assert len(transitions) == 2
    for versions in transitions.values():
        assert versions[0][0] == 0
        for current, following in zip(versions, versions[1:]):
            assert current[1] == following[0]
        assert versions[-1][1] > versions[0][0]


def test_qwen35_hybrid_cache_semantics_cover_both_cache_kinds(
    hybrid_cache_run: HybridCacheRun,
) -> None:
    package = hybrid_cache_run.package
    detections = [
        item
        for item in detect(package)
        if item.name in {"KVCacheUpdate", "StateCacheUpdate"}
    ]
    kv_updates = [item for item in detections if item.name == "KVCacheUpdate"]
    state_updates = [
        item for item in detections if item.name == "StateCacheUpdate"
    ]

    assert {item.scope for item in kv_updates} == {
        "model.layers.1.self_attn#0",
        "model.layers.1.self_attn#1",
    }
    assert {item.scope for item in state_updates} == {
        f"model.layers.0.linear_attn#{index}" for index in range(4)
    }
    assert all("2 tensor concatenation(s)" in item.evidence for item in kv_updates)
    assert all("1 storage mutation(s)" in item.evidence for item in state_updates)

    result = apply_detections(package, detections)
    assert len(result.regions) == 6
    assert not result.skipped
    errors = [
        issue for issue in validate_package(package) if issue.severity == "error"
    ]
    assert not errors, "\n".join(str(issue) for issue in errors)
