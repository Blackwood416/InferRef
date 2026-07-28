"""Semantic detection against real Hugging Face model code.

The detectors' class-name rules (``LlamaRMSNorm`` -> RMSNorm) are assumptions
about naming conventions. Those assumptions are only worth anything if they
survive contact with code nobody wrote for InferRef, so this module traces an
actual ``transformers`` model.

Skipped when the ``hf`` extra is absent. The model is built from a config with
random weights, so nothing is downloaded and the test stays hermetic.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import inferref
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.semantic import apply_detections, detect

transformers = pytest.importorskip(
    "transformers", reason="real-model detection tests need the hf extra"
)


@pytest.fixture(scope="module")
def hf_trace(tmp_path_factory) -> TracePackage:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.randint(0, 64, (1, 6))

    output = tmp_path_factory.mktemp("hf") / "trace"
    with torch.no_grad(), inferref.trace(
        output=output, capture_tensors="all", model_name="tiny-llama"
    ) as session:
        session.mark_input("input_ids", input_ids)
        session.mark_output("logits", model(input_ids=input_ids, use_cache=False).logits)
    return TracePackage.load(output)


def test_real_llama_classes_are_recognised(hf_trace: TracePackage) -> None:
    """The class-name heuristics resolve transformers' actual class names."""
    detections = detect(hf_trace)
    labelled = {d.name for d in detections}

    assert {"TransformerBlock", "Attention", "RMSNorm", "MLP", "RoPE", "Linear"} <= labelled

    # And they resolved from the real classes, not from something incidental.
    evidence = " ".join(d.evidence for d in detections)
    for class_name in (
        "LlamaDecoderLayer",
        "LlamaAttention",
        "LlamaRMSNorm",
        "LlamaMLP",
        "LlamaRotaryEmbedding",
    ):
        assert class_name in evidence, class_name


def test_real_rope_is_found_by_source_function(hf_trace: TracePackage) -> None:
    """transformers implements RoPE as a free function, not a module."""
    rope = [
        d
        for d in detect(hf_trace)
        if d.name == "RoPE" and d.method == "source_function"
    ]
    assert len(rope) == 2, "one per decoder layer"
    for detection in rope:
        assert "apply_rotary_pos_emb" in detection.evidence
        assert detection.scope.endswith("self_attn")
        assert len(detection.node_ids) > 5


def test_real_model_coverage_is_high(hf_trace: TracePackage) -> None:
    """Most of a real model should get labelled; the rest is the work list."""
    from inferref.inspect.analyze import analyze

    apply_detections(hf_trace, detect(hf_trace))
    result = analyze(hf_trace)

    assert result.semantic_coverage > 0.85, result.semantic_counts
    # What is left over is top-level plumbing (position ids, mask helpers),
    # not an unrecognised kernel — SPEC §25's "unsupported patterns" list.
    assert set(result.unlabelled_modules) <= {"<root>", "model"}


def test_real_model_regions_validate(hf_trace: TracePackage) -> None:
    apply_detections(hf_trace, detect(hf_trace))
    hf_trace.save(hf_trace.root)
    reloaded = TracePackage.load(hf_trace.root)
    errors = [i for i in validate_package(reloaded) if i.severity == "error"]
    assert not errors, "\n".join(str(i) for i in errors)


def test_real_model_rope_testcase_is_extractable(
    hf_trace: TracePackage, tmp_path: Path
) -> None:
    """The end goal: a runnable kernel testcase out of an untouched HF model."""
    from inferref.testcase.extract import extract_region

    apply_detections(hf_trace, detect(hf_trace))
    region = next(
        r for r in hf_trace.regions
        if r.name.startswith("RoPE@") and r.creation_method == "source_function"
    )
    result = extract_region(hf_trace, region, tmp_path / "rope")

    assert result.reproducible, result.missing_payloads
    assert result.inputs and result.outputs
    for name in result.inputs:
        assert (tmp_path / "rope" / "inputs" / f"{name}.irtensor").is_file()
