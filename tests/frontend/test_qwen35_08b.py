"""Opt-in validation against the official Qwen3.5-0.8B checkpoint.

Set ``INFERREF_QWEN35_MODEL`` to a local Hugging Face/ModelScope checkpoint
directory.  Normal CI skips this module's test and never downloads large
weights implicitly.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

import inferref
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.semantic import detect

transformers = pytest.importorskip(
    "transformers", reason="Qwen3.5 integration needs the hf extra"
)
Qwen3_5ForConditionalGeneration = getattr(
    transformers, "Qwen3_5ForConditionalGeneration", None
)
if Qwen3_5ForConditionalGeneration is None:
    pytest.skip(
        "installed transformers does not expose Qwen3.5",
        allow_module_level=True,
    )


def _cache_detections(package: TracePackage, name: str):
    return [item for item in detect(package) if item.name == name]


def test_official_qwen35_08b_hybrid_cache(tmp_path: Path) -> None:
    configured_path = os.environ.get("INFERREF_QWEN35_MODEL")
    if not configured_path:
        pytest.skip("set INFERREF_QWEN35_MODEL to run the 0.8B checkpoint test")
    model_path = Path(configured_path)
    if not model_path.is_dir():
        pytest.fail(f"INFERREF_QWEN35_MODEL is not a directory: {model_path}")

    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    input_ids = torch.tensor([[623, 776, 3812, 33241, 24197]])

    assert sum(parameter.numel() for parameter in model.parameters()) == 852_985_920
    assert model.config.text_config.layer_types.count("linear_attention") == 18
    assert model.config.text_config.layer_types.count("full_attention") == 6
    assert (
        model.lm_head.weight.data_ptr()
        == model.model.language_model.embed_tokens.weight.data_ptr()
    )

    with torch.inference_mode():
        reference = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        ).logits
        prefill = model(
            input_ids=input_ids[:, :4],
            use_cache=True,
            logits_to_keep=1,
        )
        decode = model(
            input_ids=input_ids[:, 4:],
            past_key_values=prefill.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )

    assert torch.allclose(decode.logits, reference, atol=1e-4, rtol=1e-4)
    assert prefill.past_key_values.get_seq_length() == 5
    assert type(prefill.past_key_values.layers[0]).__name__ == "LinearAttentionLayer"
    assert type(prefill.past_key_values.layers[3]).__name__ == "DynamicLayer"

    packages: dict[str, TracePackage] = {}
    for label, scope in {
        "linear": "model.language_model.layers.0",
        "full": "model.language_model.layers.3",
    }.items():
        output = tmp_path / label
        with torch.inference_mode(), inferref.trace(
            output=output,
            scope=scope,
            capture_tensors="metadata",
            model_name="Qwen/Qwen3.5-0.8B",
        ) as session:
            session.mark_input("prefill_input_ids", input_ids[:, :4])
            traced_prefill = model(
                input_ids=input_ids[:, :4],
                use_cache=True,
                logits_to_keep=1,
            )
            session.mark_output("prefill_logits", traced_prefill.logits)
            session.mark_input("decode_input_id", input_ids[:, 4:])
            traced_decode = model(
                input_ids=input_ids[:, 4:],
                past_key_values=traced_prefill.past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            session.mark_output("decode_logits", traced_decode.logits)
        packages[label] = TracePackage.load(output)

    assert len(_cache_detections(packages["linear"], "StateCacheUpdate")) == 4
    assert len(_cache_detections(packages["full"], "KVCacheUpdate")) == 2
    for package in packages.values():
        errors = [
            issue
            for issue in validate_package(package)
            if issue.severity == "error"
        ]
        assert not errors, "\n".join(str(issue) for issue in errors)
