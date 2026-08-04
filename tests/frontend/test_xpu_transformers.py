"""Required no-download Transformers coverage for the Windows XPU runner."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

import inferref
from inferref.agent.protocol import ENGINE_ADAPTER_FORMAT, ENGINE_ADAPTER_VERSION
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.semantic import apply_detections, detect
from inferref.suite import run_suite
from inferref.testcase.extract import extract_region

try:
    import transformers
except ImportError:
    transformers = None

REQUIRED = os.environ.get("INFERREF_REQUIRE_XPU_MODELS") == "1"


@pytest.fixture(scope="module")
def xpu() -> str:
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        if REQUIRED:
            pytest.fail("required XPU Transformers job has no usable torch.xpu")
        pytest.skip("XPU Transformers coverage requires an Intel GPU")
    if transformers is None:
        if REQUIRED:
            pytest.fail("required XPU Transformers job has no transformers package")
        pytest.skip("install the hf/qwen35 extras")
    return "xpu"


def _required_class(name: str):
    value = getattr(transformers, name, None) if transformers is not None else None
    if value is None:
        if REQUIRED:
            pytest.fail(f"required transformers class {name} is unavailable")
        pytest.skip(f"transformers does not expose {name}")
    return value


COPY_ENGINE = r"""
import json, shutil, sys
from pathlib import Path
case, output = Path(sys.argv[1]), Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=True)
manifest = json.loads((case / 'testcase.json').read_text(encoding='utf-8'))
for item in manifest['outputs']:
    shutil.copy2(case / item['payload'], output / f"{item['name']}.irtensor")
"""


def test_tiny_llama_trace_extract_and_suite(tmp_path: Path, xpu: str) -> None:
    LlamaConfig = _required_class("LlamaConfig")
    LlamaForCausalLM = _required_class("LlamaForCausalLM")
    torch.manual_seed(7)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    ).eval().to(xpu)
    input_ids = torch.tensor([[1, 7, 11, 19]], device=xpu)
    trace_dir = tmp_path / "llama-trace"
    with torch.no_grad(), inferref.trace(
        output=trace_dir,
        scope="model.layers.0",
        capture_tensors="all",
        semantic_analysis=False,
        model_name="tiny-random-llama-xpu",
    ) as session:
        session.mark_input("input_ids", input_ids)
        logits = model(input_ids=input_ids, use_cache=False).logits
        session.mark_output("logits", logits)

    package = TracePackage.load(trace_dir)
    # Transformers creates two CPU scalar control tensors inside the otherwise
    # XPU graph; the inferred manifest label must report that truthfully.
    assert package.manifest.execution.device == "mixed"
    assert sum(value.device.type == "xpu" for value in package.graph.values) > 80
    assert not [issue for issue in validate_package(package) if issue.severity == "error"]
    detections = detect(package)
    assert {item.name for item in detections} >= {"RMSNorm", "RoPE"}
    applied = apply_detections(package, detections)
    region = next(item for item in applied.regions if item.semantic.name == "RMSNorm")
    testcase_dir = tmp_path / "cases" / "rmsnorm"
    extracted = extract_region(package, region, testcase_dir)
    assert extracted.reproducible

    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps({
        "format": "inferref-suite", "format_version": "0.1", "name": "xpu-tiny-llama",
        "cases": [{"id": "rmsnorm", "testcase": "cases/rmsnorm", "tags": ["xpu", "llama"]}],
    }), encoding="utf-8")
    engine = tmp_path / "copy_engine.py"; engine.write_text(COPY_ENGINE, encoding="utf-8")
    requirements = json.loads((testcase_dir / "testcase.json").read_text())["requirements"]
    adapter = tmp_path / "copy.adapter.json"
    adapter.write_text(json.dumps({
        "format": ENGINE_ADAPTER_FORMAT, "format_version": ENGINE_ADAPTER_VERSION,
        "name": "xpu-suite-protocol-smoke", "target_device": "xpu",
        "capabilities": {"device_types": ["xpu"], "dtypes": requirements["dtypes"], "max_rank": 16, "features": ["multiple_outputs", "strided_inputs", "alias_effects", "mutation_effects"]},
        "command": ["{python}", str(engine), "{testcase}", "{output}"],
    }), encoding="utf-8")
    assert run_suite(suite_path, adapter, tmp_path / "suite-runs")["status"] == "pass"


def test_tiny_qwen35_hybrid_prefill_decode_trace(tmp_path: Path, xpu: str) -> None:
    Config = _required_class("Qwen3_5TextConfig")
    Model = _required_class("Qwen3_5ForCausalLM")
    config = Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=4, linear_num_value_heads=4,
        linear_conv_kernel_dim=4, layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=32, pad_token_id=0, tie_word_embeddings=True,
    )
    torch.manual_seed(11)
    model = Model(config).eval().to(xpu)
    ids = torch.tensor([[1, 7, 11, 19]], device=xpu)
    with torch.no_grad():
        reference = model(ids, use_cache=False).logits[:, -1:]
    trace_dir = tmp_path / "qwen-trace"
    with torch.no_grad(), inferref.trace(
        output=trace_dir, capture_tensors="metadata", model_name="tiny-qwen3.5-xpu"
    ) as session:
        session.mark_input("prefill_ids", ids[:, :3])
        prefill = model(ids[:, :3], use_cache=True)
        session.mark_input("decode_ids", ids[:, 3:])
        decode = model(ids[:, 3:], past_key_values=prefill.past_key_values, use_cache=True)
        session.mark_output("decode_logits", decode.logits)
    torch.xpu.synchronize()
    assert torch.allclose(decode.logits, reference, atol=1e-4, rtol=1e-4)
    package = TracePackage.load(trace_dir)
    labels = {item.name for item in detect(package)}
    assert {"StateCacheUpdate", "KVCacheUpdate"} <= labels
    assert not [issue for issue in validate_package(package) if issue.severity == "error"]
