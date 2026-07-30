"""Real Hugging Face KV-cache validation.

The model is a randomly initialised tiny Llama built from configuration, so the
test executes transformers' real model and cache implementations without a
network download. StaticCache is traced through prefill and one-token decode;
DynamicCache is also checked numerically because it is the common eager-mode
default but updates Python references through ``cat`` rather than mutating
tensor storage.
"""

from __future__ import annotations

import inspect
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

import inferref
from inferref.ir.package import TracePackage
from inferref.ir.validate import validate_package
from inferref.semantic import apply_detections, detect
from inferref.tensor import codec
from inferref.testcase.extract import extract_region

transformers = pytest.importorskip(
    "transformers", reason="real-model KV-cache tests need the hf extra"
)
cache_utils = pytest.importorskip("transformers.cache_utils")
DynamicCache = getattr(cache_utils, "DynamicCache", None)
StaticCache = getattr(cache_utils, "StaticCache", None)
if DynamicCache is None or StaticCache is None:  # pragma: no cover - old optional extra
    pytest.skip(
        "installed transformers does not expose DynamicCache and StaticCache",
        allow_module_level=True,
    )


@dataclass(frozen=True)
class HFCacheRun:
    trace_dir: Path
    static_max_error: float
    dynamic_max_error: float
    static_seq_length: int
    dynamic_seq_length: int
    static_cache_is_returned: bool
    dynamic_cache_is_returned: bool


def _new_dynamic_cache(config):
    parameters = inspect.signature(DynamicCache).parameters
    return DynamicCache(config=config) if "config" in parameters else DynamicCache()


def _new_static_cache(config, max_cache_len: int):
    """Construct StaticCache across the supported transformers API shapes."""
    parameters = inspect.signature(StaticCache).parameters
    kwargs = {"config": config, "max_cache_len": max_cache_len}
    if "max_batch_size" in parameters:
        kwargs["max_batch_size"] = 1
    if "device" in parameters:
        kwargs["device"] = "cpu"
    if "dtype" in parameters:
        kwargs["dtype"] = torch.float32
    return StaticCache(**kwargs)


def _argument_value(argument, values: dict[int, np.ndarray]):
    kind = argument["kind"]
    if kind == "tensor":
        return values[argument["value_id"]]
    if kind == "scalar":
        return argument["value"]
    if kind in ("list", "tuple"):
        return [_argument_value(item, values) for item in argument["items"]]
    raise AssertionError(f"unsupported replay argument: {argument}")


def _replay_cache_testcase(path: Path, manifest: dict) -> dict[int, np.ndarray]:
    """Tiny NumPy executor for the physical HF StaticCache region."""
    values = {
        entry["value_id"]: codec.read(path / entry["payload"]).as_comparable().copy()
        for entry in manifest["inputs"]
    }

    for node in manifest["nodes"]:
        name = node["canonical_name"]
        args = [_argument_value(argument, values) for argument in node["positional_args"]]
        result_id = node["result"]["value_id"]

        if name == "aten.zeros.default":
            dtype = node["keyword_args"]["dtype"]["repr"]
            result = np.zeros(args[0], dtype=np.dtype(dtype))
        elif name == "aten.arange.default":
            result = np.arange(args[0], dtype=np.int64)
        elif name in ("aten.add.Tensor", "aten.add_.Tensor"):
            result = np.asarray(args[0] + args[1])
        elif name == "aten.index_copy_.default":
            target, dimension, indices, source = args
            result = target.copy()
            moved = np.moveaxis(result, dimension, 0)
            moved_source = np.moveaxis(source, dimension, 0)
            moved[np.asarray(indices, dtype=np.int64)] = moved_source
        else:  # pragma: no cover - points directly at an upstream cache change
            raise AssertionError(f"unsupported StaticCache operator {name}")
        values[result_id] = result
    return values


@pytest.fixture(scope="module")
def hf_cache_run(tmp_path_factory) -> HFCacheRun:
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).eval()
    input_ids = torch.tensor([[1, 7, 11, 19]])
    prefill_ids = input_ids[:, :3]
    decode_ids = input_ids[:, 3:]

    with torch.no_grad():
        reference = model(input_ids=input_ids, use_cache=False).logits

        dynamic_cache = _new_dynamic_cache(config)
        dynamic_prefill = model(
            input_ids=prefill_ids,
            past_key_values=dynamic_cache,
            use_cache=True,
        )
        dynamic_decode = model(
            input_ids=decode_ids,
            past_key_values=dynamic_cache,
            use_cache=True,
        )
        dynamic_logits = torch.cat(
            [dynamic_prefill.logits, dynamic_decode.logits], dim=1
        )

    static_cache = _new_static_cache(config, max_cache_len=8)
    trace_dir = tmp_path_factory.mktemp("hf-kv-cache") / "trace"
    with torch.no_grad(), inferref.trace(
        output=trace_dir,
        capture_tensors="all",
        model_name="tiny-hf-llama-static-cache",
    ) as session:
        session.mark_input("prefill_ids", prefill_ids)
        static_prefill = model(
            input_ids=prefill_ids,
            past_key_values=static_cache,
            use_cache=True,
        )
        session.mark_output("prefill_logits", static_prefill.logits)

        session.mark_input("decode_ids", decode_ids)
        static_decode = model(
            input_ids=decode_ids,
            past_key_values=static_cache,
            use_cache=True,
        )
        session.mark_output("decode_logits", static_decode.logits)

    static_logits = torch.cat([static_prefill.logits, static_decode.logits], dim=1)
    return HFCacheRun(
        trace_dir=trace_dir,
        static_max_error=float((static_logits - reference).abs().max()),
        dynamic_max_error=float((dynamic_logits - reference).abs().max()),
        static_seq_length=int(static_cache.get_seq_length()),
        dynamic_seq_length=int(dynamic_cache.get_seq_length()),
        static_cache_is_returned=(static_decode.past_key_values is static_cache),
        dynamic_cache_is_returned=(dynamic_decode.past_key_values is dynamic_cache),
    )


def test_real_hf_prefill_decode_matches_uncached(hf_cache_run: HFCacheRun) -> None:
    assert hf_cache_run.static_max_error < 1e-5
    assert hf_cache_run.dynamic_max_error < 1e-5
    assert hf_cache_run.static_seq_length == 4
    assert hf_cache_run.dynamic_seq_length == 4
    assert hf_cache_run.static_cache_is_returned
    assert hf_cache_run.dynamic_cache_is_returned


def test_real_static_cache_records_truthful_mutations(hf_cache_run: HFCacheRun) -> None:
    package = TracePackage.load(hf_cache_run.trace_dir)
    mutated = [
        op
        for op in package.graph.ops_in_execution_order()
        if op.effects.mutated_storages
    ]

    assert len(mutated) == 12
    assert sum(op.canonical_name == "aten.add_.Tensor" for op in mutated) == 4
    assert sum(op.canonical_name == "aten.index_copy_.default" for op in mutated) == 8

    transitions: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for op in mutated:
        assert len(op.effects.mutated_storages) == 1
        mutation = op.effects.mutated_storages[0]
        transitions[mutation.storage_id].append(
            (mutation.version_before, mutation.version_after)
        )

    # Per layer: cumulative length, key cache and value cache. Prefill and
    # decode each advance every storage exactly once.
    assert len(transitions) == 6
    assert all(versions == [(0, 1), (1, 2)] for versions in transitions.values())


def test_real_static_cache_is_detected_per_layer_and_phase(
    hf_cache_run: HFCacheRun,
) -> None:
    package = TracePackage.load(hf_cache_run.trace_dir)
    detections = [d for d in detect(package) if d.name == "KVCacheUpdate"]

    assert len(detections) == 4
    assert {d.scope for d in detections} == {
        "model.layers.0.self_attn#0",
        "model.layers.0.self_attn#1",
        "model.layers.1.self_attn#0",
        "model.layers.1.self_attn#1",
    }
    assert {d.method for d in detections} == {"semantic_pattern"}
    assert {d.confidence for d in detections} == {0.95}
    assert all("3 storage mutation(s)" in d.evidence for d in detections)
    for detection in detections:
        operators = [package.graph.op(node_id) for node_id in detection.node_ids]
        assert sum(len(op.effects.mutated_storages) for op in operators) == 3
        assert sum(op.op == "index_copy_" for op in operators) == 2


def test_real_static_cache_regions_validate(hf_cache_run: HFCacheRun) -> None:
    package = TracePackage.load(hf_cache_run.trace_dir)
    detections = [d for d in detect(package) if d.name == "KVCacheUpdate"]
    result = apply_detections(package, detections)

    assert len(result.regions) == 4
    assert not result.skipped
    errors = [issue for issue in validate_package(package) if issue.severity == "error"]
    assert not errors, "\n".join(str(issue) for issue in errors)


def test_real_static_cache_testcases_are_reproducible(
    hf_cache_run: HFCacheRun, tmp_path: Path
) -> None:
    package = TracePackage.load(hf_cache_run.trace_dir)
    detections = [d for d in detect(package) if d.name == "KVCacheUpdate"]
    regions = apply_detections(package, detections).regions

    for index, region in enumerate(regions):
        testcase = extract_region(package, region, tmp_path / f"cache-{index}")
        assert testcase.reproducible, testcase.missing_payload_details
        manifest = json.loads(
            (testcase.path / "testcase.json").read_text(encoding="utf-8")
        )
        assert manifest["reproducible"] is True
        assert len(manifest["outputs"]) == 3
        for entry in manifest["inputs"] + manifest["outputs"]:
            tensor = codec.read(testcase.path / entry["payload"])
            assert list(tensor.shape) == entry["shape"]

        replayed = _replay_cache_testcase(testcase.path, manifest)
        for entry in manifest["outputs"]:
            reference = codec.read(
                testcase.path / entry["payload"]
            ).as_comparable()
            assert np.array_equal(replayed[entry["value_id"]], reference)
