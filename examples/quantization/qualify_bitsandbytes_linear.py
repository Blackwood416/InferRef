"""Qualify CPU NF4 Linear and portable workflows, without model generation/GPU.

Uses either a local read-only safetensors checkpoint or deterministic synthetic
weights (CI). The bnb engine and numpy reference do not share dequantization.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import replace
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
from bitsandbytes_linear import linear
from check_bitsandbytes_model import sha256
from safetensors import safe_open

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import EngineAdapter
from inferref.agent.service import run_engine
from inferref.alignment.hashing import (
    normalized_adapter_sha256,
    trace_metadata_sha256,
    tree_sha256,
)
from inferref.alignment.planner import plan_alignment
from inferref.alignment.runner import AlignmentRunner
from inferref.alignment.store import AlignmentStore
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import StringValue, TensorRef
from inferref.quantization import dequantize_nf4, import_bitsandbytes_quant_state
from inferref.quantization.bitsandbytes import QUANT_STATE_PROFILE
from inferref.suite import run_suite
from inferref.tensor import codec
from inferref.testcase.extract import extract_operator
from inferref.testcase.validate import validate_testcase

EXAMPLE = Path(__file__).resolve().parent
WEIGHTS = ("model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.k_proj.weight",
           "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.down_proj.weight")


def make_trace(root, item, x, y):
    tensors = {1: item.codes, **item.parameters}
    for value_id, data in ((0, x), (2, y)):
        tensors[value_id] = codec.decode(codec.encode(dtype="float16", shape=data.shape,
                                                       stride=codec.contiguous_stride(data.shape), payload=data.tobytes()))
    values = []
    for value_id, tensor in tensors.items():
        payload = f"tensors/v{value_id}.irtensor"
        codec.write(root / payload, dtype=tensor.dtype, shape=tensor.shape, stride=tensor.stride,
                    payload=tensor.payload_bytes())
        values.append(TensorValueRecord(id=value_id, dtype=tensor.dtype, shape=tensor.shape, stride=tensor.stride,
                                         producer=0 if value_id == 2 else None, consumers=(0,) if value_id in (0, 1) else (),
                                         quantization=item.spec if value_id == 1 else None,
                                         capture=CaptureInfo(mode="full", payload=payload)))
    graph = Graph(values=values, operators=[OperatorRecord(id=0, execution_index=0, namespace="inferref",
                  op="bnb_nf4_linear_fp16", positional_args=(TensorRef(0), TensorRef(1)), result=TensorRef(2),
                  keyword_args={"weight_cast": StringValue("float16"), "compute_dtype": StringValue("float16")})])
    package = TracePackage(manifest=Manifest(format_version="1.0"), graph=graph, root=root,
                           regions=[RegionRecord(id=0, name="bnb-linear", node_ids=(0,), inputs=(0, 1), outputs=(2,))])
    package.save(root)
    return package


def qualify(root, name, packed, state, *, workflows):
    root.mkdir(parents=True)
    item = import_bitsandbytes_quant_state(packed, state, profile=QUANT_STATE_PROFILE,
                                           scale_value_id=10, codebook_value_id=11).quantized
    # bnb 0.50.1 CPU canonical GEMM casts weights directly to compute dtype;
    # QuantState.dtype governs standalone dequantize_4bit, not this GEMM path.
    w = dequantize_nf4(item.codes, item.spec, values=item.parameters).astype(np.float16).astype(np.float32)
    rng = np.random.default_rng(101)
    checks = []
    package = None
    for batch_shape in ((1,), (4,), (2, 3)):
        x = (rng.standard_normal((*batch_shape, w.shape[1])) / np.sqrt(w.shape[1])).astype(np.float16)
        reference = np.einsum("...i,oi->...o", x.astype(np.float32), w, dtype=np.float32).astype(np.float16)
        actual = linear(packed, state, x)
        delta = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
        ok = bool(np.all(delta <= .002 + .002 * np.abs(reference.astype(np.float32))))
        checks.append({"input_shape": list(x.shape), "pass": ok, "max_abs_error": float(delta.max(initial=0))})
        if package is None:
            package = make_trace(root / "trace", item, x, reference)
    result = {"weight": name, "shape": list(state.shape), "linear": checks}
    if workflows:
        case = extract_operator(package, 0, root / "extracted", input_names=("x", "weight"), output_names=("y",)).path
        # The engine only sees the relocated standalone tree, not source traces/model.
        relocated = root / "relocated/case"
        shutil.copytree(case, relocated)
        result["relocation_valid"] = validate_testcase(relocated).valid
        adapter_path = EXAMPLE / "bitsandbytes_linear.adapter.json"
        engine = EngineAdapter.load(adapter_path)
        result["agent"] = run_engine(relocated, adapter_path, root / "agent-runs").status
        result["injected_error"] = execute_adapter(relocated, replace(engine, command=(*engine.command, "--inject-bug")),
                                                    root / "negative-runs")["status"]
        suite = root / "suite.json"
        suite.write_text(json.dumps({"format": "inferref-suite", "format_version": "0.1", "name": name,
                         "cases": [{"id": "linear", "testcase": "relocated/case", "tags": []}]}), encoding="utf-8")
        result["suite"] = run_suite(suite, engine, root / "suite-runs")["counts"]["pass"]
        plan = plan_alignment(package, adapter=engine)
        store = AlignmentStore.initialize(root / "alignment", trace_path=package.root, adapter_path=adapter_path,
                     trace_metadata_digest=trace_metadata_sha256(package.root), adapter_digest=normalized_adapter_sha256(engine))
        runner = AlignmentRunner(store, package, engine)
        runs = runner.run(limit=1, engine_revision="bnb-0.50.1-cpu-rc1")
        result["alignment"] = runs[0].status
        result["plan_run_agree"] = runs[0].boundary == plan.candidates[0].boundary
        scale_path = package.root / "tensors/v10.irtensor"
        scales = codec.read(scale_path).data.copy()
        codec.write_array(scale_path, scales * np.float32(1.5))
        try:
            runner.run_candidate(runs[0].boundary, engine_revision="bnb-0.50.1-cpu-rc1")
        except ValueError as exc:
            result["stale_source_rejected"] = "digest mismatch" in str(exc)
        else:
            result["stale_source_rejected"] = False
        codec.write_array(scale_path, scales)
        digest = tree_sha256(relocated)
        codec.write_array(relocated / "quantization_parameters/v10.irtensor", scales * np.float32(1.5))
        result["changed_case_hash"] = tree_sha256(relocated) != digest
        result["changed_scales"] = execute_adapter(relocated, engine, root / "scale-runs")["status"]
        codec.write_array(relocated / "quantization_parameters/v10.irtensor", scales)
    result["pass"] = all(c["pass"] for c in checks) and (not workflows or (
        result["relocation_valid"] and result["agent"] == "pass" and result["suite"] == 1
        and result["alignment"] == "pass" and result["plan_run_agree"] and result["stale_source_rejected"]
        and result["changed_case_hash"] and result["changed_scales"] == result["injected_error"] == "mismatch"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, help="omit for deterministic CI synthetic weights")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.model and (output == args.model.resolve() or args.model.resolve() in output.parents):
        parser.error("output must be outside source model")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(100)
    results = []
    if args.model:
        path = args.model / "model.safetensors"
        digest = sha256(path)
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = handle.keys()
            for index, name in enumerate(WEIGHTS):
                state = bnb.functional.QuantState.from_dict({k[len(name) + 1:]: handle.get_tensor(k)
                         for k in keys if k.startswith(name + ".")}, device=torch.device("cpu"))
                result = qualify(output / str(index), name, handle.get_tensor(name), state, workflows=True)
                results.append(result)
                print(json.dumps(result), flush=True)
        unchanged = digest == sha256(path)
    else:
        packed, state = bnb.functional.quantize_4bit(torch.randn(128, 192, dtype=torch.bfloat16),
                                                     quant_type="nf4", compress_statistics=True)
        results.append(qualify(output / "synthetic", "synthetic", packed, state, workflows=True))
        unchanged = True
    report = {"status": "pass" if unchanged and all(r["pass"] for r in results) else "fail", "results": results,
              "device": "cpu", "bitsandbytes": bnb.__version__, "torch": torch.__version__,
              "source_unchanged": unchanged, "atol": .002, "rtol": .002,
              "scope": "canonical-packing CPU Linear4bit; no AVX512 repacking, GPU or whole-model generation"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
