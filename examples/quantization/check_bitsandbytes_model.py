"""Read-only audit of real NF4 safetensors against bitsandbytes 0.50.1 on CPU.

Loads one weight at a time, not the model or remote code. Writes only beneath
the fresh output directory: a JSON report and one portable dequantization case.
This validates stored weights, not whole-model inference or GPU Linear kernels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch
from safetensors import safe_open

from inferref.compare.compare import compare_testcase
from inferref.compare.tolerance import TolerancePolicy
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import StringValue, TensorRef
from inferref.quantization import import_bitsandbytes_quant_state
from inferref.quantization.bitsandbytes import QUANT_STATE_PROFILE
from inferref.tensor import codec
from inferref.testcase.extract import extract_operator
from inferref.testcase.validate import validate_testcase


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_example(root, imported, expected, actual):
    root.mkdir()
    item = imported.quantized
    tensors = {1: item.codes, 2: expected, **item.parameters}
    values = []
    for value_id, tensor in tensors.items():
        payload = f"tensors/v{value_id}.irtensor"
        codec.write(root / payload, dtype=tensor.dtype, shape=tensor.shape, stride=tensor.stride,
                    payload=tensor.payload_bytes())
        values.append(TensorValueRecord(id=value_id, dtype=tensor.dtype, shape=tensor.shape, stride=tensor.stride,
                                         producer=0 if value_id == 2 else None,
                                         consumers=(0,) if value_id == 1 else (),
                                         quantization=item.spec if value_id == 1 else None,
                                         capture=CaptureInfo(mode="full", payload=payload)))
    graph = Graph(values=values, operators=[OperatorRecord(
        id=0, execution_index=0, namespace="inferref", op="dequantize_nf4_source",
        positional_args=(TensorRef(1),), keyword_args={"dtype": StringValue(imported.source_dtype)}, result=TensorRef(2))])
    package = TracePackage(manifest=Manifest(format_version="1.0"), graph=graph, root=root)
    package.save(root)
    extracted = extract_operator(package, 0, root.parent / "testcase", input_names=("weight",), output_names=("y",))
    validation = validate_testcase(extracted.path)
    if not validation.valid:
        raise RuntimeError(validation.to_dict())
    engine = root.parent / "bitsandbytes-output"
    codec.write_array(engine / "y.irtensor", actual, dtype=imported.source_dtype)
    comparison = compare_testcase(extracted.path, engine, policy=TolerancePolicy(override_atol=0, override_rtol=0))
    (root.parent / "comparison.json").write_text(json.dumps(comparison.to_dict(), indent=2), encoding="utf-8")
    return comparison.first_failure is None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="local model directory; safetensors only, no remote code")
    parser.add_argument("--output", type=Path, required=True, help="fresh directory outside the source model")
    parser.add_argument("--limit", type=int, default=0, help="number of NF4 weights, 0 means all")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1:
        parser.error("limit must be nonnegative and threads positive")
    model, output = args.model.resolve(), args.output.resolve()
    if output == model or model in output.parents:
        parser.error("output must be outside the read-only source model")
    files = sorted(model.glob("*.safetensors"))
    if not files:
        parser.error("no safetensors files found")
    if bnb.__version__ != "0.50.1":
        parser.error("this audit requires bitsandbytes==0.50.1")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    source_hashes = {path.name: sha256(path) for path in files}
    results = []
    example_pass = False
    available = 0
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            suffix = ".quant_state.bitsandbytes__nf4"
            weights = sorted(k[:-len(suffix)] for k in keys if k.endswith(suffix))
            available += len(weights)
            for name in weights:
                if args.limit and len(results) >= args.limit:
                    break
                packed = handle.get_tensor(name)
                state = bnb.functional.QuantState.from_dict(
                    {k[len(name) + 1:]: handle.get_tensor(k) for k in keys if k.startswith(name + ".")},
                    device=torch.device("cpu"))
                imported = import_bitsandbytes_quant_state(packed, state, profile=QUANT_STATE_PROFILE,
                                                            scale_value_id=10, codebook_value_id=11)
                expected = imported.dequantize_source()
                oracle = bnb.functional.dequantize_4bit(packed, state)
                actual = oracle.view(torch.uint16).numpy() if state.dtype == torch.bfloat16 else oracle.numpy()
                scales = (bnb.functional.dequantize_blockwise(state.absmax, state.state2) + state.offset
                          if state.nested else state.absmax)
                scale_mismatches = int(np.count_nonzero(imported.quantized.parameters[10].data != scales.numpy()))
                mismatches = int(np.count_nonzero(expected.data != actual))
                max_abs = float(np.max(np.abs(expected.as_float32() - oracle.float().numpy()), initial=0))
                if not results:
                    example_pass = save_example(output / "trace", imported, expected, actual)
                results.append({"weight": name, "file": path.name, "shape": list(state.shape),
                                "source_dtype": imported.source_dtype, "nested": state.nested,
                                "numel": expected.numel, "scale_mismatches": scale_mismatches,
                                "weight_mismatches": mismatches, "max_abs_error": max_abs})
                if len(results) % 14 == 0 or mismatches or scale_mismatches:
                    print(f"checked {len(results)} weights; last={name}; mismatches={mismatches}/{scale_mismatches}", flush=True)
    unchanged = source_hashes == {path.name: sha256(path) for path in files}
    passed = bool(results) and unchanged and example_pass and all(
        r["weight_mismatches"] == r["scale_mismatches"] == 0 for r in results)
    report = {"format": "inferref-bitsandbytes-model-audit", "format_version": "1.0",
              "status": "pass" if passed else "fail", "scope": "stored NF4 weights, not full-model inference",
              "profile": QUANT_STATE_PROFILE, "torch": torch.__version__, "bitsandbytes": bnb.__version__,
              "device": "cpu", "model": str(model), "source_sha256": source_hashes, "source_unchanged": unchanged,
              "available_nf4_weights": available, "checked_weights": len(results),
              "checked_elements": sum(r["numel"] for r in results), "portable_case_pass": example_pass,
              "results": results}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"results", "source_sha256"}}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
