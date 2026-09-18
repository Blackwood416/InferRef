"""CPU bitsandbytes 0.50.1 Linear4bit adapter; explicit fp16 weight/compute cast.

This is not linear/nf4-w4a16/v1: bnb casts dequantized weights to the compute
dtype before GEMM, whereas that reference contract keeps weights in float32.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import torch

from inferref.ir.paths import resolve_contained_path
from inferref.tensor import codec


def linear(packed, state, x):
    if bnb.__version__ != "0.50.1":
        raise ValueError("adapter requires bitsandbytes==0.50.1")
    module = bnb.nn.Linear4bit(state.shape[1], state.shape[0], bias=False,
                              compute_dtype=torch.float16, quant_type="nf4")
    module.weight = bnb.nn.Params4bit(packed.clone(), requires_grad=False, quant_state=state,
                                     quant_type="nf4", bnb_quantized=True, module=module)
    # Evaluate the public Linear forward in canonical packing, not the separate
    # AVX512-specific repacking/optional downloaded kernel path.
    module.support_avx512bf16_for_cpu = False
    module.eval()
    with torch.inference_mode():
        return module(torch.from_numpy(x.copy())).numpy()


def run(case: Path, output: Path, *, inject_bug=False):
    torch.set_num_threads(2)
    manifest = json.loads((case / "testcase.json").read_text(encoding="utf-8"))
    if len(manifest["inputs"]) != 2 or len(manifest["outputs"]) != 1:
        raise ValueError("expected two Linear inputs and one output")
    weight = next(item for item in manifest["inputs"] if "quantization" in item)
    activation = next(item for item in manifest["inputs"] if item is not weight)
    spec = weight["quantization"]
    if spec["scheme"] != "nf4/v1" or spec["group_size"] != 64:
        raise ValueError("expected NF4 blocksize 64")

    def read(item):
        return codec.read(resolve_contained_path(case, item["payload"])).data.copy()

    codes, x = read(weight).reshape(-1), read(activation)
    if x.dtype != np.float16 or manifest["outputs"][0]["dtype"] != "float16":
        raise ValueError("only explicit float16 compute/output supported")
    packed = codes[::2] << 4
    packed[:codes.size // 2] |= codes[1::2]
    parameters = {item["value_id"]: read(item) for item in manifest["quantization_parameters"]}
    state = bnb.functional.QuantState(shape=torch.Size(spec["logical_shape"]), dtype=torch.float32,
                                      absmax=torch.from_numpy(parameters[spec["scale_value_id"]]),
                                      code=torch.from_numpy(parameters[spec["codebook_value_id"]]),
                                      blocksize=64, quant_type="nf4")
    result = linear(torch.from_numpy(packed.reshape(-1, 1)), state, x)
    if inject_bug:
        result.flat[0] += 1
    codec.write_array(resolve_contained_path(output, manifest["outputs"][0]["name"] + ".irtensor"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testcase", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-bug", action="store_true")
    args = parser.parse_args()
    run(args.testcase, args.output, inject_bug=args.inject_bug)
