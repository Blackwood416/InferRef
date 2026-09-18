"""Independent numpy adapter for affine INT8 Linear (does not import its oracle)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from inferref.ir.paths import resolve_contained_path
from inferref.tensor import codec


def run(case: Path, output: Path, *, inject_bug: bool = False) -> None:
    manifest = json.loads((case / "testcase.json").read_text(encoding="utf-8"))
    if len(manifest["inputs"]) != 2 or len(manifest["outputs"]) != 1:
        raise ValueError("expected two Linear inputs and one output")
    weights = [item for item in manifest["inputs"] if "quantization" in item]
    if len(weights) != 1:
        raise ValueError("expected one explicitly quantized weight")
    weight = weights[0]
    x_record = next(item for item in manifest["inputs"] if item is not weight)
    spec = weight["quantization"]
    if spec["scheme"] not in {"affine-int8/v1", "affine-int4/v1", "affine-uint4/v1", "nf4/v1"} or spec["compute_dtype"] != "float32":
        raise ValueError("unsupported quantization semantics")

    def read(item):
        return codec.read(resolve_contained_path(case, item["payload"])).data

    raw, x = read(weight), read(x_record)
    parameters = {item["value_id"]: read(item) for item in manifest["quantization_parameters"]}
    scales = parameters[spec["scale_value_id"]]
    zeros = parameters.get(spec["zero_point_value_id"])
    decoded = np.empty(raw.shape, dtype=np.float32)
    for index in np.ndindex(raw.shape):
        if spec["scheme"] == "nf4/v1":
            flat_index = np.ravel_multi_index(index, raw.shape)
            decoded[index] = parameters[spec["codebook_value_id"]][raw[index]] * scales[flat_index // spec["group_size"]]
            continue
        if spec["granularity"] == "per_tensor":
            parameter_index = ()
        elif spec["granularity"] == "per_axis":
            parameter_index = (index[spec["axis"]],)
        elif spec["granularity"] == "per_group":
            position = list(index)
            position[spec["axis"]] //= spec["group_size"]
            parameter_index = tuple(position)
        else:
            raise ValueError("unsupported granularity")
        zero = np.float32(0 if zeros is None else zeros[parameter_index])
        decoded[index] = (np.float32(raw[index]) - zero) * scales[parameter_index]
    result = np.einsum("...i,oi->...o", x.astype(np.float32), decoded, dtype=np.float32)
    result = result.astype(x.dtype)
    if inject_bug:
        result.flat[0] += 1
    output.mkdir(parents=True, exist_ok=True)
    name = manifest["outputs"][0]["name"]
    codec.write_array(resolve_contained_path(output, f"{name}.irtensor"), result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testcase", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inject-bug", action="store_true")
    args = parser.parse_args()
    run(args.testcase, args.output, inject_bug=args.inject_bug)
