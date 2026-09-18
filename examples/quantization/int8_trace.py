"""Create a torch-free affine INT8 Linear trace with explicit per-group scales."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.quantization import (
    QuantizationSpec,
    linear_int4_reference,
    linear_int8_reference,
)
from inferref.tensor import codec


def build_trace(root: Path, *, packed: bool = False, unsigned: bool = False) -> TracePackage:
    root.mkdir(parents=True, exist_ok=False)
    packed = packed or unsigned
    storage_dtype = "uint4" if unsigned else "int4" if packed else "int8"
    arrays = {
        0: np.array([[1, 2, 3]], dtype=np.float16 if packed else np.float32),
        1: np.array([[2, 4, 6], [-2, 0, 2]], dtype=np.int8),
        10: np.array([[.5, .25], [1, 2]], dtype=np.float32),
    }
    if unsigned:
        arrays[1] = (arrays[1] + 8).astype(np.uint8)
        arrays[11] = np.full((2, 2), 8, dtype=np.uint8)
    spec = QuantizationSpec(logical_shape=(2, 3), scale_value_id=10,
                            zero_point_value_id=11 if unsigned else None,
                            scheme=f"affine-{storage_dtype}/v1", storage_dtype=storage_dtype,
                            granularity="per_group", axis=1, group_size=2)
    for value_id, array in arrays.items():
        codec.write_array(root / f"tensors/v{value_id}.irtensor", array,
                          dtype=storage_dtype if value_id == 1 else None)
    reference = linear_int4_reference if packed else linear_int8_reference
    arrays[2] = reference(arrays[0], codec.read(root / "tensors/v1.irtensor"), spec,
                          values={i: codec.read(root / f"tensors/v{i}.irtensor") for i in spec.dependencies})
    codec.write_array(root / "tensors/v2.irtensor", arrays[2])
    values = [TensorValueRecord(
        id=i, dtype=storage_dtype if i == 1 else str(array.dtype), shape=array.shape, stride=codec.contiguous_stride(array.shape),
        producer=0 if i == 2 else None, consumers=(0,) if i in (0, 1) else (),
        quantization=spec if i == 1 else None,
        capture=CaptureInfo(mode="full", payload=f"tensors/v{i}.irtensor"),
    ) for i, array in arrays.items()]
    graph = Graph(values=values, operators=[OperatorRecord(
        id=0, execution_index=0, namespace="inferref", op="linear_affine_w4a16" if packed else "linear_affine_int8",
        positional_args=(TensorRef(0), TensorRef(1)), result=TensorRef(2),
    )])
    package = TracePackage(manifest=Manifest(format_version="1.0"), graph=graph,
                           regions=[RegionRecord(id=0, name="int8-linear", node_ids=(0,), inputs=(0, 1), outputs=(2,))],
                           root=root)
    package.save(root)
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--packed", action="store_true", help="use signed INT4 weights and float16 inputs/outputs")
    parser.add_argument("--unsigned", action="store_true", help="use UINT4 weights and explicit uint8 zero points")
    args = parser.parse_args()
    print(build_trace(args.output, packed=args.packed, unsigned=args.unsigned).root)
