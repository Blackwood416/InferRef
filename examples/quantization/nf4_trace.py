"""Explicit flat QuantState import and NF4 W4A16 trace, without Torch."""

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
from inferref.quantization import import_bitsandbytes_nf4, linear_nf4_reference
from inferref.quantization.nf4 import BITSANDBYTES_PROFILE, NF4_CODEBOOK
from inferref.tensor import codec


def build_trace(root: Path) -> TracePackage:
    root.mkdir(parents=True, exist_ok=False)
    indices = np.arange(69, dtype=np.uint8) % 16
    source = indices[::2] << 4
    source[:indices.size // 2] |= indices[1::2]
    imported = import_bitsandbytes_nf4(source, {
        "shape": (3, 23), "dtype": "float16", "quant_type": "nf4", "blocksize": 64,
        "absmax": np.array([1, .5], dtype=np.float32),
        "code": np.array(NF4_CODEBOOK, dtype=np.float32),
        "nested": False, "offset": None, "state2": None,
    }, profile=BITSANDBYTES_PROFILE, scale_value_id=10, codebook_value_id=11)
    x = np.ones((1, 23), dtype=np.float16)
    y = linear_nf4_reference(x, imported.codes, imported.spec, values=imported.parameters)
    tensors = {1: imported.codes, **imported.parameters}
    for value_id, data in ((0, x), (2, y)):
        tensors[value_id] = codec.decode(codec.encode(dtype="float16", shape=data.shape,
                                                       stride=codec.contiguous_stride(data.shape), payload=data.tobytes()))
    values = []
    for value_id, tensor in tensors.items():
        payload = f"tensors/v{value_id}.irtensor"
        codec.write(root / payload, dtype=tensor.dtype, shape=tensor.shape, stride=tensor.stride,
                    payload=tensor.payload_bytes())
        values.append(TensorValueRecord(id=value_id, dtype=tensor.dtype, shape=tensor.shape, stride=tensor.stride,
                                         producer=0 if value_id == 2 else None,
                                         consumers=(0,) if value_id in (0, 1) else (),
                                         quantization=imported.spec if value_id == 1 else None,
                                         capture=CaptureInfo(mode="full", payload=payload)))
    graph = Graph(values=values, operators=[OperatorRecord(
        id=0, execution_index=0, namespace="inferref", op="linear_nf4_w4a16",
        positional_args=(TensorRef(0), TensorRef(1)), result=TensorRef(2))])
    package = TracePackage(manifest=Manifest(format_version="1.0"), graph=graph,
                           regions=[RegionRecord(id=0, name="nf4-linear", node_ids=(0,), inputs=(0, 1), outputs=(2,))], root=root)
    package.save(root)
    return package


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    print(build_trace(parser.parse_args().output).root)
