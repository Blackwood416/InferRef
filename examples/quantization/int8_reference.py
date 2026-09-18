"""Run the standalone INT8 reference API: python examples/quantization/int8_reference.py.

This is not a quantized trace exporter or an engine adapter. It demonstrates
explicit per-group weight semantics and their content fingerprint without Torch.
"""

from __future__ import annotations

import json

import numpy as np

from inferref.quantization import (
    QuantizationSpec,
    dequantize_affine,
    linear_int8_reference,
    quantization_fingerprint,
)
from inferref.tensor import codec


def tensor(values, dtype):
    data = np.asarray(values, dtype=dtype)
    return codec.decode(codec.encode(dtype=dtype, shape=data.shape,
                                    stride=codec.contiguous_stride(data.shape),
                                    payload=data.tobytes()))


def main() -> None:
    weights = tensor([[2, 4, 6], [-2, 0, 2]], "int8")
    spec = QuantizationSpec(logical_shape=(2, 3), scale_value_id=10,
                            granularity="per_group", axis=1, group_size=2)
    parameters = {10: tensor([[.5, .25], [1., 2.]], "float32")}
    inputs = np.array([[1., 2., 3.]], dtype=np.float32)
    decoded = dequantize_affine(weights, spec, values=parameters)
    output = linear_int8_reference(inputs, weights, spec, values=parameters)
    print(json.dumps({
        "descriptor": spec.to_dict(),
        "decoded_weights": decoded.tolist(),
        "linear_output": output.tolist(),
        "fingerprint": quantization_fingerprint(weights, spec, values=parameters),
        "example_scope": "standalone_reference_api",
    }, indent=2))


if __name__ == "__main__":
    main()
