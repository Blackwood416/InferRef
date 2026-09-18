"""Independent Torch affine INT8 oracle for the numpy-only reference API."""

import numpy as np
import pytest
import torch

from inferref.quantization import (
    QuantizationSpec,
    dequantize_affine,
    linear_int8_reference,
    quantize_affine,
)
from inferref.tensor import codec


def tensor(values, dtype):
    data = np.asarray(values, dtype=dtype)
    return codec.decode(codec.encode(dtype=dtype, shape=data.shape,
                                    stride=codec.contiguous_stride(data.shape),
                                    payload=data.tobytes()))


@pytest.mark.parametrize("zero_point", [-128, 0, 1, 127])
def test_torch_per_tensor_rounding_saturation_and_dequantization(zero_point):
    data = np.linspace(-160, 160, 641, dtype=np.float32)
    spec = QuantizationSpec(logical_shape=data.shape, scale_value_id=1, zero_point_value_id=2)
    params = {1: tensor(.5, "float32"), 2: tensor(zero_point, "int8")}
    quantized = torch.quantize_per_tensor(torch.tensor(data.tolist()), .5, zero_point, torch.qint8)
    expected = np.asarray(quantized.int_repr().tolist(), dtype=np.int8)
    np.testing.assert_array_equal(quantize_affine(data, spec, values=params), expected)
    actual = dequantize_affine(tensor(expected, "int8"), spec, values=params)
    np.testing.assert_array_equal(actual, np.asarray(quantized.dequantize().tolist(), np.float32))


@pytest.mark.parametrize("axis", [0, 1])
def test_torch_per_channel_weights_and_float32_linear(axis):
    data = np.arange(-9, 9, dtype=np.float32).reshape(3, 6) * .25
    count = data.shape[axis]
    scale = np.array([.25, .5, 1., 2., 4., 8.][:count], dtype=np.float32)
    zero = np.arange(count, dtype=np.int8)
    spec = QuantizationSpec(logical_shape=data.shape, scale_value_id=1, zero_point_value_id=2,
                            granularity="per_axis", axis=axis)
    params = {1: tensor(scale, "float32"), 2: tensor(zero, "int8")}
    quantized = torch.quantize_per_channel(torch.tensor(data.tolist()),
                                          torch.tensor(scale.tolist(), dtype=torch.float64),
                                          torch.tensor(zero.tolist(), dtype=torch.int64), axis,
                                          torch.qint8)
    codes = quantize_affine(data, spec, values=params)
    np.testing.assert_array_equal(codes, np.asarray(quantized.int_repr().tolist(), np.int8))
    inputs = np.arange(12, dtype=np.float32).reshape(2, 6) * .125
    actual = linear_int8_reference(inputs, tensor(codes, "int8"), spec, values=params)
    expected = torch.nn.functional.linear(torch.tensor(inputs.tolist()), quantized.dequantize())
    np.testing.assert_array_equal(actual, np.asarray(expected.tolist(), np.float32))
