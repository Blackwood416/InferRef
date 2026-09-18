"""NF4 codebook/block semantics and an explicit, non-nested bitsandbytes importer.

Profile source: bitsandbytes 0.48.2 functional.QuantState/get_4bit_type and
csrc/kernels.cu high-nibble-first decoding. No bitsandbytes/Torch dependency.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from inferref.quantization.spec import QuantizationError, QuantizationSpec
from inferref.tensor import codec

NF4_CODEBOOK = (
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
)
BITSANDBYTES_PROFILE = "bitsandbytes-0.48.2/nf4-uint8-flat/v1"


def nf4_parameters(spec: QuantizationSpec, values):
    from inferref.quantization.reference import _parameter

    if spec.scheme != "nf4/v1":
        raise QuantizationError("NF4 requires the nf4/v1 descriptor")
    scale = _parameter(values, spec.scale_value_id, "float32", spec.parameter_shape, "block scale")
    if not np.all(np.isfinite(scale) & (scale >= 0)):
        raise QuantizationError("NF4 block scales must be finite and nonnegative")
    book = _parameter(values, spec.codebook_value_id, "float32", (16,), "codebook")
    if not np.array_equal(book, np.asarray(NF4_CODEBOOK, dtype=np.float32)):
        raise QuantizationError("codebook values do not match bitsandbytes-nf4/v1")
    return scale, book


def dequantize_nf4(codes: codec.TensorView, spec: QuantizationSpec, *, values) -> np.ndarray:
    from inferref.quantization.reference import _tensor

    scale, book = nf4_parameters(spec, values)
    indices = _tensor(codes, "uint4", spec.logical_shape, "NF4 indices")
    if np.any(indices > 15):
        raise QuantizationError("NF4 codebook index outside [0, 15]")
    groups = np.arange(indices.size, dtype=np.int64) // spec.group_size
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            return (book[indices.reshape(-1)] * scale[groups]).reshape(spec.logical_shape)
        except FloatingPointError:
            raise QuantizationError("NF4 dequantization overflow in float32") from None


@dataclass(frozen=True)
class ImportedNF4:
    codes: codec.TensorView
    spec: QuantizationSpec
    parameters: Mapping[int, codec.TensorView]


def import_bitsandbytes_nf4(
    packed: np.ndarray, state: Mapping[str, Any], *, profile: str,
    scale_value_id: int, codebook_value_id: int,
) -> ImportedNF4:
    """Import a host projection of non-nested QuantState, never infer a profile.

    Exact keys: absmax/code numpy float32 arrays, shape tuple/list, blocksize=64,
    dtype='float16', quant_type='nf4', nested=False, offset=None, state2=None.
    Source packing is uint8, high nibble first; the unused odd tail nibble is
    discarded. The result is canonical low-first uint4 with zero tail padding.
    """
    if profile != BITSANDBYTES_PROFILE:
        raise QuantizationError("unsupported bitsandbytes importer profile")
    keys = {"absmax", "code", "shape", "blocksize", "dtype", "quant_type", "nested", "offset", "state2"}
    if not isinstance(state, Mapping) or set(state) != keys:
        raise QuantizationError("incomplete or unknown QuantState projection fields")
    if state["nested"] is not False or state["offset"] is not None or state["state2"] is not None:
        raise QuantizationError("nested/compressed NF4 statistics are unsupported")
    if state["dtype"] != "float16" or state["quant_type"] != "nf4" or type(state["blocksize"]) is not int or state["blocksize"] != 64:
        raise QuantizationError("profile requires float16 NF4 with blocksize=64")
    if not isinstance(state["shape"], (list, tuple)):
        raise QuantizationError("QuantState shape must be explicit")
    spec = QuantizationSpec(scheme="nf4/v1", storage_dtype="uint4", logical_shape=tuple(state["shape"]),
                            scale_value_id=scale_value_id, codebook_value_id=codebook_value_id,
                            codebook="bitsandbytes-nf4/v1", granularity="per_block", group_size=64,
                            rounding="none", saturation="none")
    return _import_nf4_codes(packed, spec, state["absmax"], state["code"])


def _import_nf4_codes(packed: np.ndarray, spec: QuantizationSpec, scales, codebook) -> ImportedNF4:
    """Canonicalize validated source semantics; source output dtype is separate."""
    count = 1
    for dim in spec.logical_shape:
        count *= dim
    if not isinstance(packed, np.ndarray) or packed.dtype != np.dtype("uint8") or packed.shape not in {( (count + 1) // 2,), ((count + 1) // 2, 1)}:
        raise QuantizationError("source must be uint8 packed bytes of shape [ceil(numel/2)] or [ceil(numel/2),1]")
    source = packed.reshape(-1)
    indices = np.empty(count, dtype=np.uint8)
    indices[::2] = source >> 4
    indices[1::2] = source[:count // 2] & 15

    def tensor(array, dtype):
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype("float32"):
            raise QuantizationError("QuantState parameters must be explicit float32 numpy arrays")
        return codec.decode(codec.encode(dtype=dtype, shape=array.shape,
                                          stride=codec.contiguous_stride(array.shape), payload=array.tobytes()))

    parameters = {spec.scale_value_id: tensor(scales, "float32"),
                  spec.codebook_value_id: tensor(codebook, "float32")}
    codes = codec.decode(codec.encode(dtype="uint4", shape=spec.logical_shape,
                                      stride=codec.contiguous_stride(spec.logical_shape),
                                      payload=codec.pack_nibbles(indices, "uint4")))
    dequantize_nf4(codes, spec, values=parameters)
    return ImportedNF4(codes, spec, parameters)
