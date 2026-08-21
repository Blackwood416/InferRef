"""Framework-independent tensor payload storage (``.irtensor``).

Depends on numpy only — never on PyTorch.
"""

from inferref.tensor.codec import (
    FLAG_CANONICAL_CONTIGUOUS,
    MAGIC,
    IRTensorError,
    TensorView,
    contiguous_stride,
    decode,
    encode,
    header_size_for_rank,
    read,
    read_stream,
    write,
    write_array,
)
from inferref.tensor.dtype_numpy import (
    bfloat16_bytes_to_float32,
    float32_to_bfloat16_bytes,
    inferref_dtype_for_numpy,
    numpy_dtype_for,
    to_comparable_float,
)

__all__ = [
    "FLAG_CANONICAL_CONTIGUOUS",
    "MAGIC",
    "IRTensorError",
    "TensorView",
    "bfloat16_bytes_to_float32",
    "contiguous_stride",
    "decode",
    "encode",
    "float32_to_bfloat16_bytes",
    "header_size_for_rank",
    "inferref_dtype_for_numpy",
    "numpy_dtype_for",
    "read",
    "read_stream",
    "to_comparable_float",
    "write",
    "write_array",
]
