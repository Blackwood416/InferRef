"""Explicit bitsandbytes 0.50.1 QuantState bridge.

The core projection is numpy-only. The object bridge imports optional Torch
and bitsandbytes only when called. One level of compressed scale statistics is
expanded into ordinary float32 parameters; nested storage is not an IR scheme.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from inferref.quantization.nf4 import ImportedNF4, _import_nf4_codes, dequantize_nf4
from inferref.quantization.spec import QuantizationError, QuantizationSpec
from inferref.tensor import codec
from inferref.tensor.dtype_numpy import float32_to_bfloat16_bytes

QUANT_STATE_PROFILE = "bitsandbytes-0.50.1/nf4-quant-state/v1"
_KEYS = {"absmax", "code", "shape", "blocksize", "dtype", "quant_type", "nested", "offset", "state2"}


@dataclass(frozen=True)
class ImportedQuantState:
    """Canonical NF4 plus source rounding/provenance, not a Linear contract.

    ``quantized`` has float32 dequantization semantics. ``source_dtype`` controls
    the separate :meth:`dequantize_source` operation matching bnb's output cast.
    Serialize a rounded output explicitly when constructing a testcase; never
    hide this required cast in optional trace extensions.
    """

    quantized: ImportedNF4
    source_dtype: str
    source_nested: bool
    profile: str = QUANT_STATE_PROFILE

    def dequantize_source(self) -> codec.TensorView:
        item = self.quantized
        data = dequantize_nf4(item.codes, item.spec, values=item.parameters)
        if self.source_dtype == "bfloat16":
            data = float32_to_bfloat16_bytes(data)
        elif self.source_dtype == "float16":
            with np.errstate(over="raise", invalid="raise", under="ignore"):
                try:
                    data = data.astype(np.float16)
                except FloatingPointError:
                    raise QuantizationError("source float16 dequantization overflow") from None
        elif self.source_dtype != "float32":
            raise QuantizationError("unsupported source dtype")
        result = codec.decode(codec.encode(dtype=self.source_dtype, shape=item.spec.logical_shape,
                                            stride=codec.contiguous_stride(item.spec.logical_shape),
                                            payload=data.tobytes()))
        if not np.all(np.isfinite(result.as_float32())):
            raise QuantizationError("source dequantization overflow")
        return result


def _array(value, dtype, shape, label):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype) or value.shape != shape:
        raise QuantizationError(f"{label} must be {dtype} with shape {shape}")
    if not np.all(np.isfinite(value)):
        raise QuantizationError(f"{label} must be finite")
    return value


def import_bitsandbytes_nf4_projection(
    packed: np.ndarray, state: Mapping[str, Any], *, profile: str,
    scale_value_id: int, codebook_value_id: int,
) -> ImportedQuantState:
    """Import exact host fields from the explicit 0.50.1 QuantState profile.

    Main blocksize is 64. Source dtype is float16/bfloat16/float32. For nested
    states, state2 has the same nine keys: float32, blocksize 256, no shape/type,
    no further nesting. Its 256-entry finite codebook is carried explicitly.
    Offset is a scalar float32 ndarray. Arithmetic is separate float32 multiply
    then add (not fused). The source objects/arrays are never modified.
    """
    if profile != QUANT_STATE_PROFILE:
        raise QuantizationError("unsupported bitsandbytes QuantState profile")
    if not isinstance(state, Mapping) or set(state) != _KEYS:
        raise QuantizationError("incomplete or unknown QuantState projection fields")
    if (state["dtype"] not in ("float16", "bfloat16", "float32") or state["quant_type"] != "nf4"
            or type(state["blocksize"]) is not int or state["blocksize"] != 64):
        raise QuantizationError("profile requires float16/bfloat16/float32 NF4 with blocksize=64")
    if type(state["nested"]) is not bool or not isinstance(state["shape"], (tuple, list)):
        raise QuantizationError("QuantState needs explicit shape and boolean nested")
    spec = QuantizationSpec(scheme="nf4/v1", storage_dtype="uint4", logical_shape=tuple(state["shape"]),
                            scale_value_id=scale_value_id, codebook_value_id=codebook_value_id,
                            codebook="bitsandbytes-nf4/v1", granularity="per_block", group_size=64,
                            rounding="none", saturation="none")
    count = spec.parameter_shape[0]
    if state["nested"]:
        inner = state["state2"]
        if not isinstance(inner, Mapping) or set(inner) != _KEYS:
            raise QuantizationError("incomplete or unknown nested QuantState fields")
        if (inner["nested"] is not False or inner["state2"] is not None or inner["offset"] is not None
                or inner["shape"] is not None or inner["quant_type"] is not None
                or inner["dtype"] != "float32" or type(inner["blocksize"]) is not int
                or inner["blocksize"] != 256):
            raise QuantizationError("only one nested scale level, float32 and blocksize=256 is supported")
        codes = _array(state["absmax"], "uint8", (count,), "compressed scales")
        book = _array(inner["code"], "float32", (256,), "nested codebook")
        maxima = _array(inner["absmax"], "float32", ((count + 255) // 256,), "nested absmax")
        offset = _array(state["offset"], "float32", (), "nested offset")
        if np.any(maxima < 0):
            raise QuantizationError("nested absmax must be nonnegative")
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            try:
                scales = book[codes] * maxima[np.arange(count, dtype=np.int64) // 256]
                scales = scales + offset
            except FloatingPointError:
                raise QuantizationError("nested scale dequantization overflow") from None
    else:
        if state["offset"] is not None or state["state2"] is not None:
            raise QuantizationError("non-nested state cannot carry offset/state2")
        scales = _array(state["absmax"], "float32", (count,), "block scales")
    item = _import_nf4_codes(packed, spec, scales, state["code"])
    return ImportedQuantState(item, state["dtype"], state["nested"])


def import_bitsandbytes_quant_state(
    packed, quant_state, *, profile: str, scale_value_id: int, codebook_value_id: int,
) -> ImportedQuantState:
    """Read an actual bnb QuantState and Torch uint8 buffer without mutating them.

    Requires installed bitsandbytes == 0.50.1. Device tensors are detached and
    copied to CPU. No backend dequantizer is used by this importer, allowing an
    independent comparison with bitsandbytes.functional.dequantize_4bit.
    """
    if profile != QUANT_STATE_PROFILE:
        raise QuantizationError("unsupported bitsandbytes QuantState profile")
    try:
        import bitsandbytes as bnb
        import torch
    except ImportError as exc:
        raise QuantizationError("QuantState object import requires optional torch and bitsandbytes==0.50.1") from exc
    if bnb.__version__ != "0.50.1":
        raise QuantizationError(f"profile requires bitsandbytes 0.50.1, found {bnb.__version__}")
    if not isinstance(quant_state, bnb.functional.QuantState):
        raise QuantizationError("expected a bitsandbytes QuantState object")

    def host(value, dtype, label):
        if not isinstance(value, torch.Tensor) or value.dtype != dtype or value.layout != torch.strided:
            raise QuantizationError(f"{label} must be a strided Torch {dtype} tensor")
        return value.detach().cpu().contiguous().numpy().copy()

    def project(state, *, inner=False):
        if not isinstance(state, bnb.functional.QuantState):
            raise QuantizationError("expected a bitsandbytes nested QuantState")
        if type(state.nested) is not bool or (inner and (state.nested or state.state2 is not None)):
            raise QuantizationError("only one nested scale level is supported")
        if state.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise QuantizationError("unsupported QuantState source dtype")
        return {
            "shape": tuple(state.shape) if state.shape is not None else None,
            "dtype": str(state.dtype).removeprefix("torch."), "blocksize": state.blocksize,
            "quant_type": state.quant_type, "nested": state.nested,
            "absmax": host(state.absmax, torch.uint8 if state.nested else torch.float32, "absmax"),
            "code": host(state.code, torch.float32, "codebook"),
            "offset": host(state.offset, torch.float32, "offset") if state.offset is not None else None,
            "state2": project(state.state2, inner=True) if state.state2 is not None else None,
        }

    return import_bitsandbytes_nf4_projection(host(packed, torch.uint8, "packed weights"), project(quant_state),
                                              profile=profile, scale_value_id=scale_value_id,
                                              codebook_value_id=codebook_value_id)
