"""Explicit affine INT8/INT4 and NF4 semantics with lazy numerical imports."""

from typing import TYPE_CHECKING

from inferref.quantization.spec import QuantizationError, QuantizationSpec

if TYPE_CHECKING:
    from inferref.quantization.bitsandbytes import (
        import_bitsandbytes_nf4_projection,
        import_bitsandbytes_quant_state,
    )
    from inferref.quantization.nf4 import dequantize_nf4, import_bitsandbytes_nf4
    from inferref.quantization.reference import (
        dequantize,
        dequantize_affine,
        linear_int4_bf16_reference,
        linear_int4_reference,
        linear_int8_reference,
        linear_nf4_bf16_reference,
        linear_nf4_reference,
        quantization_fingerprint,
        quantize_affine,
    )


def __getattr__(name: str):
    # Parsing a descriptor remains stdlib-only; load numpy for arithmetic only.
    if name in {"import_bitsandbytes_nf4_projection", "import_bitsandbytes_quant_state"}:
        from inferref.quantization import bitsandbytes

        value = getattr(bitsandbytes, name)
        globals()[name] = value
        return value
    if name in {"dequantize_nf4", "import_bitsandbytes_nf4"}:
        from inferref.quantization import nf4

        value = getattr(nf4, name)
        globals()[name] = value
        return value
    if name in {"dequantize", "dequantize_affine", "linear_nf4_reference", "linear_int8_reference", "linear_int4_reference",
                "linear_nf4_bf16_reference", "linear_int4_bf16_reference",
                "quantization_fingerprint", "quantize_affine"}:
        from inferref.quantization import reference

        value = getattr(reference, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "QuantizationError",
    "QuantizationSpec",
    "dequantize",
    "dequantize_affine",
    "dequantize_nf4",
    "import_bitsandbytes_nf4",
    "import_bitsandbytes_nf4_projection",
    "import_bitsandbytes_quant_state",
    "linear_int4_bf16_reference",
    "linear_int4_reference",
    "linear_int8_reference",
    "linear_nf4_bf16_reference",
    "linear_nf4_reference",
    "quantization_fingerprint",
    "quantize_affine",
]
