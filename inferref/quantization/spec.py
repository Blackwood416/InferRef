"""Strict, framework-independent descriptors for explicit quantization.

This is a semantic contract, not an ignorable Trace IR extension. Artifact
embedding requires trace 1.0 or testcase 0.4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

FORMAT = "inferref-quantization"
FORMAT_VERSION = "1.0"
SCHEME = "affine-int8/v1"
AFFINE_SCHEMES = {"affine-int8/v1": "int8", "affine-int4/v1": "int4", "affine-uint4/v1": "uint4"}
_MAX_I64 = (1 << 63) - 1


class QuantizationError(ValueError):
    """Invalid or unsupported quantization metadata or numerical input."""


def reject_embedded_quantization(record: dict[str, Any]) -> None:
    """Reject critical metadata in legacy artifacts or untyped unknown fields.

    Existing formats must never silently lower an explicit quantized tensor
    into a plain integer tensor. Namespaced third-party extensions are separate.
    """
    if "quantization" in record:
        raise QuantizationError(
            "embedded quantization metadata requires a typed descriptor and "
            "trace format_version 1.0 or testcase format_version 0.4"
        )


def _integer(value: Any, name: str, *, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= _MAX_I64:
        raise QuantizationError(f"{name} must be an integer in [{minimum}, {_MAX_I64}]")


@dataclass(frozen=True, kw_only=True)
class QuantizationSpec:
    """Affine INT8/INT4/UINT4 or NF4 with explicit parameter references.

    ``zero_point_value_id=None`` explicitly means a symmetric zero point of 0.
    IDs are immutable tensor value IDs, not filenames or runtime addresses.
    Per-group parameters have the logical shape with ``axis`` replaced by
    ceil(axis length / group_size); the final group may be shorter.
    NF4 uses flattened blocks and a separately referenced codebook instead
    of affine zero points. All schemes dequantize in float32.
    """

    logical_shape: tuple[int, ...]
    scale_value_id: int
    zero_point_value_id: int | None = None
    granularity: str = "per_tensor"
    axis: int | None = None
    group_size: int | None = None
    scheme: str = SCHEME
    storage_dtype: str = "int8"
    compute_dtype: str = "float32"
    rounding: str = "nearest_even"
    saturation: str = "clamp"
    codebook_value_id: int | None = None
    codebook: str | None = None

    def __post_init__(self) -> None:
        nf4 = self.scheme == "nf4/v1"
        if not isinstance(self.scheme, str) or self.scheme not in AFFINE_SCHEMES and not nf4:
            raise QuantizationError(f"unsupported scheme: {self.scheme!r}")
        if self.storage_dtype != ("uint4" if nf4 else AFFINE_SCHEMES[self.scheme]):
            raise QuantizationError(f"unsupported storage_dtype {self.storage_dtype!r} for {self.scheme}")
        for name, supported in (
            ("compute_dtype", "float32"), ("rounding", "none" if nf4 else "nearest_even"),
            ("saturation", "none" if nf4 else "clamp"),
        ):
            if getattr(self, name) != supported:
                raise QuantizationError(f"unsupported {name}: {getattr(self, name)!r}")
        if not isinstance(self.logical_shape, tuple):
            raise QuantizationError("logical_shape must be an immutable tuple")
        for dim in self.logical_shape:
            _integer(dim, "logical_shape dimension")
        _integer(self.scale_value_id, "scale_value_id")
        if nf4:
            _integer(self.codebook_value_id, "codebook_value_id")
            if self.codebook_value_id == self.scale_value_id or self.codebook != "bitsandbytes-nf4/v1":
                raise QuantizationError("NF4 requires a distinct codebook ID and bitsandbytes-nf4/v1 identity")
            if self.zero_point_value_id is not None or self.axis is not None or self.granularity != "per_block":
                raise QuantizationError("NF4 requires per_block granularity, null axis and null zero point")
            _integer(self.group_size, "group_size", minimum=1)
            return
        if self.codebook_value_id is not None or self.codebook is not None:
            raise QuantizationError("affine quantization cannot declare a codebook")
        if self.zero_point_value_id is not None:
            _integer(self.zero_point_value_id, "zero_point_value_id")
            if self.zero_point_value_id == self.scale_value_id:
                raise QuantizationError("scale and zero point must have distinct value IDs")
        if self.granularity not in ("per_tensor", "per_axis", "per_group"):
            raise QuantizationError(f"unsupported granularity: {self.granularity!r}")
        if self.granularity == "per_tensor":
            if self.axis is not None or self.group_size is not None:
                raise QuantizationError("per_tensor requires axis=null and group_size=null")
        else:
            _integer(self.axis, "axis")
            assert self.axis is not None
            if self.axis >= len(self.logical_shape):
                raise QuantizationError("axis must be less than the logical rank")
            if self.granularity == "per_axis" and self.group_size is not None:
                raise QuantizationError("per_axis requires group_size=null")
            if self.granularity == "per_group":
                _integer(self.group_size, "group_size", minimum=1)

    @property
    def code_range(self) -> tuple[int, int]:
        return {"int8": (-128, 127), "int4": (-8, 7), "uint4": (0, 15)}[self.storage_dtype]

    @property
    def zero_point_dtype(self) -> str:
        return "uint8" if self.storage_dtype == "uint4" else "int8"

    @property
    def parameter_shape(self) -> tuple[int, ...]:
        """Exact required shape of both scale and (when present) zero point."""
        if self.scheme == "nf4/v1":
            numel = 1
            for dim in self.logical_shape:
                numel *= dim
            assert self.group_size is not None
            return ((numel + self.group_size - 1) // self.group_size,)
        if self.granularity == "per_tensor":
            return ()
        assert self.axis is not None
        if self.granularity == "per_axis":
            return (self.logical_shape[self.axis],)
        assert self.group_size is not None
        shape = list(self.logical_shape)
        shape[self.axis] = (shape[self.axis] + self.group_size - 1) // self.group_size
        return tuple(shape)

    @property
    def dependencies(self) -> tuple[int, ...]:
        if self.codebook_value_id is not None:
            return (self.scale_value_id, self.codebook_value_id)
        if self.zero_point_value_id is None:
            return (self.scale_value_id,)
        return (self.scale_value_id, self.zero_point_value_id)

    def to_dict(self) -> dict[str, Any]:
        """Canonical descriptor: defaults and nulls remain explicit on the wire."""
        return {
            "format": FORMAT,
            "format_version": FORMAT_VERSION,
            "scheme": self.scheme,
            "storage_dtype": self.storage_dtype,
            "logical_shape": list(self.logical_shape),
            "scale_value_id": self.scale_value_id,
            "zero_point_value_id": self.zero_point_value_id,
            "granularity": self.granularity,
            "axis": self.axis,
            "group_size": self.group_size,
            "compute_dtype": self.compute_dtype,
            "rounding": self.rounding,
            "saturation": self.saturation,
            **({"codebook_value_id": self.codebook_value_id, "codebook": self.codebook} if self.scheme == "nf4/v1" else {}),
        }

    @classmethod
    def from_dict(cls, data: Any) -> QuantizationSpec:
        """Reject missing/unknown fields and unknown versions, without guessing."""
        if not isinstance(data, dict):
            raise QuantizationError("quantization descriptor must be an object")
        expected = {
            "format", "format_version", "scheme", "storage_dtype", "logical_shape",
            "scale_value_id", "zero_point_value_id", "granularity", "axis",
            "group_size", "compute_dtype", "rounding", "saturation",
        }
        if data.get("scheme") == "nf4/v1":
            expected.update(("codebook_value_id", "codebook"))
        if set(data) != expected:
            raise QuantizationError(
                f"quantization fields differ: missing={sorted(expected - set(data))}, "
                f"unknown={sorted(set(data) - expected, key=str)}"
            )
        if data["format"] != FORMAT or data["format_version"] != FORMAT_VERSION:
            raise QuantizationError("unsupported quantization format or format_version")
        if not isinstance(data["logical_shape"], list):
            raise QuantizationError("logical_shape must be a JSON array")
        fields = {k: v for k, v in data.items() if k not in ("format", "format_version")}
        fields["logical_shape"] = tuple(data["logical_shape"])
        return cls(**fields)
