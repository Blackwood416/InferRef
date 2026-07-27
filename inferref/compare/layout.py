"""Layout comparison (SPEC §29).

Layout is a first-class debugging concern: an engine can produce numerically
correct values in the wrong memory arrangement, and the report must say which
of those happened. The four failure classes are kept distinct::

    shape mismatch
    stride mismatch
    storage-offset mismatch
    value mismatch

The canonical example this exists for::

    Tensor values are identical when interpreted logically,
    but engine stride differs from reference stride.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferref.tensor.codec import TensorView


@dataclass
class LayoutDiff:
    """Structural differences between two tensors (SPEC §29)."""

    dtype_mismatch: bool = False
    shape_mismatch: bool = False
    stride_mismatch: bool = False
    storage_offset_mismatch: bool = False

    reference_dtype: str = ""
    actual_dtype: str = ""
    reference_shape: tuple[int, ...] = ()
    actual_shape: tuple[int, ...] = ()
    reference_stride: tuple[int, ...] = ()
    actual_stride: tuple[int, ...] = ()
    reference_storage_offset: int = 0
    actual_storage_offset: int = 0

    @property
    def comparable(self) -> bool:
        """Whether values can be compared at all.

        Shape and dtype must agree; stride and offset differences are reported
        but do not block value comparison, because payloads are stored in
        canonical logical order (IR §20).
        """
        return not (self.shape_mismatch or self.dtype_mismatch)

    @property
    def any_mismatch(self) -> bool:
        return (
            self.dtype_mismatch
            or self.shape_mismatch
            or self.stride_mismatch
            or self.storage_offset_mismatch
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dtype_mismatch": self.dtype_mismatch,
            "shape_mismatch": self.shape_mismatch,
            "stride_mismatch": self.stride_mismatch,
            "storage_offset_mismatch": self.storage_offset_mismatch,
            "reference": {
                "dtype": self.reference_dtype,
                "shape": list(self.reference_shape),
                "stride": list(self.reference_stride),
                "storage_offset": self.reference_storage_offset,
            },
            "actual": {
                "dtype": self.actual_dtype,
                "shape": list(self.actual_shape),
                "stride": list(self.actual_stride),
                "storage_offset": self.actual_storage_offset,
            },
        }

    def summary(self) -> str:
        parts: list[str] = []
        if self.dtype_mismatch:
            parts.append(f"dtype {self.reference_dtype} != {self.actual_dtype}")
        if self.shape_mismatch:
            parts.append(f"shape {list(self.reference_shape)} != {list(self.actual_shape)}")
        if self.stride_mismatch:
            parts.append(f"stride {list(self.reference_stride)} != {list(self.actual_stride)}")
        if self.storage_offset_mismatch:
            parts.append(
                f"storage_offset {self.reference_storage_offset} "
                f"!= {self.actual_storage_offset}"
            )
        return "; ".join(parts)


def diff_layout(
    reference: TensorView, actual: TensorView, *, ignore_stride: bool = False
) -> LayoutDiff:
    """Compare the layout metadata of two decoded tensors."""
    return LayoutDiff(
        dtype_mismatch=reference.dtype != actual.dtype,
        shape_mismatch=tuple(reference.shape) != tuple(actual.shape),
        stride_mismatch=(
            False if ignore_stride else tuple(reference.stride) != tuple(actual.stride)
        ),
        storage_offset_mismatch=(
            False
            if ignore_stride
            else reference.storage_offset != actual.storage_offset
        ),
        reference_dtype=reference.dtype,
        actual_dtype=actual.dtype,
        reference_shape=tuple(reference.shape),
        actual_shape=tuple(actual.shape),
        reference_stride=tuple(reference.stride),
        actual_stride=tuple(actual.stride),
        reference_storage_offset=reference.storage_offset,
        actual_storage_offset=actual.storage_offset,
    )
