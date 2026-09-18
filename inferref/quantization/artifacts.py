"""Version-gated artifact metadata; no numpy or framework imports at load time."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from inferref.quantization.dependencies import (
    DependencyClosure,
    quantization_dependencies,
)
from inferref.quantization.spec import (
    AFFINE_SCHEMES,
    QuantizationError,
    QuantizationSpec,
)


def feature(spec: QuantizationSpec) -> str:
    return f"quantization/{spec.scheme}/{spec.granularity}"


QUANTIZATION_FEATURES = frozenset(
    f"quantization/{scheme}/{granularity}"
    for scheme in AFFINE_SCHEMES
    for granularity in ("per_tensor", "per_axis", "per_group")
) | {"quantization/nf4/v1/per_block"}


def needs_quantization(requirements: Any) -> bool:
    return any(str(item).startswith("quantization/") for item in requirements.get("features", ()))


def validate_comparison(manifest: dict[str, Any], effective: Any) -> None:
    if any("quantization" in output for output in manifest.get("outputs", ())) and (
        effective.comparator != "tensor/numeric/v1" or effective.per_output
    ):
        raise QuantizationError("quantized testcase outputs support only the global numeric comparator; use trace comparison for raw codes")


def validate_registry(records: list[dict[str, Any]], roots: Iterable[int] | None = None) -> DependencyClosure:
    """Check descriptors, unique IDs, dependency shapes and unquantized params."""
    specs = {
        record["id"]: QuantizationSpec.from_dict(record["quantization"])
        for record in records if "quantization" in record
    }
    ids = [record["id"] for record in records]
    closure = quantization_dependencies(ids if roots is None else roots, specs=specs, available_ids=ids)
    registry = {record["id"]: record for record in records}
    for value_id in closure.quantized_ids:
        record, spec = registry[value_id], specs[value_id]
        if record.get("dtype") != spec.storage_dtype or record.get("shape") != list(spec.logical_shape):
            raise QuantizationError(f"value {value_id}: storage dtype/shape differs from quantization descriptor")
        for parameter_id in spec.dependencies:
            parameter = registry[parameter_id]
            dtype = "float32" if parameter_id == spec.scale_value_id else spec.zero_point_dtype
            shape = spec.parameter_shape
            if parameter_id == spec.codebook_value_id:
                dtype, shape = "float32", (16,)
            if parameter.get("dtype") != dtype or parameter.get("shape") != list(shape):
                raise QuantizationError(f"parameter {parameter_id}: expected {dtype} {shape}")
    return closure


def validate_trace_metadata(package: Any) -> DependencyClosure | None:
    from inferref.ir.version import QUANTIZED_FORMAT_VERSION, check_format_version

    check_format_version(package.manifest.format, package.manifest.format_version)
    # to_dict also prevents smuggling a core descriptor through unknown fields.
    records = [value.to_dict() for value in package.graph.values]
    if not any("quantization" in record for record in records):
        return None
    if package.manifest.format_version != QUANTIZED_FORMAT_VERSION:
        raise QuantizationError("embedded quantization requires trace format_version 1.0")
    return validate_registry(records)
