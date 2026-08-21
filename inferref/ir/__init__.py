"""InferRef Trace IR v0.1 — the stable data contract.

This package is deliberately **pure stdlib**. It must remain importable in an
environment where neither PyTorch nor numpy is installed (Trace IR v0.1
acceptance criterion #10).
"""

from inferref.ir.dtypes import (
    DTYPE_CODES,
    DTYPE_NAMES,
    dtype_code,
    dtype_itemsize,
    dtype_name_from_code,
    is_known_dtype,
)
from inferref.ir.graph import Graph, GraphIO
from inferref.ir.manifest import (
    Capture,
    Determinism,
    Environment,
    Execution,
    Manifest,
    ModelInfo,
    NamedVersion,
    SourcePolicy,
)
from inferref.ir.module import ModuleRecord
from inferref.ir.operator import AliasEffect, Effects, OperatorRecord, StorageMutation
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.source import SourceFrame, SourceRecord
from inferref.ir.storage import StorageRecord
from inferref.ir.tensor_value import CaptureInfo, Device, TensorHash, TensorValueRecord
from inferref.ir.validate import ValidationIssue, validate_package
from inferref.ir.values import (
    DictValue,
    ListValue,
    NoneValue,
    OpaqueValue,
    ScalarValue,
    StringValue,
    TensorRef,
    TupleValue,
    Value,
    value_from_dict,
    walk_tensor_refs,
)
from inferref.ir.version import (
    FORMAT,
    FORMAT_VERSION,
    INFERREF_VERSION,
    TENSOR_FORMAT_VERSION,
)

__all__ = [
    "DTYPE_CODES",
    "DTYPE_NAMES",
    "FORMAT",
    "FORMAT_VERSION",
    "INFERREF_VERSION",
    "TENSOR_FORMAT_VERSION",
    "AliasEffect",
    "Capture",
    "CaptureInfo",
    "Determinism",
    "Device",
    "DictValue",
    "Effects",
    "Environment",
    "Execution",
    "Graph",
    "GraphIO",
    "ListValue",
    "Manifest",
    "ModelInfo",
    "ModuleRecord",
    "NamedVersion",
    "NoneValue",
    "OpaqueValue",
    "OperatorRecord",
    "RegionRecord",
    "ScalarValue",
    "SourceFrame",
    "SourcePolicy",
    "SourceRecord",
    "StorageMutation",
    "StorageRecord",
    "StringValue",
    "TensorHash",
    "TensorRef",
    "TensorValueRecord",
    "TracePackage",
    "TupleValue",
    "ValidationIssue",
    "Value",
    "dtype_code",
    "dtype_itemsize",
    "dtype_name_from_code",
    "is_known_dtype",
    "validate_package",
    "value_from_dict",
    "walk_tensor_refs",
]
