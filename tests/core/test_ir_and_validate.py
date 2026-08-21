"""Trace IR schema and validation tests (IR §46, §48).

The validator's ten invariants are exercised with deliberately corrupted
packages: a validator that never fires is worthless, so each invariant has a
negative case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inferref.ir.graph import Graph, GraphIO
from inferref.ir.manifest import Manifest
from inferref.ir.module import ModuleRecord, path_matches
from inferref.ir.operator import AliasEffect, Effects, OperatorRecord, StorageMutation
from inferref.ir.package import TracePackage
from inferref.ir.region import RegionRecord
from inferref.ir.source import SourceFrame, SourceRecord
from inferref.ir.tensor_value import CaptureInfo, Device, TensorValueRecord
from inferref.ir.validate import validate_package
from inferref.ir.values import (
    DictValue,
    ListValue,
    NoneValue,
    OpaqueValue,
    ScalarValue,
    StringValue,
    TensorRef,
    TupleValue,
    value_from_dict,
)
from inferref.ir.version import check_format_version
from inferref.region.boundary import derive_boundary


def _tensor(value_id: int, shape=(2, 2), **kwargs) -> TensorValueRecord:
    return TensorValueRecord(
        id=value_id,
        dtype=kwargs.pop("dtype", "float32"),
        shape=shape,
        stride=kwargs.pop("stride", tuple(reversed([1] * len(shape)))
                          if len(shape) < 2 else (shape[1], 1)),
        storage_id=kwargs.pop("storage_id", value_id),
        storage_version=kwargs.pop("storage_version", 0),
        **kwargs,
    )


def _package() -> TracePackage:
    """A minimal but valid two-operator package."""
    values = [_tensor(1), _tensor(2), _tensor(3), _tensor(4)]
    ops = [
        OperatorRecord(
            id=1, execution_index=0, namespace="aten", op="mm", overload="default",
            positional_args=(TensorRef(1), TensorRef(2)), result=TensorRef(3),
        ),
        OperatorRecord(
            id=2, execution_index=1, namespace="aten", op="add", overload="Tensor",
            positional_args=(TensorRef(3), TensorRef(1)), result=TensorRef(4),
        ),
    ]
    graph = Graph(
        operators=ops,
        values=values,
        inputs=[GraphIO("a", TensorRef(1)), GraphIO("b", TensorRef(2))],
        outputs=[GraphIO("out", TensorRef(4))],
    )
    graph.recompute_links()
    return TracePackage(manifest=Manifest(), graph=graph)


def _errors(package: TracePackage) -> list[str]:
    return [str(i) for i in validate_package(package) if i.severity == "error"]


def test_capture_degradation_metadata_roundtrip() -> None:
    capture = CaptureInfo(
        mode="hash",
        requested_mode="full",
        degraded_reason="max_capture_elements",
        limit=1_000_000,
        logical_numel=4_194_304,
    )

    encoded = capture.to_dict()
    decoded = CaptureInfo.from_dict(encoded)

    assert encoded == {
        "mode": "hash",
        "requested_mode": "full",
        "degraded_reason": "max_capture_elements",
        "limit": 1_000_000,
        "logical_numel": 4_194_304,
    }
    assert decoded == capture


# -- happy path ------------------------------------------------------------


def test_minimal_package_is_valid() -> None:
    assert _errors(_package()) == []


def test_package_roundtrips_through_disk(tmp_path: Path) -> None:
    package = _package()
    package.modules = [ModuleRecord(id=1, path="layers.0", type="torch.nn.Module")]
    package.sources = [
        SourceRecord(id=1, primary=SourceFrame("m.py", 10, "forward"),
                     stack=(SourceFrame("m.py", 10, "forward"),))
    ]
    package.save(tmp_path / "trace")

    reloaded = TracePackage.load(tmp_path / "trace")
    assert len(reloaded.graph.operators) == 2
    assert len(reloaded.graph.values) == 4
    assert reloaded.graph.op(1).canonical_name == "aten.mm.default"
    assert reloaded.modules[0].path == "layers.0"
    assert str(reloaded.sources[0].primary) == "m.py:10 in forward"
    assert _errors(reloaded) == []


def test_package_save_to_new_root_copies_payloads(tmp_path: Path) -> None:
    import numpy as np

    from inferref.tensor import codec

    package = _package()
    source = tmp_path / "source"
    package.save(source)
    codec.write_array(source / "tensors" / "v1.irtensor", np.zeros((2, 2), np.float32))
    package.graph.value(1).capture = CaptureInfo(
        mode="full", payload="tensors/v1.irtensor"
    )
    package.save(source)

    loaded = TracePackage.load(source)
    destination = tmp_path / "destination"
    loaded.save(destination)

    copied = TracePackage.load(destination)
    assert (destination / "tensors" / "v1.irtensor").is_file()
    assert _errors(copied) == []


# -- invariants 1 & 2: reference integrity ---------------------------------


def test_invariant_2_missing_value_is_reported() -> None:
    package = _package()
    package.graph.operators[0].positional_args = (TensorRef(1), TensorRef(999))
    package.graph.recompute_links()
    assert any("missing value:999" in e for e in _errors(package))


def test_invariant_1_missing_source_is_reported() -> None:
    package = _package()
    package.sources = [SourceRecord(id=1)]
    package.graph.operators[0].source_id = 42
    assert any("missing source:42" in e for e in _errors(package))


def test_invariant_1_missing_module_is_reported() -> None:
    package = _package()
    package.modules = [ModuleRecord(id=1, path="a")]
    package.graph.operators[0].module_stack = (99,)
    assert any("missing module:99" in e for e in _errors(package))


def test_duplicate_record_ids_are_reported_before_reindex() -> None:
    package = _package()
    package.graph.values.append(_tensor(1))
    package.graph.operators.append(
        OperatorRecord(id=1, execution_index=2, namespace="aten", op="neg")
    )

    errors = _errors(package)
    assert any("duplicate value id 1" in error for error in errors)
    assert any("duplicate operator id 1" in error for error in errors)


def test_alias_effect_endpoints_must_belong_to_operator() -> None:
    package = _package()
    package.graph.op(1).effects = Effects(
        aliases=(AliasEffect(output_value_id=4, input_value_id=3, relationship="view"),)
    )

    errors = _errors(package)
    assert any("alias output value:4 is not an output" in error for error in errors)
    assert any("alias input value:3 is not an input" in error for error in errors)


def test_alias_effect_relationship_and_identity_are_validated() -> None:
    package = _package()
    package.graph.value(1).runtime_object_id = 10
    package.graph.value(3).runtime_object_id = 11
    package.graph.op(1).effects = Effects(
        aliases=(
            AliasEffect(3, 1, "same_object"),
            AliasEffect(3, 2, "view"),
            AliasEffect(3, 1, "future_alias"),
        )
    )

    errors = _errors(package)
    assert any("different runtime_object_id" in error for error in errors)
    assert any("view alias does not share one storage_id" in error for error in errors)
    assert any("unknown alias relationship 'future_alias'" in error for error in errors)


def test_duplicate_mutation_and_alias_effects_are_rejected() -> None:
    package = _package()
    mutation = StorageMutation(storage_id=1, version_before=0, version_after=1)
    alias = AliasEffect(3, 1, "unknown_alias")
    package.graph.op(1).effects = Effects(
        mutated_storages=(mutation, mutation), aliases=(alias, alias)
    )

    errors = _errors(package)
    assert any("mutated more than once" in error for error in errors)
    assert any("duplicate alias effect" in error for error in errors)


# -- invariant 3: execution order ------------------------------------------


def test_invariant_3_duplicate_execution_index() -> None:
    package = _package()
    package.graph.operators[1].execution_index = 0
    assert any("duplicate execution_index" in e for e in _errors(package))


def test_invariant_3_negative_execution_index() -> None:
    package = _package()
    package.graph.operators[0].execution_index = -5
    assert any("negative execution_index" in e for e in _errors(package))


# -- invariants 4 & 5: producer / consumer ---------------------------------


def test_invariant_4_wrong_producer() -> None:
    package = _package()
    package.graph.value(3).producer = 2      # actually produced by op 1
    assert any("producer recorded as 2" in e for e in _errors(package))


def test_invariant_5_wrong_consumers() -> None:
    package = _package()
    package.graph.value(1).consumers = (1,)  # op 2 also consumes it
    assert any("consumers recorded as" in e for e in _errors(package))


# -- invariant 6: rank consistency -----------------------------------------


def test_invariant_6_rank_mismatch() -> None:
    package = _package()
    package.graph.value(1).stride = (1,)     # shape is rank 2
    assert any("rank mismatch" in e for e in _errors(package))


def test_invariant_6_unknown_dtype() -> None:
    package = _package()
    package.graph.value(1).dtype = "float128"
    assert any("unknown dtype" in e for e in _errors(package))


# -- invariant 7: storage versions never decrease --------------------------


def test_invariant_7_version_going_backwards() -> None:
    package = _package()
    # value 1 is consumed by op 1 (index 0) and op 2 (index 1); make the later
    # observation claim an older version of the same storage.
    package.graph.value(3).storage_id = 1
    package.graph.value(3).storage_version = 5
    package.graph.value(4).storage_id = 1
    package.graph.value(4).storage_version = 2
    assert any("version went backwards" in e for e in _errors(package))


def test_invariant_7_mutation_must_advance_version() -> None:
    package = _package()
    package.graph.operators[0].effects = Effects(
        mutated_storages=(StorageMutation(storage_id=1, version_before=3,
                                          version_after=3),)
    )
    assert any("does not advance version" in e for e in _errors(package))


def test_invariant_7_mutation_cannot_skip_generations() -> None:
    package = _package()
    package.graph.operators[0].effects = Effects(
        mutated_storages=(
            StorageMutation(storage_id=1, version_before=7, version_after=9),
        )
    )

    errors = _errors(package)
    assert any("skips generations (7 -> 9)" in error for error in errors)
    assert any("does not match observed version (0 != 7)" in error for error in errors)


def test_invariant_7_mutation_must_be_connected_to_tensor_input() -> None:
    package = _package()
    package.graph.operators[0].effects = Effects(
        mutated_storages=(
            StorageMutation(storage_id=99, version_before=0, version_after=1),
        )
    )

    assert any(
        "storage:99 mutation is not connected to a tensor input" in error
        for error in _errors(package)
    )


def test_invariant_7_mutation_effect_advances_high_water_without_tensor_result() -> None:
    values = [
        _tensor(1, storage_id=1, storage_version=0),
        _tensor(2, storage_id=1, storage_version=0),
        _tensor(3, storage_id=3, storage_version=0),
    ]
    mutate = OperatorRecord(
        id=1,
        execution_index=0,
        namespace="custom",
        op="update_cache",
        overload="default",
        positional_args=(TensorRef(1),),
        result=NoneValue(),
        effects=Effects(
            mutated_storages=(
                StorageMutation(storage_id=1, version_before=0, version_after=1),
            )
        ),
    )
    stale_read = OperatorRecord(
        id=2,
        execution_index=1,
        namespace="custom",
        op="read_cache",
        overload="default",
        positional_args=(TensorRef(2),),
        result=TensorRef(3),
    )
    graph = Graph(operators=[mutate, stale_read], values=values)
    graph.recompute_links()
    package = TracePackage(manifest=Manifest(), graph=graph)

    assert any("version went backwards" in error for error in _errors(package))


def test_mutation_effect_produces_unreturned_storage_alias() -> None:
    """A base tensor at version_after is caused by copy_, not an input."""
    values = [
        _tensor(1, storage_id=1, storage_version=0),  # target before write
        _tensor(2, storage_id=1, storage_version=1),  # copy_ target result
        _tensor(3, storage_id=1, storage_version=1),  # base cache after write
        _tensor(4, shape=(1, 2), storage_id=1, storage_version=1),  # live view
        _tensor(5, storage_id=5, storage_version=0),  # source values
    ]
    copy = OperatorRecord(
        id=1,
        execution_index=0,
        namespace="aten",
        op="copy_",
        overload="default",
        positional_args=(TensorRef(1), TensorRef(5)),
        result=TensorRef(2),
        effects=Effects(
            mutated_storages=(
                StorageMutation(storage_id=1, version_before=0, version_after=1),
            )
        ),
    )
    live_slice = OperatorRecord(
        id=2,
        execution_index=1,
        namespace="aten",
        op="slice",
        overload="Tensor",
        positional_args=(TensorRef(3),),
        result=TensorRef(4),
    )
    graph = Graph(operators=[copy, live_slice], values=values)
    graph.recompute_links()

    assert graph.value(2).producer == 1  # explicit return
    assert graph.value(3).producer == 1  # storage-generation effect
    assert graph.value(4).producer == 2  # explicit downstream view wins
    assert derive_boundary(graph, [1, 2])[0] == [1, 5]

    package = TracePackage(manifest=Manifest(), graph=graph)
    assert _errors(package) == []


# -- invariants 8 & 9: regions ---------------------------------------------


def test_invariant_8_region_references_missing_op() -> None:
    package = _package()
    package.regions = [RegionRecord(id=1, name="r", node_ids=(1, 42))]
    assert any("missing op:42" in e for e in _errors(package))


def test_invariant_9_region_boundary_must_match_derivation() -> None:
    package = _package()
    package.regions = [
        RegionRecord(id=1, name="r", node_ids=(1,), inputs=(1,), outputs=(3,))
    ]
    # The real boundary for {op 1} is inputs (1, 2), outputs (3).
    assert any("recorded inputs" in e for e in _errors(package))


def test_correct_region_boundary_validates() -> None:
    package = _package()
    package.regions = [
        RegionRecord(id=1, name="r", node_ids=(1,), inputs=(1, 2), outputs=(3,))
    ]
    assert _errors(package) == []


# -- invariant 10: payload byte counts -------------------------------------


def test_invariant_10_missing_payload(tmp_path: Path) -> None:
    package = _package()
    package.graph.value(1).capture = CaptureInfo(mode="full", payload="tensors/nope.irtensor")
    package.save(tmp_path / "trace")
    assert any("missing payload" in e for e in _errors(TracePackage.load(tmp_path / "trace")))


def test_invariant_10_payload_cannot_escape_trace_root(tmp_path: Path) -> None:
    package = _package()
    package.graph.value(1).capture = CaptureInfo(
        mode="full", payload="../outside.irtensor"
    )
    package.save(tmp_path / "trace")
    (tmp_path / "outside.irtensor").write_bytes(b"not part of the trace")

    errors = _errors(TracePackage.load(tmp_path / "trace"))
    assert any("payload path escapes trace package" in error for error in errors)


def test_invariant_10_payload_shape_must_match(tmp_path: Path) -> None:
    import numpy as np

    from inferref.tensor import codec

    package = _package()
    root = tmp_path / "trace"
    package.save(root)
    # Write a payload whose header describes a different shape than the value.
    codec.write_array(root / "tensors" / "v1.irtensor", np.zeros((4, 4), np.float32))
    package.graph.value(1).capture = CaptureInfo(mode="full", payload="tensors/v1.irtensor")
    package.save(root)

    errors = _errors(TracePackage.load(root))
    assert any("shape" in e for e in errors), errors


def test_invariant_10_payload_header_must_match_value_metadata(tmp_path: Path) -> None:
    import struct

    import numpy as np

    from inferref.ir.dtypes import dtype_code
    from inferref.tensor import codec

    package = _package()
    root = tmp_path / "trace"
    package.save(root)
    payload = root / "tensors" / "v1.irtensor"
    codec.write_array(payload, np.zeros((2, 2), np.float32))
    blob = bytearray(payload.read_bytes())
    struct.pack_into("<H", blob, 8, dtype_code("int32"))
    struct.pack_into("<I", blob, 10, 0)
    struct.pack_into("<Q", blob, 18, 3)
    struct.pack_into("<2q", blob, 50, 1, 2)
    struct.pack_into("<q", blob, 66, 4)
    payload.write_bytes(blob)
    package.graph.value(1).capture = CaptureInfo(
        mode="full", payload="tensors/v1.irtensor"
    )
    package.save(root)

    errors = _errors(TracePackage.load(root))
    assert any("payload dtype int32 != value dtype float32" in error for error in errors)
    assert any("logical_numel 3" in error for error in errors)
    assert any("not marked canonical contiguous" in error for error in errors)
    assert any("payload stride" in error for error in errors)
    assert any("payload storage_offset 4" in error for error in errors)


# -- IR §46 forward compatibility ------------------------------------------


def test_unknown_fields_are_preserved_on_rewrite() -> None:
    data = {
        "id": 7,
        "kind": "tensor",
        "dtype": "float32",
        "shape": [2],
        "stride": [1],
        "a_field_from_the_future": {"nested": True},
    }
    record = TensorValueRecord.from_dict(data)
    assert record.id == 7
    assert record.unknown["a_field_from_the_future"] == {"nested": True}
    assert record.to_dict()["a_field_from_the_future"] == {"nested": True}


def test_extensions_are_preserved() -> None:
    data = {
        "id": 1, "execution_index": 0, "canonical_name": "aten.mm.default",
        "extensions": {"pytorch": {"source_fn_stack": ["torch.nn.Linear"]}},
    }
    record = OperatorRecord.from_dict(data)
    assert record.extensions["pytorch"]["source_fn_stack"] == ["torch.nn.Linear"]
    assert record.to_dict()["extensions"] == data["extensions"]


def test_canonical_name_is_recovered_when_parts_are_absent() -> None:
    record = OperatorRecord.from_dict(
        {"id": 1, "execution_index": 0, "canonical_name": "aten.slice.Tensor"}
    )
    assert (record.namespace, record.op, record.overload) == ("aten", "slice", "Tensor")
    assert record.canonical_name == "aten.slice.Tensor"


def test_unknown_value_kind_degrades_to_opaque() -> None:
    value = value_from_dict({"kind": "quantum_superposition", "value": 42})
    assert isinstance(value, OpaqueValue)
    assert value.portable is False


def test_incompatible_major_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported trace format_version"):
        check_format_version("inferref-trace", "9.0")
    with pytest.raises(ValueError, match="not an InferRef trace"):
        check_format_version("something-else", "0.1")


# -- value system ----------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        TensorRef(5),
        ScalarValue("int64", 3),
        ScalarValue.from_number(float("nan"), "float32"),
        ScalarValue.from_number(float("-inf"), "float64"),
        NoneValue(),
        StringValue("hello"),
        ListValue((TensorRef(1), ScalarValue("int64", 2))),
        TupleValue((NoneValue(), StringValue("x"))),
        DictValue(((StringValue("dim"), ScalarValue("int64", 1)),)),
        OpaqueValue("torch.dtype", "float16", True),
    ],
)
def test_value_json_roundtrip(value) -> None:
    encoded = json.loads(json.dumps(value.to_dict()))
    assert value_from_dict(encoded).to_dict() == value.to_dict()


def test_special_scalar_decoding() -> None:
    import math

    assert math.isnan(ScalarValue.from_number(float("nan"), "float32").as_number())
    assert ScalarValue.from_number(float("inf"), "float32").as_number() == math.inf
    assert ScalarValue.from_number(float("-inf"), "float32").as_number() == -math.inf
    assert ScalarValue("int64", 5).as_number() == 5


def test_tensor_value_derived_sizes() -> None:
    value = _tensor(1, shape=(2, 3, 4), dtype="float16")
    assert value.logical_numel == 24
    assert value.logical_nbytes == 48
    assert value.rank == 3


def test_device_parsing() -> None:
    assert str(Device.from_dict({"type": "cuda", "index": 1})) == "cuda:1"
    assert str(Device.from_dict("cpu")) == "cpu"
    assert str(Device.from_dict(None)) == "cpu"


# -- module path matching --------------------------------------------------


@pytest.mark.parametrize(
    "path,pattern,expected",
    [
        ("layers.0", "layers.0", True),
        ("layers.0.self_attn", "layers.0", True),
        ("layers.0", "model.layers.0", True),      # tolerate an absent root name
        ("layers.10", "layers.1", False),          # not a prefix match on digits
        ("layers.0", "layers.1", False),
        ("anything", "", True),
    ],
)
def test_path_matches(path: str, pattern: str, expected: bool) -> None:
    assert path_matches(path, pattern) is expected
