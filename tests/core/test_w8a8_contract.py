"""``linear/w8a8-int32-accum/v1`` executable contract (freeze §2, §4 and §6).

The integer/float32 arithmetic lives in ``test_w8a8_reference.py``; this module
owns the contract surface: role binding, descriptor/axis/group validation, the
float32 boundary relation, the merged requirement features, and adapter
preflight before an engine process starts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inferref.agent.executor import execute_adapter
from inferref.agent.protocol import (
    ENGINE_ADAPTER_FORMAT,
    ENGINE_ADAPTER_VERSION,
    EngineAdapter,
)
from inferref.contracts import (
    contract_boundary_issues,
    contract_input_issues,
    contract_requirements,
    get_contract,
)
from inferref.contracts.schema import FEATURE_VOCABULARY
from inferref.ir.graph import Graph
from inferref.ir.manifest import Manifest
from inferref.ir.operator import OperatorRecord
from inferref.ir.package import TracePackage
from inferref.ir.tensor_value import CaptureInfo, TensorValueRecord
from inferref.ir.values import TensorRef
from inferref.quantization import QuantizationSpec, linear_w8a8_int32_reference
from inferref.quantization.artifacts import QUANTIZATION_FEATURES
from inferref.tensor import codec
from inferref.testcase.extract import extract_operator

CONTRACT = "linear/w8a8-int32-accum/v1"
W8A8_FEATURE = "quantization/w8a8-int32-accum/v1"
PER_TENSOR = "quantization/affine-int8/v1/per_tensor"
PER_AXIS = "quantization/affine-int8/v1/per_axis"
PER_GROUP = "quantization/affine-int8/v1/per_group"

M_SIZE, N_SIZE, K_SIZE = 2, 3, 4


def _tensor(shape: list[int], dtype: str = "float32") -> dict[str, object]:
    return {"shape": shape, "dtype": dtype, "stride": []}


def _quantized(spec: QuantizationSpec, dtype: str | None = None) -> dict[str, object]:
    record = _tensor(list(spec.logical_shape), dtype or spec.storage_dtype)
    record["quantization"] = spec.to_dict()
    return record


def _spec(
    logical_shape: tuple[int, ...],
    granularity: str = "per_tensor",
    *,
    group_size: int | None = None,
    scale_value_id: int = 1,
    zero_point_value_id: int | None = None,
) -> QuantizationSpec:
    """Symmetric-or-asymmetric affine INT8 spec on the frozen axes.

    ``QuantizationSpec`` stores only non-negative axes, so the frozen per-token
    axis ``-2`` is already ``0`` here and per-group always uses axis ``1``.
    """
    return QuantizationSpec(
        logical_shape=logical_shape,
        scale_value_id=scale_value_id,
        zero_point_value_id=zero_point_value_id,
        granularity=granularity,
        axis=None if granularity == "per_tensor" else 0 if granularity == "per_axis" else 1,
        group_size=group_size if granularity == "per_group" else None,
    )


def _operands(
    x_granularity: str = "per_tensor",
    weight_granularity: str = "per_tensor",
    *,
    group_size: int | None = None,
    asymmetric: bool = False,
) -> dict[str, dict[str, object]]:
    return {
        "x": _quantized(
            _spec(
                (M_SIZE, K_SIZE),
                x_granularity,
                group_size=group_size,
                scale_value_id=1,
                zero_point_value_id=2 if asymmetric else None,
            )
        ),
        "weight": _quantized(
            _spec(
                (N_SIZE, K_SIZE),
                weight_granularity,
                group_size=group_size,
                scale_value_id=3,
                zero_point_value_id=4 if asymmetric else None,
            )
        ),
    }


# -- declaration -----------------------------------------------------------


def test_contract_declares_frozen_roles_features_and_description() -> None:
    contract = get_contract(CONTRACT)
    assert contract is not None
    assert contract.inputs == ("x", "weight")
    assert contract.outputs == ("y",)
    assert contract.features == (W8A8_FEATURE,)
    assert "no bias" in contract.description
    # The token is part of the shared feature vocabulary, so plugin contracts
    # and adapter manifests can declare it too.
    assert W8A8_FEATURE in FEATURE_VOCABULARY
    assert W8A8_FEATURE in QUANTIZATION_FEATURES


# -- input validation ------------------------------------------------------


@pytest.mark.parametrize(
    ("x_granularity", "weight_granularity", "group_size"),
    [
        ("per_tensor", "per_tensor", None),
        ("per_axis", "per_axis", None),
        ("per_group", "per_group", 2),
        ("per_group", "per_tensor", 2),
        ("per_tensor", "per_group", 2),
        ("per_axis", "per_group", 2),
    ],
)
def test_validator_accepts_every_frozen_granularity_combination(
    x_granularity: str, weight_granularity: str, group_size: int | None
) -> None:
    inputs = _operands(x_granularity, weight_granularity, group_size=group_size)
    assert contract_input_issues(CONTRACT, inputs) == []
    assert (
        contract_boundary_issues(CONTRACT, inputs, {"y": _tensor([M_SIZE, N_SIZE])})
        == []
    )


@pytest.mark.parametrize(
    ("x_granularity", "weight_granularity", "group_size"),
    [
        ("per_axis", "per_axis", None),
        ("per_group", "per_group", 2),
        ("per_group", "per_axis", 2),
    ],
)
def test_validator_accepts_asymmetric_int8_zero_points(
    x_granularity: str, weight_granularity: str, group_size: int | None
) -> None:
    inputs = _operands(
        x_granularity, weight_granularity, group_size=group_size, asymmetric=True
    )
    assert contract_input_issues(CONTRACT, inputs) == []


def test_validator_rejects_non_int8_codes_and_missing_descriptors() -> None:
    inputs = _operands()
    cases = [
        (
            {"x": _tensor([M_SIZE, K_SIZE], "float32")},
            "x must be an int8 code tensor",
        ),
        (
            {"weight": {**_quantized(_spec((N_SIZE, K_SIZE))), "dtype": "float32"}},
            "weight must be an int8 code tensor",
        ),
        (
            {"x": _tensor([M_SIZE, K_SIZE], "int8")},
            "x quantization descriptor is invalid",
        ),
        (
            {"weight": {**_tensor([N_SIZE, K_SIZE], "int8"), "quantization": None}},
            "weight quantization descriptor is invalid",
        ),
    ]
    for overrides, message in cases:
        issues = contract_input_issues(CONTRACT, {**inputs, **overrides})
        assert any(message in issue for issue in issues), (overrides, issues)


def test_validator_rejects_non_affine_int8_schemes() -> None:
    inputs = _operands()
    int4 = QuantizationSpec(
        logical_shape=(M_SIZE, K_SIZE),
        scale_value_id=1,
        scheme="affine-int4/v1",
        storage_dtype="int4",
    )
    nf4 = QuantizationSpec(
        logical_shape=(N_SIZE, K_SIZE),
        scale_value_id=3,
        codebook_value_id=4,
        scheme="nf4/v1",
        storage_dtype="uint4",
        codebook="bitsandbytes-nf4/v1",
        granularity="per_block",
        group_size=K_SIZE,
        rounding="none",
        saturation="none",
    )
    for name, spec in (("x", int4), ("weight", nf4)):
        record = {**_tensor(list(spec.logical_shape), "int8"), "quantization": spec.to_dict()}
        issues = contract_input_issues(CONTRACT, {**inputs, name: record})
        assert any(
            f"{name} must use the affine-int8/v1 scheme" in issue for issue in issues
        ), issues


def test_validator_rejects_rank_and_reduction_mismatch() -> None:
    inputs = _operands()
    rank3 = _quantized(_spec((1, M_SIZE, K_SIZE)))
    issues = contract_input_issues(CONTRACT, {**inputs, "x": rank3})
    assert any("x must be rank 2" in issue for issue in issues), issues

    shorter_k = _quantized(_spec((N_SIZE, K_SIZE + 1)))
    issues = contract_input_issues(CONTRACT, {**inputs, "weight": shorter_k})
    assert any("agree on the reduction size K" in issue for issue in issues), issues


def test_validator_rejects_axes_outside_the_freeze() -> None:
    inputs = _operands()
    for name, logical_shape in (
        ("x", (M_SIZE, K_SIZE)),
        ("weight", (N_SIZE, K_SIZE)),
    ):
        wrong_axis = QuantizationSpec(
            logical_shape=logical_shape,
            scale_value_id=1,
            granularity="per_axis",
            axis=1,
        )
        issues = contract_input_issues(
            CONTRACT, {**inputs, name: _quantized(wrong_axis)}
        )
        expected = f"{name} per_axis must use axis 0"
        assert any(expected in issue for issue in issues), issues

        wrong_group_axis = QuantizationSpec(
            logical_shape=logical_shape,
            scale_value_id=1,
            granularity="per_group",
            axis=0,
            group_size=2,
        )
        issues = contract_input_issues(
            CONTRACT, {**inputs, name: _quantized(wrong_group_axis)}
        )
        expected = f"{name} per_group must use the reduction axis 1"
        assert any(expected in issue for issue in issues), issues


def test_validator_rejects_unequal_group_sizes() -> None:
    inputs = _operands("per_group", "per_group", group_size=2)
    wider = _quantized(_spec((N_SIZE, K_SIZE), "per_group", group_size=K_SIZE))
    issues = contract_input_issues(CONTRACT, {**inputs, "weight": wider})
    assert any(
        "both per_group operands must declare the same group_size" in issue
        for issue in issues
    ), issues


def test_validator_rejects_a_shape_that_contradicts_its_descriptor() -> None:
    inputs = _operands()
    mismatched = {**inputs["x"], "shape": [M_SIZE, K_SIZE + 1]}
    issues = contract_input_issues(CONTRACT, {**inputs, "x": mismatched})
    assert any(
        "x shape must equal its quantization logical_shape" in issue
        for issue in issues
    ), issues


def test_validator_reports_missing_roles() -> None:
    inputs = _operands()
    assert contract_input_issues(CONTRACT, {"x": inputs["x"]}) == [
        "weight must be a tensor"
    ]
    assert contract_input_issues(CONTRACT, {}) == [
        "x must be a tensor",
        "weight must be a tensor",
    ]


# -- boundary relation -----------------------------------------------------


def test_relation_requires_ordinary_float32_y_of_shape_m_n() -> None:
    inputs = _operands("per_group", "per_group", group_size=2)
    assert (
        contract_boundary_issues(CONTRACT, inputs, {"y": _tensor([M_SIZE, N_SIZE])})
        == []
    )
    quantized_y = {
        **_tensor([M_SIZE, N_SIZE]),
        "quantization": _spec((M_SIZE, N_SIZE)).to_dict(),
    }
    cases = [
        ({"y": _tensor([M_SIZE, N_SIZE], "float16")}, "W8A8 y must be an ordinary float32 tensor"),
        ({"y": quantized_y}, "W8A8 y must be an ordinary float32 tensor"),
        ({}, "W8A8 y must be an ordinary float32 tensor"),
        ({"y": _tensor([N_SIZE, M_SIZE])}, "W8A8 y shape must equal [M, N]"),
        ({"y": _tensor([M_SIZE, N_SIZE + 1])}, "W8A8 y shape must equal [M, N]"),
    ]
    for outputs, message in cases:
        issues = contract_boundary_issues(CONTRACT, inputs, outputs)
        assert any(message in issue for issue in issues), (outputs, issues)


# -- requirements ----------------------------------------------------------


def test_requirements_include_the_w8a8_and_constituent_affine_tokens() -> None:
    operands = _operands("per_axis", "per_group", group_size=2)
    manifest = {
        "inputs": [
            {"name": "x", **operands["x"]},
            {"name": "weight", **operands["weight"]},
        ],
        "outputs": [{"name": "y", **_tensor([M_SIZE, N_SIZE])}],
        "nodes": [],
    }
    requirements = contract_requirements(manifest, CONTRACT)
    assert requirements["features"] == [PER_AXIS, PER_GROUP, W8A8_FEATURE]
    assert requirements["dtypes"] == ["float32", "int8"]
    assert requirements["max_rank"] == 2


# -- adapter preflight -----------------------------------------------------


def _w8a8_trace(root: Path) -> TracePackage:
    """int8 ``x [2,4]`` @ int8 ``weight [3,4]`` -> float32 ``y [2,3]``, per-tensor."""
    root.mkdir(parents=True, exist_ok=False)
    x_codes = np.array([[1, -2, 3, -4], [5, 6, -7, 8]], dtype=np.int8)
    weight_codes = np.array(
        [[1, 0, -1, 2], [3, -4, 5, -6], [7, 8, -9, 1]], dtype=np.int8
    )
    input_spec = _spec((M_SIZE, K_SIZE), scale_value_id=10)
    weight_spec = _spec((N_SIZE, K_SIZE), scale_value_id=11)
    arrays = {
        1: x_codes,
        2: weight_codes,
        10: np.array(0.5, dtype=np.float32),
        11: np.array(0.25, dtype=np.float32),
    }
    for value_id, array in arrays.items():
        codec.write_array(root / f"tensors/v{value_id}.irtensor", array)
    parameters = {
        value_id: codec.read(root / f"tensors/v{value_id}.irtensor")
        for value_id in input_spec.dependencies + weight_spec.dependencies
    }
    arrays[3] = linear_w8a8_int32_reference(
        codec.read(root / "tensors/v1.irtensor"),
        input_spec,
        codec.read(root / "tensors/v2.irtensor"),
        weight_spec,
        values=parameters,
    )
    codec.write_array(root / "tensors/v3.irtensor", arrays[3])
    values = [
        TensorValueRecord(
            id=value_id,
            dtype=str(array.dtype),
            shape=array.shape,
            stride=codec.contiguous_stride(array.shape),
            producer=0 if value_id == 3 else None,
            consumers=(0,) if value_id in (1, 2) else (),
            quantization=input_spec if value_id == 1 else (
                weight_spec if value_id == 2 else None
            ),
            capture=CaptureInfo(mode="full", payload=f"tensors/v{value_id}.irtensor"),
        )
        for value_id, array in arrays.items()
    ]
    graph = Graph(
        values=values,
        operators=[
            OperatorRecord(
                id=0,
                execution_index=0,
                namespace="inferref",
                op="linear_w8a8_int32",
                positional_args=(TensorRef(1), TensorRef(2)),
                result=TensorRef(3),
            )
        ],
    )
    package = TracePackage(manifest=Manifest(format_version="1.0"), graph=graph, root=root)
    package.save(root)
    return package


def _w8a8_case(tmp_path: Path) -> Path:
    package = _w8a8_trace(tmp_path / "trace")
    return extract_operator(
        package,
        0,
        tmp_path / "case",
        input_names=("x", "weight"),
        output_names=("y",),
        contracts=(CONTRACT,),
    ).path


def _adapter(
    path: Path, features: list[str], *, with_contract_capabilities: bool = True
) -> Path:
    """Probe adapter whose engine command must never do real work."""
    capabilities: dict[str, object] = {
        "device_types": ["cpu"],
        "dtypes": ["float32", "int8"],
        "max_rank": 2,
        "features": features,
        "contracts": [CONTRACT],
    }
    if with_contract_capabilities:
        capabilities["contract_capabilities"] = {
            CONTRACT: {
                "dtypes": ["float32", "int8"],
                "max_rank": 2,
                "features": list(features),
            }
        }
    path.write_text(
        json.dumps(
            {
                "format": ENGINE_ADAPTER_FORMAT,
                "format_version": ENGINE_ADAPTER_VERSION,
                "name": "probe",
                "target_device": "cpu",
                "capabilities": capabilities,
                "command": [
                    "{python}",
                    "-c",
                    "raise SystemExit('must not run')",
                    "{testcase}",
                    "{output}",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_preflight_rejects_a_missing_w8a8_feature_token(tmp_path: Path) -> None:
    case = _w8a8_case(tmp_path)
    adapter = _adapter(tmp_path / "adapter.json", [PER_TENSOR])
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert result["status"] == "unsupported"
    assert result["execution"] is None
    assert result["unsupported"] == [
        {"kind": "contract_feature", "contract": CONTRACT, "required": W8A8_FEATURE}
    ]


def test_preflight_accepts_a_declared_w8a8_feature_token(tmp_path: Path) -> None:
    case = _w8a8_case(tmp_path)
    adapter = _adapter(tmp_path / "adapter.json", [PER_TENSOR, W8A8_FEATURE])
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert "unsupported" not in result
    # The probe command declines to run, so reaching execution proves preflight passed.
    assert result["status"] == "execution_error"
    assert result["execution"] is not None


def test_preflight_rejects_a_missing_w8a8_token_without_a_refinement(
    tmp_path: Path,
) -> None:
    case = _w8a8_case(tmp_path)
    adapter = _adapter(
        tmp_path / "adapter.json", [PER_TENSOR], with_contract_capabilities=False
    )
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert result["status"] == "unsupported"
    assert result["execution"] is None
    assert result["unsupported"] == [
        {"kind": "contract_feature", "contract": CONTRACT, "required": W8A8_FEATURE}
    ]


def test_preflight_accepts_an_adapter_wide_w8a8_token_without_a_refinement(
    tmp_path: Path,
) -> None:
    case = _w8a8_case(tmp_path)
    adapter = _adapter(
        tmp_path / "adapter.json",
        [PER_TENSOR, W8A8_FEATURE],
        with_contract_capabilities=False,
    )
    result = execute_adapter(case, EngineAdapter.load(adapter), tmp_path / "runs")
    assert "unsupported" not in result
    # The probe command declines to run, so reaching execution proves preflight passed.
    assert result["status"] == "execution_error"
    assert result["execution"] is not None
