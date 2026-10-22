"""W8A8 int32-accumulation reference: freeze §7 positive and §8 negative matrices.

Every positive case is checked against two oracles that do not share code with
``inferref.quantization.reference``: an exact int64 group-sum oracle driven by
the frozen float32 epilogue, and a float64 dequantized matmul oracle
``(x - zx) * sx @ ((w - zw) * sw).T``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from inferref.quantization import (
    QuantizationError,
    QuantizationSpec,
    linear_w8a8_int32_reference,
)
from inferref.tensor import codec
from inferref.tensor.codec import TensorView

X_SCALE_ID, X_ZERO_ID, W_SCALE_ID, W_ZERO_ID = 11, 12, 13, 14
INT32_MAX = (1 << 31) - 1
F32_EPS = np.finfo(np.float32).eps
AXES = {"per_tensor": None, "per_axis": 0, "per_group": 1}


def tensor(data, dtype="float32"):
    array = np.asarray(data, dtype=dtype)
    return codec.decode(codec.encode(dtype=dtype, shape=array.shape,
                                    stride=codec.contiguous_stride(array.shape),
                                    payload=array.tobytes()))


def codes(rows, k_size, offset):
    """Deterministic int8 codes in [-125, 125] for the larger fixtures."""
    indices = np.arange(rows * k_size).reshape(rows, k_size)
    return ((indices * 7 + offset) % 251 - 125).astype(np.int8)


def _table(values, shape, dtype, label):
    table = np.asarray(values, dtype=dtype)
    if table.shape != shape:
        raise ValueError(f"{label} table must have shape {shape}, got {table.shape}")
    return table


def _declared_parameter(table, granularity):
    """Collapse a constant-per-group ``(rows, groups)`` table to ``parameter_shape``."""
    if granularity == "per_tensor":
        if not np.all(table == table[0, 0]):
            raise ValueError("per_tensor parameters must be constant")
        return table[0, 0]
    if granularity == "per_axis":
        if not np.all(table == table[:, :1]):
            raise ValueError("per_axis parameters must be constant along K")
        return table[:, 0]
    return table


@dataclass(frozen=True)
class Case:
    """One W8A8 fixture plus the constant-per-group tables its oracles need."""

    inputs: TensorView
    input_spec: QuantizationSpec
    weights: TensorView
    weight_spec: QuantizationSpec
    values: dict[int, TensorView]
    x_scale: np.ndarray
    x_zero: np.ndarray
    w_scale: np.ndarray
    w_zero: np.ndarray
    group_size: int

    @property
    def groups(self) -> int:
        return self.x_scale.shape[1]


def make_case(
    x_codes, w_codes, *, x_scale, w_scale, x_zero=None, w_zero=None,
    x_granularity="per_tensor", w_granularity="per_tensor", group_size=None,
) -> Case:
    """Build descriptors, parameter TensorViews and group tables for one case.

    ``x_scale``/``w_scale``/zero points are given as the ``(rows, groups)``
    tables the freeze prescribes; they are collapsed to the descriptor's exact
    ``parameter_shape`` when the granularity makes them constant.
    """
    x_codes = np.asarray(x_codes, dtype=np.int8)
    w_codes = np.asarray(w_codes, dtype=np.int8)
    if x_codes.ndim != 2 or w_codes.ndim != 2 or x_codes.shape[1] != w_codes.shape[1]:
        raise ValueError("W8A8 fixtures need rank-2 x [M, K] and weight [N, K]")
    m_size, k_size = x_codes.shape
    n_size = w_codes.shape[0]
    grouped = "per_group" in (x_granularity, w_granularity)
    if grouped and group_size is None:
        raise ValueError("per_group fixtures must declare group_size")
    group_size = group_size if grouped else max(k_size, 1)
    groups = -(-k_size // group_size)
    x_scale = _table(x_scale, (m_size, groups), np.float32, "x scale")
    w_scale = _table(w_scale, (n_size, groups), np.float32, "w scale")
    symmetric_x = x_zero is None
    symmetric_w = w_zero is None
    x_zero = (np.zeros((m_size, groups), np.int64) if symmetric_x
              else _table(x_zero, (m_size, groups), np.int64, "x zero point"))
    w_zero = (np.zeros((n_size, groups), np.int64) if symmetric_w
              else _table(w_zero, (n_size, groups), np.int64, "w zero point"))
    values = {
        X_SCALE_ID: tensor(_declared_parameter(x_scale, x_granularity), "float32"),
        W_SCALE_ID: tensor(_declared_parameter(w_scale, w_granularity), "float32"),
    }
    if not symmetric_x:
        values[X_ZERO_ID] = tensor(_declared_parameter(x_zero, x_granularity), "int8")
    if not symmetric_w:
        values[W_ZERO_ID] = tensor(_declared_parameter(w_zero, w_granularity), "int8")
    input_spec = QuantizationSpec(
        logical_shape=x_codes.shape, scale_value_id=X_SCALE_ID,
        zero_point_value_id=None if symmetric_x else X_ZERO_ID,
        granularity=x_granularity, axis=AXES[x_granularity],
        group_size=group_size if x_granularity == "per_group" else None)
    weight_spec = QuantizationSpec(
        logical_shape=w_codes.shape, scale_value_id=W_SCALE_ID,
        zero_point_value_id=None if symmetric_w else W_ZERO_ID,
        granularity=w_granularity, axis=AXES[w_granularity],
        group_size=group_size if w_granularity == "per_group" else None)
    return Case(tensor(x_codes, "int8"), input_spec, tensor(w_codes, "int8"), weight_spec,
                values, x_scale, x_zero, w_scale, w_zero, group_size)


UNSET = object()


def run(case: Case, *, inputs=UNSET, input_spec=UNSET, weights=UNSET, weight_spec=UNSET,
        values=UNSET, bias=None):
    """Call the reference, letting a caller override any argument (even with None)."""
    return linear_w8a8_int32_reference(
        case.inputs if inputs is UNSET else inputs,
        case.input_spec if input_spec is UNSET else input_spec,
        case.weights if weights is UNSET else weights,
        case.weight_spec if weight_spec is UNSET else weight_spec,
        values=case.values if values is UNSET else values, bias=bias)


# --------------------------------------------------------------------------- oracles


def oracle_group_sums(case: Case, index=None):
    """Exact int64 per-group sums, written independently of the reference.

    ``index`` selects the group of each ``(m, k)`` element so injected
    mis-grouping bugs can be expressed as an alternative table.
    """
    x_codes = case.inputs.data.astype(np.int64)
    w_codes = case.weights.data.astype(np.int64)
    m_size, k_size = x_codes.shape
    n_size = w_codes.shape[0]
    if index is None:
        index = np.broadcast_to(np.arange(k_size) // case.group_size, (m_size, k_size))
    sums = [np.zeros((m_size, n_size), dtype=np.int64) for _ in range(case.groups)]
    for m in range(m_size):
        for k in range(k_size):
            group = int(index[m, k])
            for n in range(n_size):
                sums[group][m, n] += ((x_codes[m, k] - case.x_zero[m, group])
                                      * (w_codes[n, k] - case.w_zero[n, group]))
    return sums


def oracle_float32(case: Case, sums, *, order=1):
    """Frozen epilogue: ``sx*sw``, then ``acc*scale``, added in group ``order``."""
    y = np.zeros((case.inputs.shape[0], case.weights.shape[0]), dtype=np.float32)
    for group in list(range(case.groups))[::order]:
        for m in range(y.shape[0]):
            for n in range(y.shape[1]):
                scale = np.float32(np.float32(case.x_scale[m, group])
                                   * np.float32(case.w_scale[n, group]))
                term = np.float32(np.float32(sums[group][m, n]) * scale)
                y[m, n] = np.float32(y[m, n] + term)
    return y


def oracle_dequantized(case: Case):
    """Float64 ``(x - zx) * sx @ ((w - zw) * sw).T``; powers of two stay exact."""
    k_size = case.inputs.shape[1]
    index = np.arange(k_size) // case.group_size
    x = (case.inputs.data.astype(np.float64) - case.x_zero[:, index]) * case.x_scale[:, index]
    w = (case.weights.data.astype(np.float64) - case.w_zero[:, index]) * case.w_scale[:, index]
    return x @ w.T


def _power_of_two(case: Case) -> bool:
    mantissa_x, _ = np.frexp(case.x_scale)
    mantissa_w, _ = np.frexp(case.w_scale)
    return bool(np.all(mantissa_x == np.float32(0.5)) and np.all(mantissa_w == np.float32(0.5)))


def assert_matches_dequantized(actual, dequantized, groups):
    """Documented tolerance: at most one float32 rounding per group step."""
    magnitude = max(1.0, float(np.max(np.abs(dequantized)))) if dequantized.size else 1.0
    np.testing.assert_allclose(actual, dequantized, rtol=16 * F32_EPS * max(groups, 1),
                               atol=16 * F32_EPS * magnitude)


def assert_case_matches_oracles(case: Case):
    actual = run(case)
    sums = oracle_group_sums(case)
    dequantized = oracle_dequantized(case)
    assert actual.dtype == np.float32
    assert actual.shape == (case.inputs.shape[0], case.weights.shape[0])
    assert actual.flags["C_CONTIGUOUS"]
    assert np.all(np.isfinite(actual))
    # The reference and this oracle perform the same IEEE float32 operations in
    # the same order, so they must agree bit for bit; this is what makes an
    # integer-dot-product reference (not a dequantize-then-multiply one) testable.
    np.testing.assert_array_equal(actual, oracle_float32(case, sums))
    if _power_of_two(case):
        # Power-of-two scales keep every term exact in float32, and the fixture
        # magnitudes stay below 2**24, so the float64 oracle is exact too.
        assert max(int(np.max(np.abs(group))) for group in sums) < (1 << 24)
        np.testing.assert_array_equal(actual, dequantized.astype(np.float32))
    else:
        assert_matches_dequantized(actual, dequantized, case.groups)


# ----------------------------------------------------------------- positive matrix


def _p1():
    """P1: per-tensor symmetric on both sides, K=3."""
    return make_case([[1, -2, 3], [4, 5, -6]], [[2, 0, -1], [-3, 1, 2]],
                     x_scale=[[0.5], [0.5]], w_scale=[[0.25], [0.25]])


def _p2():
    """P2: per-axis -2 (per token) x, per-axis 0 (per channel) weight."""
    return make_case(
        [[1, -2, 3, -4], [5, 6, -7, 8]],
        [[1, 2, -3, 0], [-2, 1, 0, 3], [4, -5, 6, -7]],
        x_scale=[[0.5], [0.25]], w_scale=[[0.5], [1.0], [2.0]],
        x_granularity="per_axis", w_granularity="per_axis")


def _p3():
    """P3: per-axis asymmetrical with explicit int8 zero points."""
    return make_case(
        [[1, -2, 3, -4], [5, 6, -7, 8]],
        [[1, 2, -3, 0], [-2, 1, 0, 3], [4, -5, 6, -7]],
        x_scale=[[0.5], [0.25]], w_scale=[[0.5], [0.125], [2.0]],
        x_zero=[[-1], [2]], w_zero=[[0], [-128], [127]],
        x_granularity="per_axis", w_granularity="per_axis")


def _p4():
    """P4: per-group G=4 symmetric on both sides, K=8."""
    return make_case(
        [[1, -2, 3, -4, 5, -6, 7, -8], [8, 7, 6, 5, 4, 3, 2, 1]],
        [[1, 0, -1, 2, -3, 4, -5, 6], [-6, 5, -4, 3, -2, 1, 0, -1]],
        x_scale=[[0.5, 0.25], [0.125, 1.0]], w_scale=[[0.25, 0.5], [1.0, 0.125]],
        x_granularity="per_group", w_granularity="per_group", group_size=4)


def _p5():
    """P5: per-group G=4 asymmetrical with per-group zero points."""
    return make_case(
        [[1, -2, 3, -4, 5, -6, 7, -8], [8, 7, 6, 5, 4, 3, 2, 1]],
        [[1, 0, -1, 2, -3, 4, -5, 6], [-6, 5, -4, 3, -2, 1, 0, -1]],
        x_scale=[[0.1, 0.3], [0.7, 1.7]], w_scale=[[0.2, 0.05], [1.3, 0.9]],
        x_zero=[[3, -5], [-7, 9]], w_zero=[[-1, 4], [2, -128]],
        x_granularity="per_group", w_granularity="per_group", group_size=4)


def _p6():
    """P6: mixed granularity, per-group x with per-tensor weight."""
    return make_case(
        [[1, -2, 3, -4, 5, -6, 7, -8], [-1, 2, -3, 4, -5, 6, -7, 8]],
        [[2, 0, -1, 3, -2, 1, 0, 4], [1, -1, 2, -2, 3, -3, 4, -4]],
        x_scale=[[0.5, 0.25], [1.0, 0.125]], w_scale=[[0.5, 0.5], [0.5, 0.5]],
        x_granularity="per_group", group_size=4)


def _p7():
    """P7: mixed granularity, per-tensor x with per-group G=8 weight, K=16."""
    return make_case(
        codes(2, 16, 3), codes(3, 16, 11),
        x_scale=[[0.5, 0.5], [0.5, 0.5]],
        w_scale=[[0.25, 0.5], [1.0, 0.125], [2.0, 0.5]],
        w_granularity="per_group", group_size=8)


def _p8():
    """P8: mixed granularity, per-axis x with per-group G=4 weight."""
    return make_case(
        codes(2, 8, 5), codes(2, 8, 9),
        x_scale=[[0.5, 0.5], [1.0, 1.0]], w_scale=[[0.25, 0.5], [1.0, 0.125]],
        w_zero=[[1, -2], [3, -128]],
        x_granularity="per_axis", w_granularity="per_group", group_size=4)


def _p9():
    """P9: per-group G=4 with K=10, so the last group is a shorter tail."""
    return make_case(
        codes(2, 10, 7), codes(3, 10, 1),
        x_scale=[[0.3, 0.7, 1.1], [0.05, 0.15, 0.25]],
        w_scale=[[0.2, 0.9, 0.45], [1.3, 0.35, 0.6], [0.05, 2.5, 0.85]],
        x_zero=[[2, -3, 4], [-5, 6, -7]], w_zero=[[1, -1, 0], [-128, 127, 3], [0, 9, -9]],
        x_granularity="per_group", w_granularity="per_group", group_size=4)


def _p10():
    """P10: per-tensor both sides with K=1, a scalar dot product."""
    return make_case(
        [[-128], [0], [127]], [[1], [-128], [127], [7]],
        x_scale=[[0.5], [0.5], [0.5]],
        w_scale=[[0.25], [0.25], [0.25], [0.25]])


def _p11():
    """P11: per-group G=1 on both sides, so every element has its own scale."""
    return make_case(
        [[1, -2, 3], [4, -5, 6]], [[7, -8, 9], [-10, 11, -12]],
        x_scale=[[0.5, 0.25, 2.0], [1.0, 0.125, 0.5]],
        w_scale=[[0.25, 1.0, 0.5], [2.0, 0.0625, 0.5]],
        x_granularity="per_group", w_granularity="per_group", group_size=1)


POSITIVE_CASES = {
    "P1": _p1, "P2": _p2, "P3": _p3, "P4": _p4, "P5": _p5, "P6": _p6,
    "P7": _p7, "P8": _p8, "P9": _p9, "P10": _p10, "P11": _p11,
}


@pytest.mark.parametrize("name", sorted(POSITIVE_CASES, key=lambda key: int(key[1:])))
def test_positive_matrix_matches_group_sum_and_dequantized_oracles(name):
    case = POSITIVE_CASES[name]()
    assert_case_matches_oracles(case)


def test_group_sizes_and_partial_tail_are_the_frozen_ones():
    assert _p4().groups == 2 and _p4().group_size == 4
    assert _p7().groups == 2 and _p7().group_size == 8
    assert _p9().groups == 3 and _p9().group_size == 4
    assert _p11().groups == 3 and _p11().group_size == 1
    assert _p1().groups == 1 and _p1().group_size == 3


# ----------------------------------------------------------------- negative matrix


def base_case():
    """Small per-tensor symmetric case used by the negative matrix."""
    return make_case([[1, 2, 3], [-1, 0, 1]], [[2, -1, 0], [1, 1, 1]],
                     x_scale=[[0.5], [0.5]], w_scale=[[0.25], [0.25]])


def asymmetric_case():
    """Per-axis case with explicit zero points on both sides."""
    return make_case([[1, 2, 3], [-1, 0, 1]], [[2, -1, 0], [1, 1, 1]],
                     x_scale=[[0.5], [0.25]], w_scale=[[0.25], [0.5]],
                     x_zero=[[1], [-2]], w_zero=[[0], [3]],
                     x_granularity="per_axis", w_granularity="per_axis")


@pytest.mark.parametrize("side", ["input", "weight"])
def test_n1_n2_codes_must_be_int8(side):
    case = base_case()
    quantized = case.inputs if side == "input" else case.weights
    widened = tensor(quantized.data, "float32")
    with pytest.raises(QuantizationError, match="int8"):
        if side == "input":
            run(case, inputs=widened)
        else:
            run(case, weights=widened)


@pytest.mark.parametrize("side", ["input", "weight"])
def test_n3_descriptor_is_required(side):
    case = base_case()
    with pytest.raises(QuantizationError, match="QuantizationSpec"):
        if side == "input":
            run(case, input_spec=None)
        else:
            run(case, weight_spec=None)


@pytest.mark.parametrize("spec_kwargs", [
    {"scheme": "affine-int4/v1", "storage_dtype": "int4"},
    {"scheme": "nf4/v1", "storage_dtype": "uint4", "granularity": "per_block",
     "axis": None, "group_size": 2, "codebook_value_id": 21,
     "codebook": "bitsandbytes-nf4/v1", "rounding": "none", "saturation": "none"},
])
@pytest.mark.parametrize("side", ["input", "weight"])
def test_n4_wrong_scheme_is_rejected(side, spec_kwargs):
    case = base_case()
    wrong = QuantizationSpec(logical_shape=(2, 3), scale_value_id=X_SCALE_ID, **spec_kwargs)
    with pytest.raises(QuantizationError, match="affine-int8"):
        if side == "input":
            run(case, input_spec=wrong)
        else:
            run(case, weight_spec=wrong)


@pytest.mark.parametrize("axis", [1, -1])
def test_n5_input_per_axis_must_use_the_row_axis(axis):
    # axis=-1 is rejected by the descriptor itself; axis=1 reaches the reference.
    case = base_case()
    values = {**case.values, X_SCALE_ID: tensor(np.ones(3, np.float32))}
    with pytest.raises(QuantizationError):
        spec = replace(case.input_spec, granularity="per_axis", axis=axis)
        run(case, input_spec=spec, values=values)


@pytest.mark.parametrize("axis", [1, -1])
def test_n6_weight_per_axis_must_use_the_row_axis(axis):
    case = base_case()
    values = {**case.values, W_SCALE_ID: tensor(np.ones(3, np.float32))}
    with pytest.raises(QuantizationError):
        spec = replace(case.weight_spec, granularity="per_axis", axis=axis)
        run(case, weight_spec=spec, values=values)


@pytest.mark.parametrize("side", ["input", "weight"])
def test_n7_per_group_must_use_the_reduction_axis(side):
    case = base_case()
    spec = replace((case.input_spec if side == "input" else case.weight_spec),
                   granularity="per_group", axis=0, group_size=2)
    scale_id = X_SCALE_ID if side == "input" else W_SCALE_ID
    values = {**case.values, scale_id: tensor(np.ones((1, 3), np.float32))}
    with pytest.raises(QuantizationError, match="reduction axis"):
        if side == "input":
            run(case, input_spec=spec, values=values)
        else:
            run(case, weight_spec=spec, values=values)


def test_n8_both_per_group_require_equal_group_sizes():
    case = make_case(codes(2, 8, 3), codes(2, 8, 5),
                     x_scale=[[0.5, 0.25], [1.0, 0.5]], w_scale=[[0.25, 0.5], [1.0, 0.125]],
                     x_granularity="per_group", w_granularity="per_group", group_size=4)
    mismatched = replace(case.weight_spec, group_size=2)
    values = {**case.values, W_SCALE_ID: tensor(np.ones((2, 4), np.float32))}
    with pytest.raises(QuantizationError, match="equal group sizes"):
        run(case, weight_spec=mismatched, values=values)


def test_n9_scale_shape_must_equal_parameter_shape():
    case = base_case()
    values = {**case.values, X_SCALE_ID: tensor(np.float32([0.5, 0.5]))}
    with pytest.raises(QuantizationError, match="shape"):
        run(case, values=values)


def test_n10_zero_point_shape_must_equal_scale_shape():
    case = asymmetric_case()
    values = {**case.values, X_ZERO_ID: tensor(np.zeros((1,), np.int8))}
    with pytest.raises(QuantizationError, match="shape"):
        run(case, values=values)


@pytest.mark.parametrize("zero", [np.zeros((), np.float32), np.zeros((), np.uint8)])
def test_n11_zero_point_must_be_an_int8_carrier(zero):
    # An int8 carrier cannot hold a value outside [-128, 127]; the range check is
    # defensive, and a wrongly typed carrier is the reachable rejection.
    case = asymmetric_case()
    values = {**case.values, X_ZERO_ID: tensor(zero, str(zero.dtype))}
    with pytest.raises(QuantizationError, match="zero point"):
        run(case, values=values)


@pytest.mark.parametrize("scale", [np.inf, np.nan, 0.0, -1.0])
def test_n12_scales_must_be_finite_and_strictly_positive(scale):
    case = base_case()
    values = {**case.values, W_SCALE_ID: tensor(np.float32(scale))}
    with pytest.raises(QuantizationError, match="finite and strictly positive"):
        run(case, values=values)


def test_n13_missing_parameter_values_are_rejected():
    case = asymmetric_case()
    without_scale = {k: v for k, v in case.values.items() if k != W_SCALE_ID}
    without_zero = {k: v for k, v in case.values.items() if k != X_ZERO_ID}
    with pytest.raises(QuantizationError, match="missing W8A8 weight scale"):
        run(case, values=without_scale)
    with pytest.raises(QuantizationError, match="missing W8A8 input zero point"):
        run(case, values=without_zero)


def test_n14_nested_quantized_parameters_are_not_unquantized_tensors():
    # The reference only sees unquantized carriers, so a quantized (int8) scale
    # payload is rejected here; the record-level "quantization" check on
    # parameters belongs to the contract validator.
    case = base_case()
    values = {**case.values, X_SCALE_ID: tensor(np.int8(1), "int8")}
    with pytest.raises(QuantizationError, match="float32"):
        run(case, values=values)


def test_n15_reduction_size_mismatch_is_rejected():
    case = base_case()
    weights = tensor(np.zeros((2, 4), np.int8), "int8")
    weight_spec = replace(case.weight_spec, logical_shape=(2, 4))
    with pytest.raises(QuantizationError, match="reduction size mismatch"):
        run(case, weights=weights, weight_spec=weight_spec)


def test_n16_input_rank_must_be_two():
    case = base_case()
    inputs = tensor(np.zeros((1, 2, 3), np.int8), "int8")
    input_spec = replace(case.input_spec, logical_shape=(1, 2, 3))
    with pytest.raises(QuantizationError, match="rank 2"):
        run(case, inputs=inputs, input_spec=input_spec)


def test_n17_bias_is_rejected():
    case = base_case()
    with pytest.raises(QuantizationError, match="no bias"):
        run(case, bias=np.zeros((2,), np.float32))


def overflow_case():
    """One group of 40000 elements: 40000 * (127 + 128) * (127 + 128) > INT32_MAX."""
    k_size = 40000
    return make_case(
        np.full((1, k_size), 127, np.int8), np.full((1, k_size), 127, np.int8),
        x_scale=[[1.0]], w_scale=[[1.0]], x_zero=[[-128]], w_zero=[[-128]])


def test_n18_int32_group_accumulation_overflow_is_a_hard_error():
    case = overflow_case()
    sums = oracle_group_sums(case)
    assert int(sums[0][0, 0]) > INT32_MAX
    with pytest.raises(QuantizationError, match="int32 accumulation overflow in group 0"):
        run(case)


def test_n19_non_finite_output_is_rejected():
    case = make_case([[32]], [[32]], x_scale=[[1e20]], w_scale=[[1e20]])
    with pytest.raises(QuantizationError, match="non-finite"):
        run(case)


# --------------------------------------------------------------- injected-bug matrix


def test_injected_wrong_zero_point_index_is_detected():
    # Indexing the per-group zero point by row only (row versus group).
    case = _p5()
    actual = run(case)
    first_group_only = np.repeat(case.x_zero[:, :1], case.groups, axis=1)
    assert np.any(first_group_only != case.x_zero)
    buggy = replace(case, x_zero=first_group_only)
    assert not np.array_equal(actual, oracle_float32(buggy, oracle_group_sums(buggy)))


def test_injected_wrong_group_index_is_detected():
    # Grouping by flattened element index instead of by the reduction index k.
    case = _p4()
    actual = run(case)
    m_size, k_size = case.inputs.shape
    flat = (np.arange(m_size)[:, None] * k_size + np.arange(k_size)[None, :]) // case.group_size
    flat = np.minimum(flat, case.groups - 1)
    correct = np.broadcast_to(np.arange(k_size) // case.group_size, (m_size, k_size))
    assert np.any(flat != correct)
    # The oracle harness is sound: the correct table reproduces the reference.
    np.testing.assert_array_equal(actual, oracle_float32(case, oracle_group_sums(case, correct)))
    assert not np.array_equal(actual, oracle_float32(case, oracle_group_sums(case, flat)))


def test_injected_float64_accumulation_is_detectable():
    # Terms are at most 255*255, so only the int32 range check distinguishes an
    # int32 accumulator from a float64 one; without it the result would be finite.
    case = overflow_case()
    sums = oracle_group_sums(case)
    assert np.all(np.isfinite(oracle_float32(case, sums)))
    with pytest.raises(QuantizationError, match="overflow"):
        run(case)


def test_injected_missing_scale_multiplication_is_detected():
    case = _p1()
    actual = run(case)
    buggy = np.zeros_like(actual)
    for group in oracle_group_sums(case):
        buggy += group.astype(np.float32)
    assert not np.allclose(actual, buggy)


def test_injected_scale_before_accumulation_is_detected():
    # K=64 terms with non-power-of-two scales: dequantizing the operands first
    # rounds every product, which the frozen epilogue never does. On this
    # platform 8 of 12 elements differ, by 1-2 float32 ulps.
    case = make_case(codes(3, 64, 3), codes(4, 64, 7),
                     x_scale=[[0.1], [0.1], [0.1]],
                     w_scale=[[0.3], [0.3], [0.3], [0.3]])
    actual = run(case)
    k_size = case.inputs.shape[1]
    index = np.arange(k_size) // case.group_size
    x = ((case.inputs.data.astype(np.float32) - case.x_zero[:, index].astype(np.float32))
         * case.x_scale[:, index])
    w = ((case.weights.data.astype(np.float32) - case.w_zero[:, index].astype(np.float32))
         * case.w_scale[:, index])
    assert not np.array_equal(actual, x @ w.T)


def test_injected_reversed_group_order_is_detected():
    # Terms 2**24, 1, 1: ascending order rounds both adds away, reversed order
    # does not, so the float32 additions are order sensitive here.
    case = make_case([[1, 1, 1]], [[1, 1, 1]], x_scale=[[2.0**24, 1.0, 1.0]],
                     w_scale=[[1.0, 1.0, 1.0]],
                     x_granularity="per_group", w_granularity="per_group", group_size=1)
    actual = run(case)
    sums = oracle_group_sums(case)
    ascending = oracle_float32(case, sums)
    reversed_order = oracle_float32(case, sums, order=-1)
    np.testing.assert_array_equal(actual, ascending)
    assert not np.array_equal(ascending, reversed_order)
