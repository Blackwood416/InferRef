# InferRef 0.12 W8A8 INT32 accumulation contract — design freeze

- Status: **approved 2026-10-22** — user confirmed all recommended defaults;
  I3b starts from the merged Wave 1 baseline
- Scope: `linear/w8a8-int32-accum/v1` reference semantics, validation, and tests
- Related: [0.12 scope](InferRef_0.12_Core_Infrastructure.md) §6.2,
  [execution plan](InferRef_0.12_Execution_Plan.md) I3b
- Baseline: 0.11.0 `inferref.quantization.reference` affine INT8 arithmetic

## 1. Contract

| Field | Value |
| --- | --- |
| Contract ID | `linear/w8a8-int32-accum/v1` |
| Inputs | `x` (`int8`), `weight` (`int8`) |
| Outputs | `y` (`float32`) |
| Bias | none in v1 |
| Activation accumulation | `int32` per reduction group |
| Epilogue | `float32` scale application |
| Adapter feature | `quantization/w8a8-int32-accum/v1` plus the constituent `quantization/affine-int8/v1/<granularity>` tokens |

This is an integer-dot-product contract. It is not the existing
`linear/affine-int8/v1` (which dequantizes the weight to float32 first).

## 2. Tensors and descriptors

### 2.1 Shapes

- `x`: rank 2, `[M, K]` (`M` rows/tokens, `K` reduction dimension).
- `weight`: rank 2, `[N, K]` (`N` output features).
- `y`: rank 2, `[M, N]`, `float32`.
- `x.shape[-1] == weight.shape[1] == K`.
- Rank > 2 `x` is out of scope for v1; reject it. This keeps per-axis
  activation scales unambiguous.

### 2.2 Quantization descriptors

Both tensors carry an explicit `QuantizationSpec` with
`scheme="affine-int8/v1"`, `storage_dtype="int8"`,
`logical_shape` equal to the tensor shape, `compute_dtype="float32"`, and
`format="inferref-quantization"`, `format_version="1.0"`.

`rounding` and `saturation` describe how the stored codes were produced. They
are validated by `QuantizationSpec` but do not change the integer reference
(the reference consumes the stored codes as-is).

### 2.3 Allowed granularity and axes

| Tensor | Granularity | `axis` | `group_size` | Scale / zero-point shape |
| --- | --- | --- | --- | --- |
| `x` | `per_tensor` | `null` | `null` | `()` |
| `x` | `per_axis` | `-2` (normalized `0`, per row/token) | `null` | `(M,)` |
| `x` | `per_group` | `-1` (normalized `1`) | `G` | `(M, ceil(K/G))` |
| `weight` | `per_tensor` | `null` | `null` | `()` |
| `weight` | `per_axis` | `0` (per output channel) | `null` | `(N,)` |
| `weight` | `per_group` | `-1` (normalized `1`) | `G` | `(N, ceil(K/G))` |

Rules:

1. `x` per-axis with any axis other than `-2`/`0` is rejected.
2. `weight` per-axis with any axis other than `0` is rejected.
3. Per-group always uses the reduction axis `-1`; any other axis is rejected.
4. If **both** sides are `per_group`, their group sizes must be equal.
5. If only one side is `per_group`, the other side may be `per_tensor` or
   `per_axis`; its scale is constant inside every group.
6. `group_size >= 1`; `ceil(K/G)` is the number of groups.
7. Scales are ordinary unquantized `float32` tensors, finite and strictly
   positive. Zero points are ordinary unquantized `int8` tensors in
   `[-128, 127]`, with exactly the same shape as the scale. Missing zero point
   means `0`.

## 3. Integer accumulation

For output element `(m, n)` and group `g` (groups are ordered `0..G-1` along
`K`):

```text
acc_g =
    sum over k in group g of
        ( int32(x_code[m,k]) - int32(zx[m,g]) )
      * ( int32(w_code[n,k]) - int32(zw[n,g]) )
```

- If `zx` is absent, `zx = 0`; if `zw` is absent, `zw = 0`.
- For `per_tensor`, the zero point is a scalar.
- For `per_axis` `x`, `zx[m]` is constant over `k`; for `per_axis` `weight`,
  `zw[n]` is constant over `k`.
- For `per_group`, `zx[m,g]` and `zw[n,g]` are constant inside group `g`.
- Compute each group sum in `int64` and require
  `INT32_MIN <= acc_g <= INT32_MAX`. If a group sum is outside the `int32`
  range, raise `QuantizationError` ("int32 accumulation overflow in group g").
  Do not wrap or saturate.
- A group sum is exact integer arithmetic; no float64 accumulation is allowed
  inside the reduction.

The maximum per-element product is `255 * 255 = 65025`; overflow is possible
for long reductions with large zero-point offsets, so the explicit check is
required.

## 4. Float32 epilogue

Groups are accumulated in ascending `g` order. For each group:

```text
sx = float32(scale_x for this group)     # scalar
sw = float32(scale_w for this group)     # scalar
scale = float32(sx * sw)                 # one float32 multiply
term = float32(float32(acc_g) * scale)   # one float32 multiply
y[m,n] = float32(y[m,n] + term)          # one float32 add
```

`y` starts at `float32(0)`. If the final `term` or `y` is non-finite, raise
`QuantizationError` ("non-finite W8A8 output"). Do not reorder groups, do not
fuse the two scale multiplications into float64, and do not use `fma` that
changes the rounding of the two explicit multiplications.

For `per_tensor`/`per_axis` on both sides, `G = 1` and the single group covers
all `K` elements.

## 5. Reference API

Add to `inferref/quantization/reference.py`:

```python
def linear_w8a8_int32_reference(
    inputs,            # TensorView carrying int8 x codes
    input_spec,        # QuantizationSpec for x
    weights,           # TensorView carrying int8 weight codes
    weight_spec,       # QuantizationSpec for weight
    *,
    values,            # parameter value_id -> TensorView
    bias=None,
) -> np.ndarray:
    """int8 x int8 -> int32 group accumulation -> float32 scaled output."""
```

- `inputs` and `weights` are quantized code `TensorView`s (int8), while
  `input_spec` and `weight_spec` are their explicit descriptors.
- `values` follows the existing `TensorView` parameter convention.
- `bias` must be `None`; a non-`None` bias raises `QuantizationError`.
- Validate both descriptors, parameter closure, shapes, axes, group
  compatibility, scales, and zero points before allocating `y`.
- Return a C-contiguous `float32` array of shape `(M, N)`.

Export it from `inferref/quantization/__init__.py`.

## 6. Contract integration

Register:

```python
ExecutableContract(
    id="linear/w8a8-int32-accum/v1",
    inputs=("x", "weight"),
    outputs=("y",),
    validate_inputs=_w8a8_inputs,
    validate_relation=_w8a8_relation,
    features=("quantization/w8a8-int32-accum/v1",),
    description="int8xint8 per-group int32 accumulation, float32 epilogue, no bias",
)
```

- `_w8a8_inputs` performs the descriptor/shape/axis/group checks in §2 and
  verifies that the parameter tensors exist and have the exact shapes.
- `_w8a8_relation` requires `y` to be ordinary `float32` with shape `(M, N)`.
- The contract feature token is merged into testcase requirements by the
  contract registry. Add the token to the quantization feature vocabulary so
  adapter manifests can declare it.
- Preflight must return `unsupported` before launching an engine that does not
  declare `quantization/w8a8-int32-accum/v1` and every constituent
  affine-int8 granularity token.

## 7. Positive test matrix

| Case | `x` granularity | `weight` granularity | Notes |
| --- | --- | --- | --- |
| P1 | per-tensor, symmetric | per-tensor, symmetric | smallest oracle case, `K=3` |
| P2 | per-axis `-2`, symmetric | per-axis `0`, symmetric | per-token / per-channel |
| P3 | per-axis `-2`, asymmetric | per-axis `0`, asymmetric | explicit int8 zero points |
| P4 | per-group `G=4`, symmetric | per-group `G=4`, symmetric | `K=8` |
| P5 | per-group `G=4`, asymmetric | per-group `G=4`, asymmetric | zero points per group |
| P6 | per-group `G=4` | per-tensor | mixed granularity |
| P7 | per-tensor | per-group `G=8` | mixed granularity |
| P8 | per-axis `-2` | per-group `G=4` | mixed granularity |
| P9 | per-group `G=4`, `K=10` | per-group `G=4`, `K=10` | partial tail group |
| P10 | per-tensor | per-tensor | `K=1`, scalar dot product |
| P11 | per-group `G=1` | per-group `G=1` | per-element scales; verifies group count |

For every positive case:

1. Compute an independent `int64` group-sum oracle and assert the reference's
   internal/epilogue result matches the expected float32 values.
2. Compute a float64 dequantized oracle
   `(x - zx) * sx @ ((w - zw) * sw).T`; for exact power-of-two scales assert
   exact equality, otherwise assert a documented float32 tolerance.
3. Assert the output dtype, shape, contiguity, and no NaN/Inf.

## 8. Negative and injected-bug matrix

Must raise `QuantizationError` (or the contract validation error) before any
engine launch:

| Case | Expected rejection |
| --- | --- |
| N1 | `x` dtype `float32` or not `int8` |
| N2 | `weight` dtype `float32` or not `int8` |
| N3 | missing `quantization` on `x` or `weight` |
| N4 | wrong scheme (`affine-int4/v1`, `nf4/v1`) |
| N5 | `x` per-axis axis `-1` or `1` |
| N6 | `weight` per-axis axis `-1` or `1` |
| N7 | per-group axis other than `-1` |
| N8 | both per-group with different `G` |
| N9 | scale shape not equal to `parameter_shape` |
| N10 | zero-point shape not equal to scale shape |
| N11 | zero-point dtype not `int8` / out of range |
| N12 | scale non-finite, zero, or negative |
| N13 | missing scale/zero-point parameter |
| N14 | quantized scale/zero-point (nested quantization) |
| N15 | `K` mismatch between `x` and `weight` |
| N16 | `x` rank 3 |
| N17 | `bias` supplied |
| N18 | int32 group accumulation overflow |
| N19 | non-finite final output |
| N20 | `y` not `float32` / wrong shape (contract relation) |

Injected-bug tests (must fail the comparison, not silently pass):

- Wrong zero-point index (row versus group).
- Wrong group index (`k // G` vs `flat_index // G`).
- float64 accumulation instead of int32 (detectable on large accumulations).
- Missing scale multiplication.
- Scale applied before accumulation.
- Group order reversed in the epilogue.

## 9. Open questions for sign-off

**Approval:** on 2026-10-22 the user confirmed all recommended defaults below.
The freeze is closed; I3b implements these semantics without further scope
questions. Any later change requires a new freeze revision.

Recommended defaults are marked **[R]**.

1. **[R]** `x` is rank 2 `[M, K]`; rank > 2 rejected in v1.
2. **[R]** `x` per-axis is axis `-2` (per token); `weight` per-axis is axis `0`
   (per output channel).
3. **[R]** Explicit asymmetric `int8` zero points are allowed; absent means 0.
4. **[R]** Any group sum outside `int32` is a hard error; no wrap/saturate.
5. **[R]** Epilogue order is exactly: `sx*sw` in float32, then `acc*scale` in
   float32, then add in ascending group order.
6. **[R]** No bias in v1; a `bias` input in the reference is an error. A future
   `linear/w8a8-int32-accum-bias/v1` may be added later.
7. **[R]** `rounding`/`saturation` are descriptor metadata only; the reference
   uses the stored int8 codes.
8. **[R]** Adapter must declare
   `quantization/w8a8-int32-accum/v1` plus every constituent
   `quantization/affine-int8/v1/<granularity>` token.

If any answer differs, I3b changes the freeze document first, then implements.
