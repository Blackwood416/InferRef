# Native FP8 and quantization development plan

Status: 0.11.0 shipped the Q1–Q4 milestones below. Historical acceptance
sections are preserved as dated evidence; see the [0.11.0 release notes](../releases/0.11.0.md)
and the [2026-10-22 release audit](../dev/status/2026-10-22-v0110-release-audit.md).
C++ execution of quantized testcases remains 0.12 scope.
This document distinguishes shipped code from planned work; it does not claim
that an FP8-capable reader can execute an FP8 kernel or a quantized model.

## Ordered milestones

Implementation and acceptance are owned by the primary agent in this task.
Each milestone must pass its own checks before the next changes the wire or
semantic contract. No release/version bump is implied by this plan.

| Milestone | Scope | State |
| --- | --- | --- |
| Q1 | Native E4M3FN/E5M2 storage, decoding, comparison, capture, interoperability | Implemented; local validation recorded below |
| Q2 | Versioned quantized-tensor semantics, exercised with existing INT8 storage | Implemented; local integration acceptance below |
| Q3 | Packed INT4/UINT4 and affine quantized Linear | Implemented; local acceptance below |
| Q4 | Explicit NF4 codebook/block-scale semantics and importer | Implemented; flat profile plus real QuantState bridge below |

FNUZ variants are a separate follow-up to Q1, not aliases for the implemented
FP8 types. A native FP8 encoder/casting policy is also outside Q1: callers
provide bytes already encoded by their framework or engine.

## Q1: native FP8 contract

| IR dtype | Wire code | Bytes/element | Bias | Special encodings |
| --- | --- | --- | --- | --- |
| `float8_e4m3fn` | 16 | 1 | 7 | Magnitude `0x7f` is NaN; no infinity |
| `float8_e5m2` | 17 | 1 | 15 | Magnitude `0x7c` is infinity; `0x7d`–`0x7f` are NaN |

Both have signed zero and subnormals. E4M3FN's largest finite magnitude is
448; E5M2's is 57344. Minimum positive subnormals are 2^-9 and 2^-16,
respectively. The sign bit is bit 7. Definitions follow the
[ONNX FP8 encoding tables](https://onnx.ai/onnx/technical/float8.html).

The `.irtensor` header and version stay at 1: the canonical payload is still
one byte per logical element. Existing codes 1–15 are unchanged. Old readers
reject codes 16/17 as unknown. Current readers must not silently substitute
uint8, FNUZ, or any scaled format when a dtype is unrecognized.

### Python / C++ use

```python
import numpy as np
from inferref.tensor import read, write_array

# Already-encoded E4M3FN bytes: +0, -0, 1, -2, max finite, NaN.
raw = np.array([0x00, 0x80, 0x38, 0xc0, 0x7e, 0x7f], dtype=np.uint8)
write_array("fp8.irtensor", raw, dtype="float8_e4m3fn")
t = read("fp8.irtensor")
assert t.data.dtype == np.uint8  # lossless bit-pattern carrier
values = t.as_float32()         # decoded numerical values
```

`float8_bytes_to_float32(raw, dtype)` is also available from `inferref.tensor`.
It requires uint8 input. `write_array(..., dtype=FP8)` rejects numeric float
arrays instead of truncating their values into integer carrier bytes. Omitting
the dtype on a uint8 array continues to mean uint8, not FP8. `encode(payload=)`
and native C++ `WriteIRTensor` accept already-encoded canonical bytes.
C++ `IRTensor::AsFloat32()` and `DecodeFloat8()` perform software widening.

The PyTorch frontend captures native tensors whose exact Torch dtype is one
of these names, including views, scalar tensors, and empty tensors. Capture
does not require `Tensor.numpy()` or assume FP8 arithmetic is implemented on
the selected device. Torch is still optional for reading/comparing tensors.

Comparison decodes FP8 before computing metrics; bytes are not treated as
integers. Default atol/rtol are both zero. Existing numerical semantics apply:
signed zeros compare equal, matching NaNs compare equal, and infinity signs
must agree. Byte equality is a distinct check (`data`/payload/hash), not the
numerical comparator's promise. Explicit tolerance overrides remain supported.
Different dtype names are incompatible even when their decoded values agree.

Adapter `capabilities.dtypes` must declare the exact FP8 name. A uint8-only
adapter is unsupported for FP8, even though the payload carrier is uint8.
No built-in example adapter is automatically upgraded to run FP8 kernels.

### Q1 acceptance

- Exhaust all 256 byte patterns per dtype with independently constructed
  exponent-bin oracles, including signed zero/subnormals/NaN/Inf.
- Compare all encodings against Torch in frontend tests when dtype is exposed.
- Core decoder tests run without importing Torch or requiring `ml_dtypes`.
- Raw payloads survive Python -> C++ -> Python byte-for-byte, including NaNs.
- Test scalar/empty/noncontiguous capture, invalid payload lengths, explicit
  adapter declarations, and trace CLI pass/fail/tolerance paths.
- Existing dtype codes, headers, and regression tests remain valid.

## Q2 tasks: quantization metadata, first with INT8

Q2 is split into two independently verifiable tasks. **Q2a is implemented**:
strict standalone metadata, affine INT8 numerical primitives, a float32 Linear
reference, and dependency-sensitive content fingerprinting. **Q2b is implemented**:
versioned trace/testcase integration, transitive extraction, adapter preflight,
comparison domains, and Alignment cache/evidence wiring. See the
[complete INT8 workflow](../../examples/quantization/README.md) for supported
execution and explicit unsupported paths.

### Q2a standalone API (unreleased)

```python
from inferref.quantization import (
    QuantizationSpec, dequantize_affine, quantize_affine,
    linear_int8_reference, quantization_fingerprint,
)
```

`QuantizationSpec` parsing requires only the standard library. Numerical APIs
load numpy lazily and never require Torch. An executable example is:

```bash
python examples/quantization/int8_reference.py
```

It returns decoded weights `[[1, 2, 1.5], [-2, 0, 4]]` and Linear output
`[[9.5, 10.0]]`. This is a reference calculation, not an engine adapter or a
trace exporter; `int8_trace.py` and `int8_numpy.adapter.json` implement the
separate end-to-end example.

The canonical descriptor has `format="inferref-quantization"`, exact
`format_version="1.0"`, and `scheme="affine-int8/v1"`. Every field, including
nulls and defaults, must be present in JSON; missing fields, unknown fields,
unknown schemes/versions, booleans in integer fields, and invalid dimensions
are rejected. Programmatic constructors provide explicit documented defaults.

| Field | Current contract |
| --- | --- |
| `storage_dtype` | `int8` only; raw codes retain signed INT8 semantics |
| `logical_shape` | Nonnegative signed-64-bit dimensions; scalar and empty supported |
| `scale_value_id` | Required nonnegative integer tensor-value ID |
| `zero_point_value_id` | Distinct integer ID, or explicit null meaning symmetric zero |
| `granularity` | `per_tensor`, `per_axis`, or `per_group` |
| `axis` | Null for per-tensor; canonical nonnegative axis otherwise |
| `group_size` | Positive integer for per-group only; explicit null otherwise |
| `compute_dtype` | `float32` |
| `rounding` | `nearest_even`, applied to the float32 quotient before adding zero point |
| `saturation` | `clamp` to [-128, 127] |

Parameters are supplied as an explicit `values: Mapping[int, TensorView]`.
Scales must have float32 payloads, be finite and strictly positive. Zero points
must have int8 payloads (no implicit conversion from out-of-range integers).
Parameter shapes are exact, not arbitrary broadcastable arrays:

- Per-tensor: rank-0 scalar `()`.
- Per-axis: vector `(logical_shape[axis],)`, shared along all other axes.
- Per-group: logical shape with the grouped axis replaced by
  `ceil(logical_shape[axis] / group_size)`. Other axes have independent
  parameters. The final group can be shorter. A zero-length grouped axis has
  zero groups; empty scale arrays contain no invalid entries.

`dequantize_affine(codes, spec, values=parameters)` computes
`(float32(codes) - float32(zero_point)) * scale`, returning a float32 array.
`quantize_affine(source, spec, values=parameters)` requires finite float32
source, rounds the float32 division to nearest even, adds zero point, clamps
and returns an int8 array. Finite division overflow saturates; dequantization
overflow is an explicit error. Float32 underflow is allowed independently of
the caller's numpy error settings. No implicit float64/float16 casting occurs.
These are specified arithmetic rules, not a promise that every framework's
reciprocal-based quantizer is bit-identical for every scale.

`linear_int8_reference(inputs, weights, spec, values=parameters, bias=None)`
uses weights `[out_features, in_features]`, float32 inputs `[..., in_features]`,
and optional float32 bias `[out_features]`. It computes
`inputs @ dequantize(weights).T + bias` with float32 outputs. This is **not** a
W8A8 integer-accumulation kernel. Compare engines against the same codes and
parameters; compare against original unquantized weights only when separately
measuring quantization loss. BLAS reduction order can vary by platform.

`quantization_fingerprint` hashes a domain-separated canonical descriptor and
length-delimited encoded bytes for codes, scale and zero point. Changing any
referenced payload or semantic field invalidates it; changing filesystem paths
or mapping insertion order does not. Layout metadata is included conservatively.
Alignment separately hashes the entire testcase tree and binds quantized trace
identity to graph metadata plus all captured code and parameter payloads.
It does not substitute this weights-only fingerprint for full artifact identity.

Legacy trace/testcase records reject a core `quantization` field rather
than silently comparing the underlying integers. Trace 1.0 supports the typed
`TensorValueRecord.quantization` field; extraction emits testcase 0.4.
Do not hide required semantics
in namespaced extensions: they remain optional, uninterpreted vendor data.
Nested/quantized parameter tensors and reference cycles are not supported by
this standalone API: parameters must be ordinary unquantized `TensorView`s.
Plain writers retain their existing format-version defaults.

### Q2b integration checklist

The following checklist is implemented for affine INT8 and extended by Q3/Q4
below. Q4 adds codebook dependencies. Nested parameter tensors remain unsupported
in the IR; the separate real QuantState importer expands one scale level first.

1. Define a versioned `QuantizationSpec` with explicit scheme, logical shape,
   storage dtype, scale/zero-point tensor references, granularity (per-tensor,
   per-axis, per-group), axis/group mapping, compute/output dtype, and casting
   rules. Specify symmetric zero points and tail groups; do not infer them.
2. Add a fail-closed format/capability gate. Required semantics cannot live only
   in an extension old readers ignore. Decide trace/testcase version migration
   before writing artifacts. Keep old non-quantized artifacts readable.
3. Validate scale/zero-point dtype, shape, finiteness, positivity where required,
   integer range, missing references, cycles, and incompatible axes/groups.
4. Extend extraction and copying with transitive quantization dependencies:
   scales, zero points, codebooks and nested state must travel with the testcase.
5. Extend adapter preflight with scheme/granularity support, not merely storage
   dtype support. Unknown/unsupported schemes must stop before engine execution.
6. Extend testcase/content hashes and alignment evidence to include every
   semantic parameter and referenced artifact; changed scales invalidate reuse.
7. Implement numpy reference dequantization for affine INT8 and a small explicit
   quantized Linear contract. Verify the engine against the *same quantized
   weights*, not full-precision original weights. Keep accumulation precision
   and output rounding explicit.
8. Integrate comparison/reporting, extraction, Agent, Suite and Alignment paths;
   add a runnable documented positive case and injected-error negative case.

Acceptance: malformed metadata is rejected, no-Torch decoding works, dependency
closure survives relocation, parameter changes invalidate cached evidence,
unsupported engines never launch, and the INT8 Linear example passes/fails in
the expected directions. Define raw-code, dequantized-value, and kernel-output
comparison modes explicitly; do not conflate quantization loss with kernel bugs.

Migration constraint: old trace readers accept every `0.x` version, so a new
minor version alone cannot protect required quantization semantics. Q2b must
use a rejecting major-version gate for quantized traces (with legacy plain
trace reading retained), and a new explicitly recognized testcase version.

### Q2b implemented integration design

The stdlib-only `inferref.quantization.dependencies.quantization_dependencies`
projection helper is implemented and tested independently. It returns sorted
value/parameter/quantized ID sets, preserves shared parameters and parameters
already selected as roots, rejects missing/duplicate IDs and cycles, and rejects
nested quantized parameters. Its iterative traversal handles malformed chains
beyond Python's recursion limit. Extraction and validation share this helper;
artifact validators additionally check descriptor/payload metadata and parameter
contents.

The implementation follows these gates. Parsing a new version alone never
authorizes execution of an incomplete or unsupported artifact.

1. **Version gate and typed records.** Use trace `1.0` and testcase `0.4` for
   quantized artifacts only. Plain writers retain their existing versions.
   Introduce a typed `TensorValueRecord.quantization` descriptor and reject
   it under legacy versions. A future trace `1.x` minor version is not
   automatically supported. Pre-Q2 Python and current C++ testcase readers
   reject `0.4`; pre-Q2 trace readers reject major `1`. Current Python readers
   support the new versions while retaining legacy reading.
2. **Dependency closure and extraction.** Project scale/zero-point value IDs
   along with quantized roots. Reject missing IDs, self references, cycles,
   nested quantized parameters, and conflicting duplicate identities before
   copying anything. Parameter records are included in projected `values`;
   their payloads live in a separate `quantization_parameters` collection and
   directory. They are not appended to executable operator input roles.
   Validate exact descriptor, metadata and payload consistency between all
   appearances of a value ID. Extraction stays staged and atomic.
3. **Evidence integrity.** Copied parameter payloads participate in testcase
   tree hashing. Quantized trace identity must also bind the code and parameter
   payloads, not just metadata JSON: otherwise editing a scale in place can
   leave an existing Alignment session looking current. Add regression tests
   for changed content with unchanged paths, relocation, missing files, and
   path/symlink containment. Do not replace the full testcase digest with a
   weights-only fingerprint.
4. **Capability preflight and execution.** Derive versioned quantization
   feature requirements by scheme/granularity, including parameter dtypes and
   ranks. Require explicit support; a legacy adapter with no capability
   declaration must not execute a quantized testcase as `unchecked`.
   Planner and runner must consume the same expanded requirements. Existing
   float/INT8-only adapters remain unsupported until explicitly extended.
5. **Comparison semantics and complete example.** Report whether comparison
   concerns raw codes, dequantized values, or float32 kernel outputs. For
   dequantized comparison, resolve each side's own parameters so equal codes
   with different scales cannot pass accidentally. Until a mode is implemented,
   reject it; do not silently fall back to raw integer comparison. Finish with
   an explicit affine INT8 Linear contract, an independent numpy adapter,
   pass/injected-error examples, and Agent/Suite/Alignment smoke tests.

Release gate: all five items must pass before the README advertises an
end-to-end quantized trace workflow. Q3 packed storage work depends on that gate.

Current comparison ABI:

- Trace comparison defaults to `dequantized`, resolving each side's parameters.
  API `quantization_mode="raw_codes"` / CLI `--quantization-mode raw_codes`
  explicitly compares only codes/layouts; the report names its domain.
- Ordinary float32 kernel outputs compare normally and report
  `float32_kernel_output` for quantized testcases. Quantized testcase reference
  outputs are dequantized against float32 engine outputs; a bare INT8 engine
  output is not interpreted using reference scales.
- Quantized engine-output manifests and custom/per-output comparator policies
  on quantized testcase outputs fail closed. Use two traces for two quantized
  outputs with independent descriptors. No automatic Torch qint8 capture or C++
  testcase 0.4 execution is implied.
- `linear/affine-int8/v1` binds `x,weight -> y`, requires float32 x/y and an
  explicit affine INT8 weight, computes float32 x @ dequantize(weight).T, and
  has no bias. The standalone reference API also supports optional bias.
- Adapter v0.2 accepts the three exact features
  `quantization/affine-int8/v1/per_tensor`, `/per_axis`, `/per_group`.
  Parameter dtype/rank requirements are included. Undeclared legacy adapters
  cannot execute these testcases as unchecked.

## Q3 tasks: packed INT4/UINT4

Implemented wire contract: `int4` code 18 and `uint4` code 19 require tensor
format **2**; byte-addressable types still require format 1. Header layout is
unchanged, but `payload_nbytes = ceil(logical_numel / 2)`. The first logical
element is the low nibble; odd tails have a zero high nibble. Both Python and
C++ reject noncanonical padding, truncation, shape/length inconsistencies, and
dtype/version mismatches. Old readers reject version 2. No hardware blocking is
inferred. `dtype_bits` / `dtype_payload_nbytes` replace size multiplication;
`dtype_itemsize` rejects sub-byte dtypes rather than returning a fraction.

Python `TensorView.data` contains **unpacked** logical int8/uint8 codes with the
logical shape; `payload_bytes()` returns canonical packed bytes, while `nbytes`
reports packed size. `write_array(..., dtype="int4"/"uint4")` takes logical codes
with the matching integer carrier, checks range, and packs them. No implicit
float casts/truncation. Native Torch packed capture is explicitly unsupported.
C++ `IRTensor.data` holds packed bytes; `AsFloat32()` decodes signed/unsigned
codes. `DataTypeSize` returns zero for packed types; use `DataTypeBits` and
`PayloadSize`. Typed byte-addressable payload views remain rejected.

Affine descriptors use `affine-int4/v1` or `affine-uint4/v1` and the matching
storage dtype. Scale rules are unchanged. Zero points are ordinary int8 or
uint8 tensors, restricted respectively to [-8, 7] or [0, 15]. Feature tokens
follow `quantization/<scheme>/<granularity>`. `linear/affine-w4a16/v1` requires
float16 x/y, computes dequantization and matmul in float32, and rounds once to
float16 output (overflow rejects). It has no bias and is not integer accumulation.
`linear_int4_reference` implements this reference contract. The independent
numpy adapter also supports these semantics.

1. Introduce bit-size and payload-size helpers. Never return a fractional byte
   from `dtype_itemsize` or assume `numel * itemsize` still applies everywhere.
2. Define the packed payload version and backward-reading behavior; audit codec,
   graph validation, TensorView, C++ reader/writer, extraction and reporting.
3. Canonical packing: first element in low nibble, next in high nibble, signed
   two's-complement INT4, and zero unused high nibble for odd numel. Reject
   noncanonical padding. Hardware-specific blocked layouts need explicit
   import/canonicalization rather than implicit interpretation.
4. Connect Q2 scale/group semantics and demonstrate W4A16 Linear end-to-end.

Acceptance: exhaustive nibble values/byte pairs, odd lengths, zero/scalar
tensors, sign extension, invalid padding/truncation, Python/C++ goldens, and
the same capability/extraction/hash/CLI gates as Q2. The packing baseline is
documented in [ONNX INT4](https://onnx.ai/onnx/technical/int4.html).

## Q4 tasks: NF4

Implemented descriptor: `scheme="nf4/v1"`, `storage_dtype="uint4"`,
`granularity="per_block"`, `axis=null`, `zero_point_value_id=null`, positive
`group_size`, `compute_dtype="float32"`, `rounding="none"`, `saturation="none"`.
In addition to the existing common fields it requires `codebook_value_id` and
`codebook="bitsandbytes-nf4/v1"`. The latter identifies the exact 16-entry
float32 NF4 table, whose payload must agree with that identity. Codebook values
are not inferred from raw uint4 storage. Affine descriptors retain their exact
existing field set and reject codebook fields.

Block numbering follows the flattened row-major **logical tensor**, not each
row independently. Scales are a float32 vector of length
`ceil(numel/group_size)`, finite and **nonnegative** (zero represents an all-zero
block). Codebook parameters have shape `[16]`; both parameters are ordinary,
unquantized immutable tensors. Nested parameters/cycles are rejected. The
reference computes `float32(codebook[index] * scale[flat_index // group_size])`.
`dequantize_nf4` is explicit; `dequantize` dispatches by descriptor. Affine
quantization APIs reject NF4 rather than interpreting it as integer codes.
No NF4 encoder/nearest-codebook quantizer is provided by this milestone.

`linear/nf4-w4a16/v1` takes float16 x and explicit NF4 weight `[out,in]`, does
float32 dequantization and matmul, then one float16 output conversion; no bias.
`linear_nf4_reference` implements it. The exact adapter feature is
`quantization/nf4/v1/per_block`. Scale and codebook bytes are included in
extraction, preflight dtype/rank requirements, standalone fingerprints,
trace/session identity, and testcase/evidence tree hashes. Dedup includes the
descriptor, conservatively retaining parameter IDs to avoid semantic merging.

### Pinned bitsandbytes importer profile

`import_bitsandbytes_nf4(packed, state, profile=..., scale_value_id=...,
codebook_value_id=...)` requires the exact profile
`bitsandbytes-0.48.2/nf4-uint8-flat/v1`. It consumes a **host projection** of
QuantState, not arbitrary pickles, framework objects, or packed state_dict JSON.
All nine state keys must be present, with no unknown keys:

| Key | Accepted representation |
| --- | --- |
| `shape` | Explicit tuple/list of nonnegative integer logical dimensions |
| `dtype` | Exact string `float16` |
| `quant_type` | Exact string `nf4` |
| `blocksize` | Exact integer 64 |
| `absmax` | float32 numpy vector with one scale per block |
| `code` | Exact 16-entry float32 numpy NF4 table |
| `nested` | Boolean false |
| `offset`, `state2` | Explicit null/None |

Source payload must be uint8 of shape `[ceil(numel/2)]` or `[ceil(numel/2),1]`.
bitsandbytes source order is **high nibble first**. The importer swaps it into
InferRef's low-first canonical order and zeros unused final padding. An unused
source tail nibble is discarded, not interpreted as a logical value. Inputs
are copied, not mutated. It returns `ImportedNF4(codes, spec, parameters)`;
the example constructs a version-gated trace from those artifacts.

The profile is grounded in the version-pinned upstream
[QuantState and NF4 table](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.2/bitsandbytes/functional.py)
and [native unpacking](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.48.2/csrc/kernels.cu).
Compressed/nested statistics, offsets, FP4, bfloat16 source state, other block
sizes, custom codebooks and other source packing profiles remain explicitly
unsupported. This is a bounded compatibility profile, not a claim to import
every bitsandbytes release or all quantized vLLM/TensorRT-LLM weights.

### Real QuantState bridge (0.50.1)

The separate `bitsandbytes-0.50.1/nf4-quant-state/v1` profile is exposed through
`import_bitsandbytes_quant_state(packed_torch_tensor, quant_state, profile=...,
scale_value_id=..., codebook_value_id=...)`. It requires runtime bitsandbytes
0.50.1 and real `functional.QuantState` objects. This pins the consumer/runtime
contract; it does not guess which library version originally wrote a checkpoint.
Optional dependencies are imported only by this object bridge. The companion
`import_bitsandbytes_nf4_projection` remains numpy-only.

Main NF4 blocksize is 64, with source output dtype float16, bfloat16 or float32.
Host projections have the same nine explicit keys as the flat profile. For
`nested=True`, main absmax is a uint8 vector of block-scale indices; `offset`
is a scalar float32 array. `state2` is another exact nine-key projection with
float32 dtype, blocksize 256, a finite 256-entry float32 codebook, nonnegative
float32 absmax vector of length `ceil(number_of_main_scales/256)`, null shape,
quant_type, offset and state2, and `nested=False`. Unknown/incomplete state,
nonfinite values, wrong shapes/types, negative effective scales and deeper
nesting are rejected. The 256-entry scale codebook is explicitly supplied data,
not inferred from NF4's separate 16-entry weight codebook.

Scale reconstruction is separate float32 multiply then add:
`effective_scale[i] = state2.code[absmax[i]] * state2.absmax[i // 256] + offset`.
This implementation does not call the bitsandbytes dequantizer, enabling an
independent oracle check. Source objects are detached/copied without mutation.
The normalized `ImportedQuantState.quantized` stores ordinary float32 effective
scales, canonical packed codes and the NF4 codebook; existing extraction,
preflight, hashing and comparison operate on those values. Original compressed
state is not retained or replayed. Equivalent expanded scales intentionally
have equivalent canonical semantics regardless of original compressed encoding.

`ImportedQuantState.dequantize_source()` explicitly rounds the float32 NF4
values to the source dtype (nearest-even bfloat16/float16), returning a typed
TensorView. Overflow is rejected. Source dtype is not inferred from a model's
compute dtype and is not silently substituted into `linear/nf4-w4a16/v1`.
Trace/testcase outputs must record that cast as actual typed output values; a
required cast cannot live only in optional provenance fields. Raw quantized
trace comparison still means float32 canonical dequantization by default.

See the [runnable real-checkpoint audit](../../examples/quantization/README.md#real-bitsandbytes-quantstate-and-double-quantization).
Its implementation was checked against installed 0.50.1 `functional.QuantState`,
`dequantize_blockwise`, `dequantize_4bit` and CPU backend implementations, then
executed against that runtime; no remote model code is loaded.

1. Define a separate NF4 scheme with explicit codebook identity/values and
   block scales. NF4 is not signed INT4 and not FP4.
2. Add an explicit importer for a documented bitsandbytes QuantState profile,
   including shape, block size, quantization type, scale state and, when
   supported, offset/nested scale state. Reject unknown/incomplete profiles.
3. Canonicalize source packing into Q3 ordering; do not assume all uint8
   weight buffers have ONNX nibble ordering or the same codebook.
4. Implement reference dequantization, extraction dependencies, capability
   declaration and NF4 Linear fixtures. Add nested statistics only with
   independent reference vectors and explicit unsupported handling otherwise.

Acceptance: all codebook entries, block boundaries/tails, packed ordering,
scale changes, negative importer cases, no-Torch core reference vectors, and
engine tests separate from full-precision quantization-quality measurements.
See [bitsandbytes functional/QuantState reference](https://huggingface.co/docs/bitsandbytes/reference/functional).

## Local Q1 validation

On Windows, 2026-09-17:

- Initial complete regression: 867 passed, 9 skipped, 3 deselected. After adding
  malformed-payload/CLI cases and extraction assertions, the focused final
  FP8/CLI/frontend run passed all 42 tests (not an additional 42 unique tests).
- Final core suite in Python 3.12 without Torch installed: 610 passed,
  1 skipped, 3 deselected.
- Torch 2.1.2 compatibility run: all 12 FP8 capture/extraction tests passed.
- Rebuilt MSVC C++ tests: 4/4 passed. All 512 encodings and Python/C++ byte
  roundtrips passed, including running the Python driver without Torch.
- Ruff, compileall and CircleCI YAML parsing passed.
- Wheel/sdist builds, isolated install/CLI checks and Twine checks passed.
  These are local unreleased artifacts under the unchanged source version,
  not replacements for published v0.10.0 assets.

Remote CI is not implied by local success; the CircleCI Linux/Windows C++ jobs
include the FP8 exhaustive test and cross-language roundtrip driver.

## Local Q2a / dependency-projection validation

On Windows, 2026-09-17:

- Q2a complete regression: 944 passed, 9 skipped, 3 deselected; JUnit report
  saved locally at `.scratch/q2a-regression.xml`. The optional Torch oracle
  emitted one upstream quantized-tensor API deprecation warning. Runtime
  reference arithmetic does not depend on that API or on Torch.
- After adding the 16 dependency-projection cases, final no-Torch Python 3.12
  core suite: 693 passed, 1 skipped, 3 deselected, with report at
  `.scratch/q2-final-core.xml`. The focused numerical/metadata/projection
  suite has 83 passing tests (these overlap the full/core counts).
- All six optional Torch 2.1.2 oracle tests passed. The three slow core gates
  passed. Existing C++ regression tests remained 4/4; C++ quantization
  reference arithmetic was not added by Q2a.
- Ruff, compileall, Agent workflow checks, and local documentation links passed.
- Local wheel/sdist builds, Twine checks and isolated installs passed. The
  installed smoke test now imports and exercises the new INT8 API and dependency
  projection outside the checkout, without Torch. These are unreleased local
  artifacts, not new published v0.10.0 assets.

This historical evidence covers Q2a. Q2b adds artifact and execution tests;
its acceptance is recorded separately below.

## Local Q2b acceptance

On Windows, 2026-09-17: no-Torch Python 3.12 core suite **723 passed, 1 skipped,
3 deselected** (`.scratch/q2b-core-final.xml`); frontend **165 passed, 8 skipped**
with the known optional Torch quantizer deprecation warning; Agent/docs **101
passed, 1 skipped**. Integration tests cover version rejection, dependency and
payload corruption before launch, relocation, internal symlink rejection for
hashing, independent dequantized/raw-code comparisons, quantized reference
outputs, explicit capability preflight, Suite/Agent execution, and plan/run
agreement. Long-lived runners also reject changed source bytes before reuse.

The documented commands were executed in `.scratch/q2b-workflow`: Agent pass;
Alignment plan recommended `region:0`, run executed `region:0: pass`; injected
error produced first divergence at `inferref.linear_affine_int8.default`,
reference 9.5 vs actual 10.5. Ruff and Agent workflow checks passed.
These are local unreleased results, not remote CI or published release claims.

## Local Q3 acceptance

On Windows, 2026-09-17: no-Torch core **757 passed, 1 skipped, 3 deselected**
(`.scratch/q3-core.xml`). C++ **5/5** tests passed, including all 512 signed and
unsigned byte-pair decodings, odd padding and version gates. Python/C++/Python
wire bytes matched for both packed dtypes across exhaustive pairs, odd tails,
scalar and empty shapes. The existing 512 FP8 goldens still matched. CircleCI
Linux/Windows jobs now run the packed goldens too; remote CI was not run here.
Signed/unsigned W4A16 pass and injected-error cases, extraction, preflight,
Suite, Alignment and cache identity checks passed. The documented signed INT4
CLI extraction and Agent execution were also run and passed.

## Local Q4 and final Q2–Q4 acceptance

Final verification on Windows, 2026-09-17–18:

- No-Torch Python 3.12 core: **783 passed, 1 skipped, 3 deselected**
  (`.scratch/quantization-final-core.xml`). Frontend/Agent/docs: **266 passed,
  9 skipped**, with one upstream Torch quantizer deprecation warning
  (`.scratch/quantization-final-extended.xml`). The six optional Torch 2.1.2
  reference checks also passed.
- All three slow installation gates passed in the pip-equipped development
  environment (`.scratch/quantization-final-slow.xml`). The initial no-Torch
  environment attempt could not run one install gate because pip was absent;
  this was an environment prerequisite, not a skipped product failure.
- C++ tests remain **5/5**, and exhaustive FP8/packed cross-language wire
  roundtrips passed again. Ruff, compileall and Agent workflow checks passed.
- NF4 tests cover all codebook indices, flattened block boundaries across rows,
  tail blocks, zero scales, packing conversion, invalid/incomplete/nested source
  states, extraction, capability rejection, Agent/Suite/Alignment, independent
  dequantized comparisons and parameter-sensitive cache identity. Dedup keeps
  affine UINT4 and NF4 semantics distinct. Planning validates parameters without
  loading/dequantizing entire weight payloads.
- Documented INT4 and NF4 CLI extraction and Agent runs returned `pass` in
  `.scratch/quant-final-cli`; independent adapter injected-error tests failed
  as expected. Validation followed the InferRef Agent capability/preflight and
  execution workflow rather than relying solely on the numerical helper API.
- Wheel and sdist were built and independently installed outside the checkout,
  with numpy-only INT8/NF4/packed-codec and CLI smoke checks. These remain local
  unreleased artifacts under source version 0.10.0, not published release assets.

At the initial Q4 acceptance, NF4 evidence used pinned upstream
source definitions, independent fixed reference vectors and a separate numpy
engine; an actual bitsandbytes runtime/GPU kernel was **not** executed locally.
Nested/double quantization, automatic native quantized Torch capture, general
vendor packing import, FNUZ and C++ quantized-testcase execution remain outside
that scope. The subsequent real QuantState acceptance below supersedes the
runtime/one-level import limitation, not the GPU or C++ execution limitations.
Remote CI and publication have not been performed.

## Real QuantState acceptance

On Windows, 2026-09-18, CPU Torch 2.13.0 and bitsandbytes 0.50.1:

- Read the supplied local `qwen3-0.6B-bnb-nf4-offline` safetensors checkpoint via
  actual `QuantState.from_dict`, without loading the full model or remote code.
  All 196 NF4 weights use bfloat16 source dtype, blocksize 64, and one compressed
  scale level of blocksize 256. All **440,401,920** decoded weight elements and
  all expanded scales matched actual bitsandbytes exactly; max absolute error 0.
- Source `model.safetensors` SHA-256 before/after:
  `075ec08938f270c54e93f07bb8df57a19297b6c4c4b831595059e81a7ba6ce05`.
- The first weight, `model.layers.0.mlp.down_proj.weight` (1024 x 3072), was
  exported as a quantized trace 1.0 / testcase 0.4 with bfloat16 dequantized
  output. Agent context validation passed; Agent compare at atol=rtol=0 passed.
  Detailed local report: `.scratch/qwen3-real-nf4-20260918/report.json`.
- Synthetic real-runtime checks quantize, serialize, restore and import each
  float16/bfloat16/float32 x nested/non-nested combination, verify output and
  source immutability, and include a safetensors-to-portable-case smoke test.
  Core tests independently check compressed-scale tails, rounding and malformed
  state without Torch. Model weights are not committed as test fixtures.
- Final no-Torch core regression: **813 passed, 1 skipped, 3 deselected**
  (`.scratch/bnb-final-core.xml`). The eight real-runtime/portable-case tests
  plus six existing Torch affine-reference checks passed, including rejection
  paths and an injected output mismatch
  (`.scratch/bnb-quant-state-frontend.xml`). Ruff and compileall passed.

This is stored-weight and scale-decoding acceptance on CPU, not full-model token
generation, fused low-bit GEMM, GPU execution, arbitrary bnb versions, deeper
nesting or C++ quantized testcase support.


## Whole-model generation and GPU qualification

On Windows, 2026-09-21, Intel Arc A770 Graphics (Level Zero driver 1.15.38308)
and PyTorch 2.13.0 + bitsandbytes 0.50.1 / Aila native SYCL engine:

- Compiled reference causal generation prefill and multi-step autoregressive
  decode into an InferRef Scenario v0.1 (`scenario.json` + `cases/prefill`,
  `cases/decode-0`, `cases/decode-1`). Each step testcase provides self-contained
  input prefix tokens and reference float32 logits.
- Target execution on Intel Arc A770 GPU via Aila's native SYCL engine adapter
  (`aila-qwen3-nf4`, `AilaQwen3Nf4InferRefAdapter.exe`).
  Loaded all 28 transformer layers, 196 NF4 weights (513.78 MB) and allocated
  KV cache on device.
- Step-by-step token parity: Reference and Aila GPU both predicted identical
  token sequences: [16, 13, 576] ("1. The") with 100% token match.
- Numerical alignment: 28-layer bfloat16 accumulation on Arc A770 GPU achieved
  0.99988 cosine similarity and passed tolerance comparison across all steps.
- Device evidence: Every step emitted valid `inferref-sycl-device` v0.1 evidence
  confirming execution on 'Intel(R) Arc(TM) A770 Graphics' (backend 'ext_oneapi_level_zero').
- Relocation, Agent, Suite (3/3 passed) and Alignment Orchestrator verification
  passed; injected bug (`--inject-bug`) was detected immediately.
  Detailed report: '.scratch/whole-model-qwen3-gpu-qual-5/report.json'.
The earlier `whole-model-qwen3-gpu-qual-final` run was an intermediate attempt
whose decode comparisons failed (Suite 1/3); it must not be cited as
qualification evidence. The final `qual-5` run passed Suite 3/3 and Alignment.
