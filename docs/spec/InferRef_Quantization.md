# Native FP8 and quantization development plan

Status: incremental implementation on the unreleased development branch.
This document distinguishes shipped code from planned work; it does not claim
that an FP8-capable reader can execute an FP8 kernel or a quantized model.

## Ordered milestones

Implementation and acceptance are owned by the primary agent in this task.
Each milestone must pass its own checks before the next changes the wire or
semantic contract. No release/version bump is implied by this plan.

| Milestone | Scope | State |
| --- | --- | --- |
| Q1 | Native E4M3FN/E5M2 storage, decoding, comparison, capture, interoperability | Implemented; local validation recorded below |
| Q2 | Versioned quantized-tensor semantics, exercised with existing INT8 storage | Planned |
| Q3 | Packed INT4/UINT4 and affine quantized Linear | Depends on Q2 |
| Q4 | Explicit NF4 codebook/block-scale semantics and importer | Depends on Q2/Q3 |

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

Q2 is not implemented yet. The following are required development tasks, not
fields that current readers understand.

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

## Q3 tasks: packed INT4/UINT4

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
