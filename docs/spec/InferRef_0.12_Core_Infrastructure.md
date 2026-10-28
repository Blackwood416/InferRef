# InferRef 0.12.0: Native Quantized Testcase ABI, Hardware Quantization Contracts, and Hierarchical Alignment Infrastructure

> **Status:** Scope locked 2026-10-22. M1–M3 are the 0.12.0 release gate;
> M4 is P1 and may ship in 0.12.0 or a later 0.12.x. The W8A8 formula was
> signed off on 2026-10-22; I3b starts from the merged Wave 1 baseline. See the
> [execution plan](InferRef_0.12_Execution_Plan.md) and
> [W8A8 contract freeze](InferRef_0.12_W8A8_Contract.md).
> Linux verification is deferred until WSL is enabled by the user; Windows and
> the local A770 are the current verification platforms.

## 1. Scope & Release Decision

InferRef 0.11.0 established the Python-side quantized trace (`1.0`) and testcase (`0.4`) pipeline, native `float8_e4m3fn` / `float8_e5m2` and packed `int4` / `uint4` tensor codecs, the `bitsandbytes 0.50.1` `QuantState` bridge, and whole-model causal generation qualification on Intel Arc A770 (Level Zero / SYCL).

However, four core infrastructure gaps remain between the Python reference pipeline and native C++/GPU engine development:

1. **C++ Testcase 0.4 & Quantization Parameter ABI Gap**: `cpp/include/inferref/irtensor.hpp` already decodes Format 2 (`int4`/`uint4`) and FP8, but `cpp/include/inferref/testcase.hpp` explicitly rejects `format_version == "0.4"` and cannot load `quantization_parameters` or expose `QuantizationSpec` descriptors to native C++ adapters and runtime bridges.
2. **Activation DType & Hardware-Accumulation Contract Gap**: Built-in `linear/nf4-w4a16/v1` and `linear/affine-w4a16/v1` in `inferref/contracts/builtin.py` hard-require `float16` activations/outputs and omit `bfloat16` (which Qwen3-0.6B NF4 uses natively) as well as optional `bias`. Furthermore, hardware low-bit kernels (such as Intel XMX DPAS `int8` or CUDA TensorCore INT8) require an explicit integer-accumulation reference contract (`W8A8` -> `int32` accumulator -> dequantized output) rather than float32 weight-dequantization-only semantics.
3. **Core Transformer Region Contract Coverage**: `inferref region detect` automatically identifies `SwiGLU` and `Attention`, but built-in executable contracts only cover `Linear`, `RMSNorm`, `RoPE`, and `KVCacheUpdate`.
4. **Whole-Model Scenario to Sub-Region Narrowing Gap**: `qualify_whole_model_generation.py` compiles multi-step `prefill` / `decode` into Scenario v0.1, but binds alignment to a single-node synthetic trace (`tokens -> logits`). When a whole-model step diverges, `inferref align narrow` cannot drill down into per-layer / per-region boundaries without re-tracing and manually re-binding.

**0.12.0 Goal:**

> Complete the core Python/C++ infrastructure closure for quantized testcases, hardware-accurate low-bit contracts, and hierarchical multi-layer alignment before expanding to external third-party formats or UI tooling.

## 2. Confirmed 0.11.0 Baseline

The 0.12.0 specification builds on these verified facts in the 0.11.0 tree:

- `inferref/quantization/spec.py` defines `QuantizationSpec` (`format="inferref-quantization"`, `format_version="1.0"`) with schemes `affine-int8/v1`, `affine-int4/v1`, `affine-uint4/v1`, and `nf4/v1`.
- `inferref/testcase/extract.py` emits `format_version="0.4"` whenever any projected input, output, or value carries a `quantization` descriptor, placing referenced scale, zero-point, and codebook payloads in `quantization_parameters/` and listing their metadata in the top-level `quantization_parameters` array of `testcase.json`.
- `cpp/include/inferref/irtensor.hpp` already implements `DataType::Int4` (code 18), `DataType::UInt4` (code 19), `DataType::Float8E4M3FN` (code 16), `DataType::Float8E5M2` (code 17), `DataTypeBits()`, `PayloadSize()`, and `AsFloat32()`.
- `cpp/include/inferref/testcase.hpp` (`Testcase::Load`) currently accepts only `0.1`, `0.2`, and `0.3`, and parses only `inputs` and `outputs`.
- `inferref/adapter/scaffold.py` generates C++ standard and `--runtime-bridge` projects, deriving `adapter.json` capabilities via `derive_requirements(manifest)`.
- `inferref/alignment/` implements persistent sessions (`L1`–`L9`), boundary discovery (`region`, `module`, `op`), content-addressed testcase caching, and causal narrowing over `TracePackage.graph`.

## 3. Non-Goals for 0.12.0

- No automatic PyTorch `__torch_dispatch__` subclass interception for third-party quantized tensor wrappers (`bitsandbytes` `Params4bit` / TorchAO) during live `inferref trace`; explicit host/bridge projection and layer-wise trace compilation remain the canonical path in 0.12.0.
- No GGUF / AWQ / GPTQ third-party checkpoint importers (deferred to 0.13+).
- No web viewer or interactive HTML graph GUI (`SPEC §64 Phase 6`).
- No breaking changes to `.irtensor` v1/v2 binary wire layouts or Trace IR `1.0`.

## 4. Terminology

| Term | Meaning |
| --- | --- |
| Quantized Testcase (`0.4`) | Standalone testcase carrying `quantization` descriptors on roles/values and a top-level `quantization_parameters` collection |
| Quantization Parameter | Immutable auxiliary tensor (`scale`, `zero_point`, `codebook`) referenced by `value_id` from a `QuantizationSpec` |
| Dequantized Helper View | C++ convenience API on `inferref::Testcase` that reconstructs float32 logical values from a quantized input tensor and its bound parameters |
| W8A8 Integer Accumulation | Quantized Linear semantics where `int8` input and `int8` weight accumulate in `int32` before applying input/weight scales |
| Hierarchical Trace Package | A whole-model or multi-layer `TracePackage` containing nested `TransformerBlock`, `Attention`, `MLP`, and leaf operator boundaries alongside end-to-end step I/O |

## 5. Milestone M1 (P0): C++ Testcase 0.4 & Quantization ABI (`cpp/include/inferref/`)

### 5.1 Motivation

A native C++/SYCL or CUDA engine validating a quantized region (`linear/nf4-w4a16/v1`, `linear/affine-int8/v1`, etc.) needs direct, dependency-free access to:
1. the packed or integer weight `IRTensor` (`inputs/weight.irtensor`),
2. the attached `QuantizationSpec` metadata (`scheme`, `granularity`, `axis`, `group_size`, `scale_value_id`, `zero_point_value_id`, `codebook_value_id`), and
3. the corresponding parameter `IRTensor` payloads (`quantization_parameters/v10.irtensor`, etc.).

### 5.2 C++ Data Structures (`cpp/include/inferref/testcase.hpp`)

Extend `inferref::Testcase` and add `inferref::QuantizationSpec` (in `testcase.hpp` or a dedicated header `cpp/include/inferref/quantization.hpp` included by `testcase.hpp`):

```cpp
namespace inferref
{

struct QuantizationSpec
{
    std::string format;               // "inferref-quantization"
    std::string format_version;       // "1.0"
    std::string scheme;               // "affine-int8/v1", "affine-int4/v1", "affine-uint4/v1", "nf4/v1"
    std::string storage_dtype;        // "int8", "int4", "uint4"
    std::vector<std::int64_t> logical_shape;
    std::int64_t scale_value_id = -1;
    bool has_zero_point = false;
    std::int64_t zero_point_value_id = -1;
    bool has_codebook = false;
    std::int64_t codebook_value_id = -1;
    std::string codebook;             // e.g. "bitsandbytes-nf4/v1" or empty
    std::string granularity;          // "per_tensor", "per_axis", "per_group", "per_block"
    bool has_axis = false;
    std::int64_t axis = 0;
    bool has_group_size = false;
    std::int64_t group_size = 0;
    std::string compute_dtype;        // "float32"
    std::string rounding;             // "nearest_even", "none"
    std::string saturation;           // "clamp", "none"
};

class Testcase
{
public:
    static Testcase Load(const std::string &testcase_dir);

    std::string FormatVersion() const;
    std::string RegionName() const;
    std::vector<std::string> InputNames() const;
    std::vector<std::string> OutputNames() const;

    IRTensor Input(const std::string &name) const;
    std::map<std::string, IRTensor> Inputs() const;

    // New in 0.12.0: quantization metadata & parameter access
    bool HasInputQuantization(const std::string &name) const;
    const QuantizationSpec &InputQuantization(const std::string &name) const;

    std::vector<std::int64_t> QuantizationParameterIds() const;
    IRTensor QuantizationParameter(std::int64_t value_id) const;
    std::map<std::int64_t, IRTensor> QuantizationParameters() const;

    // Convenience helper: dequantize a quantized input role to canonical float32
    std::vector<float> DequantizeInputToFloat32(const std::string &name) const;

    void SetOutputDir(const std::string &output_dir);
    void WriteOutput(const std::string &name, const IRTensor &tensor);
    void WriteOutputs(
        const std::map<std::string, IRTensor> &outputs,
        const std::string &caller_label = "engine");
    void Finish();
};

} // namespace inferref
```

### 5.3 Validation & Fail-Closed Rules in C++

Mirror the Python `validate_testcase` guarantees in `Testcase::Load`:

1. **Version Gate**:
   - Accept `format_version` in `{"0.1", "0.2", "0.3", "0.4"}`.
   - If `quantization_parameters` or any role `quantization` field is present when `format_version != "0.4"`, throw `TestcaseError("quantization metadata requires testcase format_version '0.4'")`.
2. **Safe Path & Payload Integrity**:
   - Every entry in `quantization_parameters` must be an object with non-negative integer `id` (or `value_id` matching `testcase.json` schema), valid `payload` passing `IsSafeRelativePath()`, and matching `dtype` / `shape` / `nbytes` against the loaded `.irtensor` header.
   - Duplicate parameter IDs or missing referenced `scale_value_id` / `zero_point_value_id` / `codebook_value_id` must fail closed during `Testcase::Load` before any engine kernel runs.
3. **Reference Dequantization Parity (`DequantizeInputToFloat32`)**:
   - Implement C++ reference dequantization for `affine-int8/v1`, `affine-int4/v1`, `affine-uint4/v1` (`per_tensor`, `per_axis`, `per_group`), and `nf4/v1` (`per_block`), matching `inferref/quantization/reference.py` and `inferref/quantization/nf4.py` bit-exactly in `float32`.

### 5.4 Runtime Bridge & Adapter Scaffolder Integration

1. **Runtime Bridge (`cpp/include/inferref/bridge.hpp`)**:
   - Keep existing `DebugInvoke(region_name, inputs)` working unchanged for unquantized testcases.
   - Add an overloaded `DebugInvokeEx` (or `RunBridge(argc, argv, const TestcaseInvoke &invoke)`) that passes `const Testcase &tc` alongside `region_name` and `inputs`, allowing a single C++ runtime bridge to query `tc.HasInputQuantization("weight")`, `tc.InputQuantization("weight")`, and `tc.QuantizationParameter(id)`.
   - If an engine registers only the legacy 2-argument `DebugInvoke` and receives a `0.4` quantized testcase, `RunBridge` exits with a dedicated exit code `kBridgeUnsupportedQuantization = 4` and diagnostic message rather than silently ignoring scales.
2. **Adapter Scaffolder (`inferref/adapter/scaffold.py`)**:
   - When scaffolding from a `0.4` testcase, generated `main.cpp` includes comments/snippets showing `testcase.InputQuantization(...)` and `testcase.QuantizationParameters()`, and `adapter.json` automatically populates `capabilities.features` with the required `quantization/<scheme>/<granularity>` tokens from `derive_requirements()`.

## 6. Milestone M2 (P0): Hardware-Accurate Quantization & Core Transformer Contracts

### 6.1 Extend W4A16 & NF4 Contracts to Support `bfloat16` and Optional `bias`

In 0.11.0, `linear/affine-w4a16/v1` and `linear/nf4-w4a16/v1` reject `bfloat16` activations (`x`) and outputs (`y`), even though modern checkpoints (such as Qwen3-0.6B NF4) run in `bfloat16`.

Add two explicit contract families and optional `bias` support across `inferref/contracts/builtin.py` and `inferref/quantization/`:

| Contract ID | Inputs | Outputs | Semantics |
| --- | --- | --- | --- |
| `linear/affine-int8/v1` | `x`, `weight` | `y` | Existing: `float32` `x`, affine `int8` `weight` -> `float32` `y` |
| `linear/affine-int8-bias/v1` | `x`, `weight`, `bias` | `y` | `float32` `x`, affine `int8` `weight`, `float32` `bias [out]` -> `float32` `y` |
| `linear/affine-w4a16/v1` | `x`, `weight` | `y` | Existing: `float16` `x` & `y`, `float32` dequant/accum |
| `linear/affine-w4a16-bf16/v1` | `x`, `weight` | `y` | `bfloat16` `x` & `y`, affine `int4`/`uint4` `weight`, `float32` dequant/accum, nearest-even `bfloat16` output |
| `linear/nf4-w4a16/v1` | `x`, `weight` | `y` | Existing: `float16` `x` & `y`, `nf4/v1` `weight`, `float32` dequant/accum |
| `linear/nf4-w4a16-bf16/v1` | `x`, `weight` | `y` | `bfloat16` `x` & `y`, `nf4/v1` `weight`, `float32` dequant/accum, nearest-even `bfloat16` output |
| `linear/w8a8-int32-accum/v1` | `x`, `weight` | `y` | Both `x` (`per_tensor`, `per_axis`, or `per_group`) and `weight` are `affine-int8/v1` with matching group structure; integer dot product accumulates in `int32` per group before multiplying by `float32(scale_x * scale_w)` into `float32` `y` |

### 6.2 `linear/w8a8-int32-accum/v1` Specification

Hardware XMX DPAS (`sycl::ext::oneapi::experimental::matrix::joint_matrix_mad`) and CUDA `dp4a`/TensorCore kernels compute integer dot products over `int8` tiles and rescale per group or per channel in the epilogue.

1. **Input validation (`_w8a8_linear_inputs`)**:
   - `x` must have `dtype == "int8"` and a valid `QuantizationSpec` with `scheme == "affine-int8/v1"` and `zero_point_value_id is None` (symmetric activations) or explicit `int8` zero-point.
   - `weight` must have `dtype == "int8"`, rank 2 `[out_features, in_features]`, and `scheme == "affine-int8/v1"`.
   - Reduction dimension `x.shape[-1] == weight.shape[1]` must match.
   - If either `x` or `weight` uses `per_group` along the reduction axis (`axis = -1`), both must share compatible group sizes (either identical `group_size` or one is `per_tensor`/`per_axis`).
2. **Reference evaluation (`linear_w8a8_int32_reference` in `inferref/quantization/reference.py`)**:
   - Cast `x` codes and `weight` codes (minus `zero_point` if present) to `int32`.
   - For each reduction group $g$, compute exact `int32` sum:
     $$A_{m, n, g} = \sum_{k \in g} (x_{m, k} - zp_{x}) \cdot (w_{n, k} - zp_{w})$$
     checking that no `int32` overflow occurs.
   - Convert $A_{m, n, g}$ to `float32`, multiply by `scale_x[..., g] * scale_w[n, g]`, and accumulate across groups $g$ in canonical order into `float32` output `y`.
3. **Adapter capability token**:
   - Require `quantization/w8a8-int32-accum/v1` in addition to the constituent `quantization/affine-int8/v1/<granularity>` tokens.

### 6.3 Core Transformer Region Contracts (`SwiGLU` & `Attention`)

Add two built-in region contracts to `inferref/contracts/builtin.py` so every region class emitted by `inferref region detect` has a first-class contract:

1. **`swiglu/mlp/v1`**:
   - **Inputs**: `("x", "gate_weight", "up_weight", "down_weight")`
   - **Outputs**: `("y",)`
   - **Validation**: `x` has shape `[..., hidden_size]`; `gate_weight` and `up_weight` have shape `[intermediate_size, hidden_size]`; `down_weight` has shape `[hidden_size, intermediate_size]`; `y` matches `x` shape and dtype.
2. **`attention/sdpa-gqa/v1`**:
   - **Inputs**: `("query", "key", "value")`
   - **Outputs**: `("attn_output",)`
   - **Validation**: Rank 4 `[batch, heads, seq_q, head_dim]` for `query` and `attn_output`; `key` and `value` have shape `[batch, kv_heads, seq_kv, head_dim]` where `heads % kv_heads == 0` (supporting MHA, GQA, and MQA) and matching dtype.

## 7. Milestone M3 (P0): Hierarchical Whole-Model Trace & Alignment Narrowing

### 7.1 Problem in 0.11.0

In `examples/quantization/qualify_whole_model_generation.py`, `make_trace_package()` builds a 1-operator trace (`qwen3.causal_generation_prefill`). While sufficient to prove `AlignmentRunner` executes the whole-model testcase, `inferref align plan` and `inferref align narrow` only see 1 boundary (`region:0`). If an engine's 28-layer GPU forward pass fails on token 1, the developer cannot run `inferref align narrow` to binary-search which layer (`layers.0` ... `layers.27`) or sub-region (`self_attn`, `mlp`, `input_layernorm`) caused the divergence.

### 7.2 Multi-Layer Trace & Hierarchical Boundary Compilation

Add `inferref/frontend/pytorch/hierarchical.py` (or extend `inferref/testcase/` & `qualify_whole_model_generation.py`) with a **Hierarchical Generation Trace Compiler**:

1. **Module-Hooked Step Capture (`compile_step_trace`)**:
   - During reference generation of a chosen step (`prefill` or `decode-k`), attach lightweight forward hooks to:
     - Top-level model (`tokens -> logits`),
     - Each decoder layer (`layers.<i>`: `hidden_states_in -> hidden_states_out`),
     - Each sub-block (`layers.<i>.input_layernorm`, `layers.<i>.self_attn`, `layers.<i>.post_attention_layernorm`, `layers.<i>.mlp`),
     - Leaf projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) with their imported `QuantizationSpec` descriptors attached when the model uses `bitsandbytes` or affine quantization.
2. **Nested `RegionRecord` Hierarchy**:
   - Emit nested `RegionRecord` spans over the step's operator execution indices:
     - `WholeModel@step` (spans all operators `0..N-1`)
     - `TransformerBlock@layers.<i>` (spans operators of layer `i`)
     - `Attention@layers.<i>.self_attn` and `SwiGLU@layers.<i>.mlp`
     - `Linear@layers.<i>.mlp.down_proj` (leaf quantized operator/region)
3. **Planner Coarse-to-Fine Binary Search (`inferref/alignment/planner.py`)**:
   - Verify and harden `plan_alignment()` and `narrow_alignment()` when a parent boundary (`WholeModel@step`) has >10 ordered child `TransformerBlock@layers.<i>` regions:
     - When `WholeModel@step` is `failing` and no child `TransformerBlock` has been tested yet, the planner's top candidate must select the **midpoint layer** (`layers.<N//2>`) or **first layer** (`layers.0`) with deterministic binary-search priority (`reason: "binary search midpoint of failing serial chain"`), reducing 28-layer fault isolation from $O(L)$ to $O(\log_2 L)$ (at most 5 runs to isolate the exact failing layer, plus 2 runs to isolate the exact sub-region/projection).

### 7.3 Batch Region Execution in Runtime Bridge (`--batch-manifest`)

Launching a separate OS process and reloading a multi-hundred-megabyte checkpoint for each alignment step dominates wall-clock time.

Add optional batch execution support to Engine Adapter v0.2 (`capabilities.features: ["batch_execution"]`) and `cpp/include/inferref/bridge.hpp`:

```bash
inferref-bridge \
  --batch-manifest batch.json
```

Where `batch.json` lists an ordered sequence of `{id, testcase_dir, output_dir}` items. `RunBridge` initializes the engine once, invokes `DebugInvoke` (or `DebugInvokeEx`) for each item in process, writes each item's `manifest.json`, and emits `batch-results.json`. `AlignmentRunner.run(limit=K)` automatically uses `--batch-manifest` when `K > 1` and the adapter declares `"batch_execution"`.

## 8. Milestone M4 (P1): Multi-Bug Agent Repair Benchmark v0.3

### 8.1 Motivation

Currently `examples/agent_eval/` contains a single benchmark (`rope_sign`). To support the controlled ablation mentioned in `README.md` (`full-diagnostics` vs `PASS/FAIL-only` vs `no-InferRef`), 0.12.0 expands the benchmark suite to cover quantized and stateful bugs.

### 8.2 New Benchmark Cases under `examples/agent_eval/`

1. **`rope_sign`** (existing baseline): sign flip in `rotate_half`.
2. **`nf4_block_scale_offset`** (new in 0.12.0): off-by-one group/block indexing when dequantizing NF4 weights across row boundaries (`flat_index // group_size` vs `col_index // group_size`).
3. **`kv_cache_index_stride`** (new in 0.12.0): `kv-cache/indexed-update/v1` writes into the wrong sequence slice during decode step 1.

Add `--diagnostic-mode {full,status_only}` to `inferref agent evaluate` so the MCP evaluation server can run controlled ablations with identical holdouts and attestation guarantees.

Implementation status: completed on local main for 0.12.x. The `v0.12.0` tag and
release remain M1–M3; M4 did not ship in that release.

## 9. Compatibility & Wire Versioning

| Artifact / Protocol | 0.11.0 Version | 0.12.0 Version | Compatibility Rule |
| --- | --- | --- | --- |
| `.irtensor` | `1` (byte), `2` (packed) | `1`, `2` (unchanged) | 100% wire-compatible |
| Trace IR | `0.1` (plain), `1.0` (quantized) | `0.1`, `1.0` (unchanged) | 100% wire-compatible |
| Testcase | `0.1`–`0.3` (plain/comp), `0.4` (quantized) | `0.1`–`0.4` (unchanged) | C++ `testcase.hpp` upgraded from `0.1–0.3` to `0.1–0.4` |
| Engine Adapter | `0.2` | `0.2` (additive features) | New capability tokens (`quantization/w8a8-int32-accum/v1`, `batch_execution`) are additive |
| Alignment Session | `0.1` | `0.1` (unchanged) | Binary-search planner scoring is deterministic and backwards-compatible |

## 10. Acceptance Criteria

### 10.1 C++ Testcase 0.4 & Quantization Helper (M1)
- [ ] `inferref::Testcase::Load` reads `0.4` quantized testcases (`affine-int8/v1`, `affine-int4/v1`, `affine-uint4/v1`, `nf4/v1`) without PyTorch or Python.
- [ ] `Testcase::Load` rejects `quantization` fields or `quantization_parameters` when `format_version != "0.4"`.
- [ ] `Testcase::Load` rejects path traversal, missing parameter payloads, duplicate parameter IDs, and header/manifest shape mismatches.
- [ ] `Testcase::DequantizeInputToFloat32` matches Python `dequantize()` bit-for-bit across INT8, packed INT4/UINT4 (including odd tails), and NF4 (including cross-row blocks).
- [ ] `inferref adapter scaffold` on a `0.4` testcase produces a compilable C++ project that passes `ctest` in CI.

### 10.2 Hardware Quantization & Transformer Contracts (M2)
- [ ] `linear/affine-int8-bias/v1`, `linear/affine-w4a16-bf16/v1`, `linear/nf4-w4a16-bf16/v1`, and `linear/w8a8-int32-accum/v1` are registered in `inferref contract list` and covered by positive and negative validator tests.
- [ ] `linear_w8a8_int32_reference` matches exact `int32` per-group accumulation and detects injected accumulator/scale bugs.
- [ ] `swiglu/mlp/v1` and `attention/sdpa-gqa/v1` validate MHA, GQA, and MQA shapes and dtypes.

### 10.3 Hierarchical Whole-Model Alignment & Batch Bridge (M3)
- [ ] `compile_step_trace` produces a valid multi-layer `TracePackage` (`WholeModel` -> `TransformerBlock` -> `Attention`/`SwiGLU` -> `Linear`) that passes `inferref validate`.
- [ ] Injected error in a single layer (e.g. `layers.19.mlp.down_proj`) is isolated by `inferref align run` + `inferref align narrow` in $\le \lceil \log_2(28) \rceil + 3 = 8$ boundary evaluations.
- [ ] C++ Runtime Bridge `--batch-manifest` executes multiple testcases in a single process invocation and matches single-testcase execution results.

### 10.4 Multi-Bug Agent Evaluation (M4)
- [x] `nf4_block_scale_offset` and `kv_cache_index_stride` benchmarks pass deterministic fake-agent CI checks (both repair and holdout verification).
- [x] `--diagnostic-mode status_only` strips element-level coordinates and metrics from MCP responses while preserving audit/attestation state machine invariants.

## 11. Implementation Tasks & File Mapping

### I1: C++ Quantization Spec & Testcase 0.4 Reader
- **Files**:
  - `cpp/include/inferref/quantization.hpp` (new)
  - `cpp/include/inferref/testcase.hpp`
  - `cpp/examples/testcase_selftest/main.cpp`
  - `scripts/generate_cpp_golden.py`
- **Outcome**: Section 5.2 & 5.3 implemented and verified by C++ self-tests on Linux and Windows.

### I2: C++ Runtime Bridge Quantized Dispatch & Batch Mode + Scaffolder Update
- **Files**:
  - `cpp/include/inferref/bridge.hpp`
  - `cpp/examples/bridge_selftest/main.cpp`
  - `inferref/adapter/scaffold.py`
  - `inferref/agent/executor.py`
  - `tests/core/test_adapter_scaffold.py`
- **Outcome**: Section 5.4 and Section 7.3 implemented; scaffold compile test covers quantized `0.4` testcase.

### I3: Hardware Quantization Contracts (`bf16`, `bias`, `w8a8-int32-accum`) & Core Region Contracts
- **Files**:
  - `inferref/quantization/reference.py`
  - `inferref/contracts/builtin.py`
  - `inferref/testcase/requirements.py`
  - `tests/core/test_contracts.py`
  - `tests/core/test_quantization.py`
- **Outcome**: Section 6.1, 6.2, and 6.3 implemented with 100% numpy-only core test coverage.

### I4: Hierarchical Multi-Layer Trace Compiler & Binary-Search Alignment Planner
- **Files**:
  - `inferref/alignment/planner.py`
  - `inferref/alignment/narrowing.py`
  - `examples/quantization/qualify_whole_model_generation.py`
  - `tests/core/test_alignment_planner.py`
  - `tests/core/test_alignment_e2e.py`
- **Outcome**: Section 7.1 and 7.2 implemented; serial-chain midpoint binary search verified on 28-layer fixture.

### I5: Multi-Bug Agent Benchmark Suite & Diagnostic Ablation Flag
- **Files**:
  - `inferref/agent/evaluation.py`
  - `inferref/agent/evaluation_mcp.py`
  - `examples/agent_eval/nf4_block_scale_offset/`
  - `examples/agent_eval/kv_cache_index_stride/`
  - `tests/agent/test_repair_evaluation.py`
- **Outcome**: Section 8 implemented and verified in CI.

## 12. Dispatch Order (Execution Waves)

```text
Wave 1 (Parallel Foundation):
  ├── I1: C++ Quantization Spec & Testcase 0.4 Reader
  └── I3: Hardware Quantization Contracts (bf16, bias, W8A8) & SwiGLU/GQA Contracts

Wave 2 (Depends on Wave 1):
  ├── I2: Runtime Bridge Quantized Dispatch, Batch Mode & Scaffolder 0.4 Support
  └── I4: Hierarchical Multi-Layer Trace Compiler & Binary-Search Alignment Planner

Wave 3 (Depends on Wave 1 & 2):
  └── I5: Multi-Bug Agent Benchmark Suite & Diagnostic Mode Ablation

Wave 4 (Release Gate):
  └── Full no-Torch core pytest, frontend pytest, C++ ctest (Linux + Windows),
      Ruff lint, Agent workflow skill check, and Intel Arc A770 SYCL gate.
```

## 13. Release Theme

```text
0.11: "Python 与 SYCL 适配器能否跑通量化算子与整网生成验证？"

0.12: "原生 C++ 引擎能否零胶水直读 0.4 量化用例，并在整网发散时通过二分编排自动定位到具体算子？"
```

One-sentence goal:

> InferRef 0.12.0 closes the Python-to-C++ quantized testcase gap and upgrades whole-model alignment from flat end-to-end checks to logarithmic layer-and-region fault isolation.
