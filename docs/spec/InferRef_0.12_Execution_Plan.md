# InferRef 0.12.0 execution plan (PM)

- Status: approved 2026-10-22; scope locked; W8A8 formula pending sign-off
- Source scope: [InferRef_0.12_Core_Infrastructure.md](InferRef_0.12_Core_Infrastructure.md)
- W8A8 details: [InferRef_0.12_W8A8_Contract.md](InferRef_0.12_W8A8_Contract.md)
- 0.11 baseline audit: [2026-10-22 release audit](../dev/status/2026-10-22-v0110-release-audit.md)

## 1. Scope lock

| Milestone | Priority | Release rule |
| --- | --- | --- |
| M1 C++ Testcase 0.4 + quantization ABI | P0 | 0.12.0 release gate |
| M2 hardware contracts + SwiGLU/SDPA contracts | P0 | 0.12.0 release gate |
| M3 hierarchical trace + binary-search narrowing + batch bridge | P0 | 0.12.0 release gate |
| M4 multi-bug Agent benchmark + diagnostic mode | P1 | May ship in 0.12.0 or 0.12.x |

Confirmed decisions (user, 2026-10-22):

1. 0.12.0 gate is M1–M3; M4 is P1.
2. Add explicit new contract IDs; unify naming:
   `linear/affine-int8-bias/v1`, `linear/affine-w4a16-bf16/v1`,
   `linear/nf4-w4a16-bf16/v1`, `linear/w8a8-int32-accum/v1`.
3. W8A8 semantics freeze before I3b; see the dedicated freeze document.
4. C++ dequantization acceptance is **float32 bit equality**, not `atol`.
5. M3 CI uses a synthetic 28-layer fixture; a real Qwen3 checkpoint is optional
   local evidence only.
6. Batch execution is opt-in via adapter feature `batch_execution`; used only
   when `K > 1`; a failed item does not abort the batch; the process exits
   non-zero at the end.
7. A770 local run is attempted; all Linux verification is deferred until WSL is
   enabled. Do not claim Linux coverage before that.
8. PM edits only design/planning documents; development, testing, acceptance,
   and release are dispatched to sub-agents.
9. No calendar deadline; parallelism may be chosen by the PM.

## 2. Work packages and file ownership

One writer per worktree. Wave 2/3 branches are based on the merged Wave 1
commit. A work package must not edit files owned by another package.

### I1 — C++ Testcase 0.4 and quantization ABI (Wave 1, P0)

Owner: C++ developer. Branch `feat/0.12-i1-cpp-testcase-04`.

Exclusive files:

- `cpp/include/inferref/quantization.hpp` (new)
- `cpp/include/inferref/testcase.hpp`
- `cpp/include/inferref/json.hpp` (only if the parser needs null/array support;
  report the change)
- `cpp/examples/testcase_selftest/main.cpp`
- `cpp/examples/quantization_selftest/main.cpp` (optional new target)
- `cpp/CMakeLists.txt`
- `scripts/generate_cpp_golden.py`
- `tests/core/test_cpp_quantization_goldens.py` (new)
- `tests/fixtures/cpp_quantization_goldens/` (new, generated goldens)

Deliverables:

1. `inferref::QuantizationSpec` parser/validator with null-aware
   `zero_point_value_id`, `axis`, and `group_size`; `codebook` only for NF4.
2. `Testcase::Load` accepts 0.1–0.4. Top-level `quantization_parameters`
   entries use the actual `value_id` key. Reject quantization metadata when
   `format_version != "0.4"`.
3. Fail-closed checks: path traversal, missing/duplicate parameter IDs, header
   versus manifest dtype/shape/nbytes mismatch, missing scale/zero/codebook
   references, unreferenced/duplicate parameters.
4. `Testcase::DequantizeInputToFloat32` for `affine-int8/v1`,
   `affine-int4/v1`, `affine-uint4/v1` (`per_tensor`, `per_axis`, `per_group`)
   and `nf4/v1` (`per_block`, flattened row-major blocks). Bit-exact with
   `inferref/quantization/reference.py` and `inferref/quantization/nf4.py`.
5. Python golden generator writes float32 bit patterns (hex or uint32) plus
   descriptors and payloads; C++ self-test compares bits. Cover odd INT4 tails,
   INT8, per-tensor/per-axis/per-group, asymmetric zero points, zero scales,
   NF4 codebook indices 0–15, and blocks crossing row boundaries.
6. No Python/numpy dependency in the C++ reader or dequantizer.

Acceptance gate:

- New C++ self-test and `ctest` pass on Windows.
- Python-generated goldens match C++ bit-for-bit for all supported schemes.
- Negative tests for every fail-closed rule.
- `scripts/generate_cpp_golden.py` is deterministic and checked in CI.
- C++ code remains portable; Linux build is written but **unverified** until
  WSL is enabled.

### I3a — bias, bf16, SwiGLU, SDPA contracts (Wave 1, P0)

Owner: Python core developer. Branch `feat/0.12-i3a-hardware-contracts`.

Exclusive files:

- `inferref/contracts/builtin.py`
- `inferref/contracts/schema.py` (feature vocabulary only, if needed)
- `inferref/quantization/reference.py`
- `inferref/quantization/__init__.py`
- `inferref/quantization/artifacts.py` (feature tokens only, if needed)
- `inferref/testcase/requirements.py` (only if feature derivation needs it)
- `tests/core/test_contracts.py`
- `tests/core/test_quantization.py`
- `tests/core/test_contract_registry.py` (approved deviation: only the minimal
  builtin-ID expectation-list edits required by the new contracts)

Deliverables:

1. `linear/affine-int8-bias/v1`: `x` float32, `weight` affine INT8,
   `bias` float32 `[out_features]`, `y` float32; no integer accumulation.
2. `linear/affine-w4a16-bf16/v1` and `linear/nf4-w4a16-bf16/v1`: `x` and `y`
   bfloat16, weight affine INT4/UINT4 or NF4, float32 dequantization and
   matmul, one nearest-even bfloat16 output conversion, no bias.
3. `swiglu/mlp/v1`: inputs `x`, `gate_weight`, `up_weight`, `down_weight`;
   output `y`; validate shapes, dtypes, and `y` shape/dtype.
4. `attention/sdpa-gqa/v1`: inputs `query`, `key`, `value`; output
   `attn_output`; validate rank-4 MHA/GQA/MQA shapes, `heads % kv_heads == 0`,
   head-dim equality, and dtype equality.
5. Do not implement W8A8 in I3a; that is I3b after sign-off.
6. Positive and negative validator tests; `inferref contract list --json`
   includes the new IDs; existing contracts remain unchanged.

Acceptance gate:

- New contract validators reject every negative case and accept positive cases.
- bf16 reference matches float32 dequantization/accumulation and a single
  nearest-even bf16 output conversion; no float16-only assumptions.
- Full core suite green; no frontend regressions.
- numpy-only; no Torch import in core paths.

### I3b — W8A8 INT32 accumulation (Wave 1b, P0; approved 2026-10-22)

Owner: same Python core developer as I3a, separate branch
`feat/0.12-i3b-w8a8`, based on the merged Wave 1 commit. The freeze is approved;
use [InferRef_0.12_W8A8_Contract.md](InferRef_0.12_W8A8_Contract.md) as the
implemented semantics.

Exclusive files: same set as I3a, plus `tests/core/test_w8a8_reference.py`.

Deliverables:

1. `linear/w8a8-int32-accum/v1` with the frozen semantics in
   [InferRef_0.12_W8A8_Contract.md](InferRef_0.12_W8A8_Contract.md).
2. `linear_w8a8_int32_reference` in float32 epilogue with exact int32 group
   accumulation, zero-point handling, overflow rejection, and no bias.
3. Contract feature `quantization/w8a8-int32-accum/v1` registered and merged
   into requirements; adapter preflight rejects missing capability before launch.
4. Positive/negative tests and injected-bug tests (wrong zero-point, wrong
   group, float64 accumulation, missing scale, overflow).

Acceptance gate: matches the independent int32/oracle test; all overflow and
shape/descriptor violations fail closed; full core suite green.

### I2 — Runtime bridge quantized dispatch, batch mode, scaffolder (Wave 2, P0)

Owner: C++/adapter developer. Branch `feat/0.12-i2-bridge-batch`.

Depends on I1 merged. Exclusive files:

- `cpp/include/inferref/bridge.hpp`
- `cpp/examples/bridge_selftest/main.cpp`
- `inferref/adapter/scaffold.py`
- `inferref/agent/executor.py`
- `tests/core/test_adapter_scaffold.py`
- `tests/core/test_bridge_batch.py` (new)

Deliverables:

1. Legacy `DebugInvoke` keeps working for unquantized testcases.
2. Add `DebugInvokeEx` / `RunBridge` overload passing `const Testcase &` so an
   engine can query `HasInputQuantization`, `InputQuantization`, and
   `QuantizationParameter`.
3. A legacy 2-argument bridge receiving a 0.4 testcase exits with
   `kBridgeUnsupportedQuantization = 4` and a diagnostic; never silently
   ignores scales.
4. Scaffolder on a 0.4 testcase emits compilable snippets for quantization
   queries and populates `capabilities.features` from `derive_requirements`.
5. `--batch-manifest batch.json` executes ordered `{id, testcase_dir,
   output_dir}` items in one process; writes per-item manifests and
   `batch-results.json`; partial failure is recorded; final exit code non-zero
   if any item fails.
6. Adapter feature `batch_execution`; `AlignmentRunner.run(limit=K)` uses batch
   only when `K > 1` and the adapter declares the feature; otherwise sequential
   fallback is explicit.

Acceptance gate:

- C++ self-test for quantized dispatch and batch ordering/partial failure.
- Scaffold compile test for a 0.4 testcase (`ctest`/CMake).
- A fake adapter proves one process launch for K>1 and correct per-item output.
- Existing adapter v0.2 and scaffold tests remain green.

### I4 — Hierarchical trace and binary-search narrowing (Wave 2, P0)

Owner: alignment/trace developer. Branch `feat/0.12-i4-hierarchical-align`.

Depends on I3a/I3b for contract-aware testcases. Exclusive files:

- `inferref/frontend/pytorch/hierarchical.py` (new; or an equivalent module)
- `inferref/alignment/planner.py`
- `inferref/alignment/narrowing.py`
- `examples/quantization/qualify_whole_model_generation.py`
- `tests/core/test_hierarchical_trace.py` (new)
- `tests/core/test_alignment_planner.py`
- `tests/core/test_alignment_e2e.py`

Deliverables:

1. `compile_step_trace` attaches forward hooks for the whole model, each
   decoder layer, each sub-block (`input_layernorm`, `self_attn`,
   `post_attention_layernorm`, `mlp`), and leaf projections; emits nested
   `RegionRecord` spans.
2. Optional explicit quantization-spec attachment for leaf projections. Do not
   claim automatic live capture of third-party quantized tensors; use an
   explicit projection/attach path.
3. Synthetic 28-layer fixture (no 440 MB checkpoint) in CI; real Qwen3 only as
   optional local evidence.
4. Planner binary-search priority for a failing serial chain: deterministic
   midpoint/first-layer policy with reason
   `"binary search midpoint of failing serial chain"`; 28 layers isolated in
   at most `ceil(log2(28)) + 3 = 8` boundary evaluations.
5. Fix the 0.11 report semantics: `qualify_whole_model_generation.py` derives
   top-level `status` from comparisons/Suite, renames the reference device field
   to `reference_device`, and points to the final evidence run. Add a regression
   that a failed comparison cannot report `status: pass`.
6. No regression to existing flat trace alignment; plan/run agreement remains.

Acceptance gate:

- Synthetic 28-layer trace validates and narrows a single injected layer defect
  in ≤8 evaluations (counted by a fake runner).
- Nested regions are deterministic and serialized with stable order.
- The report-status regression fails on the old hard-coded behavior.
- Existing alignment/planner tests green.

### I5 — Multi-bug Agent benchmark and diagnostic mode (Wave 3, P1)

Owner: Agent/evaluation developer. Branch `feat/0.12-i5-agent-bench`.

Depends on M1–M3 interfaces. Exclusive files:

- `inferref/agent/evaluation.py`
- `inferref/agent/evaluation_mcp.py`
- `examples/agent_eval/nf4_block_scale_offset/` (new)
- `examples/agent_eval/kv_cache_index_stride/` (new)
- `tests/agent/test_repair_evaluation.py`

Deliverables:

1. `nf4_block_scale_offset`: off-by-one flat block indexing across row
   boundaries; visible and holdout cases; deterministic fake-agent repair.
2. `kv_cache_index_stride`: wrong decoded sequence slice in decode step 1;
   visible and holdout cases.
3. `--diagnostic-mode {full,status_only}` on `inferref agent evaluate`;
   `status_only` strips element-level coordinates/metrics from MCP responses
   while preserving audit/attestation invariants.
4. Existing `rope_sign` benchmark remains green.

Acceptance gate: deterministic fake-agent pass/fail; holdouts verified; mode
ablation cannot leak element coordinates; attestation FSM invariants hold.

## 3. Waves

```text
Wave 0 (this change): docs, spec commit, 0.11 audit, W8A8 freeze.
Wave 1 (parallel): I1 + I3a.
Wave 1b: I3b after W8A8 sign-off.
Wave 2 (parallel after Wave 1): I2 + I4.
Wave 3: I5.
Wave 4 (release gate): full verification, evidence, release notes.
```

A Wave 2 branch is based on the merged Wave 1 commit. Do not run two writers in
one worktree.

## 4. Verification matrix

| Platform | Scope | Status |
| --- | --- | --- |
| Windows local | Python core + frontend, C++ ctest, Ruff, Agent skill | Required for every WP |
| Windows A770 | whole-model / XPU gate; optional per WP | Attempt locally; record evidence |
| Linux (WSL) | core/agent/frontend/cpp Linux | **Deferred** until user enables WSL |
| Remote CircleCI | Linux/Windows matrix | Unverified; network currently blocked |
| CUDA/ROCm | accelerator contracts | Not available |
| Real Qwen3 checkpoint | optional local evidence | Not a CI dependency |

No sub-agent may report Linux coverage until WSL is enabled and the commands are
actually run. Windows-only results must say so.

## 5. Definition of done (each WP)

- Focused new tests fail before the change and pass after it.
- Full relevant suite green on Windows: `tests/core`, `tests/frontend`,
  `tests/agent`, `tests/docs` as applicable; C++ ctest for C++ changes.
- `ruff check inferref tests scripts` passes.
- No-Torch core acceptance remains intact (subprocess torch-block test).
- Compatibility rules respected: `.irtensor` v1/v2, Trace IR 0.1/1.0 unchanged;
  testcase 0.1–0.4 readable; adapter v0.2 additive only.
- User-visible changes recorded in `CHANGELOG.md` under `Unreleased`.
- Evidence paths and exact command outputs included in the WP report.
- No unrelated refactors or generated artifacts in the patch.

## 6. Dispatch briefs

### I1 brief

> Implement 0.12 M1 in an isolated worktree. Read
> `docs/spec/InferRef_0.12_Core_Infrastructure.md` §5,
> `docs/spec/InferRef_0.12_Execution_Plan.md` I1, and
> `inferref/quantization/reference.py`. Only edit the I1 files listed in the
> execution plan. Mirror Python `validate_testcase` fail-closed rules in
> `Testcase::Load`; the top-level key is `value_id`. Add float32 bit-exact
> goldens and a C++ self-test. Run `ctest` on Windows. Do not claim Linux.
> Report changed files, commands, and exact results.

### I3a brief

> Implement 0.12 M2 non-W8A8 parts in an isolated worktree. Read the spec §6 and
> the execution plan I3a. Only edit the I3a files. Keep core numpy-only; do not
> import Torch. Add positive/negative contract tests. Do not implement W8A8.
> Run the focused and full core suites. Report exact results and any deviations.

### I3b brief

> Implement 0.12 W8A8 only after the freeze document is signed off. Follow
> `docs/spec/InferRef_0.12_W8A8_Contract.md` exactly: int32 per-group
> accumulation, explicit zero-point subtraction, int32 overflow rejection,
> float32 epilogue order, no bias. Add independent oracle/injected-bug tests.
> Register the adapter feature token. Report exact results.

### I2 brief

> Implement 0.12 bridge/scaffolder/batch after I1 is merged. Only edit I2 files.
> Preserve legacy `DebugInvoke`; add `DebugInvokeEx`; exit code 4 for a legacy
> bridge receiving 0.4. Implement `--batch-manifest` with per-item status and a
> non-zero final exit on failure. Prove one-process batch execution with a fake
> adapter. Run ctest and adapter/scaffold tests.

### I4 brief

> Implement 0.12 hierarchical trace and narrowing after I3. Only edit I4 files.
> Use a synthetic 28-layer fixture in CI; real Qwen3 is optional local evidence.
> Add deterministic midpoint binary search and measure ≤8 evaluations. Fix the
> hard-coded `status: pass` in the whole-model example and add a regression.
> Preserve existing alignment behavior.

### I5 brief

> Implement 0.12 M4 after M1–M3. Add the two benchmarks and
> `--diagnostic-mode status_only`; preserve attestation/audit invariants. Run
> the Agent evaluation tests with deterministic fake agents. P1: do not block
> the 0.12.0 release gate on this WP unless the user promotes it.

## 7. Release checklist (Wave 4)

- [ ] Windows: full core/frontend/agent/docs pytest; C++ ctest; Ruff; Agent skill.
- [ ] No-Torch core test.
- [ ] A770: fresh XPU gate evidence and, if applicable, whole-model re-run.
- [ ] Linux: **only after WSL is enabled**; otherwise record "not run".
- [ ] `CHANGELOG.md` Unreleased → 0.12.0; release notes `docs/releases/0.12.0.md`.
- [ ] Version bump in `pyproject.toml` and `inferref/ir/version.py`.
- [ ] Distribution build/check/install in a fresh directory.
- [ ] Durable evidence summary under `docs/dev/attestations/` (0.11 and 0.12).
- [ ] 0.12 status snapshot in `docs/dev/status/` with gaps and unverified items.
- [ ] Tag/release only after the user approves; remote CI results must be green
      where available.

## 8. Risk register

| Risk | Mitigation |
| --- | --- |
| C++/Python dequantization bit mismatch | float32 bit goldens; no fast-math; cover LUT constants, odd tails, cross-row NF4 |
| W8A8 axis/zero-point ambiguity | freeze document; x rank 2, per-axis x=-2, per-axis weight=0; explicit sign-off |
| Contract-ID proliferation | explicit IDs confirmed; per-contract capability tests; naming symmetry |
| Hierarchical hooks are model-specific | synthetic fixture + explicit spec attachment; real model optional |
| Binary-search tie-break nondeterminism | deterministic midpoint/first policy and fake-runner evaluation count |
| Batch bridge partial failure | per-item status, `batch-results.json`, final non-zero exit; one-process test |
| 0.11 evidence not durable | committed summary + hashes; do not rely on `.scratch` |
| WSL/Linux unavailable | record "not run"; do not claim; run after user enables WSL |
| A770/oneAPI environment drift | record driver/oneAPI versions; local xpu-env path; fresh evidence dirs |
