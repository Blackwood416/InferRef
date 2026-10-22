# v0.11.0 release evidence audit and 0.12 readiness

- Date: 2026-10-22
- Auditor: PM session
- Scope: shipped 0.11.0 content, release evidence, leftovers, and 0.12 prerequisites
- Environment: Windows 11; Python 3.13.11; torch 2.13.0+cpu; MSVC 19.x + Ninja;
  Intel Arc A770 (driver 32.0.101.9150, Level Zero 1.15.38308); oneAPI 2026.1
  installed at `C:\Program Files (x86)\Intel\oneAPI` but not on `PATH`;
  WSL is **not** enabled; GitHub is unreachable from this host.

## 1. Conclusion

1. The 0.11.0 tree is healthy on the local Windows gates: **1088 passed,
   9 skipped, 3 deselected**; C++ `ctest` **5/5**; Ruff, the Agent workflow
   check, and the distribution checker pass.
2. The release notes' core claims are supported by the final `qual-5` evidence:
   3/3 Suite, 100% token parity, cosine 0.99988, A770 device evidence,
   Alignment state `verified`, and the injected bug detected.
3. Two evidence/documentation defects must be fixed before 0.12 M3 reuses the
   same machinery:
   - `docs/spec/InferRef_Quantization.md` cites the **failed intermediate**
     `whole-model-qwen3-gpu-qual-final` report instead of `qual-5`.
   - `examples/quantization/qualify_whole_model_generation.py` hard-codes
     `"status": "pass"` and never derives it from comparisons. The failed
     `qual-final` run therefore also reports `status: pass`.
4. The A770 and real bitsandbytes evidence exists only under `.scratch/`, which
   is gitignored. There is no committed attestation or durable evidence summary.
5. Remote tag/release/CI and **all Linux verification are unverified**: WSL is
   off and the network cannot reach GitHub. Linux work is deferred until the
   user enables WSL.

## 2. Version and repository

| Item | Value |
| --- | --- |
| Branch / commit | `main` @ `7361777` (`chore: bump version to 0.11.0`, 2026-09-21) |
| Tag | `v0.11.0` (local) |
| `origin/main` | same commit as local `main` |
| Remote tag/release | **unverified** — `git ls-remote` failed: proxy `127.0.0.1:443` unreachable |
| Version sources | `pyproject.toml` 0.11.0; `inferref --version` 0.11.0 |
| Untracked at audit start | `docs/spec/Inferref_0.12_Core_Infrastructure.md` (now committed as `InferRef_0.12_Core_Infrastructure.md`) |
| Stale branch | `codex/scenario-v0.1`: 43 commits behind `main`, zero unique commits |
| `dist/` | no 0.11.0 artifacts (latest is 0.9.2); 0.11.0 wheel/sdist live in `.scratch/release-0.11.0-dist/` |
| PyPI | `inferref` returns 404; this is expected — `docs/DISTRIBUTION.md` states PyPI publication is a separate maintainer step |

## 3. Local verification matrix

| Gate | Command | Result | Notes |
| --- | --- | --- | --- |
| Python full suite | `.venv\Scripts\python.exe -m pytest -q -rs` | **1088 passed, 9 skipped, 3 deselected** in 255.67 s | Windows, py3.13, torch 2.13.0+cpu |
| C++ | `ctest --test-dir cpp/build -C Release --output-on-failure` | **5/5 passed** | `irtensor`, `fp8`, `packed`, `testcase`, `bridge` |
| Ruff | `.venv\Scripts\ruff.exe check inferref tests scripts` | **All checks passed** | |
| Agent workflow skill | `.venv\Scripts\python.exe scripts/check_agent_workflow_skill.py --repo .` | **OK** (6 files, 43 commands, 144 flags, 14 MCP tools) | |
| Distribution | `.venv\Scripts\python.exe scripts/check_distributions.py .scratch/release-0.11.0-dist` | **pass** (`version=0.11.0`, `installed=false`) | Wheel 361,142 B; sdist 774,846 B |
| XPU SYCL gate (existing evidence) | `.scratch/xpu-sycl-gate/report.json` | **13/13 pass** | A770, driver 1.15.38308, 16,706,793,472 B global memory |
| Whole-model generation (existing evidence) | `.scratch/whole-model-qwen3-gpu-qual-5/report.json` | **pass** | 3/3 Suite; details in §5 |

Default-suite skips (all environment-gated, not product failures):

- 1 POSIX-only interpreter test (`os.name != "nt"`).
- 4 accelerator-contract tests: the default `.venv` is CPU-only.
- 1 Qwen3.5 0.8B checkpoint test: `INFERREF_QWEN35_MODEL` is not set.
- 2 XPU Transformers tests: the default `.venv` torch is CPU-only. A separate
  `E:\InferRefRunner\xpu-env` has `torch 2.13.0+xpu` and reports XPU available.

The 3 deselected tests are `slow` installation gates (default `-m "not slow"`).

## 4. 0.11 acceptance audit

| Milestone / claim | Evidence | Verdict |
| --- | --- | --- |
| Q1 native FP8 storage, decoding, Python/C++ roundtrips | `tests/core` FP8 tests, `cpp/examples/fp8_selftest`, full suite + ctest | Verified by tests; no FP8 GPU execution claimed |
| Q2b quantized INT8 trace/testcase/preflight/comparison | `tests/core/test_quantization*.py`, `tests/frontend/test_quantization_reference.py`, example workflow | Verified |
| Q3 packed INT4/UINT4 codec and affine Linear | exhaustive nibble/byte-pair tests, Python/C++ goldens, ctest 5/5 | Verified |
| Q4 NF4 codebook/block semantics and bnb importer | NF4 tests, real QuantState audit, ctest | Verified on CPU |
| Real bitsandbytes 0.50.1 QuantState | `.scratch/qwen3-real-nf4-20260918/report.json`: 196/196 weights, 440,401,920 elements, `max_abs_error=0.0`, source SHA-256 unchanged | Verified on CPU; weights not committed |
| Whole-model causal generation + A770 execution | `qual-5`: 3/3 Suite, tokens `[16, 13, 576]` ("1. The"), cosine `0.999882 / 0.999881 / 0.999877`, A770 device evidence, Alignment `verified`, injected bug detected | Verified on A770; **wrong report cited in the spec** |
| Generic XPU/SYCL gate | `.scratch/xpu-sycl-gate/report.json`: 13/13 pass, A770 driver 1.15.38308 | Verified |
| Package distributions | `.scratch/release-0.11.0-dist`: wheel/sdist, checker pass | Local build verified; remote release unverified |
| Remote CI / GitHub release | CircleCI and GitHub Actions config exist | **Unverified** — network blocked |
| C++ execution of quantized testcases | Explicitly "not qualified" in 0.11.0 release notes | Deferred to 0.12 M1 |

## 5. Evidence chain and key numbers

### 5.1 Final whole-model GPU qualification (`qual-5`)

Source: `.scratch/whole-model-qwen3-gpu-qual-5/`

- `report.json`: `status=pass`, all three steps `adapter_status=pass` and
  `compare_pass=true`; `suite_passed=3`, `suite_total=3`;
  `alignment_status=pass`; `injected_error_detected=true`; `plan_run_agree=true`.
- Per-step top tokens: `16`, `13`, `576`; all `token_match=true`.
- Per-step device evidence: `inferref-sycl-device` v0.1, backend
  `ext_oneapi_level_zero`, `Intel(R) Arc(TM) A770 Graphics`.
- `suite-runs/inferref-suite-run.json`: `counts = {total:3, pass:3,
  unsupported:0, partial:0, failed:0, rejected:0}`.
- Metrics from the prefill suite run: `max_abs_error=0.1875`,
  `mean_abs_error=0.030611`, `rmse=0.039153`, `cosine_similarity=0.999882094`;
  comparison override `atol=0.25`, `rtol=0.05`.
- `alignment/alignment.json`: boundary `region:0` state `verified`, session
  status `verified`.
- `alignment/evidence/ev-28fe32ea-376b-4935-be5f-b81882cd9a8e.json`:
  `result=pass`, `outputs_match=true`, adapter SHA-256
  `e36e4051c6bdfdee7eb0df358be4594647bb5a7668a2fa6a82b662480d7a4b66`.

### 5.2 Failed intermediate (`qual-final`) — do not cite

Source: `.scratch/whole-model-qwen3-gpu-qual-final/report.json`

- `status=pass` (the hard-coded field), but `suite_passed=1`, `suite_total=3`.
- `decode-0`: comparison fail, `cosine_similarity=0.253674568`,
  `max_abs_error=14.01171875`.
- `decode-1`: comparison fail, `cosine_similarity=0.317245166`,
  `max_abs_error=13.421875`.
- The spec currently points here. It must point to `qual-5`.

### 5.3 Real bitsandbytes audit

Source: `.scratch/qwen3-real-nf4-20260918/report.json`

- `status=pass`; `available_nf4_weights=196`; `checked_weights=196`;
  `checked_elements=440401920`; `max_abs_error=0.0`.
- `model.safetensors` SHA-256:
  `075ec08938f270c54e93f07bb8df57a19297b6c4c4b831595059e81a7ba6ce05`;
  `source_unchanged=true`.

### 5.4 Generic XPU gate

Source: `.scratch/xpu-sycl-gate/report.json`

- `counts = {total:13, pass:13, failed:0, rejected:0, unsupported:0}`.
- All 13 cases emit A770 device evidence with driver `1.15.38308`.

## 6. Findings and leftovers

| ID | Severity | Finding | Owner / next action |
| --- | --- | --- | --- |
| A1 | P0 | `docs/spec/InferRef_Quantization.md` cites `qual-final` (failed) as the whole-model report | PM docs fix in this change |
| A2 | P0 | `qualify_whole_model_generation.py` hard-codes `"status": "pass"`; `qual-final` proves it can report pass while comparisons fail | 0.12 I4: derive status from comparisons/Suite; add regression |
| A3 | P0 | A770/bnb/whole-model evidence is only in gitignored `.scratch/`; no durable attestation | Release agent: commit a small evidence summary + hashes under `docs/dev/attestations/` |
| A4 | P1 | No automated test covers whole-model generation; it is a manual GPU example | 0.12 I4: synthetic 28-layer fixture in CI; real model optional |
| A5 | P1 | Top-level `device` in the whole-model report is the **reference** device (`cpu`), while engine device evidence is per-step A770 | 0.12 I4: rename/clarify fields |
| A6 | P1 | `docs/spec/InferRef_Quantization.md` still says "unreleased development branch"; Q4 sections mention source 0.10.0 | PM docs fix: mark 0.11.0 shipped, preserve historical acceptance |
| A7 | P1 | No 0.11.0 status snapshot; `docs/dev/status` index stopped at 2026-09-03 v0.10.0 | This document + index update |
| A8 | P1 | No `CHANGELOG.md` `Unreleased` section | Release agent at 0.12 start |
| A9 | P1 | Stale branch `codex/scenario-v0.1` (43 behind, no unique commits) | Release agent: delete after confirmation |
| A10 | P1 | Default `.venv` is CPU-only, so XPU frontend tests skip; XPU env is separate | Document in CI/release checklist; use `INFERREF_XPU_PYTHON` |
| A11 | P1 | Remote tag/release/CircleCI status and all Linux gates unverified (WSL off, network blocked) | Release checklist; run when WSL/network available |
| A12 | P2 | `dist/` has no 0.11.0 artifacts; release artifacts live in `.scratch/release-0.11.0-dist/` | Clarify distribution policy; do not treat `dist/` as authoritative |

## 7. Not verified / explicit limits

- **Linux**: WSL is not enabled. No Linux pytest, no Linux C++ build, no
  `cpp-linux` / `core-linux` / `frontend-linux` CI result was produced locally.
  Do not claim Linux coverage for 0.11.0 or 0.12.0 until WSL is enabled.
- **Remote CI / GitHub release / tag push**: network to GitHub is unavailable.
- **CUDA / ROCm**: no hardware; accelerator-contract tests skip.
- **Other bitsandbytes versions, deeper nesting, vendor fused low-bit kernels**:
  explicitly outside 0.11.0 scope.
- **C++ execution of 0.4 quantized testcases**: explicitly not qualified in
  0.11.0; this is 0.12 M1.
- **W8A8 integer accumulation**: not implemented; 0.12 M2.
- **Real YOLO26 / Aila model integration**: external, manual.

## 8. 0.12 prerequisites and next steps

Completed by this audit:

- 0.11.0 evidence audit (this document) and status index entry.
- Corrected the `qual-final` pointer and stale status line in the quantization plan.
- Committed and renamed the 0.12 spec to `docs/spec/InferRef_0.12_Core_Infrastructure.md`.
- Added `docs/spec/InferRef_0.12_Execution_Plan.md` with work packages and gates.
- Added `docs/spec/InferRef_0.12_W8A8_Contract.md` for the integer-accumulation
  formula freeze.
- Recorded the Linux/WSL deferral and the A770 local-run environment.

Next:

1. User signs off the W8A8 formula freeze.
2. Wave 1: I1 (C++ 0.4 ABI) and I3a (bias/bf16/SwiGLU/SDPA) in isolated
   worktrees; I3b (W8A8) after sign-off.
3. Wave 2: I2 (bridge/scaffolder/batch) and I4 (hierarchical trace/planner).
4. Wave 3: I5 (Agent benchmark + diagnostic mode).
5. Release agent: `CHANGELOG` Unreleased, durable 0.11 evidence summary, remote
   tag/release checks, Linux gates when WSL is enabled.

## 9. Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q -rs
ctest --test-dir cpp/build -C Release --output-on-failure
.\.venv\Scripts\ruff.exe check inferref tests scripts
.\.venv\Scripts\python.exe scripts/check_agent_workflow_skill.py --repo .
.\.venv\Scripts\python.exe scripts/check_distributions.py .scratch/release-0.11.0-dist

# Existing evidence (read-only):
Get-Content .scratch/xpu-sycl-gate/report.json
Get-Content .scratch/whole-model-qwen3-gpu-qual-5/report.json
Get-Content .scratch/whole-model-qwen3-gpu-qual-5/suite-runs/inferref-suite-run.json
Get-Content .scratch/qwen3-real-nf4-20260918/report.json

# A770 gate (requires oneAPI env; writes new evidence):
.\scripts\run_xpu_sycl_gate.ps1 `
  -Python E:/InferRefRunner/xpu-env/Scripts/python.exe `
  -ExpectedDeviceNameRegex "Arc.*A770"
```

Linux commands are intentionally absent until WSL is enabled.
