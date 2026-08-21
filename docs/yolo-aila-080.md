# YOLO26 / Aila TFCM record (0.8.0)

TFCM is Time-to-First-Correct-Model: the wall-clock time from "I have a model
family and a runtime" to the first `inferref agent run` / `inferref agent
compare` result of `pass` for a region of that model.

The 0.8.0 Adapter DX exists because the YOLO26 / Aila experiment showed that
the numerical harness works across a new model family, but writing per-region
C++ adapter glue still took too long. This document is the runbook that
produces a TFCM record, plus the first record produced during 0.8.0
development.

## 1. TFCM runbook

Every step below is a real `inferref` command on current `main`. Record the
wall-clock time of steps 3 through 8 on a clean checkout; that total is the
TFCM for the selected region.

```bash
# 1. Produce a reference trace of the model family
python examples/mini_llama/run_trace.py --output trace/

# 2. Find the semantic regions and inspect the trace
inferref region detect trace/
inferref agent context trace/ --json

# 3. Extract the smallest engine boundary (contract-bound when possible)
inferref agent extract trace/ --region "RoPE@layers.0.self_attn" \
  -o repro/rope \
  --input-names cos,sin,query,key \
  --output-names q_embed,k_embed \
  --contract rope/rotate-half/v1 --json

# 4. Scaffold the adapter project from the extracted testcase
inferref adapter scaffold repro/rope --language cpp --output adapter/
# or for a runtime that dispatches many regions:
inferref adapter scaffold repro/rope --language cpp --runtime-bridge \
  --output bridge/

# 5. Implement engine dispatch
#    standard mode:  replace RunYourEngine in adapter/main.cpp
#    bridge mode:    replace the DebugInvoke callback in bridge/main.cpp

# 6. Build (point INFERREF_CPP_INCLUDE at this checkout's cpp/include)
cmake -S adapter -B adapter/build \
  -DINFERREF_CPP_INCLUDE=/path/to/inferref/cpp/include
cmake --build adapter/build

# 7. Run the engine through the adapter
inferref agent run repro/rope --adapter adapter/adapter.json \
  --runs-dir runs --json

# 8. Record the result
inferref agent compare repro/rope runs/<run-id>/output --json
```

For a stateful model family (prefill -> decode), the same loop applies per
scenario step; validate the chain first with
`inferref scenario validate scenarios/kv-chain --json` and run it through one
adapter with `inferref agent run_scenario`.

To apply the runbook to the YOLO26 workflow, substitute the model's trace
producer for step 1, its detected regions for step 2, and the runtime's real
dispatch implementation for step 5. The rest of the toolchain is model-agnostic.

## 2. First TFCM record

- **Date:** 2026-08-21
- **Revision:** `80a29485016c693532f0b92495c9a66e4f1609ce`
- **Environment:** Windows 11, MSVC 19.50 + Ninja, Python 3.13, InferRef 0.8.0
  scope (C++ testcase helper, adapter scaffold, runtime bridge).
- **Target:** MiniLlama RoPE region (extraction) and the kv-chain prefill
  fixture (bridge end-to-end), as the repository's stand-in for a new model
  family. Real YOLO26 / Aila model integration is the target workflow and
  remains manual.
- **Loop measured (clean scratch directory):**

  | Step | Wall clock | Result |
  | --- | --- | --- |
  | trace (2-layer MiniLlama) | ~10 s | trace written, 109+ operators |
  | region detect | <1 s | RoPE / RMSNorm / Attention / SwiGLU regions |
  | agent extract (contract-bound RoPE) | <1 s | `pass`, reproducible |
  | adapter scaffold (standard and bridge mode) | <1 s | four files, `adapter.json` passes `EngineAdapter.load()` |
  | C++ build (Ninja + MSVC) | ~5 s | zero warnings |
  | bridge run + compare | <2 s | `pass` on kv-cache/append/v1 via `inferref agent compare` |

- **First correct model verdict:** `pass` for the KV prefill region through
  the runtime bridge with a stub `DebugInvoke` implementing KV append,
  compared with `inferref agent compare` against the fixture reference.
- **Artifacts:** development scratch directories; the runbook commands and
  the C++ self-tests are committed and reproducible in CI.
- **Known gaps:**
  - The first record uses repository fixtures, not a real YOLO26 model; the
    runbook is the bridge to that workflow.
  - No comparison against a 0.7.x baseline is claimed (the 0.8 release does
    not have one).
  - A full multi-region model requires the `DebugInvoke` callback to dispatch
    per region key; the bridge itself is region-agnostic.
