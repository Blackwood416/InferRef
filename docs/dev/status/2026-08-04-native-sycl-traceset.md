# InferRef 0.5.0 — native SYCL and engine matrices

## Delivered

- optional `INFERREF_BUILD_SYCL=ON` native C++/SYCL testcase engine;
- direct `.irtensor` RMSNorm, RoPE, KV append, indexed update, and packed-cache
  execution without Python or PyTorch in the engine;
- versioned 13-case XPU corpus with FP32/FP16/BF16 and weight-free Tiny
  Llama/Qwen3.5 prefill/decode slices;
- repeatable `--adapter` Suite matrices and static HTML + JSON reports;
- XPU manual gate with all-output Python/C++ verdict agreement and three
  deliberately corrupted negative cases;
- native dependency inspection that rejects Python/PyTorch linkage.

## Local A770 result

Intel oneAPI 2026.1 with `icx-cl /fsycl` built the engine successfully. All 13
positive corpus cases passed on Intel Arc A770. RMSNorm, RoPE, and indexed
KV-cache injected errors were rejected by both comparators.

The first native run exposed two useful integration defects that are now fixed:
the oneAPI 2026 compiler no longer accepts `/Qsycl`, and an indexed cache update
must apply to every outer batch/head group rather than only the first one.

## Boundary

This release proves the portable testcase/codec/comparator boundary with a real
non-PyTorch engine. It does not yet claim Linux/WSL XPU support, CUDA runner
coverage, or a production-optimized kernel library.
