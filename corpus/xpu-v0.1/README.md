# InferRef XPU corpus v0.1

Small deterministic testcases for RMSNorm, RoPE, KV append/indexed update and
packed KV cache, including weight-free Tiny Llama and Tiny Qwen3.5
prefill/single-step/multi-step decode slices. They contain no model weights and
are regenerated with:

```bash
python corpus/xpu-v0.1/generate.py
```

`suite.json` tags the cases as prefill/decode TraceSet slices. The payloads are
portable logical tensors; `xpu` describes the native SYCL gate, not a mandatory
execution device for other adapters.
