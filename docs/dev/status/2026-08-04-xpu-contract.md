# InferRef 0.3.1 — accelerator and Intel XPU contract

## Outcome

InferRef's only hardware contract is no longer CUDA-specific. Payload capture
now has an explicit asynchronous-device materialization boundary shared by
CUDA/ROCm and Intel XPU, and execution device metadata is inferred from the
actual values in the trace.

## Delivered

- generic accelerator discovery, synchronization and host materialization;
- `inferref doctor [--device DEVICE] [--json]`;
- automatic `execution.device` inference (`cpu`, `xpu:0`, `cuda:0`, `mixed`,
  or `unknown`);
- parameterized CUDA/XPU contract tests;
- a manual-only Windows XPU self-hosted workflow that never overwrites the
  runner's PyTorch environment;
- a reproducible XPU runner provisioning script and operations guide.

## Local hardware evidence

Validated on Windows with an Intel Arc A770 and `torch 2.13.0+xpu`:

- FP16 and BF16 full payload capture;
- view aliasing and mutation storage versions;
- pinned CPU/XPU storage identity separation;
- host materialization from work enqueued on a non-default XPU stream.

The non-default physical device test is skipped because the host has one XPU.
Linux/WSL XPU and CUDA remain future hardware gates; they use the same test
contract.
