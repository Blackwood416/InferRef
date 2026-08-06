# Windows Intel XPU and native SYCL gate

The manual XPU workflow targets a self-hosted Intel Arc A770. It uses the
preconfigured Python named by `INFERREF_XPU_PYTHON`; the workflow neither runs
`setup-python` nor installs packages.

For a local gate, install Visual Studio C++ tools, Ninja, and Intel oneAPI, then
run:

```powershell
./scripts/run_xpu_sycl_gate.ps1 `
  -Python E:/InferRefRunner/xpu-env/Scripts/python.exe `
  -ExpectedDeviceNameRegex "Arc.*A770"
```

The script initializes both Visual Studio and oneAPI environments, builds with
`icx-cl /fsycl`, runs the versioned XPU corpus, produces HTML/JSON Suite
reports, and verifies Python/C++ comparator agreement. It also executes
deliberately corrupted RMSNorm, RoPE, and KV-cache outputs and requires both
comparators to reject them.

Evidence directories are one-run-one-directory: the script refuses to reuse an
existing evidence root unless `-CleanEvidenceDirectory` is explicitly passed,
and the workflow scopes each run as
`xpu-sycl-evidence/<run-id>-<attempt>`. This prevents stale output tensors,
device evidence, or negative-corruption files from being read as if they
belonged to the current run.

The engine creates its queue with `gpu_selector_v`, rejects non-Intel devices
and non-Level-Zero backends, and writes `inferref-sycl-device` v0.1 evidence for
every case. The gate fixes `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`, validates
every evidence record, and uploads an aggregate `device-evidence.json` with
device name, vendor, driver, backend, type, and global memory.

Two qualification levels are supported. The generic gate passes any Intel
Level Zero GPU (`-ExpectedDeviceNameRegex` empty). Passing
`-ExpectedDeviceNameRegex "Arc.*A770"` additionally requires every case's
device name to match, so a runner that silently falls back to an iGPU or a
different Arc SKU fails the workflow instead of staying green.

`inferref_sycl_engine` is a native executable and does not import or link
Python/PyTorch. It reads and writes `.irtensor` directly. The gate records its
native dependency table and rejects Python/PyTorch dependencies.

The first release covers Windows. Linux/WSL XPU and CUDA/ROCm share the frontend
accelerator contract but are not release blockers.
