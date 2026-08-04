# Windows Intel XPU and native SYCL gate

The manual XPU workflow targets a self-hosted Intel Arc A770. It uses the
preconfigured Python named by `INFERREF_XPU_PYTHON`; the workflow neither runs
`setup-python` nor installs packages.

For a local gate, install Visual Studio C++ tools, Ninja, and Intel oneAPI, then
run:

```powershell
./scripts/run_xpu_sycl_gate.ps1 `
  -Python E:/InferRefRunner/xpu-env/Scripts/python.exe
```

The script initializes both Visual Studio and oneAPI environments, builds with
`icx-cl /fsycl`, runs the versioned XPU corpus, produces HTML/JSON Suite
reports, and verifies Python/C++ comparator agreement. It also executes
deliberately corrupted RMSNorm, RoPE, and KV-cache outputs and requires both
comparators to reject them.

`inferref_sycl_engine` is a native executable and does not import or link
Python/PyTorch. It reads and writes `.irtensor` directly. The gate records its
native dependency table and rejects Python/PyTorch dependencies.

The first release covers Windows. Linux/WSL XPU and CUDA/ROCm share the frontend
accelerator contract but are not release blockers.
