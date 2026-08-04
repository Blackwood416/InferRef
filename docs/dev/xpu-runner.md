# Windows Intel XPU self-hosted runner

The XPU gate is deliberately manual. A self-hosted runner executes checked-out
code with the runner account's privileges, so the workflow is not triggered by
pull requests.

## Provision the immutable Python environment

From an elevated PowerShell prompt:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\provision-xpu-runner.ps1
```

This creates `E:\InferRefRunner\xpu-env` with Python 3.13.11 and
`torch==2.13.0+xpu`. The workflow never runs `setup-python` or `pip`; update the
environment only by deliberately rerunning the provisioning script.

Set the repository Actions variable `INFERREF_XPU_PYTHON` to:

```text
E:\InferRefRunner\xpu-env\Scripts\python.exe
```

## Register the runner

Use GitHub's **Settings → Actions → Runners → New self-hosted runner** commands,
install under `E:\InferRefRunner\actions-runner`, and add the custom label
`xpu`. Run it as a Windows service so the labels are:

```text
self-hosted, windows, x64, xpu
```

No registration token or credential belongs in this repository. Once online,
start **Intel XPU compatibility** from the Actions `workflow_dispatch` page.
