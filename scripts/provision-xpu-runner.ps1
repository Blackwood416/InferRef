[CmdletBinding()]
param(
    [string]$Root = 'E:\InferRefRunner',
    [string]$PythonVersion = '3.13.11'
)

$ErrorActionPreference = 'Stop'
$environment = Join-Path $Root 'xpu-env'
New-Item -ItemType Directory -Force -Path $Root | Out-Null

uv python install $PythonVersion
uv venv --python $PythonVersion $environment
$python = Join-Path $environment 'Scripts\python.exe'
uv pip install --python $python `
    --index-url https://download.pytorch.org/whl/xpu `
    'torch==2.13.0+xpu'
uv pip install --python $python `
    'numpy==2.5.1' `
    'pytest>=7,<10' `
    'transformers==5.14.1'

& $python -c @'
import torch
print(f"torch={torch.__version__}")
print(f"xpu_available={torch.xpu.is_available()}")
print(f"xpu_devices={torch.xpu.device_count()}")
if not torch.xpu.is_available():
    raise SystemExit("XPU is not available in the provisioned environment")
x = torch.arange(8, device="xpu")
torch.xpu.synchronize()
print(x.cpu())
'@

Write-Host "XPU Python: $python"
Write-Host 'Set repository variable INFERREF_XPU_PYTHON to this path.'
