param(
    [string]$Python = "",
    [string]$BuildDirectory = "cpp/build-sycl-icx",
    [string]$EvidenceDirectory = ".scratch/xpu-sycl-gate"
)

$ErrorActionPreference = "Stop"
$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Python) {
    $Python = Join-Path $Repository ".venv/Scripts/python.exe"
}
$Python = (Resolve-Path $Python).Path
$BuildDirectory = [IO.Path]::GetFullPath((Join-Path $Repository $BuildDirectory))
$EvidenceDirectory = [IO.Path]::GetFullPath((Join-Path $Repository $EvidenceDirectory))

$VsWhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
if (-not (Test-Path -LiteralPath $VsWhere -PathType Leaf)) {
    throw "vswhere.exe was not found"
}
$VsRoot = (& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath).Trim()
$VsDevCmd = Join-Path $VsRoot "Common7/Tools/VsDevCmd.bat"
$OneApiSetVars = Join-Path ${env:ProgramFiles(x86)} "Intel/oneAPI/setvars.bat"
if (-not (Test-Path -LiteralPath $VsDevCmd -PathType Leaf) -or -not (Test-Path -LiteralPath $OneApiSetVars -PathType Leaf)) {
    throw "Visual Studio C++ or oneAPI setvars.bat was not found"
}

# Import the toolchain environment into this PowerShell process as well as the
# build subprocess, so the native engine can resolve the SYCL runtime DLLs.
$EnvironmentCommand = "call `"$VsDevCmd`" -arch=amd64 >nul && call `"$OneApiSetVars`" intel64 --force >nul && set"
foreach ($Line in (& cmd.exe /d /s /c $EnvironmentCommand)) {
    if ($Line -match '^([^=]+)=(.*)$') {
        Set-Item -Path "Env:$($Matches[1])" -Value $Matches[2]
    }
}
$env:ONEAPI_DEVICE_SELECTOR = "level_zero:gpu"

New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null
$SuiteRuns = Join-Path $EvidenceDirectory "suite-runs"
$Engine = Join-Path $BuildDirectory "inferref_sycl_engine.exe"
$AdapterTemplate = Join-Path $Repository "examples/sycl_engine/inferref_sycl.adapter.json"
$Adapter = Join-Path $EvidenceDirectory "effective-sycl.adapter.json"
$AdapterDocument = Get-Content -LiteralPath $AdapterTemplate -Raw | ConvertFrom-Json
$AdapterDocument.command[0] = $Engine
$AdapterDocument | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $Adapter -Encoding utf8
$Suite = Join-Path $Repository "corpus/xpu-v0.1/suite.json"
$BuildCommand = @(
    "call `"$VsDevCmd`" -arch=amd64",
    "call `"$OneApiSetVars`" intel64 --force",
    "cmake -S `"$Repository/cpp`" -B `"$BuildDirectory`" -G Ninja -DINFERREF_BUILD_SYCL=ON -DCMAKE_CXX_COMPILER=icx-cl.exe -DCMAKE_BUILD_TYPE=Release",
    "cmake --build `"$BuildDirectory`" --target inferref_sycl_engine inferref_compare",
    "`"$Python`" -m inferref.cli.main suite run `"$Suite`" --adapter `"$Adapter`" --runs-dir `"$SuiteRuns`" --json > `"$EvidenceDirectory/suite-run.json`"",
    "`"$Python`" -m inferref.cli.main suite report `"$EvidenceDirectory/suite-run.json`" --output `"$EvidenceDirectory/report.html`""
) -join " && "
& cmd.exe /d /s /c $BuildCommand
if ($LASTEXITCODE -ne 0) { throw "SYCL build or positive suite gate failed ($LASTEXITCODE)" }

$Comparator = Join-Path $BuildDirectory "inferref_compare.exe"
$SuiteReport = Get-Content -LiteralPath (Join-Path $EvidenceDirectory "suite-run.json") -Raw | ConvertFrom-Json
$DeviceEvidence = @()
foreach ($CaseResult in $SuiteReport.cases) {
    $Run = $CaseResult.results[0].run
    $Device = Get-Content -LiteralPath (Join-Path $Run.output "inferref-sycl-device.json") -Raw | ConvertFrom-Json
    if ($Device.device_type -ne "gpu" -or $Device.backend -ne "ext_oneapi_level_zero" -or $Device.vendor -notmatch "Intel") {
        throw "case $($CaseResult.id) did not execute on an Intel Level Zero GPU"
    }
    $DeviceEvidence += [PSCustomObject]@{ case = $CaseResult.id; device = $Device }
    $Manifest = Get-Content -LiteralPath (Join-Path $Run.testcase "testcase.json") -Raw | ConvertFrom-Json
    foreach ($Output in $Manifest.outputs) {
        & $Comparator (Join-Path $Run.testcase $Output.payload) (Join-Path $Run.output "$($Output.name).irtensor") | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "C++ comparator disagreed with Python PASS for $($CaseResult.id)/$($Output.name)"
        }
    }
}
$DeviceEvidence | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $EvidenceDirectory "device-evidence.json") -Encoding utf8
$NegativeCases = @(
    @{ Name = "rmsnorm"; Case = "rmsnorm-float32"; Output = "y" },
    @{ Name = "rope"; Case = "rope-dim4"; Output = "q_embed" },
    @{ Name = "kv-cache"; Case = "kv-index"; Output = "cache_out" }
)
foreach ($Item in $NegativeCases) {
    $CaseDirectory = Join-Path $Repository "corpus/xpu-v0.1/cases/$($Item.Case)"
    $NegativeOutput = Join-Path $EvidenceDirectory "negative-$($Item.Name)"
    New-Item -ItemType Directory -Force -Path $NegativeOutput | Out-Null
    & $Engine --testcase $CaseDirectory --output $NegativeOutput --inject-error *> (Join-Path $NegativeOutput "engine.log")
    if ($LASTEXITCODE -ne 0) { throw "$($Item.Name) negative engine run failed to execute" }
    & $Comparator (Join-Path $CaseDirectory "reference/$($Item.Output).irtensor") (Join-Path $NegativeOutput "$($Item.Output).irtensor") *> (Join-Path $NegativeOutput "cpp-compare.txt")
    if ($LASTEXITCODE -eq 0) { throw "$($Item.Name) injected error was not rejected by the C++ comparator" }
    & $Python -m inferref.cli.main compare $CaseDirectory $NegativeOutput --json *> (Join-Path $NegativeOutput "python-compare.json")
    if ($LASTEXITCODE -eq 0) { throw "$($Item.Name) injected error was not rejected by the Python comparator" }
}

# The executable is native: its dependency table must not contain Python or PyTorch.
$Dependencies = (& dumpbin /dependents $Engine) -join "`n"
if ($Dependencies -match "(?i)(python|torch)") { throw "native SYCL engine unexpectedly depends on Python/PyTorch" }
$Dependencies | Set-Content -LiteralPath (Join-Path $EvidenceDirectory "native-dependencies.txt") -Encoding utf8
Write-Host "SYCL gate PASS: $($SuiteReport.counts.pass) positive cells and 3 injected-error cases"
