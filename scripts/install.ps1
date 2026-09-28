param([string]$Conda = 'conda', [switch]$SkipModels)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimePath = Join-Path $projectRoot '.envs/runtime'
$runtimePython = Join-Path $runtimePath 'python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) {
    & $Conda create --yes --prefix $runtimePath python=3.13 pip
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
}
& $runtimePython -m pip install -r (Join-Path $projectRoot 'requirements/cuda128.txt')
if ($LASTEXITCODE -ne 0) { throw 'CUDA Python dependency installation failed' }
& $runtimePython -m pip install --no-build-isolation -r (Join-Path $projectRoot 'requirements/windows-py313.lock.txt')
if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
& $runtimePython -m pip install --no-deps -e (Join-Path $projectRoot 'third_party/ml-sharp') -e (Join-Path $projectRoot 'third_party/DepthMaster/third_party/utils3d')
if ($LASTEXITCODE -ne 0) { throw 'Local source registration failed' }
& $runtimePython (Join-Path $PSScriptRoot 'patch_runtime.py')
if ($LASTEXITCODE -ne 0) { throw 'Windows compiler compatibility patch failed' }
Push-Location (Join-Path $projectRoot 'pano-web')
try { npm.cmd ci; if ($LASTEXITCODE -ne 0) { throw 'Frontend dependency installation failed' } }
finally { Pop-Location }
if (-not $SkipModels) {
    & $runtimePython (Join-Path $PSScriptRoot 'download_models.py')
    if ($LASTEXITCODE -ne 0) { throw 'Model download or verification failed' }
}
Push-Location $projectRoot
try {
    $checkArguments = @('scripts/check_install.py', '--gpu', '--report', '.runtime/installation-check.json')
    if ($SkipModels) { $checkArguments += '--skip-models' }
    & (Join-Path $projectRoot 'run_cuda_python.cmd') @checkArguments
    if ($LASTEXITCODE -ne 0) { throw 'Installation check failed' }
}
finally { Pop-Location }
