<#
.SYNOPSIS
    Build SageAttention3 (Blackwell) wheel using uv-managed Python environments.

.DESCRIPTION
    Creates a local Python virtual environment per version (reusable across runs),
    installs the correct PyTorch build with CUDA support, and produces a wheel
    into the dist/ folder.

.PARAMETER TorchVersion
    PyTorch version to install, e.g. "2.8.0", "2.9.0".
    If not specified, installs the latest available.

.PARAMETER PythonVersions
    One or more Python version strings, e.g. "3.13", "3.14".
    Defaults to "3.13" if not specified.

.EXAMPLE
    .\build.ps1 2.9.0 3.13
    .\build.ps1 -TorchVersion 2.8.0 -PythonVersions 3.13,3.14
    .\build.ps1                         # latest torch, Python 3.13
#>

param(
    [Parameter(Position = 0)]
    [string]$TorchVersion,

    [Parameter(Position = 1, ValueFromRemainingArguments)]
    [string[]]$PythonVersions = @("3.13")
)

$ErrorActionPreference = "Continue"

$ProjectRoot = $PSScriptRoot
$EnvsDir     = Join-Path $ProjectRoot ".buildenvs"
$DistDir     = Join-Path $ProjectRoot "dist"

# Ensure output directories exist
if (-not (Test-Path $EnvsDir)) { New-Item -ItemType Directory -Path $EnvsDir | Out-Null }
if (-not (Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir | Out-Null }

# ---------- Prerequisites ----------

# Verify uv is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed or not on PATH. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

# Activate Visual Studio developer environment if cl.exe is not already on PATH
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vsWhere)) {
        Write-Error "Cannot find vswhere.exe. Install Visual Studio Build Tools."
        exit 1
    }
    $vsInstallDir = & $vsWhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $vsInstallDir) {
        Write-Error "Cannot find Visual Studio installation with C++ tools."
        exit 1
    }
    $vcvarsall = Join-Path $vsInstallDir "VC\Auxiliary\Build\vcvarsall.bat"
    if (-not (Test-Path $vcvarsall)) {
        Write-Error "Cannot find vcvarsall.bat at $vcvarsall"
        exit 1
    }
    Write-Host "Activating Visual Studio environment ..." -ForegroundColor Yellow
    $envVars = cmd /c "`"$vcvarsall`" amd64 && set" 2>&1
    foreach ($line in $envVars) {
        if ($line -match "^([^=]+)=(.*)$") {
            [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], "Process")
        }
    }
    if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
        Write-Error "Failed to activate Visual Studio environment — cl.exe still not on PATH."
        exit 1
    }
    Write-Host "Visual Studio environment activated. cl.exe: $(Get-Command cl | Select-Object -ExpandProperty Source)" -ForegroundColor Green
}

# Ensure CUDA is discoverable
if (-not $env:CUDA_HOME) {
    if ($env:CUDA_PATH) {
        $env:CUDA_HOME = $env:CUDA_PATH
    } else {
        $cudaDir = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" `
            -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | Select-Object -First 1
        if ($cudaDir) {
            $env:CUDA_HOME = $cudaDir.FullName
            $env:CUDA_PATH = $cudaDir.FullName
        } else {
            Write-Error "Cannot find CUDA installation. Set CUDA_HOME or CUDA_PATH."
            exit 1
        }
    }
}
Write-Host "Using CUDA_HOME: $env:CUDA_HOME" -ForegroundColor Magenta

# Verify CUDA >= 12.8
$nvccVer = & "$env:CUDA_HOME\bin\nvcc.exe" --version 2>&1 | Select-String "release (\d+)\.(\d+)"
if ($nvccVer) {
    $cudaMajor = [int]$nvccVer.Matches[0].Groups[1].Value
    $cudaMinor = [int]$nvccVer.Matches[0].Groups[2].Value
    if ($cudaMajor -lt 12 -or ($cudaMajor -eq 12 -and $cudaMinor -lt 8)) {
        Write-Error "SageAttention3 requires CUDA >= 12.8. Found: $cudaMajor.$cudaMinor"
        exit 1
    }
    $torchCudaTag = "cu${cudaMajor}${cudaMinor}"
} else {
    Write-Error "Failed to detect CUDA version from nvcc."
    exit 1
}
Write-Host "CUDA version: $cudaMajor.$cudaMinor (index tag: $torchCudaTag)" -ForegroundColor Magenta

$torchIndexUrl = "https://download.pytorch.org/whl/$torchCudaTag"

# Tell setuptools we already activated the VC environment
$env:DISTUTILS_USE_SDK = "1"

# Limit parallelism to avoid OOM during CUDA compilation
$env:MAX_JOBS = "2"

# ---------- Build loop ----------

foreach ($PyVer in $PythonVersions) {
    $envPath = Join-Path $EnvsDir "py$PyVer"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " Building sageattn3 wheel for Python $PyVer" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Create venv if it doesn't already exist (reusable across runs)
    $pythonExe = Join-Path $envPath "Scripts\python.exe"
    if (-not (Test-Path $pythonExe)) {
        Write-Host "[1/4] Creating virtual environment at $envPath ..." -ForegroundColor Yellow
        uv venv --python $PyVer $envPath
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create venv for Python $PyVer"
            continue
        }
    } else {
        Write-Host "[1/4] Reusing existing virtual environment at $envPath" -ForegroundColor Green
    }

    # Install / update build dependencies
    Write-Host "[2/4] Installing build dependencies ..." -ForegroundColor Yellow
    uv pip install --python $pythonExe setuptools wheel packaging numpy ninja einops
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install base dependencies for Python $PyVer"
        continue
    }

    if ($TorchVersion) {
        uv pip install --python $pythonExe "torch==$TorchVersion" --index-url $torchIndexUrl
    } else {
        uv pip install --python $pythonExe torch --index-url $torchIndexUrl
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to install torch for Python $PyVer"
        continue
    }

    # Read back installed torch version for wheel naming
    $installedTorch = & $pythonExe -c "import torch; print(torch.__version__)" 2>&1
    $torchTag = ($installedTorch -replace '\+.*$','') -replace '\.',''
    Write-Host "  Installed PyTorch: $installedTorch (tag: torch$torchTag)" -ForegroundColor Magenta

    # Build the wheel
    Write-Host "[3/4] Building wheel ..." -ForegroundColor Yellow

    # Clean previous build artifacts to avoid stale objects
    $buildDir = Join-Path $ProjectRoot "build"
    if (Test-Path $buildDir) {
        Write-Host "  Cleaning previous build directory ..." -ForegroundColor Yellow
        Remove-Item -Recurse -Force $buildDir
    }

    Push-Location $ProjectRoot
    try {
        & $pythonExe setup.py bdist_wheel --dist-dir $DistDir 2>&1 | Tee-Object -FilePath (Join-Path $ProjectRoot "build_log_py${PyVer}.txt")
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Wheel build failed for Python $PyVer. See build_log_py${PyVer}.txt"
            continue
        }
    } finally {
        Pop-Location
    }

    # Rename wheel to include torch version
    # e.g. sageattn3-1.0.0-cp313-cp313-win_amd64.whl
    #   -> sageattn3-1.0.0+torch290-cp313-cp313-win_amd64.whl
    $cpTag = "cp$($PyVer -replace '\.','')"
    $builtWheel = Get-ChildItem $DistDir -Filter "sageattn3-*-${cpTag}-${cpTag}-*.whl" |
        Where-Object { $_.Name -notmatch '\+torch' } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($builtWheel) {
        $newName = $builtWheel.Name -replace '(sageattn3-[^-]+)(-)', "`$1+torch${torchTag}`$2"
        $newPath = Join-Path $DistDir $newName
        if ($newName -ne $builtWheel.Name) {
            Move-Item $builtWheel.FullName $newPath -Force
            Write-Host "  Renamed: $($builtWheel.Name) -> $newName" -ForegroundColor Yellow
        }
    }

    Write-Host "[4/4] Done for Python $PyVer" -ForegroundColor Green
}

# ---------- Summary ----------

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " All wheels in: $DistDir" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Get-ChildItem $DistDir -Filter "sageattn3-*.whl" | ForEach-Object { Write-Host "  $_" }
