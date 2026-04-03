<#
.SYNOPSIS
    Build SageAttention wheels for one or more Python versions using uv.

.DESCRIPTION
    Creates a local Python virtual environment per version (reusable across runs),
    installs build dependencies, and produces wheels into a shared dist/ folder.

.PARAMETER PythonVersions
    One or more Python version strings, e.g. "3.11", "3.12", "3.13".
    Defaults to "3.11" if not specified.

.PARAMETER TorchVersion
    PyTorch version to install, e.g. "2.9.0", "2.11.0".
    If not specified, installs the latest available.

.EXAMPLE
    .\build.ps1 2.9.0 3.10 3.11
#>

param(
    [Parameter(Position = 0)]
    [string]$TorchVersion,

    [Parameter(Position = 1, ValueFromRemainingArguments)]
    [string[]]$PythonVersions = @("3.11")
)

$ErrorActionPreference = "Continue"

$ProjectRoot = $PSScriptRoot
$EnvsDir     = Join-Path $ProjectRoot ".buildenvs"
$DistDir     = Join-Path $ProjectRoot "dist"

# Ensure output directories exist
if (-not (Test-Path $EnvsDir)) { New-Item -ItemType Directory -Path $EnvsDir | Out-Null }
if (-not (Test-Path $DistDir))  { New-Item -ItemType Directory -Path $DistDir  | Out-Null }

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
    $vsInstallDir = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
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
    # Run vcvarsall.bat and capture the resulting environment variables
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

# Ensure CUDA is discoverable by torch
if (-not $env:CUDA_HOME) {
    if ($env:CUDA_PATH) {
        $env:CUDA_HOME = $env:CUDA_PATH
    } else {
        # Auto-detect from default install location
        $cudaDir = Get-ChildItem "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA" -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
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

# Auto-detect supported architectures from nvcc
$nvccExe = Join-Path $env:CUDA_HOME "bin\nvcc.exe"
$supportedArchs = & $nvccExe --list-gpu-arch 2>$null | ForEach-Object {
    if ($_ -match "compute_(\d+)") { $Matches[1] }
}
# Map from SageAttention's SUPPORTED_ARCHS format to compute capabilities
# Only include architectures that nvcc actually supports
$wantedArchs = @("8.0", "8.6", "8.9", "9.0", "10.0", "12.0", "12.1")
$archList = @()
foreach ($arch in $wantedArchs) {
    $computeNum = $arch -replace "\.", ""
    if ($supportedArchs -contains $computeNum) {
        $archList += $arch
    }
}
if ($archList.Count -eq 0) {
    Write-Error "No supported architectures found for nvcc at $nvccExe"
    exit 1
}
$env:TORCH_CUDA_ARCH_LIST = $archList -join ";"
Write-Host "Target architectures: $env:TORCH_CUDA_ARCH_LIST" -ForegroundColor Magenta

# Tell setuptools/distutils we already activated the VC environment
$env:DISTUTILS_USE_SDK = "1"

# Limit parallelism to avoid out-of-memory during CUDA compilation
# Each .cu file compiles for multiple architectures with --threads=8 internally
$env:MAX_JOBS = "2"

foreach ($PyVer in $PythonVersions) {
    $envPath = Join-Path $EnvsDir "py$PyVer"
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host " Building wheel for Python $PyVer" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    # Create venv if it doesn't already exist (reusable across runs)
    if (-not (Test-Path (Join-Path $envPath "Scripts\python.exe"))) {
        Write-Host "[1/4] Creating virtual environment at $envPath ..." -ForegroundColor Yellow
        uv venv --python $PyVer $envPath
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Failed to create venv for Python $PyVer"
            continue
        }
    } else {
        Write-Host "[1/4] Reusing existing virtual environment at $envPath" -ForegroundColor Green
    }

    $pythonExe = Join-Path $envPath "Scripts\python.exe"

    # Determine PyTorch CUDA index URL from installed CUDA version
    $nvccVer = & "$env:CUDA_HOME\bin\nvcc.exe" --version 2>&1 | Select-String "release (\d+)\.(\d+)"
    if ($nvccVer) {
        $cudaMajor = $nvccVer.Matches[0].Groups[1].Value
        $cudaMinor = $nvccVer.Matches[0].Groups[2].Value
        $torchCudaTag = "cu${cudaMajor}${cudaMinor}"
    } else {
        $torchCudaTag = "cu128"
    }
    $torchIndexUrl = "https://download.pytorch.org/whl/$torchCudaTag"
    Write-Host "  Using PyTorch index: $torchIndexUrl" -ForegroundColor Magenta

    # Install / update build dependencies (torch with CUDA support)
    # Pin setuptools/wheel/packaging to match pyproject.toml build-system requires
    Write-Host "[2/4] Installing build dependencies ..." -ForegroundColor Yellow
    uv pip install --python $pythonExe "setuptools>=62,<75" "wheel>=0.38,<0.44" "packaging>=21,<24" numpy
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

    # Read back the actual installed torch version for wheel naming
    $installedTorch = & $pythonExe -c "import torch; print(torch.__version__)" 2>&1
    $torchTag = ($installedTorch -replace '\+.*$','') -replace '\.',''
    Write-Host "  Installed PyTorch: $installedTorch (tag: torch$torchTag)" -ForegroundColor Magenta

    # Build the wheel
    Write-Host "[3/4] Building wheel ..." -ForegroundColor Yellow
    Push-Location $ProjectRoot
    try {
        & $pythonExe setup.py bdist_wheel --dist-dir $DistDir
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Wheel build failed for Python $PyVer"
            continue
        }
    } finally {
        Pop-Location
    }

    # Rename wheel to include torch version infix
    # e.g. sageattention-2.2.0-cp310-cp310-win_amd64.whl
    #   -> sageattention-2.2.0+torch290-cp310-cp310-win_amd64.whl
    $cpTag = "cp$($PyVer -replace '\.','')"
    $builtWheel = Get-ChildItem $DistDir -Filter "sageattention-*-${cpTag}-${cpTag}-*.whl" |
        Where-Object { $_.Name -notmatch '\+torch' } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($builtWheel) {
        $newName = $builtWheel.Name -replace '(sageattention-[^-]+)(-)', "`$1+torch${torchTag}`$2"
        $newPath = Join-Path $DistDir $newName
        if ($newName -ne $builtWheel.Name) {
            Move-Item $builtWheel.FullName $newPath -Force
            Write-Host "  Renamed: $($builtWheel.Name) -> $newName" -ForegroundColor Yellow
        }
    }

    Write-Host "[4/4] Done for Python $PyVer" -ForegroundColor Green
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host " All wheels accumulated in: $DistDir" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Get-ChildItem $DistDir -Filter *.whl | ForEach-Object { Write-Host "  $_" }
