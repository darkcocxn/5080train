[CmdletBinding()]
param(
    [string]$Python = "3.13",
    [switch]$SkipUvInstall,
    [switch]$SkipPythonInstall,
    [switch]$AllowLockUpdate
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $ProjectRoot

function Write-Step {
    param([string]$Message)

    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Get-UvPath {
    $uvCommand = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCommand) {
        return $uvCommand.Source
    }

    $candidates = @()
    if ($env:USERPROFILE) {
        $candidates += Join-Path $env:USERPROFILE ".local\bin\uv.exe"
        $candidates += Join-Path $env:USERPROFILE ".cargo\bin\uv.exe"
    }
    if ($env:LOCALAPPDATA) {
        $candidates += Join-Path $env:LOCALAPPDATA "uv\uv.exe"
        $candidates += Join-Path $env:LOCALAPPDATA "Programs\uv\uv.exe"
    }

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return $null
}

function Add-DirectoryToProcessPath {
    param([string]$Directory)

    if (-not $Directory) {
        return
    }

    $pathParts = $env:Path -split [IO.Path]::PathSeparator
    if ($pathParts -notcontains $Directory) {
        $env:Path = "$Directory$([IO.Path]::PathSeparator)$env:Path"
    }
}

function Invoke-Checked {
    param(
        [string]$FilePath,
        [string[]]$Arguments
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $FilePath $($Arguments -join ' ')"
    }
}

if (-not (Test-Path -LiteralPath "pyproject.toml")) {
    throw "pyproject.toml was not found. Run this script from the project checkout."
}

if (-not (Test-Path -LiteralPath "uv.lock")) {
    Write-Warning "uv.lock was not found. The sync step may create or update the lock file."
}

$uvPath = Get-UvPath
if (-not $uvPath) {
    if ($SkipUvInstall) {
        throw "uv is not installed or not on PATH. Re-run without -SkipUvInstall, or install uv first."
    }

    Write-Step "Installing uv"
    Invoke-Expression (Invoke-RestMethod "https://astral.sh/uv/install.ps1")
    $uvPath = Get-UvPath
}

if (-not $uvPath) {
    throw "uv installer finished, but uv.exe was not found. Restart the terminal and run this script again."
}

Add-DirectoryToProcessPath (Split-Path -Parent $uvPath)

Write-Step "Using uv"
Invoke-Checked $uvPath @("--version")

if (-not $SkipPythonInstall) {
    Write-Step "Ensuring Python $Python is available"
    Invoke-Checked $uvPath @("python", "install", $Python)
}

$syncArgs = @("sync", "--no-install-project", "--python", $Python)
if (-not $AllowLockUpdate) {
    $syncArgs += "--locked"
}

Write-Step "Syncing project environment"
Invoke-Checked $uvPath $syncArgs

Write-Step "Verifying Python packages"
Invoke-Checked $uvPath @(
    "run",
    "python",
    "-c",
    "import sys; print('python', sys.version.split()[0]); print('executable', sys.executable); import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available())"
)

Write-Host ""
Write-Host "Environment is ready."
Write-Host "Activate it with: .\.venv\Scripts\Activate.ps1"
Write-Host "Or run scripts through uv, for example: uv run python <script.py>"
