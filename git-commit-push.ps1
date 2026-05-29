[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $ProjectRoot

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Arguments -join ' ')"
    }
}

Write-Host "Repository: $ProjectRoot"
Invoke-Git @("rev-parse", "--is-inside-work-tree")

Write-Host "Running: git push"
Invoke-Git @("push")

Write-Host ""
Invoke-Git @("status", "--short", "--branch")
