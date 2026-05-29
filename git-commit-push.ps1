[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $ProjectRoot

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    if ($DryRun) {
        Write-Host "DRY RUN: git $($Arguments -join ' ')"
        return
    }

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Arguments -join ' ')"
    }
}

function Get-GitOutput {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    $output = & git @Arguments 2>&1
    if ($LASTEXITCODE -ne 0) {
        return $null
    }
    return @($output)
}

function Test-GitSuccess {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & git @Arguments *> $null
    return ($LASTEXITCODE -eq 0)
}

function Read-MenuChoice {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string[]]$Options
    )

    if ($Options.Count -lt 1) {
        throw "No options were provided for menu: $Title"
    }

    while ($true) {
        Write-Host ""
        Write-Host $Title -ForegroundColor Cyan
        for ($idx = 0; $idx -lt $Options.Count; $idx++) {
            Write-Host ("  {0}. {1}" -f ($idx + 1), $Options[$idx])
        }
        $answer = Read-Host "Enter choice number"
        $choice = 0
        if ([int]::TryParse($answer, [ref]$choice) -and $choice -ge 1 -and $choice -le $Options.Count) {
            return ($choice - 1)
        }
        Write-Host "Invalid choice. Pick a number from the list." -ForegroundColor Yellow
    }
}

function Read-RequiredText {
    param([Parameter(Mandatory = $true)][string]$Prompt)

    while ($true) {
        $value = Read-Host $Prompt
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            return $value.Trim()
        }
        Write-Host "Value is required." -ForegroundColor Yellow
    }
}

function Select-Remote {
    $remotes = @(Get-GitOutput @("remote") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($remotes.Count -lt 1) {
        throw "No git remotes are configured."
    }
    $idx = Read-MenuChoice "Choose remote" $remotes
    return $remotes[$idx]
}

function Get-CurrentBranch {
    $branch = @(Get-GitOutput @("branch", "--show-current"))
    if ($branch.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($branch[0])) {
        return $branch[0].Trim()
    }
    return $null
}

Write-Host "Repository: $ProjectRoot"
Invoke-Git @("rev-parse", "--is-inside-work-tree")

$branch = Get-CurrentBranch
if ($branch) {
    Write-Host "Current branch: $branch"
} else {
    Write-Host "Current checkout is detached; pushing needs an explicit target." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Current status:"
Invoke-Git @("status", "--short", "--branch")

$stageOptions = @(
    "Use already staged changes only",
    "Stage all changes, including untracked files (git add -A)",
    "Stage tracked modifications/deletions only (git add -u)",
    "Enter specific paths to stage",
    "Abort"
)
$stageChoice = Read-MenuChoice "Choose what to stage" $stageOptions
switch ($stageChoice) {
    0 { Write-Host "Keeping current staged set." }
    1 { Invoke-Git @("add", "-A") }
    2 { Invoke-Git @("add", "-u") }
    3 {
        $pathText = Read-RequiredText "Enter paths separated by semicolon (;)"
        $paths = @($pathText -split ";" | ForEach-Object { $_.Trim() } | Where-Object { $_ })
        if ($paths.Count -lt 1) {
            throw "No paths were provided."
        }
        Invoke-Git (@("add", "--") + $paths)
    }
    4 {
        Write-Host "Aborted before staging."
        exit 0
    }
}

if (Test-GitSuccess @("diff", "--cached", "--quiet")) {
    Write-Host ""
    Write-Host "No staged changes are available for commit." -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Staged summary:"
Invoke-Git @("diff", "--cached", "--stat")

while ($true) {
    $reviewChoice = Read-MenuChoice "Choose next action before commit" @(
        "Continue to commit",
        "Show full staged diff",
        "Show current status",
        "Abort"
    )
    if ($reviewChoice -eq 0) { break }
    if ($reviewChoice -eq 1) { Invoke-Git @("diff", "--cached") }
    if ($reviewChoice -eq 2) { Invoke-Git @("status", "--short", "--branch") }
    if ($reviewChoice -eq 3) {
        Write-Host "Aborted before commit."
        exit 0
    }
}

$commitMessage = Read-RequiredText "Commit message"
Invoke-Git @("commit", "-m", $commitMessage)

$upstream = @(Get-GitOutput @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"))
$pushOptions = @()
$pushActions = @()
if ($upstream.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($upstream[0])) {
    $pushOptions += "Push to configured upstream ($($upstream[0].Trim()))"
    $pushActions += "upstream"
}
if ($branch) {
    $pushOptions += "Choose remote, push current branch with the same branch name"
    $pushActions += "same-name"
    $pushOptions += "Choose remote, push current branch and set upstream (-u)"
    $pushActions += "set-upstream"
}
$pushOptions += "Choose remote and enter remote branch name"
$pushActions += "manual-ref"
$pushOptions += "Skip push; keep commit local"
$pushActions += "skip"

$pushChoice = Read-MenuChoice "Choose push target" $pushOptions
$pushAction = $pushActions[$pushChoice]

switch ($pushAction) {
    "upstream" {
        Invoke-Git @("push")
    }
    "same-name" {
        $remote = Select-Remote
        Invoke-Git @("push", $remote, $branch)
    }
    "set-upstream" {
        $remote = Select-Remote
        Invoke-Git @("push", "-u", $remote, $branch)
    }
    "manual-ref" {
        $remote = Select-Remote
        $remoteBranch = Read-RequiredText "Remote branch name"
        Invoke-Git @("push", $remote, "HEAD:$remoteBranch")
    }
    "skip" {
        Write-Host "Commit created locally; push skipped by your choice."
    }
}

Write-Host ""
Write-Host "Done."
