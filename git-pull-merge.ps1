[CmdletBinding()]
param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSCommandPath
Set-Location -LiteralPath $ProjectRoot

$script:CreatedStash = $false

function Invoke-Git {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    if ($DryRun) {
        Write-Host "演练模式，不执行: git $($Arguments -join ' ')"
        return
    }

    & git @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git 命令失败: git $($Arguments -join ' ')"
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
        throw "菜单没有可选项: $Title"
    }

    while ($true) {
        Write-Host ""
        Write-Host $Title -ForegroundColor Cyan
        for ($idx = 0; $idx -lt $Options.Count; $idx++) {
            Write-Host ("  {0}. {1}" -f ($idx + 1), $Options[$idx])
        }
        $answer = Read-Host "请输入选项编号"
        $choice = 0
        if ([int]::TryParse($answer, [ref]$choice) -and $choice -ge 1 -and $choice -le $Options.Count) {
            return ($choice - 1)
        }
        Write-Host "无效选择，请输入清单中的编号。" -ForegroundColor Yellow
    }
}

function Read-RequiredText {
    param([Parameter(Mandatory = $true)][string]$Prompt)

    while ($true) {
        $value = Read-Host $Prompt
        if (-not [string]::IsNullOrWhiteSpace($value)) {
            return $value.Trim()
        }
        Write-Host "这里必须输入内容。" -ForegroundColor Yellow
    }
}

function Select-Remote {
    $remotes = @(Get-GitOutput @("remote") | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
    if ($remotes.Count -lt 1) {
        throw "当前仓库没有配置任何远端 remote。"
    }
    $idx = Read-MenuChoice "选择要从哪个远端 remote 获取更新" $remotes
    return $remotes[$idx]
}

function Get-CurrentBranch {
    $branch = @(Get-GitOutput @("branch", "--show-current"))
    if ($branch.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($branch[0])) {
        return $branch[0].Trim()
    }
    return $null
}

function Push-NewStash {
    param([switch]$IncludeUntracked)

    $before = @(Get-GitOutput @("stash", "list"))
    $stashArgs = @("stash", "push", "-m", "拉取合并前自动临时储藏")
    if ($IncludeUntracked) {
        $stashArgs = @("stash", "push", "--include-untracked", "-m", "拉取合并前自动临时储藏")
    }

    Invoke-Git $stashArgs
    if ($DryRun) {
        return $false
    }

    $after = @(Get-GitOutput @("stash", "list"))
    if ($after.Count -gt $before.Count) {
        return $true
    }
    if ($after.Count -gt 0 -and ($before.Count -eq 0 -or $after[0] -ne $before[0])) {
        return $true
    }

    Write-Host "Git 没有创建新的临时储藏 stash。" -ForegroundColor Yellow
    return $false
}

Write-Host "仓库路径: $ProjectRoot"
Invoke-Git @("rev-parse", "--is-inside-work-tree")

$branch = Get-CurrentBranch
if ($branch) {
    Write-Host "当前分支: $branch"
} else {
    Write-Host "当前不是普通分支，而是 detached HEAD 状态。" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "当前状态:"
Invoke-Git @("status", "--short", "--branch")

$isDirty = -not ((Test-GitSuccess @("diff", "--quiet")) -and (Test-GitSuccess @("diff", "--cached", "--quiet")))
$untracked = @(Get-GitOutput @("ls-files", "--others", "--exclude-standard") | Where-Object { $_ })
if ($isDirty -or $untracked.Count -gt 0) {
    $dirtyChoice = Read-MenuChoice "检测到本地有未提交改动，请选择如何处理" @(
        "继续，不处理本地改动：保留当前文件；如果和远端改动冲突，后续合并可能失败",
        "只临时储藏已跟踪文件：收起已被 Git 跟踪的修改/删除；不包含新增未跟踪文件",
        "临时储藏已跟踪和未跟踪文件：连新增文件一起收起，合并后可选择恢复",
        "中止：不执行 fetch 或 merge"
    )
    switch ($dirtyChoice) {
        0 { Write-Host "继续保留当前工作区改动。" }
        1 {
            $script:CreatedStash = Push-NewStash
        }
        2 {
            $script:CreatedStash = Push-NewStash -IncludeUntracked
        }
        3 {
            Write-Host "已中止：尚未执行 fetch。"
            exit 0
        }
    }
}

$remote = Select-Remote
$fetchChoice = Read-MenuChoice "选择获取远端信息的方式" @(
    "获取远端最新信息：执行 git fetch；只更新远端跟踪分支，不直接修改当前文件",
    "获取并清理远端已删除分支：执行 git fetch --prune；会删除本地过期的远端跟踪分支",
    "跳过获取：不联网，只使用本地已有的远端跟踪分支信息",
    "中止：不执行 merge"
)
switch ($fetchChoice) {
    0 { Invoke-Git @("fetch", $remote) }
    1 { Invoke-Git @("fetch", "--prune", $remote) }
    2 { Write-Host "已按选择跳过 fetch。" }
    3 {
        Write-Host "已中止：尚未执行 merge。"
        exit 0
    }
}

$upstream = @(Get-GitOutput @("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"))
$sourceOptions = @()
$sourceRefs = @()
if ($upstream.Count -gt 0 -and -not [string]::IsNullOrWhiteSpace($upstream[0])) {
    $sourceOptions += "使用当前分支配置的上游分支 ($($upstream[0].Trim()))：通常是最常用的远端对应分支"
    $sourceRefs += $upstream[0].Trim()
}

$remoteBranches = @(
    Get-GitOutput @("for-each-ref", "--format=%(refname:short)", "refs/remotes/$remote") |
        Where-Object { $_ -and ($_ -notmatch "/HEAD$") } |
        Sort-Object
)
foreach ($remoteBranch in $remoteBranches) {
    $sourceOptions += "合并远端分支 $remoteBranch：把该远端分支的提交合入当前分支"
    $sourceRefs += $remoteBranch
}
$sourceOptions += "手动输入来源引用：例如 5080train/master，适合列表里没有目标分支时使用"
$sourceRefs += "__manual__"
$sourceOptions += "中止：不执行 merge"
$sourceRefs += "__abort__"

$sourceChoice = Read-MenuChoice "选择要合并到当前分支的来源" $sourceOptions
$sourceRef = $sourceRefs[$sourceChoice]
if ($sourceRef -eq "__abort__") {
    Write-Host "已中止：尚未执行 merge。"
    exit 0
}
if ($sourceRef -eq "__manual__") {
    $sourceRef = Read-RequiredText "请输入来源引用，例如 5080train/master"
}

$mergeChoice = Read-MenuChoice "选择合并方式" @(
    "仅允许快进合并：执行 git merge --ff-only；本地没有额外提交时才会成功，历史最干净",
    "普通合并：执行 git merge；能快进就快进，否则创建一次合并提交",
    "尽量创建合并提交：执行 git merge --no-ff；用于明确保留一次合并记录",
    "压缩合并：执行 git merge --squash；只把改动放进工作区/暂存区，不自动提交",
    "中止：不执行 merge"
)

try {
    switch ($mergeChoice) {
        0 { Invoke-Git @("merge", "--ff-only", $sourceRef) }
        1 { Invoke-Git @("merge", $sourceRef) }
        2 { Invoke-Git @("merge", "--no-ff", $sourceRef) }
        3 { Invoke-Git @("merge", "--squash", $sourceRef) }
        4 {
            Write-Host "已中止：尚未执行 merge。"
            exit 0
        }
    }
} catch {
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Write-Host "合并没有干净完成。请查看状态并手动处理冲突。" -ForegroundColor Yellow
    & git status --short --branch
    exit 1
}

if ($script:CreatedStash) {
    $stashChoice = Read-MenuChoice "本脚本刚才创建了临时储藏 stash，请选择下一步" @(
        "立即恢复这个 stash：执行 git stash pop；如果恢复时冲突，需要手动解决",
        "保留 stash 以后再恢复：稍后可用 git stash list / git stash pop 手动处理"
    )
    if ($stashChoice -eq 0) {
        Invoke-Git @("stash", "pop")
    } else {
        Write-Host "已保留 stash。需要时使用 git stash list / git stash pop。"
    }
}

Write-Host ""
Write-Host "最终状态:"
Invoke-Git @("status", "--short", "--branch")
Write-Host ""
Write-Host "完成。"
