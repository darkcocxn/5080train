param(
    [string]$LogPath = ""
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($LogPath)) {
    $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $LogPath = "C:\Users\jiaotong1\5080train\history\v2-to-v9-watcher-$stamp.log"
}

function Write-Log {
    param([string]$Message)
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
    Add-Content -Path $LogPath -Value $line
}

function Get-TrainingProcess {
    param([string]$ScriptName)
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -like "*$ScriptName*" -and $_.Name -match "powershell|python|uv"
    }
}

Write-Log "Watcher started."

while ($true) {
    $v2Processes = Get-TrainingProcess -ScriptName "2dcnnv2/2dcnnv2.py"
    if ($v2Processes) {
        Start-Sleep -Seconds 60
        continue
    }

    Write-Log "No active V2 process found."

    $v9Processes = Get-TrainingProcess -ScriptName "2dcnnv9/2dcnnv9.py"
    if ($v9Processes) {
        Write-Log "V9 is already running. Exiting watcher."
        break
    }

    $runStamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $stdout = "C:\Users\jiaotong1\5080train\history\v9-newdata-$runStamp.out.log"
    $stderr = "C:\Users\jiaotong1\5080train\history\v9-newdata-$runStamp.err.log"
    $envCmd = @'
$env:SURMOD_NEW_DATA_DIR = 'C:\Users\jiaotong1\5080train\newdata'
$env:SURMOD_WAVELET_IMAGE_DIR = 'C:\Users\jiaotong1\5080train\newdata\Newdatabase\Scalogram\rare_waves-20260531-140353'
uv run python 2dcnnv9/2dcnnv9.py
'@

    Write-Log "Launching V9. STDOUT=$stdout STDERR=$stderr"
    $proc = Start-Process -FilePath powershell.exe `
        -ArgumentList @("-NoProfile", "-Command", $envCmd) `
        -WorkingDirectory "C:\Users\jiaotong1\5080train" `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden
    Write-Log "Started V9 launcher PID=$($proc.Id)"
    break
}

Write-Log "Watcher exiting."
