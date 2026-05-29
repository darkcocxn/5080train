@echo off
setlocal
cd /d "%~dp0"

set "PROJECT_ROOT=%CD%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $root=(Resolve-Path -LiteralPath $env:PROJECT_ROOT).Path; $scripts=@(Get-ChildItem -LiteralPath $root -Filter '*2DCNN-*.py'); if($scripts.Count -lt 1){ throw 'Training script not found.' }; $script=$scripts[0]; $python=Join-Path $root '.venv\Scripts\python.exe'; $useVenv=Test-Path -LiteralPath $python; if(-not $useVenv){ $uv=Get-Command uv -ErrorAction SilentlyContinue; if(-not $uv){ throw 'Neither .venv Python nor uv was found. Run setup-env.cmd first.' } }; Write-Host ('Training script: ' + $script.Name); if($env:REMOTE_TRAIN_DRY_RUN -eq '1'){ if($useVenv){ Write-Host ('Would run: ' + $python + ' ' + $script.FullName) } else { Write-Host ('Would run: uv run python ' + $script.FullName) }; exit 0 }; if($useVenv){ & $python $script.FullName } else { & uv run python $script.FullName }; exit $LASTEXITCODE"
exit /b %ERRORLEVEL%
