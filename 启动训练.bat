@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $root=(Resolve-Path -LiteralPath '%~dp0').Path; $script=Get-ChildItem -LiteralPath $root -Filter '*2DCNN*.py' | Where-Object { $_.Name -notlike '2DCNN*' -and (($_.BaseName.ToCharArray() | Where-Object { $_ -eq '-' }).Count -ge 4) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1; if(-not $script){ throw 'Strong scalar fusion training script not found.' }; $python=Join-Path $root '.venv\Scripts\python.exe'; if(Test-Path -LiteralPath $python){ & $python $script.FullName } else { & uv run python $script.FullName }; exit $LASTEXITCODE"
exit /b %ERRORLEVEL%
