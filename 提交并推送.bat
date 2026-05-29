@echo off
setlocal
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0git-commit-push.ps1"
exit /b %ERRORLEVEL%
