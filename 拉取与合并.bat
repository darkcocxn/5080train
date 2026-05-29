@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0git-pull-merge.ps1" %*
exit /b %ERRORLEVEL%
