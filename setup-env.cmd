@echo off
setlocal

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup-env.ps1" %*
exit /b %ERRORLEVEL%
