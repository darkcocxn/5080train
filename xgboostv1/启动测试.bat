@echo off
cd /d "%~dp0.."
uv run python xgboostv1/xgboostv1test.py
pause
