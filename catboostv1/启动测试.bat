@echo off
cd /d "%~dp0.."
uv run python catboostv1/catboostv1test.py
pause
