@echo off
cd /d "%~dp0.."
uv run python 2dcnnv6/2dcnnv6ensemble.py --weight-v2 0.60
pause
