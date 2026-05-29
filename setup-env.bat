@echo off
setlocal

cd /d "%~dp0"

if not exist "pyproject.toml" (
    echo pyproject.toml not found. Run this script from the project root.
    pause
    exit /b 1
)

echo.
echo ==> Checking uv...
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ==> Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    where uv >nul 2>&1
    if %ERRORLEVEL% neq 0 (
        echo uv install failed. Restart terminal and retry.
        pause
        exit /b 1
    )
)
uv --version

echo.
echo ==> Ensuring Python 3.13...
uv python install 3.13
if %ERRORLEVEL% neq 0 (
    echo Python install failed.
    pause
    exit /b 1
)

echo.
echo ==> Syncing project environment...
uv sync --no-install-project --python 3.13 --locked
if %ERRORLEVEL% neq 0 (
    echo Sync failed, retrying without --locked...
    uv sync --no-install-project --python 3.13
    if %ERRORLEVEL% neq 0 (
        echo Sync failed.
        pause
        exit /b 1
    )
)

echo.
echo ==> Verifying Python packages...
uv run python -c "import sys; print('python', sys.version.split()[0]); import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available())"

echo.
echo Environment is ready.
echo Activate: .venv\Scripts\activate.bat
echo Or run:   uv run python ^<script.py^>
pause
