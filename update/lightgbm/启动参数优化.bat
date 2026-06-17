@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

set DEFAULT_ARGS=--trials 30 --data-use-ratio 0.30 --n-jobs 8

if "%~1"=="" (
    echo Running LightGBM Optuna/TPE with default pre-search args:
    echo %DEFAULT_ARGS%
    uv run python update\lightgbm\tune_optuna_tpe.py %DEFAULT_ARGS%
) else (
    echo Running LightGBM Optuna/TPE with custom args:
    echo %*
    uv run python update\lightgbm\tune_optuna_tpe.py %*
)

pause
