@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

set DEFAULT_ARGS=--trials 15 --data-use-ratio 0.30 --num-epochs 30 --num-workers 0 --device auto

if "%~1"=="" (
    echo Running LSTM Optuna/TPE with default pre-search args:
    echo %DEFAULT_ARGS%
    uv run python update\lstm\tune_optuna_tpe.py %DEFAULT_ARGS%
) else (
    echo Running LSTM Optuna/TPE with custom args:
    echo %*
    uv run python update\lstm\tune_optuna_tpe.py %*
)

pause
