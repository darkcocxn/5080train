@echo off
chcp 65001 >nul
cd /d "%~dp0..\.."

set DEFAULT_ARGS=--trials 20 --data-use-ratio 0.30 --num-epochs 40 --device auto

if "%~1"=="" (
    echo Running MLP Optuna/TPE with default pre-search args:
    echo %DEFAULT_ARGS%
    uv run python update\mlp\tune_optuna_tpe.py %DEFAULT_ARGS%
) else (
    echo Running MLP Optuna/TPE with custom args:
    echo %*
    uv run python update\mlp\tune_optuna_tpe.py %*
)

pause
