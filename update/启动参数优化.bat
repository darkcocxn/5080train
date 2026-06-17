@echo off
chcp 65001 >nul
cd /d "%~dp0.."

echo ========================================
echo Optuna/TPE 参数优化启动菜单
echo ========================================
echo 1. Random Forest
echo 2. XGBoost
echo 3. LightGBM
echo 4. CatBoost
echo 5. MLP
echo 6. LSTM
echo 7. WaveNet
echo 8. 2D-CNN
echo 9. 仅检查全部搜索空间 dry-run
echo ========================================
set /p choice=请输入序号:

if "%choice%"=="1" call "%~dp0randomforest\启动参数优化.bat" & exit /b
if "%choice%"=="2" call "%~dp0xgboost\启动参数优化.bat" & exit /b
if "%choice%"=="3" call "%~dp0lightgbm\启动参数优化.bat" & exit /b
if "%choice%"=="4" call "%~dp0catboost\启动参数优化.bat" & exit /b
if "%choice%"=="5" call "%~dp0mlp\启动参数优化.bat" & exit /b
if "%choice%"=="6" call "%~dp0lstm\启动参数优化.bat" & exit /b
if "%choice%"=="7" call "%~dp0wavenet\启动参数优化.bat" & exit /b
if "%choice%"=="8" call "%~dp02dcnn\启动参数优化.bat" & exit /b
if "%choice%"=="9" goto dryrun_all

echo 无效序号。
pause
exit /b 1

:dryrun_all
uv run python update\randomforest\tune_optuna_tpe.py --dry-run --startup-trials 1
uv run python update\xgboost\tune_optuna_tpe.py --dry-run --startup-trials 1
uv run python update\lightgbm\tune_optuna_tpe.py --dry-run --startup-trials 1
uv run python update\catboost\tune_optuna_tpe.py --dry-run --startup-trials 1
uv run python update\mlp\tune_optuna_tpe.py --dry-run --startup-trials 1
uv run python update\lstm\tune_optuna_tpe.py --dry-run --startup-trials 1
uv run python update\wavenet\tune_optuna_tpe.py --dry-run --startup-trials 1
uv run python update\2dcnn\tune_optuna_tpe.py --dry-run --startup-trials 1
pause
