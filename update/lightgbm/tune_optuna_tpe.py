# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lightgbmv1.lightgbmv1 import Config, build_model  # noqa: E402
from update.common.optuna_tpe_common import run_sklearn_tpe_tuning  # noqa: E402


def suggest_params(trial):
    return {
        "N_ESTIMATORS": trial.suggest_int("N_ESTIMATORS", 500, 5000, step=250),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 0.005, 0.08, log=True),
        "NUM_LEAVES": trial.suggest_int("NUM_LEAVES", 15, 255, log=True),
        "MAX_DEPTH": trial.suggest_categorical("MAX_DEPTH", [-1, 4, 6, 8, 10, 12, 16]),
        "MIN_CHILD_SAMPLES": trial.suggest_int("MIN_CHILD_SAMPLES", 10, 200),
        "SUBSAMPLE": trial.suggest_float("SUBSAMPLE", 0.60, 1.00),
        "COLSAMPLE_BYTREE": trial.suggest_float("COLSAMPLE_BYTREE", 0.60, 1.00),
        "REG_LAMBDA": trial.suggest_float("REG_LAMBDA", 1.0e-3, 30.0, log=True),
        "REG_ALPHA": trial.suggest_float("REG_ALPHA", 1.0e-6, 5.0, log=True),
        "WAVEFORM_FEATURE_COUNT": trial.suggest_categorical("WAVEFORM_FEATURE_COUNT", [64, 128, 256, 384]),
    }


if __name__ == "__main__":
    run_sklearn_tpe_tuning(
        "lightgbm",
        Config,
        build_model,
        suggest_params,
        model_env_prefixes=("SURMOD_LGB_",),
        default_trials=50,
    )
