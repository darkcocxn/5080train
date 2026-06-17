# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from randomforestv1.randomforestv1 import Config, build_model  # noqa: E402
from update.common.optuna_tpe_common import run_sklearn_tpe_tuning  # noqa: E402


def suggest_params(trial):
    return {
        "N_ESTIMATORS": trial.suggest_int("N_ESTIMATORS", 200, 1200, step=100),
        "MAX_DEPTH": trial.suggest_categorical("MAX_DEPTH", [None, 6, 8, 10, 12, 16, 20, 24, 32]),
        "MIN_SAMPLES_LEAF": trial.suggest_int("MIN_SAMPLES_LEAF", 1, 8),
        "MIN_SAMPLES_SPLIT": trial.suggest_int("MIN_SAMPLES_SPLIT", 2, 20),
        "MAX_FEATURES": trial.suggest_categorical("MAX_FEATURES", ["sqrt", "log2", 0.35, 0.50, 0.70, 1.0]),
        "WAVEFORM_FEATURE_COUNT": trial.suggest_categorical("WAVEFORM_FEATURE_COUNT", [64, 128, 256, 384]),
    }


if __name__ == "__main__":
    run_sklearn_tpe_tuning(
        "randomforest",
        Config,
        build_model,
        suggest_params,
        model_env_prefixes=("SURMOD_RF_",),
        default_trials=40,
    )
