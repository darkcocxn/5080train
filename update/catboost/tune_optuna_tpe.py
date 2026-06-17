# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catboostv1.catboostv1 import Config, build_model  # noqa: E402
from update.common.optuna_tpe_common import run_sklearn_tpe_tuning  # noqa: E402


def suggest_params(trial):
    return {
        "ITERATIONS": trial.suggest_int("ITERATIONS", 500, 6000, step=250),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 0.005, 0.08, log=True),
        "DEPTH": trial.suggest_int("DEPTH", 4, 10),
        "L2_LEAF_REG": trial.suggest_float("L2_LEAF_REG", 1.0, 30.0, log=True),
        "SUBSAMPLE": trial.suggest_float("SUBSAMPLE", 0.60, 1.00),
        "WAVEFORM_FEATURE_COUNT": trial.suggest_categorical("WAVEFORM_FEATURE_COUNT", [64, 128, 256, 384]),
    }


if __name__ == "__main__":
    run_sklearn_tpe_tuning(
        "catboost",
        Config,
        build_model,
        suggest_params,
        model_env_prefixes=("SURMOD_CAT_",),
        default_trials=50,
    )
