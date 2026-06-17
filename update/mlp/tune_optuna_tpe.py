# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mlpv1.mlpv1 import Config  # noqa: E402
from tabular_model_common import train_torch_mlp_tabular_model  # noqa: E402
from update.common.optuna_tpe_common import run_torch_mlp_tpe_tuning  # noqa: E402


def suggest_params(trial):
    return {
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [256, 512, 768, 1024]),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 1.0e-5, 3.0e-3, log=True),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1.0e-6, 1.0e-3, log=True),
        "SMOOTH_L1_BETA": trial.suggest_categorical("SMOOTH_L1_BETA", [0.3, 0.5, 1.0, 1.5, 2.0]),
        "EARLY_STOPPING_PATIENCE": trial.suggest_categorical("EARLY_STOPPING_PATIENCE", [15, 20, 25, 30]),
        "MLP_HIDDEN_DIM": trial.suggest_categorical("MLP_HIDDEN_DIM", [128, 192, 256, 384, 512]),
        "MLP_BLOCK_COUNT": trial.suggest_int("MLP_BLOCK_COUNT", 2, 5),
        "MLP_HIDDEN_MULT": trial.suggest_categorical("MLP_HIDDEN_MULT", [2, 3, 4]),
        "MLP_DROPOUT": trial.suggest_float("MLP_DROPOUT", 0.05, 0.35),
        "WAVEFORM_FEATURE_COUNT": trial.suggest_categorical("WAVEFORM_FEATURE_COUNT", [64, 128, 256, 384]),
    }


if __name__ == "__main__":
    run_torch_mlp_tpe_tuning(
        "mlp",
        Config,
        train_torch_mlp_tabular_model,
        suggest_params,
        default_trials=30,
    )
