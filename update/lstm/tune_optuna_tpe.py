# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lstmv1.lstmv1 as lstm_module  # noqa: E402
from sequence_model_common import apply_common_environment_overrides  # noqa: E402
from update.common.optuna_tpe_common import run_history_module_tpe_tuning  # noqa: E402


BASE_CONFIG = lstm_module.Config


def _channels(value: str) -> list[int]:
    return [int(item) for item in value.split("-")]


def suggest_params(trial):
    stem_key = trial.suggest_categorical("SEQ_STEM_CHANNELS_KEY", ["24-48", "32-64", "48-96", "64-128"])
    return {
        "SEQ_LEN": trial.suggest_categorical("SEQ_LEN", [1024, 2048, 4096]),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [32, 64, 96]),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 3.0e-5, 5.0e-4, log=True),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1.0e-6, 5.0e-4, log=True),
        "GRAD_CLIP_NORM": trial.suggest_categorical("GRAD_CLIP_NORM", [0.5, 0.8, 1.0, 1.5]),
        "SEQ_STEM_CHANNELS": _channels(stem_key),
        "SEQ_STEM_DROPOUT": trial.suggest_float("SEQ_STEM_DROPOUT", 0.03, 0.20),
        "LSTM_HIDDEN_DIM": trial.suggest_categorical("LSTM_HIDDEN_DIM", [96, 128, 192, 256]),
        "LSTM_NUM_LAYERS": trial.suggest_int("LSTM_NUM_LAYERS", 1, 3),
        "LSTM_DROPOUT": trial.suggest_float("LSTM_DROPOUT", 0.05, 0.35),
        "LSTM_ATTENTION_DIM": trial.suggest_categorical("LSTM_ATTENTION_DIM", [64, 96, 128, 192]),
        "SEQUENCE_PROJECTOR_DIM": trial.suggest_categorical("SEQUENCE_PROJECTOR_DIM", [128, 192, 256, 384]),
        "SCALAR_EMBED_DIM": trial.suggest_categorical("SCALAR_EMBED_DIM", [128, 192, 256]),
        "SCALAR_RES_BLOCKS": trial.suggest_int("SCALAR_RES_BLOCKS", 2, 5),
        "SCALAR_RES_DROPOUT": trial.suggest_float("SCALAR_RES_DROPOUT", 0.08, 0.30),
        "FUSION_BILINEAR_DIM": trial.suggest_categorical("FUSION_BILINEAR_DIM", [48, 64, 80, 112]),
        "FUSION_OUTPUT_DIM": trial.suggest_categorical("FUSION_OUTPUT_DIM", [256, 384, 512]),
        "HEAD_DROPOUT": trial.suggest_float("HEAD_DROPOUT", 0.10, 0.35),
    }


def train_once(config):
    apply_common_environment_overrides(config)
    lstm_module.Config = config
    return lstm_module.train_sequence_model(config, lstm_module.build_model)


if __name__ == "__main__":
    run_history_module_tpe_tuning(
        "lstm",
        lstm_module,
        BASE_CONFIG,
        train_once,
        suggest_params,
        default_trials=20,
    )
