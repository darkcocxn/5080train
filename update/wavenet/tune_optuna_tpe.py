# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import wavenetv1.wavenetv1 as wavenet_module  # noqa: E402
from sequence_model_common import apply_common_environment_overrides  # noqa: E402
from update.common.optuna_tpe_common import run_history_module_tpe_tuning  # noqa: E402


BASE_CONFIG = wavenet_module.Config


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split("-")]


def suggest_params(trial):
    dilation_key = trial.suggest_categorical(
        "WAVENET_DILATIONS_KEY",
        ["1-2-4-8-16-32", "1-2-4-8-16-32-64", "1-2-4-8-16-32-64-128-256"],
    )
    return {
        "SEQ_LEN": trial.suggest_categorical("SEQ_LEN", [2048, 4096]),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [48, 64, 96]),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 3.0e-5, 7.0e-4, log=True),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1.0e-6, 5.0e-4, log=True),
        "GRAD_CLIP_NORM": trial.suggest_categorical("GRAD_CLIP_NORM", [0.5, 0.8, 1.0, 1.5]),
        "WAVENET_RESIDUAL_CHANNELS": trial.suggest_categorical("WAVENET_RESIDUAL_CHANNELS", [32, 48, 64, 96]),
        "WAVENET_SKIP_CHANNELS": trial.suggest_categorical("WAVENET_SKIP_CHANNELS", [64, 96, 128, 192]),
        "WAVENET_KERNEL_SIZE": trial.suggest_categorical("WAVENET_KERNEL_SIZE", [3, 5]),
        "WAVENET_DILATION_CYCLES": trial.suggest_int("WAVENET_DILATION_CYCLES", 2, 4),
        "WAVENET_DILATIONS_PER_CYCLE": _ints(dilation_key),
        "WAVENET_DROPOUT": trial.suggest_float("WAVENET_DROPOUT", 0.04, 0.25),
        "WAVENET_ATTENTION_DIM": trial.suggest_categorical("WAVENET_ATTENTION_DIM", [64, 96, 128, 192]),
        "SEQUENCE_PROJECTOR_DIM": trial.suggest_categorical("SEQUENCE_PROJECTOR_DIM", [192, 256, 288, 384]),
        "SCALAR_EMBED_DIM": trial.suggest_categorical("SCALAR_EMBED_DIM", [128, 192, 256]),
        "SCALAR_RES_BLOCKS": trial.suggest_int("SCALAR_RES_BLOCKS", 2, 5),
        "SCALAR_RES_DROPOUT": trial.suggest_float("SCALAR_RES_DROPOUT", 0.08, 0.30),
        "FUSION_BILINEAR_DIM": trial.suggest_categorical("FUSION_BILINEAR_DIM", [48, 64, 80, 112]),
        "FUSION_OUTPUT_DIM": trial.suggest_categorical("FUSION_OUTPUT_DIM", [256, 384, 512]),
        "HEAD_DROPOUT": trial.suggest_float("HEAD_DROPOUT", 0.10, 0.35),
    }


def train_once(config):
    apply_common_environment_overrides(config)
    wavenet_module.Config = config
    return wavenet_module.train_sequence_model(config, wavenet_module.build_model)


if __name__ == "__main__":
    run_history_module_tpe_tuning(
        "wavenet",
        wavenet_module,
        BASE_CONFIG,
        train_once,
        suggest_params,
        default_trials=20,
    )
