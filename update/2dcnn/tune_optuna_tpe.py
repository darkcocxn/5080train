# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from update.common.optuna_tpe_common import run_history_module_tpe_tuning  # noqa: E402


def load_2dcnn_module():
    script_path = PROJECT_ROOT / "2dcnnv11" / "2dcnnv11.py"
    spec = importlib.util.spec_from_file_location("surmod_2dcnnv11_train", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load 2D-CNN script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ints(value: str) -> list[int]:
    return [int(item) for item in value.split("-")]


def suggest_params(trial):
    image_size = trial.suggest_categorical("IMAGE_SIZE_KEY", [128, 160, 192])
    channels_key = trial.suggest_categorical(
        "CNN_CHANNELS_KEY",
        ["24-56-112-176", "32-72-144-224", "40-80-160-256"],
    )
    return {
        "IMAGE_SIZE": (int(image_size), int(image_size)),
        "BATCH_SIZE": trial.suggest_categorical("BATCH_SIZE", [48, 64, 96]),
        "LEARNING_RATE": trial.suggest_float("LEARNING_RATE", 3.0e-5, 5.0e-4, log=True),
        "WEIGHT_DECAY": trial.suggest_float("WEIGHT_DECAY", 1.0e-6, 5.0e-4, log=True),
        "GRAD_CLIP_NORM": trial.suggest_categorical("GRAD_CLIP_NORM", [0.5, 0.8, 1.0, 1.5]),
        "CNN_CHANNELS": _ints(channels_key),
        "CNN_DROPOUT": trial.suggest_float("CNN_DROPOUT", 0.04, 0.20),
        "CNN_PROJECTOR_DIM": trial.suggest_categorical("CNN_PROJECTOR_DIM", [192, 256, 288, 384]),
        "CNN_PROJECTOR_DROPOUT": trial.suggest_float("CNN_PROJECTOR_DROPOUT", 0.05, 0.25),
        "SCALAR_EMBED_DIM": trial.suggest_categorical("SCALAR_EMBED_DIM", [128, 192, 256]),
        "SCALAR_RES_BLOCKS": trial.suggest_int("SCALAR_RES_BLOCKS", 2, 5),
        "SCALAR_RES_DROPOUT": trial.suggest_float("SCALAR_RES_DROPOUT", 0.08, 0.30),
        "FUSION_BILINEAR_DIM": trial.suggest_categorical("FUSION_BILINEAR_DIM", [48, 64, 80, 112]),
        "FUSION_OUTPUT_DIM": trial.suggest_categorical("FUSION_OUTPUT_DIM", [256, 384, 512]),
        "FUSION_DROPOUT": trial.suggest_float("FUSION_DROPOUT", 0.08, 0.28),
        "HEAD_DROPOUT": trial.suggest_float("HEAD_DROPOUT", 0.10, 0.35),
        "TIME_FREQ_MASK_PROB": trial.suggest_float("TIME_FREQ_MASK_PROB", 0.10, 0.45),
    }


def main():
    cnn_module = load_2dcnn_module()
    base_config = cnn_module.Config

    def train_once(config):
        cnn_module.Config = config
        history = cnn_module.train()
        cnn_module.plot_results(history)
        return history

    run_history_module_tpe_tuning(
        "2dcnn",
        cnn_module,
        base_config,
        train_once,
        suggest_params,
        default_trials=20,
    )


if __name__ == "__main__":
    main()
