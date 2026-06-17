# -*- coding: utf-8 -*-
"""PyTorch MLP v1 training script for the non-image surrogate baseline."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tabular_model_common import BaseTabularConfig, train_torch_mlp_tabular_model  # noqa: E402


class Config(BaseTabularConfig):
    SCRIPT_DIR = SCRIPT_DIR
    PROJECT_ROOT = PROJECT_ROOT
    MODEL_TAG = "mlpv1"
    ENV_PREFIX = "MLP"
    ALGORITHM_DISPLAY_NAME = "MLP V1"
    INPUT_MODE = "scalar_plus_downsampled_waveform"
    MODEL_FAMILY = "tabular_mlp_waveform_scalar_surrogate"
    ARCHITECTURE_REVISION = "mlpv1_residual_scalar_plus_downsampled_waveform"
    LITERATURE_BASIS = BaseTabularConfig.LITERATURE_BASIS + [
        "MLP is included as the neural-network tabular/time-history baseline against tree ensembles and 2D-CNN models.",
    ]
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / MODEL_TAG
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_ROOT_DIR
    BEST_MODEL_NAME = "best_mlp_model.pth"
    FINAL_MODEL_NAME = "mlp_model.pth"

    USE_STANDARD_SCALER = True
    BATCH_SIZE = 512
    NUM_EPOCHS = 180
    LEARNING_RATE = 1.0e-3
    WEIGHT_DECAY = 1.0e-4
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 8
    MIN_LR = 1.0e-6
    EARLY_STOPPING_PATIENCE = 25
    SMOOTH_L1_BETA = 1.0
    GRAD_CLIP_NORM = 1.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MLP_HIDDEN_DIM = 256
    MLP_BLOCK_COUNT = 3
    MLP_HIDDEN_MULT = 2
    MLP_DROPOUT = 0.15


if __name__ == "__main__":
    print(f"Running MLP v1 on device: {Config.DEVICE}")
    train_torch_mlp_tabular_model(Config)
