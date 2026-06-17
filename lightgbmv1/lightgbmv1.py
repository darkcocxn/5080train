# -*- coding: utf-8 -*-
"""LightGBM v1 training script for the non-image surrogate baseline."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tabular_model_common import BaseTabularConfig, train_sklearn_tabular_model  # noqa: E402


class Config(BaseTabularConfig):
    SCRIPT_DIR = SCRIPT_DIR
    PROJECT_ROOT = PROJECT_ROOT
    MODEL_TAG = "lightgbmv1"
    ENV_PREFIX = "LIGHTGBM"
    ALGORITHM_DISPLAY_NAME = "LightGBM V1"
    INPUT_MODE = "scalar_plus_downsampled_waveform"
    MODEL_FAMILY = "tabular_lightgbm_waveform_scalar_surrogate"
    ARCHITECTURE_REVISION = "lightgbmv1_scalar_plus_downsampled_waveform"
    LITERATURE_BASIS = BaseTabularConfig.LITERATURE_BASIS + [
        "LightGBM is included as an efficient histogram-boosting baseline for large tabular surrogate datasets.",
    ]
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / MODEL_TAG
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_ROOT_DIR
    BEST_MODEL_NAME = "best_lightgbm_model.pkl"

    N_ESTIMATORS = 3000
    LEARNING_RATE = 0.03
    NUM_LEAVES = 63
    MAX_DEPTH = -1
    MIN_CHILD_SAMPLES = 50
    SUBSAMPLE = 0.85
    COLSAMPLE_BYTREE = 0.85
    REG_LAMBDA = 3.0
    REG_ALPHA = 0.0


def build_model(config):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise ImportError("lightgbm is not installed. Install it with: uv add lightgbm") from exc

    return LGBMRegressor(
        objective=os.environ.get("SURMOD_LGB_OBJECTIVE", "regression"),
        n_estimators=int(os.environ.get("SURMOD_LGB_N_ESTIMATORS", config.N_ESTIMATORS)),
        learning_rate=float(os.environ.get("SURMOD_LGB_LEARNING_RATE", config.LEARNING_RATE)),
        num_leaves=int(os.environ.get("SURMOD_LGB_NUM_LEAVES", config.NUM_LEAVES)),
        max_depth=int(os.environ.get("SURMOD_LGB_MAX_DEPTH", config.MAX_DEPTH)),
        min_child_samples=int(os.environ.get("SURMOD_LGB_MIN_CHILD_SAMPLES", config.MIN_CHILD_SAMPLES)),
        subsample=float(os.environ.get("SURMOD_LGB_SUBSAMPLE", config.SUBSAMPLE)),
        colsample_bytree=float(os.environ.get("SURMOD_LGB_COLSAMPLE_BYTREE", config.COLSAMPLE_BYTREE)),
        reg_lambda=float(os.environ.get("SURMOD_LGB_REG_LAMBDA", config.REG_LAMBDA)),
        reg_alpha=float(os.environ.get("SURMOD_LGB_REG_ALPHA", config.REG_ALPHA)),
        random_state=int(config.SEED),
        n_jobs=int(os.environ.get("SURMOD_N_JOBS", "-1")),
        verbose=int(os.environ.get("SURMOD_LGB_VERBOSE", "-1")),
    )


if __name__ == "__main__":
    train_sklearn_tabular_model(Config, build_model)
