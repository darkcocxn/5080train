# -*- coding: utf-8 -*-
"""XGBoost v1 training script for the non-image surrogate baseline."""

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
    MODEL_TAG = "xgboostv1"
    ENV_PREFIX = "XGBOOST"
    ALGORITHM_DISPLAY_NAME = "XGBoost V1"
    INPUT_MODE = "scalar_plus_downsampled_waveform"
    MODEL_FAMILY = "tabular_xgboost_waveform_scalar_surrogate"
    ARCHITECTURE_REVISION = "xgboostv1_scalar_plus_downsampled_waveform"
    LITERATURE_BASIS = BaseTabularConfig.LITERATURE_BASIS + [
        "XGBoost is included as a strong gradient-boosted tree baseline in recent seismic response prediction studies.",
    ]
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / MODEL_TAG
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_ROOT_DIR
    BEST_MODEL_NAME = "best_xgboost_model.pkl"

    N_ESTIMATORS = 2000
    LEARNING_RATE = 0.03
    MAX_DEPTH = 5
    MIN_CHILD_WEIGHT = 3.0
    SUBSAMPLE = 0.85
    COLSAMPLE_BYTREE = 0.85
    REG_LAMBDA = 3.0
    REG_ALPHA = 0.0


def build_model(config):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError("xgboost is not installed. Install it with: uv add xgboost") from exc

    return XGBRegressor(
        objective=os.environ.get("SURMOD_XGB_OBJECTIVE", "reg:squarederror"),
        n_estimators=int(os.environ.get("SURMOD_XGB_N_ESTIMATORS", config.N_ESTIMATORS)),
        learning_rate=float(os.environ.get("SURMOD_XGB_LEARNING_RATE", config.LEARNING_RATE)),
        max_depth=int(os.environ.get("SURMOD_XGB_MAX_DEPTH", config.MAX_DEPTH)),
        min_child_weight=float(os.environ.get("SURMOD_XGB_MIN_CHILD_WEIGHT", config.MIN_CHILD_WEIGHT)),
        subsample=float(os.environ.get("SURMOD_XGB_SUBSAMPLE", config.SUBSAMPLE)),
        colsample_bytree=float(os.environ.get("SURMOD_XGB_COLSAMPLE_BYTREE", config.COLSAMPLE_BYTREE)),
        reg_lambda=float(os.environ.get("SURMOD_XGB_REG_LAMBDA", config.REG_LAMBDA)),
        reg_alpha=float(os.environ.get("SURMOD_XGB_REG_ALPHA", config.REG_ALPHA)),
        tree_method=os.environ.get("SURMOD_XGB_TREE_METHOD", "hist"),
        device=os.environ.get("SURMOD_XGB_DEVICE", "cpu"),
        n_jobs=int(os.environ.get("SURMOD_N_JOBS", "-1")),
        random_state=int(config.SEED),
        verbosity=int(os.environ.get("SURMOD_XGB_VERBOSITY", "1")),
    )


if __name__ == "__main__":
    train_sklearn_tabular_model(Config, build_model)
