# -*- coding: utf-8 -*-
"""CatBoost v1 training script for the non-image surrogate baseline."""

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
    MODEL_TAG = "catboostv1"
    ENV_PREFIX = "CATBOOST"
    ALGORITHM_DISPLAY_NAME = "CatBoost V1"
    INPUT_MODE = "scalar_plus_downsampled_waveform"
    MODEL_FAMILY = "tabular_catboost_waveform_scalar_surrogate"
    ARCHITECTURE_REVISION = "catboostv1_scalar_plus_downsampled_waveform"
    LITERATURE_BASIS = BaseTabularConfig.LITERATURE_BASIS + [
        "CatBoost is included because recent steel-frame seismic surrogate studies report it as a competitive tabular model.",
    ]
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / MODEL_TAG
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_ROOT_DIR
    BEST_MODEL_NAME = "best_catboost_model.pkl"

    ITERATIONS = 4000
    LEARNING_RATE = 0.03
    DEPTH = 6
    L2_LEAF_REG = 3.0
    SUBSAMPLE = 0.85


def build_model(config):
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:
        raise ImportError("catboost is not installed. Install it with: uv add catboost") from exc

    return CatBoostRegressor(
        loss_function=os.environ.get("SURMOD_CAT_LOSS_FUNCTION", "RMSE"),
        iterations=int(os.environ.get("SURMOD_CAT_ITERATIONS", config.ITERATIONS)),
        learning_rate=float(os.environ.get("SURMOD_CAT_LEARNING_RATE", config.LEARNING_RATE)),
        depth=int(os.environ.get("SURMOD_CAT_DEPTH", config.DEPTH)),
        l2_leaf_reg=float(os.environ.get("SURMOD_CAT_L2_LEAF_REG", config.L2_LEAF_REG)),
        subsample=float(os.environ.get("SURMOD_CAT_SUBSAMPLE", config.SUBSAMPLE)),
        bootstrap_type=os.environ.get("SURMOD_CAT_BOOTSTRAP_TYPE", "Bernoulli"),
        random_seed=int(config.SEED),
        thread_count=int(os.environ.get("SURMOD_N_JOBS", "-1")),
        verbose=int(os.environ.get("SURMOD_CAT_VERBOSE", "100")),
        allow_writing_files=False,
    )


if __name__ == "__main__":
    train_sklearn_tabular_model(Config, build_model)
