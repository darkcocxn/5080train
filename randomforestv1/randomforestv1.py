# -*- coding: utf-8 -*-
"""Random Forest v1 training script for the non-image surrogate baseline."""

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
    MODEL_TAG = "randomforestv1"
    ENV_PREFIX = "RANDOMFOREST"
    ALGORITHM_DISPLAY_NAME = "Random Forest V1"
    INPUT_MODE = "scalar_plus_downsampled_waveform"
    MODEL_FAMILY = "tabular_random_forest_waveform_scalar_surrogate"
    ARCHITECTURE_REVISION = "randomforestv1_scalar_plus_downsampled_waveform"
    LITERATURE_BASIS = BaseTabularConfig.LITERATURE_BASIS + [
        "Random Forest is included as a stable bagging-tree baseline frequently used in structural response surrogate comparisons.",
    ]
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / MODEL_TAG
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_ROOT_DIR
    BEST_MODEL_NAME = "best_randomforest_model.pkl"

    N_ESTIMATORS = 500
    MAX_DEPTH = None
    MIN_SAMPLES_LEAF = 2
    MIN_SAMPLES_SPLIT = 2
    MAX_FEATURES = "sqrt"


def _optional_int(value: str | None):
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "none":
        return None
    return int(value)


def build_model(config):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=int(os.environ.get("SURMOD_RF_N_ESTIMATORS", config.N_ESTIMATORS)),
        max_depth=_optional_int(os.environ.get("SURMOD_RF_MAX_DEPTH")) if os.environ.get("SURMOD_RF_MAX_DEPTH") else config.MAX_DEPTH,
        min_samples_leaf=int(os.environ.get("SURMOD_RF_MIN_SAMPLES_LEAF", config.MIN_SAMPLES_LEAF)),
        min_samples_split=int(os.environ.get("SURMOD_RF_MIN_SAMPLES_SPLIT", config.MIN_SAMPLES_SPLIT)),
        max_features=os.environ.get("SURMOD_RF_MAX_FEATURES", config.MAX_FEATURES),
        bootstrap=True,
        n_jobs=int(os.environ.get("SURMOD_N_JOBS", "-1")),
        random_state=int(config.SEED),
        verbose=int(os.environ.get("SURMOD_RF_VERBOSE", "1")),
    )


if __name__ == "__main__":
    train_sklearn_tabular_model(Config, build_model)
