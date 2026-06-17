# -*- coding: utf-8 -*-
"""Evaluate the XGBoost v1 non-image surrogate baseline."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from tabular_model_common import evaluate_sklearn_tabular_model  # noqa: E402
from xgboostv1 import Config  # noqa: E402


if __name__ == "__main__":
    evaluate_sklearn_tabular_model(Config)
