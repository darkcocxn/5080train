# -*- coding: utf-8 -*-
"""Evaluate the WaveNet/TCN sequence surrogate."""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from sequence_model_common import evaluate_sequence_model  # noqa: E402
from wavenetv1 import Config, build_model  # noqa: E402


if __name__ == "__main__":
    print(f"Running WaveNet v1 evaluation on device: {Config.DEVICE}")
    evaluate_sequence_model(Config, build_model)
