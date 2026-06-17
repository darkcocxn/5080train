# -*- coding: utf-8 -*-
"""
WaveNet/TCN sequence surrogate training script.

This model reads raw ground-motion sequences directly and fuses them with the
same scalar features used by the current 2D-CNN branch.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from sequence_model_common import (  # noqa: E402
    ScalarFeatureEncoder,
    SequenceFusionRegressor,
    apply_common_environment_overrides,
    build_conv1d_norm,
    train_sequence_model,
)


class Config:
    SCRIPT_DIR = SCRIPT_DIR
    PROJECT_ROOT = PROJECT_ROOT

    CSV_DIR_CANDIDATES = (
        PROJECT_ROOT / "newdata",
        PROJECT_ROOT / "CSV-dataset",
        PROJECT_ROOT / "数据集",
    )
    CSV_DIR = next((path for path in CSV_DIR_CANDIDATES if path.exists()), CSV_DIR_CANDIDATES[0])
    TRAIN_CSV_PATH = None
    VAL_CSV_PATH = None
    TEST_CSV_PATH = None
    DATASET_PREFIX = "opensees_surrogate_dataset_floors_3_to_7_"
    TRAIN_FILE_PATTERN = f"{DATASET_PREFIX}*_train.csv"
    VAL_FILE_PATTERN = f"{DATASET_PREFIX}*_val.csv"
    TEST_FILE_PATTERN = f"{DATASET_PREFIX}*_test.csv"

    MODEL_TAG = "wavenetv1"
    ENV_PREFIX = "WAVENET"
    MODEL_FAMILY = "wave_sequence_wavenet_scalar_fusion"
    ARCHITECTURE_REVISION = "wavenetv1_dilated_tcn_attention_tailaware"
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / MODEL_TAG
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_ROOT_DIR
    UNIQUE_MODEL_RUN_DIR = True

    BEST_WEIGHTS_NAME = "best_wavenet_model.pth"
    MAE_WEIGHTS_NAME = "best_wavenet_mae_model.pth"
    FOCUS_WEIGHTS_NAME = "best_wavenet_focus_model.pth"
    EXTREME_UNDER_WEIGHTS_NAME = "best_wavenet_extreme_under_model.pth"
    FINAL_WEIGHTS_NAME = "wavenet_model.pth"
    EMA_WEIGHTS_NAME = "ema_wavenet_model.pth"
    MODEL_WEIGHTS_PATH = MODEL_ROOT_DIR / BEST_WEIGHTS_NAME

    SAMPLE_ID_COL = "sample_id"
    TXT_COL = "txt_path"
    LABEL_COL = "max_drift_ratio_raw"
    STATUS_COL = "analysis_status"
    WAVE_DT_COL = "wave_dt"
    DAMPER_LAYOUT_COL = "damper_layout"
    BASE_SCALAR_COLS = ["num_floors", "floor_mass", "floor_height", "k_base_1_4", "Fy_add"]
    WAVE_FEATURE_COLS = [
        "period_1_sec",
        "wave_pga",
        "wave_rms",
        "wave_mean_abs",
        "wave_cav",
        "wave_arias_proxy",
        "wave_duration_5_95",
        "wave_zero_crossing_rate",
        "wave_dominant_freq",
        "wave_spectral_centroid",
        "wave_predominant_period",
        "wave_intensity_score",
    ]
    WAVE_LOG_FEATURE_COLS = list(WAVE_FEATURE_COLS)
    USE_WAVE_DERIVED_FEATURES = True
    SCALAR_COLS = list(BASE_SCALAR_COLS)
    MAX_DAMPER_FLOORS = 7

    LABEL_SCALE = 1000.0
    SCALE_TARGET = True
    DATA_USE_RATIO = 1.0
    SEED = 42

    TARGET_DT = 0.01
    SEQ_LEN = 4096
    SEQUENCE_CHANNELS = ["acc_scaled", "abs_acc_scaled"]
    SEQUENCE_CLIP_VALUE = 8.0
    CACHE_SEQUENCES = True

    BATCH_SIZE = 96
    LEARNING_RATE = 2.0e-4
    WEIGHT_DECAY = 1.0e-4
    NUM_EPOCHS = 120
    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 5
    MIN_LR = 3e-7
    EARLY_STOPPING_PATIENCE = 20
    WARMUP_EPOCHS = 5
    WARMUP_START_FACTOR = 0.25
    GRAD_CLIP_NORM = 1.0
    SMOOTH_L1_BETA = 1.0
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0
    USE_AMP = True
    USE_EMA = True
    EMA_DECAY = 0.998
    OPTIMIZER_NO_DECAY_NORM_AND_BIAS = True

    SEQ_NORM = "group"
    SEQ_GROUP_NORM_MAX_GROUPS = 8
    WAVENET_RESIDUAL_CHANNELS = 64
    WAVENET_SKIP_CHANNELS = 128
    WAVENET_KERNEL_SIZE = 3
    WAVENET_DILATION_CYCLES = 3
    WAVENET_DILATIONS_PER_CYCLE = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    WAVENET_DROPOUT = 0.10
    WAVENET_ATTENTION_DIM = 128
    SEQUENCE_PROJECTOR_DIM = 288
    SEQUENCE_PROJECTOR_DROPOUT = 0.12

    SCALAR_NORM = "layer"
    SCALAR_EMBED_DIM = 192
    SCALAR_INPUT_DROPOUT = 0.09
    SCALAR_RES_BLOCKS = 4
    SCALAR_RES_HIDDEN_MULT = 2
    SCALAR_RES_DROPOUT = 0.18
    SCALAR_RESIDUAL_SCALE_INIT = 0.10

    FUSION_MODE = "gated_bilinear"
    FUSION_BILINEAR_DIM = 80
    FUSION_OUTPUT_DIM = 384
    FUSION_DROPOUT = 0.18
    FUSION_INTERACTION_SCALE_INIT = 0.32
    HEAD_HIDDEN_DIMS = [384, 128]
    HEAD_DROPOUT = 0.25

    USE_TARGET_WEIGHTED_LOSS = True
    USE_DENSITY_WEIGHTED_LOSS = True
    LOSS_DENSITY_BIN_COUNT = 40
    LOSS_DENSITY_SMOOTH_KERNEL_SIZE = 7
    LOSS_DENSITY_SMOOTH_SIGMA = 1.35
    LOSS_DENSITY_ALPHA = 0.58
    LOSS_WEIGHT_MIN = 0.5
    LOSS_WEIGHT_MAX = 14.0
    USE_SMOOTH_TAIL_WEIGHTS = True
    TAIL_WEIGHT_TRANSITION_WIDTH = 0.003
    ADAPTIVE_TAIL_THRESHOLDS = True
    ENGINEERING_TAIL_THRESHOLDS = [0.005, 0.010, 0.020]
    FALLBACK_TAIL_THRESHOLDS = [0.003, 0.005, 0.007]
    TAIL_MIN_POSITIVE_COUNT = 96
    TAIL_LOSS_WEIGHTS = [1.20, 1.65, 2.40]

    USE_TAIL_UNDERPREDICTION_LOSS = True
    TAIL_UNDERPREDICTION_START_EPOCH = 6
    TAIL_UNDERPREDICTION_RAMP_EPOCHS = 12
    TAIL_UNDERPREDICTION_THRESHOLD = 0.010
    TAIL_UNDERPREDICTION_WEIGHT = 0.14
    EXTREME_TAIL_UNDERPREDICTION_THRESHOLD = 0.020
    EXTREME_TAIL_UNDERPREDICTION_WEIGHT = 0.34
    TAIL_UNDERPREDICTION_MAX_LOSS = 2.4
    USE_TAIL_RELATIVE_UNDER_LOSS = True
    TAIL_RELATIVE_UNDER_WEIGHT = 0.018
    EXTREME_TAIL_RELATIVE_UNDER_WEIGHT = 0.040
    USE_TAIL_PINBALL_LOSS = True
    TAIL_PINBALL_TAU = 0.58
    EXTREME_TAIL_PINBALL_TAU = 0.64
    TAIL_PINBALL_WEIGHT = 0.008
    EXTREME_TAIL_PINBALL_WEIGHT = 0.016

    USE_TAIL_CLASSIFICATION_AUX = True
    TAIL_CLASSIFICATION_THRESHOLDS = [0.003, 0.005, 0.007]
    TAIL_CLASSIFICATION_LOSS_WEIGHTS = [0.010, 0.024, 0.044]
    TAIL_CLASSIFICATION_HIDDEN_DIM = 96
    TAIL_CLASSIFICATION_DROPOUT = 0.10
    TAIL_CLASSIFICATION_INIT_BIAS = -3.0
    TAIL_CLASSIFICATION_POS_WEIGHT_MAX = 26.0
    TAIL_CLASSIFICATION_RAMP_EPOCHS = 10

    USE_TAIL_CORRECTION_HEAD = True
    USE_TAIL_CORRECTION_GATE = True
    TAIL_CORRECTION_HIDDEN_DIM = 96
    TAIL_CORRECTION_DROPOUT = 0.10
    TAIL_CORRECTION_INIT_BIAS = -4.2
    TAIL_CORRECTION_GATE_INIT_BIAS = -1.7
    USE_TAIL_PROB_GATED_CORRECTION = True
    TAIL_PROB_GATE_INDEX = 1
    TAIL_PROB_GATE_DETACH = True
    TAIL_PROB_GATE_POWER = 1.10

    USE_WEIGHTED_SAMPLER = True
    TARGET_BIN_COUNT = 8
    SAMPLER_POWER = 0.85
    SAMPLER_MIN_WEIGHT = 0.35
    SAMPLER_MAX_WEIGHT = 12.0
    SAMPLER_TAIL_BOOSTS = [1.08, 1.35, 1.80]
    SAMPLER_NUM_SAMPLES_MULTIPLIER = 1.12

    VAL_MID_TAIL_LOW = 0.005
    VAL_MID_TAIL_HIGH = 0.010
    VAL_TAIL_THRESHOLD = 0.010
    VAL_EXTREME_TAIL_THRESHOLD = 0.020
    VAL_FOCUS_RMSE_WEIGHT = 0.18
    VAL_FOCUS_TAIL_MAE_WEIGHT = 0.25
    VAL_FOCUS_EXTREME_TAIL_MAE_WEIGHT = 0.45
    VAL_FOCUS_TAIL_UNDER_WEIGHT = 0.10
    VAL_FOCUS_EXTREME_TAIL_UNDER_WEIGHT = 0.20
    SELECTION_FOCUS_WEIGHT = 0.022
    SELECTION_MID_TAIL_WEIGHT = 0.055

    TEST_SEEDS = [42, 2026, 123]
    TEST_SAMPLE_RATIO = 0.8
    TEST_SAMPLE_WITH_REPLACEMENT = False


apply_common_environment_overrides(Config)
Config.MODEL_ROOT_DIR.mkdir(parents=True, exist_ok=True)
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


class DilatedResidualBlock(nn.Module):
    def __init__(self, dilation: int):
        super().__init__()
        residual_channels = Config.WAVENET_RESIDUAL_CHANNELS
        skip_channels = Config.WAVENET_SKIP_CHANNELS
        kernel_size = Config.WAVENET_KERNEL_SIZE
        padding = dilation * (kernel_size - 1) // 2
        self.filter_conv = nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.gate_conv = nn.Conv1d(
            residual_channels,
            residual_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.dropout = nn.Dropout(Config.WAVENET_DROPOUT)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.norm = build_conv1d_norm(Config, residual_channels)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        gated = self.dropout(gated)
        skip = self.skip_conv(gated)
        residual = self.residual_conv(gated)
        return self.norm((x + residual) / math.sqrt(2.0)), skip


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(input_dim, Config.WAVENET_ATTENTION_DIM),
            nn.Tanh(),
            nn.Linear(Config.WAVENET_ATTENTION_DIM, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        weights = torch.softmax(self.score(x_t), dim=1)
        return torch.sum(weights * x_t, dim=1)


class WaveNetSequenceSurrogate(nn.Module):
    def __init__(self, sequence_channels: int, num_scalars: int):
        super().__init__()
        residual_channels = Config.WAVENET_RESIDUAL_CHANNELS
        skip_channels = Config.WAVENET_SKIP_CHANNELS
        self.input_projection = nn.Sequential(
            nn.Conv1d(sequence_channels, residual_channels, kernel_size=1, bias=False),
            build_conv1d_norm(Config, residual_channels),
            nn.SiLU(inplace=True),
        )
        dilations = Config.WAVENET_DILATIONS_PER_CYCLE * Config.WAVENET_DILATION_CYCLES
        self.blocks = nn.ModuleList(DilatedResidualBlock(dilation) for dilation in dilations)
        self.post = nn.Sequential(
            nn.SiLU(inplace=True),
            nn.Conv1d(skip_channels, skip_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Dropout(Config.WAVENET_DROPOUT),
        )
        self.attention_pool = TemporalAttentionPooling(skip_channels)
        pooled_dim = skip_channels * 3
        self.sequence_projector = nn.Sequential(
            nn.Linear(pooled_dim, Config.SEQUENCE_PROJECTOR_DIM),
            nn.GELU(),
            nn.Dropout(Config.SEQUENCE_PROJECTOR_DROPOUT),
        )
        self.scalar_encoder = ScalarFeatureEncoder(Config, num_scalars)
        self.regressor = SequenceFusionRegressor(
            Config,
            sequence_dim=Config.SEQUENCE_PROJECTOR_DIM,
            scalar_dim=self.scalar_encoder.out_dim,
        )

    def forward(self, sequences: torch.Tensor, scalars: torch.Tensor, return_aux: bool = False):
        x = self.input_projection(sequences)
        skip_total = None
        for block in self.blocks:
            x, skip = block(x)
            skip_total = skip if skip_total is None else skip_total + skip
        assert skip_total is not None
        x_skip = self.post(skip_total / math.sqrt(float(len(self.blocks))))
        mean_pool = torch.mean(x_skip, dim=2)
        max_pool = torch.amax(x_skip, dim=2)
        attn_pool = self.attention_pool(x_skip)
        seq_features = self.sequence_projector(torch.cat((mean_pool, max_pool, attn_pool), dim=1))
        scalar_features = self.scalar_encoder(scalars)
        return self.regressor(seq_features, scalar_features, return_aux=return_aux)


def build_model(sequence_channels: int, num_scalars: int) -> nn.Module:
    return WaveNetSequenceSurrogate(sequence_channels, num_scalars)


if __name__ == "__main__":
    print(f"Running WaveNet v1 on device: {Config.DEVICE}")
    train_sequence_model(Config, build_model)
