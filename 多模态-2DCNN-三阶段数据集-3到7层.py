# -*- coding: utf-8 -*-
"""
2D-CNN 多模态训练脚本（三阶段数据集，小波图版，3 到 7 层）
图像分支: 使用小波图 PNG
物理参数分支: 使用 MLP 处理结构参数

适配数据集:
1. 三阶段 3 到 7 层 train / val CSV；
2. 标签使用 `max_drift_ratio_raw`；
3. 自动过滤 `analysis_status != ok` 的失败样本；
4. 使用 warmup、EMA 权重、尾部加权采样和梯度裁剪提高验证稳定性；
5. 使用标量 LayerNorm、平滑尾部权重、尾部低估 ramp-up 和轻量时频遮挡增强；
6. 保存 scaler、最优权重和 training_metadata，供配套测试脚本复用。
"""

import os
import sys
import time
import json
import hashlib
import re
import random
from contextlib import nullcontext
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
for SUPPORT_DIR in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(SUPPORT_DIR) not in sys.path:
        sys.path.append(str(SUPPORT_DIR))

from floors_3_to_7_utils import (
    resolve_image_path,
    sample_dataframe_by_group,
)


DEFAULT_DATASET_DIR_NAMES = ("CSV-dataset", "数据集")


def _first_existing_path(*candidates: Path | None) -> Path:
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        raise ValueError("至少需要提供一个候选路径")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _latest_existing_dir(parent: Path, *patterns: str) -> Path | None:
    if not parent.exists():
        return None

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in parent.glob(pattern) if path.is_dir())
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


class Config:
    SCRIPT_DIR = SCRIPT_DIR
    PROJECT_ROOT = PROJECT_ROOT
    RAW_DATA_DIR = PROJECT_ROOT / "rawdata"
    CSV_DIR_CANDIDATES = (
        PROJECT_ROOT / DEFAULT_DATASET_DIR_NAMES[0],
        PROJECT_ROOT / DEFAULT_DATASET_DIR_NAMES[1],
    )
    CSV_DIR = next((path for path in CSV_DIR_CANDIDATES if path.exists()), CSV_DIR_CANDIDATES[0])

    TRAIN_CSV_PATH = None
    VAL_CSV_PATH = None
    DATASET_PREFIX = "opensees_surrogate_dataset_floors_3_to_7_"
    TRAIN_FILE_PATTERN = f"{DATASET_PREFIX}*_train.csv"
    VAL_FILE_PATTERN = f"{DATASET_PREFIX}*_val.csv"
    WAVELET_IMAGE_DIR = _first_existing_path(
        _latest_existing_dir(RAW_DATA_DIR / "Scalogram", "6000-1-*"),
        RAW_DATA_DIR / "Scalogram" / "6000-1",
        RAW_DATA_DIR / "Scalogram" / "6000-uniform-scale-0.1-1.0",
        RAW_DATA_DIR / "Scalogram" / "6000",
        PROJECT_ROOT / "Raw data file" / "Scalogram" / "6000-1",
        PROJECT_ROOT / "Raw data file" / "Scalogram" / "6000",
    )
    FORCE_WAVELET_IMAGE_DIR = True
    MODEL_ROOT_DIR = SCRIPT_DIR / "model-2dcnn-3stage-rf-3to7"
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_DIR
    UNIQUE_MODEL_RUN_DIR = True

    IMAGE_COL = "image_path"
    TXT_COL = "txt_path"
    LABEL_COL = "max_drift_ratio_raw"
    STATUS_COL = "analysis_status"
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
    DAMPER_LAYOUT_COL = "damper_layout"
    MAX_DAMPER_FLOORS = 7
    LABEL_SCALE = 1000.0
    SCALE_TARGET = True

    DATA_USE_RATIO = 1.0
    VAL_SPLIT = 0.2
    SEED = 42

    IMAGE_SIZE = (160, 160)
    IMAGE_NORMALIZE_MEAN = [0.5, 0.5, 0.5]
    IMAGE_NORMALIZE_STD = [0.5, 0.5, 0.5]
    BATCH_SIZE = 96
    LEARNING_RATE = 1.5e-4
    WEIGHT_DECAY = 3e-5
    NUM_EPOCHS = 160

    SCHEDULER_FACTOR = 0.5
    SCHEDULER_PATIENCE = 6
    MIN_LR = 5e-7
    EARLY_STOPPING_PATIENCE = 28
    WARMUP_EPOCHS = 5
    WARMUP_START_FACTOR = 0.25
    GRAD_CLIP_NORM = 0.8
    SMOOTH_L1_BETA = 1.0
    USE_TARGET_WEIGHTED_LOSS = True
    USE_DENSITY_WEIGHTED_LOSS = True
    LOSS_DENSITY_BIN_COUNT = 32
    LOSS_DENSITY_SMOOTH_KERNEL_SIZE = 5
    LOSS_DENSITY_SMOOTH_SIGMA = 1.0
    LOSS_DENSITY_ALPHA = 0.50
    LOSS_WEIGHT_MIN = 0.5
    LOSS_WEIGHT_MAX = 14.0
    LOSS_WEIGHT_GE_005 = 1.5
    LOSS_WEIGHT_GE_010 = 3.8
    LOSS_WEIGHT_GE_020 = 14.0
    USE_SMOOTH_TAIL_WEIGHTS = True
    TAIL_WEIGHT_TRANSITION_WIDTH = 0.003
    USE_TAIL_UNDERPREDICTION_LOSS = True
    TAIL_UNDERPREDICTION_START_EPOCH = 5
    TAIL_UNDERPREDICTION_RAMP_EPOCHS = 8
    TAIL_UNDERPREDICTION_THRESHOLD = 0.010
    TAIL_UNDERPREDICTION_WEIGHT = 0.18
    EXTREME_TAIL_UNDERPREDICTION_THRESHOLD = 0.020
    EXTREME_TAIL_UNDERPREDICTION_WEIGHT = 0.50
    TAIL_UNDERPREDICTION_MAX_LOSS = 2.2
    VAL_TAIL_THRESHOLD = 0.010
    VAL_EXTREME_TAIL_THRESHOLD = 0.020
    VAL_FOCUS_RMSE_WEIGHT = 0.18
    VAL_FOCUS_TAIL_MAE_WEIGHT = 0.25
    VAL_FOCUS_EXTREME_TAIL_MAE_WEIGHT = 0.45
    VAL_FOCUS_TAIL_UNDER_WEIGHT = 0.10
    VAL_FOCUS_EXTREME_TAIL_UNDER_WEIGHT = 0.20
    USE_EMA = False
    EMA_DECAY = 0.98

    USE_AMP = True
    CACHE_IMAGES = True
    USE_TRAIN_IMAGE_AUGMENTATION = True
    TIME_FREQ_MASK_PROB = 0.35
    TIME_MASK_COUNT = 1
    TIME_MASK_MAX_FRACTION = 0.08
    FREQ_MASK_COUNT = 1
    FREQ_MASK_MAX_FRACTION = 0.06
    TIME_FREQ_MASK_FILL_VALUE = 0.5

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 8

    USE_WEIGHTED_SAMPLER = True
    TARGET_BIN_COUNT = 8
    SAMPLER_POWER = 0.85
    SAMPLER_MIN_WEIGHT = 0.35
    SAMPLER_MAX_WEIGHT = 15.0
    SAMPLER_TAIL_BOOST_GE_005 = 1.0
    SAMPLER_TAIL_BOOST_GE_010 = 1.5
    SAMPLER_TAIL_BOOST_GE_020 = 4.0
    SAMPLER_NUM_SAMPLES_MULTIPLIER = 1.10

    CNN_BACKBONE = "scalar_film_residual"
    CNN_CHANNELS = [32, 64, 128, 192]
    CNN_KERNEL_SIZE = 3
    CNN_POOL_SIZES = [2, 2, 2, 2]
    CNN_NORM = "group"
    CNN_GROUP_NORM_MAX_GROUPS = 8
    CNN_DROPOUT = 0.08
    CNN_POOL_OUTPUT = (4, 4)
    CNN_PROJECTOR_DIM = 256
    CNN_PROJECTOR_DROPOUT = 0.10
    CNN_FILM_IDENTITY_INIT = True
    CNN_FILM_GATE_INIT_BIAS = 3.0

    MLP_HIDDEN_LAYERS = [64, 128]
    SCALAR_NORM = "layer"
    MLP_DROPOUT = 0.1

    HEAD_HIDDEN_DIMS = [256, 64]
    HEAD_DROPOUT = 0.20
    USE_TAIL_CORRECTION_HEAD = True
    USE_TAIL_CORRECTION_GATE = True
    TAIL_CORRECTION_HIDDEN_DIM = 64
    TAIL_CORRECTION_DROPOUT = 0.05
    TAIL_CORRECTION_INIT_BIAS = -4.0
    TAIL_CORRECTION_GATE_INIT_BIAS = -1.5
    USE_TAIL_CLASSIFICATION_AUX = True
    TAIL_CLASSIFICATION_THRESHOLDS = [0.010, 0.020]
    TAIL_CLASSIFICATION_LOSS_WEIGHTS = [0.025, 0.050]
    TAIL_CLASSIFICATION_HIDDEN_DIM = 64
    TAIL_CLASSIFICATION_DROPOUT = 0.05
    TAIL_CLASSIFICATION_INIT_BIAS = -3.0
    TAIL_CLASSIFICATION_POS_WEIGHT_MAX = 30.0
    TAIL_CLASSIFICATION_RAMP_EPOCHS = 8
    OPTIMIZER_NO_DECAY_NORM_AND_BIAS = True
    SAVE_ALTERNATE_BEST_CHECKPOINTS = True


Config.MODEL_ROOT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Running on device: {Config.DEVICE}")

if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def build_scalar_norm(num_features: int) -> nn.Module:
    norm_type = str(Config.SCALAR_NORM).lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(num_features)
    if norm_type == "layer":
        return nn.LayerNorm(num_features)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"不支持的标量归一化类型: {Config.SCALAR_NORM}")


def _resolve_group_count(num_channels: int, max_groups: int) -> int:
    max_groups = max(1, min(int(max_groups), int(num_channels)))
    for group_count in range(max_groups, 0, -1):
        if int(num_channels) % group_count == 0:
            return group_count
    return 1


def build_cnn_norm(num_channels: int) -> nn.Module:
    norm_type = str(Config.CNN_NORM).lower()
    if norm_type == "batch":
        return nn.BatchNorm2d(num_channels)
    if norm_type == "group":
        group_count = _resolve_group_count(num_channels, Config.CNN_GROUP_NORM_MAX_GROUPS)
        return nn.GroupNorm(group_count, num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm2d(num_channels, affine=True)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"不支持的 CNN 归一化类型: {Config.CNN_NORM}")


class RandomTimeFrequencyMask:
    def __init__(
        self,
        prob: float,
        time_mask_count: int,
        time_mask_max_fraction: float,
        freq_mask_count: int,
        freq_mask_max_fraction: float,
        fill_value: float,
    ):
        self.prob = float(prob)
        self.time_mask_count = int(time_mask_count)
        self.time_mask_max_fraction = float(time_mask_max_fraction)
        self.freq_mask_count = int(freq_mask_count)
        self.freq_mask_max_fraction = float(freq_mask_max_fraction)
        self.fill_value = float(fill_value)

    def _mask_axis(self, tensor: torch.Tensor, axis: int, count: int, max_fraction: float) -> None:
        if count <= 0 or max_fraction <= 0.0:
            return
        axis_size = int(tensor.shape[axis])
        max_width = max(1, int(round(axis_size * max_fraction)))
        for _ in range(count):
            width = int(torch.randint(1, max_width + 1, (1,)).item())
            if width >= axis_size:
                start = 0
                width = axis_size
            else:
                start = int(torch.randint(0, axis_size - width + 1, (1,)).item())
            if axis == 1:
                tensor[:, start : start + width, :] = self.fill_value
            elif axis == 2:
                tensor[:, :, start : start + width] = self.fill_value

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.prob <= 0.0 or float(torch.rand(()).item()) >= self.prob:
            return tensor
        tensor = tensor.clone()
        self._mask_axis(tensor, axis=2, count=self.time_mask_count, max_fraction=self.time_mask_max_fraction)
        self._mask_axis(tensor, axis=1, count=self.freq_mask_count, max_fraction=self.freq_mask_max_fraction)
        return tensor


def build_image_transform(train: bool) -> transforms.Compose:
    transform_steps = [
        transforms.Resize(Config.IMAGE_SIZE),
        transforms.ToTensor(),
    ]
    if train and Config.USE_TRAIN_IMAGE_AUGMENTATION:
        transform_steps.append(
            RandomTimeFrequencyMask(
                prob=Config.TIME_FREQ_MASK_PROB,
                time_mask_count=Config.TIME_MASK_COUNT,
                time_mask_max_fraction=Config.TIME_MASK_MAX_FRACTION,
                freq_mask_count=Config.FREQ_MASK_COUNT,
                freq_mask_max_fraction=Config.FREQ_MASK_MAX_FRACTION,
                fill_value=Config.TIME_FREQ_MASK_FILL_VALUE,
            )
        )
    transform_steps.append(
        transforms.Normalize(mean=Config.IMAGE_NORMALIZE_MEAN, std=Config.IMAGE_NORMALIZE_STD)
    )
    return transforms.Compose(transform_steps)


class MultimodalDataset(Dataset):
    def __init__(self, image_paths, scalar_data, labels, sample_weights=None, transform=None):
        self.image_paths = image_paths
        self.scalar_data = torch.tensor(scalar_data, dtype=torch.float32)
        target_values = labels * Config.LABEL_SCALE if Config.SCALE_TARGET else labels
        self.labels = torch.tensor(target_values, dtype=torch.float32)
        if sample_weights is None:
            sample_weights = np.ones(len(labels), dtype=np.float32)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self.transform = transform
        self._path_cache = {}

        if Config.CACHE_IMAGES:
            unique_paths = set(image_paths)
            print(f">>> Pre-loading {len(unique_paths)} unique images into memory (total rows: {len(image_paths)})...")
            for path in tqdm(unique_paths, desc="Caching images"):
                self._path_cache[path] = self._load_raw_image(path)
            print(">>> Image caching complete.")

        print(f">>> Dataset size: {len(self.image_paths)} samples")

    def _load_raw_image(self, img_path):
        try:
            return Image.open(img_path).convert("RGB")
        except Exception as exc:
            print(f"Warning: Error loading image {img_path}, using black image. Error: {exc}")
            return Image.new("RGB", Config.IMAGE_SIZE, (0, 0, 0))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = self._path_cache[img_path] if img_path in self._path_cache else self._load_raw_image(img_path)
        if self.transform:
            image = self.transform(image)
        scalars = self.scalar_data[idx]
        label = self.labels[idx]
        sample_weight = self.sample_weights[idx]
        return image, scalars, label, sample_weight


class ResidualConvBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int, dropout: float):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = build_cnn_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = build_cnn_norm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                build_cnn_norm(out_channels),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + identity)
        out = self.pool(out)
        out = self.dropout(out)
        return out


class ConditionalResidualConvBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int,
        dropout: float,
        scalar_dim: int,
    ):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = build_cnn_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = build_cnn_norm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout2d(dropout)
        self.film = nn.Linear(scalar_dim, out_channels * 2)
        self.gate = nn.Linear(scalar_dim, out_channels)

        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                build_cnn_norm(out_channels),
            )

        if Config.CNN_FILM_IDENTITY_INIT:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, Config.CNN_FILM_GATE_INIT_BIAS)

    def forward(self, x, scalar_features):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out + identity)

        gamma, beta = self.film(scalar_features).chunk(2, dim=1)
        gate = torch.sigmoid(self.gate(scalar_features))
        out = out * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        out = out * gate[:, :, None, None]

        out = self.pool(out)
        out = self.dropout(out)
        return out


class MultimodalPredictor(nn.Module):
    def __init__(self, num_scalars):
        super(MultimodalPredictor, self).__init__()

        c_ch = Config.CNN_CHANNELS
        mlp_h = Config.MLP_HIDDEN_LAYERS
        head_h = Config.HEAD_HIDDEN_DIMS

        self.mlp = nn.Sequential(
            nn.Linear(num_scalars, mlp_h[0]),
            build_scalar_norm(mlp_h[0]),
            nn.ReLU(),
            nn.Linear(mlp_h[0], mlp_h[1]),
            build_scalar_norm(mlp_h[1]),
            nn.ReLU(),
            nn.Dropout(Config.MLP_DROPOUT),
        )
        self.mlp_out_dim = mlp_h[1]
        self.uses_scalar_film = Config.CNN_BACKBONE == "scalar_film_residual"

        if Config.CNN_BACKBONE == "legacy":
            self.cnn = nn.Sequential(
                nn.Conv2d(3, c_ch[0], kernel_size=3, padding=1),
                build_cnn_norm(c_ch[0]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(Config.CNN_DROPOUT),

                nn.Conv2d(c_ch[0], c_ch[1], kernel_size=3, padding=1),
                build_cnn_norm(c_ch[1]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(Config.CNN_DROPOUT),

                nn.Conv2d(c_ch[1], c_ch[2], kernel_size=3, padding=1),
                build_cnn_norm(c_ch[2]),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(Config.CNN_DROPOUT),

                nn.AdaptiveAvgPool2d(Config.CNN_POOL_OUTPUT),
                nn.Flatten(),
            )
            pool_h, pool_w = Config.CNN_POOL_OUTPUT
            self.cnn_out_dim = c_ch[2] * pool_h * pool_w
        elif self.uses_scalar_film:
            self.cnn_stem = nn.Sequential(
                nn.Conv2d(
                    3,
                    c_ch[0],
                    kernel_size=Config.CNN_KERNEL_SIZE,
                    padding=Config.CNN_KERNEL_SIZE // 2,
                    bias=False,
                ),
                build_cnn_norm(c_ch[0]),
                nn.ReLU(inplace=True),
            )
            self.cnn_blocks = nn.ModuleList()
            in_channels = c_ch[0]
            for out_channels, pool_size in zip(c_ch, Config.CNN_POOL_SIZES):
                self.cnn_blocks.append(
                    ConditionalResidualConvBlock2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=Config.CNN_KERNEL_SIZE,
                        pool_size=pool_size,
                        dropout=Config.CNN_DROPOUT,
                        scalar_dim=self.mlp_out_dim,
                    )
                )
                in_channels = out_channels
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.cnn_out_dim = c_ch[-1] * 2
        else:
            blocks = [
                nn.Conv2d(3, c_ch[0], kernel_size=Config.CNN_KERNEL_SIZE, padding=Config.CNN_KERNEL_SIZE // 2, bias=False),
                build_cnn_norm(c_ch[0]),
                nn.ReLU(inplace=True),
            ]
            in_channels = c_ch[0]
            for out_channels, pool_size in zip(c_ch, Config.CNN_POOL_SIZES):
                blocks.append(
                    ResidualConvBlock2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=Config.CNN_KERNEL_SIZE,
                        pool_size=pool_size,
                        dropout=Config.CNN_DROPOUT,
                    )
                )
                in_channels = out_channels
            self.cnn = nn.Sequential(*blocks)
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.cnn_out_dim = c_ch[-1] * 2

        self.cnn_projector = nn.Sequential(
            nn.Linear(self.cnn_out_dim, Config.CNN_PROJECTOR_DIM),
            nn.ReLU(),
            nn.Dropout(Config.CNN_PROJECTOR_DROPOUT),
        )

        fusion_dim = Config.CNN_PROJECTOR_DIM + self.mlp_out_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, head_h[0]),
            nn.ReLU(),
            nn.Dropout(Config.HEAD_DROPOUT),
            nn.Linear(head_h[0], head_h[1]),
            nn.ReLU(),
            nn.Linear(head_h[1], 1),
        )
        self.use_tail_correction_head = Config.USE_TAIL_CORRECTION_HEAD
        if self.use_tail_correction_head:
            self.tail_correction_head = nn.Sequential(
                nn.Linear(fusion_dim, Config.TAIL_CORRECTION_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(Config.TAIL_CORRECTION_DROPOUT),
                nn.Linear(Config.TAIL_CORRECTION_HIDDEN_DIM, 1),
            )
            self.tail_correction_activation = nn.Softplus()
            nn.init.constant_(self.tail_correction_head[-1].bias, Config.TAIL_CORRECTION_INIT_BIAS)
            self.use_tail_correction_gate = Config.USE_TAIL_CORRECTION_GATE
            if self.use_tail_correction_gate:
                self.tail_correction_gate = nn.Linear(fusion_dim, 1)
                nn.init.zeros_(self.tail_correction_gate.weight)
                nn.init.constant_(self.tail_correction_gate.bias, Config.TAIL_CORRECTION_GATE_INIT_BIAS)
        self.use_tail_classification_aux = Config.USE_TAIL_CLASSIFICATION_AUX
        if self.use_tail_classification_aux:
            self.tail_classifier_head = nn.Sequential(
                nn.Linear(fusion_dim, Config.TAIL_CLASSIFICATION_HIDDEN_DIM),
                nn.ReLU(),
                nn.Dropout(Config.TAIL_CLASSIFICATION_DROPOUT),
                nn.Linear(Config.TAIL_CLASSIFICATION_HIDDEN_DIM, len(Config.TAIL_CLASSIFICATION_THRESHOLDS)),
            )
            nn.init.constant_(self.tail_classifier_head[-1].bias, Config.TAIL_CLASSIFICATION_INIT_BIAS)

    def forward(self, img, scalars, return_aux: bool = False):
        x_scalar = self.mlp(scalars)
        if self.uses_scalar_film:
            x_img = self.cnn_stem(img)
            for block in self.cnn_blocks:
                x_img = block(x_img, x_scalar)
            avg_features = self.avg_pool(x_img).flatten(1)
            max_features = self.max_pool(x_img).flatten(1)
            x_img = torch.cat((avg_features, max_features), dim=1)
        else:
            x_img = self.cnn(img)
        if Config.CNN_BACKBONE != "legacy" and not self.uses_scalar_film:
            avg_features = self.avg_pool(x_img).flatten(1)
            max_features = self.max_pool(x_img).flatten(1)
            x_img = torch.cat((avg_features, max_features), dim=1)
        x_img = self.cnn_projector(x_img)
        x_fused = torch.cat((x_img, x_scalar), dim=1)
        base_pred = self.head(x_fused)
        aux_outputs = {}
        if self.use_tail_classification_aux:
            aux_outputs["tail_logits"] = self.tail_classifier_head(x_fused)
        if self.use_tail_correction_head:
            tail_correction = self.tail_correction_activation(self.tail_correction_head(x_fused))
            if self.use_tail_correction_gate:
                tail_correction = tail_correction * torch.sigmoid(self.tail_correction_gate(x_fused))
            pred = base_pred + tail_correction
        else:
            pred = base_pred
        if return_aux:
            return pred, aux_outputs
        return pred


def resolve_wavelet_image_path(raw_value) -> str:
    raw_path = Path(str(raw_value))
    if Config.FORCE_WAVELET_IMAGE_DIR:
        return str(Config.WAVELET_IMAGE_DIR / raw_path.name)
    return resolve_image_path(raw_value, Config.WAVELET_IMAGE_DIR)


def build_image_paths(values):
    image_paths: list[str] = []
    missing_paths: list[str] = []

    for value in values:
        image_path = Path(resolve_wavelet_image_path(value))
        image_paths.append(str(image_path))
        if not image_path.exists():
            missing_paths.append(str(image_path))

    if missing_paths:
        preview = "\n".join(f"  - {path}" for path in missing_paths[:5])
        raise FileNotFoundError(
            f"小波图目录缺少 {len(missing_paths)} 个图片，当前目录: {Config.WAVELET_IMAGE_DIR}\n"
            f"{preview}"
        )

    return np.array(image_paths, dtype=object)


def get_existing_dataset_dirs() -> list[Path]:
    existing_dirs = [path for path in Config.CSV_DIR_CANDIDATES if path.exists()]
    return existing_dirs or [Config.CSV_DIR]


def format_dataset_dir_text() -> str:
    return ", ".join(str(path) for path in get_existing_dataset_dirs())


def resolve_explicit_csv_path(explicit_path: str | Path) -> Path:
    path = Path(explicit_path)
    candidate_paths: list[Path] = []

    if path.is_absolute():
        candidate_paths.append(path)
    else:
        candidate_paths.append(Config.PROJECT_ROOT / path)
        candidate_paths.extend(dataset_dir / path for dataset_dir in get_existing_dataset_dirs())

    seen: set[Path] = set()
    ordered_candidates: list[Path] = []
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered_candidates.append(candidate)

    for candidate in ordered_candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in ordered_candidates)
    raise FileNotFoundError(f"显式指定的 CSV 文件不存在: {explicit_path}；已检查: {searched}")


def resolve_csv_path(explicit_path: str | Path | None, pattern: str) -> Path:
    if explicit_path:
        return resolve_explicit_csv_path(explicit_path)

    candidates: list[Path] = []
    for dataset_dir in get_existing_dataset_dirs():
        candidates.extend(dataset_dir.glob(pattern))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"在以下目录中未找到匹配 '{pattern}' 的文件: {format_dataset_dir_text()}")
    return candidates[0]


def dataset_base_name_from_csv(csv_path: Path) -> str:
    for suffix in ("_train.csv", "_val.csv", "_test.csv"):
        if csv_path.name.endswith(suffix):
            return csv_path.name[: -len(suffix)]
    raise ValueError(f"无法从文件名解析数据集基名: {csv_path.name}")


def build_compact_artifact_name(dataset_base: str, prefix: str) -> str:
    stamp_match = re.search(r"(\d{8}-\d{6})$", str(dataset_base))
    stamp = stamp_match.group(1) if stamp_match else "nostamp"
    digest = hashlib.sha1(str(dataset_base).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{stamp}-{digest}"


def resolve_latest_complete_dataset_paths() -> tuple[Path, Path, str]:
    candidates: list[tuple[float, Path, Path, str]] = []
    incomplete_bases: list[str] = []

    for dataset_dir in get_existing_dataset_dirs():
        for train_csv_path in dataset_dir.glob(Config.TRAIN_FILE_PATTERN):
            dataset_base = dataset_base_name_from_csv(train_csv_path)
            val_csv_path = train_csv_path.with_name(f"{dataset_base}_val.csv")
            if not val_csv_path.exists():
                incomplete_bases.append(str(train_csv_path))
                continue

            candidate_mtime = max(train_csv_path.stat().st_mtime, val_csv_path.stat().st_mtime)
            summary_path = train_csv_path.with_name(f"{dataset_base}_summary.json")
            if summary_path.exists():
                candidate_mtime = max(candidate_mtime, summary_path.stat().st_mtime)

            candidates.append((candidate_mtime, train_csv_path, val_csv_path, dataset_base))

    if not candidates:
        if incomplete_bases:
            bases_text = ", ".join(sorted(set(incomplete_bases)))
            raise FileNotFoundError(
                "未找到包含配套 train / val 文件的完整数据集运行。"
                f"以下 train 文件缺少对应 val 文件: {bases_text}"
            )
        raise FileNotFoundError(
            f"在以下目录中未找到匹配 '{Config.TRAIN_FILE_PATTERN}' 的训练集文件: {format_dataset_dir_text()}"
        )

    _, train_csv_path, val_csv_path, dataset_base = max(candidates, key=lambda item: item[0])
    return train_csv_path, val_csv_path, dataset_base


def resolve_dataset_paths() -> tuple[Path, Path, str]:
    if Config.TRAIN_CSV_PATH and Config.VAL_CSV_PATH:
        train_csv_path = resolve_csv_path(Config.TRAIN_CSV_PATH, Config.TRAIN_FILE_PATTERN)
        val_csv_path = resolve_csv_path(Config.VAL_CSV_PATH, Config.VAL_FILE_PATTERN)
        train_base = dataset_base_name_from_csv(train_csv_path)
        val_base = dataset_base_name_from_csv(val_csv_path)
        if train_base != val_base:
            raise ValueError(
                "显式指定的 train / val CSV 不属于同一数据集运行："
                f"train={train_csv_path.name}, val={val_csv_path.name}"
            )
        return train_csv_path, val_csv_path, train_base

    if Config.TRAIN_CSV_PATH and not Config.VAL_CSV_PATH:
        train_csv_path = resolve_csv_path(Config.TRAIN_CSV_PATH, Config.TRAIN_FILE_PATTERN)
        dataset_base = dataset_base_name_from_csv(train_csv_path)
        val_csv_path = train_csv_path.with_name(f"{dataset_base}_val.csv")
        if not val_csv_path.exists():
            raise FileNotFoundError(f"未找到与 train 配套的 val 文件: {val_csv_path}")
        return train_csv_path, val_csv_path, dataset_base

    if Config.VAL_CSV_PATH and not Config.TRAIN_CSV_PATH:
        val_csv_path = resolve_csv_path(Config.VAL_CSV_PATH, Config.VAL_FILE_PATTERN)
        dataset_base = dataset_base_name_from_csv(val_csv_path)
        train_csv_path = val_csv_path.with_name(f"{dataset_base}_train.csv")
        if not train_csv_path.exists():
            raise FileNotFoundError(f"未找到与 val 配套的 train 文件: {train_csv_path}")
        return train_csv_path, val_csv_path, dataset_base

    return resolve_latest_complete_dataset_paths()


def configure_model_dir(dataset_base: str) -> Path:
    base_model_run_name = build_compact_artifact_name(dataset_base, prefix="model")
    if Config.UNIQUE_MODEL_RUN_DIR:
        model_run_name = f"{base_model_run_name}-train{time.strftime('%Y%m%d-%H%M%S')}"
    else:
        model_run_name = base_model_run_name
    model_dir = Config.MODEL_ROOT_DIR / model_run_name
    model_dir.mkdir(parents=True, exist_ok=True)
    Config.MODEL_DIR = model_dir
    Config.SAVE_DIR = model_dir
    return model_dir


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_amp_autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type="cuda", enabled=True)


def build_grad_scaler(enabled: bool):
    return torch.amp.GradScaler(device=Config.DEVICE.type, enabled=enabled)


def build_adamw_optimizer(model: nn.Module) -> optim.Optimizer:
    if not Config.OPTIMIZER_NO_DECAY_NORM_AND_BIAS:
        return optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for param_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower_name = param_name.lower()
        is_norm_or_bias = (
            param.ndim <= 1
            or lower_name.endswith(".bias")
            or ".bn" in lower_name
            or "norm" in lower_name
            or "layernorm" in lower_name
            or "groupnorm" in lower_name
        )
        if is_norm_or_bias:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {"params": decay_params, "weight_decay": Config.WEIGHT_DECAY},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return optim.AdamW(param_groups, lr=Config.LEARNING_RATE)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                tensor = tensor.detach()
                shadow_tensor = self.shadow[name]
                if torch.is_floating_point(shadow_tensor):
                    shadow_tensor.mul_(self.decay).add_(tensor, alpha=1.0 - self.decay)
                else:
                    shadow_tensor.copy_(tensor)

    def store(self, model: nn.Module) -> None:
        self.backup = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: tensor.detach().clone()
            for name, tensor in self.shadow.items()
        }


def apply_lr_warmup(optimizer: optim.Optimizer, epoch_idx: int) -> None:
    if Config.WARMUP_EPOCHS <= 0 or epoch_idx >= Config.WARMUP_EPOCHS:
        return

    progress = float(epoch_idx + 1) / float(Config.WARMUP_EPOCHS)
    factor = Config.WARMUP_START_FACTOR + (1.0 - Config.WARMUP_START_FACTOR) * progress
    for param_group in optimizer.param_groups:
        param_group["lr"] = Config.LEARNING_RATE * factor


def save_training_metadata(metadata: dict) -> None:
    path = Config.MODEL_DIR / "training_metadata.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f">>> Training metadata saved to {path}")


def validate_dataframe(df: pd.DataFrame, csv_path: Path) -> None:
    required_cols = Config.BASE_SCALAR_COLS + [Config.IMAGE_COL, Config.TXT_COL, Config.LABEL_COL, Config.STATUS_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV 文件缺少必要列: {missing_cols}，源文件: {csv_path}")


def filter_valid_rows(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df[Config.STATUS_COL] == "ok"].copy()
    filtered = filtered.dropna(subset=Config.BASE_SCALAR_COLS + [Config.IMAGE_COL, Config.TXT_COL, Config.LABEL_COL])
    return filtered.reset_index(drop=True)


def parse_damper_layout_flags(layout_value, num_floors: int, layout_width: int) -> list[int]:
    floor_count = max(0, min(int(num_floors), int(layout_width)))
    flags = [0] * int(layout_width)
    if floor_count <= 0:
        return flags

    if layout_value is None or (isinstance(layout_value, float) and np.isnan(layout_value)):
        parsed_bits = [1] * floor_count
    else:
        text = str(layout_value).strip()
        bits = re.findall(r"[01]", text)
        if bits:
            parsed_bits = [int(bit) for bit in bits[:floor_count]]
            if len(parsed_bits) < floor_count:
                parsed_bits.extend([1] * (floor_count - len(parsed_bits)))
        else:
            parsed_bits = [1] * floor_count

    flags[:floor_count] = parsed_bits[:floor_count]
    return flags


def get_available_wave_feature_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in Config.WAVE_FEATURE_COLS if col in df.columns]


def _finite_series(values, default: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(array, nan=default, posinf=default, neginf=default)


def add_wave_derived_features(feature_df: pd.DataFrame) -> list[str]:
    if not Config.USE_WAVE_DERIVED_FEATURES:
        return []

    derived_feature_names: list[str] = []
    eps = 1e-6

    for col in Config.WAVE_LOG_FEATURE_COLS:
        if col not in feature_df.columns:
            continue
        values = np.maximum(_finite_series(feature_df[col]), 0.0)
        feature_name = f"log1p_{col}"
        feature_df[feature_name] = np.log1p(values).astype(np.float32)
        derived_feature_names.append(feature_name)

    if "period_1_sec" not in feature_df.columns:
        return derived_feature_names

    period = np.clip(_finite_series(feature_df["period_1_sec"], default=eps), eps, None)

    if "wave_predominant_period" in feature_df.columns:
        wave_period = np.clip(_finite_series(feature_df["wave_predominant_period"], default=eps), eps, None)
        ratio = np.clip(wave_period / period, eps, 20.0)
        inverse_ratio = np.clip(period / wave_period, eps, 20.0)
        feature_df["wave_to_structure_period_ratio"] = ratio.astype(np.float32)
        feature_df["structure_to_wave_period_ratio"] = inverse_ratio.astype(np.float32)
        feature_df["wave_structure_period_log_gap"] = np.abs(np.log(ratio)).astype(np.float32)
        derived_feature_names.extend(
            [
                "wave_to_structure_period_ratio",
                "structure_to_wave_period_ratio",
                "wave_structure_period_log_gap",
            ]
        )

    if "wave_dominant_freq" in feature_df.columns:
        dominant_freq = np.maximum(_finite_series(feature_df["wave_dominant_freq"]), 0.0)
        feature_df["wave_dominant_freq_x_period"] = np.clip(dominant_freq * period, 0.0, 20.0).astype(np.float32)
        derived_feature_names.append("wave_dominant_freq_x_period")

    if "wave_spectral_centroid" in feature_df.columns:
        centroid = np.maximum(_finite_series(feature_df["wave_spectral_centroid"]), 0.0)
        feature_df["wave_spectral_centroid_x_period"] = np.clip(centroid * period, 0.0, 20.0).astype(np.float32)
        derived_feature_names.append("wave_spectral_centroid_x_period")

    for col in ("wave_pga", "wave_cav", "wave_arias_proxy", "wave_intensity_score"):
        if col not in feature_df.columns:
            continue
        values = np.maximum(_finite_series(feature_df[col]), 0.0)
        feature_name = f"{col}_x_period"
        feature_df[feature_name] = (values * period).astype(np.float32)
        derived_feature_names.append(feature_name)

    return derived_feature_names


def add_structure_derived_features(feature_df: pd.DataFrame) -> list[str]:
    eps = 1e-6
    derived_feature_names: list[str] = []
    floors = np.maximum(_finite_series(feature_df["num_floors"], default=1.0), 1.0)
    mass = np.maximum(_finite_series(feature_df["floor_mass"]), eps)
    height = np.maximum(_finite_series(feature_df["floor_height"]), eps)
    stiffness = np.maximum(_finite_series(feature_df["k_base_1_4"]), eps)
    fy_add = np.maximum(_finite_series(feature_df["Fy_add"]), eps)
    period = np.maximum(_finite_series(feature_df.get("period_1_sec", np.zeros(len(feature_df)))), eps)

    feature_values = {
        "inv_k_base_1_4": 1.0 / stiffness,
        "inv_Fy_add": 1.0 / fy_add,
        "mass_to_stiffness": mass / stiffness,
        "height_to_stiffness": height / stiffness,
        "period_squared": period ** 2,
        "num_floors_x_period": floors * period,
        "floor_mass_x_period": mass * period,
        "floor_height_x_period": height * period,
        "flexibility_x_period": (mass / stiffness) * period,
        "strength_inverse_x_period": (1.0 / fy_add) * period,
    }

    for feature_name, values in feature_values.items():
        feature_df[feature_name] = values.astype(np.float32)
        derived_feature_names.append(feature_name)

    return derived_feature_names


def add_tail_risk_features(feature_df: pd.DataFrame) -> list[str]:
    eps = 1e-6
    floors = np.maximum(_finite_series(feature_df["num_floors"], default=1.0), 1.0)
    mass = np.maximum(_finite_series(feature_df["floor_mass"]), eps)
    stiffness = np.maximum(_finite_series(feature_df["k_base_1_4"]), eps)
    period = np.maximum(_finite_series(feature_df.get("period_1_sec", np.zeros(len(feature_df)))), eps)
    intensity = np.maximum(_finite_series(feature_df.get("wave_intensity_score", np.zeros(len(feature_df)))), 0.0)
    arias = np.maximum(_finite_series(feature_df.get("wave_arias_proxy", np.zeros(len(feature_df)))), 0.0)
    cav = np.maximum(_finite_series(feature_df.get("wave_cav", np.zeros(len(feature_df)))), 0.0)
    sparse_ratio = np.maximum(_finite_series(feature_df["damper_sparse_ratio"]), 0.0)
    flexibility = mass / stiffness

    tail_risk = period * floors * flexibility * (1.0 + sparse_ratio)
    intensity_tail_risk = tail_risk * (1.0 + intensity)
    arias_tail_risk = tail_risk * (1.0 + arias)
    cav_tail_risk = tail_risk * (1.0 + cav)
    feature_values = {
        "tail_risk_proxy": tail_risk,
        "log1p_tail_risk_proxy": np.log1p(np.maximum(tail_risk, 0.0)),
        "wave_intensity_tail_risk": intensity_tail_risk,
        "wave_arias_tail_risk": arias_tail_risk,
        "wave_cav_tail_risk": cav_tail_risk,
    }

    derived_feature_names: list[str] = []
    for feature_name, values in feature_values.items():
        feature_df[feature_name] = values.astype(np.float32)
        derived_feature_names.append(feature_name)

    return derived_feature_names


def build_scalar_feature_frame(df: pd.DataFrame, layout_width: int) -> tuple[pd.DataFrame, list[str]]:
    wave_feature_cols = get_available_wave_feature_cols(df)
    feature_df = df[Config.BASE_SCALAR_COLS + wave_feature_cols].copy()
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        if feature_df[col].isna().any():
            median_value = feature_df[col].median()
            if pd.isna(median_value):
                median_value = 0.0
            feature_df[col] = feature_df[col].fillna(median_value)
    derived_feature_names = add_wave_derived_features(feature_df)
    structure_derived_feature_names = add_structure_derived_features(feature_df)
    layout_feature_names = [f"damper_story_{idx}" for idx in range(1, int(layout_width) + 1)]
    layout_matrix = np.zeros((len(df), int(layout_width)), dtype=np.float32)

    for row_idx, row in enumerate(df.itertuples(index=False)):
        num_floors = int(getattr(row, "num_floors"))
        layout_value = getattr(row, Config.DAMPER_LAYOUT_COL, None) if hasattr(row, Config.DAMPER_LAYOUT_COL) else None
        layout_matrix[row_idx, :] = parse_damper_layout_flags(layout_value, num_floors, layout_width)

    for col_idx, feature_name in enumerate(layout_feature_names):
        feature_df[feature_name] = layout_matrix[:, col_idx]

    feature_df["damper_install_count"] = layout_matrix.sum(axis=1)
    feature_df["damper_install_ratio"] = np.divide(
        feature_df["damper_install_count"].astype(np.float32),
        np.maximum(feature_df["num_floors"].astype(np.float32), 1.0),
    )
    feature_df["damper_sparse_ratio"] = 1.0 - feature_df["damper_install_ratio"].astype(np.float32)
    tail_risk_feature_names = add_tail_risk_features(feature_df)

    scalar_feature_names = (
        Config.BASE_SCALAR_COLS
        + wave_feature_cols
        + derived_feature_names
        + structure_derived_feature_names
        + layout_feature_names
        + [
            "damper_install_count",
            "damper_install_ratio",
            "damper_sparse_ratio",
        ]
        + tail_risk_feature_names
    )
    return feature_df[scalar_feature_names], scalar_feature_names


def _build_target_bin_series(labels: pd.Series) -> pd.Series:
    unique_count = int(labels.nunique())
    if unique_count <= 1:
        return pd.Series(np.zeros(len(labels), dtype=int), index=labels.index)

    q = min(Config.TARGET_BIN_COUNT, unique_count)
    try:
        bins = pd.qcut(labels, q=q, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(np.zeros(len(labels), dtype=int), index=labels.index)

    return pd.Series(np.asarray(bins, dtype=np.int64), index=labels.index)


def _gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    sigma = max(float(sigma), 1e-6)
    offsets = np.arange(size, dtype=np.float64) - size // 2
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel_sum = float(kernel.sum())
    return kernel / max(kernel_sum, 1e-12)


def _smooth_tail_boost_np(values: np.ndarray, threshold: float, target_weight: float) -> np.ndarray:
    if not Config.USE_SMOOTH_TAIL_WEIGHTS:
        return np.where(values >= threshold, target_weight, 1.0)
    width = max(float(Config.TAIL_WEIGHT_TRANSITION_WIDTH), 1e-8)
    progress = np.clip((values - (threshold - width)) / width, 0.0, 1.0)
    progress = progress * progress * (3.0 - 2.0 * progress)
    return 1.0 + (float(target_weight) - 1.0) * progress


def _smooth_tail_boost_torch(labels_raw: torch.Tensor, threshold: float, target_weight: float) -> torch.Tensor:
    if not Config.USE_SMOOTH_TAIL_WEIGHTS:
        return torch.where(
            labels_raw >= threshold,
            torch.full_like(labels_raw, float(target_weight)),
            torch.ones_like(labels_raw),
        )
    width = max(float(Config.TAIL_WEIGHT_TRANSITION_WIDTH), 1e-8)
    progress = torch.clamp((labels_raw - (threshold - width)) / width, 0.0, 1.0)
    progress = progress * progress * (3.0 - 2.0 * progress)
    return 1.0 + (float(target_weight) - 1.0) * progress


def build_train_sampler(train_df: pd.DataFrame):
    if not Config.USE_WEIGHTED_SAMPLER:
        return None, {"enabled": False}

    floor_series = train_df["num_floors"].round().astype(int).astype(str)
    labels = train_df[Config.LABEL_COL].astype(float)
    target_bins = _build_target_bin_series(labels)
    balance_keys = floor_series + "_bin" + target_bins.astype(str)

    counts = balance_keys.value_counts()
    median_count = float(counts.median())
    weights = balance_keys.map(lambda key: (median_count / counts[key]) ** Config.SAMPLER_POWER)
    weights = weights.to_numpy(dtype=np.float64).copy()

    label_values = labels.to_numpy(dtype=np.float64)
    weights *= _smooth_tail_boost_np(label_values, 0.005, Config.SAMPLER_TAIL_BOOST_GE_005)
    weights *= _smooth_tail_boost_np(label_values, 0.010, Config.SAMPLER_TAIL_BOOST_GE_010)
    weights *= _smooth_tail_boost_np(label_values, 0.020, Config.SAMPLER_TAIL_BOOST_GE_020)
    weights = np.clip(weights, Config.SAMPLER_MIN_WEIGHT, Config.SAMPLER_MAX_WEIGHT)
    sampler_num_samples = max(1, int(round(len(weights) * Config.SAMPLER_NUM_SAMPLES_MULTIPLIER)))

    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=sampler_num_samples,
        replacement=True,
    )

    sampler_summary = {
        "enabled": True,
        "target_bin_count": int(target_bins.nunique()),
        "joint_group_count": int(balance_keys.nunique()),
        "num_samples": int(sampler_num_samples),
        "num_samples_multiplier": float(Config.SAMPLER_NUM_SAMPLES_MULTIPLIER),
        "tail_count_ge_0p005": int(np.sum(label_values >= 0.005)),
        "tail_count_ge_0p010": int(np.sum(label_values >= 0.010)),
        "tail_count_ge_0p020": int(np.sum(label_values >= 0.020)),
        "tail_boost_ge_0p005": float(Config.SAMPLER_TAIL_BOOST_GE_005),
        "tail_boost_ge_0p010": float(Config.SAMPLER_TAIL_BOOST_GE_010),
        "tail_boost_ge_0p020": float(Config.SAMPLER_TAIL_BOOST_GE_020),
        "smooth_tail_weights": bool(Config.USE_SMOOTH_TAIL_WEIGHTS),
        "tail_weight_transition_width": float(Config.TAIL_WEIGHT_TRANSITION_WIDTH),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }
    return sampler, sampler_summary


def build_label_density_weights(labels_raw: np.ndarray) -> np.ndarray:
    labels_raw = np.asarray(labels_raw, dtype=np.float64)
    if labels_raw.size == 0 or not Config.USE_DENSITY_WEIGHTED_LOSS:
        return np.ones(labels_raw.shape, dtype=np.float32)

    label_space = np.log1p(np.maximum(labels_raw, 0.0) * Config.LABEL_SCALE)
    unique_count = int(np.unique(label_space).size)
    bin_count = max(2, min(Config.LOSS_DENSITY_BIN_COUNT, unique_count))
    counts, bin_edges = np.histogram(label_space, bins=bin_count)
    counts = counts.astype(np.float64)
    density_kernel = _gaussian_kernel(Config.LOSS_DENSITY_SMOOTH_KERNEL_SIZE, Config.LOSS_DENSITY_SMOOTH_SIGMA)
    pad_width = int(len(density_kernel) // 2)
    padded_counts = np.pad(counts, pad_width=pad_width, mode="edge")
    smoothed_counts = np.convolve(padded_counts, density_kernel, mode="valid")
    smoothed_counts = np.maximum(smoothed_counts, 1.0)

    bin_indices = np.searchsorted(bin_edges[1:-1], label_space, side="right")
    median_density = float(np.median(smoothed_counts))
    weights = (median_density / smoothed_counts[bin_indices]) ** Config.LOSS_DENSITY_ALPHA
    weights = np.clip(weights, Config.LOSS_WEIGHT_MIN, Config.LOSS_WEIGHT_MAX)
    weights = weights / max(float(np.mean(weights)), 1e-8)
    weights = np.clip(weights, Config.LOSS_WEIGHT_MIN, Config.LOSS_WEIGHT_MAX)
    return weights.astype(np.float32)


def build_tail_classification_summary(labels_raw: np.ndarray) -> dict:
    labels_raw = np.asarray(labels_raw, dtype=np.float64)
    thresholds = [float(threshold) for threshold in Config.TAIL_CLASSIFICATION_THRESHOLDS]
    total_count = int(labels_raw.size)
    positive_counts: list[int] = []
    pos_weights: list[float] = []
    for threshold in thresholds:
        positive_count = int(np.sum(labels_raw >= threshold))
        negative_count = max(0, total_count - positive_count)
        positive_counts.append(positive_count)
        if positive_count <= 0:
            pos_weight = 1.0
        else:
            pos_weight = np.sqrt(float(negative_count) / float(positive_count))
        pos_weights.append(float(np.clip(pos_weight, 1.0, Config.TAIL_CLASSIFICATION_POS_WEIGHT_MAX)))

    return {
        "enabled": bool(Config.USE_TAIL_CLASSIFICATION_AUX),
        "thresholds": thresholds,
        "loss_weights": [float(value) for value in Config.TAIL_CLASSIFICATION_LOSS_WEIGHTS],
        "positive_counts": positive_counts,
        "total_count": total_count,
        "pos_weights": pos_weights,
        "pos_weight_max": float(Config.TAIL_CLASSIFICATION_POS_WEIGHT_MAX),
        "ramp_epochs": int(Config.TAIL_CLASSIFICATION_RAMP_EPOCHS),
    }


def apply_tail_loss_multipliers(labels_scaled: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if not Config.USE_TARGET_WEIGHTED_LOSS:
        return weights

    labels_raw = labels_scaled / float(Config.LABEL_SCALE) if Config.SCALE_TARGET else labels_scaled
    weights = torch.maximum(weights, _smooth_tail_boost_torch(labels_raw, 0.005, Config.LOSS_WEIGHT_GE_005))
    weights = torch.maximum(weights, _smooth_tail_boost_torch(labels_raw, 0.010, Config.LOSS_WEIGHT_GE_010))
    weights = torch.maximum(weights, _smooth_tail_boost_torch(labels_raw, 0.020, Config.LOSS_WEIGHT_GE_020))
    weights = torch.clamp(weights, min=Config.LOSS_WEIGHT_MIN, max=Config.LOSS_WEIGHT_MAX)
    return weights


def calculate_tail_classification_loss(
    aux_outputs: dict,
    labels_scaled: torch.Tensor,
    pos_weights: torch.Tensor,
    epoch_num: int,
) -> torch.Tensor:
    if not Config.USE_TAIL_CLASSIFICATION_AUX:
        return labels_scaled.new_zeros(())
    tail_logits = aux_outputs.get("tail_logits")
    if tail_logits is None:
        return labels_scaled.new_zeros(())

    labels_raw = labels_scaled / float(Config.LABEL_SCALE) if Config.SCALE_TARGET else labels_scaled
    thresholds = torch.tensor(
        Config.TAIL_CLASSIFICATION_THRESHOLDS,
        dtype=labels_raw.dtype,
        device=labels_raw.device,
    ).view(1, -1)
    targets = (labels_raw >= thresholds).to(dtype=tail_logits.dtype)
    loss_weights = torch.tensor(
        Config.TAIL_CLASSIFICATION_LOSS_WEIGHTS,
        dtype=tail_logits.dtype,
        device=tail_logits.device,
    ).view(1, -1)
    pos_weights = pos_weights.to(dtype=tail_logits.dtype, device=tail_logits.device).view(1, -1)
    bce = nn.functional.binary_cross_entropy_with_logits(
        tail_logits,
        targets,
        pos_weight=pos_weights,
        reduction="none",
    )
    ramp_epochs = max(int(Config.TAIL_CLASSIFICATION_RAMP_EPOCHS), 1)
    ramp_factor = min(1.0, max(0.0, float(epoch_num) / float(ramp_epochs)))
    return ramp_factor * torch.mean(bce * loss_weights)


def calculate_tail_underprediction_loss(
    preds: torch.Tensor,
    labels_scaled: torch.Tensor,
    epoch_num: int | None = None,
) -> torch.Tensor:
    if not Config.USE_TAIL_UNDERPREDICTION_LOSS:
        return preds.new_zeros(())
    if epoch_num is not None and epoch_num < Config.TAIL_UNDERPREDICTION_START_EPOCH:
        return preds.new_zeros(())

    labels_raw = labels_scaled / float(Config.LABEL_SCALE) if Config.SCALE_TARGET else labels_scaled
    under_error = torch.relu(labels_scaled - preds)
    total_loss = preds.new_zeros(())
    if epoch_num is None:
        ramp_factor = 1.0
    else:
        ramp_epochs = max(int(Config.TAIL_UNDERPREDICTION_RAMP_EPOCHS), 1)
        ramp_factor = min(1.0, max(0.0, (epoch_num - Config.TAIL_UNDERPREDICTION_START_EPOCH + 1) / ramp_epochs))

    tail_mask = labels_raw >= Config.TAIL_UNDERPREDICTION_THRESHOLD
    if torch.any(tail_mask):
        tail_errors = under_error[tail_mask]
        total_loss = total_loss + ramp_factor * Config.TAIL_UNDERPREDICTION_WEIGHT * nn.functional.smooth_l1_loss(
            tail_errors,
            torch.zeros_like(tail_errors),
            beta=Config.SMOOTH_L1_BETA,
            reduction="mean",
        )

    extreme_mask = labels_raw >= Config.EXTREME_TAIL_UNDERPREDICTION_THRESHOLD
    if torch.any(extreme_mask):
        extreme_errors = under_error[extreme_mask]
        total_loss = total_loss + ramp_factor * Config.EXTREME_TAIL_UNDERPREDICTION_WEIGHT * nn.functional.smooth_l1_loss(
            extreme_errors,
            torch.zeros_like(extreme_errors),
            beta=Config.SMOOTH_L1_BETA,
            reduction="mean",
        )

    return torch.clamp(total_loss, max=Config.TAIL_UNDERPREDICTION_MAX_LOSS)


def calculate_regression_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, float]:
    errors = y_pred_raw - y_true_raw
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true_raw - np.mean(y_true_raw)) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def calculate_validation_focus_metrics(y_true_raw: np.ndarray, y_pred_raw: np.ndarray) -> dict[str, float]:
    errors = y_pred_raw - y_true_raw
    under_errors = np.maximum(y_true_raw - y_pred_raw, 0.0)
    tail_mask = y_true_raw >= Config.VAL_TAIL_THRESHOLD
    extreme_tail_mask = y_true_raw >= Config.VAL_EXTREME_TAIL_THRESHOLD
    if np.any(tail_mask):
        tail_errors = errors[tail_mask]
        tail_mae = float(np.mean(np.abs(tail_errors)))
        tail_bias = float(np.mean(tail_errors))
        tail_under_mae = float(np.mean(under_errors[tail_mask]))
    else:
        tail_mae = float(np.mean(np.abs(errors)))
        tail_bias = float(np.mean(errors))
        tail_under_mae = float(np.mean(under_errors))

    if np.any(extreme_tail_mask):
        extreme_tail_errors = errors[extreme_tail_mask]
        extreme_tail_mae = float(np.mean(np.abs(extreme_tail_errors)))
        extreme_tail_bias = float(np.mean(extreme_tail_errors))
        extreme_tail_under_mae = float(np.mean(under_errors[extreme_tail_mask]))
    else:
        extreme_tail_mae = tail_mae
        extreme_tail_bias = tail_bias
        extreme_tail_under_mae = tail_under_mae

    metrics = calculate_regression_metrics(y_true_raw, y_pred_raw)
    focus_score = (
        metrics["mae"]
        + Config.VAL_FOCUS_RMSE_WEIGHT * metrics["rmse"]
        + Config.VAL_FOCUS_TAIL_MAE_WEIGHT * tail_mae
        + Config.VAL_FOCUS_EXTREME_TAIL_MAE_WEIGHT * extreme_tail_mae
        + Config.VAL_FOCUS_TAIL_UNDER_WEIGHT * tail_under_mae
        + Config.VAL_FOCUS_EXTREME_TAIL_UNDER_WEIGHT * extreme_tail_under_mae
    )
    return {
        "focus_score": float(focus_score),
        "tail_mae": float(tail_mae),
        "tail_bias": float(tail_bias),
        "tail_under_mae": float(tail_under_mae),
        "tail_count": int(np.sum(tail_mask)),
        "extreme_tail_mae": float(extreme_tail_mae),
        "extreme_tail_bias": float(extreme_tail_bias),
        "extreme_tail_under_mae": float(extreme_tail_under_mae),
        "extreme_tail_count": int(np.sum(extreme_tail_mask)),
    }


def prepare_data():
    train_csv_path, val_csv_path, dataset_base = resolve_dataset_paths()
    model_dir = configure_model_dir(dataset_base)
    print(f">>> Using dataset run:       {dataset_base}")
    print(f">>> Loading training CSV from: {train_csv_path}")
    print(f">>> Loading validation CSV from: {val_csv_path}")
    print(f">>> Dataset search dirs:    {format_dataset_dir_text()}")
    print(f">>> Wavelet image dir:      {Config.WAVELET_IMAGE_DIR}")
    print(f">>> Model run name:         {model_dir.name}")
    print(f">>> Model artifacts dir:    {model_dir}")

    try:
        train_df = pd.read_csv(train_csv_path, low_memory=False)
        val_df = pd.read_csv(val_csv_path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Error reading CSV: {exc}")

    validate_dataframe(train_df, train_csv_path)
    validate_dataframe(val_df, val_csv_path)
    train_df = filter_valid_rows(train_df)
    val_df = filter_valid_rows(val_df)
    train_df = sample_dataframe_by_group(train_df, Config.TXT_COL, Config.DATA_USE_RATIO, Config.SEED)

    layout_width = int(max(Config.MAX_DAMPER_FLOORS, train_df["num_floors"].max(), val_df["num_floors"].max()))
    train_scalar_df, scalar_feature_names = build_scalar_feature_frame(train_df, layout_width=layout_width)
    val_scalar_df, _ = build_scalar_feature_frame(val_df, layout_width=layout_width)
    Config.SCALAR_COLS = list(scalar_feature_names)
    print(f">>> Scalar feature columns ({len(scalar_feature_names)}): {scalar_feature_names}")

    train_paths = build_image_paths(train_df[Config.IMAGE_COL].values)
    val_paths = build_image_paths(val_df[Config.IMAGE_COL].values)
    train_labels = train_df[Config.LABEL_COL].values.astype(np.float32)
    val_labels = val_df[Config.LABEL_COL].values.astype(np.float32)
    train_loss_weights = build_label_density_weights(train_labels)
    val_loss_weights = np.ones_like(val_labels, dtype=np.float32)
    tail_classification_summary = build_tail_classification_summary(train_labels)
    train_scalars_raw = train_scalar_df.values.astype(np.float32)
    val_scalars_raw = val_scalar_df.values.astype(np.float32)
    train_unique_txt_paths = list(dict.fromkeys(train_df[Config.TXT_COL].astype(str).values))
    val_unique_txt_paths = list(dict.fromkeys(val_df[Config.TXT_COL].astype(str).values))
    train_unique_image_paths = list(dict.fromkeys(train_paths))
    val_unique_image_paths = list(dict.fromkeys(val_paths))
    print(f">>> Unique train txt files: {len(train_unique_txt_paths)}")
    print(f">>> Unique val txt files:   {len(val_unique_txt_paths)}")
    print(f">>> Unique train images:    {len(train_unique_image_paths)}")
    print(f">>> Unique val images:      {len(val_unique_image_paths)}")
    print(f">>> Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    print(
        ">>> Train loss weights: "
        f"min={train_loss_weights.min():.3f}, "
        f"max={train_loss_weights.max():.3f}, "
        f"mean={train_loss_weights.mean():.3f}"
    )

    print(">>> Fitting StandardScaler on training scalars...")
    scalar_scaler = StandardScaler()
    train_scalars_norm = scalar_scaler.fit_transform(train_scalars_raw)
    val_scalars_norm = scalar_scaler.transform(val_scalars_raw)

    scalar_scaler_path = model_dir / "scalar_scaler.pkl"
    joblib.dump(scalar_scaler, scalar_scaler_path)
    print(f">>> Scalar scaler saved to {scalar_scaler_path}")

    metadata = {
        "dataset_base": dataset_base,
        "model_run_name": model_dir.name,
        "unique_model_run_dir": Config.UNIQUE_MODEL_RUN_DIR,
        "train_csv_path": str(train_csv_path),
        "val_csv_path": str(val_csv_path),
        "dataset_search_dirs": [str(path) for path in get_existing_dataset_dirs()],
        "model_dir": str(model_dir),
        "image_col": Config.IMAGE_COL,
        "txt_col": Config.TXT_COL,
        "damper_layout_col": Config.DAMPER_LAYOUT_COL,
        "layout_feature_width": int(layout_width),
        "label_col": Config.LABEL_COL,
        "scale_target": Config.SCALE_TARGET,
        "label_scale": Config.LABEL_SCALE,
        "base_scalar_cols": list(Config.BASE_SCALAR_COLS),
        "wave_feature_cols": get_available_wave_feature_cols(train_df),
        "wave_derived_features_enabled": bool(Config.USE_WAVE_DERIVED_FEATURES),
        "wave_log_feature_cols": list(Config.WAVE_LOG_FEATURE_COLS),
        "scalar_feature_names": list(scalar_feature_names),
        "num_scalar_features": int(len(scalar_feature_names)),
        "best_weights_name": "best_2dcnn_model.pth",
        "best_weights_metric": "val_mae_raw",
        "last_weights_name": "multimodal_2dcnn_model.pth",
        "final_weights_name": "multimodal_2dcnn_model.pth",
        "wavelet_image_dir": str(Config.WAVELET_IMAGE_DIR),
        "force_wavelet_image_dir": bool(Config.FORCE_WAVELET_IMAGE_DIR),
        "image_size": list(Config.IMAGE_SIZE),
        "image_normalize_mean": list(Config.IMAGE_NORMALIZE_MEAN),
        "image_normalize_std": list(Config.IMAGE_NORMALIZE_STD),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_unique_waves": int(len(train_unique_txt_paths)),
        "val_unique_waves": int(len(val_unique_txt_paths)),
        "train_unique_images": int(len(train_unique_image_paths)),
        "val_unique_images": int(len(val_unique_image_paths)),
        "train_loss_weight_summary": {
            "min": float(train_loss_weights.min()),
            "max": float(train_loss_weights.max()),
            "mean": float(train_loss_weights.mean()),
        },
        "tail_classification": tail_classification_summary,
    }

    train_transform = build_image_transform(train=True)
    val_transform = build_image_transform(train=False)

    train_dataset = MultimodalDataset(
        train_paths,
        train_scalars_norm,
        train_labels,
        sample_weights=train_loss_weights,
        transform=train_transform,
    )
    val_dataset = MultimodalDataset(
        val_paths,
        val_scalars_norm,
        val_labels,
        sample_weights=val_loss_weights,
        transform=val_transform,
    )

    pin_memory = Config.DEVICE.type == "cuda"
    persistent_workers = Config.NUM_WORKERS > 0
    sampler, sampler_summary = build_train_sampler(train_df)
    metadata["use_weighted_sampler"] = bool(Config.USE_WEIGHTED_SAMPLER)
    metadata["sampler_summary"] = sampler_summary
    train_drop_last = len(train_dataset) > Config.BATCH_SIZE and len(train_dataset) % Config.BATCH_SIZE == 1
    metadata["train_drop_last"] = bool(train_drop_last)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=train_drop_last,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, metadata


def train():
    set_global_seed(Config.SEED)
    train_loader, val_loader, metadata = prepare_data()
    use_amp = Config.USE_AMP and Config.DEVICE.type == "cuda"

    model = MultimodalPredictor(num_scalars=int(metadata["num_scalar_features"])).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n>>> Model Architecture:")
    print(model)
    print(f"\n>>> Total parameters: {total_params:,}")
    print(f">>> Trainable parameters: {trainable_params:,}\n")

    metadata.update(
        {
            "model_family": "multimodal_2dcnn",
            "best_weights_name": "best_2dcnn_model.pth",
            "best_weights_metric": "val_mae_raw",
            "best_weights_are_ema": Config.USE_EMA,
            "selection_weights_name": "best_2dcnn_focus_model.pth",
            "last_weights_name": "multimodal_2dcnn_model.pth",
            "ema_weights_name": "ema_2dcnn_model.pth" if Config.USE_EMA else None,
            "total_params": int(total_params),
            "trainable_params": int(trainable_params),
            "optimizer": "AdamW",
            "optimizer_no_decay_norm_and_bias": Config.OPTIMIZER_NO_DECAY_NORM_AND_BIAS,
            "loss": "SmoothL1Loss",
            "smooth_l1_beta": Config.SMOOTH_L1_BETA,
            "learning_rate": Config.LEARNING_RATE,
            "weight_decay": Config.WEIGHT_DECAY,
            "batch_size": Config.BATCH_SIZE,
            "grad_clip_norm": Config.GRAD_CLIP_NORM,
            "use_amp": bool(use_amp),
            "use_ema": Config.USE_EMA,
            "ema_decay": Config.EMA_DECAY if Config.USE_EMA else None,
            "target_weighted_loss": {
                "enabled": Config.USE_TARGET_WEIGHTED_LOSS,
                "density_enabled": Config.USE_DENSITY_WEIGHTED_LOSS,
                "density_bin_count": Config.LOSS_DENSITY_BIN_COUNT,
                "density_smooth_kernel_size": Config.LOSS_DENSITY_SMOOTH_KERNEL_SIZE,
                "density_smooth_sigma": Config.LOSS_DENSITY_SMOOTH_SIGMA,
                "density_alpha": Config.LOSS_DENSITY_ALPHA,
                "weight_min": Config.LOSS_WEIGHT_MIN,
                "weight_max": Config.LOSS_WEIGHT_MAX,
                "ge_0p005": Config.LOSS_WEIGHT_GE_005,
                "ge_0p010": Config.LOSS_WEIGHT_GE_010,
                "ge_0p020": Config.LOSS_WEIGHT_GE_020,
                "smooth_tail_weights": Config.USE_SMOOTH_TAIL_WEIGHTS,
                "tail_weight_transition_width": Config.TAIL_WEIGHT_TRANSITION_WIDTH,
            },
            "tail_underprediction_loss": {
                "enabled": Config.USE_TAIL_UNDERPREDICTION_LOSS,
                "start_epoch": Config.TAIL_UNDERPREDICTION_START_EPOCH,
                "ramp_epochs": Config.TAIL_UNDERPREDICTION_RAMP_EPOCHS,
                "tail_threshold": Config.TAIL_UNDERPREDICTION_THRESHOLD,
                "tail_weight": Config.TAIL_UNDERPREDICTION_WEIGHT,
                "extreme_tail_threshold": Config.EXTREME_TAIL_UNDERPREDICTION_THRESHOLD,
                "extreme_tail_weight": Config.EXTREME_TAIL_UNDERPREDICTION_WEIGHT,
                "max_loss": Config.TAIL_UNDERPREDICTION_MAX_LOSS,
            },
            "validation_selection": {
                "metric": "val_mae_raw",
                "tie_breaker": "focus_score",
                "tail_threshold": Config.VAL_TAIL_THRESHOLD,
                "extreme_tail_threshold": Config.VAL_EXTREME_TAIL_THRESHOLD,
                "rmse_weight": Config.VAL_FOCUS_RMSE_WEIGHT,
                "tail_mae_weight": Config.VAL_FOCUS_TAIL_MAE_WEIGHT,
                "extreme_tail_mae_weight": Config.VAL_FOCUS_EXTREME_TAIL_MAE_WEIGHT,
                "tail_under_weight": Config.VAL_FOCUS_TAIL_UNDER_WEIGHT,
                "extreme_tail_under_weight": Config.VAL_FOCUS_EXTREME_TAIL_UNDER_WEIGHT,
            },
            "warmup": {
                "epochs": Config.WARMUP_EPOCHS,
                "start_factor": Config.WARMUP_START_FACTOR,
            },
            "image_augmentation": {
                "enabled": Config.USE_TRAIN_IMAGE_AUGMENTATION,
                "time_freq_mask_prob": Config.TIME_FREQ_MASK_PROB,
                "time_mask_count": Config.TIME_MASK_COUNT,
                "time_mask_max_fraction": Config.TIME_MASK_MAX_FRACTION,
                "freq_mask_count": Config.FREQ_MASK_COUNT,
                "freq_mask_max_fraction": Config.FREQ_MASK_MAX_FRACTION,
                "fill_value": Config.TIME_FREQ_MASK_FILL_VALUE,
            },
            "scheduler": {
                "type": "ReduceLROnPlateau",
                "monitor": "val_mae_raw",
                "factor": Config.SCHEDULER_FACTOR,
                "patience": Config.SCHEDULER_PATIENCE,
                "min_lr": Config.MIN_LR,
            },
            "sampler_params": {
                "target_bin_count": Config.TARGET_BIN_COUNT,
                "power": Config.SAMPLER_POWER,
                "min_weight": Config.SAMPLER_MIN_WEIGHT,
                "max_weight": Config.SAMPLER_MAX_WEIGHT,
                "tail_boost_ge_0p005": Config.SAMPLER_TAIL_BOOST_GE_005,
                "tail_boost_ge_0p010": Config.SAMPLER_TAIL_BOOST_GE_010,
                "tail_boost_ge_0p020": Config.SAMPLER_TAIL_BOOST_GE_020,
                "num_samples_multiplier": Config.SAMPLER_NUM_SAMPLES_MULTIPLIER,
                "smooth_tail_weights": Config.USE_SMOOTH_TAIL_WEIGHTS,
                "tail_weight_transition_width": Config.TAIL_WEIGHT_TRANSITION_WIDTH,
            },
            "architecture": {
                "cnn_backbone": Config.CNN_BACKBONE,
                "channels": Config.CNN_CHANNELS,
                "kernel_size": Config.CNN_KERNEL_SIZE,
                "pool_sizes": Config.CNN_POOL_SIZES,
                "cnn_norm": Config.CNN_NORM,
                "cnn_group_norm_max_groups": Config.CNN_GROUP_NORM_MAX_GROUPS,
                "cnn_dropout": Config.CNN_DROPOUT,
                "pool_output": list(Config.CNN_POOL_OUTPUT),
                "projector_dim": Config.CNN_PROJECTOR_DIM,
                "projector_dropout": Config.CNN_PROJECTOR_DROPOUT,
                "film_identity_init": Config.CNN_FILM_IDENTITY_INIT,
                "film_gate_init_bias": Config.CNN_FILM_GATE_INIT_BIAS,
                "scalar_layers": Config.MLP_HIDDEN_LAYERS,
                "scalar_norm": Config.SCALAR_NORM,
                "scalar_dropout": Config.MLP_DROPOUT,
                "head_hidden_dims": Config.HEAD_HIDDEN_DIMS,
                "head_dropout": Config.HEAD_DROPOUT,
                "tail_correction_head": Config.USE_TAIL_CORRECTION_HEAD,
                "tail_correction_gate": Config.USE_TAIL_CORRECTION_GATE,
                "tail_correction_hidden_dim": Config.TAIL_CORRECTION_HIDDEN_DIM,
                "tail_correction_dropout": Config.TAIL_CORRECTION_DROPOUT,
                "tail_correction_init_bias": Config.TAIL_CORRECTION_INIT_BIAS,
                "tail_correction_gate_init_bias": Config.TAIL_CORRECTION_GATE_INIT_BIAS,
                "tail_classification_aux": Config.USE_TAIL_CLASSIFICATION_AUX,
                "tail_classification_thresholds": list(Config.TAIL_CLASSIFICATION_THRESHOLDS),
                "tail_classification_hidden_dim": Config.TAIL_CLASSIFICATION_HIDDEN_DIM,
                "tail_classification_dropout": Config.TAIL_CLASSIFICATION_DROPOUT,
                "tail_classification_init_bias": Config.TAIL_CLASSIFICATION_INIT_BIAS,
            },
        }
    )
    save_training_metadata(metadata)

    criterion = nn.SmoothL1Loss(beta=Config.SMOOTH_L1_BETA, reduction="none")
    optimizer = build_adamw_optimizer(model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        min_lr=Config.MIN_LR,
    )
    scaler = build_grad_scaler(use_amp)
    ema = ModelEMA(model, Config.EMA_DECAY) if Config.USE_EMA else None
    tail_class_pos_weights = torch.tensor(
        metadata.get("tail_classification", {}).get(
            "pos_weights",
            [1.0] * len(Config.TAIL_CLASSIFICATION_THRESHOLDS),
        ),
        dtype=torch.float32,
        device=Config.DEVICE,
    )

    best_val_mae_raw = float("inf")
    best_val_loss = float("inf")
    best_val_focus_score = float("inf")
    best_val_tail_mae_raw = float("inf")
    best_val_extreme_tail_mae_raw = float("inf")
    best_val_tail_under_mae_raw = float("inf")
    best_val_extreme_tail_under_mae_raw = float("inf")
    best_raw_focus_score = float("inf")
    best_raw_focus_epoch = 0
    best_mae_checkpoint = float("inf")
    best_mae_epoch = 0
    best_extreme_under_checkpoint = float("inf")
    best_extreme_under_epoch = 0
    best_epoch = 0
    early_stop_counter = 0
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae_raw": [],
        "val_rmse_raw": [],
        "val_r2": [],
        "val_focus_score": [],
        "val_tail_mae_raw": [],
        "val_tail_bias_raw": [],
        "val_tail_under_mae_raw": [],
        "val_tail_count": [],
        "val_extreme_tail_mae_raw": [],
        "val_extreme_tail_bias_raw": [],
        "val_extreme_tail_under_mae_raw": [],
        "val_extreme_tail_count": [],
        "val_selection_score": [],
        "lr": [],
    }

    print(f"\n>>> Start Training for {Config.NUM_EPOCHS} epochs...")
    print(f"    AMP: {'ON' if use_amp else 'OFF'} | Early Stopping patience: {Config.EARLY_STOPPING_PATIENCE}")
    start_time = time.time()

    for epoch in range(Config.NUM_EPOCHS):
        apply_lr_warmup(optimizer, epoch)
        model.train()
        train_loss = 0.0
        train_seen_samples = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} [Train]")
        for imgs, scalars, labels, sample_weights in loop:
            imgs = imgs.to(Config.DEVICE, non_blocking=True)
            scalars = scalars.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True).unsqueeze(1)
            sample_weights = sample_weights.to(Config.DEVICE, non_blocking=True).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with build_amp_autocast_context(use_amp):
                if Config.USE_TAIL_CLASSIFICATION_AUX:
                    preds, aux_outputs = model(imgs, scalars, return_aux=True)
                else:
                    preds = model(imgs, scalars)
                    aux_outputs = {}
                loss_per_sample = criterion(preds, labels)
                loss_weights = apply_tail_loss_multipliers(labels, sample_weights)
                loss = (loss_per_sample * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
                loss = loss + calculate_tail_underprediction_loss(preds, labels, epoch_num=epoch + 1)
                loss = loss + calculate_tail_classification_loss(
                    aux_outputs,
                    labels,
                    tail_class_pos_weights,
                    epoch_num=epoch + 1,
                )

            scaler.scale(loss).backward()
            if Config.GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)

            train_loss += loss.item() * imgs.size(0)
            train_seen_samples += imgs.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        epoch_train_loss = train_loss / max(train_seen_samples, 1)

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        model.eval()
        val_loss = 0.0
        pred_batches = []
        true_batches = []
        with torch.no_grad():
            for imgs, scalars, labels, _sample_weights in val_loader:
                imgs = imgs.to(Config.DEVICE, non_blocking=True)
                scalars = scalars.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True).unsqueeze(1)

                with build_amp_autocast_context(use_amp):
                    preds = model(imgs, scalars)
                    loss = criterion(preds, labels).mean()

                val_loss += loss.item() * imgs.size(0)
                pred_batches.append(preds.detach().cpu().numpy().reshape(-1))
                true_batches.append(labels.detach().cpu().numpy().reshape(-1))

        epoch_val_loss = val_loss / len(val_loader.dataset)
        pred_scaled = np.concatenate(pred_batches).astype(np.float32)
        true_scaled = np.concatenate(true_batches).astype(np.float32)
        if Config.SCALE_TARGET:
            pred_raw = pred_scaled / Config.LABEL_SCALE
            true_raw = true_scaled / Config.LABEL_SCALE
        else:
            pred_raw = pred_scaled
            true_raw = true_scaled

        metrics = calculate_regression_metrics(true_raw, pred_raw)
        focus_metrics = calculate_validation_focus_metrics(true_raw, pred_raw)
        current_lr = float(optimizer.param_groups[0]["lr"])

        checkpoint_state = ema.state_dict() if ema is not None else model.state_dict()
        if ema is not None:
            ema.restore(model)

        selection_score = float(metrics["mae"])
        if epoch + 1 > Config.WARMUP_EPOCHS:
            scheduler.step(selection_score)

        history["train_loss"].append(float(epoch_train_loss))
        history["val_loss"].append(float(epoch_val_loss))
        history["val_mae_raw"].append(metrics["mae"])
        history["val_rmse_raw"].append(metrics["rmse"])
        history["val_r2"].append(metrics["r2"])
        history["val_focus_score"].append(focus_metrics["focus_score"])
        history["val_tail_mae_raw"].append(focus_metrics["tail_mae"])
        history["val_tail_bias_raw"].append(focus_metrics["tail_bias"])
        history["val_tail_under_mae_raw"].append(focus_metrics["tail_under_mae"])
        history["val_tail_count"].append(focus_metrics["tail_count"])
        history["val_extreme_tail_mae_raw"].append(focus_metrics["extreme_tail_mae"])
        history["val_extreme_tail_bias_raw"].append(focus_metrics["extreme_tail_bias"])
        history["val_extreme_tail_under_mae_raw"].append(focus_metrics["extreme_tail_under_mae"])
        history["val_extreme_tail_count"].append(focus_metrics["extreme_tail_count"])
        history["val_selection_score"].append(float(selection_score))
        history["lr"].append(current_lr)

        improved = (metrics["mae"] < best_val_mae_raw) or (
            np.isclose(metrics["mae"], best_val_mae_raw)
            and focus_metrics["focus_score"] < best_val_focus_score
        )
        if Config.SAVE_ALTERNATE_BEST_CHECKPOINTS:
            if focus_metrics["focus_score"] < best_raw_focus_score:
                best_raw_focus_score = focus_metrics["focus_score"]
                best_raw_focus_epoch = epoch + 1
                torch.save(checkpoint_state, Config.SAVE_DIR / "best_2dcnn_focus_model.pth")
            if metrics["mae"] < best_mae_checkpoint:
                best_mae_checkpoint = metrics["mae"]
                best_mae_epoch = epoch + 1
                torch.save(checkpoint_state, Config.SAVE_DIR / "best_2dcnn_mae_model.pth")
            if focus_metrics["extreme_tail_under_mae"] < best_extreme_under_checkpoint:
                best_extreme_under_checkpoint = focus_metrics["extreme_tail_under_mae"]
                best_extreme_under_epoch = epoch + 1
                torch.save(checkpoint_state, Config.SAVE_DIR / "best_2dcnn_extreme_under_model.pth")

        if improved:
            best_val_focus_score = focus_metrics["focus_score"]
            best_val_mae_raw = metrics["mae"]
            best_val_loss = epoch_val_loss
            best_val_tail_mae_raw = focus_metrics["tail_mae"]
            best_val_extreme_tail_mae_raw = focus_metrics["extreme_tail_mae"]
            best_val_tail_under_mae_raw = focus_metrics["tail_under_mae"]
            best_val_extreme_tail_under_mae_raw = focus_metrics["extreme_tail_under_mae"]
            best_epoch = epoch + 1
            early_stop_counter = 0
            torch.save(checkpoint_state, Config.SAVE_DIR / "best_2dcnn_model.pth")
            tqdm.write(
                "  >> Best model saved "
                f"(Selection MAE: {selection_score:.6f}, "
                f"Focus score: {focus_metrics['focus_score']:.6f}, "
                f"MAE: {best_val_mae_raw:.6f}, "
                f"tail MAE: {best_val_tail_mae_raw:.6f}, "
                f"tail under: {best_val_tail_under_mae_raw:.6f}, "
                f"extreme tail MAE: {focus_metrics['extreme_tail_mae']:.6f}, "
                f"extreme under: {best_val_extreme_tail_under_mae_raw:.6f})"
            )
        else:
            early_stop_counter += 1

        tqdm.write(
            f"Ep {epoch + 1}: "
            f"Train SmoothL1={epoch_train_loss:.4f} | "
            f"Val SmoothL1={epoch_val_loss:.4f} | "
            f"Val MAE={metrics['mae']:.6f} | "
            f"Val RMSE={metrics['rmse']:.6f} | "
            f"Val R2={metrics['r2']:.4f} | "
            f"Val Score={focus_metrics['focus_score']:.6f} | "
            f"Selection MAE={selection_score:.6f} | "
                f"Tail MAE={focus_metrics['tail_mae']:.6f} | "
                f"Tail Bias={focus_metrics['tail_bias']:.6f} | "
                f"Tail Under={focus_metrics['tail_under_mae']:.6f} | "
                f"Extreme Tail MAE={focus_metrics['extreme_tail_mae']:.6f} | "
                f"Extreme Bias={focus_metrics['extreme_tail_bias']:.6f} | "
                f"Extreme Under={focus_metrics['extreme_tail_under_mae']:.6f} | "
                f"LR={current_lr:.2e} | "
            f"EarlyStop={early_stop_counter}/{Config.EARLY_STOPPING_PATIENCE}"
        )

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            tqdm.write(f"\n>>> Early stopping triggered at epoch {epoch + 1}.")
            break

    final_model_path = Config.SAVE_DIR / "multimodal_2dcnn_model.pth"
    torch.save(model.state_dict(), final_model_path)
    if ema is not None:
        torch.save(ema.state_dict(), Config.SAVE_DIR / "ema_2dcnn_model.pth")
    total_minutes = (time.time() - start_time) / 60

    metadata.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_mae_raw": float(best_val_mae_raw),
            "best_val_loss": float(best_val_loss),
            "best_val_focus_score": float(best_val_focus_score),
            "best_val_tail_mae_raw": float(best_val_tail_mae_raw),
            "best_val_extreme_tail_mae_raw": float(best_val_extreme_tail_mae_raw),
            "best_val_tail_under_mae_raw": float(best_val_tail_under_mae_raw),
            "best_val_extreme_tail_under_mae_raw": float(best_val_extreme_tail_under_mae_raw),
            "alternate_best_checkpoints": {
                "enabled": bool(Config.SAVE_ALTERNATE_BEST_CHECKPOINTS),
                "focus": {
                    "weights_name": "best_2dcnn_focus_model.pth",
                    "epoch": int(best_raw_focus_epoch),
                    "score": float(best_raw_focus_score),
                },
                "mae": {
                    "weights_name": "best_2dcnn_mae_model.pth",
                    "epoch": int(best_mae_epoch),
                    "mae": float(best_mae_checkpoint),
                },
                "extreme_under": {
                    "weights_name": "best_2dcnn_extreme_under_model.pth",
                    "epoch": int(best_extreme_under_epoch),
                    "mae": float(best_extreme_under_checkpoint),
                },
            },
            "total_train_minutes": float(total_minutes),
            "history_keys": list(history.keys()),
        }
    )
    save_training_metadata(metadata)

    history_path = Config.MODEL_DIR / "training_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)
    print(f">>> Training history saved to: {history_path}")

    print(f"\n>>> Training Complete. Model saved to {final_model_path}")
    print(
        f">>> Best model saved with Val MAE: {best_val_mae_raw:.6f}, "
        f"Focus score: {best_val_focus_score:.6f}, "
        f"Tail MAE: {best_val_tail_mae_raw:.6f}, "
        f"Tail Under: {best_val_tail_under_mae_raw:.6f}, "
        f"Extreme Tail MAE: {best_val_extreme_tail_mae_raw:.6f}, "
        f"Extreme Under: {best_val_extreme_tail_under_mae_raw:.6f} at epoch {best_epoch}"
    )
    print(f">>> Total Time: {total_minutes:.1f} mins")
    return history


def plot_results(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train Loss (SmoothL1)")
    plt.plot(history["val_loss"], label="Val Loss (SmoothL1)")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history["val_mae_raw"], label="Val MAE", color="orange")
    if "val_focus_score" in history:
        plt.plot(history["val_focus_score"], label="Val Focus Score", color="green", alpha=0.8)
    if "val_selection_score" in history:
        plt.plot(history["val_selection_score"], label="Selection MAE", color="black", alpha=0.65)
    if "val_tail_mae_raw" in history:
        plt.plot(history["val_tail_mae_raw"], label="Tail MAE", color="red", alpha=0.8)
    if "val_extreme_tail_mae_raw" in history:
        plt.plot(history["val_extreme_tail_mae_raw"], label="Extreme Tail MAE", color="purple", alpha=0.8)
    if "val_extreme_tail_under_mae_raw" in history:
        plt.plot(
            history["val_extreme_tail_under_mae_raw"],
            label="Extreme Under MAE",
            color="brown",
            alpha=0.8,
        )
    plt.title("Validation Metrics")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.semilogy(history["train_loss"], label="Train (log)")
    plt.semilogy(history["val_loss"], label="Val (log)")
    plt.title("Loss (Log Scale)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = Config.SAVE_DIR / "training_curves.png"
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f">>> Training curves saved to {fig_path}")


if __name__ == "__main__":
    train_history = train()
    plot_results(train_history)
    if Config.SCALE_TARGET:
        print("\n提示: 模型训练目标已放大 1000 倍。")
        print("例如: 模型输出 5.0，实际代表 drift ratio 为 0.005。")
