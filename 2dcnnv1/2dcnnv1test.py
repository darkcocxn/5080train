# -*- coding: utf-8 -*-
"""
2D-CNN 多模态测试脚本（三阶段数据集，小波图版，3 到 7 层）
用于评估 `2dcnnv1.py` 训练得到的模型。
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for SUPPORT_DIR in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(SUPPORT_DIR) not in sys.path:
        sys.path.append(str(SUPPORT_DIR))

from floors_3_to_7_utils import (
    build_seed_metrics_dataframe,
    build_seed_metrics_report,
    calculate_metrics,
    calculate_relative_errors,
    print_seed_summary,
    resolve_image_path,
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
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / "2dcnnv1"
    MODEL_DIR = MODEL_ROOT_DIR

    TEST_CSV_PATH = None
    DATASET_PREFIX = "opensees_surrogate_dataset_floors_3_to_7_"
    TEST_FILE_PATTERN = f"{DATASET_PREFIX}*_test.csv"
    WAVELET_IMAGE_DIR = _first_existing_path(
        _latest_existing_dir(RAW_DATA_DIR / "Scalogram", "6000-1-*"),
        RAW_DATA_DIR / "Scalogram" / "6000-1",
        RAW_DATA_DIR / "Scalogram" / "6000-uniform-scale-0.1-1.0",
        RAW_DATA_DIR / "Scalogram" / "6000",
        PROJECT_ROOT / "Raw data file" / "Scalogram" / "6000-1",
        PROJECT_ROOT / "Raw data file" / "Scalogram" / "6000",
    )
    FORCE_WAVELET_IMAGE_DIR = True
    MODEL_WEIGHTS_PATH = MODEL_DIR / "best_2dcnn_model.pth"
    MODEL_WEIGHTS_NAME_OVERRIDE = os.environ.get("SURMOD_2DCNN_WEIGHTS_NAME")
    SCALER_PATH = MODEL_DIR / "scalar_scaler.pkl"
    TRAINING_METADATA_PATH = MODEL_DIR / "training_metadata.json"

    IMAGE_COL = "image_path"
    TXT_COL = "txt_path"
    LABEL_COL = "max_drift_ratio_raw"
    STATUS_COL = "analysis_status"
    YIELDED_COL = "steel02_yielded"
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
    POST_PROCESS_FACTOR = 1.0

    TEST_SEEDS = [42, 2026, 123]
    TEST_SAMPLE_RATIO = 0.8
    TEST_SAMPLE_WITH_REPLACEMENT = False

    IMAGE_SIZE = (128, 128)
    IMAGE_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_NORMALIZE_STD = [0.229, 0.224, 0.225]
    BATCH_SIZE = 64
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4

    CNN_BACKBONE = "legacy"
    CNN_CHANNELS = [32, 64, 128]
    CNN_KERNEL_SIZE = 3
    CNN_POOL_SIZES = [2, 2, 2]
    CNN_NORM = "batch"
    CNN_GROUP_NORM_MAX_GROUPS = 8
    CNN_DROPOUT = 0.2
    CNN_POOL_OUTPUT = (4, 4)
    CNN_PROJECTOR_DIM = 256
    CNN_PROJECTOR_DROPOUT = 0.3
    CNN_FILM_IDENTITY_INIT = True
    CNN_FILM_GATE_INIT_BIAS = 3.0
    MLP_HIDDEN_LAYERS = [64, 128]
    SCALAR_NORM = "batch"
    MLP_DROPOUT = 0.1
    HEAD_HIDDEN_DIMS = [256, 64]
    HEAD_DROPOUT = 0.4
    USE_TAIL_CORRECTION_HEAD = False
    USE_TAIL_CORRECTION_GATE = False
    TAIL_CORRECTION_HIDDEN_DIM = 64
    TAIL_CORRECTION_DROPOUT = 0.05
    TAIL_CORRECTION_INIT_BIAS = -4.0
    TAIL_CORRECTION_GATE_INIT_BIAS = -1.5
    USE_TAIL_CLASSIFICATION_AUX = False
    TAIL_CLASSIFICATION_THRESHOLDS = [0.010, 0.020]
    TAIL_CLASSIFICATION_HIDDEN_DIM = 64
    TAIL_CLASSIFICATION_DROPOUT = 0.05
    TAIL_CLASSIFICATION_INIT_BIAS = -3.0


print(f"Running on device: {Config.DEVICE}")
Config.MODEL_ROOT_DIR.mkdir(parents=True, exist_ok=True)


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


class TestDataset(Dataset):
    def __init__(self, image_paths, scalar_data, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.scalar_data = torch.tensor(scalar_data, dtype=torch.float32)
        self.transform = transform
        print(f">>> Test dataset size: {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", Config.IMAGE_SIZE)

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        scalars = self.scalar_data[idx]
        return image, scalars, label, img_path


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


def configure_model_artifact_paths(model_dir: Path) -> None:
    Config.MODEL_DIR = model_dir
    Config.MODEL_WEIGHTS_PATH = model_dir / "best_2dcnn_model.pth"
    Config.SCALER_PATH = model_dir / "scalar_scaler.pkl"
    Config.TRAINING_METADATA_PATH = model_dir / "training_metadata.json"


def _has_legacy_2dcnn_artifacts(path: Path) -> bool:
    return (path / "best_2dcnn_model.pth").exists() and (path / "scalar_scaler.pkl").exists()


def resolve_model_dir() -> Path:
    if Config.MODEL_DIR != Config.MODEL_ROOT_DIR:
        model_dir = Path(Config.MODEL_DIR)
        if not model_dir.exists():
            raise FileNotFoundError(f"指定的模型目录不存在: {model_dir}")
        return model_dir

    candidates: list[Path] = []
    search_roots = (Config.MODEL_ROOT_DIR,)
    for root_dir in search_roots:
        legacy_metadata_path = root_dir / "training_metadata.json"
        if legacy_metadata_path.exists():
            candidates.append(root_dir)
        elif _has_legacy_2dcnn_artifacts(root_dir):
            candidates.append(root_dir)

        if root_dir.exists():
            candidates.extend(
                path
                for path in root_dir.iterdir()
                if path.is_dir() and ((path / "training_metadata.json").exists() or _has_legacy_2dcnn_artifacts(path))
            )

    if not candidates:
        raise FileNotFoundError(
            "未找到可用的 2DCNN 模型运行目录或 training_metadata.json。"
        )

    return max(
        candidates,
        key=lambda path: (path / "training_metadata.json").stat().st_mtime
        if (path / "training_metadata.json").exists()
        else path.stat().st_mtime,
    )


def load_training_metadata() -> dict:
    if not Config.TRAINING_METADATA_PATH.exists():
        return {}
    with Config.TRAINING_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    Config.LABEL_COL = metadata.get("label_col", Config.LABEL_COL)
    Config.SCALE_TARGET = bool(metadata.get("scale_target", Config.SCALE_TARGET))
    Config.LABEL_SCALE = float(metadata.get("label_scale", Config.LABEL_SCALE))
    if metadata.get("image_size"):
        Config.IMAGE_SIZE = tuple(metadata["image_size"])
    if metadata.get("image_normalize_mean"):
        Config.IMAGE_NORMALIZE_MEAN = list(metadata["image_normalize_mean"])
    if metadata.get("image_normalize_std"):
        Config.IMAGE_NORMALIZE_STD = list(metadata["image_normalize_std"])
    if metadata.get("wavelet_image_dir"):
        Config.WAVELET_IMAGE_DIR = Path(metadata["wavelet_image_dir"])
    if "force_wavelet_image_dir" in metadata:
        Config.FORCE_WAVELET_IMAGE_DIR = bool(metadata["force_wavelet_image_dir"])
    best_weights_name = Config.MODEL_WEIGHTS_NAME_OVERRIDE or metadata.get("best_weights_name")
    if best_weights_name:
        Config.MODEL_WEIGHTS_PATH = Config.MODEL_DIR / best_weights_name
    scalar_feature_names = metadata.get("scalar_feature_names")
    if scalar_feature_names:
        Config.SCALAR_COLS = list(scalar_feature_names)
    wave_feature_cols = metadata.get("wave_feature_cols")
    if wave_feature_cols is not None:
        Config.WAVE_FEATURE_COLS = list(wave_feature_cols)
    wave_log_feature_cols = metadata.get("wave_log_feature_cols")
    if wave_log_feature_cols is not None:
        Config.WAVE_LOG_FEATURE_COLS = list(wave_log_feature_cols)
    if "wave_derived_features_enabled" in metadata:
        Config.USE_WAVE_DERIVED_FEATURES = bool(metadata["wave_derived_features_enabled"])
    layout_feature_width = metadata.get("layout_feature_width")
    if layout_feature_width is not None:
        Config.MAX_DAMPER_FLOORS = max(int(layout_feature_width), Config.MAX_DAMPER_FLOORS)

    architecture = metadata.get("architecture") or {}
    if architecture:
        Config.CNN_BACKBONE = architecture.get("cnn_backbone", Config.CNN_BACKBONE)
        Config.CNN_CHANNELS = list(architecture.get("channels", Config.CNN_CHANNELS))
        Config.CNN_KERNEL_SIZE = int(architecture.get("kernel_size", Config.CNN_KERNEL_SIZE))
        Config.CNN_POOL_SIZES = list(architecture.get("pool_sizes", Config.CNN_POOL_SIZES))
        Config.CNN_NORM = str(architecture.get("cnn_norm", Config.CNN_NORM))
        Config.CNN_GROUP_NORM_MAX_GROUPS = int(
            architecture.get("cnn_group_norm_max_groups", Config.CNN_GROUP_NORM_MAX_GROUPS)
        )
        Config.CNN_DROPOUT = float(architecture.get("cnn_dropout", Config.CNN_DROPOUT))
        Config.CNN_POOL_OUTPUT = tuple(architecture.get("pool_output", Config.CNN_POOL_OUTPUT))
        Config.CNN_PROJECTOR_DIM = int(architecture.get("projector_dim", Config.CNN_PROJECTOR_DIM))
        Config.CNN_PROJECTOR_DROPOUT = float(
            architecture.get("projector_dropout", Config.CNN_PROJECTOR_DROPOUT)
        )
        Config.CNN_FILM_IDENTITY_INIT = bool(
            architecture.get("film_identity_init", Config.CNN_FILM_IDENTITY_INIT)
        )
        Config.CNN_FILM_GATE_INIT_BIAS = float(
            architecture.get("film_gate_init_bias", Config.CNN_FILM_GATE_INIT_BIAS)
        )
        Config.MLP_HIDDEN_LAYERS = list(architecture.get("scalar_layers", Config.MLP_HIDDEN_LAYERS))
        Config.SCALAR_NORM = str(architecture.get("scalar_norm", Config.SCALAR_NORM))
        Config.MLP_DROPOUT = float(architecture.get("scalar_dropout", Config.MLP_DROPOUT))
        Config.HEAD_HIDDEN_DIMS = list(architecture.get("head_hidden_dims", Config.HEAD_HIDDEN_DIMS))
        Config.HEAD_DROPOUT = float(architecture.get("head_dropout", Config.HEAD_DROPOUT))
        Config.USE_TAIL_CORRECTION_HEAD = bool(
            architecture.get("tail_correction_head", Config.USE_TAIL_CORRECTION_HEAD)
        )
        Config.USE_TAIL_CORRECTION_GATE = bool(
            architecture.get("tail_correction_gate", Config.USE_TAIL_CORRECTION_GATE)
        )
        Config.TAIL_CORRECTION_HIDDEN_DIM = int(
            architecture.get("tail_correction_hidden_dim", Config.TAIL_CORRECTION_HIDDEN_DIM)
        )
        Config.TAIL_CORRECTION_DROPOUT = float(
            architecture.get("tail_correction_dropout", Config.TAIL_CORRECTION_DROPOUT)
        )
        Config.TAIL_CORRECTION_INIT_BIAS = float(
            architecture.get("tail_correction_init_bias", Config.TAIL_CORRECTION_INIT_BIAS)
        )
        Config.TAIL_CORRECTION_GATE_INIT_BIAS = float(
            architecture.get("tail_correction_gate_init_bias", Config.TAIL_CORRECTION_GATE_INIT_BIAS)
        )
        Config.USE_TAIL_CLASSIFICATION_AUX = bool(
            architecture.get("tail_classification_aux", Config.USE_TAIL_CLASSIFICATION_AUX)
        )
        Config.TAIL_CLASSIFICATION_THRESHOLDS = list(
            architecture.get("tail_classification_thresholds", Config.TAIL_CLASSIFICATION_THRESHOLDS)
        )
        Config.TAIL_CLASSIFICATION_HIDDEN_DIM = int(
            architecture.get("tail_classification_hidden_dim", Config.TAIL_CLASSIFICATION_HIDDEN_DIM)
        )
        Config.TAIL_CLASSIFICATION_DROPOUT = float(
            architecture.get("tail_classification_dropout", Config.TAIL_CLASSIFICATION_DROPOUT)
        )
        Config.TAIL_CLASSIFICATION_INIT_BIAS = float(
            architecture.get("tail_classification_init_bias", Config.TAIL_CLASSIFICATION_INIT_BIAS)
        )

    if Config.TEST_CSV_PATH is None and metadata.get("train_csv_path"):
        train_csv_path = Path(metadata["train_csv_path"])
        train_name = train_csv_path.name
        if train_name.endswith("_train.csv"):
            paired_test_path = train_csv_path.with_name(train_name.replace("_train.csv", "_test.csv"))
            if paired_test_path.exists():
                Config.TEST_CSV_PATH = str(paired_test_path)
            else:
                Config.TEST_FILE_PATTERN = train_name.replace("_train.csv", "_test.csv")
    return metadata


def resolve_test_csv_path() -> Path:
    if Config.TEST_CSV_PATH:
        return resolve_explicit_csv_path(Config.TEST_CSV_PATH)

    candidates: list[Path] = []
    for dataset_dir in get_existing_dataset_dirs():
        candidates.extend(dataset_dir.glob(Config.TEST_FILE_PATTERN))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"在以下目录中未找到匹配 '{Config.TEST_FILE_PATTERN}' 的测试集文件: {format_dataset_dir_text()}"
        )
    return candidates[0]


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


def build_scalar_feature_frame(
    df: pd.DataFrame,
    layout_width: int,
    expected_feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    wave_feature_cols = [col for col in Config.WAVE_FEATURE_COLS if col in df.columns]
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

    if expected_feature_names:
        missing_features = [col for col in expected_feature_names if col not in feature_df.columns]
        if missing_features:
            raise KeyError(f"测试集构造标量特征时缺少训练所需列: {missing_features}")
        scalar_feature_names = list(expected_feature_names)

    return feature_df[scalar_feature_names], scalar_feature_names


def build_result_paths(test_csv_path):
    dataset_base = dataset_base_name_from_csv(test_csv_path)
    result_tag = build_compact_artifact_name(dataset_base, prefix="test")
    result_base = Config.MODEL_DIR / result_tag
    return (
        result_base.with_name(result_base.name + "_results.csv"),
        result_base.with_name(result_base.name + "_seed_metrics.csv"),
        result_base.with_name(result_base.name + "_plot.svg"),
    )


def load_test_data():
    model_dir = resolve_model_dir()
    configure_model_artifact_paths(model_dir)
    metadata = load_training_metadata()
    print(f">>> Using model directory: {model_dir}")
    print(f">>> Dataset search dirs: {format_dataset_dir_text()}")
    if metadata:
        print(f">>> Loaded training metadata from: {Config.TRAINING_METADATA_PATH}")
    print(f">>> Wavelet image dir: {Config.WAVELET_IMAGE_DIR}")

    test_csv_path = resolve_test_csv_path()
    result_save_path, seed_result_path, fig_path = build_result_paths(test_csv_path)
    print(f">>> Loading CSV from: {test_csv_path}")
    if metadata.get("dataset_base"):
        metadata_base = str(metadata["dataset_base"])
        test_base = dataset_base_name_from_csv(test_csv_path)
        if metadata_base != test_base:
            raise ValueError(
                "模型运行目录与测试集不匹配："
                f"metadata dataset_base={metadata_base}, test dataset_base={test_base}"
            )

    try:
        df = pd.read_csv(test_csv_path, low_memory=False)
    except Exception as exc:
        print(f"Error reading CSV: {exc}")
        return None

    validate_dataframe(df, test_csv_path)
    df = filter_valid_rows(df)

    image_paths = build_image_paths(df[Config.IMAGE_COL].values)
    labels = df[Config.LABEL_COL].values.astype(np.float32)
    if Config.YIELDED_COL in df.columns:
        yielded_flags = df[Config.YIELDED_COL].fillna(0).astype(bool).values
        yielded_col_used = Config.YIELDED_COL
    elif "steel02_yielded" in df.columns:
        yielded_flags = df["steel02_yielded"].fillna(0).astype(bool).values
        yielded_col_used = "steel02_yielded"
    elif "steel01_yielded" in df.columns:
        yielded_flags = df["steel01_yielded"].fillna(0).astype(bool).values
        yielded_col_used = "steel01_yielded"
    else:
        yielded_flags = np.zeros(len(df), dtype=bool)
        yielded_col_used = None
    if yielded_col_used:
        print(
            f">>> Yield marker: {yielded_col_used}, yielded rows: "
            f"{int(np.sum(yielded_flags))}/{len(yielded_flags)}"
        )
    else:
        print(">>> Yield marker: none, yielded rows: 0")
    expected_feature_names = metadata.get("scalar_feature_names")
    if expected_feature_names:
        layout_width = int(
            max(
                Config.MAX_DAMPER_FLOORS,
                df["num_floors"].max(),
                len([col for col in expected_feature_names if str(col).startswith("damper_story_")]),
            )
        )
        scalar_feature_df, scalar_feature_names = build_scalar_feature_frame(
            df,
            layout_width=layout_width,
            expected_feature_names=expected_feature_names,
        )
        Config.SCALAR_COLS = list(scalar_feature_names)
    else:
        scalar_feature_df = df[Config.BASE_SCALAR_COLS].copy()
        Config.SCALAR_COLS = list(Config.BASE_SCALAR_COLS)
    print(f">>> Scalar feature columns ({len(Config.SCALAR_COLS)}): {Config.SCALAR_COLS}")
    raw_scalars = scalar_feature_df.values.astype(np.float32)

    print(f">>> Loading Scalar Scaler from: {Config.SCALER_PATH}")
    try:
        scaler = joblib.load(Config.SCALER_PATH)
        scaler_feature_count = getattr(scaler, "n_features_in_", None)
        if scaler_feature_count is not None and int(scaler_feature_count) != int(raw_scalars.shape[1]):
            raise ValueError(
                "测试脚本构造的标量特征维度与训练时 scaler 不一致："
                f"scaler={int(scaler_feature_count)}, test_features={int(raw_scalars.shape[1])}, "
                f"features={Config.SCALAR_COLS}"
            )
        scalars_norm = scaler.transform(raw_scalars)
    except FileNotFoundError:
        print("Error: Scalar scaler file not found!")
        return None

    transform = transforms.Compose(
        [
            transforms.Resize(Config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMAGE_NORMALIZE_MEAN, std=Config.IMAGE_NORMALIZE_STD),
        ]
    )

    dataset = TestDataset(image_paths, scalars_norm, labels, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=(Config.DEVICE.type == "cuda"),
    )

    return {
        "loader": loader,
        "df": df,
        "scalar_feature_df": scalar_feature_df,
        "yielded_flags": yielded_flags,
        "yielded_col_used": yielded_col_used,
        "image_paths": image_paths,
        "result_save_path": result_save_path,
        "seed_result_path": seed_result_path,
        "fig_path": fig_path,
    }


def evaluate():
    test_data = load_test_data()
    if test_data is None:
        return

    test_loader = test_data["loader"]
    test_df = test_data["df"]
    scalar_feature_df = test_data["scalar_feature_df"]
    yielded_flags = test_data["yielded_flags"]
    yielded_col_used = test_data["yielded_col_used"]
    image_paths = test_data["image_paths"]
    result_save_path = test_data["result_save_path"]
    seed_result_path = test_data["seed_result_path"]
    fig_path = test_data["fig_path"]

    print(f"\n>>> Loading model weights from: {Config.MODEL_WEIGHTS_PATH}")
    model = MultimodalPredictor(num_scalars=len(Config.SCALAR_COLS)).to(Config.DEVICE)
    try:
        model.load_state_dict(torch.load(Config.MODEL_WEIGHTS_PATH, map_location=Config.DEVICE))
        print(">>> Weights loaded successfully.")
    except FileNotFoundError:
        print("Error: .pth weight file not found!")
        return

    model.eval()
    true_labels = []
    pred_labels = []
    img_paths_list = []

    print(f"\n>>> Starting inference with Post-Process Factor: {Config.POST_PROCESS_FACTOR} ...")
    with torch.no_grad():
        for images, scalars, labels, paths in tqdm(test_loader, desc="Evaluating"):
            images = images.to(Config.DEVICE, non_blocking=True)
            scalars = scalars.to(Config.DEVICE, non_blocking=True)

            raw_preds = model(images, scalars)
            base_preds = raw_preds.cpu().numpy().flatten()
            if Config.SCALE_TARGET:
                base_preds = base_preds / Config.LABEL_SCALE
            corrected_preds = base_preds * Config.POST_PROCESS_FACTOR

            true_labels.extend(labels.numpy())
            pred_labels.extend(corrected_preds)
            img_paths_list.extend(paths)

    true_labels = np.asarray(true_labels, dtype=np.float32)
    pred_labels = np.asarray(pred_labels, dtype=np.float32)
    metrics = calculate_metrics(true_labels, pred_labels)
    relative_errors = calculate_relative_errors(true_labels, pred_labels)

    print("\n" + "=" * 60)
    print("2D-CNN Multimodal Model Evaluation Results (Floors 3 to 7)")
    print(f"(Post-process Factor: x{Config.POST_PROCESS_FACTOR})")
    print("=" * 60)
    print(f"  Samples  : {metrics['Samples']}")
    print(f"  R2 Score : {metrics['R2']:.4f}")
    print(f"  MAE      : {metrics['MAE']:.6f} (rad)")
    print(f"  RMSE     : {metrics['RMSE']:.6f} (rad)")
    print(f"  MAPE     : {metrics['MAPE']:.2f}%")
    print("=" * 60 + "\n")

    seed_metrics_df = build_seed_metrics_dataframe(
        true_labels,
        pred_labels,
        Config.TEST_SEEDS,
        sample_ratio=Config.TEST_SAMPLE_RATIO,
        sample_with_replacement=Config.TEST_SAMPLE_WITH_REPLACEMENT,
    )
    print_seed_summary(seed_metrics_df)

    columns = ["sample_id", "split", "stage", Config.TXT_COL, Config.IMAGE_COL] + Config.BASE_SCALAR_COLS + [Config.DAMPER_LAYOUT_COL, Config.LABEL_COL]
    yield_columns = [col for col in ("steel01_yielded", "steel02_yielded") if col in test_df.columns]
    available_columns = list(dict.fromkeys([col for col in columns + yield_columns if col in test_df.columns]))
    result_df = test_df[available_columns].copy()
    for feature_name in Config.SCALAR_COLS:
        result_df[f"feature__{feature_name}"] = scalar_feature_df[feature_name].values
    result_df["TXT_Path"] = test_df[Config.TXT_COL].values
    result_df["Image_Path"] = img_paths_list
    result_df["True_Drift"] = true_labels
    result_df["Pred_Drift"] = pred_labels
    result_df["Abs_Error"] = np.abs(true_labels - pred_labels)
    result_df["Error_Pct"] = relative_errors
    result_df["Yielded_Flag_Used"] = yielded_flags
    result_df["Yielded_Column_Used"] = yielded_col_used or ""
    result_df.to_csv(result_save_path, index=False, encoding="utf-8-sig")
    print(f">>> Detailed predictions saved to: {result_save_path}")

    build_seed_metrics_report(seed_metrics_df).to_csv(seed_result_path, index=False, encoding="utf-8-sig")
    print(f">>> Multi-seed summary saved to: {seed_result_path}")

    plot_results(
        true_labels,
        pred_labels,
        yielded_flags,
        metrics["R2"],
        metrics["MAE"],
        metrics["RMSE"],
        fig_path,
        yielded_label=yielded_col_used or "Yielded",
    )


def plot_results(y_true, y_pred, yielded_flags, r2, mae, rmse, fig_path, yielded_label="Yielded"):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    yielded_flags = np.asarray(yielded_flags, dtype=bool)
    elastic_flags = ~yielded_flags
    if np.any(elastic_flags):
        plt.scatter(y_true[elastic_flags], y_pred[elastic_flags], alpha=0.3, s=5, c="blue", label="Predictions")
    if np.any(yielded_flags):
        plt.scatter(
            y_true[yielded_flags],
            y_pred[yielded_flags],
            alpha=0.5,
            s=5,
            c="orange",
            label=f"Yielded ({yielded_label})",
        )
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Fit (y=x)")
    plt.title(f"True vs Predicted Drift Ratio\n(R2={r2:.4f})")
    plt.xlabel("True Drift Ratio")
    plt.ylabel("Predicted Drift Ratio")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, color="purple", alpha=0.7, edgecolor="black")
    plt.axvline(x=0, color="r", linestyle="--", linewidth=2)
    plt.title(f"Error Distribution (Pred - True)\nMAE={mae:.6f}, RMSE={rmse:.6f}")
    plt.xlabel("Error Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    rel_errors = calculate_relative_errors(y_true, y_pred)
    rel_errors_clipped = np.clip(rel_errors, 0, 100)
    plt.hist(rel_errors_clipped, bins=50, color="green", alpha=0.7, edgecolor="black")
    plt.axvline(
        x=np.median(rel_errors),
        color="r",
        linestyle="--",
        linewidth=2,
        label=f"Median: {np.median(rel_errors):.1f}%",
    )
    plt.title("Relative Error Distribution (%)")
    plt.xlabel("Relative Error (%)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f">>> Evaluation plot saved to: {fig_path}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("2D-CNN Multimodal Model Testing Script (Floors 3 to 7)")
    print("=" * 60 + "\n")

    evaluate()

    if Config.SCALE_TARGET:
        print("\n提示: 预测值由模型输出除以 1000 还原得到。")
        print("例如: 模型输出 5.0，还原后代表 drift ratio 为 0.005。")
