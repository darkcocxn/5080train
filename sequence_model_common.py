# -*- coding: utf-8 -*-
"""
Shared utilities for sequence-based surrogate models.

The LSTM and WaveNet scripts keep their own version directories, while this
module centralizes data loading, scalar features, losses, metrics, and the
train/eval loops so both algorithms stay comparable.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm


def _as_path(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(str(value))


def parse_env_path_list(value: str | None) -> tuple[Path, ...]:
    if not value:
        return ()
    parts = [item.strip() for item in value.split(os.pathsep)]
    return tuple(Path(item).expanduser() for item in parts if item)


def apply_common_environment_overrides(config) -> None:
    dataset_dirs = parse_env_path_list(
        os.environ.get("SURMOD_DATASET_DIRS") or os.environ.get("SURMOD_DATASET_DIR")
    )
    if dataset_dirs:
        config.CSV_DIR_CANDIDATES = dataset_dirs
        config.CSV_DIR = next((path for path in dataset_dirs if path.exists()), dataset_dirs[0])

    for env_name, attr_name in (
        ("SURMOD_TRAIN_CSV", "TRAIN_CSV_PATH"),
        ("SURMOD_VAL_CSV", "VAL_CSV_PATH"),
        ("SURMOD_TEST_CSV", "TEST_CSV_PATH"),
    ):
        value = os.environ.get(env_name)
        if value:
            setattr(config, attr_name, Path(value).expanduser())

    model_root_dir = os.environ.get("SURMOD_MODEL_ROOT_DIR")
    if model_root_dir:
        path = Path(model_root_dir).expanduser()
        config.MODEL_ROOT_DIR = path
        config.MODEL_DIR = path
        config.SAVE_ROOT_DIR = path
        config.SAVE_DIR = path

    data_use_ratio = os.environ.get("SURMOD_DATA_USE_RATIO")
    if data_use_ratio:
        config.DATA_USE_RATIO = float(data_use_ratio)

    num_epochs = os.environ.get("SURMOD_NUM_EPOCHS")
    if num_epochs:
        config.NUM_EPOCHS = int(num_epochs)

    batch_size = os.environ.get("SURMOD_BATCH_SIZE")
    if batch_size:
        config.BATCH_SIZE = int(batch_size)

    num_workers = os.environ.get("SURMOD_NUM_WORKERS")
    if num_workers:
        config.NUM_WORKERS = int(num_workers)


def get_existing_dataset_dirs(config) -> list[Path]:
    existing_dirs = [path for path in config.CSV_DIR_CANDIDATES if path.exists()]
    return existing_dirs or [config.CSV_DIR]


def format_dataset_dir_text(config) -> str:
    return ", ".join(str(path) for path in get_existing_dataset_dirs(config))


def resolve_explicit_csv_path(config, explicit_path: str | Path) -> Path:
    path = Path(explicit_path).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(config.PROJECT_ROOT / path)
        candidates.extend(dataset_dir / path for dataset_dir in get_existing_dataset_dirs(config))

    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)

    for candidate in ordered:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Explicit CSV does not exist: {explicit_path}; checked: "
        + ", ".join(str(candidate) for candidate in ordered)
    )


def resolve_csv_path(config, explicit_path: str | Path | None, pattern: str) -> Path:
    if explicit_path:
        return resolve_explicit_csv_path(config, explicit_path)

    candidates: list[Path] = []
    for dataset_dir in get_existing_dataset_dirs(config):
        candidates.extend(dataset_dir.glob(pattern))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No CSV matched '{pattern}' in: {format_dataset_dir_text(config)}"
        )
    return candidates[0]


def dataset_base_name_from_csv(csv_path: Path) -> str:
    for suffix in ("_train.csv", "_val.csv", "_test.csv"):
        if csv_path.name.endswith(suffix):
            return csv_path.name[: -len(suffix)]
    raise ValueError(f"Cannot parse dataset base name from: {csv_path.name}")


def build_compact_artifact_name(dataset_base: str, prefix: str) -> str:
    stamp_match = re.search(r"(\d{8}-\d{6})", str(dataset_base))
    stamp = stamp_match.group(1) if stamp_match else "nostamp"
    digest = hashlib.sha1(str(dataset_base).encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{stamp}-{digest}"


def resolve_latest_complete_dataset_paths(config) -> tuple[Path, Path, str]:
    candidates: list[tuple[float, Path, Path, str]] = []
    incomplete: list[str] = []
    for dataset_dir in get_existing_dataset_dirs(config):
        for train_csv in dataset_dir.glob(config.TRAIN_FILE_PATTERN):
            base = dataset_base_name_from_csv(train_csv)
            val_csv = train_csv.with_name(f"{base}_val.csv")
            if not val_csv.exists():
                incomplete.append(str(train_csv))
                continue
            mtime = max(train_csv.stat().st_mtime, val_csv.stat().st_mtime)
            summary_path = train_csv.with_name(f"{base}_summary.json")
            if summary_path.exists():
                mtime = max(mtime, summary_path.stat().st_mtime)
            candidates.append((mtime, train_csv, val_csv, base))

    if not candidates:
        if incomplete:
            raise FileNotFoundError(
                "No complete train/val dataset found. Missing val for: "
                + ", ".join(sorted(set(incomplete)))
            )
        raise FileNotFoundError(
            f"No train CSV matched '{config.TRAIN_FILE_PATTERN}' in: "
            f"{format_dataset_dir_text(config)}"
        )
    _, train_csv, val_csv, base = max(candidates, key=lambda item: item[0])
    return train_csv, val_csv, base


def resolve_dataset_paths(config) -> tuple[Path, Path, str]:
    if config.TRAIN_CSV_PATH and config.VAL_CSV_PATH:
        train_csv = resolve_csv_path(config, config.TRAIN_CSV_PATH, config.TRAIN_FILE_PATTERN)
        val_csv = resolve_csv_path(config, config.VAL_CSV_PATH, config.VAL_FILE_PATTERN)
        train_base = dataset_base_name_from_csv(train_csv)
        val_base = dataset_base_name_from_csv(val_csv)
        if train_base != val_base:
            raise ValueError(f"Train/val CSV bases differ: {train_base} vs {val_base}")
        return train_csv, val_csv, train_base

    if config.TRAIN_CSV_PATH and not config.VAL_CSV_PATH:
        train_csv = resolve_csv_path(config, config.TRAIN_CSV_PATH, config.TRAIN_FILE_PATTERN)
        base = dataset_base_name_from_csv(train_csv)
        val_csv = train_csv.with_name(f"{base}_val.csv")
        if not val_csv.exists():
            raise FileNotFoundError(f"Paired val CSV does not exist: {val_csv}")
        return train_csv, val_csv, base

    if config.VAL_CSV_PATH and not config.TRAIN_CSV_PATH:
        val_csv = resolve_csv_path(config, config.VAL_CSV_PATH, config.VAL_FILE_PATTERN)
        base = dataset_base_name_from_csv(val_csv)
        train_csv = val_csv.with_name(f"{base}_train.csv")
        if not train_csv.exists():
            raise FileNotFoundError(f"Paired train CSV does not exist: {train_csv}")
        return train_csv, val_csv, base

    return resolve_latest_complete_dataset_paths(config)


def configure_model_dir(config, dataset_base: str) -> Path:
    config.MODEL_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    if not config.UNIQUE_MODEL_RUN_DIR:
        config.MODEL_DIR = config.MODEL_ROOT_DIR
        config.SAVE_DIR = config.MODEL_ROOT_DIR
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        return config.MODEL_DIR

    run_name = build_compact_artifact_name(dataset_base, prefix="model")
    run_name = f"{run_name}-train{time.strftime('%Y%m%d-%H%M%S')}"
    model_dir = config.MODEL_ROOT_DIR / run_name
    model_dir.mkdir(parents=True, exist_ok=False)
    config.MODEL_DIR = model_dir
    config.SAVE_DIR = model_dir
    return model_dir


def resolve_model_dir(config) -> Path:
    root = Path(config.MODEL_ROOT_DIR)
    explicit_model_dir = os.environ.get(f"SURMOD_{config.ENV_PREFIX}_MODEL_DIR")
    if explicit_model_dir:
        path = Path(explicit_model_dir).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Model directory does not exist: {path}")
        return path

    if (root / "training_metadata.json").exists():
        return root
    candidates = [
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "training_metadata.json").exists()
    ] if root.exists() else []
    if not candidates:
        raise FileNotFoundError(f"No model run with training_metadata.json found under: {root}")
    return max(candidates, key=lambda path: (path / "training_metadata.json").stat().st_mtime)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def sample_dataframe_by_group(df: pd.DataFrame, group_col: str, ratio: float, seed: int) -> pd.DataFrame:
    if ratio >= 1.0:
        print(f">>> Using 100% of data: {len(df)} samples")
        return df.reset_index(drop=True)
    unique_groups = df[group_col].drop_duplicates().tolist()
    if not unique_groups:
        return df.reset_index(drop=True)
    use_count = max(1, int(round(len(unique_groups) * ratio)))
    use_count = min(use_count, len(unique_groups))
    rng = np.random.RandomState(seed)
    selected = set(rng.choice(unique_groups, size=use_count, replace=False))
    sampled = df[df[group_col].isin(selected)].reset_index(drop=True)
    print(
        f">>> Using {ratio * 100:.1f}% groups: "
        f"{use_count}/{len(unique_groups)} groups, {len(sampled)}/{len(df)} rows"
    )
    return sampled


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_amp_autocast_context(config, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast(device_type=config.DEVICE.type, enabled=True)


def build_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_adamw_optimizer(config, model: nn.Module) -> optim.Optimizer:
    if not getattr(config, "OPTIMIZER_NO_DECAY_NORM_AND_BIAS", True):
        return optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    decay_params = []
    no_decay_params = []
    norm_keywords = ("norm", "bn", "ln")
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if param.ndim <= 1 or name.endswith(".bias") or any(key in lower for key in norm_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.LEARNING_RATE,
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
        self.backup: dict[str, torch.Tensor] | None = None

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, tensor in model.state_dict().items():
                tensor = tensor.detach()
                shadow = self.shadow[name]
                if torch.is_floating_point(shadow):
                    shadow.mul_(self.decay).add_(tensor, alpha=1.0 - self.decay)
                else:
                    shadow.copy_(tensor)

    def store(self, model: nn.Module) -> None:
        self.backup = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def restore(self, model: nn.Module) -> None:
        if self.backup is None:
            return
        model.load_state_dict(self.backup, strict=True)
        self.backup = None

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}


def apply_lr_warmup(config, optimizer: optim.Optimizer, epoch_idx: int) -> None:
    if config.WARMUP_EPOCHS <= 0 or epoch_idx >= config.WARMUP_EPOCHS:
        return
    progress = float(epoch_idx + 1) / float(config.WARMUP_EPOCHS)
    factor = config.WARMUP_START_FACTOR + (1.0 - config.WARMUP_START_FACTOR) * progress
    for param_group in optimizer.param_groups:
        param_group["lr"] = config.LEARNING_RATE * factor


def build_scalar_norm(config, num_features: int) -> nn.Module:
    norm_type = str(config.SCALAR_NORM).lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(num_features)
    if norm_type == "layer":
        return nn.LayerNorm(num_features)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported scalar norm: {config.SCALAR_NORM}")


def _resolve_group_count(num_channels: int, max_groups: int) -> int:
    max_groups = max(1, min(int(max_groups), int(num_channels)))
    for group_count in range(max_groups, 0, -1):
        if int(num_channels) % group_count == 0:
            return group_count
    return 1


def build_conv1d_norm(config, num_channels: int) -> nn.Module:
    norm_type = str(config.SEQ_NORM).lower()
    if norm_type == "batch":
        return nn.BatchNorm1d(num_channels)
    if norm_type == "group":
        return nn.GroupNorm(_resolve_group_count(num_channels, config.SEQ_GROUP_NORM_MAX_GROUPS), num_channels)
    if norm_type == "instance":
        return nn.InstanceNorm1d(num_channels, affine=True)
    if norm_type in {"none", "identity"}:
        return nn.Identity()
    raise ValueError(f"Unsupported 1D norm: {config.SEQ_NORM}")


class ScalarResidualBlock(nn.Module):
    def __init__(self, config, dim: int, hidden_mult: int, dropout: float):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)
        self.norm = build_scalar_norm(config, dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(config.SCALAR_RESIDUAL_SCALE_INIT)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.residual_scale * self.net(self.norm(x))


class ScalarFeatureEncoder(nn.Module):
    def __init__(self, config, num_scalars: int):
        super().__init__()
        dim = int(config.SCALAR_EMBED_DIM)
        blocks: list[nn.Module] = [
            nn.Linear(num_scalars, dim),
            build_scalar_norm(config, dim),
            nn.GELU(),
            nn.Dropout(config.SCALAR_INPUT_DROPOUT),
        ]
        blocks.extend(
            ScalarResidualBlock(
                config,
                dim=dim,
                hidden_mult=config.SCALAR_RES_HIDDEN_MULT,
                dropout=config.SCALAR_RES_DROPOUT,
            )
            for _ in range(config.SCALAR_RES_BLOCKS)
        )
        blocks.append(build_scalar_norm(config, dim))
        self.encoder = nn.Sequential(*blocks)
        self.out_dim = dim

    def forward(self, scalars: torch.Tensor) -> torch.Tensor:
        return self.encoder(scalars)


class GatedBilinearFusionBlock(nn.Module):
    def __init__(self, config, sequence_dim: int, scalar_dim: int):
        super().__init__()
        bilinear_dim = int(config.FUSION_BILINEAR_DIM)
        output_dim = int(config.FUSION_OUTPUT_DIM)
        self.seq_gate = nn.Linear(scalar_dim, sequence_dim)
        self.scalar_gate = nn.Linear(sequence_dim, scalar_dim)
        self.seq_bilinear = nn.Linear(sequence_dim, bilinear_dim, bias=False)
        self.scalar_bilinear = nn.Linear(scalar_dim, bilinear_dim, bias=False)
        self.interaction_scale = nn.Parameter(torch.tensor(float(config.FUSION_INTERACTION_SCALE_INIT)))
        self.out = nn.Sequential(
            nn.Linear(sequence_dim + scalar_dim + bilinear_dim, output_dim),
            build_scalar_norm(config, output_dim),
            nn.GELU(),
            nn.Dropout(config.FUSION_DROPOUT),
        )
        nn.init.zeros_(self.seq_gate.weight)
        nn.init.zeros_(self.seq_gate.bias)
        nn.init.zeros_(self.scalar_gate.weight)
        nn.init.zeros_(self.scalar_gate.bias)

    def forward(self, seq_features: torch.Tensor, scalar_features: torch.Tensor) -> torch.Tensor:
        seq_factor = 1.0 + 0.5 * torch.tanh(self.seq_gate(scalar_features))
        scalar_factor = 1.0 + 0.5 * torch.tanh(self.scalar_gate(seq_features))
        seq_gated = seq_features * seq_factor
        scalar_gated = scalar_features * scalar_factor
        bilinear = self.interaction_scale.clamp(0.0, 2.0) * (
            self.seq_bilinear(seq_gated) * self.scalar_bilinear(scalar_gated)
        )
        return self.out(torch.cat((seq_gated, scalar_gated, bilinear), dim=1))


class SequenceFusionRegressor(nn.Module):
    def __init__(self, config, sequence_dim: int, scalar_dim: int):
        super().__init__()
        self.config = config
        if config.FUSION_MODE == "gated_bilinear":
            self.fusion = GatedBilinearFusionBlock(config, sequence_dim, scalar_dim)
            fusion_dim = int(config.FUSION_OUTPUT_DIM)
        elif config.FUSION_MODE == "concat":
            self.fusion = None
            fusion_dim = int(sequence_dim + scalar_dim)
        else:
            raise ValueError(f"Unsupported fusion mode: {config.FUSION_MODE}")

        h0, h1 = config.HEAD_HIDDEN_DIMS
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, h0),
            nn.GELU(),
            nn.Dropout(config.HEAD_DROPOUT),
            nn.Linear(h0, h1),
            nn.GELU(),
            nn.Linear(h1, 1),
        )

        self.use_tail_classifier = bool(config.USE_TAIL_CLASSIFICATION_AUX)
        if self.use_tail_classifier:
            self.tail_classifier_head = nn.Sequential(
                nn.Linear(fusion_dim, config.TAIL_CLASSIFICATION_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(config.TAIL_CLASSIFICATION_DROPOUT),
                nn.Linear(config.TAIL_CLASSIFICATION_HIDDEN_DIM, len(config.TAIL_CLASSIFICATION_THRESHOLDS)),
            )
            nn.init.constant_(self.tail_classifier_head[-1].bias, config.TAIL_CLASSIFICATION_INIT_BIAS)

        self.use_tail_correction = bool(config.USE_TAIL_CORRECTION_HEAD)
        if self.use_tail_correction:
            self.tail_correction_head = nn.Sequential(
                nn.Linear(fusion_dim, config.TAIL_CORRECTION_HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(config.TAIL_CORRECTION_DROPOUT),
                nn.Linear(config.TAIL_CORRECTION_HIDDEN_DIM, 1),
            )
            self.tail_correction_activation = nn.Softplus()
            nn.init.constant_(self.tail_correction_head[-1].bias, config.TAIL_CORRECTION_INIT_BIAS)
            if config.USE_TAIL_CORRECTION_GATE:
                self.tail_correction_gate = nn.Linear(fusion_dim, 1)
                nn.init.zeros_(self.tail_correction_gate.weight)
                nn.init.constant_(self.tail_correction_gate.bias, config.TAIL_CORRECTION_GATE_INIT_BIAS)
            else:
                self.tail_correction_gate = None

    def forward(
        self,
        sequence_features: torch.Tensor,
        scalar_features: torch.Tensor,
        return_aux: bool = False,
    ):
        if self.fusion is None:
            fused = torch.cat((sequence_features, scalar_features), dim=1)
        else:
            fused = self.fusion(sequence_features, scalar_features)

        base_pred = self.head(fused)
        aux: dict[str, torch.Tensor] = {}
        if self.use_tail_classifier:
            aux["tail_logits"] = self.tail_classifier_head(fused)

        if self.use_tail_correction:
            correction = self.tail_correction_activation(self.tail_correction_head(fused))
            if self.tail_correction_gate is not None:
                gate = torch.sigmoid(self.tail_correction_gate(fused))
                if self.config.USE_TAIL_PROB_GATED_CORRECTION and self.use_tail_classifier:
                    gate_index = min(
                        max(int(self.config.TAIL_PROB_GATE_INDEX), 0),
                        aux["tail_logits"].shape[1] - 1,
                    )
                    tail_prob_gate = torch.sigmoid(aux["tail_logits"][:, gate_index : gate_index + 1])
                    if self.config.TAIL_PROB_GATE_DETACH:
                        tail_prob_gate = tail_prob_gate.detach()
                    if self.config.TAIL_PROB_GATE_POWER != 1.0:
                        tail_prob_gate = tail_prob_gate.clamp_min(1e-6).pow(self.config.TAIL_PROB_GATE_POWER)
                    gate = gate * tail_prob_gate
                correction = correction * gate
            pred = base_pred + correction
        else:
            pred = base_pred

        if return_aux:
            return pred, aux
        return pred


def validate_dataframe(config, df: pd.DataFrame, csv_path: Path) -> None:
    required = config.BASE_SCALAR_COLS + [config.TXT_COL, config.LABEL_COL, config.STATUS_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns {missing}: {csv_path}")


def filter_valid_rows(config, df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df[config.STATUS_COL] == "ok"].copy()
    filtered[config.LABEL_COL] = pd.to_numeric(filtered[config.LABEL_COL], errors="coerce")
    filtered = filtered.replace([np.inf, -np.inf], np.nan)
    filtered = filtered.dropna(subset=config.BASE_SCALAR_COLS + [config.TXT_COL, config.LABEL_COL])
    return filtered.reset_index(drop=True)


def parse_damper_layout_flags(layout_value, num_floors: int, layout_width: int) -> list[int]:
    floor_count = max(0, min(int(num_floors), int(layout_width)))
    flags = [0] * int(layout_width)
    if floor_count <= 0:
        return flags
    if layout_value is None or (isinstance(layout_value, float) and np.isnan(layout_value)):
        parsed_bits = [1] * floor_count
    else:
        bits = re.findall(r"[01]", str(layout_value).strip())
        if bits:
            parsed_bits = [int(bit) for bit in bits[:floor_count]]
            if len(parsed_bits) < floor_count:
                parsed_bits.extend([1] * (floor_count - len(parsed_bits)))
        else:
            parsed_bits = [1] * floor_count
    flags[:floor_count] = parsed_bits[:floor_count]
    return flags


def get_available_wave_feature_cols(config, df: pd.DataFrame) -> list[str]:
    return [col for col in config.WAVE_FEATURE_COLS if col in df.columns]


def _finite_series(values, default: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.nan_to_num(array, nan=default, posinf=default, neginf=default)


def add_wave_derived_features(config, feature_df: pd.DataFrame) -> list[str]:
    if not config.USE_WAVE_DERIVED_FEATURES:
        return []
    names: list[str] = []
    eps = 1e-6
    for col in config.WAVE_LOG_FEATURE_COLS:
        if col not in feature_df.columns:
            continue
        values = np.maximum(_finite_series(feature_df[col]), 0.0)
        name = f"log1p_{col}"
        feature_df[name] = np.log1p(values).astype(np.float32)
        names.append(name)

    if "period_1_sec" not in feature_df.columns:
        return names
    period = np.clip(_finite_series(feature_df["period_1_sec"], default=eps), eps, None)
    if "wave_predominant_period" in feature_df.columns:
        wave_period = np.clip(_finite_series(feature_df["wave_predominant_period"], default=eps), eps, None)
        ratio = np.clip(wave_period / period, eps, 20.0)
        inverse_ratio = np.clip(period / wave_period, eps, 20.0)
        feature_df["wave_to_structure_period_ratio"] = ratio.astype(np.float32)
        feature_df["structure_to_wave_period_ratio"] = inverse_ratio.astype(np.float32)
        feature_df["wave_structure_period_log_gap"] = np.abs(np.log(ratio)).astype(np.float32)
        names.extend([
            "wave_to_structure_period_ratio",
            "structure_to_wave_period_ratio",
            "wave_structure_period_log_gap",
        ])
    if "wave_dominant_freq" in feature_df.columns:
        values = np.maximum(_finite_series(feature_df["wave_dominant_freq"]), 0.0)
        feature_df["wave_dominant_freq_x_period"] = np.clip(values * period, 0.0, 20.0).astype(np.float32)
        names.append("wave_dominant_freq_x_period")
    if "wave_spectral_centroid" in feature_df.columns:
        values = np.maximum(_finite_series(feature_df["wave_spectral_centroid"]), 0.0)
        feature_df["wave_spectral_centroid_x_period"] = np.clip(values * period, 0.0, 20.0).astype(np.float32)
        names.append("wave_spectral_centroid_x_period")
    for col in ("wave_pga", "wave_cav", "wave_arias_proxy", "wave_intensity_score"):
        if col not in feature_df.columns:
            continue
        values = np.maximum(_finite_series(feature_df[col]), 0.0)
        name = f"{col}_x_period"
        feature_df[name] = (values * period).astype(np.float32)
        names.append(name)
    return names


def add_structure_derived_features(feature_df: pd.DataFrame) -> list[str]:
    eps = 1e-6
    floors = np.maximum(_finite_series(feature_df["num_floors"], default=1.0), 1.0)
    mass = np.maximum(_finite_series(feature_df["floor_mass"]), eps)
    height = np.maximum(_finite_series(feature_df["floor_height"]), eps)
    stiffness = np.maximum(_finite_series(feature_df["k_base_1_4"]), eps)
    fy_add = np.maximum(_finite_series(feature_df["Fy_add"]), eps)
    period = np.maximum(_finite_series(feature_df.get("period_1_sec", np.zeros(len(feature_df)))), eps)
    values = {
        "inv_k_base_1_4": 1.0 / stiffness,
        "inv_Fy_add": 1.0 / fy_add,
        "mass_to_stiffness": mass / stiffness,
        "height_to_stiffness": height / stiffness,
        "period_squared": period**2,
        "num_floors_x_period": floors * period,
        "floor_mass_x_period": mass * period,
        "floor_height_x_period": height * period,
        "flexibility_x_period": (mass / stiffness) * period,
        "strength_inverse_x_period": (1.0 / fy_add) * period,
    }
    for name, arr in values.items():
        feature_df[name] = arr.astype(np.float32)
    return list(values.keys())


def add_tail_risk_features(feature_df: pd.DataFrame) -> list[str]:
    eps = 1e-6
    floors = np.maximum(_finite_series(feature_df["num_floors"], default=1.0), 1.0)
    mass = np.maximum(_finite_series(feature_df["floor_mass"]), eps)
    stiffness = np.maximum(_finite_series(feature_df["k_base_1_4"]), eps)
    period = np.maximum(_finite_series(feature_df.get("period_1_sec", np.zeros(len(feature_df)))), eps)
    intensity = np.maximum(_finite_series(feature_df.get("wave_intensity_score", np.zeros(len(feature_df)))), 0.0)
    arias = np.maximum(_finite_series(feature_df.get("wave_arias_proxy", np.zeros(len(feature_df)))), 0.0)
    cav = np.maximum(_finite_series(feature_df.get("wave_cav", np.zeros(len(feature_df)))), 0.0)
    sparse = np.maximum(_finite_series(feature_df["damper_sparse_ratio"]), 0.0)
    tail_risk = period * floors * (mass / stiffness) * (1.0 + sparse)
    values = {
        "tail_risk_proxy": tail_risk,
        "log1p_tail_risk_proxy": np.log1p(np.maximum(tail_risk, 0.0)),
        "wave_intensity_tail_risk": tail_risk * (1.0 + intensity),
        "wave_arias_tail_risk": tail_risk * (1.0 + arias),
        "wave_cav_tail_risk": tail_risk * (1.0 + cav),
    }
    for name, arr in values.items():
        feature_df[name] = arr.astype(np.float32)
    return list(values.keys())


def build_scalar_feature_frame(
    config,
    df: pd.DataFrame,
    layout_width: int,
    expected_feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    wave_cols = get_available_wave_feature_cols(config, df)
    feature_df = df[config.BASE_SCALAR_COLS + wave_cols].copy()
    for col in feature_df.columns:
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")
        if feature_df[col].isna().any():
            median = feature_df[col].median()
            feature_df[col] = feature_df[col].fillna(0.0 if pd.isna(median) else median)

    derived = add_wave_derived_features(config, feature_df)
    structure_derived = add_structure_derived_features(feature_df)
    layout_names = [f"damper_story_{idx}" for idx in range(1, int(layout_width) + 1)]
    layout_matrix = np.zeros((len(df), int(layout_width)), dtype=np.float32)
    for row_idx, row in enumerate(df.itertuples(index=False)):
        num_floors = int(getattr(row, "num_floors"))
        layout_value = getattr(row, config.DAMPER_LAYOUT_COL, None) if hasattr(row, config.DAMPER_LAYOUT_COL) else None
        layout_matrix[row_idx, :] = parse_damper_layout_flags(layout_value, num_floors, layout_width)

    for col_idx, name in enumerate(layout_names):
        feature_df[name] = layout_matrix[:, col_idx]
    feature_df["damper_install_count"] = layout_matrix.sum(axis=1)
    feature_df["damper_install_ratio"] = np.divide(
        feature_df["damper_install_count"].astype(np.float32),
        np.maximum(feature_df["num_floors"].astype(np.float32), 1.0),
    )
    feature_df["damper_sparse_ratio"] = 1.0 - feature_df["damper_install_ratio"].astype(np.float32)
    tail_names = add_tail_risk_features(feature_df)
    feature_names = (
        config.BASE_SCALAR_COLS
        + wave_cols
        + derived
        + structure_derived
        + layout_names
        + ["damper_install_count", "damper_install_ratio", "damper_sparse_ratio"]
        + tail_names
    )
    if expected_feature_names:
        for name in expected_feature_names:
            if name not in feature_df.columns:
                feature_df[name] = 0.0
        feature_names = list(expected_feature_names)
    return feature_df[feature_names].astype(np.float32), feature_names


def _parse_h5_reference(reference: str) -> tuple[Path, str | None]:
    text = reference[len("h5://") :]
    if "|" in text:
        path_text, selector = text.rsplit("|", 1)
        selector = selector.strip()
    else:
        path_text, selector = text, None
    return Path(path_text).expanduser(), selector


def _fallback_h5_path(config, path: Path) -> Path:
    if path.exists():
        return path
    matches = list((config.PROJECT_ROOT / "newdata").rglob(path.name))
    if matches:
        return matches[0]
    matches = list(config.PROJECT_ROOT.rglob(path.name))
    if matches:
        return matches[0]
    return path


def _collect_h5_datasets(group, prefix: str = ""):
    import h5py  # type: ignore

    datasets = []
    for key, value in group.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(value, h5py.Dataset):
            if np.issubdtype(value.dtype, np.number):
                datasets.append((name, value))
        elif isinstance(value, h5py.Group):
            datasets.extend(_collect_h5_datasets(value, name))
    return datasets


def _select_h5_dataset(handle, selector: str | None):
    if selector and not selector.isdigit() and selector in handle:
        return handle[selector], None
    datasets = _collect_h5_datasets(handle)
    if not datasets:
        raise ValueError("HDF5 file has no numeric datasets")
    preferred = []
    for name, dataset in datasets:
        lower = name.lower()
        score = 0
        if any(token in lower for token in ("wave", "acc", "data", "record", "motion")):
            score += 10
        score += min(int(dataset.ndim), 3)
        score += 1 if dataset.shape and max(dataset.shape) > 100 else 0
        preferred.append((score, name, dataset))
    preferred.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return preferred[0][2], selector


def load_h5_sequence(config, reference: str) -> np.ndarray:
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "h5py is required for h5:// wave references. Install it with "
            "`uv add h5py>=3.11.0` or point SURMOD_DATASET_DIRS to a txt-based dataset."
        ) from exc

    path, selector = _parse_h5_reference(reference)
    path = _fallback_h5_path(config, path)
    if not path.exists():
        raise FileNotFoundError(f"HDF5 wave file does not exist: {path}")

    with h5py.File(path, "r") as handle:
        dataset, selector = _select_h5_dataset(handle, selector)
        if selector and selector.isdigit() and dataset.ndim >= 2:
            index = int(selector)
            if index >= dataset.shape[0] and index - 1 >= 0:
                index -= 1
            index = max(0, min(index, dataset.shape[0] - 1))
            values = np.asarray(dataset[index], dtype=np.float32).reshape(-1)
        else:
            values = np.asarray(dataset, dtype=np.float32).reshape(-1)
    return values


def load_txt_sequence(path: Path, wave_dt: float | None = None) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float32)
    data = np.asarray(data, dtype=np.float32).reshape(-1)
    if data.size > 16 and wave_dt is not None and np.isfinite(wave_dt):
        first = float(data[0])
        if 0.0 < first < 1.0 and abs(first - float(wave_dt)) <= max(1e-6, 0.05 * abs(float(wave_dt))):
            data = data[1:]
    return data


def resample_sequence(values: np.ndarray, source_dt: float | None, target_dt: float | None) -> np.ndarray:
    if target_dt is None or source_dt is None or not np.isfinite(source_dt):
        return values.astype(np.float32, copy=False)
    source_dt = float(source_dt)
    target_dt = float(target_dt)
    if source_dt <= 0.0 or target_dt <= 0.0 or abs(source_dt - target_dt) < 1e-9:
        return values.astype(np.float32, copy=False)
    if len(values) < 2:
        return values.astype(np.float32, copy=False)
    duration = (len(values) - 1) * source_dt
    target_count = max(2, int(round(duration / target_dt)) + 1)
    source_t = np.linspace(0.0, duration, num=len(values), dtype=np.float64)
    target_t = np.linspace(0.0, duration, num=target_count, dtype=np.float64)
    return np.interp(target_t, source_t, values.astype(np.float64)).astype(np.float32)


def pad_or_truncate_sequence(values: np.ndarray, seq_len: int) -> np.ndarray:
    seq_len = int(seq_len)
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(values) >= seq_len:
        return values[:seq_len].astype(np.float32, copy=False)
    padded = np.zeros(seq_len, dtype=np.float32)
    padded[: len(values)] = values
    return padded


def load_wave_sequence(config, reference, seq_len: int | None = None, wave_dt=None) -> np.ndarray:
    reference_text = str(reference).strip()
    if not reference_text or reference_text.lower() == "nan":
        raise ValueError("Empty wave reference")
    if reference_text.startswith("h5://"):
        values = load_h5_sequence(config, reference_text)
    else:
        path = Path(reference_text).expanduser()
        if not path.is_absolute():
            path = config.PROJECT_ROOT / path
        if not path.exists():
            matches = list(config.PROJECT_ROOT.rglob(path.name))
            if matches:
                path = matches[0]
        if not path.exists():
            raise FileNotFoundError(f"Wave txt file does not exist: {path}")
        values = load_txt_sequence(path, wave_dt=float(wave_dt) if wave_dt is not None else None)
    source_dt = float(wave_dt) if wave_dt is not None and pd.notna(wave_dt) else None
    values = resample_sequence(values, source_dt=source_dt, target_dt=config.TARGET_DT)
    return pad_or_truncate_sequence(values, int(seq_len or config.SEQ_LEN))


def build_sequence_channels(config, sequence: np.ndarray, sequence_scaler: dict) -> np.ndarray:
    mean = float(sequence_scaler.get("mean", 0.0))
    std = max(float(sequence_scaler.get("std", 1.0)), 1e-8)
    scaled = np.clip((sequence.astype(np.float32) - mean) / std, -config.SEQUENCE_CLIP_VALUE, config.SEQUENCE_CLIP_VALUE)
    channels = []
    for channel in config.SEQUENCE_CHANNELS:
        if channel == "acc_scaled":
            channels.append(scaled)
        elif channel == "abs_acc_scaled":
            channels.append(np.clip(np.abs(sequence.astype(np.float32)) / std, 0.0, config.SEQUENCE_CLIP_VALUE))
        elif channel == "energy_cumsum":
            energy = np.cumsum(np.square(sequence.astype(np.float32)))
            denom = float(energy[-1]) if energy.size and energy[-1] > 0 else 1.0
            channels.append((energy / denom).astype(np.float32))
        else:
            raise ValueError(f"Unsupported sequence channel: {channel}")
    return np.stack(channels, axis=0).astype(np.float32)


def fit_sequence_scaler(config, references, wave_dts) -> dict:
    unique: dict[str, object] = {}
    for ref, dt in zip(references, wave_dts):
        unique.setdefault(str(ref), dt)
    total = 0
    sum_value = 0.0
    sum_square = 0.0
    max_abs = 0.0
    print(f">>> Fitting sequence scaler on {len(unique)} unique wave references...")
    for ref, dt in tqdm(list(unique.items()), desc="Sequence scaler"):
        seq = load_wave_sequence(config, ref, seq_len=config.SEQ_LEN, wave_dt=dt)
        total += int(seq.size)
        sum_value += float(np.sum(seq, dtype=np.float64))
        sum_square += float(np.sum(np.square(seq, dtype=np.float64)))
        max_abs = max(max_abs, float(np.max(np.abs(seq))) if seq.size else 0.0)
    mean = sum_value / max(total, 1)
    variance = max(sum_square / max(total, 1) - mean * mean, 1e-12)
    std = math.sqrt(variance)
    return {
        "mean": float(mean),
        "std": float(std),
        "max_abs": float(max_abs),
        "target_dt": None if config.TARGET_DT is None else float(config.TARGET_DT),
        "seq_len": int(config.SEQ_LEN),
        "channels": list(config.SEQUENCE_CHANNELS),
        "clip_value": float(config.SEQUENCE_CLIP_VALUE),
    }


class SequenceDataset(Dataset):
    def __init__(
        self,
        config,
        df: pd.DataFrame,
        scalars: np.ndarray,
        labels: np.ndarray,
        sequence_scaler: dict,
        sample_weights: np.ndarray | None = None,
        cache_sequences: bool = True,
    ):
        self.config = config
        self.refs = df[config.TXT_COL].astype(str).values
        self.wave_dts = (
            pd.to_numeric(df[config.WAVE_DT_COL], errors="coerce").values
            if config.WAVE_DT_COL in df.columns
            else np.full(len(df), np.nan)
        )
        self.sample_ids = (
            df[config.SAMPLE_ID_COL].astype(str).values
            if config.SAMPLE_ID_COL in df.columns
            else np.arange(len(df)).astype(str)
        )
        self.scalars = torch.tensor(scalars, dtype=torch.float32)
        target = labels * config.LABEL_SCALE if config.SCALE_TARGET else labels
        self.labels = torch.tensor(target, dtype=torch.float32)
        if sample_weights is None:
            sample_weights = np.ones(len(df), dtype=np.float32)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        self.sequence_scaler = sequence_scaler
        self.cache_sequences = bool(cache_sequences)
        self._cache: dict[tuple[str, float], np.ndarray] = {}
        print(f">>> Sequence dataset size: {len(self.refs)} samples")

    def __len__(self) -> int:
        return len(self.refs)

    def _load_channels(self, idx: int) -> np.ndarray:
        ref = str(self.refs[idx])
        dt_value = self.wave_dts[idx]
        dt_key = float(dt_value) if np.isfinite(dt_value) else float("nan")
        key = (ref, dt_key)
        if self.cache_sequences and key in self._cache:
            return self._cache[key]
        seq = load_wave_sequence(self.config, ref, seq_len=self.config.SEQ_LEN, wave_dt=dt_value)
        channels = build_sequence_channels(self.config, seq, self.sequence_scaler)
        if self.cache_sequences:
            self._cache[key] = channels
        return channels

    def __getitem__(self, idx: int):
        sequence = torch.tensor(self._load_channels(idx), dtype=torch.float32)
        return (
            sequence,
            self.scalars[idx],
            self.labels[idx],
            self.sample_weights[idx],
            self.sample_ids[idx],
        )


def _build_target_bin_series(config, labels: pd.Series) -> pd.Series:
    unique_count = int(labels.nunique())
    if unique_count <= 1:
        return pd.Series(np.zeros(len(labels), dtype=int), index=labels.index)
    q = min(config.TARGET_BIN_COUNT, unique_count)
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
    return kernel / max(float(kernel.sum()), 1e-12)


def _smooth_tail_boost_np(config, values: np.ndarray, threshold: float, target_weight: float) -> np.ndarray:
    if not config.USE_SMOOTH_TAIL_WEIGHTS:
        return np.where(values >= threshold, target_weight, 1.0)
    width = max(float(config.TAIL_WEIGHT_TRANSITION_WIDTH), 1e-8)
    progress = np.clip((values - (threshold - width)) / width, 0.0, 1.0)
    progress = progress * progress * (3.0 - 2.0 * progress)
    return 1.0 + (float(target_weight) - 1.0) * progress


def _smooth_tail_boost_torch(config, labels_raw: torch.Tensor, threshold: float, target_weight: float) -> torch.Tensor:
    if not config.USE_SMOOTH_TAIL_WEIGHTS:
        return torch.where(
            labels_raw >= threshold,
            torch.full_like(labels_raw, float(target_weight)),
            torch.ones_like(labels_raw),
        )
    width = max(float(config.TAIL_WEIGHT_TRANSITION_WIDTH), 1e-8)
    progress = torch.clamp((labels_raw - (threshold - width)) / width, 0.0, 1.0)
    progress = progress * progress * (3.0 - 2.0 * progress)
    return 1.0 + (float(target_weight) - 1.0) * progress


def configure_adaptive_tail_policy(config, train_labels: np.ndarray, val_labels: np.ndarray | None = None) -> dict:
    train_labels = np.asarray(train_labels, dtype=np.float64)
    val_labels = np.asarray([] if val_labels is None else val_labels, dtype=np.float64)
    engineering = [float(value) for value in config.ENGINEERING_TAIL_THRESHOLDS]
    fallback = [float(value) for value in config.FALLBACK_TAIL_THRESHOLDS]
    anchor = engineering[1] if len(engineering) >= 2 else engineering[0]
    anchor_count = int(np.sum(train_labels >= anchor))
    use_fallback = bool(config.ADAPTIVE_TAIL_THRESHOLDS and anchor_count < int(config.TAIL_MIN_POSITIVE_COUNT))
    thresholds = sorted(fallback if use_fallback else engineering)
    if len(thresholds) < 3:
        raise ValueError("At least 3 tail thresholds are required")

    config.TAIL_CLASSIFICATION_THRESHOLDS = thresholds
    config.TAIL_CLASSIFICATION_LOSS_WEIGHTS = [0.010, 0.024, 0.044] if use_fallback else [0.014, 0.030, 0.058]
    config.TAIL_UNDERPREDICTION_THRESHOLD = thresholds[1]
    config.EXTREME_TAIL_UNDERPREDICTION_THRESHOLD = thresholds[2]
    config.VAL_MID_TAIL_LOW = thresholds[0]
    config.VAL_MID_TAIL_HIGH = thresholds[1]
    config.VAL_TAIL_THRESHOLD = thresholds[1]
    config.VAL_EXTREME_TAIL_THRESHOLD = thresholds[2]
    config.TAIL_PROB_GATE_INDEX = min(1, len(thresholds) - 1)

    train_counts = [int(np.sum(train_labels >= threshold)) for threshold in thresholds]
    val_counts = [int(np.sum(val_labels >= threshold)) for threshold in thresholds] if val_labels.size else []
    policy = {
        "enabled": bool(config.ADAPTIVE_TAIL_THRESHOLDS),
        "mode": "fallback_mid_tail" if use_fallback else "engineering_tail",
        "reason": f"train count >= {anchor:.4f} is {anchor_count}, min required {config.TAIL_MIN_POSITIVE_COUNT}",
        "engineering_thresholds": engineering,
        "fallback_thresholds": fallback,
        "active_thresholds": thresholds,
        "train_counts": train_counts,
        "val_counts": val_counts,
        "classification_loss_weights": list(config.TAIL_CLASSIFICATION_LOSS_WEIGHTS),
    }
    print(f">>> Adaptive tail policy: {policy['mode']} | thresholds={thresholds} | train_counts={train_counts}")
    return policy


def build_train_sampler(config, train_df: pd.DataFrame):
    if not config.USE_WEIGHTED_SAMPLER:
        return None, {"enabled": False}
    floor_series = train_df["num_floors"].round().astype(int).astype(str)
    labels = train_df[config.LABEL_COL].astype(float)
    target_bins = _build_target_bin_series(config, labels)
    balance_keys = floor_series + "_bin" + target_bins.astype(str)
    counts = balance_keys.value_counts()
    median_count = float(counts.median())
    weights = balance_keys.map(
        lambda key: (median_count / counts[key]) ** config.SAMPLER_POWER
    ).to_numpy(dtype=np.float64, copy=True)
    label_values = labels.to_numpy(dtype=np.float64)
    for threshold, boost in zip(config.TAIL_CLASSIFICATION_THRESHOLDS, config.SAMPLER_TAIL_BOOSTS):
        weights *= _smooth_tail_boost_np(config, label_values, float(threshold), float(boost))
    weights = np.clip(weights, config.SAMPLER_MIN_WEIGHT, config.SAMPLER_MAX_WEIGHT)
    num_samples = max(1, int(round(len(weights) * config.SAMPLER_NUM_SAMPLES_MULTIPLIER)))
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=True)
    return sampler, {
        "enabled": True,
        "target_bin_count": int(target_bins.nunique()),
        "joint_group_count": int(balance_keys.nunique()),
        "num_samples": int(num_samples),
        "active_tail_thresholds": [float(value) for value in config.TAIL_CLASSIFICATION_THRESHOLDS],
        "active_tail_counts": [int(np.sum(label_values >= threshold)) for threshold in config.TAIL_CLASSIFICATION_THRESHOLDS],
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        "weight_mean": float(weights.mean()),
    }


def build_label_density_weights(config, labels_raw: np.ndarray) -> np.ndarray:
    labels_raw = np.asarray(labels_raw, dtype=np.float64)
    if labels_raw.size == 0 or not config.USE_DENSITY_WEIGHTED_LOSS:
        return np.ones(labels_raw.shape, dtype=np.float32)
    label_space = np.log1p(np.maximum(labels_raw, 0.0) * config.LABEL_SCALE)
    unique_count = int(np.unique(label_space).size)
    bin_count = max(2, min(config.LOSS_DENSITY_BIN_COUNT, unique_count))
    counts, bin_edges = np.histogram(label_space, bins=bin_count)
    counts = counts.astype(np.float64)
    kernel = _gaussian_kernel(config.LOSS_DENSITY_SMOOTH_KERNEL_SIZE, config.LOSS_DENSITY_SMOOTH_SIGMA)
    pad_width = int(len(kernel) // 2)
    smoothed = np.convolve(np.pad(counts, pad_width=pad_width, mode="edge"), kernel, mode="valid")
    smoothed = np.maximum(smoothed, 1.0)
    bin_indices = np.searchsorted(bin_edges[1:-1], label_space, side="right")
    weights = (float(np.median(smoothed)) / smoothed[bin_indices]) ** config.LOSS_DENSITY_ALPHA
    weights = np.clip(weights, config.LOSS_WEIGHT_MIN, config.LOSS_WEIGHT_MAX)
    weights = weights / max(float(np.mean(weights)), 1e-8)
    weights = np.clip(weights, config.LOSS_WEIGHT_MIN, config.LOSS_WEIGHT_MAX)
    return weights.astype(np.float32)


def build_tail_classification_summary(config, labels_raw: np.ndarray) -> dict:
    labels_raw = np.asarray(labels_raw, dtype=np.float64)
    total = int(labels_raw.size)
    positive_counts = []
    pos_weights = []
    for threshold in config.TAIL_CLASSIFICATION_THRESHOLDS:
        positive = int(np.sum(labels_raw >= threshold))
        negative = max(0, total - positive)
        positive_counts.append(positive)
        if positive <= 0:
            pos_weight = 1.0
        else:
            pos_weight = math.sqrt(float(negative) / float(positive))
        pos_weights.append(float(np.clip(pos_weight, 1.0, config.TAIL_CLASSIFICATION_POS_WEIGHT_MAX)))
    return {
        "enabled": bool(config.USE_TAIL_CLASSIFICATION_AUX),
        "thresholds": [float(value) for value in config.TAIL_CLASSIFICATION_THRESHOLDS],
        "loss_weights": [float(value) for value in config.TAIL_CLASSIFICATION_LOSS_WEIGHTS],
        "positive_counts": positive_counts,
        "total_count": total,
        "pos_weights": pos_weights,
        "pos_weight_max": float(config.TAIL_CLASSIFICATION_POS_WEIGHT_MAX),
        "ramp_epochs": int(config.TAIL_CLASSIFICATION_RAMP_EPOCHS),
    }


def apply_tail_loss_multipliers(config, labels_scaled: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if not config.USE_TARGET_WEIGHTED_LOSS:
        return weights
    labels_raw = labels_scaled / float(config.LABEL_SCALE) if config.SCALE_TARGET else labels_scaled
    for threshold, target_weight in zip(config.TAIL_CLASSIFICATION_THRESHOLDS, config.TAIL_LOSS_WEIGHTS):
        weights = torch.maximum(weights, _smooth_tail_boost_torch(config, labels_raw, threshold, target_weight))
    return torch.clamp(weights, min=config.LOSS_WEIGHT_MIN, max=config.LOSS_WEIGHT_MAX)


def calculate_tail_classification_loss(
    config,
    aux_outputs: dict,
    labels_scaled: torch.Tensor,
    pos_weights: torch.Tensor,
    epoch_num: int,
) -> torch.Tensor:
    if not config.USE_TAIL_CLASSIFICATION_AUX:
        return labels_scaled.new_zeros(())
    tail_logits = aux_outputs.get("tail_logits")
    if tail_logits is None:
        return labels_scaled.new_zeros(())
    labels_raw = labels_scaled / float(config.LABEL_SCALE) if config.SCALE_TARGET else labels_scaled
    thresholds = torch.tensor(config.TAIL_CLASSIFICATION_THRESHOLDS, dtype=labels_raw.dtype, device=labels_raw.device).view(1, -1)
    targets = (labels_raw >= thresholds).to(dtype=tail_logits.dtype)
    loss_weights = torch.tensor(config.TAIL_CLASSIFICATION_LOSS_WEIGHTS, dtype=tail_logits.dtype, device=tail_logits.device).view(1, -1)
    pos_weights = pos_weights.to(dtype=tail_logits.dtype, device=tail_logits.device).view(1, -1)
    bce = nn.functional.binary_cross_entropy_with_logits(tail_logits, targets, pos_weight=pos_weights, reduction="none")
    ramp = min(1.0, max(0.0, float(epoch_num) / float(max(config.TAIL_CLASSIFICATION_RAMP_EPOCHS, 1))))
    return ramp * torch.mean(bce * loss_weights)


def calculate_tail_underprediction_loss(config, preds: torch.Tensor, labels_scaled: torch.Tensor, epoch_num: int | None = None) -> torch.Tensor:
    if not config.USE_TAIL_UNDERPREDICTION_LOSS:
        return preds.new_zeros(())
    if epoch_num is not None and epoch_num < config.TAIL_UNDERPREDICTION_START_EPOCH:
        return preds.new_zeros(())
    labels_raw = labels_scaled / float(config.LABEL_SCALE) if config.SCALE_TARGET else labels_scaled
    preds_raw = preds / float(config.LABEL_SCALE) if config.SCALE_TARGET else preds
    under_error = torch.relu(labels_scaled - preds)
    signed_error = labels_scaled - preds
    relative_under = torch.relu(labels_raw - preds_raw) / torch.clamp(labels_raw.abs(), min=5e-4)
    total = preds.new_zeros(())
    if epoch_num is None:
        ramp = 1.0
    else:
        ramp = min(
            1.0,
            max(0.0, (epoch_num - config.TAIL_UNDERPREDICTION_START_EPOCH + 1) / max(config.TAIL_UNDERPREDICTION_RAMP_EPOCHS, 1)),
        )
    for threshold, base_weight, rel_weight, tau, pin_weight in (
        (
            config.TAIL_UNDERPREDICTION_THRESHOLD,
            config.TAIL_UNDERPREDICTION_WEIGHT,
            config.TAIL_RELATIVE_UNDER_WEIGHT,
            config.TAIL_PINBALL_TAU,
            config.TAIL_PINBALL_WEIGHT,
        ),
        (
            config.EXTREME_TAIL_UNDERPREDICTION_THRESHOLD,
            config.EXTREME_TAIL_UNDERPREDICTION_WEIGHT,
            config.EXTREME_TAIL_RELATIVE_UNDER_WEIGHT,
            config.EXTREME_TAIL_PINBALL_TAU,
            config.EXTREME_TAIL_PINBALL_WEIGHT,
        ),
    ):
        mask = labels_raw >= threshold
        if not torch.any(mask):
            continue
        errors = under_error[mask]
        total = total + ramp * base_weight * nn.functional.smooth_l1_loss(
            errors,
            torch.zeros_like(errors),
            beta=config.SMOOTH_L1_BETA,
            reduction="mean",
        )
        if config.USE_TAIL_RELATIVE_UNDER_LOSS:
            total = total + ramp * rel_weight * torch.mean(relative_under[mask])
        if config.USE_TAIL_PINBALL_LOSS:
            signed = signed_error[mask]
            pinball = torch.maximum(tau * signed, (tau - 1.0) * signed)
            total = total + ramp * pin_weight * torch.mean(pinball)
    return torch.clamp(total, max=config.TAIL_UNDERPREDICTION_MAX_LOSS)


def calculate_relative_errors(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-9) * 100.0


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    ss_res = float(np.sum(errors**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 0.0 if ss_tot <= 0.0 else float(1.0 - ss_res / ss_tot)
    mape = float(np.mean(calculate_relative_errors(y_true, y_pred)))
    bias = float(np.mean(errors))
    return {"Samples": int(len(y_true)), "R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape, "Bias": bias}


def calculate_validation_focus_metrics(config, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    errors = y_pred - y_true
    under = np.maximum(y_true - y_pred, 0.0)
    mid_mask = (y_true >= config.VAL_MID_TAIL_LOW) & (y_true < config.VAL_MID_TAIL_HIGH)
    tail_mask = y_true >= config.VAL_TAIL_THRESHOLD
    extreme_mask = y_true >= config.VAL_EXTREME_TAIL_THRESHOLD

    def masked_stats(mask, fallback_mask=None):
        active = mask if np.any(mask) else (fallback_mask if fallback_mask is not None and np.any(fallback_mask) else np.ones_like(mask, dtype=bool))
        return (
            float(np.mean(np.abs(errors[active]))),
            float(np.mean(errors[active])),
            float(np.mean(under[active])),
            int(np.sum(mask)),
        )

    mid_mae, mid_bias, mid_under, mid_count = masked_stats(mid_mask)
    tail_mae, tail_bias, tail_under, tail_count = masked_stats(tail_mask)
    ext_mae, ext_bias, ext_under, ext_count = masked_stats(extreme_mask, tail_mask)
    metrics = calculate_regression_metrics(y_true, y_pred)
    focus = (
        metrics["MAE"]
        + config.VAL_FOCUS_RMSE_WEIGHT * metrics["RMSE"]
        + config.VAL_FOCUS_TAIL_MAE_WEIGHT * tail_mae
        + config.VAL_FOCUS_EXTREME_TAIL_MAE_WEIGHT * ext_mae
        + config.VAL_FOCUS_TAIL_UNDER_WEIGHT * tail_under
        + config.VAL_FOCUS_EXTREME_TAIL_UNDER_WEIGHT * ext_under
    )
    return {
        "focus_score": float(focus),
        "mid_tail_mae": mid_mae,
        "mid_tail_bias": mid_bias,
        "mid_tail_under_mae": mid_under,
        "mid_tail_count": mid_count,
        "tail_mae": tail_mae,
        "tail_bias": tail_bias,
        "tail_under_mae": tail_under,
        "tail_count": tail_count,
        "extreme_tail_mae": ext_mae,
        "extreme_tail_bias": ext_bias,
        "extreme_tail_under_mae": ext_under,
        "extreme_tail_count": ext_count,
    }


def calculate_selection_score(config, metrics: dict[str, float], focus: dict[str, float]) -> float:
    return float(
        metrics["MAE"]
        + config.SELECTION_FOCUS_WEIGHT * focus["focus_score"]
        + config.SELECTION_MID_TAIL_WEIGHT * focus["mid_tail_mae"]
    )


def build_seed_metrics_dataframe(y_true, y_pred, seeds, sample_ratio=0.8, sample_with_replacement=False) -> pd.DataFrame:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sample_size = max(2, min(int(round(len(y_true) * sample_ratio)), len(y_true)))
    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y_true), size=sample_size, replace=sample_with_replacement)
        metrics = calculate_regression_metrics(y_true[idx], y_pred[idx])
        rows.append({"Seed": seed, **metrics})
    return pd.DataFrame(rows)


def build_seed_metrics_report(seed_df: pd.DataFrame) -> pd.DataFrame:
    report = seed_df.copy()
    numeric_cols = ["R2", "MAE", "RMSE", "MAPE", "Bias"]
    mean_row = {"Seed": "mean", "Samples": seed_df["Samples"].mean()}
    std_row = {"Seed": "std", "Samples": seed_df["Samples"].std(ddof=0)}
    for col in numeric_cols:
        mean_row[col] = seed_df[col].mean()
        std_row[col] = seed_df[col].std(ddof=0)
    return pd.concat([report, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def print_seed_summary(seed_df: pd.DataFrame) -> None:
    summary = build_seed_metrics_report(seed_df).tail(2).reset_index(drop=True)
    mean_row = summary.iloc[0]
    std_row = summary.iloc[1]
    print("-" * 72)
    print(
        f"Resampling mean/std | R2={mean_row['R2']:.4f} +/- {std_row['R2']:.4f} | "
        f"MAE={mean_row['MAE']:.6f} +/- {std_row['MAE']:.6f} | "
        f"RMSE={mean_row['RMSE']:.6f} +/- {std_row['RMSE']:.6f} | "
        f"MAPE={mean_row['MAPE']:.2f}% +/- {std_row['MAPE']:.2f}%"
    )
    print("-" * 72)


def build_tail_threshold_metrics(y_true: np.ndarray, y_pred: np.ndarray, thresholds: list[float]) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        true_pos_mask = y_true >= threshold
        pred_pos_mask = y_pred >= threshold
        tp = int(np.sum(true_pos_mask & pred_pos_mask))
        fp = int(np.sum(~true_pos_mask & pred_pos_mask))
        fn = int(np.sum(true_pos_mask & ~pred_pos_mask))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        rows.append(
            {
                "threshold": float(threshold),
                "true_count": int(np.sum(true_pos_mask)),
                "pred_count": int(np.sum(pred_pos_mask)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return pd.DataFrame(rows)


def build_drift_bin_report(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    bins = [
        ("<0.001", y_true < 0.001),
        ("0.001-0.002", (y_true >= 0.001) & (y_true < 0.002)),
        ("0.002-0.003", (y_true >= 0.002) & (y_true < 0.003)),
        ("0.003-0.005", (y_true >= 0.003) & (y_true < 0.005)),
        ("0.005-0.010", (y_true >= 0.005) & (y_true < 0.010)),
        (">=0.010", y_true >= 0.010),
    ]
    rows = []
    errors = y_pred - y_true
    for name, mask in bins:
        count = int(np.sum(mask))
        if count == 0:
            rows.append({"drift_bin": name, "count": 0, "mae": np.nan, "bias": np.nan, "under_rate": np.nan})
            continue
        rows.append(
            {
                "drift_bin": name,
                "count": count,
                "mae": float(np.mean(np.abs(errors[mask]))),
                "bias": float(np.mean(errors[mask])),
                "under_rate": float(np.mean(errors[mask] < 0.0)),
            }
        )
    return pd.DataFrame(rows)


def prepare_training_data(config):
    train_csv, val_csv, dataset_base = resolve_dataset_paths(config)
    model_dir = configure_model_dir(config, dataset_base)
    print(f">>> Dataset run:       {dataset_base}")
    print(f">>> Train CSV:         {train_csv}")
    print(f">>> Val CSV:           {val_csv}")
    print(f">>> Dataset dirs:      {format_dataset_dir_text(config)}")
    print(f">>> Model artifacts:   {model_dir}")

    train_df = pd.read_csv(train_csv, low_memory=False)
    val_df = pd.read_csv(val_csv, low_memory=False)
    validate_dataframe(config, train_df, train_csv)
    validate_dataframe(config, val_df, val_csv)
    train_df = filter_valid_rows(config, train_df)
    val_df = filter_valid_rows(config, val_df)
    train_df = sample_dataframe_by_group(train_df, config.TXT_COL, config.DATA_USE_RATIO, config.SEED)

    layout_width = int(max(config.MAX_DAMPER_FLOORS, train_df["num_floors"].max(), val_df["num_floors"].max()))
    train_scalar_df, feature_names = build_scalar_feature_frame(config, train_df, layout_width)
    val_scalar_df, _ = build_scalar_feature_frame(config, val_df, layout_width, expected_feature_names=feature_names)
    config.SCALAR_COLS = list(feature_names)
    print(f">>> Scalar features ({len(feature_names)}): {feature_names}")

    train_labels = train_df[config.LABEL_COL].values.astype(np.float32)
    val_labels = val_df[config.LABEL_COL].values.astype(np.float32)
    adaptive_policy = configure_adaptive_tail_policy(config, train_labels, val_labels)
    loss_weights = build_label_density_weights(config, train_labels)
    tail_summary = build_tail_classification_summary(config, train_labels)

    scalar_scaler = StandardScaler()
    train_scalars = scalar_scaler.fit_transform(train_scalar_df.values.astype(np.float32))
    val_scalars = scalar_scaler.transform(val_scalar_df.values.astype(np.float32))
    joblib.dump(scalar_scaler, model_dir / "scalar_scaler.pkl")

    wave_dts = train_df[config.WAVE_DT_COL].values if config.WAVE_DT_COL in train_df.columns else np.full(len(train_df), np.nan)
    sequence_scaler = fit_sequence_scaler(config, train_df[config.TXT_COL].astype(str).values, wave_dts)
    joblib.dump(sequence_scaler, model_dir / "sequence_scaler.pkl")

    metadata = {
        "dataset_base": dataset_base,
        "model_run_name": model_dir.name,
        "train_csv_path": str(train_csv),
        "val_csv_path": str(val_csv),
        "dataset_search_dirs": [str(path) for path in get_existing_dataset_dirs(config)],
        "model_dir": str(model_dir),
        "model_family": config.MODEL_FAMILY,
        "architecture_revision": config.ARCHITECTURE_REVISION,
        "txt_col": config.TXT_COL,
        "sample_id_col": config.SAMPLE_ID_COL,
        "wave_dt_col": config.WAVE_DT_COL,
        "label_col": config.LABEL_COL,
        "scale_target": bool(config.SCALE_TARGET),
        "label_scale": float(config.LABEL_SCALE),
        "layout_feature_width": int(layout_width),
        "base_scalar_cols": list(config.BASE_SCALAR_COLS),
        "wave_feature_cols": get_available_wave_feature_cols(config, train_df),
        "scalar_feature_names": list(feature_names),
        "num_scalar_features": int(len(feature_names)),
        "sequence_config": {
            "seq_len": int(config.SEQ_LEN),
            "target_dt": None if config.TARGET_DT is None else float(config.TARGET_DT),
            "channels": list(config.SEQUENCE_CHANNELS),
            "num_channels": int(len(config.SEQUENCE_CHANNELS)),
            "clip_value": float(config.SEQUENCE_CLIP_VALUE),
        },
        "sequence_scaler": sequence_scaler,
        "best_weights_name": config.BEST_WEIGHTS_NAME,
        "last_weights_name": config.FINAL_WEIGHTS_NAME,
        "best_weights_are_ema": bool(config.USE_EMA),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_unique_waves": int(train_df[config.TXT_COL].nunique()),
        "val_unique_waves": int(val_df[config.TXT_COL].nunique()),
        "train_loss_weight_summary": {
            "min": float(loss_weights.min()),
            "max": float(loss_weights.max()),
            "mean": float(loss_weights.mean()),
        },
        "adaptive_tail_policy": adaptive_policy,
        "tail_classification": tail_summary,
    }
    save_json(model_dir / "training_metadata.json", metadata)

    train_dataset = SequenceDataset(
        config,
        train_df,
        train_scalars,
        train_labels,
        sequence_scaler,
        sample_weights=loss_weights,
        cache_sequences=config.CACHE_SEQUENCES,
    )
    val_dataset = SequenceDataset(
        config,
        val_df,
        val_scalars,
        val_labels,
        sequence_scaler,
        sample_weights=np.ones_like(val_labels, dtype=np.float32),
        cache_sequences=config.CACHE_SEQUENCES,
    )
    sampler, sampler_summary = build_train_sampler(config, train_df)
    metadata["use_weighted_sampler"] = bool(config.USE_WEIGHTED_SAMPLER)
    metadata["sampler_summary"] = sampler_summary
    save_json(model_dir / "training_metadata.json", metadata)

    pin_memory = config.DEVICE.type == "cuda"
    persistent_workers = config.NUM_WORKERS > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=len(train_dataset) > config.BATCH_SIZE and len(train_dataset) % config.BATCH_SIZE == 1,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return train_loader, val_loader, metadata


def train_sequence_model(config, model_factory: Callable[[int, int], nn.Module]):
    set_global_seed(config.SEED)
    train_loader, val_loader, metadata = prepare_training_data(config)
    use_amp = config.USE_AMP and config.DEVICE.type == "cuda"
    model = model_factory(int(metadata["sequence_config"]["num_channels"]), int(metadata["num_scalar_features"])).to(config.DEVICE)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(model)
    print(f">>> Total parameters: {total_params:,}")
    print(f">>> Trainable parameters: {trainable_params:,}")
    metadata.update({"total_params": int(total_params), "trainable_params": int(trainable_params)})
    save_json(config.MODEL_DIR / "training_metadata.json", metadata)

    criterion = nn.SmoothL1Loss(beta=config.SMOOTH_L1_BETA, reduction="none")
    optimizer = build_adamw_optimizer(config, model)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.MIN_LR,
    )
    scaler = build_grad_scaler(use_amp)
    ema = ModelEMA(model, config.EMA_DECAY) if config.USE_EMA else None
    pos_weights = torch.tensor(
        metadata.get("tail_classification", {}).get("pos_weights", [1.0] * len(config.TAIL_CLASSIFICATION_THRESHOLDS)),
        dtype=torch.float32,
        device=config.DEVICE,
    )

    best_selection = float("inf")
    best_mae = float("inf")
    best_focus = float("inf")
    best_extreme_under = float("inf")
    best_epoch = 0
    early_stop = 0
    history = {key: [] for key in [
        "train_loss",
        "val_loss",
        "val_mae_raw",
        "val_rmse_raw",
        "val_r2",
        "val_mape",
        "val_focus_score",
        "val_mid_tail_mae_raw",
        "val_tail_mae_raw",
        "val_tail_under_mae_raw",
        "val_extreme_tail_mae_raw",
        "val_extreme_tail_under_mae_raw",
        "val_selection_score",
        "lr",
    ]}

    start_time = time.time()
    print(f">>> Start training {config.MODEL_TAG} for {config.NUM_EPOCHS} epochs")
    for epoch in range(config.NUM_EPOCHS):
        apply_lr_warmup(config, optimizer, epoch)
        model.train()
        train_loss = 0.0
        seen = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.NUM_EPOCHS} [Train]")
        for sequences, scalars, labels, weights, _sample_ids in loop:
            sequences = sequences.to(config.DEVICE, non_blocking=True)
            scalars = scalars.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True).unsqueeze(1)
            weights = weights.to(config.DEVICE, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with build_amp_autocast_context(config, use_amp):
                preds, aux = model(sequences, scalars, return_aux=True)
                loss_per_sample = criterion(preds, labels)
                loss_weights = apply_tail_loss_multipliers(config, labels, weights)
                loss = (loss_per_sample * loss_weights).sum() / loss_weights.sum().clamp_min(1.0)
                loss = loss + calculate_tail_underprediction_loss(config, preds, labels, epoch_num=epoch + 1)
                loss = loss + calculate_tail_classification_loss(config, aux, labels, pos_weights, epoch_num=epoch + 1)
            scaler.scale(loss).backward()
            if config.GRAD_CLIP_NORM is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            batch_size = int(sequences.size(0))
            train_loss += float(loss.item()) * batch_size
            seen += batch_size
            loop.set_postfix(loss=f"{float(loss.item()):.4f}")
        epoch_train_loss = train_loss / max(seen, 1)

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        model.eval()
        val_loss = 0.0
        pred_batches = []
        true_batches = []
        with torch.no_grad():
            for sequences, scalars, labels, _weights, _sample_ids in val_loader:
                sequences = sequences.to(config.DEVICE, non_blocking=True)
                scalars = scalars.to(config.DEVICE, non_blocking=True)
                labels = labels.to(config.DEVICE, non_blocking=True).unsqueeze(1)
                with build_amp_autocast_context(config, use_amp):
                    preds = model(sequences, scalars)
                    loss = criterion(preds, labels).mean()
                val_loss += float(loss.item()) * int(sequences.size(0))
                pred_batches.append(preds.detach().cpu().numpy().reshape(-1))
                true_batches.append(labels.detach().cpu().numpy().reshape(-1))
        checkpoint_state = ema.state_dict() if ema is not None else model.state_dict()
        if ema is not None:
            ema.restore(model)

        pred_scaled = np.concatenate(pred_batches).astype(np.float32)
        true_scaled = np.concatenate(true_batches).astype(np.float32)
        pred_raw = pred_scaled / config.LABEL_SCALE if config.SCALE_TARGET else pred_scaled
        true_raw = true_scaled / config.LABEL_SCALE if config.SCALE_TARGET else true_scaled
        metrics = calculate_regression_metrics(true_raw, pred_raw)
        focus = calculate_validation_focus_metrics(config, true_raw, pred_raw)
        selection = calculate_selection_score(config, metrics, focus)
        if epoch + 1 > config.WARMUP_EPOCHS:
            scheduler.step(selection)

        current_lr = float(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(float(epoch_train_loss))
        history["val_loss"].append(float(val_loss / max(len(val_loader.dataset), 1)))
        history["val_mae_raw"].append(metrics["MAE"])
        history["val_rmse_raw"].append(metrics["RMSE"])
        history["val_r2"].append(metrics["R2"])
        history["val_mape"].append(metrics["MAPE"])
        history["val_focus_score"].append(focus["focus_score"])
        history["val_mid_tail_mae_raw"].append(focus["mid_tail_mae"])
        history["val_tail_mae_raw"].append(focus["tail_mae"])
        history["val_tail_under_mae_raw"].append(focus["tail_under_mae"])
        history["val_extreme_tail_mae_raw"].append(focus["extreme_tail_mae"])
        history["val_extreme_tail_under_mae_raw"].append(focus["extreme_tail_under_mae"])
        history["val_selection_score"].append(selection)
        history["lr"].append(current_lr)

        if focus["focus_score"] < best_focus:
            best_focus = focus["focus_score"]
            torch.save(checkpoint_state, config.SAVE_DIR / config.FOCUS_WEIGHTS_NAME)
        if metrics["MAE"] < best_mae:
            best_mae = metrics["MAE"]
            torch.save(checkpoint_state, config.SAVE_DIR / config.MAE_WEIGHTS_NAME)
        if focus["extreme_tail_under_mae"] < best_extreme_under:
            best_extreme_under = focus["extreme_tail_under_mae"]
            torch.save(checkpoint_state, config.SAVE_DIR / config.EXTREME_UNDER_WEIGHTS_NAME)

        improved = selection < best_selection or (np.isclose(selection, best_selection) and metrics["MAE"] <= best_mae)
        if improved:
            best_selection = selection
            best_epoch = epoch + 1
            early_stop = 0
            torch.save(checkpoint_state, config.SAVE_DIR / config.BEST_WEIGHTS_NAME)
            tqdm.write(
                f"  >> Best saved | selection={selection:.6f} | MAE={metrics['MAE']:.6f} | "
                f"tail={focus['tail_mae']:.6f} | extreme_under={focus['extreme_tail_under_mae']:.6f}"
            )
        else:
            early_stop += 1

        tqdm.write(
            f"Ep {epoch + 1}: train={epoch_train_loss:.4f} | val={history['val_loss'][-1]:.4f} | "
            f"MAE={metrics['MAE']:.6f} | RMSE={metrics['RMSE']:.6f} | R2={metrics['R2']:.4f} | "
            f"MAPE={metrics['MAPE']:.2f}% | focus={focus['focus_score']:.6f} | "
            f"selection={selection:.6f} | lr={current_lr:.2e} | early={early_stop}/{config.EARLY_STOPPING_PATIENCE}"
        )
        if early_stop >= config.EARLY_STOPPING_PATIENCE:
            tqdm.write(f">>> Early stopping at epoch {epoch + 1}")
            break

    torch.save(model.state_dict(), config.SAVE_DIR / config.FINAL_WEIGHTS_NAME)
    if ema is not None:
        torch.save(ema.state_dict(), config.SAVE_DIR / config.EMA_WEIGHTS_NAME)
    total_minutes = (time.time() - start_time) / 60.0
    metadata.update(
        {
            "best_epoch": int(best_epoch),
            "best_val_selection_score": float(best_selection),
            "best_val_mae_raw": float(best_mae),
            "best_val_focus_score": float(best_focus),
            "best_extreme_tail_under_mae_raw": float(best_extreme_under),
            "total_train_minutes": float(total_minutes),
            "history_keys": list(history.keys()),
        }
    )
    save_json(config.SAVE_DIR / "training_metadata.json", metadata)
    save_json(config.SAVE_DIR / "training_history.json", history)
    pd.DataFrame(history).to_csv(config.SAVE_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_training_curves(config, history)
    print(f">>> Training complete in {total_minutes:.1f} min. Best epoch: {best_epoch}")
    return history


def plot_training_curves(config, history: dict) -> None:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(history["val_mae_raw"], label="MAE")
    plt.plot(history["val_selection_score"], label="Selection")
    plt.plot(history["val_tail_mae_raw"], label="Tail MAE")
    plt.title("Validation")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.semilogy(history["train_loss"], label="Train")
    plt.semilogy(history["val_loss"], label="Val")
    plt.title("Loss Log")
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path = config.SAVE_DIR / "training_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f">>> Training curves saved to: {path}")


def resolve_test_csv_path(config, metadata: dict) -> Path:
    if config.TEST_CSV_PATH:
        return resolve_csv_path(config, config.TEST_CSV_PATH, config.TEST_FILE_PATTERN)
    train_path_text = metadata.get("train_csv_path")
    if train_path_text:
        train_path = Path(train_path_text)
        base = dataset_base_name_from_csv(train_path)
        paired = train_path.with_name(f"{base}_test.csv")
        if paired.exists():
            return paired
    return resolve_csv_path(config, None, config.TEST_FILE_PATTERN)


def build_result_paths(config, test_csv: Path) -> tuple[Path, Path, Path, Path, Path]:
    base = dataset_base_name_from_csv(test_csv)
    artifact_name = build_compact_artifact_name(base, f"{config.MODEL_TAG}_test")
    result_base = config.MODEL_DIR / artifact_name
    return (
        result_base.with_suffix(".csv"),
        result_base.with_name(result_base.name + "_seed_metrics.csv"),
        result_base.with_name(result_base.name + "_tail_metrics.csv"),
        result_base.with_name(result_base.name + "_drift_bins.csv"),
        result_base.with_suffix(".svg"),
    )


def load_training_metadata(config) -> dict:
    model_dir = resolve_model_dir(config)
    config.MODEL_DIR = model_dir
    config.SAVE_DIR = model_dir
    metadata_path = model_dir / "training_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    config.SCALAR_COLS = list(metadata.get("scalar_feature_names", []))
    config.SEQ_LEN = int(metadata.get("sequence_config", {}).get("seq_len", config.SEQ_LEN))
    config.SEQUENCE_CHANNELS = list(metadata.get("sequence_config", {}).get("channels", config.SEQUENCE_CHANNELS))
    config.TARGET_DT = metadata.get("sequence_config", {}).get("target_dt", config.TARGET_DT)
    tail_policy = metadata.get("adaptive_tail_policy", {})
    active_thresholds = tail_policy.get("active_thresholds")
    if active_thresholds:
        config.TAIL_CLASSIFICATION_THRESHOLDS = [float(value) for value in active_thresholds]
        config.TAIL_UNDERPREDICTION_THRESHOLD = config.TAIL_CLASSIFICATION_THRESHOLDS[1]
        config.EXTREME_TAIL_UNDERPREDICTION_THRESHOLD = config.TAIL_CLASSIFICATION_THRESHOLDS[2]
        config.VAL_MID_TAIL_LOW = config.TAIL_CLASSIFICATION_THRESHOLDS[0]
        config.VAL_MID_TAIL_HIGH = config.TAIL_CLASSIFICATION_THRESHOLDS[1]
        config.VAL_TAIL_THRESHOLD = config.TAIL_CLASSIFICATION_THRESHOLDS[1]
        config.VAL_EXTREME_TAIL_THRESHOLD = config.TAIL_CLASSIFICATION_THRESHOLDS[2]
    override_name = os.environ.get(f"SURMOD_{config.ENV_PREFIX}_WEIGHTS_NAME")
    weights_name = override_name or metadata.get("best_weights_name") or config.BEST_WEIGHTS_NAME
    config.MODEL_WEIGHTS_PATH = model_dir / weights_name
    print(f">>> Model dir: {model_dir}")
    print(f">>> Weights:   {config.MODEL_WEIGHTS_PATH}")
    return metadata


def load_test_data(config, metadata: dict):
    test_csv = resolve_test_csv_path(config, metadata)
    result_paths = build_result_paths(config, test_csv)
    print(f">>> Test CSV:  {test_csv}")
    df = pd.read_csv(test_csv, low_memory=False)
    validate_dataframe(config, df, test_csv)
    df = filter_valid_rows(config, df)
    labels = df[config.LABEL_COL].values.astype(np.float32)
    expected = metadata.get("scalar_feature_names") or config.SCALAR_COLS
    layout_width = int(max(
        metadata.get("layout_feature_width", config.MAX_DAMPER_FLOORS),
        config.MAX_DAMPER_FLOORS,
        df["num_floors"].max(),
    ))
    scalar_df, feature_names = build_scalar_feature_frame(config, df, layout_width, expected_feature_names=expected)
    config.SCALAR_COLS = list(feature_names)
    scaler = joblib.load(config.MODEL_DIR / "scalar_scaler.pkl")
    scalars = scaler.transform(scalar_df.values.astype(np.float32))
    sequence_scaler = joblib.load(config.MODEL_DIR / "sequence_scaler.pkl")
    dataset = SequenceDataset(
        config,
        df,
        scalars,
        labels,
        sequence_scaler,
        sample_weights=np.ones_like(labels, dtype=np.float32),
        cache_sequences=config.CACHE_SEQUENCES,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.DEVICE.type == "cuda",
        persistent_workers=config.NUM_WORKERS > 0,
    )
    return {
        "loader": loader,
        "df": df,
        "scalar_df": scalar_df,
        "labels": labels,
        "test_csv": test_csv,
        "result_paths": result_paths,
    }


def evaluate_sequence_model(config, model_factory: Callable[[int, int], nn.Module]) -> None:
    metadata = load_training_metadata(config)
    test_data = load_test_data(config, metadata)
    model = model_factory(len(config.SEQUENCE_CHANNELS), len(config.SCALAR_COLS)).to(config.DEVICE)
    state_dict = torch.load(config.MODEL_WEIGHTS_PATH, map_location=config.DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    true_values = []
    pred_values = []
    sample_ids = []
    with torch.no_grad():
        for sequences, scalars, labels, _weights, batch_sample_ids in tqdm(test_data["loader"], desc="Evaluating"):
            sequences = sequences.to(config.DEVICE, non_blocking=True)
            scalars = scalars.to(config.DEVICE, non_blocking=True)
            preds = model(sequences, scalars)
            pred = preds.detach().cpu().numpy().reshape(-1)
            if config.SCALE_TARGET:
                pred = pred / float(config.LABEL_SCALE)
                true = labels.numpy().reshape(-1) / float(config.LABEL_SCALE)
            else:
                true = labels.numpy().reshape(-1)
            pred_values.extend(pred)
            true_values.extend(true)
            sample_ids.extend([str(value) for value in batch_sample_ids])

    y_true = np.asarray(true_values, dtype=np.float32)
    y_pred = np.asarray(pred_values, dtype=np.float32)
    metrics = calculate_regression_metrics(y_true, y_pred)
    print("=" * 72)
    print(f"{config.MODEL_TAG} evaluation")
    print(f"Samples: {metrics['Samples']}")
    print(f"R2     : {metrics['R2']:.6f}")
    print(f"MAE    : {metrics['MAE']:.9f}")
    print(f"RMSE   : {metrics['RMSE']:.9f}")
    print(f"MAPE   : {metrics['MAPE']:.2f}%")
    print(f"Bias   : {metrics['Bias']:.9f}")
    print("=" * 72)

    seed_df = build_seed_metrics_dataframe(
        y_true,
        y_pred,
        config.TEST_SEEDS,
        sample_ratio=config.TEST_SAMPLE_RATIO,
        sample_with_replacement=config.TEST_SAMPLE_WITH_REPLACEMENT,
    )
    print_seed_summary(seed_df)
    threshold_metrics = build_tail_threshold_metrics(y_true, y_pred, [0.003, 0.005, 0.007, 0.010, 0.020])
    drift_report = build_drift_bin_report(y_true, y_pred)

    result_path, seed_path, tail_path, bin_path, fig_path = test_data["result_paths"]
    df = test_data["df"].copy()
    result_df = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "True_Drift": y_true,
            "Pred_Drift": y_pred,
            "Abs_Error": np.abs(y_true - y_pred),
            "Error_Pct": calculate_relative_errors(y_true, y_pred),
        }
    )
    for col in (config.TXT_COL, "split", "stage", "num_floors", "wave_cluster", "steel01_yielded", "steel02_yielded"):
        if col in df.columns:
            result_df[col] = df[col].values
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")
    build_seed_metrics_report(seed_df).to_csv(seed_path, index=False, encoding="utf-8-sig")
    threshold_metrics.to_csv(tail_path, index=False, encoding="utf-8-sig")
    drift_report.to_csv(bin_path, index=False, encoding="utf-8-sig")
    plot_evaluation_results(y_true, y_pred, metrics, fig_path)
    print(f">>> Predictions saved to: {result_path}")
    print(f">>> Seed metrics saved to: {seed_path}")
    print(f">>> Tail metrics saved to: {tail_path}")
    print(f">>> Drift-bin report saved to: {bin_path}")
    print(f">>> Plot saved to: {fig_path}")


def plot_evaluation_results(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, fig_path: Path) -> None:
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred, alpha=0.35, s=6)
    min_val = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_val = max(float(np.max(y_true)), float(np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2)
    plt.title(f"True vs Predicted\nR2={metrics['R2']:.4f}")
    plt.xlabel("True Drift")
    plt.ylabel("Predicted Drift")
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 2)
    errors = y_pred - y_true
    plt.hist(errors, bins=50, alpha=0.75, edgecolor="black")
    plt.axvline(0.0, color="r", linestyle="--", linewidth=2)
    plt.title(f"Error\nMAE={metrics['MAE']:.6f}")
    plt.xlabel("Pred - True")
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 3)
    rel_errors = np.clip(calculate_relative_errors(y_true, y_pred), 0.0, 100.0)
    plt.hist(rel_errors, bins=50, alpha=0.75, edgecolor="black")
    plt.title("Relative Error")
    plt.xlabel("Error (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, format="svg", bbox_inches="tight")
    plt.close()
