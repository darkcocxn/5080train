from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_MODULE_NAME = "surmod_v8_train_base_for_v9"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_base_module() -> ModuleType:
    return importlib.import_module(BASE_MODULE_NAME)


def configure_v9(base: ModuleType) -> None:
    cfg = base.Config

    cfg.MODEL_ROOT_DIR = PROJECT_ROOT / "output" / "2dcnnv9"
    cfg.MODEL_DIR = cfg.MODEL_ROOT_DIR
    cfg.SAVE_ROOT_DIR = cfg.MODEL_ROOT_DIR
    cfg.SAVE_DIR = cfg.MODEL_ROOT_DIR
    cfg.MODEL_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    cfg.NUM_EPOCHS = 160
    cfg.SCHEDULER_PATIENCE = 3
    cfg.EARLY_STOPPING_PATIENCE = 16
    cfg.MIN_LR = 3e-7

    cfg.LOSS_WEIGHT_MAX = 24.0
    cfg.LOSS_WEIGHT_GE_005 = 1.8
    cfg.LOSS_WEIGHT_GE_010 = 5.2
    cfg.LOSS_WEIGHT_GE_020 = 18.0
    cfg.LOSS_WEIGHT_GE_030 = 24.0

    cfg.TAIL_UNDERPREDICTION_START_EPOCH = 5
    cfg.TAIL_UNDERPREDICTION_RAMP_EPOCHS = 12
    cfg.TAIL_UNDERPREDICTION_WEIGHT = 0.18
    cfg.EXTREME_TAIL_UNDERPREDICTION_WEIGHT = 0.60
    cfg.SEVERE_TAIL_UNDERPREDICTION_THRESHOLD = 0.030
    cfg.SEVERE_TAIL_UNDERPREDICTION_WEIGHT = 0.85
    cfg.TAIL_RELATIVE_UNDER_WEIGHT = 0.04
    cfg.EXTREME_TAIL_RELATIVE_UNDER_WEIGHT = 0.10
    cfg.SEVERE_TAIL_RELATIVE_UNDER_WEIGHT = 0.20
    cfg.TAIL_UNDERPREDICTION_MAX_LOSS = 3.0
    cfg.UNDERPREDICTION_REL_EPS = 5e-4

    cfg.VAL_SEVERE_TAIL_THRESHOLD = 0.030
    cfg.VAL_FOCUS_MID_TAIL_MAE_WEIGHT = 0.12
    cfg.VAL_FOCUS_RMSE_WEIGHT = 0.22
    cfg.VAL_FOCUS_TAIL_MAE_WEIGHT = 0.38
    cfg.VAL_FOCUS_EXTREME_TAIL_MAE_WEIGHT = 0.82
    cfg.VAL_FOCUS_SEVERE_TAIL_MAE_WEIGHT = 1.10
    cfg.VAL_FOCUS_TAIL_UNDER_WEIGHT = 0.20
    cfg.VAL_FOCUS_EXTREME_TAIL_UNDER_WEIGHT = 0.50
    cfg.VAL_FOCUS_SEVERE_TAIL_UNDER_WEIGHT = 0.78
    cfg.VAL_FOCUS_TAIL_LOW_BIAS_WEIGHT = 0.18
    cfg.VAL_FOCUS_EXTREME_TAIL_LOW_BIAS_WEIGHT = 0.55
    cfg.VAL_FOCUS_SEVERE_TAIL_LOW_BIAS_WEIGHT = 0.90

    cfg.SELECTION_METRIC = "val_conservative_tail_score_v9"
    cfg.SELECTION_FOCUS_WEIGHT = 0.045
    cfg.SELECTION_MID_TAIL_WEIGHT = 0.08
    cfg.SELECTION_TAIL_WEIGHT = 0.18
    cfg.SELECTION_EXTREME_TAIL_WEIGHT = 0.40
    cfg.SELECTION_EXTREME_UNDER_WEIGHT = 0.30
    cfg.SELECTION_EXTREME_NEG_BIAS_WEIGHT = 0.28
    cfg.SELECTION_SEVERE_TAIL_WEIGHT = 0.55
    cfg.SELECTION_SEVERE_UNDER_WEIGHT = 0.40
    cfg.SELECTION_SEVERE_NEG_BIAS_WEIGHT = 0.32

    cfg.SAMPLER_MAX_WEIGHT = 24.0
    cfg.SAMPLER_TAIL_BOOST_GE_005 = 1.15
    cfg.SAMPLER_TAIL_BOOST_GE_010 = 2.4
    cfg.SAMPLER_TAIL_BOOST_GE_020 = 7.0
    cfg.SAMPLER_TAIL_BOOST_GE_030 = 12.0
    cfg.SAMPLER_NUM_SAMPLES_MULTIPLIER = 1.25
    cfg.SAMPLER_HIGH_FLOOR_BOOST = 0.35
    cfg.SAMPLER_LOW_STIFFNESS_BOOST = 0.25
    cfg.SAMPLER_LOW_STRENGTH_BOOST = 0.18
    cfg.SAMPLER_HIGH_FLOOR_TAIL_INTERACTION = 0.30

    cfg.TAIL_CLASSIFICATION_THRESHOLDS = [0.010, 0.020, 0.030]
    cfg.TAIL_CLASSIFICATION_LOSS_WEIGHTS = [0.020, 0.065, 0.180]
    cfg.TAIL_CLASSIFICATION_POS_WEIGHT_MAX = 40.0
    cfg.TAIL_CLASSIFICATION_RAMP_EPOCHS = 12

    cfg.TAIL_CORRECTION_INIT_BIAS = -3.6
    cfg.TAIL_CORRECTION_GATE_INIT_BIAS = -0.8
    cfg.TAIL_PROB_GATE_POWER = 0.90
    cfg.TAIL_EXTREME_PROB_GATE_FLOOR = 0.82

    cfg.ARCHITECTURE_REVISION_V9 = "tail_conservative_selection_v9"


def install_v9_patches(base: ModuleType) -> None:
    cfg = base.Config

    def smoothstep(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 0.0, 1.0)
        return clipped * clipped * (3.0 - 2.0 * clipped)

    def safe_numeric(df: pd.DataFrame, column: str, default: float) -> np.ndarray:
        if column not in df.columns:
            return np.full(len(df), default, dtype=np.float64)
        series = pd.to_numeric(df[column], errors="coerce").fillna(default)
        return series.to_numpy(dtype=np.float64)

    def percentile_rank(values: np.ndarray) -> np.ndarray:
        return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)

    def build_train_sampler_v9(train_df: pd.DataFrame):
        if not cfg.USE_WEIGHTED_SAMPLER:
            return None, {"enabled": False}

        floor_series = train_df["num_floors"].round().astype(int).astype(str)
        labels = train_df[cfg.LABEL_COL].astype(float)
        target_bins = base._build_target_bin_series(labels)
        balance_keys = floor_series + "_bin" + target_bins.astype(str)

        counts = balance_keys.value_counts()
        median_count = float(counts.median())
        weights = balance_keys.map(lambda key: (median_count / counts[key]) ** cfg.SAMPLER_POWER)
        weights = weights.to_numpy(dtype=np.float64).copy()

        label_values = labels.to_numpy(dtype=np.float64)
        weights *= base._smooth_tail_boost_np(label_values, 0.005, cfg.SAMPLER_TAIL_BOOST_GE_005)
        weights *= base._smooth_tail_boost_np(label_values, 0.010, cfg.SAMPLER_TAIL_BOOST_GE_010)
        weights *= base._smooth_tail_boost_np(label_values, 0.020, cfg.SAMPLER_TAIL_BOOST_GE_020)
        weights *= base._smooth_tail_boost_np(label_values, 0.030, cfg.SAMPLER_TAIL_BOOST_GE_030)

        floors = safe_numeric(train_df, "num_floors", 3.0)
        high_floor_progress = smoothstep((floors - 5.0) / 2.0)
        weights *= 1.0 + cfg.SAMPLER_HIGH_FLOOR_BOOST * high_floor_progress

        stiffness = np.maximum(safe_numeric(train_df, "k_base_1_4", 1.0), 1.0)
        low_stiffness_rank = percentile_rank(1.0 / stiffness)
        weights *= 1.0 + cfg.SAMPLER_LOW_STIFFNESS_BOOST * low_stiffness_rank

        strength = np.maximum(safe_numeric(train_df, "Fy_add", 1.0), 1.0)
        low_strength_rank = percentile_rank(1.0 / strength)
        weights *= 1.0 + cfg.SAMPLER_LOW_STRENGTH_BOOST * low_strength_rank * high_floor_progress

        tail_progress = smoothstep(label_values / 0.03)
        weights *= 1.0 + cfg.SAMPLER_HIGH_FLOOR_TAIL_INTERACTION * high_floor_progress * tail_progress

        weights = np.clip(weights, cfg.SAMPLER_MIN_WEIGHT, cfg.SAMPLER_MAX_WEIGHT)
        sampler_num_samples = max(1, int(round(len(weights) * cfg.SAMPLER_NUM_SAMPLES_MULTIPLIER)))

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
            "num_samples_multiplier": float(cfg.SAMPLER_NUM_SAMPLES_MULTIPLIER),
            "tail_count_ge_0p005": int(np.sum(label_values >= 0.005)),
            "tail_count_ge_0p010": int(np.sum(label_values >= 0.010)),
            "tail_count_ge_0p020": int(np.sum(label_values >= 0.020)),
            "tail_count_ge_0p030": int(np.sum(label_values >= 0.030)),
            "tail_boost_ge_0p005": float(cfg.SAMPLER_TAIL_BOOST_GE_005),
            "tail_boost_ge_0p010": float(cfg.SAMPLER_TAIL_BOOST_GE_010),
            "tail_boost_ge_0p020": float(cfg.SAMPLER_TAIL_BOOST_GE_020),
            "tail_boost_ge_0p030": float(cfg.SAMPLER_TAIL_BOOST_GE_030),
            "high_floor_boost": float(cfg.SAMPLER_HIGH_FLOOR_BOOST),
            "low_stiffness_boost": float(cfg.SAMPLER_LOW_STIFFNESS_BOOST),
            "low_strength_boost": float(cfg.SAMPLER_LOW_STRENGTH_BOOST),
            "high_floor_tail_interaction": float(cfg.SAMPLER_HIGH_FLOOR_TAIL_INTERACTION),
            "smooth_tail_weights": bool(cfg.USE_SMOOTH_TAIL_WEIGHTS),
            "tail_weight_transition_width": float(cfg.TAIL_WEIGHT_TRANSITION_WIDTH),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
        }
        return sampler, sampler_summary

    def apply_tail_loss_multipliers_v9(labels_scaled: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if not cfg.USE_TARGET_WEIGHTED_LOSS:
            return weights

        labels_raw = labels_scaled / float(cfg.LABEL_SCALE) if cfg.SCALE_TARGET else labels_scaled
        weights = torch.maximum(weights, base._smooth_tail_boost_torch(labels_raw, 0.005, cfg.LOSS_WEIGHT_GE_005))
        weights = torch.maximum(weights, base._smooth_tail_boost_torch(labels_raw, 0.010, cfg.LOSS_WEIGHT_GE_010))
        weights = torch.maximum(weights, base._smooth_tail_boost_torch(labels_raw, 0.020, cfg.LOSS_WEIGHT_GE_020))
        weights = torch.maximum(weights, base._smooth_tail_boost_torch(labels_raw, 0.030, cfg.LOSS_WEIGHT_GE_030))
        return torch.clamp(weights, min=cfg.LOSS_WEIGHT_MIN, max=cfg.LOSS_WEIGHT_MAX)

    def calculate_tail_underprediction_loss_v9(
        preds: torch.Tensor,
        labels_scaled: torch.Tensor,
        epoch_num: int | None = None,
    ) -> torch.Tensor:
        if not cfg.USE_TAIL_UNDERPREDICTION_LOSS:
            return preds.new_zeros(())
        if epoch_num is not None and epoch_num < cfg.TAIL_UNDERPREDICTION_START_EPOCH:
            return preds.new_zeros(())

        labels_raw = labels_scaled / float(cfg.LABEL_SCALE) if cfg.SCALE_TARGET else labels_scaled
        preds_raw = preds / float(cfg.LABEL_SCALE) if cfg.SCALE_TARGET else preds
        under_error_scaled = torch.relu(labels_scaled - preds)
        under_error_raw = torch.relu(labels_raw - preds_raw)

        if epoch_num is None:
            ramp_factor = 1.0
        else:
            ramp_epochs = max(int(cfg.TAIL_UNDERPREDICTION_RAMP_EPOCHS), 1)
            ramp_factor = min(
                1.0,
                max(0.0, (epoch_num - cfg.TAIL_UNDERPREDICTION_START_EPOCH + 1) / ramp_epochs),
            )

        def penalty_for_mask(
            mask: torch.Tensor,
            absolute_weight: float,
            relative_weight: float,
        ) -> torch.Tensor:
            if not torch.any(mask):
                return preds.new_zeros(())
            abs_loss = nn.functional.smooth_l1_loss(
                under_error_scaled[mask],
                torch.zeros_like(under_error_scaled[mask]),
                beta=cfg.SMOOTH_L1_BETA,
                reduction="mean",
            )
            denom = labels_raw[mask].clamp_min(float(cfg.UNDERPREDICTION_REL_EPS))
            rel_under = under_error_raw[mask] / denom
            rel_loss = torch.mean(rel_under.pow(2))
            return absolute_weight * abs_loss + relative_weight * rel_loss

        total_loss = preds.new_zeros(())
        total_loss = total_loss + penalty_for_mask(
            labels_raw >= cfg.TAIL_UNDERPREDICTION_THRESHOLD,
            cfg.TAIL_UNDERPREDICTION_WEIGHT,
            cfg.TAIL_RELATIVE_UNDER_WEIGHT,
        )
        total_loss = total_loss + penalty_for_mask(
            labels_raw >= cfg.EXTREME_TAIL_UNDERPREDICTION_THRESHOLD,
            cfg.EXTREME_TAIL_UNDERPREDICTION_WEIGHT,
            cfg.EXTREME_TAIL_RELATIVE_UNDER_WEIGHT,
        )
        total_loss = total_loss + penalty_for_mask(
            labels_raw >= cfg.SEVERE_TAIL_UNDERPREDICTION_THRESHOLD,
            cfg.SEVERE_TAIL_UNDERPREDICTION_WEIGHT,
            cfg.SEVERE_TAIL_RELATIVE_UNDER_WEIGHT,
        )
        return torch.clamp(ramp_factor * total_loss, max=cfg.TAIL_UNDERPREDICTION_MAX_LOSS)

    def calculate_validation_focus_metrics_v9(
        y_true_raw: np.ndarray,
        y_pred_raw: np.ndarray,
    ) -> dict[str, float]:
        errors = y_pred_raw - y_true_raw
        under_errors = np.maximum(y_true_raw - y_pred_raw, 0.0)
        mid_tail_mask = (y_true_raw >= cfg.VAL_MID_TAIL_LOW) & (y_true_raw < cfg.VAL_MID_TAIL_HIGH)
        tail_mask = y_true_raw >= cfg.VAL_TAIL_THRESHOLD
        extreme_tail_mask = y_true_raw >= cfg.VAL_EXTREME_TAIL_THRESHOLD
        severe_tail_mask = y_true_raw >= cfg.VAL_SEVERE_TAIL_THRESHOLD

        def masked_metrics(mask: np.ndarray) -> tuple[float, float, float]:
            if np.any(mask):
                segment_errors = errors[mask]
                return (
                    float(np.mean(np.abs(segment_errors))),
                    float(np.mean(segment_errors)),
                    float(np.mean(under_errors[mask])),
                )
            return (
                float(np.mean(np.abs(errors))),
                float(np.mean(errors)),
                float(np.mean(under_errors)),
            )

        mid_tail_mae, mid_tail_bias, mid_tail_under_mae = masked_metrics(mid_tail_mask)
        tail_mae, tail_bias, tail_under_mae = masked_metrics(tail_mask)
        extreme_tail_mae, extreme_tail_bias, extreme_tail_under_mae = masked_metrics(extreme_tail_mask)
        severe_tail_mae, severe_tail_bias, severe_tail_under_mae = masked_metrics(severe_tail_mask)

        tail_low_bias_penalty = max(-tail_bias, 0.0)
        extreme_tail_low_bias_penalty = max(-extreme_tail_bias, 0.0)
        severe_tail_low_bias_penalty = max(-severe_tail_bias, 0.0)

        metrics = base.calculate_regression_metrics(y_true_raw, y_pred_raw)
        focus_score = (
            metrics["mae"]
            + cfg.VAL_FOCUS_MID_TAIL_MAE_WEIGHT * mid_tail_mae
            + cfg.VAL_FOCUS_RMSE_WEIGHT * metrics["rmse"]
            + cfg.VAL_FOCUS_TAIL_MAE_WEIGHT * tail_mae
            + cfg.VAL_FOCUS_EXTREME_TAIL_MAE_WEIGHT * extreme_tail_mae
            + cfg.VAL_FOCUS_SEVERE_TAIL_MAE_WEIGHT * severe_tail_mae
            + cfg.VAL_FOCUS_TAIL_UNDER_WEIGHT * tail_under_mae
            + cfg.VAL_FOCUS_EXTREME_TAIL_UNDER_WEIGHT * extreme_tail_under_mae
            + cfg.VAL_FOCUS_SEVERE_TAIL_UNDER_WEIGHT * severe_tail_under_mae
            + cfg.VAL_FOCUS_TAIL_LOW_BIAS_WEIGHT * tail_low_bias_penalty
            + cfg.VAL_FOCUS_EXTREME_TAIL_LOW_BIAS_WEIGHT * extreme_tail_low_bias_penalty
            + cfg.VAL_FOCUS_SEVERE_TAIL_LOW_BIAS_WEIGHT * severe_tail_low_bias_penalty
        )
        return {
            "focus_score": float(focus_score),
            "mid_tail_mae": float(mid_tail_mae),
            "mid_tail_bias": float(mid_tail_bias),
            "mid_tail_under_mae": float(mid_tail_under_mae),
            "mid_tail_count": int(np.sum(mid_tail_mask)),
            "tail_mae": float(tail_mae),
            "tail_bias": float(tail_bias),
            "tail_under_mae": float(tail_under_mae),
            "tail_count": int(np.sum(tail_mask)),
            "extreme_tail_mae": float(extreme_tail_mae),
            "extreme_tail_bias": float(extreme_tail_bias),
            "extreme_tail_under_mae": float(extreme_tail_under_mae),
            "extreme_tail_count": int(np.sum(extreme_tail_mask)),
            "severe_tail_mae": float(severe_tail_mae),
            "severe_tail_bias": float(severe_tail_bias),
            "severe_tail_under_mae": float(severe_tail_under_mae),
            "severe_tail_count": int(np.sum(severe_tail_mask)),
            "tail_low_bias_penalty": float(tail_low_bias_penalty),
            "extreme_tail_low_bias_penalty": float(extreme_tail_low_bias_penalty),
            "severe_tail_low_bias_penalty": float(severe_tail_low_bias_penalty),
        }

    def calculate_selection_score_v9(metrics: dict[str, float], focus_metrics: dict[str, float]) -> float:
        return float(
            metrics["mae"]
            + cfg.SELECTION_FOCUS_WEIGHT * focus_metrics["focus_score"]
            + cfg.SELECTION_MID_TAIL_WEIGHT * focus_metrics["mid_tail_mae"]
            + cfg.SELECTION_TAIL_WEIGHT * focus_metrics["tail_mae"]
            + cfg.SELECTION_EXTREME_TAIL_WEIGHT * focus_metrics["extreme_tail_mae"]
            + cfg.SELECTION_EXTREME_UNDER_WEIGHT * focus_metrics["extreme_tail_under_mae"]
            + cfg.SELECTION_EXTREME_NEG_BIAS_WEIGHT * focus_metrics["extreme_tail_low_bias_penalty"]
            + cfg.SELECTION_SEVERE_TAIL_WEIGHT * focus_metrics["severe_tail_mae"]
            + cfg.SELECTION_SEVERE_UNDER_WEIGHT * focus_metrics["severe_tail_under_mae"]
            + cfg.SELECTION_SEVERE_NEG_BIAS_WEIGHT * focus_metrics["severe_tail_low_bias_penalty"]
        )

    base.build_train_sampler = build_train_sampler_v9
    base.apply_tail_loss_multipliers = apply_tail_loss_multipliers_v9
    base.calculate_tail_underprediction_loss = calculate_tail_underprediction_loss_v9
    base.calculate_validation_focus_metrics = calculate_validation_focus_metrics_v9
    base.calculate_selection_score = calculate_selection_score_v9


def candidate_checkpoint_score(history: dict[str, list[Any]], index: int) -> float:
    tail_bias = float(history["val_tail_bias_raw"][index])
    extreme_bias = float(history["val_extreme_tail_bias_raw"][index])
    return float(
        float(history["val_mae_raw"][index])
        + 0.18 * float(history["val_mid_tail_mae_raw"][index])
        + 0.55 * float(history["val_tail_mae_raw"][index])
        + 0.95 * float(history["val_extreme_tail_mae_raw"][index])
        + 0.45 * float(history["val_tail_under_mae_raw"][index])
        + 0.85 * float(history["val_extreme_tail_under_mae_raw"][index])
        + 0.30 * max(-tail_bias, 0.0)
        + 0.70 * max(-extreme_bias, 0.0)
    )


def postprocess_training_metadata(base: ModuleType) -> None:
    model_dir = base.Config.MODEL_DIR
    metadata_path = model_dir / "training_metadata.json"
    history_path = model_dir / "training_history.json"
    if not metadata_path.exists() or not history_path.exists():
        return

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    alternate = metadata.get("alternate_best_checkpoints") or {}

    candidates: list[dict[str, Any]] = []
    raw_candidates = [
        ("best_2dcnn_model.pth", metadata.get("best_epoch")),
        ("best_2dcnn_focus_model.pth", (alternate.get("focus") or {}).get("epoch")),
        ("best_2dcnn_mae_model.pth", (alternate.get("mae") or {}).get("epoch")),
        ("best_2dcnn_extreme_under_model.pth", (alternate.get("extreme_under") or {}).get("epoch")),
    ]

    for weights_name, epoch in raw_candidates:
        if not epoch:
            continue
        weights_path = model_dir / weights_name
        if not weights_path.exists():
            continue
        index = int(epoch) - 1
        if index < 0 or index >= len(history.get("val_mae_raw", [])):
            continue
        tail_bias = float(history["val_tail_bias_raw"][index])
        extreme_bias = float(history["val_extreme_tail_bias_raw"][index])
        candidates.append(
            {
                "weights_name": weights_name,
                "epoch": int(epoch),
                "candidate_score": candidate_checkpoint_score(history, index),
                "val_mae_raw": float(history["val_mae_raw"][index]),
                "val_tail_mae_raw": float(history["val_tail_mae_raw"][index]),
                "val_extreme_tail_mae_raw": float(history["val_extreme_tail_mae_raw"][index]),
                "val_tail_under_mae_raw": float(history["val_tail_under_mae_raw"][index]),
                "val_extreme_tail_under_mae_raw": float(history["val_extreme_tail_under_mae_raw"][index]),
                "tail_negative_bias_penalty": float(max(-tail_bias, 0.0)),
                "extreme_negative_bias_penalty": float(max(-extreme_bias, 0.0)),
            }
        )

    chosen = None
    if candidates:
        chosen = min(candidates, key=lambda item: (item["candidate_score"], item["val_mae_raw"]))
        metadata["best_weights_name"] = chosen["weights_name"]
        metadata["best_weights_metric"] = "v9_conservative_candidate_score"

    metadata["v9_overrides"] = {
        "architecture_revision": base.Config.ARCHITECTURE_REVISION_V9,
        "loss_weight_ge_030": float(base.Config.LOSS_WEIGHT_GE_030),
        "sampler_tail_boost_ge_030": float(base.Config.SAMPLER_TAIL_BOOST_GE_030),
        "tail_classification_thresholds": list(base.Config.TAIL_CLASSIFICATION_THRESHOLDS),
        "severe_tail_threshold": float(base.Config.VAL_SEVERE_TAIL_THRESHOLD),
        "tail_relative_under_weight": float(base.Config.TAIL_RELATIVE_UNDER_WEIGHT),
        "extreme_tail_relative_under_weight": float(base.Config.EXTREME_TAIL_RELATIVE_UNDER_WEIGHT),
        "severe_tail_relative_under_weight": float(base.Config.SEVERE_TAIL_RELATIVE_UNDER_WEIGHT),
    }
    metadata["v9_default_checkpoint_selection"] = {
        "metric": "v9_conservative_candidate_score",
        "chosen": chosen,
        "candidates": candidates,
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    if chosen is not None:
        print(
            ">>> V9 default evaluation weights: "
            f"{chosen['weights_name']} (epoch {chosen['epoch']}, score {chosen['candidate_score']:.6f})"
        )


def main() -> None:
    base = load_base_module()
    configure_v9(base)
    install_v9_patches(base)

    train_history = base.train()
    postprocess_training_metadata(base)
    base.plot_results(train_history)

    if base.Config.SCALE_TARGET:
        print("\nHint: the model predicts scaled targets and restores drift by dividing by 1000.")


if __name__ == "__main__":
    main()
