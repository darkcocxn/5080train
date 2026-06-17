# -*- coding: utf-8 -*-
"""Shared training and evaluation utilities for non-image tabular baselines."""

from __future__ import annotations

import inspect
import os
import random
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from sequence_model_common import (
    apply_common_environment_overrides,
    build_compact_artifact_name,
    build_drift_bin_report,
    build_scalar_feature_frame,
    build_seed_metrics_dataframe,
    build_seed_metrics_report,
    build_sequence_channels,
    build_tail_threshold_metrics,
    calculate_regression_metrics,
    calculate_relative_errors,
    configure_model_dir,
    dataset_base_name_from_csv,
    filter_valid_rows,
    fit_sequence_scaler,
    format_dataset_dir_text,
    get_available_wave_feature_cols,
    get_existing_dataset_dirs,
    load_wave_sequence,
    print_seed_summary,
    resolve_csv_path,
    resolve_dataset_paths,
    resolve_model_dir,
    sample_dataframe_by_group,
    save_json,
    validate_dataframe,
)


class BaseTabularConfig:
    PROJECT_ROOT = Path(__file__).resolve().parent
    SCRIPT_DIR = PROJECT_ROOT

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

    MODEL_TAG = "tabular_baseline"
    ENV_PREFIX = "TABULAR"
    MODEL_FAMILY = "tabular_non_image_surrogate"
    ARCHITECTURE_REVISION = "v1"
    MODEL_ROOT_DIR = PROJECT_ROOT / "output" / MODEL_TAG
    MODEL_DIR = MODEL_ROOT_DIR
    SAVE_ROOT_DIR = MODEL_ROOT_DIR
    SAVE_DIR = MODEL_ROOT_DIR
    UNIQUE_MODEL_RUN_DIR = True
    BEST_MODEL_NAME = "best_model.pkl"

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
    IMPUTER_STRATEGY = "median"
    USE_STANDARD_SCALER = False

    INCLUDE_WAVEFORM_FEATURES = True
    TARGET_DT = 0.01
    SEQ_LEN = 4096
    WAVEFORM_FEATURE_COUNT = 256
    SEQUENCE_CHANNELS = ["acc_scaled"]
    SEQUENCE_CLIP_VALUE = 8.0

    USE_SAMPLE_WEIGHT = True
    SAMPLE_WEIGHT_STEPS = [(0.005, 1.5), (0.010, 3.0), (0.020, 8.0)]
    SAMPLE_WEIGHT_MAX = 24.0

    TEST_SEEDS = [42, 2026, 123]
    TEST_SAMPLE_RATIO = 0.8
    TEST_SAMPLE_WITH_REPLACEMENT = False

    PROBLEM_STATEMENT = (
        "Predict max_drift_ratio_raw from structural parameters, earthquake-derived "
        "features or raw acceleration time history, and damper layout features."
    )
    LITERATURE_BASIS = [
        "Breiman (2001) Random Forests: robust ensemble tree baseline for nonlinear tabular regression.",
        "Chen and Guestrin (2016) XGBoost: scalable gradient boosted tree baseline.",
        "Ke et al. (2017) LightGBM: efficient histogram-based gradient boosting baseline.",
        "Prokhorenkova et al. (2018) CatBoost: ordered boosting baseline for tabular data.",
        "Recent seismic surrogate-model studies commonly compare RF/XGBoost/LightGBM/CatBoost/NN baselines against CNN or sequence models.",
    ]


def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def apply_tabular_environment_overrides(config) -> None:
    apply_common_environment_overrides(config)
    for env_name, attr_name in (
        ("SURMOD_INCLUDE_WAVEFORM", "INCLUDE_WAVEFORM_FEATURES"),
        ("SURMOD_USE_SAMPLE_WEIGHT", "USE_SAMPLE_WEIGHT"),
        ("SURMOD_UNIQUE_MODEL_RUN_DIR", "UNIQUE_MODEL_RUN_DIR"),
    ):
        if os.environ.get(env_name) is not None:
            setattr(config, attr_name, parse_bool(os.environ.get(env_name), getattr(config, attr_name)))

    for env_name, attr_name, caster in (
        ("SURMOD_TABULAR_SEQ_LEN", "SEQ_LEN", int),
        ("SURMOD_TABULAR_WAVEFORM_FEATURE_COUNT", "WAVEFORM_FEATURE_COUNT", int),
        ("SURMOD_TARGET_DT", "TARGET_DT", float),
        ("SURMOD_LABEL_SCALE", "LABEL_SCALE", float),
        ("SURMOD_SEED", "SEED", int),
        ("SURMOD_TEST_SAMPLE_RATIO", "TEST_SAMPLE_RATIO", float),
    ):
        value = os.environ.get(env_name)
        if value is not None and str(value).strip() != "":
            setattr(config, attr_name, caster(value))

    channels = os.environ.get("SURMOD_TABULAR_SEQUENCE_CHANNELS")
    if channels:
        config.SEQUENCE_CHANNELS = [item.strip() for item in channels.split(",") if item.strip()]


def read_valid_csv(config, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    validate_dataframe(config, df, csv_path)
    return filter_valid_rows(config, df)


def get_wave_dts(config, df: pd.DataFrame) -> np.ndarray:
    if getattr(config, "WAVE_DT_COL", "wave_dt") in df.columns:
        return pd.to_numeric(df[getattr(config, "WAVE_DT_COL")], errors="coerce").to_numpy()
    return np.full(len(df), np.nan)


def get_waveform_feature_names(config, sequence_scaler: dict | None = None) -> list[str]:
    channels = list((sequence_scaler or {}).get("channels", getattr(config, "SEQUENCE_CHANNELS", ["acc_scaled"])))
    count = int(getattr(config, "WAVEFORM_FEATURE_COUNT", 256))
    names: list[str] = []
    for channel in channels:
        clean = "".join(ch if ch.isalnum() else "_" for ch in str(channel)).strip("_")
        for idx in range(count):
            names.append(f"waveform_{clean}_{idx:04d}")
    return names


def build_waveform_feature_frame(config, df: pd.DataFrame, sequence_scaler: dict) -> tuple[pd.DataFrame, list[str]]:
    names = get_waveform_feature_names(config, sequence_scaler)
    if len(df) == 0:
        return pd.DataFrame(columns=names, dtype=np.float32), names

    feature_count = int(getattr(config, "WAVEFORM_FEATURE_COUNT", 256))
    values = np.empty((len(df), len(names)), dtype=np.float32)
    refs = df[config.TXT_COL].astype(str).to_numpy()
    wave_dts = get_wave_dts(config, df)
    cache: dict[tuple[str, float | None], np.ndarray] = {}

    for row_idx in tqdm(range(len(df)), desc="Waveform features"):
        ref = str(refs[row_idx])
        dt_value = wave_dts[row_idx]
        dt_key = float(dt_value) if np.isfinite(dt_value) else None
        key = (ref, dt_key)
        cached = cache.get(key)
        if cached is None:
            sequence = load_wave_sequence(config, ref, seq_len=int(config.SEQ_LEN), wave_dt=dt_value)
            channels = build_sequence_channels(config, sequence, sequence_scaler)
            if channels.shape[1] <= 1:
                selected = channels
            else:
                index = np.linspace(0, channels.shape[1] - 1, feature_count).round().astype(np.int64)
                selected = channels[:, index]
            cached = selected.reshape(-1).astype(np.float32, copy=False)
            cache[key] = cached
        values[row_idx, :] = cached

    return pd.DataFrame(values, columns=names), names


def build_tabular_feature_frame(
    config,
    df: pd.DataFrame,
    layout_width: int,
    scalar_feature_names: list[str] | None = None,
    include_waveform: bool | None = None,
    sequence_scaler: dict | None = None,
    waveform_feature_names: list[str] | None = None,
    all_feature_names: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    scalar_df, scalar_names = build_scalar_feature_frame(
        config,
        df,
        int(layout_width),
        expected_feature_names=scalar_feature_names,
    )
    frames = [scalar_df]
    waveform_names: list[str] = []
    use_waveform = bool(getattr(config, "INCLUDE_WAVEFORM_FEATURES", False) if include_waveform is None else include_waveform)
    if use_waveform:
        if sequence_scaler is None:
            raise ValueError("sequence_scaler is required when INCLUDE_WAVEFORM_FEATURES=True")
        wave_df, waveform_names = build_waveform_feature_frame(config, df, sequence_scaler)
        if waveform_feature_names:
            for name in waveform_feature_names:
                if name not in wave_df.columns:
                    wave_df[name] = 0.0
            wave_df = wave_df[list(waveform_feature_names)]
            waveform_names = list(waveform_feature_names)
        frames.append(wave_df)

    feature_df = pd.concat(frames, axis=1)
    feature_names = list(scalar_names) + list(waveform_names)
    if all_feature_names:
        for name in all_feature_names:
            if name not in feature_df.columns:
                feature_df[name] = 0.0
        feature_df = feature_df[list(all_feature_names)]
        feature_names = list(all_feature_names)
    return feature_df.astype(np.float32), list(scalar_names), list(waveform_names)


def build_simple_sample_weights(config, labels_raw: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels_raw, dtype=np.float64)
    weights = np.ones(labels.shape, dtype=np.float32)
    if not bool(getattr(config, "USE_SAMPLE_WEIGHT", True)):
        return weights
    steps = getattr(config, "SAMPLE_WEIGHT_STEPS", [(0.005, 1.5), (0.010, 3.0), (0.020, 8.0)])
    for threshold, multiplier in steps:
        weights[labels >= float(threshold)] *= float(multiplier)
    max_weight = float(getattr(config, "SAMPLE_WEIGHT_MAX", 24.0))
    weights = np.clip(weights, 1.0, max_weight)
    weights = weights / max(float(weights.mean()), 1e-8)
    return weights.astype(np.float32)


def calculate_selection_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    metrics = calculate_regression_metrics(y_true, y_pred)
    errors = y_pred - y_true
    tail_mask = y_true >= 0.010
    tail_mae = float(np.mean(np.abs(errors[tail_mask]))) if np.any(tail_mask) else metrics["MAE"]
    under_mae = float(np.mean(np.maximum(y_true[tail_mask] - y_pred[tail_mask], 0.0))) if np.any(tail_mask) else 0.0
    return float(metrics["MAE"] + 0.20 * metrics["RMSE"] + 0.30 * tail_mae + 0.20 * under_mae)


def prepare_tabular_training_data(config) -> dict:
    set_global_seed(config.SEED)
    train_csv, val_csv, dataset_base = resolve_dataset_paths(config)
    model_dir = configure_model_dir(config, dataset_base)
    print(f">>> Dataset run:       {dataset_base}")
    print(f">>> Train CSV:         {train_csv}")
    print(f">>> Val CSV:           {val_csv}")
    print(f">>> Dataset dirs:      {format_dataset_dir_text(config)}")
    print(f">>> Model artifacts:   {model_dir}")

    train_df = read_valid_csv(config, train_csv)
    val_df = read_valid_csv(config, val_csv)
    train_df = sample_dataframe_by_group(train_df, config.TXT_COL, config.DATA_USE_RATIO, config.SEED)
    layout_width = int(max(config.MAX_DAMPER_FLOORS, train_df["num_floors"].max(), val_df["num_floors"].max()))

    sequence_scaler = None
    if bool(getattr(config, "INCLUDE_WAVEFORM_FEATURES", False)):
        wave_dts = get_wave_dts(config, train_df)
        sequence_scaler = fit_sequence_scaler(config, train_df[config.TXT_COL].astype(str).to_numpy(), wave_dts)
        print(
            f">>> Waveform input enabled: seq_len={config.SEQ_LEN}, "
            f"features/channel={config.WAVEFORM_FEATURE_COUNT}, channels={config.SEQUENCE_CHANNELS}"
        )
    else:
        print(">>> Waveform input disabled; using scalar/derived CSV features only.")

    train_features, scalar_names, waveform_names = build_tabular_feature_frame(
        config,
        train_df,
        layout_width,
        include_waveform=bool(getattr(config, "INCLUDE_WAVEFORM_FEATURES", False)),
        sequence_scaler=sequence_scaler,
    )
    val_features, _, _ = build_tabular_feature_frame(
        config,
        val_df,
        layout_width,
        scalar_feature_names=scalar_names,
        include_waveform=bool(getattr(config, "INCLUDE_WAVEFORM_FEATURES", False)),
        sequence_scaler=sequence_scaler,
        waveform_feature_names=waveform_names,
        all_feature_names=list(train_features.columns),
    )
    feature_names = list(train_features.columns)
    config.SCALAR_COLS = list(scalar_names)
    print(f">>> Scalar features:    {len(scalar_names)}")
    print(f">>> Waveform features:  {len(waveform_names)}")
    print(f">>> Total features:     {len(feature_names)}")

    y_train_raw = train_df[config.LABEL_COL].to_numpy(dtype=np.float32)
    y_val_raw = val_df[config.LABEL_COL].to_numpy(dtype=np.float32)
    y_train = y_train_raw * float(config.LABEL_SCALE) if config.SCALE_TARGET else y_train_raw
    y_val = y_val_raw * float(config.LABEL_SCALE) if config.SCALE_TARGET else y_val_raw
    sample_weights = build_simple_sample_weights(config, y_train_raw)

    metadata = {
        "dataset_base": dataset_base,
        "model_run_name": model_dir.name,
        "algorithm_display_name": getattr(config, "ALGORITHM_DISPLAY_NAME", config.MODEL_TAG),
        "input_mode": getattr(config, "INPUT_MODE", "scalar_plus_downsampled_waveform"),
        "problem_statement": getattr(config, "PROBLEM_STATEMENT", BaseTabularConfig.PROBLEM_STATEMENT),
        "literature_basis": list(getattr(config, "LITERATURE_BASIS", BaseTabularConfig.LITERATURE_BASIS)),
        "train_csv_path": str(train_csv),
        "val_csv_path": str(val_csv),
        "dataset_search_dirs": [str(path) for path in get_existing_dataset_dirs(config)],
        "model_dir": str(model_dir),
        "model_tag": config.MODEL_TAG,
        "model_family": config.MODEL_FAMILY,
        "architecture_revision": config.ARCHITECTURE_REVISION,
        "sample_id_col": config.SAMPLE_ID_COL,
        "txt_col": config.TXT_COL,
        "wave_dt_col": config.WAVE_DT_COL,
        "label_col": config.LABEL_COL,
        "scale_target": bool(config.SCALE_TARGET),
        "label_scale": float(config.LABEL_SCALE),
        "layout_feature_width": int(layout_width),
        "base_scalar_cols": list(config.BASE_SCALAR_COLS),
        "wave_feature_cols": get_available_wave_feature_cols(config, train_df),
        "scalar_feature_names": list(scalar_names),
        "waveform_feature_names": list(waveform_names),
        "feature_names": list(feature_names),
        "num_features": int(len(feature_names)),
        "include_waveform_features": bool(getattr(config, "INCLUDE_WAVEFORM_FEATURES", False)),
        "sequence_scaler": sequence_scaler,
        "sequence_config": {
            "seq_len": int(getattr(config, "SEQ_LEN", 0)),
            "waveform_feature_count": int(getattr(config, "WAVEFORM_FEATURE_COUNT", 0)),
            "target_dt": None if getattr(config, "TARGET_DT", None) is None else float(config.TARGET_DT),
            "channels": list(getattr(config, "SEQUENCE_CHANNELS", [])),
            "clip_value": float(getattr(config, "SEQUENCE_CLIP_VALUE", 0.0)),
        },
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "train_unique_waves": int(train_df[config.TXT_COL].nunique()),
        "val_unique_waves": int(val_df[config.TXT_COL].nunique()),
        "sample_weight_summary": {
            "enabled": bool(getattr(config, "USE_SAMPLE_WEIGHT", True)),
            "min": float(sample_weights.min()),
            "max": float(sample_weights.max()),
            "mean": float(sample_weights.mean()),
        },
    }
    save_json(model_dir / "training_metadata.json", metadata)
    return {
        "train_df": train_df,
        "val_df": val_df,
        "train_features": train_features,
        "val_features": val_features,
        "y_train": y_train.astype(np.float32),
        "y_val": y_val.astype(np.float32),
        "y_train_raw": y_train_raw,
        "y_val_raw": y_val_raw,
        "sample_weights": sample_weights,
        "metadata": metadata,
        "model_dir": model_dir,
    }


def inverse_target(config, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if config.SCALE_TARGET:
        return values / float(config.LABEL_SCALE)
    return values


def build_result_paths(config, csv_path: Path, phase: str) -> dict[str, Path]:
    base = dataset_base_name_from_csv(csv_path)
    artifact_name = build_compact_artifact_name(base, f"{config.MODEL_TAG}_{phase}")
    result_base = config.MODEL_DIR / artifact_name
    return {
        "predictions": result_base.with_suffix(".csv"),
        "metrics": result_base.with_name(result_base.name + "_metrics.json"),
        "seed_metrics": result_base.with_name(result_base.name + "_seed_metrics.csv"),
        "tail_metrics": result_base.with_name(result_base.name + "_tail_metrics.csv"),
        "drift_bins": result_base.with_name(result_base.name + "_drift_bins.csv"),
        "plot": result_base.with_suffix(".svg"),
    }


def write_evaluation_outputs(
    config,
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    paths: dict[str, Path],
    phase: str,
) -> dict:
    metrics = calculate_regression_metrics(y_true, y_pred)
    selection_score = calculate_selection_score(y_true, y_pred)
    metrics = {**metrics, "SelectionScore": float(selection_score), "phase": phase}
    save_json(paths["metrics"], metrics)

    sample_ids = (
        df[config.SAMPLE_ID_COL].astype(str).to_numpy()
        if config.SAMPLE_ID_COL in df.columns
        else np.arange(len(df)).astype(str)
    )
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
            result_df[col] = df[col].to_numpy()
    result_df.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")

    seed_df = build_seed_metrics_dataframe(
        y_true,
        y_pred,
        config.TEST_SEEDS,
        sample_ratio=config.TEST_SAMPLE_RATIO,
        sample_with_replacement=config.TEST_SAMPLE_WITH_REPLACEMENT,
    )
    build_seed_metrics_report(seed_df).to_csv(paths["seed_metrics"], index=False, encoding="utf-8-sig")
    build_tail_threshold_metrics(y_true, y_pred, [0.003, 0.005, 0.007, 0.010, 0.015, 0.020]).to_csv(
        paths["tail_metrics"], index=False, encoding="utf-8-sig"
    )
    build_drift_bin_report(y_true, y_pred).to_csv(paths["drift_bins"], index=False, encoding="utf-8-sig")
    plot_evaluation_results(y_true, y_pred, metrics, paths["plot"])
    print_seed_summary(seed_df)
    print(f">>> {phase} predictions saved to: {paths['predictions']}")
    print(f">>> {phase} metrics saved to: {paths['metrics']}")
    return metrics


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


def maybe_fit_with_sample_weight(pipeline: Pipeline, estimator, x_train, y_train, sample_weights: np.ndarray | None):
    fit_params = {}
    if sample_weights is not None:
        try:
            signature = inspect.signature(estimator.fit)
            if "sample_weight" in signature.parameters:
                fit_params["model__sample_weight"] = sample_weights
        except (TypeError, ValueError):
            fit_params["model__sample_weight"] = sample_weights
    try:
        return pipeline.fit(x_train, y_train, **fit_params)
    except TypeError as exc:
        if "sample_weight" not in str(exc):
            raise
        print(">>> Estimator rejected sample_weight; refitting without sample weights.")
        return pipeline.fit(x_train, y_train)


def save_feature_importance(pipeline: Pipeline, feature_names: list[str], path: Path) -> None:
    model = pipeline.named_steps.get("model")
    values = None
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=np.float64)
    elif hasattr(model, "get_feature_importance"):
        try:
            values = np.asarray(model.get_feature_importance(), dtype=np.float64)
        except Exception:
            values = None
    if values is None or len(values) != len(feature_names):
        return
    df = pd.DataFrame({"feature": feature_names, "importance": values})
    df.sort_values("importance", ascending=False).to_csv(path, index=False, encoding="utf-8-sig")
    print(f">>> Feature importance saved to: {path}")


def train_sklearn_tabular_model(config, model_factory: Callable) -> None:
    apply_tabular_environment_overrides(config)
    data = prepare_tabular_training_data(config)
    estimator = model_factory(config)
    steps = [("imputer", SimpleImputer(strategy=getattr(config, "IMPUTER_STRATEGY", "median")))]
    if bool(getattr(config, "USE_STANDARD_SCALER", False)):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps)

    x_train = data["train_features"].to_numpy(dtype=np.float32)
    x_val = data["val_features"].to_numpy(dtype=np.float32)
    print(f">>> Fitting {config.MODEL_TAG} on shape={x_train.shape}...")
    maybe_fit_with_sample_weight(
        pipeline,
        estimator,
        x_train,
        data["y_train"],
        data["sample_weights"] if bool(getattr(config, "USE_SAMPLE_WEIGHT", True)) else None,
    )

    val_pred_raw = inverse_target(config, pipeline.predict(x_val))
    paths = build_result_paths(config, Path(data["metadata"]["val_csv_path"]), "val")
    val_metrics = write_evaluation_outputs(config, data["val_df"], data["y_val_raw"], val_pred_raw, paths, "val")
    print("=" * 72)
    print(f"{config.MODEL_TAG} validation | R2={val_metrics['R2']:.6f} | MAE={val_metrics['MAE']:.9f} | RMSE={val_metrics['RMSE']:.9f}")
    print("=" * 72)

    model_path = config.MODEL_DIR / getattr(config, "BEST_MODEL_NAME", "best_model.pkl")
    joblib.dump(pipeline, model_path)
    save_feature_importance(pipeline, data["metadata"]["feature_names"], config.MODEL_DIR / "feature_importance.csv")
    metadata = data["metadata"]
    metadata.update(
        {
            "best_model_name": model_path.name,
            "val_metrics": val_metrics,
            "estimator_class": estimator.__class__.__name__,
            "uses_standard_scaler": bool(getattr(config, "USE_STANDARD_SCALER", False)),
        }
    )
    save_json(config.MODEL_DIR / "training_metadata.json", metadata)
    print(f">>> Model saved to: {model_path}")


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


def load_tabular_metadata(config) -> dict:
    model_dir = resolve_model_dir(config)
    config.MODEL_DIR = model_dir
    config.SAVE_DIR = model_dir
    metadata_path = model_dir / "training_metadata.json"
    with metadata_path.open("r", encoding="utf-8") as file:
        import json

        metadata = json.load(file)
    config.SCALAR_COLS = list(metadata.get("scalar_feature_names", []))
    seq_config = metadata.get("sequence_config", {})
    if seq_config:
        config.SEQ_LEN = int(seq_config.get("seq_len", getattr(config, "SEQ_LEN", 4096)))
        config.WAVEFORM_FEATURE_COUNT = int(seq_config.get("waveform_feature_count", getattr(config, "WAVEFORM_FEATURE_COUNT", 256)))
        config.TARGET_DT = seq_config.get("target_dt", getattr(config, "TARGET_DT", None))
        config.SEQUENCE_CHANNELS = list(seq_config.get("channels", getattr(config, "SEQUENCE_CHANNELS", ["acc_scaled"])))
        config.SEQUENCE_CLIP_VALUE = float(seq_config.get("clip_value", getattr(config, "SEQUENCE_CLIP_VALUE", 8.0)))
    print(f">>> Model dir: {model_dir}")
    return metadata


def build_test_features(config, metadata: dict, test_df: pd.DataFrame) -> pd.DataFrame:
    include_waveform = bool(metadata.get("include_waveform_features", False))
    sequence_scaler = metadata.get("sequence_scaler")
    layout_width = int(max(metadata.get("layout_feature_width", config.MAX_DAMPER_FLOORS), config.MAX_DAMPER_FLOORS, test_df["num_floors"].max()))
    features, _, _ = build_tabular_feature_frame(
        config,
        test_df,
        layout_width,
        scalar_feature_names=list(metadata.get("scalar_feature_names", [])),
        include_waveform=include_waveform,
        sequence_scaler=sequence_scaler,
        waveform_feature_names=list(metadata.get("waveform_feature_names", [])),
        all_feature_names=list(metadata.get("feature_names", [])),
    )
    return features


def evaluate_sklearn_tabular_model(config) -> None:
    apply_tabular_environment_overrides(config)
    metadata = load_tabular_metadata(config)
    test_csv = resolve_test_csv_path(config, metadata)
    print(f">>> Test CSV: {test_csv}")
    test_df = read_valid_csv(config, test_csv)
    features = build_test_features(config, metadata, test_df)
    model_path = config.MODEL_DIR / metadata.get("best_model_name", getattr(config, "BEST_MODEL_NAME", "best_model.pkl"))
    pipeline = joblib.load(model_path)
    y_true = test_df[config.LABEL_COL].to_numpy(dtype=np.float32)
    y_pred = inverse_target(config, pipeline.predict(features.to_numpy(dtype=np.float32)))
    paths = build_result_paths(config, test_csv, "test")
    metrics = write_evaluation_outputs(config, test_df, y_true, y_pred, paths, "test")
    print("=" * 72)
    print(f"{config.MODEL_TAG} test | R2={metrics['R2']:.6f} | MAE={metrics['MAE']:.9f} | RMSE={metrics['RMSE']:.9f} | MAPE={metrics['MAPE']:.2f}%")
    print("=" * 72)


class TabularTensorDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray, sample_weights: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.reshape(-1, 1), dtype=torch.float32)
        self.sample_weights = torch.tensor(sample_weights.reshape(-1, 1), dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, idx: int):
        return self.features[idx], self.targets[idx], self.sample_weights[idx]


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_mult: int, dropout: float):
        super().__init__()
        hidden_dim = int(dim * hidden_mult)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.net(self.norm(x))


class TabularMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, block_count: int, dropout: float, hidden_mult: int):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        layers.extend(ResidualMLPBlock(hidden_dim, hidden_mult=hidden_mult, dropout=dropout) for _ in range(block_count))
        layers.append(nn.LayerNorm(hidden_dim))
        self.backbone = nn.Sequential(*layers)
        head_hidden = max(64, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_mlp_model_from_config(config, input_dim: int) -> TabularMLPRegressor:
    return TabularMLPRegressor(
        input_dim=int(input_dim),
        hidden_dim=int(getattr(config, "MLP_HIDDEN_DIM", 256)),
        block_count=int(getattr(config, "MLP_BLOCK_COUNT", 3)),
        dropout=float(getattr(config, "MLP_DROPOUT", 0.15)),
        hidden_mult=int(getattr(config, "MLP_HIDDEN_MULT", 2)),
    )


def transform_with_preprocessor(preprocessor: dict, features: np.ndarray) -> np.ndarray:
    x = preprocessor["imputer"].transform(features)
    x = preprocessor["scaler"].transform(x)
    return x.astype(np.float32)


def predict_torch_mlp(config, model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for features, _targets, _weights in loader:
            features = features.to(config.DEVICE)
            pred = model(features).detach().cpu().numpy().reshape(-1)
            preds.append(pred.astype(np.float32))
    return np.concatenate(preds, axis=0) if preds else np.asarray([], dtype=np.float32)


def plot_training_history(history: list[dict], path: Path) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(df["epoch"], df["MAE"], label="val_mae")
    if "val_selection_score" in df.columns:
        plt.plot(df["epoch"], df["val_selection_score"], label="val_score")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, format="svg", bbox_inches="tight")
    plt.close()


def train_torch_mlp_tabular_model(config) -> None:
    apply_tabular_environment_overrides(config)
    data = prepare_tabular_training_data(config)
    x_train_raw = data["train_features"].to_numpy(dtype=np.float32)
    x_val_raw = data["val_features"].to_numpy(dtype=np.float32)
    imputer = SimpleImputer(strategy=getattr(config, "IMPUTER_STRATEGY", "median"))
    scaler = StandardScaler()
    x_train = scaler.fit_transform(imputer.fit_transform(x_train_raw)).astype(np.float32)
    x_val = scaler.transform(imputer.transform(x_val_raw)).astype(np.float32)
    preprocessor = {"imputer": imputer, "scaler": scaler}
    joblib.dump(preprocessor, config.MODEL_DIR / "feature_preprocessor.pkl")

    train_dataset = TabularTensorDataset(x_train, data["y_train"], data["sample_weights"])
    val_dataset = TabularTensorDataset(x_val, data["y_val"], np.ones_like(data["y_val"], dtype=np.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_mlp_model_from_config(config, x_train.shape[1]).to(config.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.MIN_LR,
    )
    criterion = nn.SmoothL1Loss(beta=config.SMOOTH_L1_BETA, reduction="none")

    best_score = float("inf")
    best_epoch = 0
    history = []
    best_path = config.MODEL_DIR / getattr(config, "BEST_MODEL_NAME", "best_mlp_model.pth")
    print(model)
    print(f">>> Fitting {config.MODEL_TAG} on shape={x_train.shape}...")

    for epoch in range(1, int(config.NUM_EPOCHS) + 1):
        model.train()
        train_loss_total = 0.0
        train_count = 0
        for features, targets, weights in train_loader:
            features = features.to(config.DEVICE)
            targets = targets.to(config.DEVICE)
            weights = weights.to(config.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            preds = model(features)
            loss = (criterion(preds, targets) * weights).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step()
            batch_size = int(features.shape[0])
            train_loss_total += float(loss.item()) * batch_size
            train_count += batch_size

        y_val_pred_scaled = predict_torch_mlp(config, model, val_loader)
        y_val_pred = inverse_target(config, y_val_pred_scaled)
        val_metrics = calculate_regression_metrics(data["y_val_raw"], y_val_pred)
        val_score = calculate_selection_score(data["y_val_raw"], y_val_pred)
        scheduler.step(val_score)
        train_loss = train_loss_total / max(train_count, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_selection_score": val_score, **val_metrics})
        print(
            f"Epoch {epoch:03d}/{config.NUM_EPOCHS} | loss={train_loss:.6f} | "
            f"val_score={val_score:.9f} | val_mae={val_metrics['MAE']:.9f} | val_r2={val_metrics['R2']:.5f}"
        )
        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        elif epoch - best_epoch >= int(config.EARLY_STOPPING_PATIENCE):
            print(f">>> Early stopping at epoch {epoch}; best epoch={best_epoch}, best score={best_score:.9f}")
            break

    torch.save(model.state_dict(), config.MODEL_DIR / getattr(config, "FINAL_MODEL_NAME", "mlp_model.pth"))
    pd.DataFrame(history).to_csv(config.MODEL_DIR / "training_history.csv", index=False, encoding="utf-8-sig")
    plot_training_history(history, config.MODEL_DIR / "training_curves.svg")

    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE), strict=True)
    val_pred_raw = inverse_target(config, predict_torch_mlp(config, model, val_loader))
    paths = build_result_paths(config, Path(data["metadata"]["val_csv_path"]), "val")
    val_metrics = write_evaluation_outputs(config, data["val_df"], data["y_val_raw"], val_pred_raw, paths, "val")
    metadata = data["metadata"]
    metadata.update(
        {
            "best_model_name": best_path.name,
            "last_model_name": getattr(config, "FINAL_MODEL_NAME", "mlp_model.pth"),
            "feature_preprocessor_name": "feature_preprocessor.pkl",
            "val_metrics": val_metrics,
            "mlp_architecture": {
                "hidden_dim": int(config.MLP_HIDDEN_DIM),
                "block_count": int(config.MLP_BLOCK_COUNT),
                "dropout": float(config.MLP_DROPOUT),
                "hidden_mult": int(config.MLP_HIDDEN_MULT),
            },
        }
    )
    save_json(config.MODEL_DIR / "training_metadata.json", metadata)
    print(f">>> Best MLP weights saved to: {best_path}")


def evaluate_torch_mlp_tabular_model(config) -> None:
    apply_tabular_environment_overrides(config)
    metadata = load_tabular_metadata(config)
    test_csv = resolve_test_csv_path(config, metadata)
    print(f">>> Test CSV: {test_csv}")
    test_df = read_valid_csv(config, test_csv)
    features = build_test_features(config, metadata, test_df)
    preprocessor = joblib.load(config.MODEL_DIR / metadata.get("feature_preprocessor_name", "feature_preprocessor.pkl"))
    x_test = transform_with_preprocessor(preprocessor, features.to_numpy(dtype=np.float32))
    test_targets = test_df[config.LABEL_COL].to_numpy(dtype=np.float32)
    dummy_scaled = test_targets * float(config.LABEL_SCALE) if config.SCALE_TARGET else test_targets
    dataset = TabularTensorDataset(x_test, dummy_scaled, np.ones_like(test_targets, dtype=np.float32))
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    arch = metadata.get("mlp_architecture", {})
    config.MLP_HIDDEN_DIM = int(arch.get("hidden_dim", config.MLP_HIDDEN_DIM))
    config.MLP_BLOCK_COUNT = int(arch.get("block_count", config.MLP_BLOCK_COUNT))
    config.MLP_DROPOUT = float(arch.get("dropout", config.MLP_DROPOUT))
    config.MLP_HIDDEN_MULT = int(arch.get("hidden_mult", config.MLP_HIDDEN_MULT))
    model = build_mlp_model_from_config(config, x_test.shape[1]).to(config.DEVICE)
    weights_path = config.MODEL_DIR / metadata.get("best_model_name", getattr(config, "BEST_MODEL_NAME", "best_mlp_model.pth"))
    model.load_state_dict(torch.load(weights_path, map_location=config.DEVICE), strict=True)
    y_pred = inverse_target(config, predict_torch_mlp(config, model, loader))
    paths = build_result_paths(config, test_csv, "test")
    metrics = write_evaluation_outputs(config, test_df, test_targets, y_pred, paths, "test")
    print("=" * 72)
    print(f"{config.MODEL_TAG} test | R2={metrics['R2']:.6f} | MAE={metrics['MAE']:.9f} | RMSE={metrics['RMSE']:.9f} | MAPE={metrics['MAPE']:.2f}%")
    print("=" * 72)
