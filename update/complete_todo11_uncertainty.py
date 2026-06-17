# -*- coding: utf-8 -*-
"""Complete TODO11 uncertainty and conformal calibration artifacts.

This is an evaluation-only script. It reads frozen final multi-seed prediction
files, builds seed ensembles, calibrates prediction intervals on validation
predictions, and evaluates coverage on the locked test split. It does not train
or modify any model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
PROTOCOL_PATH = DEFAULT_OUTPUT_ROOT / "protocol" / "protocol_lock.json"
TARGET = "max_drift_ratio_raw"
STRUCTURE_COLS = ("num_floors", "floor_mass", "floor_height", "k_base_1_4", "Fy_add", "damper_layout")
TARGET_COVERAGE = 0.90
CALIBRATION_LEVELS = (0.50, 0.70, 0.80, 0.90, 0.95)
FIGURE_MODELS = ("2dcnn", "wavenet", "lstm", "lightgbm", "randomforest", "xgboost")
INTERVAL_METHODS = {
    "raw_seed_quantile_90": ("raw_lower_90", "raw_upper_90", "raw_width_90", "raw_covered_90"),
    "split_conformal_abs_90": ("abs_lower_90", "abs_upper_90", "abs_width_90", "abs_covered_90"),
    "split_conformal_adaptive_90": (
        "adaptive_lower_90",
        "adaptive_upper_90",
        "adaptive_width_90",
        "adaptive_covered_90",
    ),
    "ensemble_cqr_90": ("cqr_lower_90", "cqr_upper_90", "cqr_width_90", "cqr_covered_90"),
}
MAIN_METHOD = "split_conformal_adaptive_90"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(out):
        return out
    return None


def fmt(value: Any, digits: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}g}"


def conformal_quantile(scores: np.ndarray, coverage: float) -> float:
    clean = np.sort(np.asarray(scores, dtype=float)[np.isfinite(scores)])
    if clean.size == 0:
        return float("nan")
    alpha = 1.0 - coverage
    index = int(math.ceil((clean.size + 1) * (1.0 - alpha))) - 1
    index = min(max(index, 0), clean.size - 1)
    return float(clean[index])


def mean_std_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None, None, None
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        return mean, 0.0, 0.0
    std = statistics.stdev(clean)
    return mean, std, 1.96 * std / math.sqrt(len(clean))


def parse_layout_count(value: Any) -> int:
    text = str(value).strip().replace("(", "").replace(")", "")
    if not text:
        return 0
    count = 0
    for part in text.split(","):
        try:
            count += int(float(part.strip()))
        except ValueError:
            continue
    return count


def structure_signature(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in STRUCTURE_COLS:
        if col == "damper_layout":
            parts.append(df[col].astype(str).str.replace(" ", "", regex=False))
        elif col in {"num_floors"}:
            parts.append(pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64").astype(str))
        else:
            parts.append(pd.to_numeric(df[col], errors="coerce").round(6).astype(str))
    signature = parts[0]
    for part in parts[1:]:
        signature = signature + "|" + part
    return signature


def load_protocol_frames(protocol_path: Path) -> dict[str, pd.DataFrame]:
    protocol = load_json(protocol_path)
    dataset = protocol["dataset"]
    paths = {
        "train": Path(dataset["train_csv"]["path"]),
        "val": Path(dataset["val_csv"]["path"]),
        "test": Path(dataset["test_csv_reserved_locked"]["path"]),
    }
    usecols = [
        "sample_id",
        "txt_path",
        "num_floors",
        "floor_mass",
        "floor_height",
        "k_base_1_4",
        "Fy_add",
        "damper_layout",
        "wave_cluster",
        TARGET,
        "steel01_yielded",
        "steel02_yielded",
    ]
    frames: dict[str, pd.DataFrame] = {}
    for split, path in paths.items():
        df = pd.read_csv(path, usecols=usecols)
        df["row_id"] = np.arange(len(df), dtype=int)
        df["structure_signature"] = structure_signature(df)
        df["damper_install_count"] = df["damper_layout"].map(parse_layout_count)
        frames[split] = df
    train_waves = set(frames["train"]["txt_path"].astype(str))
    trainval_waves = train_waves | set(frames["val"]["txt_path"].astype(str))
    train_structures = set(frames["train"]["structure_signature"].astype(str))
    trainval_structures = train_structures | set(frames["val"]["structure_signature"].astype(str))
    for split, df in frames.items():
        df["unseen_wave_vs_train"] = ~df["txt_path"].astype(str).isin(train_waves)
        df["unseen_wave_vs_trainval"] = ~df["txt_path"].astype(str).isin(trainval_waves)
        df["unseen_structure_vs_train"] = ~df["structure_signature"].astype(str).isin(train_structures)
        df["unseen_structure_vs_trainval"] = ~df["structure_signature"].astype(str).isin(trainval_structures)
        frames[split] = df
    return frames


def read_prediction(path: Path, source_df: pd.DataFrame) -> np.ndarray:
    df = pd.read_csv(path)
    if "y_pred" in df.columns:
        pred_col = "y_pred"
    elif "Pred_Drift" in df.columns:
        pred_col = "Pred_Drift"
    else:
        raise ValueError(f"Cannot find prediction column in {path}")
    if len(df) != len(source_df):
        raise ValueError(f"Prediction row count mismatch in {path}: {len(df)} vs {len(source_df)}")
    if "sample_id" in df.columns:
        pred_keys = df["sample_id"].astype(str).to_numpy()
        source_keys = source_df["sample_id"].astype(str).to_numpy()
        if not np.array_equal(pred_keys, source_keys):
            raise ValueError(f"sample_id alignment failed for {path}")
    elif "txt_path" in df.columns:
        pred_keys = df["txt_path"].astype(str).to_numpy()
        source_keys = source_df["txt_path"].astype(str).to_numpy()
        if not np.array_equal(pred_keys, source_keys):
            raise ValueError(f"txt_path alignment failed for {path}")
    return pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)


def discover_prediction_sets(output_root: Path, frames: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    final_dir = output_root / "final_multiseed"
    prediction_sets: dict[str, dict[str, Any]] = {}
    for model_dir in sorted(path for path in final_dir.iterdir() if path.is_dir() and path.name != "logs"):
        model = model_dir.name
        seed_payloads: dict[int, dict[str, Path]] = {}
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            val_path = seed_dir / "predictions_val.csv"
            test_path = seed_dir / "predictions_test.csv"
            if val_path.exists() and test_path.exists():
                seed_payloads[seed] = {"val": val_path, "test": test_path}
        if len(seed_payloads) < 2:
            continue
        split_preds: dict[str, np.ndarray] = {}
        seeds = sorted(seed_payloads)
        for split in ("val", "test"):
            split_preds[split] = np.vstack([read_prediction(seed_payloads[seed][split], frames[split]) for seed in seeds])
        prediction_sets[model] = {"seeds": seeds, "predictions": split_preds}
    if not prediction_sets:
        raise FileNotFoundError(f"No usable multi-seed prediction sets found under {final_dir}")
    return prediction_sets


def build_ensemble_frame(model: str, split: str, source: pd.DataFrame, seed_predictions: np.ndarray) -> pd.DataFrame:
    preds = np.asarray(seed_predictions, dtype=float)
    out = source.copy()
    out["model"] = model
    out["split"] = split
    out["n_seeds"] = int(preds.shape[0])
    out["y_true"] = pd.to_numeric(source[TARGET], errors="coerce").to_numpy(dtype=float)
    out["pred_mean"] = preds.mean(axis=0)
    out["pred_var_seed"] = preds.var(axis=0, ddof=1 if preds.shape[0] > 1 else 0)
    out["pred_std_seed"] = np.sqrt(out["pred_var_seed"].to_numpy(dtype=float))
    out["pred_q05_seed"] = np.quantile(preds, 0.05, axis=0)
    out["pred_q50_seed"] = np.quantile(preds, 0.50, axis=0)
    out["pred_q95_seed"] = np.quantile(preds, 0.95, axis=0)
    out["pred_min_seed"] = preds.min(axis=0)
    out["pred_max_seed"] = preds.max(axis=0)
    out["signed_error"] = out["pred_mean"] - out["y_true"]
    out["abs_error"] = out["signed_error"].abs()
    return out


def fit_conformal_params(val_df: pd.DataFrame) -> dict[str, Any]:
    pred_std = val_df["pred_std_seed"].to_numpy(dtype=float)
    abs_error = val_df["abs_error"].to_numpy(dtype=float)
    median_std = float(np.nanmedian(pred_std)) if np.isfinite(pred_std).any() else 0.0
    median_abs = float(np.nanmedian(abs_error)) if np.isfinite(abs_error).any() else 0.0
    std_floor = max(1e-10, 0.10 * median_std, 0.01 * median_abs)
    scaled_score = abs_error / (pred_std + std_floor)
    cqr_score = np.maximum.reduce(
        [
            val_df["pred_q05_seed"].to_numpy(dtype=float) - val_df["y_true"].to_numpy(dtype=float),
            val_df["y_true"].to_numpy(dtype=float) - val_df["pred_q95_seed"].to_numpy(dtype=float),
            np.zeros(len(val_df), dtype=float),
        ]
    )
    params: dict[str, Any] = {
        "std_floor": std_floor,
        "levels": {},
    }
    for level in CALIBRATION_LEVELS:
        level_key = f"{int(round(level * 100)):02d}"
        params["levels"][level_key] = {
            "coverage": level,
            "q_abs": conformal_quantile(abs_error, level),
            "q_scaled": conformal_quantile(scaled_score, level),
            "q_cqr": conformal_quantile(cqr_score, level),
        }
    return params


def apply_intervals(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    std_floor = float(params["std_floor"])
    level_params = params["levels"]["90"]
    q_abs = float(level_params["q_abs"])
    q_scaled = float(level_params["q_scaled"])
    q_cqr = float(level_params["q_cqr"])
    adaptive_radius = q_scaled * (out["pred_std_seed"].to_numpy(dtype=float) + std_floor)
    out["raw_lower_90"] = out["pred_q05_seed"]
    out["raw_upper_90"] = out["pred_q95_seed"]
    out["raw_width_90"] = out["raw_upper_90"] - out["raw_lower_90"]
    out["raw_covered_90"] = (out["y_true"] >= out["raw_lower_90"]) & (out["y_true"] <= out["raw_upper_90"])
    out["abs_lower_90"] = out["pred_mean"] - q_abs
    out["abs_upper_90"] = out["pred_mean"] + q_abs
    out["abs_width_90"] = out["abs_upper_90"] - out["abs_lower_90"]
    out["abs_covered_90"] = (out["y_true"] >= out["abs_lower_90"]) & (out["y_true"] <= out["abs_upper_90"])
    out["adaptive_lower_90"] = out["pred_mean"] - adaptive_radius
    out["adaptive_upper_90"] = out["pred_mean"] + adaptive_radius
    out["adaptive_width_90"] = out["adaptive_upper_90"] - out["adaptive_lower_90"]
    out["adaptive_covered_90"] = (out["y_true"] >= out["adaptive_lower_90"]) & (
        out["y_true"] <= out["adaptive_upper_90"]
    )
    out["cqr_lower_90"] = out["pred_q05_seed"] - q_cqr
    out["cqr_upper_90"] = out["pred_q95_seed"] + q_cqr
    out["cqr_width_90"] = out["cqr_upper_90"] - out["cqr_lower_90"]
    out["cqr_covered_90"] = (out["y_true"] >= out["cqr_lower_90"]) & (out["y_true"] <= out["cqr_upper_90"])
    return out


def summarize_interval(
    df: pd.DataFrame,
    model: str,
    split: str,
    method: str,
    group_type: str,
    group_value: str,
    mask: pd.Series | np.ndarray,
    claim_scope: str,
) -> dict[str, Any]:
    lower_col, upper_col, width_col, covered_col = INTERVAL_METHODS[method]
    sub = df.loc[mask].copy()
    row: dict[str, Any] = {
        "model": model,
        "split": split,
        "interval_method": method,
        "target_coverage": TARGET_COVERAGE,
        "group_type": group_type,
        "group_value": group_value,
        "claim_scope": claim_scope,
        "n": int(len(sub)),
    }
    if sub.empty:
        row.update(
            {
                "coverage": None,
                "coverage_error": None,
                "avg_interval_width": None,
                "median_interval_width": None,
                "mean_abs_error": None,
                "p95_abs_error": None,
                "mean_pred_std": None,
                "mean_y_true": None,
            }
        )
        return row
    coverage = float(sub[covered_col].mean())
    row.update(
        {
            "coverage": coverage,
            "coverage_error": coverage - TARGET_COVERAGE,
            "avg_interval_width": float(sub[width_col].mean()),
            "median_interval_width": float(sub[width_col].median()),
            "mean_abs_error": float(sub["abs_error"].mean()),
            "p95_abs_error": float(np.quantile(sub["abs_error"].to_numpy(dtype=float), 0.95)),
            "mean_pred_std": float(sub["pred_std_seed"].mean()),
            "mean_y_true": float(sub["y_true"].mean()),
            "lower_mean": float(sub[lower_col].mean()),
            "upper_mean": float(sub[upper_col].mean()),
        }
    )
    return row


def group_rows(df: pd.DataFrame, model: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = list(INTERVAL_METHODS)
    groups: list[tuple[str, str, pd.Series | np.ndarray, str]] = [
        ("all", "all", np.ones(len(df), dtype=bool), "final_locked_test" if split == "test" else "calibration_diagnostic"),
        ("ood", "exact_unseen_wave_vs_trainval", df["unseen_wave_vs_trainval"], "ood_final" if split == "test" else "diagnostic"),
        (
            "ood",
            "exact_unseen_structure_vs_trainval",
            df["unseen_structure_vs_trainval"],
            "ood_final" if split == "test" else "diagnostic",
        ),
        ("ood", "exact_unseen_wave_vs_train", df["unseen_wave_vs_train"], "diagnostic" if split == "val" else "final_locked_test"),
        (
            "ood",
            "exact_unseen_structure_vs_train",
            df["unseen_structure_vs_train"],
            "diagnostic" if split == "val" else "final_locked_test",
        ),
    ]
    for col in ("num_floors", "wave_cluster", "damper_install_count", "steel02_yielded"):
        for value in sorted(df[col].dropna().unique().tolist()):
            groups.append((col, str(value), df[col] == value, "final_locked_test" if split == "test" else "calibration_diagnostic"))
    for method in methods:
        for group_type, group_value, mask, claim_scope in groups:
            rows.append(summarize_interval(df, model, split, method, group_type, group_value, mask, claim_scope))
    return rows


def tail_rows(df: pd.DataFrame, model: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y = df["y_true"].to_numpy(dtype=float)
    tail_defs: list[tuple[str, pd.Series | np.ndarray, str]] = [
        ("all", np.ones(len(df), dtype=bool), "final_locked_test" if split == "test" else "calibration_diagnostic"),
        ("target >= split_p90", y >= float(np.quantile(y, 0.90)), "final_locked_test" if split == "test" else "calibration_diagnostic"),
        ("target >= split_p95", y >= float(np.quantile(y, 0.95)), "final_locked_test" if split == "test" else "calibration_diagnostic"),
        ("target >= split_p99", y >= float(np.quantile(y, 0.99)), "final_locked_test" if split == "test" else "calibration_diagnostic"),
        ("target >= 0.005", y >= 0.005, "final_locked_test" if split == "test" else "calibration_diagnostic"),
        ("target >= 0.010", y >= 0.010, "final_locked_test_absent_if_n0" if split == "test" else "diagnostic_high_drift"),
        ("target >= 0.015", y >= 0.015, "final_locked_test_absent_if_n0" if split == "test" else "diagnostic_high_drift"),
        ("target >= 0.020", y >= 0.020, "final_locked_test_absent_if_n0" if split == "test" else "diagnostic_high_drift"),
        ("steel01_yielded=1", df["steel01_yielded"].astype(int) == 1, "final_locked_test_absent_if_n0" if split == "test" else "diagnostic_yielded"),
        ("steel02_yielded=1", df["steel02_yielded"].astype(int) == 1, "final_locked_test" if split == "test" else "calibration_diagnostic"),
        ("OOD exact unseen wave vs train+val", df["unseen_wave_vs_trainval"], "ood_final" if split == "test" else "diagnostic"),
        (
            "OOD exact unseen structure vs train+val",
            df["unseen_structure_vs_trainval"],
            "ood_final" if split == "test" else "diagnostic",
        ),
    ]
    for method in INTERVAL_METHODS:
        for tail_definition, mask, claim_scope in tail_defs:
            rows.append(summarize_interval(df, model, split, method, "tail", tail_definition, mask, claim_scope))
    return rows


def correlation_rows(df: pd.DataFrame, model: str, split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variables = {
        "pred_std_seed": "pred_std_seed",
        "raw_interval_width": "raw_width_90",
        "adaptive_interval_width": "adaptive_width_90",
        "cqr_interval_width": "cqr_width_90",
    }
    high_error_threshold = float(np.quantile(df["abs_error"].to_numpy(dtype=float), 0.90))
    high_mask = df["abs_error"] >= high_error_threshold
    for name, col in variables.items():
        spearman = float(df[["abs_error", col]].corr(method="spearman").iloc[0, 1])
        pearson = float(df[["abs_error", col]].corr(method="pearson").iloc[0, 1])
        top_mean = float(df.loc[high_mask, col].mean()) if high_mask.any() else None
        other_mean = float(df.loc[~high_mask, col].mean()) if (~high_mask).any() else None
        ratio = top_mean / other_mean if other_mean and math.isfinite(other_mean) and other_mean != 0.0 else None
        rows.append(
            {
                "model": model,
                "split": split,
                "uncertainty_variable": name,
                "n": int(len(df)),
                "high_error_definition": "top_10pct_abs_error_within_model_split",
                "high_error_threshold": high_error_threshold,
                "spearman_abs_error_vs_uncertainty": spearman,
                "pearson_abs_error_vs_uncertainty": pearson,
                "mean_uncertainty_high_error": top_mean,
                "mean_uncertainty_other": other_mean,
                "high_error_uncertainty_ratio": ratio,
            }
        )
    return rows


def calibration_curve_rows(
    ensemble_frames: dict[tuple[str, str], pd.DataFrame], params_by_model: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, params in params_by_model.items():
        test_df = ensemble_frames[(model, "test")]
        for level in CALIBRATION_LEVELS:
            level_key = f"{int(round(level * 100)):02d}"
            p = params["levels"][level_key]
            radius = float(p["q_scaled"]) * (test_df["pred_std_seed"].to_numpy(dtype=float) + float(params["std_floor"]))
            lower = test_df["pred_mean"].to_numpy(dtype=float) - radius
            upper = test_df["pred_mean"].to_numpy(dtype=float) + radius
            covered = (test_df["y_true"].to_numpy(dtype=float) >= lower) & (test_df["y_true"].to_numpy(dtype=float) <= upper)
            rows.append(
                {
                    "model": model,
                    "split": "test",
                    "interval_family": "split_conformal_adaptive",
                    "nominal_coverage": level,
                    "empirical_coverage": float(covered.mean()),
                    "coverage_error": float(covered.mean() - level),
                    "avg_interval_width": float(np.mean(upper - lower)),
                    "n": int(len(test_df)),
                }
            )
    return rows


def make_calibration_curve(curve_rows: list[dict[str, Any]], output_root: Path) -> tuple[Path, Path]:
    fig_dir = output_root / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    curve_df = pd.DataFrame(curve_rows)
    plt.figure(figsize=(7.5, 6.0))
    plt.plot([0.45, 1.0], [0.45, 1.0], color="black", linestyle="--", linewidth=1.0, label="ideal")
    for model in FIGURE_MODELS:
        sub = curve_df.loc[curve_df["model"] == model].sort_values("nominal_coverage")
        if sub.empty:
            continue
        plt.plot(sub["nominal_coverage"], sub["empirical_coverage"], marker="o", linewidth=1.8, label=model)
    plt.xlabel("Nominal coverage")
    plt.ylabel("Empirical locked-test coverage")
    plt.title("Split conformal adaptive interval calibration")
    plt.xlim(0.45, 0.98)
    plt.ylim(0.45, 1.0)
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, frameon=True)
    plt.tight_layout()
    png = fig_dir / "calibration_curve.png"
    svg = fig_dir / "calibration_curve.svg"
    plt.savefig(png, dpi=220)
    plt.savefig(svg)
    plt.close()
    return png, svg


def width_error_rows(ensemble_frames: dict[tuple[str, str], pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in FIGURE_MODELS:
        key = (model, "test")
        if key not in ensemble_frames:
            continue
        df = ensemble_frames[key].copy()
        if df["adaptive_width_90"].nunique(dropna=True) <= 1:
            df["width_bin"] = 0
        else:
            ranks = df["adaptive_width_90"].rank(method="first")
            df["width_bin"] = pd.qcut(ranks, q=10, labels=False, duplicates="drop")
        for bin_id, sub in df.groupby("width_bin", dropna=False):
            rows.append(
                {
                    "model": model,
                    "split": "test",
                    "interval_method": MAIN_METHOD,
                    "width_bin": int(bin_id) if pd.notna(bin_id) else -1,
                    "n": int(len(sub)),
                    "mean_interval_width": float(sub["adaptive_width_90"].mean()),
                    "mean_abs_error": float(sub["abs_error"].mean()),
                    "coverage": float(sub["adaptive_covered_90"].mean()),
                    "mean_pred_std": float(sub["pred_std_seed"].mean()),
                }
            )
    return rows


def make_width_error_plot(rows: list[dict[str, Any]], output_root: Path) -> tuple[Path, Path]:
    fig_dir = output_root / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    plt.figure(figsize=(8.0, 5.8))
    for model in FIGURE_MODELS:
        sub = df.loc[df["model"] == model].sort_values("mean_interval_width")
        if sub.empty:
            continue
        plt.plot(sub["mean_interval_width"], sub["mean_abs_error"], marker="o", linewidth=1.8, label=model)
    plt.xlabel("Mean adaptive 90% interval width by width decile")
    plt.ylabel("Mean absolute error")
    plt.title("Interval width vs. empirical error")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, frameon=True)
    plt.tight_layout()
    png = fig_dir / "interval_width_vs_error.png"
    svg = fig_dir / "interval_width_vs_error.svg"
    plt.savefig(png, dpi=220)
    plt.savefig(svg)
    plt.close()
    return png, svg


def make_table7(coverage_by_group: pd.DataFrame) -> pd.DataFrame:
    sub = coverage_by_group.loc[
        (coverage_by_group["split"] == "test")
        & (coverage_by_group["interval_method"].isin(["split_conformal_adaptive_90", "ensemble_cqr_90"]))
        & (
            ((coverage_by_group["group_type"] == "all") & (coverage_by_group["group_value"] == "all"))
            | (
                (coverage_by_group["group_type"] == "ood")
                & (coverage_by_group["group_value"].isin(["exact_unseen_wave_vs_trainval", "exact_unseen_structure_vs_trainval"]))
            )
        )
    ].copy()
    sub = sub.sort_values(["interval_method", "coverage_error", "avg_interval_width"], key=lambda series: series.abs() if series.name == "coverage_error" else series)
    cols = [
        "model",
        "interval_method",
        "group_type",
        "group_value",
        "n",
        "coverage",
        "coverage_error",
        "avg_interval_width",
        "median_interval_width",
        "mean_abs_error",
        "mean_pred_std",
        "claim_scope",
    ]
    return sub[cols]


def markdown_table(df: pd.DataFrame, max_rows: int = 25) -> str:
    use = df.head(max_rows).copy()
    if use.empty:
        return "_No rows._"
    for col in use.columns:
        if pd.api.types.is_numeric_dtype(use[col]):
            use[col] = use[col].map(lambda value: fmt(value) if pd.notna(value) else "NA")
    use = use.astype(str)
    headers = list(use.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in use.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "/") for col in headers) + " |")
    return "\n".join(lines)


def create_report(
    output_root: Path,
    table7: pd.DataFrame,
    coverage_tail: pd.DataFrame,
    correlations: pd.DataFrame,
    manifest: dict[str, Any],
    outputs: dict[str, str],
) -> Path:
    report_path = (
        PROJECT_ROOT
        / "md_files"
        / "02_research_and_paper"
        / "literature_reviews"
        / "TODO11_不确定性与conformal校准完成报告_20260617.md"
    )
    main_rows = table7.loc[
        (table7["model"] == "2dcnn")
        & (table7["interval_method"] == MAIN_METHOD)
        & (table7["group_type"] == "all")
    ]
    tail_2dcnn = coverage_tail.loc[
        (coverage_tail["model"] == "2dcnn")
        & (coverage_tail["split"] == "test")
        & (coverage_tail["interval_method"] == MAIN_METHOD)
        & (coverage_tail["group_value"].isin(["target >= 0.005", "target >= 0.010", "steel01_yielded=1"]))
    ].copy()
    corr_2dcnn = correlations.loc[
        (correlations["model"] == "2dcnn")
        & (correlations["split"] == "test")
        & (correlations["uncertainty_variable"].isin(["pred_std_seed", "adaptive_interval_width"]))
    ].copy()
    lines = [
        "# TODO11 不确定性与 conformal 校准完成报告",
        "",
        "生成日期：2026-06-17",
        "",
        "## 1. 工作范围",
        "",
        "本轮只读取 TODO5 frozen final multi-seed predictions 与 TODO1 locked protocol CSV，不训练、不调参、不改测试集。方法是：对每个模型的 5 个 seed 构造 ensemble 均值、方差和经验 5/50/95 分位数；在 validation split 上拟合 split conformal 校准半径；在 locked test 上报告 90% prediction interval coverage、average interval width、tail coverage、OOD coverage 和误差-不确定性相关性。",
        "",
        "## 2. 已生成文件",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 3. TODO11 完成标准核对",
            "",
            "| TODO11 要求 | 本轮状态 | 证据文件 |",
            "|---|---|---|",
            "| 基于 5 seed ensemble 输出预测均值和方差 | 完成；每个样本含 `pred_mean`, `pred_var_seed`, `pred_std_seed` | `prediction_intervals.csv` |",
            "| 输出 5%、50%、95% 分位预测 | 完成；采用 5-seed empirical quantile，不启动新 quantile-regression 训练 | `prediction_intervals.csv` |",
            "| 使用 validation/calibration split 做 conformal calibration | 完成；validation 作为 calibration split，locked test 独立评估 | `conformal_calibration_params.json` |",
            "| 报告 90% prediction interval coverage | 完成；含 raw、absolute conformal、adaptive conformal、ensemble-CQR 四种区间 | `coverage_by_group.csv`, `table7_uncertainty_calibration.csv` |",
            "| 报告 average interval width | 完成 | `coverage_by_group.csv`, `coverage_by_tail.csv` |",
            "| 单独报告 tail coverage 和 OOD coverage | 完成；test OOD 为 exact unseen-wave/structure，high-drift 独立样本缺失则显式标记 | `coverage_by_tail.csv` |",
            "| 检查高误差样本是否对应更高不确定性 | 完成；输出 Spearman/Pearson 和 top-error uncertainty ratio | `uncertainty_error_correlation.csv`, `interval_width_vs_error.png` |",
            "",
            "## 4. 主结果摘录",
            "",
        ]
    )
    if not main_rows.empty:
        row = main_rows.iloc[0]
        lines.append(
            f"- `2dcnn` locked-test adaptive conformal 90% interval: coverage = `{fmt(row['coverage'])}`, "
            f"average width = `{fmt(row['avg_interval_width'])}`, mean absolute error = `{fmt(row['mean_abs_error'])}`."
        )
    if not tail_2dcnn.empty:
        lines.append("")
        lines.append("`2dcnn` tail/OOD coverage 摘录：")
        lines.append("")
        lines.append(markdown_table(tail_2dcnn[["group_value", "n", "coverage", "avg_interval_width", "claim_scope"]], max_rows=8))
    if not corr_2dcnn.empty:
        lines.append("")
        lines.append("`2dcnn` 误差-不确定性相关性摘录：")
        lines.append("")
        lines.append(
            markdown_table(
                corr_2dcnn[
                    [
                        "uncertainty_variable",
                        "spearman_abs_error_vs_uncertainty",
                        "high_error_uncertainty_ratio",
                    ]
                ],
                max_rows=8,
            )
        )
    lines.extend(
        [
            "",
            "## 5. 论文主张边界",
            "",
            "- 可以写：模型不只报告点预测，还给出了模型间 seed spread、empirical quantile 和 conformal-calibrated 90% prediction interval；locked test 上覆盖率和区间宽度已独立报告。",
            "- 可以写：locked test 的 OOD coverage 等价于 exact unseen-wave / exact unseen-structure coverage，因为 TODO10 已确认 test 的 exact `txt_path` 与 structure signature 100% 未出现在 train+val。",
            "- 必须谨慎写：validation 曾用于 HPO/early stopping，因此 split conformal 的严格有限样本保证不如专门保留 calibration set 干净；本文应表述为“post-hoc validation-calibrated intervals with independent locked-test empirical coverage”。",
            "- 不能写：已经训练了专门的 quantile regression 模型。本轮为了不干扰训练任务，采用的是 5-seed empirical quantile + conformal correction。",
            "- 不能写：高漂移或主体屈服工况下区间已经独立校准。TODO10 已确认 locked test 中 `target >= 0.010` 和 `steel01_yielded=1` 样本为 0；validation high-drift 只能诊断。",
            "",
            "## 6. 参考文献与规范",
            "",
            "以下文献已于 2026-06-17 核对，基础方法文献用于定义 deep ensemble / conformal / CQR，2025-2026 文献用于说明 conformal UQ 在复杂学习场景和物理 surrogate model 中仍是前沿可靠性工具，REFORMS 用于约束 coverage、区间宽度和 claim boundary 的报告规范。",
            "",
            "1. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS. https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles",
            "2. Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. NeurIPS. https://papers.neurips.cc/paper/8613-conformalized-quantile-regression",
            "3. Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4), 494-591. https://arxiv.org/abs/2107.07511",
            "4. Gao, R., & Liu, W. (2025). Model uncertainty quantification by conformal prediction in continual learning. ICML 2025, PMLR 267:18453-18469. https://proceedings.mlr.press/v267/gao25i.html",
            "5. Gopakumar, V., Gray, A., Oskarsson, J., Giles, D., Zanisi, L., Kusner, M., Pamela, S., & Deisenroth, M. P. (2026). Uncertainty quantification of surrogate models using conformal prediction. Machine Learning: Science and Technology. https://www.deisenroth.cc/publication/gopakumar-2026/",
            "6. Kapoor, S., Cantrell, E. M., Peng, K., et al. (2024). REFORMS: Consensus-based recommendations for machine-learning-based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452",
            "7. Gao, J., Du, K., & Qi, J. (2025). Quantifying the uncertainty of structural parameters using machine learning-based surrogate models. ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering. DOI: 10.1061/AJRUA6.RUENG-1550",
            "",
            "## 7. TODO11 状态结论",
            "",
            f"TODO11 已完成为 non-training uncertainty/conformal evaluation。共处理 `{manifest['model_count']}` 个模型、`{manifest['seed_count_per_complete_model']}` 个 seed、validation calibration rows `{manifest['calibration_rows']}`、locked test rows `{manifest['test_rows']}`。若论文必须声称“专门 quantile regression 模型”，需要另开 TODO11b 训练 0.05/0.50/0.95 quantile models；当前版本已经足以支撑 conformal interval coverage 和工程用户不确定性提示。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Complete TODO11 uncertainty and conformal calibration artifacts.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    args = parser.parse_args()

    output_root = args.output_root
    uncertainty_dir = output_root / "uncertainty"
    table_dir = output_root / "paper_tables"
    figure_dir = output_root / "paper_figures"
    stat_dir = output_root / "statistics"
    uncertainty_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    stat_dir.mkdir(parents=True, exist_ok=True)

    frames = load_protocol_frames(args.protocol)
    prediction_sets = discover_prediction_sets(output_root, frames)

    ensemble_frames: dict[tuple[str, str], pd.DataFrame] = {}
    conformal_params: dict[str, dict[str, Any]] = {}
    prediction_interval_rows: list[dict[str, Any]] = []
    coverage_group_rows: list[dict[str, Any]] = []
    coverage_tail_rows: list[dict[str, Any]] = []
    correlation_output_rows: list[dict[str, Any]] = []
    model_seed_counts: dict[str, int] = {}

    output_columns = [
        "model",
        "split",
        "n_seeds",
        "row_id",
        "sample_id",
        "txt_path",
        "num_floors",
        "wave_cluster",
        "damper_install_count",
        "steel01_yielded",
        "steel02_yielded",
        "unseen_wave_vs_trainval",
        "unseen_structure_vs_trainval",
        "y_true",
        "pred_mean",
        "pred_var_seed",
        "pred_std_seed",
        "pred_q05_seed",
        "pred_q50_seed",
        "pred_q95_seed",
        "abs_error",
        "raw_lower_90",
        "raw_upper_90",
        "raw_width_90",
        "raw_covered_90",
        "abs_lower_90",
        "abs_upper_90",
        "abs_width_90",
        "abs_covered_90",
        "adaptive_lower_90",
        "adaptive_upper_90",
        "adaptive_width_90",
        "adaptive_covered_90",
        "cqr_lower_90",
        "cqr_upper_90",
        "cqr_width_90",
        "cqr_covered_90",
    ]

    for model, payload in prediction_sets.items():
        seeds = payload["seeds"]
        model_seed_counts[model] = len(seeds)
        val_df = build_ensemble_frame(model, "val", frames["val"], payload["predictions"]["val"])
        params = fit_conformal_params(val_df)
        conformal_params[model] = {"seeds": seeds, **params}
        for split in ("val", "test"):
            ens = build_ensemble_frame(model, split, frames[split], payload["predictions"][split])
            ens = apply_intervals(ens, params)
            ensemble_frames[(model, split)] = ens
            prediction_interval_rows.extend(ens[output_columns].to_dict("records"))
            coverage_group_rows.extend(group_rows(ens, model, split))
            coverage_tail_rows.extend(tail_rows(ens, model, split))
            correlation_output_rows.extend(correlation_rows(ens, model, split))

    prediction_intervals_path = uncertainty_dir / "prediction_intervals.csv"
    write_csv(prediction_intervals_path, prediction_interval_rows, fields=output_columns)

    coverage_group_path = uncertainty_dir / "coverage_by_group.csv"
    write_csv(coverage_group_path, coverage_group_rows)
    coverage_tail_path = uncertainty_dir / "coverage_by_tail.csv"
    write_csv(coverage_tail_path, coverage_tail_rows)
    correlation_path = uncertainty_dir / "uncertainty_error_correlation.csv"
    write_csv(correlation_path, correlation_output_rows)
    conformal_params_path = uncertainty_dir / "conformal_calibration_params.json"
    save_json(conformal_params_path, conformal_params)

    curve = calibration_curve_rows(ensemble_frames, conformal_params)
    curve_path = uncertainty_dir / "calibration_curve_points.csv"
    write_csv(curve_path, curve)
    calibration_png, calibration_svg = make_calibration_curve(curve, output_root)

    width_error = width_error_rows(ensemble_frames)
    width_error_path = uncertainty_dir / "interval_width_vs_error_points.csv"
    write_csv(width_error_path, width_error)
    width_error_png, width_error_svg = make_width_error_plot(width_error, output_root)

    coverage_group_df = pd.DataFrame(coverage_group_rows)
    coverage_tail_df = pd.DataFrame(coverage_tail_rows)
    correlations_df = pd.DataFrame(correlation_output_rows)
    table7 = make_table7(coverage_group_df)
    table7_path = table_dir / "table7_uncertainty_calibration.csv"
    table7.to_csv(table7_path, index=False, encoding="utf-8-sig")
    table7_md = table_dir / "table7_uncertainty_calibration.md"
    table7_md.write_text(markdown_table(table7, max_rows=40) + "\n", encoding="utf-8")

    seed_count_values = sorted(set(model_seed_counts.values()))
    manifest: dict[str, Any] = {
        "run_name": "todo11_uncertainty_conformal_completion",
        "generated_at": "2026-06-17",
        "status": {
            "five_seed_ensemble_mean_variance": "complete",
            "empirical_quantile_5_50_95": "complete_from_seed_ensemble_no_new_training",
            "dedicated_quantile_regression_training": "not_run_to_avoid_training_interference",
            "validation_conformal_calibration": "complete_post_hoc_validation_calibration",
            "locked_test_coverage": "complete",
            "tail_and_ood_coverage": "complete_with_high_drift_absence_flagged",
            "error_uncertainty_association": "complete",
        },
        "model_count": len(prediction_sets),
        "models": sorted(prediction_sets),
        "seed_count_by_model": model_seed_counts,
        "seed_count_per_complete_model": seed_count_values,
        "calibration_split": "val",
        "calibration_rows": int(len(frames["val"])),
        "test_rows": int(len(frames["test"])),
        "target_coverage": TARGET_COVERAGE,
        "main_interval_method": MAIN_METHOD,
        "claim_boundaries": [
            "Validation was previously used in HPO/early stopping; report intervals as post-hoc validation-calibrated with independent locked-test empirical coverage.",
            "Dedicated quantile regression models were not trained in this non-interference TODO11 run.",
            "Locked test has no target >= 0.010 or steel01_yielded=1 samples; high-drift interval coverage requires the TODO10 OpenSees stress plan.",
            "Raw seed quantile intervals from only five seeds are diagnostic and should not be used as calibrated uncertainty claims without conformal correction.",
        ],
    }
    outputs = {
        "prediction_intervals": str(prediction_intervals_path),
        "coverage_by_group": str(coverage_group_path),
        "coverage_by_tail": str(coverage_tail_path),
        "uncertainty_error_correlation": str(correlation_path),
        "conformal_calibration_params": str(conformal_params_path),
        "calibration_curve_points": str(curve_path),
        "interval_width_vs_error_points": str(width_error_path),
        "table7_uncertainty_calibration": str(table7_path),
        "table7_uncertainty_calibration_md": str(table7_md),
        "calibration_curve_png": str(calibration_png),
        "calibration_curve_svg": str(calibration_svg),
        "interval_width_vs_error_png": str(width_error_png),
        "interval_width_vs_error_svg": str(width_error_svg),
    }
    manifest["outputs"] = outputs
    report_path = create_report(output_root, table7, coverage_tail_df, correlations_df, manifest, outputs)
    outputs["todo11_completion_report"] = str(report_path)
    manifest_path = stat_dir / "todo11_completion_manifest.json"
    save_json(manifest_path, manifest)

    print(f">>> TODO11 prediction intervals: {prediction_intervals_path}")
    print(f">>> TODO11 coverage by group: {coverage_group_path}")
    print(f">>> TODO11 coverage by tail: {coverage_tail_path}")
    print(f">>> TODO11 Table 7: {table7_path}")
    print(f">>> TODO11 calibration curve: {calibration_png}")
    print(f">>> TODO11 width/error figure: {width_error_png}")
    print(f">>> TODO11 manifest: {manifest_path}")
    print(f">>> TODO11 report: {report_path}")


if __name__ == "__main__":
    main()
