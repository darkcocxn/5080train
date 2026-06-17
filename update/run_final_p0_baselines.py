"""Run TODO5 lightweight P0 baselines on the locked train/val/test split.

The script trains CPU-only sanity/linear/tree baselines declared by TODO4 and
writes per-seed predictions and metrics. It does not start Optuna or any deep
learning/GPU model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_JSON = PROJECT_ROOT / "publication_eval_20260614" / "protocol" / "protocol_lock.json"
TARGET_COL = "max_drift_ratio_raw"
STATUS_COL = "analysis_status"
SPLIT_COL = "split"
DEFAULT_MODELS = ("dummy_mean", "dummy_median", "ridge", "elasticnet", "extratrees", "histgradientboosting")
TAIL_THRESHOLDS = (0.005, 0.010, 0.015, 0.020)
EXCLUDE_COLUMNS = {
    "sample_id",
    "split",
    "stage",
    "round_idx",
    "txt_path",
    "image_path",
    TARGET_COL,
    "max_drift_ratio_q1e4",
    "steel01_yielded",
    "steel02_yielded",
    "peak_damper_force",
    "yield_margin",
    "peak_story_id",
    "analysis_status",
    "failure_reason",
    "failure_category",
    "analysis_return_code",
    "analysis_failed_step",
    "analysis_failed_time",
    "nonconvergence_flag",
}
PREFERRED_CATEGORICAL = ("damper_layout", "period_compliant", "period_check_status", "wave_cluster")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def protocol_paths(protocol: dict[str, Any]) -> tuple[Path, Path, Path]:
    dataset = protocol["dataset"]
    return (
        Path(dataset["train_csv"]["path"]).resolve(),
        Path(dataset["val_csv"]["path"]).resolve(),
        Path(dataset["test_csv_reserved_locked"]["path"]).resolve(),
    )


def read_valid_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if STATUS_COL in df.columns:
        df = df[df[STATUS_COL].astype(str) == "ok"].copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.dropna(subset=[TARGET_COL]).reset_index(drop=True)


def infer_feature_columns(train_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    candidates = [column for column in train_df.columns if column not in EXCLUDE_COLUMNS]
    categorical = [column for column in PREFERRED_CATEGORICAL if column in candidates]
    numeric: list[str] = []
    for column in candidates:
        if column in categorical:
            continue
        converted = pd.to_numeric(train_df[column], errors="coerce")
        if converted.notna().sum() > 0:
            numeric.append(column)
        else:
            categorical.append(column)
    return numeric, categorical


def prepare_features(df: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]) -> pd.DataFrame:
    feature_df = pd.DataFrame(index=df.index)
    for column in numeric_cols:
        feature_df[column] = pd.to_numeric(df[column], errors="coerce") if column in df.columns else np.nan
    for column in categorical_cols:
        feature_df[column] = df[column].astype("string").fillna("<missing>") if column in df.columns else "<missing>"
    return feature_df


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=False)


def model_factory(name: str, seed: int, n_jobs: int) -> Any:
    if name == "dummy_mean":
        return DummyRegressor(strategy="mean")
    if name == "dummy_median":
        return DummyRegressor(strategy="median")
    if name == "ridge":
        return Ridge(alpha=1.0)
    if name == "elasticnet":
        return ElasticNet(alpha=1e-4, l1_ratio=0.1, max_iter=20000, random_state=seed)
    if name == "extratrees":
        return ExtraTreesRegressor(
            n_estimators=400,
            max_features=0.8,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "histgradientboosting":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=500,
            max_leaf_nodes=31,
            l2_regularization=1e-4,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=seed,
        )
    raise ValueError(f"Unknown P0 baseline: {name}")


def build_pipeline(name: str, seed: int, numeric_cols: list[str], categorical_cols: list[str], n_jobs: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", model_factory(name, seed, n_jobs)),
        ]
    )


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    error = y_pred - y_true
    metrics: dict[str, Any] = {
        "samples": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        "mape": float(np.mean(np.abs(error) / np.maximum(np.abs(y_true), 1e-12)) * 100.0),
        "smape": smape(y_true, y_pred),
        "bias": float(np.mean(error)),
        "max_underprediction": float(np.max(y_true - y_pred)),
        "p95_abs_error": float(np.quantile(np.abs(error), 0.95)),
    }
    for q in (0.90, 0.95):
        threshold = float(np.quantile(y_true, q))
        mask = y_true >= threshold
        prefix = f"tail_p{int(q * 100)}"
        metrics[f"{prefix}_threshold"] = threshold
        metrics[f"{prefix}_count"] = int(mask.sum())
        metrics[f"{prefix}_mae"] = float(mean_absolute_error(y_true[mask], y_pred[mask])) if mask.any() else None
        metrics[f"{prefix}_rmse"] = float(math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))) if mask.any() else None
    for threshold in TAIL_THRESHOLDS:
        true_mask = y_true >= threshold
        pred_mask = y_pred >= threshold
        tp = int(np.logical_and(true_mask, pred_mask).sum())
        true_count = int(true_mask.sum())
        pred_count = int(pred_mask.sum())
        dangerous = int(np.logical_and(true_mask, ~pred_mask).sum())
        key = f"thr_{threshold:.3f}"
        metrics[f"{key}_true_count"] = true_count
        metrics[f"{key}_pred_count"] = pred_count
        metrics[f"{key}_recall"] = tp / true_count if true_count else None
        metrics[f"{key}_precision"] = tp / pred_count if pred_count else None
        metrics[f"{key}_dangerous_under_count"] = dangerous
        metrics[f"{key}_dangerous_under_rate"] = dangerous / true_count if true_count else None
    return metrics


def prediction_rows(df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, model: str, seed: int, phase: str) -> list[dict[str, Any]]:
    columns_to_copy = [column for column in ("sample_id", "txt_path", "num_floors", "wave_cluster", "steel01_yielded", "steel02_yielded") if column in df.columns]
    rows: list[dict[str, Any]] = []
    for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
        row = {
            "model": model,
            "seed": seed,
            "phase": phase,
            "row_index": idx,
            "y_true": float(truth),
            "y_pred": float(pred),
            "error": float(pred - truth),
            "abs_error": float(abs(pred - truth)),
        }
        for column in columns_to_copy:
            row[column] = df.iloc[idx][column]
        rows.append(row)
    return rows


def flatten_metrics(model: str, seed: int, phase: str, metrics: dict[str, Any], elapsed: float) -> dict[str, Any]:
    row = {"model": model, "seed": seed, "phase": phase, "fit_seconds": elapsed}
    row.update(metrics)
    return row


def aggregate_test_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    test_rows = [row for row in rows if row["phase"] == "test"]
    if not test_rows:
        return []
    metric_names = [key for key in test_rows[0].keys() if key not in {"model", "seed", "phase"} and isinstance(test_rows[0].get(key), (int, float))]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in test_rows:
        grouped.setdefault(row["model"], []).append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in grouped.items():
        summary: dict[str, Any] = {"model": model, "seed_count": len(model_rows)}
        for metric in metric_names:
            values = [float(row[metric]) for row in model_rows if row.get(metric) is not None and math.isfinite(float(row[metric]))]
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_std"] = std
            summary[f"{metric}_ci95_normal"] = ci95
        output.append(summary)
    return output


def run_baselines(args: argparse.Namespace) -> dict[str, Any]:
    protocol = load_json(Path(args.protocol_json).resolve())
    output_root = Path(args.output_root).resolve() if args.output_root else Path(args.protocol_json).resolve().parents[1]
    final_dir = output_root / "final_multiseed"
    table_dir = output_root / "paper_tables"
    final_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    train_csv, val_csv, test_csv = protocol_paths(protocol)
    train_df = read_valid_csv(train_csv)
    val_df = read_valid_csv(val_csv)
    test_df = read_valid_csv(test_csv)
    numeric_cols, categorical_cols = infer_feature_columns(train_df)
    x_train = prepare_features(train_df, numeric_cols, categorical_cols)
    x_val = prepare_features(val_df, numeric_cols, categorical_cols)
    x_test = prepare_features(test_df, numeric_cols, categorical_cols)
    y_train = train_df[TARGET_COL].to_numpy(dtype=float)
    y_val = val_df[TARGET_COL].to_numpy(dtype=float)
    y_test = test_df[TARGET_COL].to_numpy(dtype=float)

    seeds = [int(seed) for seed in args.seeds]
    models = list(args.models)
    all_metric_rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "run_name": "todo5_p0_baseline_final_eval",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_json": str(Path(args.protocol_json).resolve()),
        "train_csv": str(train_csv),
        "val_csv": str(val_csv),
        "test_csv": str(test_csv),
        "models": models,
        "seeds": seeds,
        "target_column": TARGET_COL,
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "non_interference_policy": [
            "CPU-only sklearn baselines.",
            "No Optuna, no torch import, no GPU model training.",
            "Existing HPO/deep learning artifacts are not modified.",
        ],
        "runs": [],
    }

    for model_name in models:
        for seed in seeds:
            run_dir = final_dir / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            pipeline = build_pipeline(model_name, seed, numeric_cols, categorical_cols, int(args.n_jobs))
            start = time.perf_counter()
            pipeline.fit(x_train, y_train)
            fit_seconds = time.perf_counter() - start
            val_pred = np.asarray(pipeline.predict(x_val), dtype=float)
            test_pred = np.asarray(pipeline.predict(x_test), dtype=float)
            val_metrics = compute_metrics(y_val, val_pred)
            test_metrics = compute_metrics(y_test, test_pred)
            val_rows = prediction_rows(val_df, y_val, val_pred, model_name, seed, "val")
            test_rows = prediction_rows(test_df, y_test, test_pred, model_name, seed, "test")
            prediction_fields = list(val_rows[0].keys()) if val_rows else []
            write_csv(run_dir / "predictions_val.csv", val_rows, prediction_fields)
            write_csv(run_dir / "predictions_test.csv", test_rows, prediction_fields)
            metadata = {
                "model": model_name,
                "seed": seed,
                "fit_seconds": fit_seconds,
                "numeric_features": numeric_cols,
                "categorical_features": categorical_cols,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
            write_json(run_dir / "metrics.json", metadata)
            if args.save_models:
                dump(pipeline, run_dir / "model.joblib")
            all_metric_rows.append(flatten_metrics(model_name, seed, "val", val_metrics, fit_seconds))
            all_metric_rows.append(flatten_metrics(model_name, seed, "test", test_metrics, fit_seconds))
            manifest["runs"].append(
                {
                    "model": model_name,
                    "seed": seed,
                    "fit_seconds": fit_seconds,
                    "run_dir": str(run_dir),
                    "test_mae": test_metrics["mae"],
                    "test_rmse": test_metrics["rmse"],
                    "test_r2": test_metrics["r2"],
                }
            )
            print(
                f"{model_name} seed={seed}: test_mae={test_metrics['mae']:.8f}, "
                f"test_rmse={test_metrics['rmse']:.8f}, fit={fit_seconds:.1f}s"
            )

    summary_rows = aggregate_test_metrics(all_metric_rows)
    metric_fields = sorted({key for row in all_metric_rows for key in row.keys()})
    write_csv(final_dir / "p0_summary_metrics_by_seed.csv", all_metric_rows, metric_fields)
    summary_fields = sorted({key for row in summary_rows for key in row.keys()})
    write_csv(final_dir / "p0_summary_metrics_mean_std_ci.csv", summary_rows, summary_fields)
    write_csv(table_dir / "table2_p0_baseline_overall_test_metrics.csv", summary_rows, summary_fields)
    write_json(final_dir / "p0_final_eval_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CPU-only P0 baseline final evaluation.")
    parser.add_argument("--protocol-json", default=str(DEFAULT_PROTOCOL_JSON))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args()
    if args.seeds is None:
        protocol = load_json(Path(args.protocol_json).resolve())
        args.seeds = protocol.get("final_evaluation_protocol", {}).get("final_training_seeds", [20260614, 20260615, 20260616, 20260617, 20260618])
    return args


def main() -> None:
    args = parse_args()
    manifest = run_baselines(args)
    print("TODO5 P0 baseline final evaluation completed.")
    print(f"Runs: {len(manifest['runs'])}")


if __name__ == "__main__":
    main()
