# -*- coding: utf-8 -*-
"""Generate TODO6-TODO8 publication tables from final multi-seed predictions.

Inputs are the per-seed artifacts written under
``publication_eval_20260614/final_multiseed/<model>/seed_<seed>/``. The script
does not train models and does not use validation results for model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
FINAL_DIR = OUTPUT_ROOT / "final_multiseed"
TABLE_DIR = OUTPUT_ROOT / "paper_tables"
STAT_DIR = OUTPUT_ROOT / "statistics"
FIGURE_DIR = OUTPUT_ROOT / "paper_figures"
REFERENCE_MODEL_CANDIDATES = ("lightgbm", "extratrees", "randomforest", "xgboost", "catboost")
THRESHOLDS = (0.005, 0.010, 0.015, 0.020)


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


def normalize_metric_key(key: str) -> str:
    mapping = {
        "Samples": "samples",
        "R2": "r2",
        "MAE": "mae",
        "RMSE": "rmse",
        "MAPE": "mape",
        "Bias": "bias",
        "SelectionScore": "selection_score",
    }
    return mapping.get(key, str(key).lower())


def normalize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in metrics.items():
        norm_key = normalize_metric_key(str(key))
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            normalized[norm_key] = float(value)
    return normalized


def row_metric(model: str, seed: int, phase: str, metrics: dict[str, Any], fit_seconds: float | None) -> dict[str, Any]:
    row = {"model": model, "seed": int(seed), "phase": phase}
    if fit_seconds is not None:
        row["fit_seconds"] = float(fit_seconds)
    row.update(normalize_metrics(metrics))
    return row


def discover_runs(final_dir: Path, include_models: set[str] | None = None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        if include_models and model not in include_models:
            continue
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            metrics_path = seed_dir / "metrics.json"
            predictions_path = seed_dir / "predictions_test.csv"
            if not metrics_path.exists() or not predictions_path.exists():
                continue
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            metrics = load_json(metrics_path)
            runs.append(
                {
                    "model": model,
                    "seed": seed,
                    "run_dir": seed_dir,
                    "metrics_path": metrics_path,
                    "predictions_path": predictions_path,
                    "metrics": metrics,
                }
            )
    return runs


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, std, ci95


def aggregate(rows: list[dict[str, Any]], phase: str = "test") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("phase") == phase:
            grouped.setdefault(str(row["model"]), []).append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        keys = sorted(
            key
            for key in {field for row in model_rows for field in row}
            if key not in {"model", "seed", "phase"} and isinstance(model_rows[0].get(key), (int, float))
        )
        out: dict[str, Any] = {"model": model, "seed_count": len(model_rows)}
        for key in keys:
            values = [float(row[key]) for row in model_rows if isinstance(row.get(key), (int, float))]
            mean, std, ci95 = mean_std_ci(values)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_ci95_normal"] = ci95
        output.append(out)
    output.sort(key=lambda row: row.get("mae_mean", float("inf")))
    return output


def aggregate_by_keys(rows: list[dict[str, Any]], keys: tuple[str, ...], sort_metric: str | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        group_key = tuple(str(row.get(key, "")) for key in keys)
        grouped.setdefault(group_key, []).append(row)
    output: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items()):
        out: dict[str, Any] = {key: value for key, value in zip(keys, group_key)}
        out["seed_count"] = len({row.get("seed") for row in group_rows})
        numeric_keys = sorted(
            key
            for key in {field for row in group_rows for field in row}
            if key not in set(keys) | {"seed"} and isinstance(group_rows[0].get(key), (int, float))
        )
        for key in numeric_keys:
            values = [float(row[key]) for row in group_rows if isinstance(row.get(key), (int, float))]
            mean, std, ci95 = mean_std_ci(values)
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_ci95_normal"] = ci95
        output.append(out)
    if sort_metric is not None:
        output.sort(key=lambda row: row.get(sort_metric, float("inf")))
    return output


def read_predictions(path: Path, model: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"y_true", "y_pred"}.issubset(df.columns):
        y_true_col, y_pred_col = "y_true", "y_pred"
    elif {"True_Drift", "Pred_Drift"}.issubset(df.columns):
        y_true_col, y_pred_col = "True_Drift", "Pred_Drift"
    else:
        raise ValueError(f"Cannot find prediction columns in {path}")
    out = pd.DataFrame(
        {
            "model": model,
            "seed": int(seed),
            "sample_id": df["sample_id"].astype(str) if "sample_id" in df.columns else np.arange(len(df)).astype(str),
            "txt_path": df["txt_path"].astype(str) if "txt_path" in df.columns else "",
            "num_floors": pd.to_numeric(df.get("num_floors", np.nan), errors="coerce"),
            "wave_cluster": pd.to_numeric(df.get("wave_cluster", np.nan), errors="coerce"),
            "steel01_yielded": pd.to_numeric(df.get("steel01_yielded", np.nan), errors="coerce"),
            "steel02_yielded": pd.to_numeric(df.get("steel02_yielded", np.nan), errors="coerce"),
            "y_true": pd.to_numeric(df[y_true_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[y_pred_col], errors="coerce"),
        }
    )
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    return out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)


def compute_tail_metrics(pred: pd.DataFrame, model: str, seed: int) -> dict[str, Any]:
    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true
    under = np.maximum(y_true - y_pred, 0.0)
    abs_error = np.abs(error)
    row: dict[str, Any] = {
        "model": model,
        "seed": int(seed),
        "samples": int(len(pred)),
        "p95_underprediction": float(np.quantile(under, 0.95)),
        "max_underprediction": float(np.max(under)),
    }
    for q in (0.90, 0.95):
        threshold = float(np.quantile(y_true, q))
        mask = y_true >= threshold
        label = f"p{int(q * 100)}"
        row[f"{label}_threshold"] = threshold
        row[f"{label}_count"] = int(mask.sum())
        row[f"{label}_mae"] = float(abs_error[mask].mean()) if mask.any() else None
        row[f"{label}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None
    for threshold in THRESHOLDS:
        true_mask = y_true >= threshold
        pred_mask = y_pred >= threshold
        dangerous = true_mask & (y_pred < threshold)
        label = f"thr_{threshold:.3f}"
        row[f"{label}_true_count"] = int(true_mask.sum())
        row[f"{label}_pred_count"] = int(pred_mask.sum())
        row[f"{label}_recall"] = float((true_mask & pred_mask).sum() / true_mask.sum()) if true_mask.any() else None
        row[f"{label}_precision"] = float((true_mask & pred_mask).sum() / pred_mask.sum()) if pred_mask.any() else None
        row[f"{label}_dangerous_under_count"] = int(dangerous.sum())
        row[f"{label}_dangerous_under_rate"] = float(dangerous.sum() / true_mask.sum()) if true_mask.any() else None
    return row


def group_metrics(pred: pd.DataFrame, group_col: str, model: str, seed: int) -> list[dict[str, Any]]:
    if group_col not in pred.columns:
        return []
    rows: list[dict[str, Any]] = []
    for group_value, group_df in pred.dropna(subset=[group_col]).groupby(group_col):
        errors = group_df["y_pred"].to_numpy(dtype=float) - group_df["y_true"].to_numpy(dtype=float)
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "group_column": group_col,
                "group_value": str(group_value),
                "samples": int(len(group_df)),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "bias": float(np.mean(errors)),
            }
        )
    return rows


def aligned_seed_frames(predictions: dict[tuple[str, int], pd.DataFrame], reference_model: str, model: str, seed: int) -> tuple[np.ndarray, np.ndarray] | None:
    ref = predictions.get((reference_model, seed))
    other = predictions.get((model, seed))
    if ref is None or other is None:
        return None
    ref_df = ref[["sample_id", "abs_error"]].rename(columns={"abs_error": "ref_abs"})
    other_df = other[["sample_id", "abs_error"]].rename(columns={"abs_error": "model_abs"})
    joined = ref_df.merge(other_df, on="sample_id", how="inner")
    if joined.empty:
        return None
    return joined["ref_abs"].to_numpy(dtype=float), joined["model_abs"].to_numpy(dtype=float)


def paired_tests(
    predictions: dict[tuple[str, int], pd.DataFrame],
    models: list[str],
    seeds: list[int],
    reference_model: str,
    bootstrap_iters: int,
    rng_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(rng_seed)
    bootstrap_rows: list[dict[str, Any]] = []
    wilcoxon_rows: list[dict[str, Any]] = []
    for model in models:
        if model == reference_model:
            continue
        for seed in seeds:
            aligned = aligned_seed_frames(predictions, reference_model, model, seed)
            if aligned is None:
                continue
            ref_abs, model_abs = aligned
            delta = model_abs - ref_abs
            n = len(delta)
            boot = np.empty(bootstrap_iters, dtype=float)
            for idx in range(bootstrap_iters):
                sample_idx = rng.integers(0, n, size=n)
                boot[idx] = float(np.mean(delta[sample_idx]))
            ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            bootstrap_rows.append(
                {
                    "reference_model": reference_model,
                    "model": model,
                    "seed": int(seed),
                    "n_pairs": int(n),
                    "delta_mae_model_minus_reference": float(np.mean(delta)),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "model_better_than_reference": bool(np.mean(delta) < 0),
                }
            )
            try:
                stat, p_value = wilcoxon(model_abs, ref_abs, zero_method="wilcox", alternative="two-sided")
            except ValueError:
                stat, p_value = float("nan"), float("nan")
            wilcoxon_rows.append(
                {
                    "reference_model": reference_model,
                    "model": model,
                    "seed": int(seed),
                    "n_pairs": int(n),
                    "wilcoxon_statistic": float(stat),
                    "p_value_two_sided": float(p_value),
                    "median_delta_abs_error": float(np.median(delta)),
                }
            )
    return bootstrap_rows, wilcoxon_rows


def pick_reference_model(aggregate_rows: list[dict[str, Any]]) -> str:
    by_model = {str(row["model"]): row for row in aggregate_rows}
    for candidate in REFERENCE_MODEL_CANDIDATES:
        if candidate in by_model:
            return candidate
    if not aggregate_rows:
        raise ValueError("No aggregate rows available.")
    return str(aggregate_rows[0]["model"])


def plot_overall_bar(aggregate_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [str(row["model"]) for row in aggregate_rows]
    means = [float(row.get("mae_mean", np.nan)) for row in aggregate_rows]
    ci = [float(row.get("mae_ci95_normal", 0.0)) for row in aggregate_rows]
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4.8))
    ax.bar(labels, means, yerr=ci, capsize=4, color="#4477aa", edgecolor="#222222", linewidth=0.6)
    ax.set_ylabel("Test MAE")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TODO6-TODO8 paper tables and paired tests.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--models", nargs="*", default=None, help="Optional model-name subset.")
    parser.add_argument("--bootstrap-iters", type=int, default=2000)
    parser.add_argument("--rng-seed", type=int, default=20260614)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    final_dir = output_root / "final_multiseed"
    table_dir = output_root / "paper_tables"
    stat_dir = output_root / "statistics"
    figure_dir = output_root / "paper_figures"
    include_models = set(args.models) if args.models else None
    runs = discover_runs(final_dir, include_models)
    if not runs:
        raise FileNotFoundError(f"No final multi-seed runs found under {final_dir}")

    metric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for run in runs:
        model = str(run["model"])
        seed = int(run["seed"])
        metrics = run["metrics"]
        fit_seconds = metrics.get("fit_seconds")
        if "val_metrics" in metrics:
            metric_rows.append(row_metric(model, seed, "val", metrics["val_metrics"], fit_seconds))
        if "test_metrics" in metrics:
            metric_rows.append(row_metric(model, seed, "test", metrics["test_metrics"], fit_seconds))
        pred = read_predictions(Path(run["predictions_path"]), model, seed)
        predictions[(model, seed)] = pred
        tail_rows.append(compute_tail_metrics(pred, model, seed))
        for group_col in ("num_floors", "wave_cluster", "steel01_yielded", "steel02_yielded"):
            group_rows.extend(group_metrics(pred, group_col, model, seed))

    aggregate_rows = aggregate(metric_rows, phase="test")
    tail_aggregate_rows = aggregate_by_keys(tail_rows, ("model",), sort_metric="p95_mae_mean")
    group_aggregate_rows = aggregate_by_keys(group_rows, ("model", "group_column", "group_value"), sort_metric="mae_mean")
    models = sorted({model for model, _seed in predictions})
    seeds = sorted({seed for _model, seed in predictions})
    reference_model = pick_reference_model(aggregate_rows)
    bootstrap_rows, wilcoxon_rows = paired_tests(
        predictions,
        models=models,
        seeds=seeds,
        reference_model=reference_model,
        bootstrap_iters=int(args.bootstrap_iters),
        rng_seed=int(args.rng_seed),
    )

    write_csv(table_dir / "tableS_seed_metrics.csv", sorted(metric_rows, key=lambda row: (row["model"], row["seed"], row["phase"])))
    write_csv(table_dir / "table2_overall_test_metrics.csv", aggregate_rows)
    write_csv(final_dir / "summary_metrics_by_seed.csv", sorted(metric_rows, key=lambda row: (row["model"], row["seed"], row["phase"])))
    write_csv(final_dir / "summary_metrics_mean_std_ci.csv", aggregate_rows)
    write_csv(table_dir / "table3_tail_safety_metrics.csv", tail_aggregate_rows)
    write_csv(table_dir / "table3_tail_safety_metrics_by_seed.csv", sorted(tail_rows, key=lambda row: (row["model"], row["seed"])))
    write_csv(table_dir / "tableS_group_metrics.csv", group_aggregate_rows)
    write_csv(table_dir / "tableS_group_metrics_by_seed.csv", sorted(group_rows, key=lambda row: (row["model"], row["seed"], row["group_column"], row["group_value"])))
    write_csv(stat_dir / "paired_bootstrap_delta.csv", bootstrap_rows)
    write_csv(stat_dir / "wilcoxon_tests.csv", wilcoxon_rows)
    plot_overall_bar(aggregate_rows, figure_dir / "model_performance_bar_ci.png")

    manifest = {
        "run_name": "todo6_8_tables_and_statistics",
        "source_final_multiseed_dir": str(final_dir),
        "models": models,
        "seeds": seeds,
        "run_count": len(runs),
        "reference_model_for_paired_tests": reference_model,
        "bootstrap_iters": int(args.bootstrap_iters),
        "outputs": {
            "final_summary_metrics_by_seed": str(final_dir / "summary_metrics_by_seed.csv"),
            "final_summary_metrics_mean_std_ci": str(final_dir / "summary_metrics_mean_std_ci.csv"),
            "table2_overall_test_metrics": str(table_dir / "table2_overall_test_metrics.csv"),
            "tableS_seed_metrics": str(table_dir / "tableS_seed_metrics.csv"),
            "table3_tail_safety_metrics": str(table_dir / "table3_tail_safety_metrics.csv"),
            "table3_tail_safety_metrics_by_seed": str(table_dir / "table3_tail_safety_metrics_by_seed.csv"),
            "tableS_group_metrics": str(table_dir / "tableS_group_metrics.csv"),
            "tableS_group_metrics_by_seed": str(table_dir / "tableS_group_metrics_by_seed.csv"),
            "paired_bootstrap_delta": str(stat_dir / "paired_bootstrap_delta.csv"),
            "wilcoxon_tests": str(stat_dir / "wilcoxon_tests.csv"),
            "model_performance_bar_ci": str(figure_dir / "model_performance_bar_ci.png"),
        },
        "notes": [
            "Prediction files are aligned by sample_id for paired tests.",
            "Paired tests use per-sample errors on the locked test split; TODO2 found repeated scenarios, so paper text should state this and optionally report scenario-aggregated sensitivity.",
            "The locked test split has no >=0.010 drift samples and no steel01_yielded=1 samples; high-drift safety claims still require TODO10.",
        ],
    }
    save_json(output_root / "statistics" / "todo6_8_manifest.json", manifest)
    save_json(final_dir / "final_multiseed_manifest.json", manifest)
    print(f">>> Aggregated {len(runs)} runs from {len(models)} models.")
    print(f">>> Reference model for paired tests: {reference_model}")
    print(f">>> Wrote {table_dir / 'table2_overall_test_metrics.csv'}")
    print(f">>> Wrote {stat_dir / 'paired_bootstrap_delta.csv'}")


if __name__ == "__main__":
    main()
