# -*- coding: utf-8 -*-
"""
Checkpoint/model ensemble evaluator for the 2D-CNN surrogate model.

Default blend:
    pred = 0.60 * pred_v2_best + 0.40 * pred_v5_mae

The default paths are intentionally explicit so the experiment is reproducible.
Tune the blend weight on validation data before using it as a final paper result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_V2_RESULTS = PROJECT_ROOT / (
    "output/model-20260523-171351-541b9ed7-train20260529-182601/"
    "test-20260523-171351-541b9ed7_results.csv"
)
DEFAULT_V5_RESULTS = PROJECT_ROOT / (
    "output/2dcnnv5/model-20260523-171351-541b9ed7-train20260531-143031/"
    "test-20260523-171351-541b9ed7_results__mae.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output/2dcnnv6/ensemble-v2best-v5mae"


def read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No rows found in {path}")
    for required in ("True_Drift", "Pred_Drift"):
        if required not in fieldnames:
            raise KeyError(f"{path} is missing required column: {required}")
    return rows, fieldnames


def calculate_metrics(true_values: list[float], pred_values: list[float]) -> dict[str, float]:
    n = len(true_values)
    if n == 0:
        return {"Samples": 0.0, "R2": 0.0, "MAE": math.nan, "RMSE": math.nan, "MAPE": math.nan, "Bias": math.nan}
    errors = [pred - true for true, pred in zip(true_values, pred_values)]
    abs_errors = [abs(error) for error in errors]
    mean_true = sum(true_values) / n
    ss_res = sum(error * error for error in errors)
    ss_tot = sum((true - mean_true) ** 2 for true in true_values)
    return {
        "Samples": float(n),
        "R2": 0.0 if ss_tot <= 0.0 else 1.0 - ss_res / ss_tot,
        "MAE": sum(abs_errors) / n,
        "RMSE": math.sqrt(ss_res / n),
        "MAPE": sum(abs(error) / max(abs(true), 1e-12) for true, error in zip(true_values, errors)) / n * 100.0,
        "Bias": sum(errors) / n,
    }


def calculate_threshold_metrics(true_values: list[float], pred_values: list[float], threshold: float) -> dict[str, float]:
    tp = fp = fn = tn = 0
    for true, pred in zip(true_values, pred_values):
        true_positive = true >= threshold
        pred_positive = pred >= threshold
        if true_positive and pred_positive:
            tp += 1
        elif (not true_positive) and pred_positive:
            fp += 1
        elif true_positive and (not pred_positive):
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else math.nan
    recall = tp / (tp + fn) if tp + fn else math.nan
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else math.nan
    return {
        "threshold": threshold,
        "positive_count": float(tp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def subset_by_range(
    true_values: list[float],
    pred_values: list[float],
    lower: float,
    upper: float,
) -> tuple[list[float], list[float]]:
    selected_true: list[float] = []
    selected_pred: list[float] = []
    for true, pred in zip(true_values, pred_values):
        if lower <= true < upper:
            selected_true.append(true)
            selected_pred.append(pred)
    return selected_true, selected_pred


def build_metric_record(name: str, true_values: list[float], pred_values: list[float]) -> dict[str, float | str]:
    record: dict[str, float | str] = {"model": name}
    record.update(calculate_metrics(true_values, pred_values))
    for label, lower, upper in (
        ("low_lt_0p001", 0.0, 0.001),
        ("core_0p001_0p005", 0.001, 0.005),
        ("mid_0p005_0p010", 0.005, 0.010),
        ("tail_ge_0p010", 0.010, float("inf")),
    ):
        part_true, part_pred = subset_by_range(true_values, pred_values, lower, upper)
        part_metrics = calculate_metrics(part_true, part_pred)
        record[f"{label}_n"] = part_metrics["Samples"]
        record[f"{label}_mae"] = part_metrics["MAE"]
        record[f"{label}_rmse"] = part_metrics["RMSE"]
        record[f"{label}_mape"] = part_metrics["MAPE"]
        record[f"{label}_bias"] = part_metrics["Bias"]
    for threshold in (0.005, 0.010, 0.015, 0.020):
        threshold_metrics = calculate_threshold_metrics(true_values, pred_values, threshold)
        suffix = str(threshold).replace(".", "p")
        record[f"f1_ge_{suffix}"] = threshold_metrics["f1"]
        record[f"precision_ge_{suffix}"] = threshold_metrics["precision"]
        record[f"recall_ge_{suffix}"] = threshold_metrics["recall"]
    return record


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def align_predictions(
    v2_rows: list[dict[str, str]],
    v5_rows: list[dict[str, str]],
) -> tuple[list[float], list[float], list[float]]:
    if len(v2_rows) != len(v5_rows):
        raise ValueError(f"Row counts differ: v2={len(v2_rows)}, v5={len(v5_rows)}")
    true_values: list[float] = []
    pred_v2: list[float] = []
    pred_v5: list[float] = []
    for index, (row_v2, row_v5) in enumerate(zip(v2_rows, v5_rows), start=1):
        true_v2 = float(row_v2["True_Drift"])
        true_v5 = float(row_v5["True_Drift"])
        if abs(true_v2 - true_v5) > 1e-10:
            raise ValueError(f"True_Drift mismatch at row {index}: {true_v2} vs {true_v5}")
        true_values.append(true_v2)
        pred_v2.append(float(row_v2["Pred_Drift"]))
        pred_v5.append(float(row_v5["Pred_Drift"]))
    return true_values, pred_v2, pred_v5


def blend_predictions(pred_v2: list[float], pred_v5: list[float], weight_v2: float) -> list[float]:
    return [weight_v2 * value_v2 + (1.0 - weight_v2) * value_v5 for value_v2, value_v5 in zip(pred_v2, pred_v5)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-results", type=Path, default=DEFAULT_V2_RESULTS)
    parser.add_argument("--v5-results", type=Path, default=DEFAULT_V5_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--weight-v2", type=float, default=0.60)
    args = parser.parse_args()

    v2_rows, v2_fieldnames = read_rows(args.v2_results)
    v5_rows, _ = read_rows(args.v5_results)
    true_values, pred_v2, pred_v5 = align_predictions(v2_rows, v5_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, float | str]] = [
        build_metric_record("v2_best", true_values, pred_v2),
        build_metric_record("v5_mae", true_values, pred_v5),
    ]
    for step in range(0, 11):
        weight_v2 = step / 10.0
        blended = blend_predictions(pred_v2, pred_v5, weight_v2)
        metric_rows.append(build_metric_record(f"blend_wv2_{weight_v2:.1f}", true_values, blended))

    selected_weight = min(max(args.weight_v2, 0.0), 1.0)
    selected_pred = blend_predictions(pred_v2, pred_v5, selected_weight)
    selected_name = f"blend_wv2_{selected_weight:.2f}".replace(".", "p")
    selected_metrics = build_metric_record(selected_name, true_values, selected_pred)

    metric_fieldnames = list(metric_rows[0].keys())
    write_csv(args.output_dir / "ensemble_grid_metrics.csv", metric_rows, metric_fieldnames)

    result_rows: list[dict[str, object]] = []
    for row, true, value_v2, value_v5, pred in zip(v2_rows, true_values, pred_v2, pred_v5, selected_pred):
        result = dict(row)
        result["Pred_Drift_v2_best"] = value_v2
        result["Pred_Drift_v5_mae"] = value_v5
        result["Pred_Drift"] = pred
        result["Abs_Error"] = abs(pred - true)
        result["Error_Pct"] = abs(pred - true) / max(abs(true), 1e-12) * 100.0
        result_rows.append(result)

    result_fieldnames = list(dict.fromkeys(v2_fieldnames + ["Pred_Drift_v2_best", "Pred_Drift_v5_mae"]))
    if "Pred_Drift" not in result_fieldnames:
        result_fieldnames.append("Pred_Drift")
    if "Abs_Error" not in result_fieldnames:
        result_fieldnames.append("Abs_Error")
    if "Error_Pct" not in result_fieldnames:
        result_fieldnames.append("Error_Pct")
    write_csv(args.output_dir / f"{selected_name}_results.csv", result_rows, result_fieldnames)

    metrics_path = args.output_dir / f"{selected_name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "weight_v2": selected_weight,
                "weight_v5": 1.0 - selected_weight,
                "v2_results": str(args.v2_results),
                "v5_results": str(args.v5_results),
                "metrics": selected_metrics,
                "note": "Use validation-tuned weights for final reporting; test-grid values are exploratory.",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Grid metrics saved to: {args.output_dir / 'ensemble_grid_metrics.csv'}")
    print(f"Selected ensemble results saved to: {args.output_dir / f'{selected_name}_results.csv'}")
    print(f"Selected ensemble metrics saved to: {metrics_path}")
    print(json.dumps(selected_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
