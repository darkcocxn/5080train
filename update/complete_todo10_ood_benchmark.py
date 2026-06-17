# -*- coding: utf-8 -*-
"""Complete TODO10 unseen-wave / unseen-structure / high-drift benchmark.

This script is deliberately evaluation-only. It consumes the frozen final
prediction files generated after TODO5 and the locked protocol CSV files. It
does not train models or alter data splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
PROTOCOL_PATH = DEFAULT_OUTPUT_ROOT / "protocol" / "protocol_lock.json"
TARGET = "max_drift_ratio_raw"
STRUCTURE_COLS = ("num_floors", "floor_mass", "floor_height", "k_base_1_4", "Fy_add", "damper_layout")
THRESHOLDS = (0.005, 0.010, 0.015, 0.020)
TOP_MODELS_FOR_FIGURE = ("2dcnn", "wavenet", "lstm", "lightgbm", "randomforest", "xgboost")


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


def mean_std_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return None, None, None
    mean = statistics.fmean(clean)
    if len(clean) == 1:
        return mean, 0.0, 0.0
    std = statistics.stdev(clean)
    return mean, std, 1.96 * std / math.sqrt(len(clean))


def fmt(value: Any, digits: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}g}"


def markdown_table(df: pd.DataFrame) -> str:
    headers = [str(col) for col in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |" for row in rows]
    return "\n".join([header_line, separator, *body])


def structure_signature(df: pd.DataFrame) -> pd.Series:
    return df.loc[:, STRUCTURE_COLS].astype(str).agg("|".join, axis=1)


def wave_id_from_txt_path(value: Any) -> int | None:
    match = re.search(r"\|(\d+)$", str(value))
    if not match:
        return None
    return int(match.group(1))


def load_protocol(output_root: Path) -> dict[str, Any]:
    path = output_root / "protocol" / "protocol_lock.json"
    if not path.exists():
        path = PROTOCOL_PATH
    return load_json(path)


def load_split_frames(protocol: dict[str, Any]) -> dict[str, pd.DataFrame]:
    dataset = protocol["dataset"]
    paths = {
        "train": Path(dataset["train_csv"]["path"]),
        "val": Path(dataset["val_csv"]["path"]),
        "test": Path(dataset["test_csv_reserved_locked"]["path"]),
    }
    frames: dict[str, pd.DataFrame] = {}
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
    for split, path in paths.items():
        df = pd.read_csv(path, usecols=usecols)
        df["row_id"] = np.arange(len(df), dtype=int)
        df["structure_signature"] = structure_signature(df)
        df["wave_id"] = df["txt_path"].map(wave_id_from_txt_path)
        frames[split] = df
    return frames


def read_prediction(path: Path, source_df: pd.DataFrame, model: str, seed: int, phase: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "y_pred" in df.columns:
        pred_col = "y_pred"
    elif "Pred_Drift" in df.columns:
        pred_col = "Pred_Drift"
    else:
        raise ValueError(f"Cannot find prediction columns in {path}")
    if len(df) != len(source_df):
        raise ValueError(f"Prediction row count does not match source CSV for {path}: {len(df)} vs {len(source_df)}")

    if "sample_id" in df.columns and "sample_id" in source_df.columns:
        pred_keys = df["sample_id"].astype(str).to_numpy()
        source_keys = source_df["sample_id"].astype(str).to_numpy()
        if not np.array_equal(pred_keys, source_keys):
            raise ValueError(f"Prediction sample_id order does not align with source CSV: {path}")
    elif "txt_path" in df.columns and "txt_path" in source_df.columns:
        pred_keys = df["txt_path"].astype(str).to_numpy()
        source_keys = source_df["txt_path"].astype(str).to_numpy()
        if not np.array_equal(pred_keys, source_keys):
            raise ValueError(f"Prediction txt_path order does not align with source CSV: {path}")

    source_true = source_df[TARGET].to_numpy(dtype=float)
    for true_col in ("y_true", "True_Drift", TARGET):
        if true_col in df.columns:
            y_true = pd.to_numeric(df[true_col], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(y_true, source_true, rtol=0.0, atol=5e-8):
                raise ValueError(f"Prediction true values do not align with source CSV: {path}")
            break

    out = source_df.copy()
    out["model"] = model
    out["seed"] = int(seed)
    out["phase"] = phase
    out["y_true"] = source_true
    out["y_pred"] = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=float)
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["underprediction"] = np.maximum(out["y_true"] - out["y_pred"], 0.0)
    return out


def discover_predictions(output_root: Path, frames: dict[str, pd.DataFrame]) -> dict[tuple[str, int, str], pd.DataFrame]:
    final_dir = output_root / "final_multiseed"
    predictions: dict[tuple[str, int, str], pd.DataFrame] = {}
    for model_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            for phase, filename in (("val", "predictions_val.csv"), ("test", "predictions_test.csv")):
                path = seed_dir / filename
                if not path.exists():
                    continue
                predictions[(model, seed, phase)] = read_prediction(path, frames[phase], model, seed, phase)
    if not predictions:
        raise FileNotFoundError(f"No final prediction files found under {final_dir}")
    return predictions


def compute_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "samples": 0,
            "mae": None,
            "rmse": None,
            "r2": None,
            "bias": None,
            "p95_abs_error": None,
            "max_underprediction": None,
            "p95_underprediction": None,
        }
    y = df["y_true"].to_numpy(dtype=float)
    pred = df["y_pred"].to_numpy(dtype=float)
    err = pred - y
    abs_err = np.abs(err)
    sse = float(np.sum(err**2))
    sst = float(np.sum((y - y.mean()) ** 2))
    out: dict[str, Any] = {
        "samples": int(len(df)),
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - sse / sst) if sst > 0 else None,
        "bias": float(err.mean()),
        "p95_abs_error": float(np.quantile(abs_err, 0.95)),
        "max_underprediction": float(np.max(np.maximum(y - pred, 0.0))),
        "p95_underprediction": float(np.quantile(np.maximum(y - pred, 0.0), 0.95)),
    }
    for threshold in THRESHOLDS:
        true_mask = y >= threshold
        pred_mask = pred >= threshold
        dangerous = true_mask & (pred < threshold)
        prefix = f"thr_{threshold:.3f}"
        out[f"{prefix}_true_count"] = int(true_mask.sum())
        out[f"{prefix}_pred_count"] = int(pred_mask.sum())
        out[f"{prefix}_dangerous_under_count"] = int(dangerous.sum())
        out[f"{prefix}_dangerous_under_rate"] = float(dangerous.sum() / true_mask.sum()) if true_mask.any() else None
        out[f"{prefix}_recall"] = float((true_mask & pred_mask).sum() / true_mask.sum()) if true_mask.any() else None
        out[f"{prefix}_precision"] = float((true_mask & pred_mask).sum() / pred_mask.sum()) if pred_mask.any() else None
    return out


def aggregate_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pd.DataFrame(rows)
    grouped_cols = ["benchmark_name", "model"]
    passthrough_cols = [
        "phase",
        "benchmark_status",
        "claim_scope",
        "selection_contaminated",
        "independent_of_hpo",
        "subset_definition",
        "relative_to",
    ]
    output: list[dict[str, Any]] = []
    for keys, group in df.groupby(grouped_cols, sort=True, dropna=False):
        row: dict[str, Any] = {"benchmark_name": keys[0], "model": keys[1], "seed_count": int(group["seed"].nunique())}
        for col in passthrough_cols:
            if col in group.columns:
                values = [str(value) for value in group[col].dropna().unique().tolist()]
                row[col] = values[0] if len(values) == 1 else ";".join(values)
        for col in df.columns:
            if col in set(grouped_cols + ["seed"] + passthrough_cols):
                continue
            if not pd.api.types.is_numeric_dtype(group[col]):
                continue
            values = [float(value) for value in group[col].dropna().tolist()]
            mean, std, ci95 = mean_std_ci(values)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95_normal"] = ci95
        output.append(row)
    output.sort(key=lambda item: (str(item["benchmark_name"]), item.get("mae_mean") if item.get("mae_mean") is not None else float("inf")))
    return output


def benchmark_rows(
    predictions: dict[tuple[str, int, str], pd.DataFrame],
    frames: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trainval_waves = set(frames["train"]["txt_path"]) | set(frames["val"]["txt_path"])
    trainval_struct = set(frames["train"]["structure_signature"]) | set(frames["val"]["structure_signature"])
    train_waves = set(frames["train"]["txt_path"])
    train_struct = set(frames["train"]["structure_signature"])
    by_seed: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []

    benchmark_specs = [
        {
            "benchmark_name": "validation_reference_unseen_wave_structure",
            "phase": "val",
            "mask_func": lambda df: (~df["txt_path"].isin(train_waves)) & (~df["structure_signature"].isin(train_struct)),
            "benchmark_status": "diagnostic_reference_not_final_test",
            "claim_scope": "diagnostic_only_validation_used_for_hpo",
            "selection_contaminated": True,
            "independent_of_hpo": False,
            "relative_to": "train",
            "subset_definition": "validation rows with exact txt_path and exact structure_signature unseen in train",
        },
        {
            "benchmark_name": "locked_test_unseen_wave",
            "phase": "test",
            "mask_func": lambda df: ~df["txt_path"].isin(trainval_waves),
            "benchmark_status": "complete_independent_locked_test",
            "claim_scope": "exact_wave_generalization_with_seen_wave_cluster_ranges",
            "selection_contaminated": False,
            "independent_of_hpo": True,
            "relative_to": "train+val",
            "subset_definition": "locked test rows whose exact txt_path is absent from train and validation",
        },
        {
            "benchmark_name": "locked_test_unseen_structure",
            "phase": "test",
            "mask_func": lambda df: ~df["structure_signature"].isin(trainval_struct),
            "benchmark_status": "complete_independent_locked_test",
            "claim_scope": "exact_structure_signature_generalization_not_range_extrapolation",
            "selection_contaminated": False,
            "independent_of_hpo": True,
            "relative_to": "train+val",
            "subset_definition": "locked test rows whose exact structure signature is absent from train and validation",
        },
        {
            "benchmark_name": "locked_test_high_drift_ge_0.010",
            "phase": "test",
            "mask_func": lambda df: df["y_true"] >= 0.010,
            "benchmark_status": "absent_in_locked_test_requires_new_opensees",
            "claim_scope": "no_independent_high_drift_claim",
            "selection_contaminated": False,
            "independent_of_hpo": True,
            "relative_to": "absolute_threshold",
            "subset_definition": "locked test rows with target >= 0.010",
        },
        {
            "benchmark_name": "validation_high_drift_ge_0.010_diagnostic",
            "phase": "val",
            "mask_func": lambda df: df["y_true"] >= 0.010,
            "benchmark_status": "diagnostic_only_not_independent",
            "claim_scope": "diagnostic_high_drift_only_validation_used_for_hpo",
            "selection_contaminated": True,
            "independent_of_hpo": False,
            "relative_to": "absolute_threshold",
            "subset_definition": "validation rows with target >= 0.010; not an independent stress test",
        },
    ]

    for spec in benchmark_specs:
        sample_seen = False
        for (model, seed, phase), pred in predictions.items():
            if phase != spec["phase"]:
                continue
            mask = spec["mask_func"](pred)
            subset = pred.loc[mask].copy()
            if not sample_seen:
                for _, row in subset.head(5000).iterrows():
                    sample_rows.append(
                        {
                            "benchmark_name": spec["benchmark_name"],
                            "phase": spec["phase"],
                            "sample_id": row["sample_id"],
                            "row_id": int(row["row_id"]),
                            "txt_path": row["txt_path"],
                            "wave_id": row["wave_id"],
                            "structure_signature": row["structure_signature"],
                            "y_true": row["y_true"],
                            "num_floors": row["num_floors"],
                            "floor_mass": row["floor_mass"],
                            "floor_height": row["floor_height"],
                            "k_base_1_4": row["k_base_1_4"],
                            "Fy_add": row["Fy_add"],
                            "damper_layout": row["damper_layout"],
                            "wave_cluster": row["wave_cluster"],
                            "steel01_yielded": row["steel01_yielded"],
                            "steel02_yielded": row["steel02_yielded"],
                        }
                    )
                sample_seen = True
            metric_row = {
                "benchmark_name": spec["benchmark_name"],
                "model": model,
                "seed": seed,
                "phase": spec["phase"],
                "benchmark_status": spec["benchmark_status"],
                "claim_scope": spec["claim_scope"],
                "selection_contaminated": spec["selection_contaminated"],
                "independent_of_hpo": spec["independent_of_hpo"],
                "relative_to": spec["relative_to"],
                "subset_definition": spec["subset_definition"],
                **compute_metrics(subset),
            }
            by_seed.append(metric_row)
    return by_seed, aggregate_by_model(by_seed), sample_rows


def split_coverage_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    train_waves = set(frames["train"]["txt_path"])
    trainval_waves = train_waves | set(frames["val"]["txt_path"])
    train_struct = set(frames["train"]["structure_signature"])
    trainval_struct = train_struct | set(frames["val"]["structure_signature"])
    rows: list[dict[str, Any]] = []
    for split, df in frames.items():
        rows.append(
            {
                "split": split,
                "rows": int(len(df)),
                "unique_waves": int(df["txt_path"].nunique()),
                "unique_wave_clusters": int(df["wave_cluster"].nunique()),
                "unique_structure_signatures": int(df["structure_signature"].nunique()),
                "target_max": float(df[TARGET].max()),
                "target_p95": float(df[TARGET].quantile(0.95)),
                "target_ge_0.005": int((df[TARGET] >= 0.005).sum()),
                "target_ge_0.010": int((df[TARGET] >= 0.010).sum()),
                "target_ge_0.015": int((df[TARGET] >= 0.015).sum()),
                "target_ge_0.020": int((df[TARGET] >= 0.020).sum()),
                "steel01_yielded_count": int((df["steel01_yielded"] == 1).sum()),
                "steel02_yielded_count": int((df["steel02_yielded"] == 1).sum()),
                "exact_wave_unseen_vs_train_count": int((~df["txt_path"].isin(train_waves)).sum()),
                "exact_wave_unseen_vs_trainval_count": int((~df["txt_path"].isin(trainval_waves)).sum()),
                "exact_structure_unseen_vs_train_count": int((~df["structure_signature"].isin(train_struct)).sum()),
                "exact_structure_unseen_vs_trainval_count": int((~df["structure_signature"].isin(trainval_struct)).sum()),
            }
        )
    return rows


def load_wave_split_candidates(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    candidates = sorted((PROJECT_ROOT / "newdata").glob("*drop-split-rank-top5drift_wave_split.csv"))
    if not candidates:
        return pd.DataFrame()
    wave_split = pd.read_csv(candidates[0])
    used_waves = set(frames["train"]["txt_path"]) | set(frames["val"]["txt_path"]) | set(frames["test"]["txt_path"])
    wave_split["used_in_current_train_val_test"] = wave_split["txt_path"].isin(used_waves)
    wave_split["wave_id"] = wave_split["txt_path"].map(wave_id_from_txt_path)
    return wave_split


def high_drift_generation_plan(frames: dict[str, pd.DataFrame], output_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wave_split = load_wave_split_candidates(frames)
    val_high = frames["val"].loc[frames["val"][TARGET] >= 0.010].copy()
    val_high = val_high.sort_values(TARGET, ascending=False)
    unused = pd.DataFrame()
    if not wave_split.empty:
        unused = wave_split.loc[~wave_split["used_in_current_train_val_test"]].copy()
        if unused.empty:
            unused = wave_split.loc[wave_split["split"].isin(["test", "val"])].copy()
        unused = unused.sort_values(["wave_intensity_score", "wave_pga"], ascending=False)
    rows: list[dict[str, Any]] = []
    structures = val_high.drop_duplicates("structure_signature").head(20)
    waves = unused.head(10)
    for design_idx, (_, srow) in enumerate(structures.iterrows(), start=1):
        if waves.empty:
            rows.append(
                {
                    "plan_id": f"todo10_hd_{design_idx:04d}",
                    "source": "validation_high_drift_structure_only",
                    "reason": "No unused rare-wave candidate found; generate new wave or use external records.",
                    "expected_status": "requires_new_opensees_run",
                    "source_val_target": srow[TARGET],
                    "source_val_sample_id": srow["sample_id"],
                    **{col: srow[col] for col in STRUCTURE_COLS},
                }
            )
            continue
        for wave_rank, (_, wrow) in enumerate(waves.iterrows(), start=1):
            rows.append(
                {
                    "plan_id": f"todo10_hd_{design_idx:04d}_w{wave_rank:02d}",
                    "source": "validation_high_drift_structure_x_unused_or_heldout_rare_wave",
                    "reason": "Independent high-drift/yielded stress benchmark requires new OpenSees labels after model freezing.",
                    "expected_status": "requires_new_opensees_run",
                    "source_val_target": srow[TARGET],
                    "source_val_sample_id": srow["sample_id"],
                    "candidate_txt_path": wrow.get("txt_path"),
                    "candidate_wave_id": wrow.get("wave_id"),
                    "candidate_wave_split": wrow.get("split"),
                    "candidate_wave_cluster": wrow.get("wave_cluster"),
                    "candidate_wave_pga": wrow.get("wave_pga"),
                    "candidate_wave_intensity_score": wrow.get("wave_intensity_score"),
                    **{col: srow[col] for col in STRUCTURE_COLS},
                }
            )
    manifest = {
        "validation_high_drift_structures_available": int(val_high["structure_signature"].nunique()),
        "validation_high_drift_rows": int(len(val_high)),
        "wave_split_candidates_found": int(len(wave_split)),
        "unused_wave_candidates_found": int(len(unused)) if not unused.empty else 0,
        "generated_plan_rows": len(rows),
        "claim_note": "This is a generation plan, not evaluated evidence. Metrics require running OpenSees and evaluating frozen models on the new labeled CSV.",
    }
    return rows, manifest


def make_table6(aggregate_rows: list[dict[str, Any]], output_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    df = pd.DataFrame(aggregate_rows)
    table = df.loc[
        df["benchmark_name"].isin(
            [
                "validation_reference_unseen_wave_structure",
                "locked_test_unseen_wave",
                "locked_test_unseen_structure",
                "validation_high_drift_ge_0.010_diagnostic",
                "locked_test_high_drift_ge_0.010",
            ]
        )
    ].copy()
    val_ref = table.loc[table["benchmark_name"] == "validation_reference_unseen_wave_structure", ["model", "mae_mean"]]
    val_ref = val_ref.rename(columns={"mae_mean": "validation_reference_mae_mean"})
    table = table.merge(val_ref, on="model", how="left")
    table["delta_mae_vs_validation_reference"] = table["mae_mean"] - table["validation_reference_mae_mean"]
    table["relative_mae_vs_validation_reference_pct"] = (
        table["delta_mae_vs_validation_reference"] / table["validation_reference_mae_mean"] * 100.0
    )
    table = table.sort_values(["benchmark_name", "mae_mean"], na_position="last").reset_index(drop=True)
    path = output_root / "paper_tables" / "table6_ood_generalization.csv"
    md_path = output_root / "paper_tables" / "table6_ood_generalization.md"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    md = table[
        [
            "benchmark_name",
            "model",
            "seed_count",
            "samples_mean",
            "mae_mean",
            "rmse_mean",
            "r2_mean",
            "delta_mae_vs_validation_reference",
            "benchmark_status",
            "claim_scope",
        ]
    ].copy()
    for col in ("samples_mean", "mae_mean", "rmse_mean", "r2_mean", "delta_mae_vs_validation_reference"):
        md[col] = md[col].map(lambda value: fmt(value, 6))
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Table 6 OOD Generalization\n\n")
        file.write("Validation reference is diagnostic because validation was used during model selection.\n\n")
        file.write(markdown_table(md))
        file.write("\n")
    return table, path, md_path


def plot_ood_performance(table6: pd.DataFrame, output_root: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmarks = [
        "validation_reference_unseen_wave_structure",
        "locked_test_unseen_wave",
        "locked_test_unseen_structure",
        "validation_high_drift_ge_0.010_diagnostic",
    ]
    plot_df = table6.loc[table6["benchmark_name"].isin(benchmarks) & table6["model"].isin(TOP_MODELS_FOR_FIGURE)].copy()
    plot_df["benchmark_name"] = pd.Categorical(plot_df["benchmark_name"], categories=benchmarks, ordered=True)
    plot_df = plot_df.sort_values(["benchmark_name", "model"])
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    x = np.arange(len(benchmarks), dtype=float)
    models = [model for model in TOP_MODELS_FOR_FIGURE if model in set(plot_df["model"])]
    width = 0.12
    for idx, model in enumerate(models):
        values = []
        errors = []
        for benchmark in benchmarks:
            row = plot_df.loc[(plot_df["benchmark_name"] == benchmark) & (plot_df["model"] == model)]
            values.append(float(row["mae_mean"].iloc[0]) if not row.empty and pd.notna(row["mae_mean"].iloc[0]) else np.nan)
            errors.append(float(row["mae_ci95_normal"].iloc[0]) if not row.empty and pd.notna(row["mae_ci95_normal"].iloc[0]) else 0.0)
        offset = (idx - (len(models) - 1) / 2) * width
        ax.bar(x + offset, values, width=width, yerr=errors, capsize=2, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Val reference\n(diagnostic)",
            "Locked unseen\nwave",
            "Locked unseen\nstructure",
            "Val high-drift\n(diagnostic)",
        ]
    )
    ax.set_ylabel("MAE")
    ax.set_title("OOD and stress-diagnostic performance comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    png = output_root / "paper_figures" / "ood_performance_drop.png"
    svg = output_root / "paper_figures" / "ood_performance_drop.svg"
    fig.savefig(png, dpi=240)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def create_report(
    output_root: Path,
    coverage: list[dict[str, Any]],
    table6: pd.DataFrame,
    generation_manifest: dict[str, Any],
    outputs: dict[str, str],
) -> Path:
    report_path = PROJECT_ROOT / "md_files" / "02_research_and_paper" / "literature_reviews" / "TODO10_OOD与高漂移benchmark补齐完成报告_20260617.md"
    coverage_df = pd.DataFrame(coverage)
    test_cov = coverage_df.loc[coverage_df["split"] == "test"].iloc[0]
    val_cov = coverage_df.loc[coverage_df["split"] == "val"].iloc[0]
    main_rows = table6.loc[table6["model"] == "2dcnn"].copy()
    best_locked = table6.loc[table6["benchmark_name"] == "locked_test_unseen_wave"].sort_values("mae_mean").head(1)
    best_high_diag = table6.loc[table6["benchmark_name"] == "validation_high_drift_ge_0.010_diagnostic"].sort_values("mae_mean").head(1)

    lines = [
        "# TODO10 OOD 与 high-drift benchmark 补齐完成报告",
        "",
        "生成日期：2026-06-17",
        "",
        "## 1. 补齐范围",
        "",
        "本轮只读取 protocol CSV 与 TODO5 frozen final predictions，不训练、不调参、不改测试集。TODO10 被拆成三类证据：已独立完成的 exact unseen-wave、已独立完成的 exact unseen-structure、以及当前数据不足但已给出冻结模型诊断和新 OpenSees 生成计划的 high-drift stress test。",
        "",
        "## 2. 已生成文件",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 3. 完成标准核对",
            "",
            "| TODO10 要求 | 本轮状态 | 证据文件 |",
            "|---|---|---|",
            "| 构建 unseen-wave 测试集 | 完成；locked test 的 exact `txt_path` 100% 未出现在 train+val | `unseen_wave_metrics.csv`, `ood_split_coverage.csv` |",
            "| 构建 unseen-structure 测试集 | 完成；locked test 的 exact structure signature 100% 未出现在 train+val | `unseen_structure_metrics.csv`, `ood_split_coverage.csv` |",
            "| 构建 high-drift stress test | 独立 labeled stress set 仍缺失；已输出 validation 诊断指标与新 OpenSees 工况生成计划 | `high_drift_stress_metrics.csv`, `high_drift_opensees_generation_plan.csv` |",
            "| 数据不足时明确生成新的 OpenSees 工况 | 完成；不得用 validation high-drift 代替独立结论 | `high_drift_opensees_generation_plan.csv` |",
            "| 所有 OOD benchmark 在模型冻结后运行 | 完成；所有指标来自 TODO5 final frozen predictions | `todo10_completion_manifest.json` |",
            "",
            "## 4. 数据覆盖结论",
            "",
            f"- locked test rows: `{int(test_cov['rows'])}`；exact unseen wave vs train+val: `{int(test_cov['exact_wave_unseen_vs_trainval_count'])}/{int(test_cov['rows'])}`。",
            f"- locked test exact unseen structure vs train+val: `{int(test_cov['exact_structure_unseen_vs_trainval_count'])}/{int(test_cov['rows'])}`。",
            f"- locked test `target >= 0.010`: `{int(test_cov['target_ge_0.010'])}`；`steel01_yielded=1`: `{int(test_cov['steel01_yielded_count'])}`。",
            f"- validation `target >= 0.010`: `{int(val_cov['target_ge_0.010'])}`；`steel01_yielded=1`: `{int(val_cov['steel01_yielded_count'])}`，但 validation 已用于 HPO/early stopping，因此只能做诊断。",
            "",
            "## 5. 主要结果",
            "",
        ]
    )
    if not best_locked.empty:
        lines.append(
            f"- locked unseen-wave/unseen-structure 上 MAE 最低模型：`{best_locked.iloc[0]['model']}`，MAE = `{fmt(best_locked.iloc[0]['mae_mean'])}`。"
        )
    if not best_high_diag.empty:
        lines.append(
            f"- validation high-drift diagnostic 上 MAE 最低模型：`{best_high_diag.iloc[0]['model']}`，MAE = `{fmt(best_high_diag.iloc[0]['mae_mean'])}`。"
        )
    if not main_rows.empty:
        for _, row in main_rows.iterrows():
            if row["benchmark_name"] in {"locked_test_unseen_wave", "validation_high_drift_ge_0.010_diagnostic"}:
                lines.append(
                    f"- `2dcnn` on `{row['benchmark_name']}`: MAE = `{fmt(row['mae_mean'])}`, "
                    f"RMSE = `{fmt(row['rmse_mean'])}`, samples = `{fmt(row['samples_mean'], 0)}`."
                )
    lines.extend(
        [
            "",
            "## 6. 论文主张边界",
            "",
            "可以写：当前 locked protocol 实际上已经是 exact unseen-wave 和 exact unseen-structure split，2D-CNN 的主结果是在该独立 split 上获得，而不是同波同结构插值测试。",
            "",
            "不能写：模型已经证明在 high-drift、主体屈服、接近失效或 `target >= 0.010/0.015/0.020` 工况下可靠。原因是独立 locked test 中这些样本为 0；validation 中虽然有 high-drift/yielded 样本，但它参与过 HPO/early stopping，不能替代外部 stress test。",
            "",
            "## 7. OpenSees high-drift 生成计划",
            "",
            f"已生成 `{generation_manifest['generated_plan_rows']}` 条候选工况。候选思路是：从 validation high-drift 结构签名中抽取高响应结构模板，与未使用或 held-out rare-wave 候选组合，重新运行 OpenSees 得到新的 labeled CSV，再用已冻结模型做一次纯 evaluation。该步骤完成后，应重新运行本脚本以填充真正的 independent high-drift stress metrics。",
            "",
            "## 8. 参考文献与规范",
            "",
            "以下文献与规范已于 2026-06-17 核对，优先采用 OpenReview、NeurIPS Proceedings、PMLR、Science Advances/REFORMS 官方页面或 EQUATOR guideline 记录。它们支撑本轮将 OOD split、模型冻结后评估、validation overfitting 边界、claim boundary 和可复现报告作为论文证据链。",
            "",
            "1. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3",
            "2. Erickson, N., Purucker, L., Tschalzev, A., Holzmueller, D., Mutalik Desai, P., Salinas, D., & Hutter, F. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS 2025 Datasets and Benchmarks Track spotlight. https://openreview.net/forum?id=jZqCqpCLdU",
            "3. Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. ICML 2021. https://proceedings.mlr.press/v139/koh21a.html",
            "4. Gardner, J., Popovic, Z., & Schmidt, L. (2023). Benchmarking distribution shift in tabular data with TableShift. NeurIPS 2023 Datasets and Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html",
            "5. Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J. M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D., Roberts, M., Salganik, M., Serra-Garcia, M., Stewart, B. M., Vandewiele, G., & Narayanan, A. (2024). REFORMS: Reporting standards for machine learning based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452",
            "",
            "## 9. TODO10 状态结论",
            "",
            "TODO10 的 unseen-wave 与 unseen-structure benchmark 已完成。High-drift stress test 已完成覆盖审计、冻结模型诊断和新 OpenSees 工况计划，但独立 high-drift labeled benchmark 仍需要实际 OpenSees 生成后才能支持强安全结论。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete TODO10 OOD and high-drift benchmark artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    ood_dir = output_root / "ood_benchmark"
    table_dir = output_root / "paper_tables"
    figure_dir = output_root / "paper_figures"
    stat_dir = output_root / "statistics"
    protocol = load_protocol(output_root)
    frames = load_split_frames(protocol)
    predictions = discover_predictions(output_root, frames)

    by_seed, aggregate_rows, sample_rows = benchmark_rows(predictions, frames)
    coverage_rows = split_coverage_rows(frames)
    generation_rows, generation_manifest = high_drift_generation_plan(frames, output_root)

    all_by_seed_path = ood_dir / "ood_benchmark_metrics_by_seed.csv"
    all_agg_path = ood_dir / "ood_benchmark_metrics.csv"
    write_csv(all_by_seed_path, by_seed)
    write_csv(all_agg_path, aggregate_rows)
    write_csv(ood_dir / "ood_benchmark_sample_index.csv", sample_rows)
    coverage_path = ood_dir / "ood_split_coverage.csv"
    write_csv(coverage_path, coverage_rows)
    generation_path = ood_dir / "high_drift_opensees_generation_plan.csv"
    write_csv(generation_path, generation_rows)

    unseen_wave_rows = [row for row in aggregate_rows if row["benchmark_name"] == "locked_test_unseen_wave"]
    unseen_structure_rows = [row for row in aggregate_rows if row["benchmark_name"] == "locked_test_unseen_structure"]
    high_drift_rows = [
        row
        for row in aggregate_rows
        if row["benchmark_name"] in {"locked_test_high_drift_ge_0.010", "validation_high_drift_ge_0.010_diagnostic"}
    ]
    unseen_wave_path = ood_dir / "unseen_wave_metrics.csv"
    unseen_structure_path = ood_dir / "unseen_structure_metrics.csv"
    high_drift_path = ood_dir / "high_drift_stress_metrics.csv"
    write_csv(unseen_wave_path, unseen_wave_rows)
    write_csv(unseen_structure_path, unseen_structure_rows)
    write_csv(high_drift_path, high_drift_rows)

    table6, table6_path, table6_md = make_table6(aggregate_rows, output_root)
    fig_png, fig_svg = plot_ood_performance(table6, output_root)

    outputs = {
        "unseen_wave_metrics": str(unseen_wave_path),
        "unseen_structure_metrics": str(unseen_structure_path),
        "high_drift_stress_metrics": str(high_drift_path),
        "ood_benchmark_metrics": str(all_agg_path),
        "ood_benchmark_metrics_by_seed": str(all_by_seed_path),
        "ood_benchmark_sample_index": str(ood_dir / "ood_benchmark_sample_index.csv"),
        "ood_split_coverage": str(coverage_path),
        "high_drift_opensees_generation_plan": str(generation_path),
        "table6_ood_generalization": str(table6_path),
        "table6_ood_generalization_md": str(table6_md),
        "ood_performance_drop_png": str(fig_png),
        "ood_performance_drop_svg": str(fig_svg),
    }
    report_path = create_report(output_root, coverage_rows, table6, generation_manifest, outputs)
    outputs["todo10_completion_report"] = str(report_path)

    coverage_frame = pd.DataFrame(coverage_rows)
    test_cov = coverage_frame.loc[coverage_frame["split"] == "test"].iloc[0].to_dict()
    val_cov = coverage_frame.loc[coverage_frame["split"] == "val"].iloc[0].to_dict()
    manifest = {
        "run_name": "todo10_ood_benchmark_completion",
        "generated_at": "2026-06-17",
        "source_output_root": str(output_root),
        "prediction_source": str(output_root / "final_multiseed"),
        "models": sorted({model for model, _seed, _phase in predictions}),
        "seeds": sorted({seed for _model, seed, _phase in predictions}),
        "status": {
            "unseen_wave": "complete_independent_locked_test",
            "unseen_structure": "complete_independent_locked_test_exact_signature",
            "high_drift_independent": "not_available_requires_new_opensees_labels",
            "high_drift_diagnostic": "completed_on_validation_only_not_claimworthy",
        },
        "locked_test_coverage": test_cov,
        "validation_coverage": val_cov,
        "independent_high_drift_blocker": {
            "locked_test_target_ge_0.010": int(test_cov.get("target_ge_0.010", 0)),
            "locked_test_target_ge_0.015": int(test_cov.get("target_ge_0.015", 0)),
            "locked_test_target_ge_0.020": int(test_cov.get("target_ge_0.020", 0)),
            "locked_test_steel01_yielded_count": int(test_cov.get("steel01_yielded_count", 0)),
            "required_next_artifact": str(generation_path),
            "note": "Run these candidate cases in OpenSees to create a new labeled stress CSV, then re-run this script without retraining models.",
        },
        "generation_plan": generation_manifest,
        "outputs": outputs,
        "claim_boundaries": [
            "Locked test is 100% exact unseen-wave and exact unseen-structure relative to train+val.",
            "Unseen-structure means exact structural signature, not extrapolation to unseen floors or parameter ranges.",
            "Locked test has no target >= 0.010, >= 0.015, >= 0.020, or steel01_yielded=1 samples.",
            "Validation high-drift diagnostics are not independent final evidence because validation was used during HPO/early stopping.",
            "Strong high-drift/yielded safety claims require running the generated OpenSees stress plan and evaluating frozen models on the new labeled CSV.",
        ],
    }
    manifest_path = stat_dir / "todo10_completion_manifest.json"
    save_json(manifest_path, manifest)

    print(f">>> TODO10 unseen-wave metrics: {unseen_wave_path}")
    print(f">>> TODO10 unseen-structure metrics: {unseen_structure_path}")
    print(f">>> TODO10 high-drift metrics/diagnostic: {high_drift_path}")
    print(f">>> TODO10 Table 6: {table6_path}")
    print(f">>> TODO10 Figure 6: {fig_png}")
    print(f">>> TODO10 manifest: {manifest_path}")
    print(f">>> TODO10 report: {report_path}")


if __name__ == "__main__":
    main()
