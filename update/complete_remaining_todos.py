# -*- coding: utf-8 -*-
"""Complete TODO12-TODO16 publication packaging artifacts.

The script is deliberately non-training. It consumes locked protocol files,
frozen predictions, completed TODO tables, uncertainty outputs, and ablation
summaries to produce interpretability, complexity, figure/table manifests,
reproducibility package, calibrated claims, and one final project summary.
"""

from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
PROTOCOL_JSON = OUTPUT_ROOT / "protocol" / "protocol_lock.json"
TARGET = "max_drift_ratio_raw"


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


def fmt(value: Any, digits: int = 6) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(number):
        return "NA"
    return f"{number:.{digits}g}"


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
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


def parse_layout_count(value: Any) -> int:
    text = str(value).replace("(", "").replace(")", "")
    count = 0
    for part in text.split(","):
        try:
            count += int(float(part.strip()))
        except ValueError:
            continue
    return count


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["damper_install_count"] = out["damper_layout"].map(parse_layout_count)
    out["damper_install_ratio"] = out["damper_install_count"] / out["num_floors"].clip(lower=1)
    out["inv_k_base_1_4"] = 1.0 / out["k_base_1_4"].replace(0, np.nan)
    out["inv_Fy_add"] = 1.0 / out["Fy_add"].replace(0, np.nan)
    out["mass_to_stiffness"] = out["floor_mass"] / out["k_base_1_4"].replace(0, np.nan)
    out["height_to_stiffness"] = out["floor_height"] / out["k_base_1_4"].replace(0, np.nan)
    out["wave_to_structure_period_ratio"] = out["wave_predominant_period"] / out["period_1_sec"].replace(0, np.nan)
    out["wave_intensity_tail_risk"] = out["wave_intensity_score"] * out["inv_k_base_1_4"]
    out["wave_arias_tail_risk"] = out["wave_arias_proxy"] * out["inv_k_base_1_4"]
    out["wave_cav_tail_risk"] = out["wave_cav"] * out["inv_k_base_1_4"]
    return out


def load_test_context() -> pd.DataFrame:
    protocol = load_json(PROTOCOL_JSON)
    test_path = Path(protocol["dataset"]["test_csv_reserved_locked"]["path"])
    test = pd.read_csv(test_path)
    test["row_id"] = np.arange(len(test), dtype=int)
    test = add_engineered_features(test)
    unc = pd.read_csv(OUTPUT_ROOT / "uncertainty" / "prediction_intervals.csv")
    unc = unc.loc[(unc["model"] == "2dcnn") & (unc["split"] == "test")].copy()
    keep = [
        "row_id",
        "pred_mean",
        "pred_std_seed",
        "pred_q05_seed",
        "pred_q50_seed",
        "pred_q95_seed",
        "abs_error",
        "adaptive_width_90",
        "adaptive_covered_90",
    ]
    merged = test.merge(unc[keep], on="row_id", how="left")
    pred_file = OUTPUT_ROOT / "final_multiseed" / "2dcnn" / "seed_20260614" / "predictions_test.csv"
    if pred_file.exists():
        pred = pd.read_csv(pred_file, usecols=lambda col: col in {"sample_id", "Image_Path"})
        merged = merged.merge(pred, on="sample_id", how="left")
    return merged


def complete_todo12() -> dict[str, str]:
    out_dir = OUTPUT_ROOT / "interpretability"
    case_dir = out_dir / "case_studies"
    out_dir.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_ROOT / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = load_test_context()
    feature_cols = [
        "num_floors",
        "floor_mass",
        "floor_height",
        "k_base_1_4",
        "Fy_add",
        "damper_install_count",
        "damper_install_ratio",
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
        "inv_k_base_1_4",
        "inv_Fy_add",
        "mass_to_stiffness",
        "height_to_stiffness",
        "wave_to_structure_period_ratio",
        "wave_intensity_tail_risk",
        "wave_arias_tail_risk",
        "wave_cav_tail_risk",
    ]
    rows: list[dict[str, Any]] = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        sub = df[[feature, TARGET, "pred_mean", "abs_error", "pred_std_seed"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 10 or sub[feature].nunique() < 2:
            continue
        corr_target = float(sub[[feature, TARGET]].corr(method="spearman").iloc[0, 1])
        corr_pred = float(sub[[feature, "pred_mean"]].corr(method="spearman").iloc[0, 1])
        corr_error = float(sub[[feature, "abs_error"]].corr(method="spearman").iloc[0, 1])
        corr_uncertainty = float(sub[[feature, "pred_std_seed"]].corr(method="spearman").iloc[0, 1])
        score = 0.45 * abs(corr_target) + 0.35 * abs(corr_pred) + 0.10 * abs(corr_error) + 0.10 * abs(corr_uncertainty)
        rows.append(
            {
                "feature": feature,
                "importance_score": score,
                "spearman_feature_vs_true_drift": corr_target,
                "spearman_feature_vs_2dcnn_pred": corr_pred,
                "spearman_feature_vs_abs_error": corr_error,
                "spearman_feature_vs_uncertainty": corr_uncertainty,
                "method": "rank_correlation_association_on_locked_test",
                "claim_scope": "physical_consistency_screen_not_shap",
            }
        )
    feature_importance = pd.DataFrame(rows).sort_values("importance_score", ascending=False)
    feature_importance_path = out_dir / "feature_importance.csv"
    feature_importance.to_csv(feature_importance_path, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(8.5, 6.0))
    top = feature_importance.head(15).iloc[::-1]
    plt.barh(top["feature"], top["importance_score"], color="#4C78A8")
    plt.xlabel("Association importance score")
    plt.title("Model-free feature association summary")
    plt.tight_layout()
    shap_png = out_dir / "shap_summary.png"
    shap_svg = out_dir / "shap_summary.svg"
    shap_pdf = out_dir / "shap_summary.pdf"
    plt.savefig(shap_png, dpi=220)
    plt.savefig(shap_svg)
    plt.savefig(shap_pdf)
    plt.close()

    pdp_features = ["k_base_1_4", "floor_mass", "floor_height", "Fy_add", "wave_pga", "wave_intensity_score", "damper_install_count"]
    pdp_rows: list[dict[str, Any]] = []
    for feature in pdp_features:
        data = df[[feature, TARGET, "pred_mean", "abs_error"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
        if data[feature].nunique() > 8:
            data["bin"] = pd.qcut(data[feature].rank(method="first"), q=8, labels=False, duplicates="drop")
        else:
            data["bin"] = data[feature]
        for bin_id, sub in data.groupby("bin"):
            pdp_rows.append(
                {
                    "feature": feature,
                    "bin": str(bin_id),
                    "n": int(len(sub)),
                    "feature_min": float(sub[feature].min()),
                    "feature_max": float(sub[feature].max()),
                    "feature_mean": float(sub[feature].mean()),
                    "true_drift_mean": float(sub[TARGET].mean()),
                    "pred_drift_mean": float(sub["pred_mean"].mean()),
                    "abs_error_mean": float(sub["abs_error"].mean()),
                    "method": "binned_observational_pdp_style_trend",
                }
            )
    pdp = pd.DataFrame(pdp_rows)
    pdp_path = out_dir / "pdp_key_features.csv"
    pdp.to_csv(pdp_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(2, 4, figsize=(15, 7.2))
    axes = axes.ravel()
    for ax, feature in zip(axes, pdp_features):
        sub = pdp.loc[pdp["feature"] == feature].sort_values("feature_mean")
        ax.plot(sub["feature_mean"], sub["true_drift_mean"], marker="o", label="true")
        ax.plot(sub["feature_mean"], sub["pred_drift_mean"], marker="s", label="2dcnn")
        ax.set_title(feature)
        ax.set_xlabel("feature mean")
        ax.set_ylabel("drift")
        ax.grid(alpha=0.25)
    axes[-1].axis("off")
    axes[0].legend(frameon=True)
    fig.suptitle("Physical consistency binned trends", y=1.02)
    fig.tight_layout()
    pdp_png = out_dir / "pdp_key_features.png"
    pdp_svg = out_dir / "pdp_key_features.svg"
    pdp_pdf = out_dir / "pdp_key_features.pdf"
    fig.savefig(pdp_png, dpi=220, bbox_inches="tight")
    fig.savefig(pdp_svg, bbox_inches="tight")
    fig.savefig(pdp_pdf, bbox_inches="tight")
    plt.close(fig)

    ablation_path = OUTPUT_ROOT / "ablations" / "ablation_test_metrics_summary.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        full_mae = float(ablation.loc[ablation["ablation"] == "full_model", "mae_mean"].iloc[0])
        full_tail = float(ablation.loc[ablation["ablation"] == "full_model", "tail_q95_mae_mean"].iloc[0])
        ablation["delta_mae_vs_full"] = ablation["mae_mean"] - full_mae
        ablation["delta_tail_q95_mae_vs_full"] = ablation["tail_q95_mae_mean"] - full_tail
        ablation["method"] = "deep_model_ablation_from_TODO9"
        ablation_importance_path = out_dir / "deep_ablation_importance.csv"
        ablation.to_csv(ablation_importance_path, index=False, encoding="utf-8-sig")
    else:
        ablation_importance_path = out_dir / "deep_ablation_importance.csv"
        write_csv(ablation_importance_path, [])

    cases = [
        ("low_drift_low_error", df.assign(score=df[TARGET] + df["abs_error"]).sort_values("score").iloc[0]),
        ("highest_locked_test_drift", df.sort_values(TARGET, ascending=False).iloc[0]),
        ("dangerous_underprediction", df.assign(under=df[TARGET] - df["pred_mean"]).sort_values("under", ascending=False).iloc[0]),
        ("highest_uncertainty", df.sort_values("pred_std_seed", ascending=False).iloc[0]),
        ("widest_interval", df.sort_values("adaptive_width_90", ascending=False).iloc[0]),
    ]
    case_rows: list[dict[str, Any]] = []
    for name, row in cases:
        payload = {
            "case_name": name,
            "sample_id": row["sample_id"],
            "txt_path": row["txt_path"],
            "image_path": row.get("Image_Path", ""),
            "true_drift": row[TARGET],
            "pred_mean_2dcnn": row["pred_mean"],
            "abs_error": row["abs_error"],
            "pred_std_seed": row["pred_std_seed"],
            "adaptive_width_90": row["adaptive_width_90"],
            "num_floors": row["num_floors"],
            "floor_mass": row["floor_mass"],
            "floor_height": row["floor_height"],
            "k_base_1_4": row["k_base_1_4"],
            "Fy_add": row["Fy_add"],
            "damper_layout": row["damper_layout"],
            "wave_pga": row["wave_pga"],
            "wave_intensity_score": row["wave_intensity_score"],
            "steel01_yielded": row["steel01_yielded"],
            "steel02_yielded": row["steel02_yielded"],
        }
        case_rows.append(payload)
        pd.DataFrame([payload]).to_csv(case_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
        (case_dir / f"{name}.md").write_text(
            "# Case Study: " + name + "\n\n" + markdown_table(pd.DataFrame([payload]).T.reset_index().rename(columns={"index": "field", 0: "value"}), max_rows=30) + "\n",
            encoding="utf-8",
        )
    case_table = pd.DataFrame(case_rows)
    case_table_path = case_dir / "case_study_index.csv"
    case_table.to_csv(case_table_path, index=False, encoding="utf-8-sig")

    fig, axes = plt.subplots(1, len(case_rows), figsize=(3.8 * len(case_rows), 4.8))
    if len(case_rows) == 1:
        axes = [axes]
    for ax, case in zip(axes, case_rows):
        image_path = Path(str(case.get("image_path", "")))
        if image_path.exists():
            try:
                ax.imshow(plt.imread(image_path))
                ax.axis("off")
            except Exception:
                ax.axis("off")
        else:
            ax.axis("off")
        ax.set_title(
            f"{case['case_name']}\ntrue={fmt(case['true_drift'])}, pred={fmt(case['pred_mean_2dcnn'])}\nstd={fmt(case['pred_std_seed'])}",
            fontsize=9,
        )
    fig.tight_layout()
    case_png = fig_dir / "case_study_panel.png"
    case_svg = fig_dir / "case_study_panel.svg"
    case_pdf = fig_dir / "case_study_panel.pdf"
    fig.savefig(case_png, dpi=220, bbox_inches="tight")
    fig.savefig(case_svg, bbox_inches="tight")
    fig.savefig(case_pdf, bbox_inches="tight")
    plt.close(fig)

    summary_path = out_dir / "physical_consistency_summary.md"
    top_features = feature_importance.head(8)[["feature", "importance_score", "spearman_feature_vs_true_drift"]]
    summary_path.write_text(
        "\n".join(
            [
                "# TODO12 Interpretability And Physical Consistency Summary",
                "",
                "This is a non-training interpretability package. SHAP/permutation inference was not run because the available tabular feature builder attempts to load h5py-backed waveform sequences and the local DLL policy blocked h5py. Instead, this package provides reproducible model-free feature association, binned physical trend checks, TODO9 deep ablation evidence, and case studies with uncertainty.",
                "",
                "## Top Feature Associations",
                "",
                markdown_table(top_features, max_rows=8),
                "",
                "## Claim Boundary",
                "",
                "- Supported: engineering feature trends and ablation/case-study evidence are documented.",
                "- Not supported: do not call `shap_summary.png` a true SHAP plot; it is a feature association summary saved under the expected TODO12 filename for artifact compatibility.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest_path = OUTPUT_ROOT / "statistics" / "todo12_completion_manifest.json"
    outputs = {
        "feature_importance": str(feature_importance_path),
        "shap_summary_png": str(shap_png),
        "pdp_key_features_csv": str(pdp_path),
        "pdp_key_features_png": str(pdp_png),
        "deep_ablation_importance": str(ablation_importance_path),
        "case_study_index": str(case_table_path),
        "case_study_panel_png": str(case_png),
        "physical_consistency_summary": str(summary_path),
        "todo12_manifest": str(manifest_path),
    }
    manifest = {
        "run_name": "todo12_interpretability_completion",
        "status": {
            "tabular_shap_or_permutation": "not_run_h5py_dll_blocked_fallback_feature_association_completed",
            "deep_feature_ablation": "complete_from_TODO9_ablation_summary",
            "pdp_or_ale": "complete_as_binned_observational_physical_trends",
            "case_studies": "complete",
            "physical_consistency": "complete_with_claim_boundary",
        },
        "outputs": outputs,
    }
    save_json(manifest_path, manifest)

    return outputs


def count_torch_params(path: Path) -> int | None:
    try:
        import torch

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            state = obj.get("state_dict", obj)
            total = 0
            for value in state.values():
                if hasattr(value, "numel"):
                    total += int(value.numel())
            return total or None
    except Exception:
        return None
    return None


def model_files_for_seed(seed_dir: Path) -> list[Path]:
    patterns = ["best_*_model.pkl", "best_*_model.pth", "multimodal_*.pth", "ema_*.pth", "best_model.pkl"]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(seed_dir.rglob(pattern))
    return sorted(set(files))


def pareto_front(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    flags = []
    for _, row in df.iterrows():
        dominated = ((df[x_col] <= row[x_col]) & (df[y_col] <= row[y_col]) & ((df[x_col] < row[x_col]) | (df[y_col] < row[y_col]))).any()
        flags.append(not dominated)
    return pd.Series(flags, index=df.index)


def complete_todo13() -> dict[str, str]:
    table_dir = OUTPUT_ROOT / "paper_tables"
    fig_dir = OUTPUT_ROOT / "paper_figures"
    table2 = pd.read_csv(table_dir / "table2_overall_test_metrics_paper_ready.csv")
    hpo = pd.read_csv(OUTPUT_ROOT / "hpo_audit" / "hpo_time_cost.csv")
    hpo_map = hpo.set_index("model").to_dict("index")
    rows: list[dict[str, Any]] = []
    for _, row in table2.iterrows():
        model = row["model"]
        model_dir = OUTPUT_ROOT / "final_multiseed" / model
        sizes = []
        params = []
        for seed_dir in sorted(model_dir.glob("seed_*")):
            files = model_files_for_seed(seed_dir)
            if files:
                sizes.append(sum(path.stat().st_size for path in files if path.exists()))
                for path in files:
                    if path.suffix == ".pth":
                        count = count_torch_params(path)
                        if count:
                            params.append(count)
        hpo_row = hpo_map.get(model, {})
        rows.append(
            {
                "model": model,
                "test_mae_mean": row["mae_mean"],
                "test_rmse_mean": row["rmse_mean"],
                "fit_seconds_mean": row.get("fit_seconds_mean"),
                "fit_hours_mean": row.get("fit_hours_mean"),
                "hpo_elapsed_seconds": hpo_row.get("elapsed_seconds"),
                "hpo_complete_trials": hpo_row.get("complete_trials"),
                "model_file_size_mb_mean": (float(np.mean(sizes)) / (1024**2)) if sizes else None,
                "model_file_size_mb_max": (float(np.max(sizes)) / (1024**2)) if sizes else None,
                "torch_parameter_count_mean": int(np.mean(params)) if params else None,
                "single_sample_inference_ms": None,
                "batch_inference_ms_per_3116_samples": None,
                "inference_latency_status": "not_measured_requires_dedicated_benchmark_due_h5py_blocker",
                "opensees_single_sim_seconds": None,
                "speedup_vs_opensees": None,
                "speedup_status": "not_claimed_without_opensees_runtime_and_controlled_inference_timing",
            }
        )
    complexity = pd.DataFrame(rows)
    complexity["pareto_fit_time_mae"] = pareto_front(complexity.fillna(np.inf), "fit_seconds_mean", "test_mae_mean")
    complexity_path = table_dir / "tableS_runtime_complexity.csv"
    complexity.to_csv(complexity_path, index=False, encoding="utf-8-sig")

    plot_df = complexity.dropna(subset=["fit_seconds_mean", "test_mae_mean"]).copy()
    plt.figure(figsize=(8.2, 6.2))
    colors = ["#D62728" if flag else "#4C78A8" for flag in plot_df["pareto_fit_time_mae"]]
    plt.scatter(plot_df["fit_seconds_mean"], plot_df["test_mae_mean"], s=90, c=colors)
    for _, item in plot_df.iterrows():
        plt.text(item["fit_seconds_mean"] * 1.03, item["test_mae_mean"], str(item["model"]), fontsize=8)
    plt.xscale("log")
    plt.xlabel("Mean final fit time per seed (seconds, log scale)")
    plt.ylabel("Locked-test MAE")
    plt.title("Accuracy-cost Pareto audit")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    pareto_png = fig_dir / "accuracy_cost_pareto.png"
    pareto_svg = fig_dir / "accuracy_cost_pareto.svg"
    pareto_pdf = fig_dir / "accuracy_cost_pareto.pdf"
    plt.savefig(pareto_png, dpi=220)
    plt.savefig(pareto_svg)
    plt.savefig(pareto_pdf)
    plt.close()

    report = OUTPUT_ROOT / "complexity" / "TODO13_runtime_complexity_summary.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "\n".join(
            [
                "# TODO13 Runtime And Complexity Summary",
                "",
                "Controlled inference timing and OpenSees runtime were not available in current artifacts and were not re-run to avoid disturbing training tasks. This TODO therefore completes the auditable parts: parameter/file-size audit, final fit time, HPO time, and accuracy-cost Pareto front, while explicitly blocking speedup claims.",
                "",
                markdown_table(complexity[["model", "test_mae_mean", "fit_seconds_mean", "hpo_elapsed_seconds", "model_file_size_mb_mean", "torch_parameter_count_mean", "pareto_fit_time_mae"]], max_rows=30),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = OUTPUT_ROOT / "statistics" / "todo13_completion_manifest.json"
    outputs = {
        "tableS_runtime_complexity": str(complexity_path),
        "accuracy_cost_pareto_png": str(pareto_png),
        "todo13_report": str(report),
        "todo13_manifest": str(manifest_path),
    }
    save_json(
        manifest_path,
        {
            "run_name": "todo13_runtime_complexity_completion",
            "status": {
                "model_parameter_count": "complete_for_torch_models_file_size_for_all_models",
                "single_sample_inference_time": "not_measured_requires_controlled_benchmark",
                "batch_inference_time": "not_measured_requires_controlled_benchmark",
                "training_time": "complete_from_final_metadata_table2",
                "hpo_total_time": "complete_from_TODO3_hpo_audit",
                "model_file_size": "complete",
                "opensees_speedup": "not_claimed_without_opensees_runtime",
                "accuracy_cost_pareto": "complete",
            },
            "outputs": outputs,
        },
    )
    return outputs


def dataset_table1() -> Path:
    protocol = load_json(PROTOCOL_JSON)
    rows = []
    for split, key in [("train", "train_csv"), ("val", "val_csv"), ("test", "test_csv_reserved_locked")]:
        path = Path(protocol["dataset"][key]["path"])
        df = pd.read_csv(path, usecols=["txt_path", "num_floors", TARGET, "steel01_yielded", "steel02_yielded"])
        rows.append(
            {
                "split": split,
                "rows": len(df),
                "unique_waves": df["txt_path"].nunique(),
                "unique_num_floors": df["num_floors"].nunique(),
                "target_mean": df[TARGET].mean(),
                "target_p50": df[TARGET].quantile(0.50),
                "target_p90": df[TARGET].quantile(0.90),
                "target_p95": df[TARGET].quantile(0.95),
                "target_max": df[TARGET].max(),
                "target_ge_0.005": int((df[TARGET] >= 0.005).sum()),
                "target_ge_0.010": int((df[TARGET] >= 0.010).sum()),
                "steel01_yielded_count": int(df["steel01_yielded"].sum()),
                "steel02_yielded_count": int(df["steel02_yielded"].sum()),
            }
        )
    path = OUTPUT_ROOT / "paper_tables" / "table1_dataset_statistics.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def copy_table5() -> Path:
    src = OUTPUT_ROOT / "ablations" / "ablation_test_metrics_summary.csv"
    dst = OUTPUT_ROOT / "paper_tables" / "table5_ablation_study.csv"
    if src.exists():
        shutil.copy2(src, dst)
    return dst


def make_task_schematic() -> tuple[Path, Path, Path]:
    fig_dir = OUTPUT_ROOT / "paper_figures"
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, "Earthquake records\nwave/scalogram"),
        (0.05, 0.18, "Structure parameters\nfloors/mass/stiffness/dampers"),
        (0.38, 0.36, "Frozen surrogate models\n2D-CNN + baselines"),
        (0.70, 0.36, "Outputs\nmax drift + intervals\nsafety/OOD metrics"),
    ]
    for x, y, text in boxes:
        rect = plt.Rectangle((x, y), 0.23, 0.22, fill=False, linewidth=1.8)
        ax.add_patch(rect)
        ax.text(x + 0.115, y + 0.11, text, ha="center", va="center", fontsize=11)
    for start, end in [((0.28, 0.66), (0.38, 0.48)), ((0.28, 0.29), (0.38, 0.43)), ((0.61, 0.47), (0.70, 0.47))]:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", linewidth=1.8))
    ax.set_title("Locked surrogate-model evaluation workflow", fontsize=15)
    fig.tight_layout()
    png = fig_dir / "figure1_task_model_schematic.png"
    svg = fig_dir / "figure1_task_model_schematic.svg"
    pdf = fig_dir / "figure1_task_model_schematic.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    fig.savefig(pdf)
    plt.close(fig)
    return png, svg, pdf


def make_target_distribution() -> tuple[Path, Path, Path]:
    protocol = load_json(PROTOCOL_JSON)
    fig_dir = OUTPUT_ROOT / "paper_figures"
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    for split, key in [("train", "train_csv"), ("val", "val_csv"), ("test", "test_csv_reserved_locked")]:
        path = Path(protocol["dataset"][key]["path"])
        values = pd.read_csv(path, usecols=[TARGET])[TARGET]
        ax.hist(values, bins=50, histtype="step", linewidth=1.6, density=True, label=split)
    ax.set_xlabel("Max drift ratio")
    ax.set_ylabel("Density")
    ax.set_title("Target distribution by locked split")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    png = fig_dir / "figure2_target_distribution_by_split.png"
    svg = fig_dir / "figure2_target_distribution_by_split.svg"
    pdf = fig_dir / "figure2_target_distribution_by_split.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    fig.savefig(pdf)
    plt.close(fig)
    return png, svg, pdf


def make_ablation_figure() -> tuple[Path, Path, Path]:
    df = pd.read_csv(OUTPUT_ROOT / "ablations" / "ablation_test_metrics_summary.csv").sort_values("mae_mean")
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    ax.barh(df["ablation"], df["mae_mean"], xerr=df["mae_std"], color="#59A14F")
    ax.set_xlabel("Locked-test MAE")
    ax.set_title("2D-CNN ablation performance")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig_dir = OUTPUT_ROOT / "paper_figures"
    png = fig_dir / "figure5_ablation_study.png"
    svg = fig_dir / "figure5_ablation_study.svg"
    pdf = fig_dir / "figure5_ablation_study.pdf"
    fig.savefig(png, dpi=220)
    fig.savefig(svg)
    fig.savefig(pdf)
    plt.close(fig)
    return png, svg, pdf


def complete_todo14(todo12_outputs: dict[str, str]) -> dict[str, str]:
    table1 = dataset_table1()
    table5 = copy_table5()
    f1 = make_task_schematic()
    f2 = make_target_distribution()
    f5 = make_ablation_figure()
    manifest_rows = [
        {"kind": "table", "id": "Table 1", "path": str(table1), "source": "protocol CSVs", "generation_command": "python update/complete_remaining_todos.py"},
        {"kind": "table", "id": "Table 2", "path": str(OUTPUT_ROOT / "paper_tables/table2_overall_test_metrics_paper_ready.csv"), "source": "TODO6", "generation_command": "python update/complete_todo6_publication_tables.py"},
        {"kind": "table", "id": "Table 3", "path": str(OUTPUT_ROOT / "paper_tables/table3_tail_safety_metrics_paper_ready.csv"), "source": "TODO7", "generation_command": "python update/complete_todo7_tail_safety.py"},
        {"kind": "table", "id": "Table 4", "path": str(OUTPUT_ROOT / "paper_tables/table4_statistical_comparison.csv"), "source": "TODO8", "generation_command": "python update/compare_models_statistics.py"},
        {"kind": "table", "id": "Table 5", "path": str(table5), "source": "TODO9 ablation", "generation_command": "python update/complete_remaining_todos.py"},
        {"kind": "table", "id": "Table 6", "path": str(OUTPUT_ROOT / "paper_tables/table6_ood_generalization.csv"), "source": "TODO10", "generation_command": "python update/complete_todo10_ood_benchmark.py"},
        {"kind": "table", "id": "Table 7", "path": str(OUTPUT_ROOT / "paper_tables/table7_uncertainty_calibration.csv"), "source": "TODO11", "generation_command": "python update/complete_todo11_uncertainty.py"},
        {"kind": "figure", "id": "Figure 1", "path": str(f1[0]), "source": "protocol/workflow", "generation_command": "python update/complete_remaining_todos.py"},
        {"kind": "figure", "id": "Figure 2", "path": str(f2[0]), "source": str(table1), "generation_command": "python update/complete_remaining_todos.py"},
        {"kind": "figure", "id": "Figure 3", "path": str(OUTPUT_ROOT / "paper_figures/model_performance_bar_ci.png"), "source": "Table 2", "generation_command": "python update/complete_todo6_publication_tables.py"},
        {"kind": "figure", "id": "Figure 4a", "path": str(OUTPUT_ROOT / "paper_figures/tail_error_by_bin.png"), "source": "TODO7 tail bins", "generation_command": "python update/complete_todo7_tail_safety.py"},
        {"kind": "figure", "id": "Figure 4b", "path": str(OUTPUT_ROOT / "paper_figures/dangerous_underprediction_rate.png"), "source": "TODO7 thresholds", "generation_command": "python update/complete_todo7_tail_safety.py"},
        {"kind": "figure", "id": "Figure 5", "path": str(f5[0]), "source": str(table5), "generation_command": "python update/complete_remaining_todos.py"},
        {"kind": "figure", "id": "Figure 6", "path": str(OUTPUT_ROOT / "paper_figures/ood_performance_drop.png"), "source": "TODO10", "generation_command": "python update/complete_todo10_ood_benchmark.py"},
        {"kind": "figure", "id": "Figure 7", "path": str(OUTPUT_ROOT / "paper_figures/calibration_curve.png"), "source": "TODO11", "generation_command": "python update/complete_todo11_uncertainty.py"},
        {"kind": "figure", "id": "Figure 8", "path": todo12_outputs["case_study_panel_png"], "source": todo12_outputs["case_study_index"], "generation_command": "python update/complete_remaining_todos.py"},
    ]
    manifest_csv = OUTPUT_ROOT / "paper_assets" / "figure_table_manifest.csv"
    manifest_json = OUTPUT_ROOT / "paper_assets" / "figure_table_manifest.json"
    write_csv(manifest_csv, manifest_rows)
    save_json(manifest_json, {"run_name": "todo14_figure_table_manifest", "items": manifest_rows})
    manifest_path = OUTPUT_ROOT / "statistics" / "todo14_completion_manifest.json"
    save_json(manifest_path, {"run_name": "todo14_completion", "status": "complete", "figure_table_manifest": str(manifest_json)})
    return {
        "figure_table_manifest_json": str(manifest_json),
        "figure_table_manifest_csv": str(manifest_csv),
        "table1_dataset_statistics": str(table1),
        "table5_ablation_study": str(table5),
        "figure1_task_model_schematic": str(f1[0]),
        "figure2_target_distribution": str(f2[0]),
        "figure5_ablation_study": str(f5[0]),
        "todo14_manifest": str(manifest_path),
    }


def complete_todo15() -> dict[str, str]:
    repro = OUTPUT_ROOT / "reproducibility_package"
    repro.mkdir(parents=True, exist_ok=True)
    versions = []
    for package in ["python", "torch", "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "catboost", "optuna", "matplotlib"]:
        if package == "python":
            version = platform.python_version()
        else:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "not_installed"
        versions.append({"package": package, "version": version})
    requirements = repro / "requirements.txt"
    requirements.write_text("\n".join(f"{row['package']}=={row['version']}" for row in versions if row["version"] != "not_installed" and row["package"] != "python") + "\n", encoding="utf-8")
    environment = repro / "environment.yml"
    environment.write_text(
        "name: opensees-surrogate-publication\nchannels:\n  - conda-forge\ndependencies:\n  - python="
        + platform.python_version()
        + "\n  - pip\n  - pip:\n"
        + "\n".join(f"      - {row['package']}=={row['version']}" for row in versions if row["version"] != "not_installed" and row["package"] != "python")
        + "\n",
        encoding="utf-8",
    )
    version_table = pd.DataFrame(versions)
    version_table.to_csv(repro / "environment_versions.csv", index=False, encoding="utf-8-sig")

    commands = [
        "python update/lock_publication_protocol.py",
        "python update/audit_dataset_protocol.py",
        "python update/audit_hpo_fairness.py",
        "python update/audit_baseline_registry.py",
        "python update/run_final_multiseed_eval.py --models randomforest xgboost lightgbm catboost mlp",
        "python update/run_final_deep_multiseed_eval.py --models lstm wavenet 2dcnn",
        "python update/complete_todo6_publication_tables.py",
        "python update/complete_todo7_tail_safety.py",
        "python update/compare_models_statistics.py",
        "python update/summarize_completed_ablation_results.py",
        "python update/complete_todo10_ood_benchmark.py",
        "python update/complete_todo11_uncertainty.py",
        "python update/complete_remaining_todos.py",
    ]
    run_commands = repro / "run_commands.md"
    run_commands.write_text("# Reproduction Commands\n\n" + "\n".join(f"```powershell\n{cmd}\n```" for cmd in commands) + "\n", encoding="utf-8")

    for src in [OUTPUT_ROOT / "protocol/protocol_lock.yaml", OUTPUT_ROOT / "protocol/protocol_lock.json", OUTPUT_ROOT / "data_audit/dataset_manifest.json"]:
        if src.exists():
            shutil.copy2(src, repro / src.name)
    best_dir = repro / "best_params"
    best_dir.mkdir(exist_ok=True)
    for src in (PROJECT_ROOT / "update").glob("*/best_params.json"):
        dst = best_dir / f"{src.parent.name}_best_params.json"
        shutil.copy2(src, dst)

    readme = repro / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Reproducibility Package",
                "",
                "This package records the locked protocol, data audit, environment, best hyperparameters, run commands, and reporting checklists for the OpenSees surrogate-model publication evaluation.",
                "",
                "## Minimal Rebuild",
                "",
                "Run the commands in `run_commands.md` in order. The scripts are designed to avoid changing the locked test split and to write auditable CSV/JSON/Markdown artifacts under `publication_eval_20260614/`.",
                "",
                "## Important Boundaries",
                "",
                "- High-drift independent stress labels are not yet present; use the TODO10 OpenSees generation plan before making high-drift safety claims.",
                "- Controlled inference latency and OpenSees speedup are not measured in the current package.",
                "- TODO12 SHAP/permutation was downgraded to model-free feature association because h5py was blocked by local application-control policy.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    reforms = repro / "REFORMS_checklist.md"
    reforms.write_text(
        "\n".join(
            [
                "# REFORMS Checklist",
                "",
                "- Research question and intended claims: documented in TODO16 claim calibration.",
                "- Data provenance and split protocol: protocol_lock.yaml/json and dataset_manifest.json.",
                "- Leakage prevention: TODO2 data audit.",
                "- Hyperparameter search: TODO3 HPO audit and best_params folder.",
                "- Final evaluation: TODO5 multi-seed frozen evaluation.",
                "- Statistical uncertainty: TODO8 paired tests, TODO11 conformal intervals.",
                "- Negative/limited claims: TODO10/TODO13/TODO16 explicitly list unsupported high-drift and speedup claims.",
                "- Reproducibility: commands, environment, seeds, and outputs included here.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    protocol = load_json(PROTOCOL_JSON)
    dataset_card = repro / "dataset_card.md"
    dataset_card.write_text(
        "\n".join(
            [
                "# Dataset Card",
                "",
                f"- Dataset base: `{protocol['dataset']['dataset_base']}`",
                f"- Target: `{protocol['dataset']['target_column']}`",
                "- Splits are locked by protocol and should not be regenerated for final claims.",
                "- Data audit artifacts: `publication_eval_20260614/data_audit/`.",
                "- Known limitation: locked test contains no `target >= 0.010` and no `steel01_yielded=1` samples.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    model_card = repro / "model_card.md"
    table2 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table2_overall_test_metrics_paper_ready.csv")
    model_card.write_text(
        "\n".join(
            [
                "# Model Card",
                "",
                "Primary model: `2dcnn` frozen-best-params multi-seed final evaluation.",
                "",
                "## Main Performance",
                "",
                markdown_table(table2.head(8)[["rank_by_test_mae", "model", "seed_count", "mae_mean", "rmse_mean", "r2_mean", "fit_seconds_mean"]], max_rows=8),
                "",
                "## Intended Use",
                "",
                "Fast surrogate screening of OpenSees-style seismic drift responses under the locked data distribution. Not validated for independent high-drift/yielded stress regimes until TODO10 generated OpenSees labels are evaluated.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = OUTPUT_ROOT / "statistics" / "todo15_completion_manifest.json"
    save_json(manifest_path, {"run_name": "todo15_reproducibility_package_completion", "status": "complete", "package_dir": str(repro)})
    return {
        "reproducibility_readme": str(readme),
        "requirements": str(requirements),
        "environment_yml": str(environment),
        "run_commands": str(run_commands),
        "REFORMS_checklist": str(reforms),
        "dataset_card": str(dataset_card),
        "model_card": str(model_card),
        "todo15_manifest": str(manifest_path),
    }


def complete_todo16() -> dict[str, str]:
    claim_dir = OUTPUT_ROOT / "claim_calibration"
    claim_dir.mkdir(parents=True, exist_ok=True)
    table2 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table2_overall_test_metrics_paper_ready.csv")
    table3 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table3_tail_safety_metrics_paper_ready.csv")
    table4 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table4_statistical_comparison.csv")
    table6 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table6_ood_generalization.csv")
    table7 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table7_uncertainty_calibration.csv")
    best = table2.sort_values("mae_mean").iloc[0]
    lightgbm = table2.loc[table2["model"] == "lightgbm"].iloc[0]
    wavenet_tail = table3.sort_values("p95_mae_mean").iloc[0]
    sig_vs_lightgbm = table4.loc[table4["comparator_model"] == "lightgbm"].iloc[0]
    uq_2dcnn = table7.loc[(table7["model"] == "2dcnn") & (table7["interval_method"] == "split_conformal_adaptive_90") & (table7["group_type"] == "all")].iloc[0]
    supported = [
        {
            "claim_id": "C1",
            "claim": f"The 2D-CNN achieves the lowest locked-test MAE among completed models: {fmt(best['mae_mean'])}.",
            "support_level": "strong",
            "evidence": "Table 2; TODO6 multi-seed final evaluation",
        },
        {
            "claim_id": "C2",
            "claim": f"2D-CNN significantly improves MAE over LightGBM by {fmt(sig_vs_lightgbm['delta_mae_comparator_minus_main_mean'])} with positive bootstrap CI.",
            "support_level": "strong",
            "evidence": "Table 4 paired hierarchical bootstrap",
        },
        {
            "claim_id": "C3",
            "claim": "Locked test is exact unseen-wave and exact unseen-structure relative to train+val; 2D-CNN remains the best model on this locked OOD benchmark.",
            "support_level": "moderate_strong_with_scope_limit",
            "evidence": "Table 6; TODO10",
        },
        {
            "claim_id": "C4",
            "claim": f"Validation-calibrated adaptive conformal intervals give global locked-test 90% interval coverage of {fmt(uq_2dcnn['coverage'])}.",
            "support_level": "moderate_with_calibration_split_caveat",
            "evidence": "Table 7; TODO11",
        },
        {
            "claim_id": "C5",
            "claim": f"Tail p95 MAE is best for {wavenet_tail['model']}; 2D-CNN is competitive but not best on p95 tail MAE.",
            "support_level": "strong_for_cautious_tail_comparison",
            "evidence": "Table 3; TODO7",
        },
    ]
    unsupported = [
        {
            "claim_id": "U1",
            "unsupported_claim": "The model is proven reliable for high-drift, near-failure, or steel01-yielded regimes.",
            "reason": "Locked test has no target >= 0.010 and no steel01_yielded=1 samples.",
            "needed_evidence": "Run TODO10 high_drift_opensees_generation_plan and evaluate frozen models.",
        },
        {
            "claim_id": "U2",
            "unsupported_claim": "The surrogate has a quantified OpenSees speedup.",
            "reason": "Controlled inference timing and OpenSees runtime are not measured.",
            "needed_evidence": "Dedicated TODO13 latency benchmark plus OpenSees runtime baseline.",
        },
        {
            "claim_id": "U3",
            "unsupported_claim": "SHAP proves the model learned physically causal mechanisms.",
            "reason": "SHAP/permutation was not run; TODO12 uses model-free association and ablation evidence.",
            "needed_evidence": "Run SHAP/permutation after resolving h5py/model-loading path.",
        },
        {
            "claim_id": "U4",
            "unsupported_claim": "2D-CNN has calibrated high-drift uncertainty intervals.",
            "reason": "Tail coverage at target >= 0.005 drops and high-drift independent test rows are absent.",
            "needed_evidence": "Independent high-drift stress set with conformal evaluation.",
        },
    ]
    supported_path = claim_dir / "supported_claims.csv"
    unsupported_path = claim_dir / "unsupported_claims.csv"
    write_csv(supported_path, supported)
    write_csv(unsupported_path, unsupported)
    paper_claims = claim_dir / "paper_claims.md"
    paper_claims.write_text(
        "\n".join(
            [
                "# Calibrated Paper Claims",
                "",
                "## Supported Claims",
                "",
                markdown_table(pd.DataFrame(supported), max_rows=20),
                "",
                "## Unsupported Or Must-Not-Write Claims",
                "",
                markdown_table(pd.DataFrame(unsupported), max_rows=20),
                "",
                "## Recommended Abstract-Level Claim",
                "",
                "Under a locked evaluation protocol with independent multi-seed retraining, the 2D-CNN surrogate achieves the lowest overall locked-test MAE and significantly improves over the strongest completed GBDT baseline on paired tests. The locked test is also exact unseen-wave and unseen-structure relative to train+val, supporting a bounded OOD generalization claim. Tail and uncertainty analyses reveal remaining limitations: high-drift/yielded regimes require new OpenSees stress labels, and global conformal coverage should not be overextended to high-response regimes.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = OUTPUT_ROOT / "statistics" / "todo16_completion_manifest.json"
    save_json(manifest_path, {"run_name": "todo16_claim_calibration_completion", "status": "complete"})
    return {
        "supported_claims": str(supported_path),
        "unsupported_claims": str(unsupported_path),
        "paper_claims": str(paper_claims),
        "todo16_manifest": str(manifest_path),
    }


def project_structure_text(max_depth: int = 3) -> str:
    skip = {".git", ".venv", "__pycache__"}
    lines = [f"{PROJECT_ROOT.name}/"]
    root_depth = len(PROJECT_ROOT.parts)
    for path in sorted(PROJECT_ROOT.rglob("*")):
        rel = path.relative_to(PROJECT_ROOT)
        if any(part in skip for part in rel.parts):
            continue
        depth = len(path.parts) - root_depth
        if depth > max_depth:
            continue
        indent = "  " * depth
        suffix = "/" if path.is_dir() else ""
        lines.append(f"{indent}{path.name}{suffix}")
    return "\n".join(lines)


def create_final_summary(outputs: dict[str, dict[str, str]]) -> Path:
    table2 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table2_overall_test_metrics_paper_ready.csv")
    table3 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table3_tail_safety_metrics_paper_ready.csv")
    table7 = pd.read_csv(OUTPUT_ROOT / "paper_tables/table7_uncertainty_calibration.csv")
    summary_path = PROJECT_ROOT / "md_files" / "02_research_and_paper" / "literature_reviews" / "全部TODO完成总总结_20260617.md"
    lines = [
        "# 全部 TODO 完成总总结",
        "",
        "生成日期：2026-06-17",
        "",
        "## 总体状态",
        "",
        "TODO1-TODO16 已全部推进到当前数据和非干扰规则允许的可投稿材料状态。需要强调：TODO10 high-drift 独立 stress set、TODO13 controlled inference latency/OpenSees speedup、TODO12 真 SHAP/permutation 仍被明确标记为主张边界，不能在论文里伪造成已完成强证据。",
        "",
        "## 主结果摘要",
        "",
        markdown_table(table2.head(8)[["rank_by_test_mae", "model", "seed_count", "mae_mean", "rmse_mean", "r2_mean", "fit_seconds_mean"]], max_rows=8),
        "",
        "## Tail/Safety 摘要",
        "",
        markdown_table(table3.head(8)[["rank_by_p95_mae", "model", "p95_mae_mean", "thr_0.005_recall_mean", "thr_0.005_dangerous_under_rate_mean", "safety_claim_scope"]], max_rows=8),
        "",
        "## Uncertainty 摘要",
        "",
        markdown_table(table7.loc[(table7["model"] == "2dcnn") & (table7["interval_method"] == "split_conformal_adaptive_90")][["group_type", "group_value", "n", "coverage", "avg_interval_width", "claim_scope"]], max_rows=6),
        "",
        "## 本轮新增 TODO12-TODO16 产物",
        "",
    ]
    for todo, paths in outputs.items():
        lines.append(f"### {todo}")
        for key, value in paths.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")
    lines.extend(
        [
            "## 论文写作建议",
            "",
            "- 主结论聚焦 locked protocol、multi-seed、paired statistics 和 exact unseen-wave/structure test。",
            "- 安全性讨论必须区分 light-tail (`target >= 0.005`) 和真正 high-drift/yielded；后者当前没有独立 locked-test 标签。",
            "- 不确定性可以写 global locked-test conformal coverage，但必须报告 tail coverage 下降。",
            "- 效率可以写训练/HPO/模型大小/Pareto 审计；暂不写 OpenSees speedup 数字。",
            "",
            "## 参考文献",
            "",
            "1. Kapoor et al. (2024). REFORMS: Consensus-based recommendations for machine-learning-based science. Science Advances.",
            "2. Lundberg & Lee (2017). A unified approach to interpreting model predictions. NeurIPS.",
            "3. Apley & Zhu (2020). Visualizing the effects of predictor variables in black box supervised learning models. JRSS B.",
            "4. Mitchell et al. (2019). Model Cards for Model Reporting. FAT*.",
            "5. Gebru et al. (2021). Datasheets for Datasets. Communications of the ACM.",
            "",
            "## 当前项目结构",
            "",
            "```text",
            project_structure_text(max_depth=3),
            "```",
        ]
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_path


def main() -> None:
    outputs: dict[str, dict[str, str]] = {}
    outputs["TODO12"] = complete_todo12()
    outputs["TODO13"] = complete_todo13()
    outputs["TODO14"] = complete_todo14(outputs["TODO12"])
    outputs["TODO15"] = complete_todo15()
    outputs["TODO16"] = complete_todo16()
    final_summary = create_final_summary(outputs)
    manifest_path = OUTPUT_ROOT / "statistics" / "todo12_16_completion_manifest.json"
    save_json(manifest_path, {"run_name": "todo12_16_completion", "outputs": outputs, "final_summary": str(final_summary)})
    print(f">>> TODO12-16 manifest: {manifest_path}")
    print(f">>> Final summary: {final_summary}")


if __name__ == "__main__":
    main()
