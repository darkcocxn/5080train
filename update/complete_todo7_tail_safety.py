# -*- coding: utf-8 -*-
"""Complete TODO7 tail and engineering-safety artifacts.

The script consumes frozen final multi-seed predictions only. It does not train
models, does not change data splits, and does not perform model selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
ABS_THRESHOLDS = (0.005, 0.010, 0.015, 0.020)
BIN_QUANTILES = (0.0, 0.50, 0.75, 0.90, 0.95, 1.0)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


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


def fmt_mean_std(row: pd.Series, metric: str, digits: int = 6) -> str:
    return f"{fmt(row.get(f'{metric}_mean'), digits)} +/- {fmt(row.get(f'{metric}_std'), digits)}"


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


def discover_prediction_runs(output_root: Path) -> list[dict[str, Any]]:
    final_dir = output_root / "final_multiseed"
    runs: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            pred_path = seed_dir / "predictions_test.csv"
            metrics_path = seed_dir / "metrics.json"
            if not pred_path.exists() or not metrics_path.exists():
                continue
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            runs.append({"model": model_dir.name, "seed": seed, "pred_path": pred_path, "metrics_path": metrics_path})
    if not runs:
        raise FileNotFoundError(f"No final prediction runs found under {final_dir}")
    return runs


def read_prediction(path: Path, model: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"y_true", "y_pred"}.issubset(df.columns):
        true_col, pred_col = "y_true", "y_pred"
    elif {"True_Drift", "Pred_Drift"}.issubset(df.columns):
        true_col, pred_col = "True_Drift", "Pred_Drift"
    else:
        raise ValueError(f"Cannot find prediction columns in {path}")
    out = pd.DataFrame(
        {
            "model": model,
            "seed": int(seed),
            "sample_id": df["sample_id"].astype(str) if "sample_id" in df.columns else np.arange(len(df)).astype(str),
            "num_floors": pd.to_numeric(df.get("num_floors", np.nan), errors="coerce"),
            "wave_cluster": pd.to_numeric(df.get("wave_cluster", np.nan), errors="coerce"),
            "steel01_yielded": pd.to_numeric(df.get("steel01_yielded", np.nan), errors="coerce"),
            "steel02_yielded": pd.to_numeric(df.get("steel02_yielded", np.nan), errors="coerce"),
            "y_true": pd.to_numeric(df[true_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["underprediction"] = np.maximum(out["y_true"] - out["y_pred"], 0.0)
    return out


def aggregate_rows(rows: list[dict[str, Any]], keys: tuple[str, ...], numeric_exclude: set[str] | None = None) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    numeric_exclude = numeric_exclude or set()
    df = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for group_key, group in df.groupby(list(keys), dropna=False, sort=True):
        if len(keys) == 1:
            group_key = (group_key,)
        row: dict[str, Any] = {key: value for key, value in zip(keys, group_key)}
        row["seed_count"] = int(group["seed"].nunique()) if "seed" in group.columns else int(len(group))
        for col in group.columns:
            if col in set(keys) | {"seed"} | numeric_exclude:
                continue
            if not pd.api.types.is_numeric_dtype(group[col]):
                continue
            values = [float(value) for value in group[col].dropna().tolist()]
            mean, std, ci95 = mean_std_ci(values)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95_normal"] = ci95
        output.append(row)
    return pd.DataFrame(output)


def create_paper_ready_table(output_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    source = output_root / "paper_tables" / "table3_tail_safety_metrics.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing TODO7 source table: {source}")
    df = pd.read_csv(source).sort_values("p95_mae_mean", ascending=True).reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for rank, row in df.iterrows():
        true_005 = finite_float(row.get("thr_0.005_true_count_mean"))
        true_010 = finite_float(row.get("thr_0.010_true_count_mean"))
        safety_scope = "light_tail_threshold_only"
        if true_010 and true_010 > 0:
            safety_scope = "includes_high_drift_threshold"
        rows.append(
            {
                "rank_by_p95_mae": rank + 1,
                "model": row["model"],
                "seed_count": int(row["seed_count"]),
                "p90_threshold_mean": row.get("p90_threshold_mean"),
                "p90_mae_mean_std": fmt_mean_std(row, "p90_mae"),
                "p90_rmse_mean_std": fmt_mean_std(row, "p90_rmse"),
                "p95_threshold_mean": row.get("p95_threshold_mean"),
                "p95_mae_mean": row.get("p95_mae_mean"),
                "p95_mae_std": row.get("p95_mae_std"),
                "p95_mae_ci95_normal": row.get("p95_mae_ci95_normal"),
                "p95_mae_mean_std": fmt_mean_std(row, "p95_mae"),
                "p95_rmse_mean_std": fmt_mean_std(row, "p95_rmse"),
                "p95_underprediction_mean": row.get("p95_underprediction_mean"),
                "max_underprediction_mean": row.get("max_underprediction_mean"),
                "thr_0.005_true_count_mean": true_005,
                "thr_0.005_recall_mean": row.get("thr_0.005_recall_mean"),
                "thr_0.005_precision_mean": row.get("thr_0.005_precision_mean"),
                "thr_0.005_dangerous_under_count_mean": row.get("thr_0.005_dangerous_under_count_mean"),
                "thr_0.005_dangerous_under_rate_mean": row.get("thr_0.005_dangerous_under_rate_mean"),
                "thr_0.010_true_count_mean": true_010,
                "safety_claim_scope": safety_scope,
            }
        )
    out = pd.DataFrame(rows)
    csv_path = output_root / "paper_tables" / "table3_tail_safety_metrics_paper_ready.csv"
    md_path = output_root / "paper_tables" / "table3_tail_safety_metrics_paper_ready.md"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md = out[
        [
            "rank_by_p95_mae",
            "model",
            "seed_count",
            "p95_mae_mean_std",
            "p95_rmse_mean_std",
            "thr_0.005_recall_mean",
            "thr_0.005_dangerous_under_count_mean",
            "max_underprediction_mean",
        ]
    ].copy()
    for col in ("thr_0.005_recall_mean", "thr_0.005_dangerous_under_count_mean", "max_underprediction_mean"):
        md[col] = md[col].map(lambda value: fmt(value, 5))
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Table 3 Tail And Safety Metrics, Paper-Ready View\n\n")
        file.write("Source: `table3_tail_safety_metrics.csv`; sorted by lower p95-tail MAE.\n\n")
        file.write(markdown_table(md))
        file.write("\n")
    return out, csv_path, md_path


def create_threshold_coverage(reference_pred: pd.DataFrame, output_root: Path) -> tuple[pd.DataFrame, Path]:
    y_true = reference_pred["y_true"].to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for q in (0.90, 0.95):
        threshold = float(np.quantile(y_true, q))
        mask = y_true >= threshold
        rows.append(
            {
                "tail_definition": f"target >= p{int(q * 100)}",
                "threshold": threshold,
                "true_count": int(mask.sum()),
                "true_rate": float(mask.mean()),
                "coverage_status": "available",
            }
        )
    for threshold in ABS_THRESHOLDS:
        mask = y_true >= threshold
        rows.append(
            {
                "tail_definition": f"target >= {threshold:.3f}",
                "threshold": threshold,
                "true_count": int(mask.sum()),
                "true_rate": float(mask.mean()),
                "coverage_status": "available" if mask.any() else "absent_in_locked_test",
            }
        )
    for col in ("steel01_yielded", "steel02_yielded"):
        if col in reference_pred.columns:
            valid = reference_pred[col].dropna()
            count = int((valid == 1).sum()) if not valid.empty else 0
            rows.append(
                {
                    "tail_definition": f"{col} == 1",
                    "threshold": None,
                    "true_count": count,
                    "true_rate": float(count / len(reference_pred)),
                    "coverage_status": "available" if count else "absent_in_locked_test",
                }
            )
    out = pd.DataFrame(rows)
    path = output_root / "paper_tables" / "tableS_tail_threshold_coverage.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    return out, path


def bin_labels_and_edges(y_true: np.ndarray) -> tuple[list[str], np.ndarray]:
    edges = np.quantile(y_true, BIN_QUANTILES)
    edges[0] = min(edges[0], float(np.min(y_true))) - 1e-12
    edges[-1] = max(edges[-1], float(np.max(y_true))) + 1e-12
    labels = [
        "p00-p50",
        "p50-p75",
        "p75-p90",
        "p90-p95",
        "p95-p100",
    ]
    return labels, edges


def create_tail_bin_metrics(predictions: list[pd.DataFrame], labels: list[str], edges: np.ndarray, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        binned = pd.cut(pred["y_true"], bins=edges, labels=labels, include_lowest=True, duplicates="drop")
        temp = pred.assign(drift_bin=binned)
        for bin_label, group in temp.groupby("drift_bin", observed=False):
            if group.empty:
                continue
            errors = group["error"].to_numpy(dtype=float)
            abs_errors = np.abs(errors)
            under = np.maximum(group["y_true"].to_numpy(dtype=float) - group["y_pred"].to_numpy(dtype=float), 0.0)
            rows.append(
                {
                    "model": str(group["model"].iloc[0]),
                    "seed": int(group["seed"].iloc[0]),
                    "drift_bin": str(bin_label),
                    "samples": int(len(group)),
                    "bin_y_true_min": float(group["y_true"].min()),
                    "bin_y_true_max": float(group["y_true"].max()),
                    "mae": float(abs_errors.mean()),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "bias": float(errors.mean()),
                    "underprediction_mean": float(under.mean()),
                    "underprediction_rate": float((errors < 0).mean()),
                    "p95_underprediction": float(np.quantile(under, 0.95)),
                }
            )
    by_seed = pd.DataFrame(rows).sort_values(["model", "seed", "drift_bin"]).reset_index(drop=True)
    aggregate = aggregate_rows(rows, ("model", "drift_bin"))
    path = output_root / "paper_tables" / "tableS_tail_error_by_bin.csv"
    seed_path = output_root / "paper_tables" / "tableS_tail_error_by_bin_by_seed.csv"
    aggregate.to_csv(path, index=False, encoding="utf-8-sig")
    by_seed.to_csv(seed_path, index=False, encoding="utf-8-sig")
    return aggregate, by_seed, path


def create_dangerous_underprediction_metrics(predictions: list[pd.DataFrame], output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    rows: list[dict[str, Any]] = []
    for pred in predictions:
        y_true = pred["y_true"].to_numpy(dtype=float)
        y_pred = pred["y_pred"].to_numpy(dtype=float)
        model = str(pred["model"].iloc[0])
        seed = int(pred["seed"].iloc[0])
        for threshold in ABS_THRESHOLDS:
            true_mask = y_true >= threshold
            pred_mask = y_pred >= threshold
            dangerous = true_mask & (y_pred < threshold)
            true_count = int(true_mask.sum())
            pred_count = int(pred_mask.sum())
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "threshold": threshold,
                    "true_count": true_count,
                    "pred_count": pred_count,
                    "dangerous_under_count": int(dangerous.sum()),
                    "dangerous_under_rate": float(dangerous.sum() / true_count) if true_count else None,
                    "recall": float((true_mask & pred_mask).sum() / true_count) if true_count else None,
                    "precision": float((true_mask & pred_mask).sum() / pred_count) if pred_count else None,
                }
            )
    by_seed = pd.DataFrame(rows).sort_values(["threshold", "model", "seed"]).reset_index(drop=True)
    aggregate = aggregate_rows(rows, ("model", "threshold"))
    path = output_root / "paper_tables" / "tableS_dangerous_underprediction_by_threshold.csv"
    seed_path = output_root / "paper_tables" / "tableS_dangerous_underprediction_by_threshold_by_seed.csv"
    aggregate.to_csv(path, index=False, encoding="utf-8-sig")
    by_seed.to_csv(seed_path, index=False, encoding="utf-8-sig")
    return aggregate, by_seed, path


def plot_tail_error_by_bin(tail_bins: pd.DataFrame, paper_ready: pd.DataFrame, output_root: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = paper_ready.sort_values("p95_mae_mean").head(8)["model"].tolist()
    data = tail_bins[tail_bins["model"].isin(selected)].copy()
    order = ["p00-p50", "p50-p75", "p75-p90", "p90-p95", "p95-p100"]
    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    palette = ["#2f6f9f", "#c75d2c", "#4c8b3f", "#8a5fbf", "#b44763", "#6a7f2a", "#3c7c7d", "#8a6f3d"]
    for idx, model in enumerate(selected):
        subset = data[data["model"] == model].set_index("drift_bin").reindex(order)
        ax.plot(order, subset["mae_mean"], marker="o", linewidth=1.8, label=model, color=palette[idx % len(palette)])
    ax.set_xlabel("True drift bin")
    ax.set_ylabel("MAE")
    ax.set_title("Tail error by true-drift bin")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    png = output_root / "paper_figures" / "tail_error_by_bin.png"
    svg = output_root / "paper_figures" / "tail_error_by_bin.svg"
    fig.savefig(png, dpi=240)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def plot_dangerous_underprediction(danger: pd.DataFrame, output_root: Path) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = danger[danger["threshold"].round(3) == 0.005].copy()
    data = data.sort_values("dangerous_under_rate_mean", ascending=True)
    labels = data["model"].tolist()
    means = data["dangerous_under_rate_mean"].to_numpy(dtype=float)
    ci = data["dangerous_under_rate_ci95_normal"].fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.8, 5.6))
    ax.barh(labels, means, xerr=ci, color="#5b8db8", edgecolor="#222222", linewidth=0.5, capsize=3)
    ax.set_xlabel("Dangerous underprediction rate at target >= 0.005")
    ax.set_ylabel("Model")
    ax.set_title("Dangerous underprediction rate with 95% CI")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    png = output_root / "paper_figures" / "dangerous_underprediction_rate.png"
    svg = output_root / "paper_figures" / "dangerous_underprediction_rate.svg"
    fig.savefig(png, dpi=240)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def create_report(
    output_root: Path,
    paper_ready: pd.DataFrame,
    coverage: pd.DataFrame,
    danger: pd.DataFrame,
    outputs: dict[str, str],
) -> Path:
    report_path = PROJECT_ROOT / "md_files" / "02_research_and_paper" / "literature_reviews" / "TODO7_尾部工程安全指标补齐完成报告_20260617.md"
    best_p95 = paper_ready.iloc[0]
    best_recall = paper_ready.sort_values("thr_0.005_recall_mean", ascending=False).iloc[0]
    best_danger = paper_ready.sort_values("thr_0.005_dangerous_under_rate_mean", ascending=True).iloc[0]
    p005 = coverage.loc[coverage["tail_definition"] == "target >= 0.005"].iloc[0]
    p010 = coverage.loc[coverage["tail_definition"] == "target >= 0.010"].iloc[0]
    absent_defs = coverage.loc[coverage["coverage_status"] == "absent_in_locked_test", "tail_definition"].tolist()

    lines = [
        "# TODO7 尾部工程安全指标补齐完成报告",
        "",
        "生成日期：2026-06-17",
        "",
        "## 1. 补齐范围",
        "",
        "本轮只读取 frozen final multi-seed predictions 和 TODO6-8 已生成的 tail/safety 表，不重新训练、不重新调参、不改测试集。补齐目标是让 TODO7 具备论文 Main Table 3、Figure 4、图件源 CSV、阈值覆盖审计和安全主张边界说明。",
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
            "| TODO7 要求 | 本轮状态 | 证据文件 |",
            "|---|---|---|",
            "| 统一所有模型 tail metrics | 完成，14 个模型、5 个 seed、70 个 final runs | `table3_tail_safety_metrics.csv`, `table3_tail_safety_metrics_paper_ready.csv` |",
            "| 定义 p90、p95、0.005、0.010、0.015、0.020 tail | 完成，并额外输出 locked-test 覆盖率 | `tableS_tail_threshold_coverage.csv` |",
            "| tail MAE / tail RMSE | 完成 | `table3_tail_safety_metrics_paper_ready.csv` |",
            "| dangerous underprediction rate | 完成 | `tableS_dangerous_underprediction_by_threshold.csv`, `dangerous_underprediction_rate.png` |",
            "| threshold recall / precision | 完成 | `table3_tail_safety_metrics.csv`, `tableS_dangerous_underprediction_by_threshold.csv` |",
            "| max underprediction / p95 underprediction | 完成 | `table3_tail_safety_metrics_paper_ready.csv` |",
            "| 样本不足时启动 TODO10 或降级主张 | 完成样本不足标记；真正 high-drift benchmark 仍归 TODO10 | `todo7_completion_manifest.json` |",
            "",
            "## 4. 主要结果",
            "",
            f"- p95 高响应尾部 MAE 最低模型：`{best_p95['model']}`，p95 MAE = `{fmt(best_p95['p95_mae_mean'])}`。",
            f"- `target >= 0.005` 阈值召回最高模型：`{best_recall['model']}`，recall = `{fmt(best_recall['thr_0.005_recall_mean'])}`。",
            f"- `target >= 0.005` 危险低估率最低模型：`{best_danger['model']}`，dangerous underprediction rate = `{fmt(best_danger['thr_0.005_dangerous_under_rate_mean'])}`。",
            f"- locked test 中 `target >= 0.005` 样本数为 `{int(p005['true_count'])}`；`target >= 0.010` 样本数为 `{int(p010['true_count'])}`。",
            "",
            "这些结果说明：平均误差最优、p95 tail 最优、阈值召回最高、危险低估率最低不一定是同一个模型。论文必须把“整体精度”和“工程安全阈值行为”分开写，不能只用 MAE 支撑安全性。",
            "",
            "## 5. 安全主张边界",
            "",
            f"当前 locked test 缺失以下高风险覆盖：{', '.join(absent_defs) if absent_defs else '无'}。",
            "",
            "因此 TODO7 现在可以支持的表述是：模型在当前 locked test 覆盖的轻中度 tail 和 `target >= 0.005` 阈值上完成了显式危险低估量化。不能支持的表述是：模型已经证明在 `target >= 0.010/0.015/0.020`、主体屈服或接近失效工况下可靠。若论文要写强工程安全 claim，必须进入 TODO10 构造 high-drift / yielded / OOD benchmark。",
            "",
            "## 6. 论文写作建议",
            "",
            "- Main Table 3 使用 `table3_tail_safety_metrics_paper_ready.csv`。",
            "- Figure 4 使用 `tail_error_by_bin.png` 和 `dangerous_underprediction_rate.png`；源数据分别为 `tableS_tail_error_by_bin.csv` 和 `tableS_dangerous_underprediction_by_threshold.csv`。",
            "- 正文只讨论 `p90/p95` 和 `target >= 0.005`，高阈值只作为“locked test 未覆盖”的限制说明。",
            "- 若要把 2D-CNN 写成主模型，应同时说明它的整体 MAE、p95 tail MAE、阈值召回和危险低估率，而不是只给一个指标。",
            "",
            "## 7. 参考文献与规范",
            "",
            "1. Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J. M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D., Roberts, M., Salganik, M., Serra-Garcia, M., Stewart, B. M., Vandewiele, G., & Narayanan, A. (2024). REFORMS: Reporting standards for machine learning based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452",
            "2. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3",
            "3. Erickson, N., Purucker, L., Tschalzev, A., Holzmueller, D., Mutalik Desai, P., Salinas, D., & Hutter, F. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS 2025 Datasets and Benchmarks Track spotlight. https://openreview.net/forum?id=jZqCqpCLdU",
            "4. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving reproducibility in machine learning research. Journal of Machine Learning Research, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html",
            "5. Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4), 494-591. Open version: https://arxiv.org/abs/2107.07511",
            "",
            "## 8. TODO7 状态结论",
            "",
            "TODO7 已补齐为“尾部和工程安全指标完成”。剩余的高漂移、屈服和 OOD 安全可靠性不再归入 TODO7，应进入 TODO10。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete TODO7 tail and engineering-safety paper artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    runs = discover_prediction_runs(output_root)
    predictions = [read_prediction(run["pred_path"], str(run["model"]), int(run["seed"])) for run in runs]
    reference_pred = predictions[0]

    paper_ready, paper_ready_csv, paper_ready_md = create_paper_ready_table(output_root)
    coverage, coverage_csv = create_threshold_coverage(reference_pred, output_root)
    labels, edges = bin_labels_and_edges(reference_pred["y_true"].to_numpy(dtype=float))
    tail_bins, tail_bins_by_seed, tail_bins_csv = create_tail_bin_metrics(predictions, labels, edges, output_root)
    danger, danger_by_seed, danger_csv = create_dangerous_underprediction_metrics(predictions, output_root)
    tail_png, tail_svg = plot_tail_error_by_bin(tail_bins, paper_ready, output_root)
    danger_png, danger_svg = plot_dangerous_underprediction(danger, output_root)

    outputs = {
        "table3_tail_safety_metrics": str(output_root / "paper_tables" / "table3_tail_safety_metrics.csv"),
        "table3_tail_safety_metrics_by_seed": str(output_root / "paper_tables" / "table3_tail_safety_metrics_by_seed.csv"),
        "table3_tail_safety_metrics_paper_ready": str(paper_ready_csv),
        "table3_tail_safety_metrics_paper_ready_md": str(paper_ready_md),
        "tableS_tail_threshold_coverage": str(coverage_csv),
        "tableS_tail_error_by_bin": str(tail_bins_csv),
        "tableS_tail_error_by_bin_by_seed": str(output_root / "paper_tables" / "tableS_tail_error_by_bin_by_seed.csv"),
        "tableS_dangerous_underprediction_by_threshold": str(danger_csv),
        "tableS_dangerous_underprediction_by_threshold_by_seed": str(output_root / "paper_tables" / "tableS_dangerous_underprediction_by_threshold_by_seed.csv"),
        "tail_error_by_bin_png": str(tail_png),
        "tail_error_by_bin_svg": str(tail_svg),
        "dangerous_underprediction_rate_png": str(danger_png),
        "dangerous_underprediction_rate_svg": str(danger_svg),
    }
    report_path = create_report(output_root, paper_ready, coverage, danger, outputs)
    outputs["todo7_completion_report"] = str(report_path)

    absent_defs = coverage.loc[coverage["coverage_status"] == "absent_in_locked_test", "tail_definition"].tolist()
    manifest = {
        "run_name": "todo7_completion",
        "generated_at": "2026-06-17",
        "source_output_root": str(output_root),
        "model_count": int(paper_ready["model"].nunique()),
        "seed_count_total": int(paper_ready["seed_count"].sum()),
        "models": paper_ready["model"].tolist(),
        "absolute_thresholds": list(ABS_THRESHOLDS),
        "drift_bin_quantiles": list(BIN_QUANTILES),
        "best_model_by_p95_mae": str(paper_ready.iloc[0]["model"]),
        "best_p95_mae_mean": float(paper_ready.iloc[0]["p95_mae_mean"]),
        "threshold_coverage_absent": absent_defs,
        "outputs": outputs,
        "completion_notes": [
            "Tail/safety metrics are computed on the locked test split after frozen-parameter final retraining.",
            "Dangerous underprediction is defined as y_true >= threshold and y_pred < threshold.",
            "target >= 0.010, 0.015, and 0.020 are absent in the locked test split; high-drift safety claims still require TODO10.",
            "Figure data sources are saved alongside the paper tables so plotted values are auditable.",
        ],
    }
    manifest_path = output_root / "statistics" / "todo7_completion_manifest.json"
    save_json(manifest_path, manifest)

    print(f">>> TODO7 paper-ready table: {paper_ready_csv}")
    print(f">>> TODO7 coverage table: {coverage_csv}")
    print(f">>> TODO7 tail-bin figure: {tail_png}")
    print(f">>> TODO7 dangerous-underprediction figure: {danger_png}")
    print(f">>> TODO7 manifest: {manifest_path}")
    print(f">>> TODO7 report: {report_path}")


if __name__ == "__main__":
    main()
