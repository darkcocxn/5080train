# -*- coding: utf-8 -*-
"""Complete TODO6 publication-ready overall-performance artifacts.

This script is intentionally read-only with respect to model runs. It consumes
the frozen final multi-seed evaluation artifacts and creates compact paper
tables, a runtime/complexity audit table, and a completion manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
MODEL_ARTIFACT_SUFFIXES = {".pkl", ".joblib", ".pickle", ".pth", ".pt", ".onnx", ".cbm", ".model"}
MODEL_ARTIFACT_NAME_TOKENS = ("best_", "_model", "model.", "weights", "checkpoint")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    ci95 = 1.96 * std / math.sqrt(len(clean))
    return mean, std, ci95


def fmt(value: Any, digits: int = 6) -> str:
    number = finite_float(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}g}"


def fmt_mean_std(row: pd.Series, metric: str, digits: int = 6) -> str:
    return f"{fmt(row.get(f'{metric}_mean'), digits)} +/- {fmt(row.get(f'{metric}_std'), digits)}"


def fmt_mean_ci(row: pd.Series, metric: str, digits: int = 6) -> str:
    mean = finite_float(row.get(f"{metric}_mean"))
    ci = finite_float(row.get(f"{metric}_ci95_normal"))
    if mean is None or ci is None:
        return "NA"
    return f"{mean:.{digits}g} [{mean - ci:.{digits}g}, {mean + ci:.{digits}g}]"


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


def create_paper_ready_table(output_root: Path) -> tuple[pd.DataFrame, Path, Path]:
    source = output_root / "paper_tables" / "table2_overall_test_metrics.csv"
    if not source.exists():
        raise FileNotFoundError(f"Missing TODO6 source table: {source}")

    df = pd.read_csv(source)
    required = {"model", "seed_count", "mae_mean", "mae_std", "mae_ci95_normal", "rmse_mean", "r2_mean"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    df = df.sort_values("mae_mean", ascending=True).reset_index(drop=True)
    best_mae = float(df.loc[0, "mae_mean"])
    rows: list[dict[str, Any]] = []
    for rank, row in df.iterrows():
        mae_mean = finite_float(row.get("mae_mean"))
        rel = None if mae_mean is None or best_mae <= 0 else (mae_mean / best_mae - 1.0) * 100.0
        rows.append(
            {
                "rank_by_test_mae": rank + 1,
                "model": row["model"],
                "seed_count": int(row["seed_count"]),
                "test_samples_mean": row.get("samples_mean"),
                "mae_mean": row.get("mae_mean"),
                "mae_std": row.get("mae_std"),
                "mae_ci95_normal": row.get("mae_ci95_normal"),
                "mae_mean_std": fmt_mean_std(row, "mae"),
                "mae_mean_ci95": fmt_mean_ci(row, "mae"),
                "rmse_mean": row.get("rmse_mean"),
                "rmse_std": row.get("rmse_std"),
                "rmse_ci95_normal": row.get("rmse_ci95_normal"),
                "rmse_mean_std": fmt_mean_std(row, "rmse"),
                "r2_mean": row.get("r2_mean"),
                "r2_std": row.get("r2_std"),
                "r2_ci95_normal": row.get("r2_ci95_normal"),
                "r2_mean_std": fmt_mean_std(row, "r2"),
                "mape_mean": row.get("mape_mean"),
                "smape_mean": row.get("smape_mean"),
                "bias_mean": row.get("bias_mean"),
                "fit_seconds_mean": row.get("fit_seconds_mean"),
                "fit_seconds_std": row.get("fit_seconds_std"),
                "fit_hours_mean": None if finite_float(row.get("fit_seconds_mean")) is None else float(row["fit_seconds_mean"]) / 3600.0,
                "relative_mae_vs_best_pct": rel,
            }
        )

    out = pd.DataFrame(rows)
    csv_path = output_root / "paper_tables" / "table2_overall_test_metrics_paper_ready.csv"
    md_path = output_root / "paper_tables" / "table2_overall_test_metrics_paper_ready.md"
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_cols = [
        "rank_by_test_mae",
        "model",
        "seed_count",
        "mae_mean_std",
        "mae_mean_ci95",
        "rmse_mean_std",
        "r2_mean_std",
        "fit_seconds_mean",
    ]
    md = out[md_cols].copy()
    md["fit_seconds_mean"] = md["fit_seconds_mean"].map(lambda value: fmt(value, 5))
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Table 2 Overall Test Metrics, Paper-Ready View\n\n")
        file.write("Source: `table2_overall_test_metrics.csv`; sorted by lower locked-test MAE.\n\n")
        file.write(markdown_table(md))
        file.write("\n")
    return out, csv_path, md_path


def is_model_artifact(path: Path) -> bool:
    lower = path.name.lower()
    if path.suffix.lower() not in MODEL_ARTIFACT_SUFFIXES:
        return False
    return any(token in lower for token in MODEL_ARTIFACT_NAME_TOKENS)


def find_training_metadata(seed_dir: Path) -> dict[str, Any] | None:
    direct = seed_dir / "training_metadata.json"
    if direct.exists():
        return load_json(direct)
    candidates = sorted(seed_dir.rglob("training_metadata.json"))
    if candidates:
        return load_json(candidates[0])
    return None


def summarize_artifacts(seed_dir: Path) -> tuple[int, int, str]:
    files = [path for path in seed_dir.rglob("*") if path.is_file() and is_model_artifact(path)]
    total_bytes = sum(path.stat().st_size for path in files)
    names = ";".join(sorted(path.name for path in files)[:12])
    return len(files), total_bytes, names


def create_complexity_tables(output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path]:
    final_dir = output_root / "final_multiseed"
    if not final_dir.exists():
        raise FileNotFoundError(f"Missing final multi-seed directory: {final_dir}")

    seed_rows: list[dict[str, Any]] = []
    for model_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            metrics_path = seed_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics = load_json(metrics_path)
            seed = int(seed_dir.name.replace("seed_", ""))
            metadata = find_training_metadata(seed_dir) or {}
            artifact_count, artifact_bytes, artifact_names = summarize_artifacts(seed_dir)
            test_metrics = metrics.get("test_metrics", {})
            seed_rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "fit_seconds": finite_float(metrics.get("fit_seconds")),
                    "fit_hours": None if finite_float(metrics.get("fit_seconds")) is None else float(metrics["fit_seconds"]) / 3600.0,
                    "test_samples": finite_float(test_metrics.get("Samples", test_metrics.get("samples"))),
                    "total_params": finite_float(metadata.get("total_params")),
                    "trainable_params": finite_float(metadata.get("trainable_params")),
                    "peak_memory_mib": finite_float(metadata.get("peak_memory_mib")),
                    "model_artifact_count": artifact_count,
                    "model_artifact_mb": artifact_bytes / (1024.0 * 1024.0),
                    "model_artifact_names": artifact_names,
                    "batch_size": finite_float(metrics.get("batch_size", metadata.get("batch_size"))),
                    "inference_seconds": finite_float(metrics.get("inference_seconds")),
                    "inference_seconds_per_sample": finite_float(metrics.get("inference_seconds_per_sample")),
                    "inference_record_status": "recorded" if finite_float(metrics.get("inference_seconds")) is not None else "not_recorded_in_final_eval",
                    "metadata_status": "available" if metadata else "not_recorded",
                    "artifact_status": "available" if artifact_count else "not_saved_or_not_detected",
                }
            )

    if not seed_rows:
        raise FileNotFoundError(f"No seed metrics found under {final_dir}")

    by_seed = pd.DataFrame(seed_rows).sort_values(["model", "seed"]).reset_index(drop=True)
    aggregate_rows: list[dict[str, Any]] = []
    numeric_cols = [
        "fit_seconds",
        "fit_hours",
        "test_samples",
        "total_params",
        "trainable_params",
        "peak_memory_mib",
        "model_artifact_count",
        "model_artifact_mb",
        "batch_size",
        "inference_seconds",
        "inference_seconds_per_sample",
    ]
    for model, group in by_seed.groupby("model", sort=True):
        row: dict[str, Any] = {"model": model, "seed_count": int(group["seed"].nunique())}
        for col in numeric_cols:
            values = [float(value) for value in group[col].dropna().tolist()]
            mean, std, ci95 = mean_std_ci(values)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_ci95_normal"] = ci95
        row["inference_record_status"] = (
            "recorded" if (group["inference_record_status"] == "recorded").all() else "not_recorded_in_final_eval"
        )
        row["parameter_record_status"] = "available" if group["trainable_params"].notna().any() else "not_recorded"
        row["artifact_record_status"] = "available" if (group["model_artifact_count"] > 0).any() else "not_saved_or_not_detected"
        row["artifact_names_example"] = next((name for name in group["model_artifact_names"].tolist() if name), "")
        aggregate_rows.append(row)

    aggregate = pd.DataFrame(aggregate_rows)
    metric_path = output_root / "paper_tables" / "tableS_complexity_runtime.csv"
    seed_path = output_root / "paper_tables" / "tableS_complexity_runtime_by_seed.csv"
    aggregate.to_csv(metric_path, index=False, encoding="utf-8-sig")
    by_seed.to_csv(seed_path, index=False, encoding="utf-8-sig")
    return aggregate, by_seed, metric_path, seed_path


def create_completion_report(
    output_root: Path,
    paper_ready: pd.DataFrame,
    complexity: pd.DataFrame,
    outputs: dict[str, str],
) -> Path:
    report_path = PROJECT_ROOT / "md_files" / "02_research_and_paper" / "literature_reviews" / "TODO6_论文主指标表补齐完成报告_20260617.md"
    best = paper_ready.iloc[0]
    deep_models = {"2dcnn", "lstm", "wavenet", "mlp"}
    classical = [model for model in paper_ready["model"].tolist() if model not in deep_models]
    missing_inference = complexity.loc[complexity["inference_record_status"] != "recorded", "model"].tolist()
    missing_params = complexity.loc[complexity["parameter_record_status"] != "available", "model"].tolist()

    lines = [
        "# TODO6 论文主指标表补齐完成报告",
        "",
        "生成日期：2026-06-17",
        "",
        "## 1. 补齐范围",
        "",
        "本轮只消费 frozen final multi-seed evaluation 的既有产物，不重新训练、不重新调参、不改动测试集。补齐目标是把 TODO6 从“已有 70-run 原始汇总表”推进到“论文可直接使用的 Table 2、Supplementary seed table、group table、复杂度/运行时间审计表和 manifest”。",
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
            "| TODO6 要求 | 本轮状态 | 证据文件 |",
            "|---|---|---|",
            f"| 汇总所有模型、所有 seed 的测试结果 | 完成，{len(paper_ready)} 个模型，{int(paper_ready['seed_count'].sum())} 个 model-seed 汇总项 | `table2_overall_test_metrics.csv`, `tableS_seed_metrics.csv` |",
            "| MAE、RMSE、R2、MAPE/SMAPE | 完成 | `table2_overall_test_metrics.csv`, `table2_overall_test_metrics_paper_ready.csv` |",
            "| 分组指标 | 完成，按现有预测文件中的 `num_floors`、`wave_cluster`、`steel01_yielded`、`steel02_yielded` 输出 | `tableS_group_metrics.csv`, `tableS_group_metrics_by_seed.csv` |",
            "| mean、std、95% CI | 完成 | `table2_overall_test_metrics.csv`, `table2_overall_test_metrics_paper_ready.csv` |",
            "| 参数量、训练时间、推理时间 | 训练时间和可得参数量/模型大小已审计；最终评估未单独记录真实 inference latency 的模型已显式标记，不做速度主张 | `tableS_complexity_runtime.csv` |",
            "",
            "## 4. Table 2 主结果",
            "",
            f"当前 locked test 主指标排名第一的是 `{best['model']}`，MAE = `{fmt(best['mae_mean'])} +/- {fmt(best['mae_std'])}`，95% CI = `{best['mae_mean_ci95']}`。该结论基于冻结参数后的独立多 seed 最终评估，而不是 Optuna 验证集最优值。",
            "",
            "可直接用于论文正文的表为 `table2_overall_test_metrics_paper_ready.csv`；完整原始数值仍保留在 `table2_overall_test_metrics.csv`，便于附录或复查。",
            "",
            "## 5. 复杂度与运行时间边界",
            "",
            f"- 已记录训练时间的模型数：{int(complexity['fit_seconds_mean'].notna().sum())}/{len(complexity)}。",
            f"- 已记录 trainable parameter count 的模型：{', '.join(complexity.loc[complexity['parameter_record_status'] == 'available', 'model'].tolist()) or '无'}。",
            f"- 未保存或未检测到模型 artifact 的模型：{', '.join(complexity.loc[complexity['artifact_record_status'] != 'available', 'model'].tolist()) or '无'}。",
            f"- 未在最终评估 metrics 中单独记录 inference latency 的模型：{', '.join(missing_inference) if missing_inference else '无'}。",
            "",
            "因此，TODO6 现在可以支撑“多 seed 主指标 + 训练成本透明报告”；但论文中暂时不能写“推理延迟显著更低/更高”这类强速度主张。若需要完整部署价值，应在 TODO13 用统一 batch size、统一硬件、warm-up、重复计时和置信区间重新做 inference benchmark。",
            "",
            "## 6. 写作建议",
            "",
            "- 正文 Table 2 使用 `table2_overall_test_metrics_paper_ready.csv`，保留 MAE、RMSE、R2、seed count、95% CI。",
            "- 附录放 `tableS_seed_metrics.csv`，证明不是单 seed cherry-picking。",
            "- 附录放 `tableS_group_metrics.csv`，说明分组误差，但不要把它写成 unseen-wave 或 unseen-structure 泛化。",
            "- 附录放 `tableS_complexity_runtime.csv`，把未记录 inference latency 的事实写清楚。",
            "- 高漂移/屈服可靠性仍不能由 TODO6 支撑，必须等待 TODO10。",
            "",
            "## 7. 参考文献与规范",
            "",
            "1. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving reproducibility in machine learning research. Journal of Machine Learning Research, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html",
            "2. Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in ML-based science. Patterns, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804",
            "3. Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J. M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D., Roberts, M., Salganik, M., Serra-Garcia, M., Stewart, B. M., Vandewiele, G., & Narayanan, A. (2024). REFORMS: Reporting standards for machine learning based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452",
            "4. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3",
            "5. Erickson, N., Purucker, L., Tschalzev, A., Holzmueller, D., Mutalik Desai, P., Salinas, D., & Hutter, F. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS 2025 Datasets and Benchmarks Track spotlight. https://openreview.net/forum?id=jZqCqpCLdU",
            "",
            "## 8. TODO6 状态结论",
            "",
            "TODO6 已补齐为“论文主指标表完成”。剩余的真正效率专题不再归入 TODO6，而应进入 TODO13；安全/OOD 主张仍进入 TODO10。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete TODO6 paper-ready overall metrics and complexity tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    paper_ready, paper_ready_csv, paper_ready_md = create_paper_ready_table(output_root)
    complexity, complexity_by_seed, complexity_csv, complexity_seed_csv = create_complexity_tables(output_root)

    outputs = {
        "table2_overall_test_metrics": str(output_root / "paper_tables" / "table2_overall_test_metrics.csv"),
        "table2_overall_test_metrics_paper_ready": str(paper_ready_csv),
        "table2_overall_test_metrics_paper_ready_md": str(paper_ready_md),
        "tableS_seed_metrics": str(output_root / "paper_tables" / "tableS_seed_metrics.csv"),
        "tableS_group_metrics": str(output_root / "paper_tables" / "tableS_group_metrics.csv"),
        "tableS_complexity_runtime": str(complexity_csv),
        "tableS_complexity_runtime_by_seed": str(complexity_seed_csv),
        "model_performance_bar_ci": str(output_root / "paper_figures" / "model_performance_bar_ci.png"),
    }
    report_path = create_completion_report(output_root, paper_ready, complexity, outputs)
    outputs["todo6_completion_report"] = str(report_path)

    manifest = {
        "run_name": "todo6_completion",
        "generated_at": "2026-06-17",
        "source_output_root": str(output_root),
        "models": paper_ready["model"].tolist(),
        "model_count": int(len(paper_ready)),
        "seed_count_total": int(paper_ready["seed_count"].sum()),
        "best_model_by_mae": str(paper_ready.iloc[0]["model"]),
        "best_mae_mean": float(paper_ready.iloc[0]["mae_mean"]),
        "outputs": outputs,
        "completion_notes": [
            "Overall locked-test metrics now have paper-ready formatted columns plus original numeric source columns.",
            "Group metrics and seed-level metrics are linked as supplementary TODO6 evidence.",
            "Complexity/runtime table reports fit time, available parameter counts, available model artifact sizes, and explicit missingness for inference latency.",
            "Inference latency was not independently recorded in the frozen final evaluation metrics; do not make latency claims until TODO13 performs a controlled benchmark.",
            "High-drift and yielded-regime safety claims remain outside TODO6 and require TODO10.",
        ],
    }
    manifest_path = output_root / "statistics" / "todo6_completion_manifest.json"
    save_json(manifest_path, manifest)

    print(f">>> TODO6 paper-ready table: {paper_ready_csv}")
    print(f">>> TODO6 complexity table: {complexity_csv}")
    print(f">>> TODO6 manifest: {manifest_path}")
    print(f">>> TODO6 report: {report_path}")


if __name__ == "__main__":
    main()
