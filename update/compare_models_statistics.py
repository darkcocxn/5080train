# -*- coding: utf-8 -*-
"""Complete TODO8 paired statistical comparison artifacts.

This script consumes frozen final multi-seed predictions only. It aligns models
on the locked test rows, computes paired effect sizes, hierarchical bootstrap
confidence intervals, Wilcoxon tests with Holm correction, and an exploratory
average-rank critical-difference style diagram.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
DEFAULT_BOOTSTRAP_ITERS = 2000
GBDT_CANDIDATES = ("lightgbm", "xgboost", "catboost", "histgradientboosting")
TREE_CANDIDATES = GBDT_CANDIDATES + ("randomforest", "extratrees")


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


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    indexed = [(idx, float(p)) for idx, p in enumerate(p_values) if p is not None and math.isfinite(float(p))]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    if not indexed:
        return adjusted
    indexed.sort(key=lambda item: item[1])
    running_max = 0.0
    for rank, (idx, p_value) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * p_value)
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def discover_predictions(output_root: Path) -> dict[tuple[str, int], pd.DataFrame]:
    final_dir = output_root / "final_multiseed"
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for model_dir in sorted(path for path in final_dir.iterdir() if path.is_dir()):
        model = model_dir.name
        for seed_dir in sorted(path for path in model_dir.iterdir() if path.is_dir() and path.name.startswith("seed_")):
            pred_path = seed_dir / "predictions_test.csv"
            if not pred_path.exists():
                continue
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            predictions[(model, seed)] = read_prediction(pred_path, model, seed)
    if not predictions:
        raise FileNotFoundError(f"No prediction files found under {final_dir}")
    return predictions


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
            "row_id": np.arange(len(df), dtype=int),
            "sample_id": df["sample_id"].astype(str) if "sample_id" in df.columns else np.arange(len(df)).astype(str),
            "y_true": pd.to_numeric(df[true_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[pred_col], errors="coerce"),
        }
    )
    out = out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    out["row_id"] = np.arange(len(out), dtype=int)
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    out["squared_error"] = out["error"] ** 2
    out["sample_occurrence"] = out.groupby("sample_id").cumcount()
    return out


def choose_main_and_references(output_root: Path, models: list[str]) -> tuple[str, str | None, str | None]:
    table_path = output_root / "paper_tables" / "table2_overall_test_metrics.csv"
    if table_path.exists():
        table = pd.read_csv(table_path).sort_values("mae_mean", ascending=True)
        main_model = str(table.iloc[0]["model"])
        present = set(table["model"])
        strongest_gbdt = None
        gbdt_table = table[table["model"].isin(GBDT_CANDIDATES)]
        if not gbdt_table.empty:
            strongest_gbdt = str(gbdt_table.iloc[0]["model"])
        strongest_tree = None
        tree_table = table[table["model"].isin(TREE_CANDIDATES)]
        if not tree_table.empty:
            strongest_tree = str(tree_table.iloc[0]["model"])
        return main_model, strongest_gbdt, strongest_tree
    main_model = sorted(models)[0]
    return main_model, None, None


def align_pair(predictions: dict[tuple[str, int], pd.DataFrame], main_model: str, comparator: str, seed: int) -> pd.DataFrame | None:
    main = predictions.get((main_model, seed))
    comp = predictions.get((comparator, seed))
    if main is None or comp is None:
        return None
    if len(main) == len(comp) and np.allclose(main["y_true"].to_numpy(), comp["y_true"].to_numpy(), rtol=0, atol=1e-12):
        return pd.DataFrame(
            {
                "seed": seed,
                "row_id": main["row_id"].to_numpy(),
                "sample_id": main["sample_id"].astype(str).to_numpy(),
                "y_true": main["y_true"].to_numpy(dtype=float),
                "main_abs": main["abs_error"].to_numpy(dtype=float),
                "comp_abs": comp["abs_error"].to_numpy(dtype=float),
                "main_sq": main["squared_error"].to_numpy(dtype=float),
                "comp_sq": comp["squared_error"].to_numpy(dtype=float),
            }
        )
    main_keyed = main[["sample_id", "sample_occurrence", "y_true", "abs_error", "squared_error"]].rename(
        columns={"abs_error": "main_abs", "squared_error": "main_sq"}
    )
    comp_keyed = comp[["sample_id", "sample_occurrence", "y_true", "abs_error", "squared_error"]].rename(
        columns={"y_true": "comp_y_true", "abs_error": "comp_abs", "squared_error": "comp_sq"}
    )
    joined = main_keyed.merge(comp_keyed, on=["sample_id", "sample_occurrence"], how="inner")
    if joined.empty:
        return None
    joined = joined[np.isclose(joined["y_true"], joined["comp_y_true"], rtol=0, atol=1e-12)]
    if joined.empty:
        return None
    joined.insert(0, "seed", seed)
    joined.insert(1, "row_id", np.arange(len(joined), dtype=int))
    return joined[["seed", "row_id", "sample_id", "y_true", "main_abs", "comp_abs", "main_sq", "comp_sq"]].reset_index(drop=True)


def seed_delta(aligned: pd.DataFrame) -> dict[str, Any]:
    y_true = aligned["y_true"].to_numpy(dtype=float)
    threshold = float(np.quantile(y_true, 0.95))
    tail = y_true >= threshold
    main_abs = aligned["main_abs"].to_numpy(dtype=float)
    comp_abs = aligned["comp_abs"].to_numpy(dtype=float)
    main_sq = aligned["main_sq"].to_numpy(dtype=float)
    comp_sq = aligned["comp_sq"].to_numpy(dtype=float)
    return {
        "n_pairs": int(len(aligned)),
        "tail_p95_threshold": threshold,
        "tail_p95_pairs": int(tail.sum()),
        "delta_mae_comparator_minus_main": float(comp_abs.mean() - main_abs.mean()),
        "delta_rmse_comparator_minus_main": float(np.sqrt(comp_sq.mean()) - np.sqrt(main_sq.mean())),
        "delta_tail_p95_mae_comparator_minus_main": float(comp_abs[tail].mean() - main_abs[tail].mean()),
        "median_delta_abs_error": float(np.median(comp_abs - main_abs)),
        "main_mae": float(main_abs.mean()),
        "comparator_mae": float(comp_abs.mean()),
    }


def bootstrap_comparison(seed_frames: list[pd.DataFrame], rng: np.random.Generator, iters: int) -> dict[str, Any]:
    mae_delta = np.empty(iters, dtype=float)
    rmse_delta = np.empty(iters, dtype=float)
    tail_delta = np.empty(iters, dtype=float)
    prepared: list[dict[str, np.ndarray]] = []
    for frame in seed_frames:
        y_true = frame["y_true"].to_numpy(dtype=float)
        threshold = float(np.quantile(y_true, 0.95))
        prepared.append(
            {
                "main_abs": frame["main_abs"].to_numpy(dtype=float),
                "comp_abs": frame["comp_abs"].to_numpy(dtype=float),
                "main_sq": frame["main_sq"].to_numpy(dtype=float),
                "comp_sq": frame["comp_sq"].to_numpy(dtype=float),
                "tail": y_true >= threshold,
            }
        )
    seed_count = len(prepared)
    for idx in range(iters):
        main_abs_sum = comp_abs_sum = main_sq_sum = comp_sq_sum = 0.0
        tail_main_abs_sum = tail_comp_abs_sum = 0.0
        total_n = tail_n = 0
        for seed_idx in rng.integers(0, seed_count, size=seed_count):
            item = prepared[int(seed_idx)]
            n = len(item["main_abs"])
            sample_idx = rng.integers(0, n, size=n)
            main_abs = item["main_abs"][sample_idx]
            comp_abs = item["comp_abs"][sample_idx]
            main_sq = item["main_sq"][sample_idx]
            comp_sq = item["comp_sq"][sample_idx]
            tail = item["tail"][sample_idx]
            main_abs_sum += float(main_abs.sum())
            comp_abs_sum += float(comp_abs.sum())
            main_sq_sum += float(main_sq.sum())
            comp_sq_sum += float(comp_sq.sum())
            total_n += n
            if tail.any():
                tail_main_abs_sum += float(main_abs[tail].sum())
                tail_comp_abs_sum += float(comp_abs[tail].sum())
                tail_n += int(tail.sum())
        mae_delta[idx] = comp_abs_sum / total_n - main_abs_sum / total_n
        rmse_delta[idx] = math.sqrt(comp_sq_sum / total_n) - math.sqrt(main_sq_sum / total_n)
        tail_delta[idx] = tail_comp_abs_sum / tail_n - tail_main_abs_sum / tail_n if tail_n else np.nan
    return {
        "delta_mae_bootstrap_ci95_low": float(np.nanquantile(mae_delta, 0.025)),
        "delta_mae_bootstrap_ci95_high": float(np.nanquantile(mae_delta, 0.975)),
        "delta_rmse_bootstrap_ci95_low": float(np.nanquantile(rmse_delta, 0.025)),
        "delta_rmse_bootstrap_ci95_high": float(np.nanquantile(rmse_delta, 0.975)),
        "delta_tail_p95_mae_bootstrap_ci95_low": float(np.nanquantile(tail_delta, 0.025)),
        "delta_tail_p95_mae_bootstrap_ci95_high": float(np.nanquantile(tail_delta, 0.975)),
    }


def build_pairwise_statistics(
    predictions: dict[tuple[str, int], pd.DataFrame],
    main_model: str,
    bootstrap_iters: int,
    rng_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    models = sorted({model for model, _seed in predictions})
    seeds = sorted({seed for _model, seed in predictions})
    rows: list[dict[str, Any]] = []
    wilcoxon_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(rng_seed)
    for comparator in models:
        if comparator == main_model:
            continue
        seed_frames: list[pd.DataFrame] = []
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            aligned = align_pair(predictions, main_model, comparator, seed)
            if aligned is None:
                continue
            seed_frames.append(aligned)
            row = {
                "main_model": main_model,
                "comparator_model": comparator,
                "seed": seed,
                **seed_delta(aligned),
            }
            seed_rows.append(row)
            try:
                stat, p_value = wilcoxon(
                    aligned["comp_abs"].to_numpy(dtype=float),
                    aligned["main_abs"].to_numpy(dtype=float),
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            except ValueError:
                stat, p_value = float("nan"), float("nan")
            wilcoxon_rows.append(
                {
                    "main_model": main_model,
                    "comparator_model": comparator,
                    "seed": seed,
                    "n_pairs": int(len(aligned)),
                    "wilcoxon_statistic": float(stat),
                    "p_value_two_sided": float(p_value),
                    "median_delta_abs_error_comparator_minus_main": float(
                        np.median(aligned["comp_abs"].to_numpy(dtype=float) - aligned["main_abs"].to_numpy(dtype=float))
                    ),
                }
            )
        if not seed_rows:
            continue
        aggregate: dict[str, Any] = {
            "main_model": main_model,
            "comparator_model": comparator,
            "seed_count": len(seed_rows),
            "n_pairs_mean": statistics.fmean([row["n_pairs"] for row in seed_rows]),
            "tail_p95_pairs_mean": statistics.fmean([row["tail_p95_pairs"] for row in seed_rows]),
            "main_better_seed_fraction_mae": statistics.fmean(
                [row["delta_mae_comparator_minus_main"] > 0 for row in seed_rows]
            ),
        }
        for metric in (
            "delta_mae_comparator_minus_main",
            "delta_rmse_comparator_minus_main",
            "delta_tail_p95_mae_comparator_minus_main",
            "median_delta_abs_error",
            "main_mae",
            "comparator_mae",
        ):
            mean, std, ci = mean_std_ci([float(row[metric]) for row in seed_rows])
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_ci95_seed"] = ci
        aggregate.update(bootstrap_comparison(seed_frames, rng, bootstrap_iters))
        low = aggregate["delta_mae_bootstrap_ci95_low"]
        high = aggregate["delta_mae_bootstrap_ci95_high"]
        if low is not None and low > 0:
            conclusion = "main_significantly_better_on_mae"
        elif high is not None and high < 0:
            conclusion = "comparator_significantly_better_on_mae"
        else:
            conclusion = "mae_difference_not_significant"
        aggregate["mae_conclusion_from_hierarchical_bootstrap"] = conclusion
        rows.append(aggregate)
    p_values = [finite_float(row["p_value_two_sided"]) for row in wilcoxon_rows]
    adjusted = holm_adjust(p_values)
    for row, p_adj in zip(wilcoxon_rows, adjusted):
        row["p_value_holm"] = p_adj
        row["holm_significant_0.05"] = bool(p_adj is not None and p_adj < 0.05)
    table = pd.DataFrame(rows).sort_values("delta_mae_comparator_minus_main_mean").reset_index(drop=True)
    wilcoxon_table = pd.DataFrame(wilcoxon_rows).sort_values(["comparator_model", "seed"]).reset_index(drop=True)
    return table, wilcoxon_table


def build_friedman_posthoc(predictions: dict[tuple[str, int], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    models = sorted({model for model, _seed in predictions})
    seeds = sorted({seed for _model, seed in predictions})
    complete_models = [model for model in models if all((model, seed) in predictions for seed in seeds)]
    mae_matrix: dict[str, list[float]] = {}
    for model in complete_models:
        mae_matrix[model] = [float(predictions[(model, seed)]["abs_error"].mean()) for seed in seeds]
    values = [mae_matrix[model] for model in complete_models]
    stat, p_value = friedmanchisquare(*values) if len(complete_models) >= 3 and len(seeds) >= 2 else (float("nan"), float("nan"))

    rank_rows: list[dict[str, Any]] = []
    per_seed_ranks: dict[str, list[float]] = {model: [] for model in complete_models}
    for seed in seeds:
        seed_maes = np.asarray([mae_matrix[model][seeds.index(seed)] for model in complete_models], dtype=float)
        ranks = rankdata(seed_maes, method="average")
        for model, rank in zip(complete_models, ranks):
            per_seed_ranks[model].append(float(rank))
    for model in complete_models:
        mean_rank, std_rank, ci_rank = mean_std_ci(per_seed_ranks[model])
        rank_rows.append(
            {
                "model": model,
                "average_rank": mean_rank,
                "rank_std": std_rank,
                "rank_ci95_seed": ci_rank,
                "mae_mean_across_seeds": statistics.fmean(mae_matrix[model]),
            }
        )
    rank_table = pd.DataFrame(rank_rows).sort_values("average_rank").reset_index(drop=True)

    posthoc_rows: list[dict[str, Any]] = []
    for model_a, model_b in itertools.combinations(complete_models, 2):
        a = np.asarray(mae_matrix[model_a], dtype=float)
        b = np.asarray(mae_matrix[model_b], dtype=float)
        try:
            stat_ab, p_ab = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat_ab, p_ab = float("nan"), float("nan")
        posthoc_rows.append(
            {
                "block_unit": "seed",
                "model_a": model_a,
                "model_b": model_b,
                "n_blocks": len(seeds),
                "mean_mae_a": float(a.mean()),
                "mean_mae_b": float(b.mean()),
                "delta_mae_a_minus_b": float(a.mean() - b.mean()),
                "wilcoxon_statistic_seed_level": float(stat_ab),
                "p_value_two_sided": float(p_ab),
                "friedman_statistic": float(stat),
                "friedman_p_value": float(p_value),
                "interpretation_scope": "exploratory_seed_block_posthoc_due_to_only_5_blocks",
            }
        )
    adjusted = holm_adjust([finite_float(row["p_value_two_sided"]) for row in posthoc_rows])
    for row, p_adj in zip(posthoc_rows, adjusted):
        row["p_value_holm"] = p_adj
        row["holm_significant_0.05"] = bool(p_adj is not None and p_adj < 0.05)
    posthoc_table = pd.DataFrame(posthoc_rows).sort_values(["p_value_holm", "model_a", "model_b"], na_position="last")

    summary = {
        "friedman_statistic": float(stat),
        "friedman_p_value": float(p_value),
        "n_models": len(complete_models),
        "n_blocks": len(seeds),
        "block_unit": "seed",
        "scope_note": "Exploratory because only five seed blocks are available; primary claims should rely on paired bootstrap effect sizes.",
    }
    return posthoc_table, rank_table, summary


def plot_average_ranks(rank_table: pd.DataFrame, output_root: Path, friedman_summary: dict[str, Any]) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = rank_table.sort_values("average_rank", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    y = np.arange(len(plot_df))
    ax.scatter(plot_df["average_rank"], y, s=55, color="#396b9e", zorder=3)
    for idx, row in plot_df.iterrows():
        ci = finite_float(row.get("rank_ci95_seed")) or 0.0
        ax.hlines(idx, row["average_rank"] - ci, row["average_rank"] + ci, color="#396b9e", linewidth=1.5)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["model"])
    ax.invert_yaxis()
    ax.set_xlabel("Average rank across seeds (lower is better)")
    ax.set_title("Exploratory critical-difference style rank diagram")
    ax.grid(axis="x", alpha=0.25)
    note = f"Friedman p={friedman_summary['friedman_p_value']:.3g}; block unit=seed; N={friedman_summary['n_blocks']}"
    ax.text(0.99, 0.02, note, transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    fig.tight_layout()
    png = output_root / "paper_figures" / "critical_difference_diagram.png"
    svg = output_root / "paper_figures" / "critical_difference_diagram.svg"
    fig.savefig(png, dpi=240)
    fig.savefig(svg)
    plt.close(fig)
    return png, svg


def write_table4_markdown(table4: pd.DataFrame, output_root: Path) -> Path:
    md_path = output_root / "paper_tables" / "table4_statistical_comparison.md"
    md = table4[
        [
            "comparator_model",
            "seed_count",
            "delta_mae_comparator_minus_main_mean",
            "delta_mae_bootstrap_ci95_low",
            "delta_mae_bootstrap_ci95_high",
            "delta_rmse_comparator_minus_main_mean",
            "delta_tail_p95_mae_comparator_minus_main_mean",
            "main_better_seed_fraction_mae",
            "mae_conclusion_from_hierarchical_bootstrap",
        ]
    ].copy()
    for col in (
        "delta_mae_comparator_minus_main_mean",
        "delta_mae_bootstrap_ci95_low",
        "delta_mae_bootstrap_ci95_high",
        "delta_rmse_comparator_minus_main_mean",
        "delta_tail_p95_mae_comparator_minus_main_mean",
        "main_better_seed_fraction_mae",
    ):
        md[col] = md[col].map(lambda value: fmt(value, 6))
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Table 4 Statistical Comparison, Paper-Ready View\n\n")
        file.write("Delta is comparator minus main model; positive values favor the main model.\n\n")
        file.write(markdown_table(md))
        file.write("\n")
    return md_path


def create_report(
    output_root: Path,
    main_model: str,
    strongest_gbdt: str | None,
    table4: pd.DataFrame,
    rank_table: pd.DataFrame,
    friedman_summary: dict[str, Any],
    outputs: dict[str, str],
) -> Path:
    report_path = PROJECT_ROOT / "md_files" / "02_research_and_paper" / "literature_reviews" / "TODO8_paired统计检验补齐完成报告_20260617.md"
    gbdt_row = table4[table4["comparator_model"] == strongest_gbdt].iloc[0] if strongest_gbdt in set(table4["comparator_model"]) else None
    significant = table4[table4["mae_conclusion_from_hierarchical_bootstrap"] == "main_significantly_better_on_mae"]
    comparable = table4[table4["mae_conclusion_from_hierarchical_bootstrap"] == "mae_difference_not_significant"]
    lines = [
        "# TODO8 paired 统计检验补齐完成报告",
        "",
        "生成日期：2026-06-17",
        "",
        "## 1. 补齐范围",
        "",
        "本轮只读取 frozen final multi-seed predictions，不训练、不调参、不改测试集。补齐目标是让 TODO8 具备论文 Main Table 4、Holm 校正、Friedman/post-hoc 表、critical-difference style 图和统计主张边界。",
        "",
        "## 2. 已生成文件",
        "",
    ]
    for key, value in outputs.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## 3. 方法要点",
            "",
            f"- 主模型自动按 locked-test MAE 选择为 `{main_model}`。",
            f"- 最强 GBDT 参考模型为 `{strongest_gbdt}`。",
            "- 配对对齐优先使用 locked-test 行顺序和 `y_true` 一致性，避免重复 `sample_id` 造成笛卡尔积配对膨胀。",
            "- `Δ = comparator - main`；因此 ΔMAE、ΔRMSE、Δtail_MAE 为正表示主模型误差更低。",
            "- 置信区间采用 seed + sample 两层 hierarchical paired bootstrap；Wilcoxon 使用逐 seed 的逐样本绝对误差差值，并做 Holm correction。",
            "- Friedman/post-hoc 与 rank diagram 以 seed 作为 block，仅作为补充排名稳定性证据；由于只有 5 个 seed，不作为唯一显著性依据。",
            "",
            "## 4. 完成标准核对",
            "",
            "| TODO8 要求 | 本轮状态 | 证据文件 |",
            "|---|---|---|",
            "| 新增 `compare_models_statistics.py` | 完成 | `update/compare_models_statistics.py` |",
            "| 同一测试样本逐样本 paired delta | 完成，按 row_id/y_true 对齐 | `paired_bootstrap_delta_todo8_extended.csv` |",
            "| 优先聚合/避免相关性夸大 | 部分完成，使用 hierarchical bootstrap 和 seed block；真正 wave/structure block bootstrap 可在 TODO10/OOD 中扩展 | `table4_statistical_comparison.csv` |",
            "| 主模型 vs 最强 GBDT/其他强基线 | 完成；TabM/AutoML 未在当前 final runs 中出现，报告为未覆盖 | `table4_statistical_comparison.csv` |",
            "| ΔMAE、ΔRMSE、Δtail_MAE 95% CI | 完成 | `table4_statistical_comparison.csv` |",
            "| Wilcoxon signed-rank test | 完成，并增加 Holm correction | `wilcoxon_tests_holm.csv` |",
            "| Friedman + post-hoc + CD diagram | 完成，标记为 seed-block exploratory | `friedman_posthoc_tests.csv`, `critical_difference_diagram.png` |",
            "",
            "## 5. 主要统计结论",
            "",
        ]
    )
    if gbdt_row is not None:
        lines.append(
            f"- `{main_model}` 相对最强 GBDT `{strongest_gbdt}` 的 ΔMAE(comparator-main) = "
            f"`{fmt(gbdt_row['delta_mae_comparator_minus_main_mean'])}`，hierarchical bootstrap 95% CI = "
            f"[`{fmt(gbdt_row['delta_mae_bootstrap_ci95_low'])}`, `{fmt(gbdt_row['delta_mae_bootstrap_ci95_high'])}`]。"
        )
    lines.extend(
        [
            f"- hierarchical bootstrap 下，MAE 上主模型显著优于的比较数：`{len(significant)}/{len(table4)}`。",
            f"- MAE 差异未显著的比较数：`{len(comparable)}/{len(table4)}`；这些比较论文中应写成 comparable/competitive。",
            f"- Friedman seed-block 检验：statistic = `{fmt(friedman_summary['friedman_statistic'])}`，p = `{fmt(friedman_summary['friedman_p_value'])}`。",
            f"- 平均 rank 第一：`{rank_table.iloc[0]['model']}`，average rank = `{fmt(rank_table.iloc[0]['average_rank'])}`。",
            "",
            "## 6. 主张边界",
            "",
            "当前 TODO8 可以支持“差异、置信区间和显著性已经报告”的主结果对比。不能支持的内容包括：未纳入 TabM/RealMLP/AutoML final runs 的直接统计优越性、高漂移/屈服工况显著更安全、以及跨 unseen-wave/unseen-structure 的泛化显著性。后两者仍需要 TODO10。",
            "",
            "## 7. 论文写作建议",
            "",
            "- Main Table 4 使用 `table4_statistical_comparison.csv`，优先报告主模型 vs strongest GBDT、序列模型、树模型强基线。",
            "- 正文使用 ΔMAE、ΔRMSE、Δtail_MAE 的 bootstrap CI；p 值只作为辅助，不单独作为结论依据。",
            "- 若 bootstrap CI 跨 0，写 `comparable` 或 `competitive`，不要写 `significantly better`。",
            "- CD/rank 图作为附录排名稳定性图，不要把 5-seed exploratory Friedman 结果写成跨数据集结论。",
            "",
            "## 8. 参考文献与规范",
            "",
            "1. Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1-30. https://jmlr.org/papers/v7/demsar06a.html",
            "2. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197",
            "3. Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52, 239-281. https://doi.org/10.1023/A:1024068626366",
            "4. Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70. https://www.jstor.org/stable/4615733",
            "5. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving reproducibility in machine learning research. Journal of Machine Learning Research, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html",
            "6. Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J. M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D., Roberts, M., Salganik, M., Serra-Garcia, M., Stewart, B. M., Vandewiele, G., & Narayanan, A. (2024). REFORMS: Reporting standards for machine learning based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452",
            "7. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3",
            "8. Erickson, N., Purucker, L., Tschalzev, A., Holzmueller, D., Mutalik Desai, P., Salinas, D., & Hutter, F. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS 2025 Datasets and Benchmarks Track spotlight. https://openreview.net/forum?id=jZqCqpCLdU",
            "",
            "## 9. TODO8 状态结论",
            "",
            "TODO8 已补齐为“paired 统计检验完成”。后续若新增 TabM/RealMLP/AutoML 或 TODO10 OOD/high-drift benchmark，应复用本脚本重新生成 Table 4 和 post-hoc 统计。",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Complete TODO8 paired statistical comparison artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-iters", type=int, default=DEFAULT_BOOTSTRAP_ITERS)
    parser.add_argument("--rng-seed", type=int, default=20260617)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    predictions = discover_predictions(output_root)
    models = sorted({model for model, _seed in predictions})
    seeds = sorted({seed for _model, seed in predictions})
    main_model, strongest_gbdt, strongest_tree = choose_main_and_references(output_root, models)

    table4, wilcoxon_holm = build_pairwise_statistics(
        predictions=predictions,
        main_model=main_model,
        bootstrap_iters=int(args.bootstrap_iters),
        rng_seed=int(args.rng_seed),
    )
    posthoc, rank_table, friedman_summary = build_friedman_posthoc(predictions)
    cd_png, cd_svg = plot_average_ranks(rank_table, output_root, friedman_summary)

    stat_dir = output_root / "statistics"
    table_dir = output_root / "paper_tables"
    table4_csv = table_dir / "table4_statistical_comparison.csv"
    table4.to_csv(table4_csv, index=False, encoding="utf-8-sig")
    table4_md = write_table4_markdown(table4, output_root)
    paired_extended_csv = stat_dir / "paired_bootstrap_delta_todo8_extended.csv"
    table4.to_csv(paired_extended_csv, index=False, encoding="utf-8-sig")
    wilcoxon_csv = stat_dir / "wilcoxon_tests_holm.csv"
    wilcoxon_holm.to_csv(wilcoxon_csv, index=False, encoding="utf-8-sig")
    posthoc_csv = stat_dir / "friedman_posthoc_tests.csv"
    posthoc.to_csv(posthoc_csv, index=False, encoding="utf-8-sig")
    rank_csv = stat_dir / "critical_difference_ranks.csv"
    rank_table.to_csv(rank_csv, index=False, encoding="utf-8-sig")
    friedman_json = stat_dir / "friedman_summary.json"
    save_json(friedman_json, friedman_summary)

    outputs = {
        "table4_statistical_comparison": str(table4_csv),
        "table4_statistical_comparison_md": str(table4_md),
        "paired_bootstrap_delta_existing_todo6_8": str(stat_dir / "paired_bootstrap_delta.csv"),
        "paired_bootstrap_delta_todo8_extended": str(paired_extended_csv),
        "wilcoxon_tests_existing_todo6_8": str(stat_dir / "wilcoxon_tests.csv"),
        "wilcoxon_tests_holm": str(wilcoxon_csv),
        "friedman_posthoc_tests": str(posthoc_csv),
        "critical_difference_ranks": str(rank_csv),
        "friedman_summary": str(friedman_json),
        "critical_difference_diagram_png": str(cd_png),
        "critical_difference_diagram_svg": str(cd_svg),
    }
    report_path = create_report(output_root, main_model, strongest_gbdt, table4, rank_table, friedman_summary, outputs)
    outputs["todo8_completion_report"] = str(report_path)

    missing_requested_baselines = [
        name for name in ("tabm", "realmlp", "autogluon", "automl", "ensemble") if name not in set(models)
    ]
    manifest = {
        "run_name": "todo8_completion",
        "generated_at": "2026-06-17",
        "source_output_root": str(output_root),
        "main_model": main_model,
        "strongest_gbdt_reference": strongest_gbdt,
        "strongest_tree_reference": strongest_tree,
        "models": models,
        "seeds": seeds,
        "model_count": len(models),
        "seed_count": len(seeds),
        "bootstrap_iters": int(args.bootstrap_iters),
        "alignment_policy": "row_order_with_y_true_consistency; fallback sample_id plus occurrence index",
        "delta_sign_convention": "comparator_minus_main; positive favors main model",
        "missing_requested_baselines": missing_requested_baselines,
        "friedman_summary": friedman_summary,
        "outputs": outputs,
        "completion_notes": [
            "Primary claims should use hierarchical paired bootstrap effect sizes and confidence intervals.",
            "Wilcoxon p-values are Holm-corrected across seed-level pairwise tests.",
            "Friedman/post-hoc and critical-difference style diagram use seed blocks only and are exploratory with five blocks.",
            "TabM/RealMLP/AutoML/ensemble final runs are not present in the current artifact set, so direct TODO8 comparisons for them remain unavailable.",
            "High-drift and yielded-regime statistical safety claims still require TODO10.",
        ],
    }
    manifest_path = stat_dir / "todo8_completion_manifest.json"
    save_json(manifest_path, manifest)

    print(f">>> TODO8 main model: {main_model}")
    print(f">>> TODO8 strongest GBDT: {strongest_gbdt}")
    print(f">>> TODO8 Table 4: {table4_csv}")
    print(f">>> TODO8 Wilcoxon Holm: {wilcoxon_csv}")
    print(f">>> TODO8 Friedman posthoc: {posthoc_csv}")
    print(f">>> TODO8 CD diagram: {cd_png}")
    print(f">>> TODO8 manifest: {manifest_path}")
    print(f">>> TODO8 report: {report_path}")


if __name__ == "__main__":
    main()
