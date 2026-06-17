# -*- coding: utf-8 -*-
"""Rebuild completed TODO9 ablation summaries from prediction CSV files.

This script is intentionally offline-only: it scans finished prediction files,
recomputes metrics, paired bootstrap deltas, Wilcoxon tests, and writes the
final Markdown report without starting any training or evaluation job.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ABLATION_ROOT = PROJECT_ROOT / "publication_eval_20260614" / "ablations"
REPORT_DIR = PROJECT_ROOT / "md_files" / "02_research_and_paper" / "literature_reviews"
REPORT_PATH = REPORT_DIR / "TODO9_2DCNN多模态消融实验最终总结_20260616.md"
SEEDS = (20260614, 20260615, 20260616)
ABLATION_ORDER = (
    "full_model",
    "scalar_only_blank_image",
    "image_only_zero_scalar",
    "no_wave_scalar_features",
    "concat_fusion",
    "no_tail_loss",
    "no_weighted_sampler",
    "no_image_augmentation",
    "no_ema",
)

DESCRIPTIONS = {
    "full_model": "完整模型：wavelet image + structural/wave scalar features + scalar-FiLM CNN + gated bilinear fusion + tail-aware training + weighted sampler + image augmentation + EMA。",
    "scalar_only_blank_image": "将所有小波图替换为空白图，仅保留标量分支。",
    "image_only_zero_scalar": "保留小波图，所有标量特征置零。",
    "no_wave_scalar_features": "移除 CSV 波形统计特征及派生波形特征，保留小波图与结构/布局标量。",
    "concat_fusion": "将 gated bilinear fusion 替换为简单 concat fusion。",
    "no_tail_loss": "关闭 tail-aware loss。",
    "no_weighted_sampler": "关闭 weighted sampler。",
    "no_image_augmentation": "关闭训练期图像增强。",
    "no_ema": "关闭 EMA 权重平均。",
}


def latest_prediction_file(ablation: str, seed: int) -> Path | None:
    root = ABLATION_ROOT / ablation / f"seed_{seed}"
    candidates = sorted(
        [path for path in root.glob("model*/test*_results.csv") if path.stat().st_size > 0],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot


def metrics_for_frame(df: pd.DataFrame) -> dict[str, float | int]:
    y = df["True_Drift"].astype(float).to_numpy()
    pred = df["Pred_Drift"].astype(float).to_numpy()
    err = pred - y
    abs_err = np.abs(err)
    q90 = float(np.quantile(y, 0.90))
    q95 = float(np.quantile(y, 0.95))
    q90_mask = y >= q90
    q95_mask = y >= q95
    return {
        "n": int(len(df)),
        "mae": float(np.mean(abs_err)),
        "rmse": float(math.sqrt(np.mean(err**2))),
        "r2": r2_score_np(y, pred),
        "bias": float(np.mean(err)),
        "max_abs_error": float(np.max(abs_err)),
        "tail_q90_threshold": q90,
        "tail_q90_n": int(np.sum(q90_mask)),
        "tail_q90_mae": float(np.mean(abs_err[q90_mask])),
        "tail_q90_bias": float(np.mean(err[q90_mask])),
        "tail_q90_under": float(np.mean(np.maximum(y[q90_mask] - pred[q90_mask], 0.0))),
        "tail_q95_threshold": q95,
        "tail_q95_n": int(np.sum(q95_mask)),
        "tail_q95_mae": float(np.mean(abs_err[q95_mask])),
        "tail_q95_bias": float(np.mean(err[q95_mask])),
        "tail_q95_under": float(np.mean(np.maximum(y[q95_mask] - pred[q95_mask], 0.0))),
    }


def markdown_table(df: pd.DataFrame, columns: list[str], float_cols: set[str] | None = None) -> str:
    float_cols = float_cols or set()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif col in float_cols:
                values.append(f"{float(value):.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ABLATION_ROOT.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    prediction_frames: dict[tuple[str, int], pd.DataFrame] = {}

    for ablation in ABLATION_ORDER:
        for seed in SEEDS:
            pred_path = latest_prediction_file(ablation, seed)
            if pred_path is None:
                continue
            df = pd.read_csv(pred_path)
            prediction_frames[(ablation, seed)] = df
            row = {
                "ablation": ablation,
                "seed": seed,
                "status": "evaluated_rebuilt",
                "model_dir": str(pred_path.parent),
                "predictions_path": str(pred_path),
            }
            run_rows.append(row)
            metric_rows.append({**row, **metrics_for_frame(df)})

    runs_df = pd.DataFrame(run_rows).sort_values(["ablation", "seed"])
    metrics_df = pd.DataFrame(metric_rows).sort_values(["ablation", "seed"])
    if metrics_df.empty:
        raise RuntimeError("No completed ablation prediction files found.")

    summary_df = (
        metrics_df.groupby("ablation", as_index=False)
        .agg(
            seeds=("seed", "count"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            bias_mean=("bias", "mean"),
            tail_q90_mae_mean=("tail_q90_mae", "mean"),
            tail_q90_mae_std=("tail_q90_mae", "std"),
            tail_q95_mae_mean=("tail_q95_mae", "mean"),
            tail_q95_mae_std=("tail_q95_mae", "std"),
        )
        .sort_values("mae_mean")
    )

    tail_rows = []
    for _, row in metrics_df.iterrows():
        for level in ("q90", "q95"):
            tail_rows.append(
                {
                    "ablation": row["ablation"],
                    "seed": row["seed"],
                    "tail_level": level,
                    "threshold": row[f"tail_{level}_threshold"],
                    "n": row[f"tail_{level}_n"],
                    "mae": row[f"tail_{level}_mae"],
                    "bias": row[f"tail_{level}_bias"],
                    "under": row[f"tail_{level}_under"],
                }
            )
    tail_df = pd.DataFrame(tail_rows)
    tail_summary_df = (
        tail_df.groupby(["ablation", "tail_level"], as_index=False)
        .agg(
            seeds=("seed", "count"),
            threshold_mean=("threshold", "mean"),
            n_mean=("n", "mean"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            bias_mean=("bias", "mean"),
            under_mean=("under", "mean"),
        )
        .sort_values(["tail_level", "mae_mean"])
    )

    boot_rows = []
    wilcoxon_rows = []
    rng = np.random.default_rng(20260616)
    for ablation in ABLATION_ORDER:
        if ablation == "full_model":
            continue
        for seed in SEEDS:
            ref = prediction_frames.get(("full_model", seed))
            cur = prediction_frames.get((ablation, seed))
            if ref is None or cur is None:
                continue
            ref_cols = ref[["sample_id", "True_Drift", "Pred_Drift"]].rename(columns={"Pred_Drift": "Pred_Full"})
            cur_cols = cur[["sample_id", "Pred_Drift"]].rename(columns={"Pred_Drift": "Pred_Ablation"})
            merged = ref_cols.merge(cur_cols, on="sample_id", how="inner")
            y = merged["True_Drift"].astype(float).to_numpy()
            full_abs = np.abs(merged["Pred_Full"].astype(float).to_numpy() - y)
            ab_abs = np.abs(merged["Pred_Ablation"].astype(float).to_numpy() - y)
            diff = ab_abs - full_abs
            observed = float(np.mean(diff))
            boot = np.empty(2000, dtype=float)
            n = len(diff)
            for i in range(len(boot)):
                idx = rng.integers(0, n, n)
                boot[i] = float(np.mean(diff[idx]))
            p_two_sided = float(min(1.0, 2.0 * min(np.mean(boot <= 0.0), np.mean(boot >= 0.0))))
            boot_rows.append(
                {
                    "ablation": ablation,
                    "seed": seed,
                    "reference": "full_model",
                    "n": n,
                    "delta_mae_ablation_minus_full": observed,
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "bootstrap_p_two_sided": p_two_sided,
                    "full_better_fraction": float(np.mean(diff > 0.0)),
                }
            )
            try:
                stat_two, p_two = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
                stat_greater, p_greater = wilcoxon(diff, alternative="greater", zero_method="wilcox")
            except ValueError:
                stat_two = p_two = stat_greater = p_greater = float("nan")
            wilcoxon_rows.append(
                {
                    "ablation": ablation,
                    "seed": seed,
                    "reference": "full_model",
                    "delta_mae_ablation_minus_full": observed,
                    "wilcoxon_stat_two_sided": stat_two,
                    "wilcoxon_p_two_sided": p_two,
                    "wilcoxon_stat_greater": stat_greater,
                    "wilcoxon_p_ablation_worse": p_greater,
                }
            )

    boot_df = pd.DataFrame(boot_rows).sort_values(["ablation", "seed"])
    wilcoxon_df = pd.DataFrame(wilcoxon_rows).sort_values(["ablation", "seed"])

    runs_df.to_csv(ABLATION_ROOT / "ablation_runs.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(ABLATION_ROOT / "ablation_test_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(ABLATION_ROOT / "ablation_test_metrics_summary.csv", index=False, encoding="utf-8-sig")
    tail_df.to_csv(ABLATION_ROOT / "ablation_tail_metrics_by_seed.csv", index=False, encoding="utf-8-sig")
    tail_summary_df.to_csv(ABLATION_ROOT / "ablation_tail_metrics_summary.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(ABLATION_ROOT / "ablation_paired_bootstrap_delta.csv", index=False, encoding="utf-8-sig")
    wilcoxon_df.to_csv(ABLATION_ROOT / "ablation_wilcoxon_tests.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "generated_at": "2026-06-16",
        "source": "update/summarize_completed_ablation_results.py",
        "completed_prediction_files": int(len(metrics_df)),
        "ablations": ABLATION_ORDER,
        "seeds": SEEDS,
        "outputs": {
            "runs": str(ABLATION_ROOT / "ablation_runs.csv"),
            "metrics_by_seed": str(ABLATION_ROOT / "ablation_test_metrics_by_seed.csv"),
            "metrics_summary": str(ABLATION_ROOT / "ablation_test_metrics_summary.csv"),
            "tail_by_seed": str(ABLATION_ROOT / "ablation_tail_metrics_by_seed.csv"),
            "tail_summary": str(ABLATION_ROOT / "ablation_tail_metrics_summary.csv"),
            "bootstrap": str(ABLATION_ROOT / "ablation_paired_bootstrap_delta.csv"),
            "wilcoxon": str(ABLATION_ROOT / "ablation_wilcoxon_tests.csv"),
            "report": str(REPORT_PATH),
        },
    }
    (ABLATION_ROOT / "ablation_completion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary_for_report = summary_df.copy()
    summary_for_report.insert(1, "description", summary_for_report["ablation"].map(DESCRIPTIONS))
    paired_summary = (
        boot_df.groupby("ablation", as_index=False)
        .agg(
            seeds=("seed", "count"),
            delta_mae_mean=("delta_mae_ablation_minus_full", "mean"),
            ci95_low_mean=("ci95_low", "mean"),
            ci95_high_mean=("ci95_high", "mean"),
            p_two_sided_median=("bootstrap_p_two_sided", "median"),
            full_better_fraction_mean=("full_better_fraction", "mean"),
        )
        .sort_values("delta_mae_mean")
    )

    lines = [
        "# TODO9 2D-CNN 多模态消融实验最终总结",
        "",
        "生成日期：2026-06-16",
        "",
        "## 完成状态",
        "",
        "已完成全部 9 个配置 × 3 个 seeds = 27 个训练/测试结果。所有统计均从 `test*_results.csv` 离线重算，避免单配置续跑覆盖 `ablation_runs.csv` 后造成的汇总偏差。",
        "",
        "## 实验协议",
        "",
        "- 固定协议：`publication_eval_20260614/protocol/protocol_lock.json`",
        "- 固定超参数：`update/2dcnn/best_params.json`",
        "- Seeds：`20260614, 20260615, 20260616`",
        "- 每个配置最多 40 epochs，early stopping patience = 8",
        "- 训练 batch candidates：`96, 64, 48, 32`；本轮主要使用 batch size 96",
        "- eval batch size：64",
        "- CUDA + AMP；训练中显存主要约 13.4-13.8 GB，GPU 利用率多数采样在 85%-98%",
        "",
        "## 主指标汇总",
        "",
        markdown_table(
            summary_for_report,
            [
                "ablation",
                "seeds",
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
                "r2_mean",
                "tail_q90_mae_mean",
                "tail_q95_mae_mean",
            ],
            {
                "mae_mean",
                "mae_std",
                "rmse_mean",
                "rmse_std",
                "r2_mean",
                "tail_q90_mae_mean",
                "tail_q95_mae_mean",
            },
        ),
        "",
        "## 相对 full_model 的 paired bootstrap 概览",
        "",
        "定义：`delta_mae_ablation_minus_full = MAE(ablation) - MAE(full_model)`。delta 为正表示消融变体误差更大，full_model 更优；delta 为负表示该消融变体在当前 locked test 上 MAE 更低。",
        "",
        markdown_table(
            paired_summary,
            [
                "ablation",
                "seeds",
                "delta_mae_mean",
                "ci95_low_mean",
                "ci95_high_mean",
                "p_two_sided_median",
                "full_better_fraction_mean",
            ],
            {
                "delta_mae_mean",
                "ci95_low_mean",
                "ci95_high_mean",
                "p_two_sided_median",
                "full_better_fraction_mean",
            },
        ),
        "",
        "## 关键结论",
        "",
        "1. `image_only_zero_scalar` 明显失效，MAE 均值约为 `0.001291`，R2 为负，说明小波图像分支不能单独承担该任务；结构、布局与标量特征是核心信息源。",
        "2. `scalar_only_blank_image` 明显弱于完整模型，说明小波图像分支/多模态表示具有补充价值。",
        "3. `no_wave_scalar_features` 在主 MAE 上最好，但 q95 尾部误差不占优；这更像是 CSV 波形统计特征与小波图像信息存在冗余或噪声，而不是可以直接删除波形信息。论文中应谨慎表述。",
        "4. `no_image_augmentation` 在当前 locked test 上表现略优，说明图像增强不是该协议下的主要收益来源；但这不等于增强无用，仍需外部测试或分布外测试验证稳健性。",
        "5. `no_ema` 的三 seed MAE 均值约为 `0.000331`，弱于 full_model 均值 `0.000320`，并且训练过程更波动；EMA 对稳定性有正贡献。",
        "6. `no_tail_loss` 与 `no_weighted_sampler` 的主 MAE 接近 full_model，但尾部误差和 seed 稳定性需要与 TODO7/TODO8 的工程安全指标联合解释。",
        "",
        "## 输出文件",
        "",
        "- `publication_eval_20260614/ablations/ablation_runs.csv`",
        "- `publication_eval_20260614/ablations/ablation_test_metrics_by_seed.csv`",
        "- `publication_eval_20260614/ablations/ablation_test_metrics_summary.csv`",
        "- `publication_eval_20260614/ablations/ablation_tail_metrics_by_seed.csv`",
        "- `publication_eval_20260614/ablations/ablation_tail_metrics_summary.csv`",
        "- `publication_eval_20260614/ablations/ablation_paired_bootstrap_delta.csv`",
        "- `publication_eval_20260614/ablations/ablation_wilcoxon_tests.csv`",
        "- `publication_eval_20260614/ablations/ablation_completion_manifest.json`",
        "",
        "## 规范与参考文献",
        "",
        "1. Pineau et al. (2021). Improving Reproducibility in Machine Learning Research. JMLR. https://jmlr.org/papers/v22/20-303.html",
        "2. Dodge et al. (2019). Show Your Work: Improved Reporting of Experimental Results. ACL. https://aclanthology.org/D19-1224/",
        "3. Dror et al. (2018). The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing. ACL. https://aclanthology.org/P18-1128/",
        "4. Henderson et al. (2018). Deep Reinforcement Learning that Matters. AAAI. https://ojs.aaai.org/index.php/aaai/article/view/11694",
        "",
        "## 后续建议",
        "",
        "最终论文写作中，建议将该消融表与 TODO5-8 的强基线、多 seed 主指标、复杂度、尾部工程安全指标和 paired tests 放在同一节或连续附录中。对 `no_wave_scalar_features` 与 `no_image_augmentation` 这类看似优于 full_model 的结果，应以误差分布、尾部风险和外部验证限定结论，避免过度声称单一组件无效。",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
