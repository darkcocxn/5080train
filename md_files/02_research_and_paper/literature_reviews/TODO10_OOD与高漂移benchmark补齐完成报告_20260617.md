# TODO10 OOD 与 high-drift benchmark 补齐完成报告

生成日期：2026-06-17

## 1. 补齐范围

本轮只读取 protocol CSV 与 TODO5 frozen final predictions，不训练、不调参、不改测试集。TODO10 被拆成三类证据：已独立完成的 exact unseen-wave、已独立完成的 exact unseen-structure、以及当前数据不足但已给出冻结模型诊断和新 OpenSees 生成计划的 high-drift stress test。

## 2. 已生成文件

- `unseen_wave_metrics`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\unseen_wave_metrics.csv`
- `unseen_structure_metrics`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\unseen_structure_metrics.csv`
- `high_drift_stress_metrics`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\high_drift_stress_metrics.csv`
- `ood_benchmark_metrics`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\ood_benchmark_metrics.csv`
- `ood_benchmark_metrics_by_seed`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\ood_benchmark_metrics_by_seed.csv`
- `ood_benchmark_sample_index`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\ood_benchmark_sample_index.csv`
- `ood_split_coverage`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\ood_split_coverage.csv`
- `high_drift_opensees_generation_plan`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\ood_benchmark\high_drift_opensees_generation_plan.csv`
- `table6_ood_generalization`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_tables\table6_ood_generalization.csv`
- `table6_ood_generalization_md`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_tables\table6_ood_generalization.md`
- `ood_performance_drop_png`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\ood_performance_drop.png`
- `ood_performance_drop_svg`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\ood_performance_drop.svg`

## 3. 完成标准核对

| TODO10 要求 | 本轮状态 | 证据文件 |
|---|---|---|
| 构建 unseen-wave 测试集 | 完成；locked test 的 exact `txt_path` 100% 未出现在 train+val | `unseen_wave_metrics.csv`, `ood_split_coverage.csv` |
| 构建 unseen-structure 测试集 | 完成；locked test 的 exact structure signature 100% 未出现在 train+val | `unseen_structure_metrics.csv`, `ood_split_coverage.csv` |
| 构建 high-drift stress test | 独立 labeled stress set 仍缺失；已输出 validation 诊断指标与新 OpenSees 工况生成计划 | `high_drift_stress_metrics.csv`, `high_drift_opensees_generation_plan.csv` |
| 数据不足时明确生成新的 OpenSees 工况 | 完成；不得用 validation high-drift 代替独立结论 | `high_drift_opensees_generation_plan.csv` |
| 所有 OOD benchmark 在模型冻结后运行 | 完成；所有指标来自 TODO5 final frozen predictions | `todo10_completion_manifest.json` |

## 4. 数据覆盖结论

- locked test rows: `3116`；exact unseen wave vs train+val: `3116/3116`。
- locked test exact unseen structure vs train+val: `3116/3116`。
- locked test `target >= 0.010`: `0`；`steel01_yielded=1`: `0`。
- validation `target >= 0.010`: `152`；`steel01_yielded=1`: `152`，但 validation 已用于 HPO/early stopping，因此只能做诊断。

## 5. 主要结果

- locked unseen-wave/unseen-structure 上 MAE 最低模型：`2dcnn`，MAE = `0.000314189`。
- validation high-drift diagnostic 上 MAE 最低模型：`mlp`，MAE = `0.0034488`。
- `2dcnn` on `locked_test_unseen_wave`: MAE = `0.000314189`, RMSE = `0.00044736`, samples = `3e+03`.
- `2dcnn` on `validation_high_drift_ge_0.010_diagnostic`: MAE = `0.00390749`, RMSE = `0.00404576`, samples = `2e+02`.

## 6. 论文主张边界

可以写：当前 locked protocol 实际上已经是 exact unseen-wave 和 exact unseen-structure split，2D-CNN 的主结果是在该独立 split 上获得，而不是同波同结构插值测试。

不能写：模型已经证明在 high-drift、主体屈服、接近失效或 `target >= 0.010/0.015/0.020` 工况下可靠。原因是独立 locked test 中这些样本为 0；validation 中虽然有 high-drift/yielded 样本，但它参与过 HPO/early stopping，不能替代外部 stress test。

## 7. OpenSees high-drift 生成计划

已生成 `200` 条候选工况。候选思路是：从 validation high-drift 结构签名中抽取高响应结构模板，与未使用或 held-out rare-wave 候选组合，重新运行 OpenSees 得到新的 labeled CSV，再用已冻结模型做一次纯 evaluation。该步骤完成后，应重新运行本脚本以填充真正的 independent high-drift stress metrics。

## 8. 参考文献与规范

以下文献与规范已于 2026-06-17 核对，优先采用 OpenReview、NeurIPS Proceedings、PMLR、Science Advances/REFORMS 官方页面或 EQUATOR guideline 记录。它们支撑本轮将 OOD split、模型冻结后评估、validation overfitting 边界、claim boundary 和可复现报告作为论文证据链。

1. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3
2. Erickson, N., Purucker, L., Tschalzev, A., Holzmueller, D., Mutalik Desai, P., Salinas, D., & Hutter, F. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS 2025 Datasets and Benchmarks Track spotlight. https://openreview.net/forum?id=jZqCqpCLdU
3. Koh, P. W., Sagawa, S., Marklund, H., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. ICML 2021. https://proceedings.mlr.press/v139/koh21a.html
4. Gardner, J., Popovic, Z., & Schmidt, L. (2023). Benchmarking distribution shift in tabular data with TableShift. NeurIPS 2023 Datasets and Benchmarks. https://proceedings.neurips.cc/paper_files/paper/2023/hash/a76a757ed479a1e6a5f8134bea492f83-Abstract-Datasets_and_Benchmarks.html
5. Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J. M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D., Roberts, M., Salganik, M., Serra-Garcia, M., Stewart, B. M., Vandewiele, G., & Narayanan, A. (2024). REFORMS: Reporting standards for machine learning based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452

## 9. TODO10 状态结论

TODO10 的 unseen-wave 与 unseen-structure benchmark 已完成。High-drift stress test 已完成覆盖审计、冻结模型诊断和新 OpenSees 工况计划，但独立 high-drift labeled benchmark 仍需要实际 OpenSees 生成后才能支持强安全结论。
