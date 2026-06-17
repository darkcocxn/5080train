# TODO8 paired 统计检验补齐完成报告

生成日期：2026-06-17

## 1. 补齐范围

本轮只读取 frozen final multi-seed predictions，不训练、不调参、不改测试集。补齐目标是让 TODO8 具备论文 Main Table 4、Holm 校正、Friedman/post-hoc 表、critical-difference style 图和统计主张边界。

## 2. 已生成文件

- `table4_statistical_comparison`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_tables\table4_statistical_comparison.csv`
- `table4_statistical_comparison_md`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_tables\table4_statistical_comparison.md`
- `paired_bootstrap_delta_existing_todo6_8`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\statistics\paired_bootstrap_delta.csv`
- `paired_bootstrap_delta_todo8_extended`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\statistics\paired_bootstrap_delta_todo8_extended.csv`
- `wilcoxon_tests_existing_todo6_8`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\statistics\wilcoxon_tests.csv`
- `wilcoxon_tests_holm`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\statistics\wilcoxon_tests_holm.csv`
- `friedman_posthoc_tests`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\statistics\friedman_posthoc_tests.csv`
- `critical_difference_ranks`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\statistics\critical_difference_ranks.csv`
- `friedman_summary`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\statistics\friedman_summary.json`
- `critical_difference_diagram_png`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\critical_difference_diagram.png`
- `critical_difference_diagram_svg`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\critical_difference_diagram.svg`

## 3. 方法要点

- 主模型自动按 locked-test MAE 选择为 `2dcnn`。
- 最强 GBDT 参考模型为 `lightgbm`。
- 配对对齐优先使用 locked-test 行顺序和 `y_true` 一致性，避免重复 `sample_id` 造成笛卡尔积配对膨胀。
- `Δ = comparator - main`；因此 ΔMAE、ΔRMSE、Δtail_MAE 为正表示主模型误差更低。
- 置信区间采用 seed + sample 两层 hierarchical paired bootstrap；Wilcoxon 使用逐 seed 的逐样本绝对误差差值，并做 Holm correction。
- Friedman/post-hoc 与 rank diagram 以 seed 作为 block，仅作为补充排名稳定性证据；由于只有 5 个 seed，不作为唯一显著性依据。

## 4. 完成标准核对

| TODO8 要求 | 本轮状态 | 证据文件 |
|---|---|---|
| 新增 `compare_models_statistics.py` | 完成 | `update/compare_models_statistics.py` |
| 同一测试样本逐样本 paired delta | 完成，按 row_id/y_true 对齐 | `paired_bootstrap_delta_todo8_extended.csv` |
| 优先聚合/避免相关性夸大 | 部分完成，使用 hierarchical bootstrap 和 seed block；真正 wave/structure block bootstrap 可在 TODO10/OOD 中扩展 | `table4_statistical_comparison.csv` |
| 主模型 vs 最强 GBDT/其他强基线 | 完成；TabM/AutoML 未在当前 final runs 中出现，报告为未覆盖 | `table4_statistical_comparison.csv` |
| ΔMAE、ΔRMSE、Δtail_MAE 95% CI | 完成 | `table4_statistical_comparison.csv` |
| Wilcoxon signed-rank test | 完成，并增加 Holm correction | `wilcoxon_tests_holm.csv` |
| Friedman + post-hoc + CD diagram | 完成，标记为 seed-block exploratory | `friedman_posthoc_tests.csv`, `critical_difference_diagram.png` |

## 5. 主要统计结论

- `2dcnn` 相对最强 GBDT `lightgbm` 的 ΔMAE(comparator-main) = `0.000145828`，hierarchical bootstrap 95% CI = [`0.000133501`, `0.000160409`]。
- hierarchical bootstrap 下，MAE 上主模型显著优于的比较数：`13/13`。
- MAE 差异未显著的比较数：`0/13`；这些比较论文中应写成 comparable/competitive。
- Friedman seed-block 检验：statistic = `64.36`，p = `8.61446e-09`。
- 平均 rank 第一：`2dcnn`，average rank = `1`。

## 6. 主张边界

当前 TODO8 可以支持“差异、置信区间和显著性已经报告”的主结果对比。不能支持的内容包括：未纳入 TabM/RealMLP/AutoML final runs 的直接统计优越性、高漂移/屈服工况显著更安全、以及跨 unseen-wave/unseen-structure 的泛化显著性。后两者仍需要 TODO10。

## 7. 论文写作建议

- Main Table 4 使用 `table4_statistical_comparison.csv`，优先报告主模型 vs strongest GBDT、序列模型、树模型强基线。
- 正文使用 ΔMAE、ΔRMSE、Δtail_MAE 的 bootstrap CI；p 值只作为辅助，不单独作为结论依据。
- 若 bootstrap CI 跨 0，写 `comparable` 或 `competitive`，不要写 `significantly better`。
- CD/rank 图作为附录排名稳定性图，不要把 5-seed exploratory Friedman 结果写成跨数据集结论。

## 8. 参考文献与规范

1. Demsar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1-30. https://jmlr.org/papers/v7/demsar06a.html
2. Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895-1923. https://doi.org/10.1162/089976698300017197
3. Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52, 239-281. https://doi.org/10.1023/A:1024068626366
4. Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2), 65-70. https://www.jstor.org/stable/4615733
5. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving reproducibility in machine learning research. Journal of Machine Learning Research, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html
6. Kapoor, S., Cantrell, E. M., Peng, K., Pham, T. H., Bail, C. A., Gundersen, O. E., Hofman, J. M., Hullman, J., Lones, M. A., Malik, M. M., Nanayakkara, P., Poldrack, R. A., Raji, I. D., Roberts, M., Salganik, M., Serra-Garcia, M., Stewart, B. M., Vandewiele, G., & Narayanan, A. (2024). REFORMS: Reporting standards for machine learning based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452
7. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing pitfalls and filling the gaps in tabular deep learning benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3
8. Erickson, N., Purucker, L., Tschalzev, A., Holzmueller, D., Mutalik Desai, P., Salinas, D., & Hutter, F. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS 2025 Datasets and Benchmarks Track spotlight. https://openreview.net/forum?id=jZqCqpCLdU

## 9. TODO8 状态结论

TODO8 已补齐为“paired 统计检验完成”。后续若新增 TabM/RealMLP/AutoML 或 TODO10 OOD/high-drift benchmark，应复用本脚本重新生成 Table 4 和 post-hoc 统计。
