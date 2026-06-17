# TODO11 不确定性与 conformal 校准完成报告

生成日期：2026-06-17

## 1. 工作范围

本轮只读取 TODO5 frozen final multi-seed predictions 与 TODO1 locked protocol CSV，不训练、不调参、不改测试集。方法是：对每个模型的 5 个 seed 构造 ensemble 均值、方差和经验 5/50/95 分位数；在 validation split 上拟合 split conformal 校准半径；在 locked test 上报告 90% prediction interval coverage、average interval width、tail coverage、OOD coverage 和误差-不确定性相关性。

## 2. 已生成文件

- `prediction_intervals`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\uncertainty\prediction_intervals.csv`
- `coverage_by_group`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\uncertainty\coverage_by_group.csv`
- `coverage_by_tail`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\uncertainty\coverage_by_tail.csv`
- `uncertainty_error_correlation`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\uncertainty\uncertainty_error_correlation.csv`
- `conformal_calibration_params`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\uncertainty\conformal_calibration_params.json`
- `calibration_curve_points`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\uncertainty\calibration_curve_points.csv`
- `interval_width_vs_error_points`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\uncertainty\interval_width_vs_error_points.csv`
- `table7_uncertainty_calibration`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_tables\table7_uncertainty_calibration.csv`
- `table7_uncertainty_calibration_md`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_tables\table7_uncertainty_calibration.md`
- `calibration_curve_png`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\calibration_curve.png`
- `calibration_curve_svg`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\calibration_curve.svg`
- `interval_width_vs_error_png`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\interval_width_vs_error.png`
- `interval_width_vs_error_svg`: `C:\Users\jiaotong1\5080train\publication_eval_20260614\paper_figures\interval_width_vs_error.svg`

## 3. TODO11 完成标准核对

| TODO11 要求 | 本轮状态 | 证据文件 |
|---|---|---|
| 基于 5 seed ensemble 输出预测均值和方差 | 完成；每个样本含 `pred_mean`, `pred_var_seed`, `pred_std_seed` | `prediction_intervals.csv` |
| 输出 5%、50%、95% 分位预测 | 完成；采用 5-seed empirical quantile，不启动新 quantile-regression 训练 | `prediction_intervals.csv` |
| 使用 validation/calibration split 做 conformal calibration | 完成；validation 作为 calibration split，locked test 独立评估 | `conformal_calibration_params.json` |
| 报告 90% prediction interval coverage | 完成；含 raw、absolute conformal、adaptive conformal、ensemble-CQR 四种区间 | `coverage_by_group.csv`, `table7_uncertainty_calibration.csv` |
| 报告 average interval width | 完成 | `coverage_by_group.csv`, `coverage_by_tail.csv` |
| 单独报告 tail coverage 和 OOD coverage | 完成；test OOD 为 exact unseen-wave/structure，high-drift 独立样本缺失则显式标记 | `coverage_by_tail.csv` |
| 检查高误差样本是否对应更高不确定性 | 完成；输出 Spearman/Pearson 和 top-error uncertainty ratio | `uncertainty_error_correlation.csv`, `interval_width_vs_error.png` |

## 4. 主结果摘录

- `2dcnn` locked-test adaptive conformal 90% interval: coverage = `0.958922`, average width = `0.00211345`, mean absolute error = `0.000302726`.

`2dcnn` tail/OOD coverage 摘录：

| group_value | n | coverage | avg_interval_width | claim_scope |
| --- | --- | --- | --- | --- |
| target >= 0.005 | 143 | 0.776224 | 0.00405127 | final_locked_test |
| target >= 0.010 | 0 | NA | NA | final_locked_test_absent_if_n0 |
| steel01_yielded=1 | 0 | NA | NA | final_locked_test_absent_if_n0 |

`2dcnn` 误差-不确定性相关性摘录：

| uncertainty_variable | spearman_abs_error_vs_uncertainty | high_error_uncertainty_ratio |
| --- | --- | --- |
| pred_std_seed | 0.405969 | 1.62086 |
| adaptive_interval_width | 0.405969 | 1.57611 |

## 5. 论文主张边界

- 可以写：模型不只报告点预测，还给出了模型间 seed spread、empirical quantile 和 conformal-calibrated 90% prediction interval；locked test 上覆盖率和区间宽度已独立报告。
- 可以写：locked test 的 OOD coverage 等价于 exact unseen-wave / exact unseen-structure coverage，因为 TODO10 已确认 test 的 exact `txt_path` 与 structure signature 100% 未出现在 train+val。
- 必须谨慎写：validation 曾用于 HPO/early stopping，因此 split conformal 的严格有限样本保证不如专门保留 calibration set 干净；本文应表述为“post-hoc validation-calibrated intervals with independent locked-test empirical coverage”。
- 不能写：已经训练了专门的 quantile regression 模型。本轮为了不干扰训练任务，采用的是 5-seed empirical quantile + conformal correction。
- 不能写：高漂移或主体屈服工况下区间已经独立校准。TODO10 已确认 locked test 中 `target >= 0.010` 和 `steel01_yielded=1` 样本为 0；validation high-drift 只能诊断。

## 6. 参考文献与规范

以下文献已于 2026-06-17 核对，基础方法文献用于定义 deep ensemble / conformal / CQR，2025-2026 文献用于说明 conformal UQ 在复杂学习场景和物理 surrogate model 中仍是前沿可靠性工具，REFORMS 用于约束 coverage、区间宽度和 claim boundary 的报告规范。

1. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. NeurIPS. https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles
2. Romano, Y., Patterson, E., & Candes, E. (2019). Conformalized quantile regression. NeurIPS. https://papers.neurips.cc/paper/8613-conformalized-quantile-regression
3. Angelopoulos, A. N., & Bates, S. (2023). Conformal prediction: A gentle introduction. Foundations and Trends in Machine Learning, 16(4), 494-591. https://arxiv.org/abs/2107.07511
4. Gao, R., & Liu, W. (2025). Model uncertainty quantification by conformal prediction in continual learning. ICML 2025, PMLR 267:18453-18469. https://proceedings.mlr.press/v267/gao25i.html
5. Gopakumar, V., Gray, A., Oskarsson, J., Giles, D., Zanisi, L., Kusner, M., Pamela, S., & Deisenroth, M. P. (2026). Uncertainty quantification of surrogate models using conformal prediction. Machine Learning: Science and Technology. https://www.deisenroth.cc/publication/gopakumar-2026/
6. Kapoor, S., Cantrell, E. M., Peng, K., et al. (2024). REFORMS: Consensus-based recommendations for machine-learning-based science. Science Advances, 10(18), eadk3452. https://www.science.org/doi/10.1126/sciadv.adk3452
7. Gao, J., Du, K., & Qi, J. (2025). Quantifying the uncertainty of structural parameters using machine learning-based surrogate models. ASCE-ASME Journal of Risk and Uncertainty in Engineering Systems, Part A: Civil Engineering. DOI: 10.1061/AJRUA6.RUENG-1550

## 7. TODO11 状态结论

TODO11 已完成为 non-training uncertainty/conformal evaluation。共处理 `14` 个模型、`[5]` 个 seed、validation calibration rows `3420`、locked test rows `3116`。若论文必须声称“专门 quantile regression 模型”，需要另开 TODO11b 训练 0.05/0.50/0.95 quantile models；当前版本已经足以支撑 conformal interval coverage 和工程用户不确定性提示。
