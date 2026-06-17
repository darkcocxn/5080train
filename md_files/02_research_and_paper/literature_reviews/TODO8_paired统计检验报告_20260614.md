# TODO8：paired 统计检验报告

生成日期：2026-06-14  
适用版本：`publication_eval_20260614`

> 2026-06-17 更新：本报告保留为阶段性记录。TODO8 的当前完成状态请以
> `TODO8_paired统计检验补齐完成报告_20260617.md` 为准；新版已自动选择
> `2dcnn` 为主模型、`lightgbm` 为最强 GBDT 参考，新增
> `compare_models_statistics.py`、`table4_statistical_comparison.csv`、
> `wilcoxon_tests_holm.csv`、`friedman_posthoc_tests.csv`、
> `critical_difference_diagram.png` 和 `todo8_completion_manifest.json`。

## 1. 目标

TODO8 的目标是把“模型 A 比模型 B 好”从单纯均值比较推进到配对统计证据。由于所有模型在同一测试样本上评估，误差是成对观测，不能把不同模型当作独立样本处理。当前采用 LightGBM 作为参考模型，因为它在 TODO6 主指标表中 MAE 最低。

本报告对应的核心产物：

- `publication_eval_20260614/statistics/paired_bootstrap_delta.csv`
- `publication_eval_20260614/statistics/wilcoxon_tests.csv`
- `publication_eval_20260614/statistics/todo6_8_manifest.json`

## 2. 方法

当前统计检验包含两条证据链：

- Paired bootstrap：对测试样本成对重采样，估计 `MAE(model) - MAE(reference)` 的 95% CI。
- Wilcoxon signed-rank test：对每个测试样本的绝对误差差值做非参数配对检验。

解释规则：

- `delta_mae_model_minus_reference > 0` 表示候选模型 MAE 高于 LightGBM，即候选模型更差。
- bootstrap 95% CI 若完全大于 0，表示候选模型在该 seed 下显著差于参考模型。
- Wilcoxon 的 p 值用于辅助判断误差分布差异，但论文中必须同时报告效应大小或 bootstrap delta，避免只报告 p 值。

## 3. 关键发现

LightGBM 作为参考模型时，多数模型在大部分 seed 上都显示出正 delta，即测试 MAE 高于 LightGBM。

典型结果：

- Random Forest 与 LightGBM 差距最小，部分 seed 的 bootstrap CI 接近或跨过 0，说明两者在部分随机种子下差距不稳定。
- XGBoost 与 CatBoost 在多数 seed 上显著差于 LightGBM，但差距仍属于强基线级别。
- ExtraTrees、MLP、Ridge、ElasticNet、HistGradientBoosting 和 dummy baselines 均明显差于 LightGBM。
- Wilcoxon 检验中 Random Forest 有若干 seed p 值不显著，进一步说明 LightGBM 对 Random Forest 的优势需要谨慎表述为“小幅但整体占优”，而不是绝对压倒。

示例 delta 量级：

| 模型 | 与 LightGBM 的关系 | 统计解释 |
|---|---|---|
| Random Forest | 最接近 | 多数 seed 为正 delta，但部分 CI 或 p 值不强 |
| XGBoost | 略差 | 多数 seed bootstrap CI 大于 0 |
| CatBoost | 略差到中等差距 | 部分 seed 差距小，部分 seed 明显 |
| ExtraTrees | 中等差距 | 所有 seed 明显差于 LightGBM |
| MLP | 明显差距 | 平均指标和配对检验均不占优 |
| Dummy baselines | 极大差距 | 只作为 sanity check |

## 4. 论文写作建议

正文统计句式建议：

> We used paired bootstrap over test samples and Wilcoxon signed-rank tests on per-sample absolute errors, using LightGBM as the strongest tabular reference. The proposed model is considered meaningfully better only if it improves the mean metric and its paired bootstrap confidence interval excludes zero in the favorable direction.

中文解释：

- 主模型若只是均值略低，不能直接声称显著优于强基线。
- 需要同时满足平均 MAE 改善、paired bootstrap 置信区间支持、Wilcoxon 或等价非参数检验支持、多个 seed 方向一致。
- 若消融实验中的完整模型对某个消融项只提升很小，建议在论文中写为“contributes marginally but consistently”，不要夸大。

## 5. 下一步与 TODO9 的衔接

TODO8 已为最终模型对比建立统计检验框架。TODO9 消融实验必须复用同一套统计逻辑：

- 完整 2D-CNN/多模态模型作为 reference。
- 每个 ablation 与完整模型做逐样本 paired bootstrap。
- 至少 3 个 seed；关键消融项建议 5 个 seed。
- 同时报告主指标、尾部指标和 paired delta。
- 若某个组件改善只体现在尾部安全指标而非整体 MAE，应明确写成工程安全贡献，不应混同为整体精度贡献。

## 6. 当前限制

- 当前 paired bootstrap 比较的是最终传统/标量基线，还未覆盖 2D-CNN 消融模型。
- 当前统计表按 seed 分开，没有做跨 seed 的层级 bootstrap；投稿前可补充 seed-level meta-analysis 或 Bayesian hierarchical estimate。
- 当前测试集高阈值样本不足，因此安全阈值统计的显著性需要 TODO10 扩展集补强。

## 7. 参考文献与规范依据

1. NeurIPS. Paper Checklist Guidelines. https://neurips.cc/public/guides/PaperChecklist
2. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving Reproducibility in Machine Learning Research. JMLR, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html
3. Demsar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR, 7, 1-30. https://jmlr.org/papers/v7/demsar06a.html
4. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
5. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3
