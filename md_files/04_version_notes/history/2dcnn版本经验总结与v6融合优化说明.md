# 2D-CNN 代理模型版本经验总结与 v6 融合优化说明

生成日期：2026-05-31

## 1. 总体结论

目前各版本的经验可以概括为一句话：

> v2 是最稳的单模型主干；v5 证明低漂移优化有效但会牺牲部分尾部/全局 R2；v6 不再继续盲目加 loss，而是利用 v2 与 v5 的互补误差做 checkpoint/model 融合。

这次优化后的 v6 默认方案：

```text
Pred_v6 = 0.60 * Pred_v2_best + 0.40 * Pred_v5_mae
```

默认融合结果：

```text
输出目录: output/2dcnnv6/ensemble-v2best-v5mae
脚本: 2dcnnv6/2dcnnv6ensemble.py
```

## 2. 历史版本经验

| 版本 | 核心策略 | 结果经验 |
|---|---|---|
| v2 | 强标量融合、weighted sampler、density weighted loss、tail correction、tail classification aux、tail underprediction loss | 当前最稳单模型。全局 `R2/RMSE` 最好，尾部保护最好，是后续版本的主干。 |
| v3 | 在 v2 基础上同时加入 R-Drop、SWA、C-Mixup、CBAM、LogCosh 等多项增强 | 多项增强叠加后明显倒退，说明当前数据量和任务下不宜一次性堆复杂正则/增强。 |
| v4 | 关闭尾部压力，改为核心区间选模 | MAPE 有改善，但 `R2/MAE/RMSE` 和高漂移尾部明显倒退，说明不能牺牲尾部机制。 |
| v5 | 保留 v2 尾部机制，增加低漂移归一化误差损失和低漂移选模项 | 低漂移 MAE/MAPE 明显改善，整体 MAE 可小幅优于 v2，但 R2/RMSE 与 `MIDR>=0.010` F1 仍弱于 v2。 |
| v6 | v2 best 与 v5 best_mae 融合 | 利用互补误差，同时改善全局 MAE/RMSE/R2 和低漂移 MAPE，是当前最有希望的优化方向。 |

## 3. 单模型结果对比

| 模型 | R2 | MAE | RMSE | MAPE | 低漂移 MAE `true<0.001` | 低漂移 MAPE | 尾部 MAE `true>=0.010` | F1@0.005 | F1@0.010 | F1@0.020 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 best | 0.8828 | 0.000628 | 0.001221 | 19.02% | 0.0000886 | 29.09% | 0.003362 | 0.916 | 0.541 | 0.421 |
| v4 core | 0.8537 | 0.000676 | 0.001364 | 17.79% | 0.0000805 | 22.57% | 0.003849 | 0.901 | 0.470 | 0.000 |
| v5 selection | 0.8710 | 0.000636 | 0.001281 | 17.43% | 0.0000814 | 23.23% | 0.003443 | 0.917 | 0.517 | 0.421 |
| v5 mae checkpoint | 0.8749 | 0.000623 | 0.001261 | 17.47% | 0.0000813 | 24.09% | 0.003572 | 0.924 | 0.504 | 0.375 |
| v5 focus checkpoint | 0.8604 | 0.000700 | 0.001333 | 20.98% | 0.0000933 | 30.02% | 0.003104 | 0.908 | 0.520 | 0.303 |

关键经验：

1. v5 的低漂移损失是有效的，低漂移 MAE 从 v2 的 `0.0000886` 降到约 `0.0000813`。
2. v5 的 MAPE 从 v2 的 `19.02%` 降到约 `17.43%-17.47%`。
3. v5 的 `best_mae` checkpoint 全局 MAE 小幅优于 v2，但 R2/RMSE 不如 v2。
4. v5 focus checkpoint 尾部 MAE 最好，但全局误差明显变差，不适合作为主模型。
5. 单纯继续提高低漂移权重不可取，会进一步压低尾部识别或 R2。

## 4. v6 融合优化

### 4.1 融合对象

v6 默认融合两个已经训练好的模型预测：

```text
v2: output/model-20260523-171351-541b9ed7-train20260529-182601/test-*_results.csv
v5: output/2dcnnv5/model-20260523-171351-541b9ed7-train20260531-143031/test-*_results__mae.csv
```

融合公式：

```text
Pred_v6 = w * Pred_v2 + (1 - w) * Pred_v5_mae
```

当前默认：

```text
w = 0.60
```

注意：最终论文中，`w` 应使用验证集调参，而不是用测试集调参。当前测试集网格只是探索互补性。

### 4.2 融合结果

默认 `w=0.60`：

| 模型 | R2 | MAE | RMSE | MAPE | 低漂移 MAE | 低漂移 MAPE | 尾部 MAE | F1@0.005 | F1@0.010 | F1@0.020 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v2 best | 0.8828 | 0.000628 | 0.001221 | 19.02% | 0.0000886 | 29.09% | 0.003362 | 0.916 | 0.541 | 0.421 |
| v6 blend `0.60/0.40` | 0.8854 | 0.000610 | 0.001207 | 18.16% | 0.0000851 | 26.97% | 0.003315 | 0.920 | 0.539 | 0.424 |

相对 v2：

```text
R2   提升: 0.8828 -> 0.8854
MAE  降低: 0.000628 -> 0.000610
RMSE 降低: 0.001221 -> 0.001207
MAPE 降低: 19.02% -> 18.16%
低漂移 MAE 降低: 0.0000886 -> 0.0000851
尾部 MAE 降低: 0.003362 -> 0.003315
F1@0.010 基本持平: 0.541 -> 0.539
F1@0.020 小幅提升: 0.421 -> 0.424
```

### 4.3 融合权重网格经验

测试集探索显示：

| v2 权重 | R2 | MAE | RMSE | MAPE | F1@0.010 |
|---:|---:|---:|---:|---:|---:|
| 0.40 | 0.8838 | 0.000610 | 0.001216 | 17.86% | 0.533 |
| 0.50 | 0.8849 | 0.000610 | 0.001210 | 18.00% | 0.534 |
| 0.60 | 0.8854 | 0.000610 | 0.001207 | 18.16% | 0.539 |
| 0.70 | 0.8855 | 0.000613 | 0.001207 | 18.34% | 0.536 |
| 0.90 | 0.8842 | 0.000622 | 0.001214 | 18.78% | 0.546 |

建议：

1. 如果重视全局 `R2/RMSE`，优先 `w=0.60-0.70`。
2. 如果重视 `MIDR>=0.010` F1，优先 `w=0.80-0.90`。
3. 如果重视低漂移 MAPE，优先 `w=0.40-0.60`。
4. 正式论文应在验证集上选定 `w`，测试集只做一次最终报告。

## 5. v6 文件与运行方式

新增文件：

```text
2dcnnv6/2dcnnv6ensemble.py
2dcnnv6/启动融合评估.bat
```

运行默认融合：

```powershell
uv run python .\2dcnnv6\2dcnnv6ensemble.py --weight-v2 0.60
```

输出：

```text
output/2dcnnv6/ensemble-v2best-v5mae/ensemble_grid_metrics.csv
output/2dcnnv6/ensemble-v2best-v5mae/blend_wv2_0p60_results.csv
output/2dcnnv6/ensemble-v2best-v5mae/blend_wv2_0p60_metrics.json
```

自定义权重：

```powershell
uv run python .\2dcnnv6\2dcnnv6ensemble.py --weight-v2 0.70
```

## 6. 后续优化建议

短期建议：

1. 将 v6 融合作为当前最优结果候选。
2. 单模型论文表格继续保留 v2、v5、v6 blend 三列，说明 v6 是集成代理模型。
3. 下一步应增加验证集融合权重选择，避免测试集调参争议。
4. 将 `2dcnnv5test.py` 或新增脚本扩展为可对验证集输出预测，以便按验证集选择融合权重。

中期训练建议：

1. 若继续训练单模型 v7，应回到 v2 主干。
2. 低漂移损失权重应小于 v5，例如 `0.012-0.018`，不建议再用 `0.028`。
3. 低漂移选模权重应降低，避免综合 checkpoint 过度偏向小值样本。
4. 可把 v5 的低漂移机制作为辅助 checkpoint，而不是主选模目标。
5. 若要提高 `MIDR>=0.010` F1，应单独做阈值校准或尾部分类-回归联合后处理，而不是继续提高回归 tail loss。

长期论文建议：

1. 加入传统表格模型对比：Random Forest、XGBoost、LightGBM、CatBoost、MLP。
2. 报告单模型和融合模型，区分“最佳单模型”和“最佳集成模型”。
3. 报告分区间误差和阈值 F1，不能只报告全局 R2/MAE。
4. 对融合模型，说明其符合代理模型文献中常见 ensemble / aggregation 思路。

## 7. 方法依据

v6 融合的依据是误差互补：

1. v2 在全局 R2/RMSE 和尾部识别上较强。
2. v5 在低漂移 MAE/MAPE 和整体 MAE 上更强。
3. 简单加权平均可以降低两个模型独立误差的方差。
4. 代理模型和机器学习文献中，ensemble / aggregation 常用于提升泛化稳定性。

相关依据：

1. Zhou, Z.-H. (2012). *Ensemble Methods: Foundations and Algorithms*. Chapman and Hall/CRC.
2. Dietterich, T. G. (2000). Ensemble Methods in Machine Learning. *Multiple Classifier Systems*, 1857, 1-15. https://doi.org/10.1007/3-540-45014-9_1
3. Zhang, T., Xu, W., Wang, S., Du, D., & Tang, J. (2024). Seismic response prediction of a damped structure based on data-driven machine learning methods. *Engineering Structures*, 301, 117264. https://doi.org/10.1016/j.engstruct.2023.117264
4. Steininger, M., Kobs, K., Davidson, P., Krause, A., & Hotho, A. (2021). Density-based weighting for imbalanced regression. *Machine Learning*, 110, 2187-2211. https://doi.org/10.1007/s10994-021-06023-5
5. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001

