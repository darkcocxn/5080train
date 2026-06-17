# 2dcnnv7 文献驱动训练优化说明

生成日期：2026-05-31

## 1. 问题复盘

v2 是当前最稳的单模型主干：

```text
R2   = 0.8806 / 0.8828
MAE  = 0.000632 / 0.000628
RMSE = 0.001235 / 0.001221
MAPE = 18.99% / 19.02%
```

v5 证明低漂移优化有效，但也暴露了副作用：

```text
v5 best_mae:
R2   = 0.8749
MAE  = 0.000623
RMSE = 0.001261
MAPE = 17.47%
低漂移 MAE = 0.0000813
```

结论：

1. v5 的低漂移损失确实降低了低漂移误差和 MAPE。
2. v5 低漂移约束过早、过强，导致 `R2/RMSE` 和 `MIDR>=0.010` F1 弱于 v2。
3. 继续加重低漂移 loss 是错误方向。
4. v7 应回到 v2 主干，只做延迟、弱权重、受 RMSE 约束的训练优化。

## 2. 文献与文档依据

### 2.1 深度不均衡回归 DIR

Yang et al. (2021) 提出 Deep Imbalanced Regression，指出连续标签任务不能简单照搬分类不均衡方法，因为相邻标签之间有连续性。他们提出 Label Distribution Smoothing (LDS) 和 Feature Distribution Smoothing (FDS)，核心思想是利用连续标签邻域关系平滑标签分布和特征统计。

对本项目的启发：

```text
最大层间位移角是连续标签，不应该用硬切分阈值完全主导训练。
低漂移、中漂移、尾部样本应通过平滑权重和弱约束连续过渡。
```

### 2.2 DenseWeight / DenseLoss

Steininger et al. (2021) 提出 DenseWeight 和 DenseLoss：根据目标值密度给样本加权，使稀有连续响应区间在训练中获得更高权重。

对本项目的启发：

```text
继续保留 v2 的 density weighted loss。
不要删除 weighted sampler 和 tail weighting。
低漂移也可视作 MAPE 敏感区间，但不能把它当作唯一重点。
```

### 2.3 Balanced MSE

Ren et al. (2022) 指出，视觉回归中的标签分布不均衡会导致普通 MSE 与目标测试分布之间存在统计偏差。Balanced MSE 通过考虑标签分布来缓解不均衡回归。

对本项目的启发：

```text
训练目标需要考虑标签分布，而不是只优化全局平均误差。
但 v5 已证明重新加权过强会牺牲整体泛化，因此 v7 采用弱低漂移项。
```

### 2.4 SmoothL1 / Huber

PyTorch 官方文档说明 `SmoothL1Loss` 与 Huber loss 密切相关：小误差区间为二次项，大误差区间近似 L1。它适合存在异常值或尾部样本的回归任务。

对本项目的启发：

```text
继续使用 SmoothL1Loss，不改成 MSE。
低漂移归一化误差也使用 SmoothL1，而不是直接使用 MAPE 或 MSE。
```

### 2.5 MAPE 小分母问题

Hyndman and Koehler (2006) 指出，常见百分比误差指标在真实值接近 0 时会出现不稳定或失真。

对本项目的启发：

```text
不直接把 MAPE 作为训练损失。
使用 abs(error) / (abs(y_true) + eps) 形式，加入工程容忍尺度 eps。
```

### 2.6 结构地震响应代理模型

Zhang et al. (2024) 在阻尼结构地震响应预测中比较传统 ML、CNN、SWT 和 aggregation model，说明不同模型/训练目标在不同响应状态下可能互补。

对本项目的启发：

```text
v2 与 v5 的互补性是真实存在的。
但这次 v7 先优化训练本身，而不是继续使用融合结果替代训练改进。
```

## 3. v7 训练优化策略

v7 从 v2 主干出发，吸收 v5 的低漂移经验，但显著降低其训练影响。

### 3.1 保留 v2 主干

继续保留：

```text
WeightedRandomSampler
Density weighted loss
Tail loss multiplier
Tail underprediction loss
Tail correction head
Tail probability gated correction
Extreme probability gate blend
Tail classification auxiliary head
EMA
SmoothL1Loss
```

这些机制不能删除，因为 v4 已证明删除尾部压力会造成明显倒退。

### 3.2 低漂移损失延迟启用

v5 从 epoch 1 就启用低漂移损失。v7 改为：

```python
LOW_DRIFT_ACCURACY_START_EPOCH = 12
LOW_DRIFT_ACCURACY_RAMP_EPOCHS = 12
```

原因：

```text
前 10 个 epoch 先让模型学习主分布、尾部分类和尾部校正。
低漂移精修放到中后期，避免早期梯度被小值样本牵引。
```

### 3.3 低漂移权重显著降低

v5：

```python
LOW_DRIFT_ACCURACY_WEIGHT = 0.028
LOW_DRIFT_RELATIVE_EPS = 0.00035
```

v7：

```python
LOW_DRIFT_ACCURACY_WEIGHT = 0.012
LOW_DRIFT_RELATIVE_EPS = 0.0005
LOW_DRIFT_ACCURACY_MAX_LOSS = 0.20
```

原因：

```text
降低权重，减少对 R2/RMSE 和尾部 F1 的副作用。
增大 eps，缓解极小真实值导致的相对误差放大。
降低 max_loss，避免低漂移项在异常 batch 中主导训练。
```

### 3.4 选模加入 RMSE 防退化项

v5 的综合 selection 选到了 epoch 21，但 v5 的 `best_mae` / `best_low` checkpoint 在部分指标上更好，说明 selection 偏早且偏软。

v7 新选模：

```text
selection_score =
    MAE
  + 0.08  * RMSE
  + 0.018 * focus_score
  + 0.025 * low_drift_mae
  + 0.020 * low_drift_norm_mae * eps
  + 0.04  * mid_tail_mae
```

作用：

```text
MAE 仍是主指标。
RMSE 防止 R2/RMSE 退化。
focus_score 继续保护尾部。
低漂移只作为弱约束，而不是主导项。
```

### 3.5 增加多 checkpoint 保存

v7 新增：

```text
best_2dcnn_rmse_model.pth
best_2dcnn_r2_model.pth
```

并保留：

```text
best_2dcnn_model.pth
best_2dcnn_mae_model.pth
best_2dcnn_focus_model.pth
best_2dcnn_extreme_under_model.pth
best_2dcnn_low_drift_model.pth
```

原因：

```text
本任务存在多目标冲突，单一 selection 不一定选出所有论文指标最优的 checkpoint。
保留多个 checkpoint 有助于训练后做严肃对比，而不是靠猜。
```

## 4. 新增文件

```text
2dcnnv7/2dcnnv7.py
2dcnnv7/2dcnnv7test.py
2dcnnv7/启动训练.bat
2dcnnv7/启动测试.bat
```

输出目录：

```text
output/2dcnnv7
```

## 5. 预期目标

相对 v2：

```text
R2 不低于 0.880
MAE 低于 0.000628
RMSE 不高于 0.00123
MAPE 低于 19%
true < 0.001 的 MAE 低于 0.0000886
F1@0.010 尽量不低于 0.54
```

如果 v7 失败，优先判断：

1. 如果 MAPE 改善但 R2/RMSE 退化，继续降低低漂移 loss 权重。
2. 如果 R2/RMSE 改善但 MAPE 无改善，保持训练 loss，微调 selection 低漂移权重。
3. 如果尾部 F1 下降，提高 `SELECTION_FOCUS_WEIGHT` 或降低 `LOW_DRIFT_ACCURACY_WEIGHT`。

## 6. 运行方式

训练：

```powershell
uv run python .\2dcnnv7\2dcnnv7.py
```

测试：

```powershell
uv run python .\2dcnnv7\2dcnnv7test.py
```

也可以双击：

```text
2dcnnv7/启动训练.bat
2dcnnv7/启动测试.bat
```

## 7. 参考文献与文档

1. Yang, Y., Zha, K., Chen, Y.-C., Wang, H., & Katabi, D. (2021). Delving into Deep Imbalanced Regression. *Proceedings of the 38th International Conference on Machine Learning*, PMLR 139:11842-11851. https://proceedings.mlr.press/v139/yang21m.html

2. Steininger, M., Kobs, K., Davidson, P., Krause, A., & Hotho, A. (2021). Density-based weighting for imbalanced regression. *Machine Learning*, 110, 2187-2211. https://doi.org/10.1007/s10994-021-06023-5

3. Ren, J., Zhang, M., Yu, C., & Liu, Z. (2022). Balanced MSE for Imbalanced Visual Regression. *CVPR 2022*. https://openaccess.thecvf.com/content/CVPR2022/html/Ren_Balanced_MSE_for_Imbalanced_Visual_Regression_CVPR_2022_paper.html

4. PyTorch Documentation. `torch.nn.SmoothL1Loss`. https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.SmoothL1Loss.html

5. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001

6. Zhang, T., Xu, W., Wang, S., Du, D., & Tang, J. (2024). Seismic response prediction of a damped structure based on data-driven machine learning methods. *Engineering Structures*, 301, 117264. https://doi.org/10.1016/j.engstruct.2023.117264

