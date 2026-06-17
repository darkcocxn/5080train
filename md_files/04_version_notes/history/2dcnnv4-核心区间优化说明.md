# 2dcnnv4 核心区间优化说明

## 目标

v4 基于当前最优的 v2 版本派生，目标从“尾部低估控制”切换为“优化数据集样本丰富区间的整体训练效果”。

当前测试集大多数样本集中在 `true < 0.010` 的核心区间，因此 v4 将最佳 checkpoint 的选择重点放在该区间，而不是继续围绕 `>=0.010` 和 `>=0.020` 尾部样本调参。

## 新增文件

```text
2dcnnv4/2dcnnv4.py
2dcnnv4/2dcnnv4test.py
2dcnnv4/启动训练.bat
2dcnnv4/启动测试.bat
```

模型输出目录：

```text
output/2dcnnv4
```

## 设计原则

1. 保留 v2 已验证有效的主干：`scalar_film_residual` CNN、残差标量编码器、`gated_bilinear` 融合、EMA、AdamW、SmoothL1Loss。
2. 不继承 v3 的复杂组合：不启用 R-Drop、SWA、C-Mixup、CBAM、LogCosh、额外高斯模糊和随机擦除。
3. 暂时不优化尾部：关闭尾部低估惩罚、尾部 correction head、尾部分类辅助 head。
4. 回到自然训练分布：关闭 weighted sampler、density loss weight、tail loss multiplier，让高频样本主导训练目标。
5. 用核心区间指标选模型：优先优化 `true < 0.010` 样本的验证 MAE，同时保留少量全局 MAE/RMSE 约束，避免模型只局部变好。

## 关键改动

### 1. 训练目标改为核心区间

新增验证核心区间：

```text
VAL_CORE_MAX = 0.010
```

新增 history 字段：

```text
val_core_mae_raw
val_core_bias_raw
val_core_under_mae_raw
val_core_count
```

最佳模型选择公式改为：

```text
selection_score =
    1.00 * core_mae
  + 0.15 * global_mae
  + 0.05 * global_rmse
```

这样 checkpoint 会主要服从核心区间表现，但仍然避免整体指标失控。

### 2. 关闭尾部专用训练压力

```text
USE_TAIL_UNDERPREDICTION_LOSS = False
USE_TAIL_CORRECTION_HEAD = False
USE_TAIL_PROB_GATED_CORRECTION = False
USE_EXTREME_PROB_GATE_BLEND = False
USE_TAIL_CLASSIFICATION_AUX = False
```

动机：v2 的尾部 correction 会主动向上修正高风险样本；在不关心尾部的实验目标下，它可能增加低值和中值样本的偏差。v4 先移除这条分支，让主回归头直接学习主要分布。

### 3. 回到自然样本分布

```text
USE_WEIGHTED_SAMPLER = False
USE_TARGET_WEIGHTED_LOSS = False
USE_DENSITY_WEIGHTED_LOSS = False
LOSS_WEIGHT_GE_005 = 1.0
LOSS_WEIGHT_GE_010 = 1.0
LOSS_WEIGHT_GE_020 = 1.0
```

动机：v2 为尾部做了采样和 loss 权重倾斜；v4 的目标是提升样本丰富区域，所以训练不再人为放大稀有尾部样本。

### 4. 保留但减弱时频 Mask 增广

```text
TIME_FREQ_MASK_PROB = 0.20
```

v2 为 `0.35`。v4 保留轻量时频遮挡，用来抑制对固定小波图局部纹理的记忆，但降低概率，减少对核心区间精细拟合的干扰。

### 5. 学习率调度更耐心

```text
SCHEDULER_PATIENCE = 5
EARLY_STOPPING_PATIENCE = 24
```

v4 用核心区间 score 做调度，验证曲线可能和 v2 的 tail-focused score 不同，因此稍微放宽 patience。

## 运行方式

训练：

```powershell
uv run python .\2dcnnv4\2dcnnv4.py
```

测试：

```powershell
uv run python .\2dcnnv4\2dcnnv4test.py
```

也可以直接双击：

```text
2dcnnv4/启动训练.bat
2dcnnv4/启动测试.bat
```

## 训练后重点看

优先看：

```text
test-*_seed_metrics.csv
test-*_results.csv
training_history.json
training_curves.png
```

建议对比口径：

```text
seed mean: R2 / MAE / RMSE / MAPE
test_holdout: R2 / MAE / RMSE / MAPE
true < 0.001
0.001 <= true < 0.005
0.005 <= true < 0.010
true < 0.010 汇总核心区间
各楼层 MAE，尤其 3-6 层
```

尾部指标仍可记录，但本轮不作为主要成败标准。

## 预期结果

理想情况下，v4 应该相对 v2：

```text
核心区间 MAE 下降
低值样本 MAPE 下降
test_holdout MAE/RMSE 下降或持平
整体 seed mean MAE 小幅下降或持平
尾部 true >= 0.010 可能变差，可接受
```

如果 v4 整体 MAE 下降但尾部明显变差，说明实验目标达成；后续可以做一个“v4-core + 轻尾部保护”的折中版。

## 参考文献

1. Huber, P. J. (1964). Robust Estimation of a Location Parameter. The Annals of Mathematical Statistics, 35(1), 73-101. https://doi.org/10.1214/aoms/1177703732
2. Loshchilov, I., & Hutter, F. (2017). Decoupled Weight Decay Regularization. https://arxiv.org/abs/1711.05101
3. Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2017). FiLM: Visual Reasoning with a General Conditioning Layer. https://arxiv.org/abs/1709.07871
4. Wu, Y., & He, K. (2018). Group Normalization. https://arxiv.org/abs/1803.08494
5. Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019). SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition. https://arxiv.org/abs/1904.08779
6. Prechelt, L. (1998). Automatic early stopping using cross validation: quantifying the criteria. Neural Networks, 11(4), 761-767. https://pubmed.ncbi.nlm.nih.gov/12662814/
