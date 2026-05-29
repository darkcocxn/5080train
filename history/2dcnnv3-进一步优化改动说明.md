# 进一步优化改动说明 2026-05-30

## 基线模型性能 (train20260529-182601)

| 指标 | 数值 |
|---|---:|
| R2 (seed mean) | 0.8806 |
| MAE (seed mean) | 0.000632 |
| RMSE (seed mean) | 0.001235 |
| MAPE (seed mean) | 18.99% |
| Tail (>=0.010) MAE | 0.00310 |
| Extreme Tail (>=0.020) MAE | 0.00478 |
| 参数量 | 1,955,097 |
| Train-Val Loss Gap | 0.13 vs 0.42 (3.2x) |

## 问题诊断

### 1. 过拟合（最核心瓶颈）

训练损失 0.13 vs 验证损失 0.42，差距达 3.2 倍。这说明模型在训练集上过度拟合，尽管已有 dropout、weight decay、EMA、梯度裁剪等正则化手段。

### 2. 图像多样性极低

75,009 个训练样本仅共享 167 张唯一小波图。CNN 可以轻易记住所有图像模式，导致图像分支几乎不提供有效泛化信号。

### 3. 学习率调度不够稳定

ReduceLROnPlateau 受验证指标噪声影响，在多模态模型中容易过早或过晚降低学习率。

### 4. 损失函数

SmoothL1Loss 在 beta 点处有不连续的二阶导数，对极端值的梯度行为不够理想。

### 5. 尾部预测

MAPE ~19%，极端尾部低估率仍偏高。需要更好的样本利用策略。

## 新增优化措施

### 新文件

- `多模态-2DCNN-三阶段数据集-3到7层-强化标量融合-进一步优化.py`（训练脚本）
- `2DCNN测试-三阶段数据集-3到7层-强化标量融合-进一步优化.py`（测试脚本）

### 新模型输出目录

```text
model-2dcnn-3stage-rf-3to7-strong-scalar-fusion-v2
```

### 1. R-Drop 正则化 (USE_R_DROP=True, R_DROP_ALPHA=1.0)

**原理**：每个 batch 做两次前向传播（不同 dropout mask），用 MSE 惩罚两次输出的不一致性。这是目前对抗 dropout-based 过拟合最有效的方法之一。

**实现**：在训练循环中，正常前向传播后再做一次不共享 dropout 的前向，计算 `MSE(pred1, pred2)` 作为额外正则项。

**预期效果**：直接缩小 train-val gap，提高泛化能力。

**参考**：R-Drop: Regularized Dropout for Neural Networks (NeurIPS 2021, IEEE TPAMI 2024 extension)

### 2. CosineAnnealingWarmRestarts (USE_COSINE_ANNEALING=True)

**原理**：学习率按余弦曲线周期性变化（T_0=15 epochs, T_mult=2），比 ReduceLROnPlateau 更可预测、更稳定。每次重启可帮助逃离局部极小值。

**参数**：
```text
COSINE_T_0 = 15      # 第一个周期
COSINE_T_MULT = 2    # 周期倍增：15 -> 30 -> 60
COSINE_ETA_MIN = 1e-6
```

**预期效果**：训练更稳定，避免过早降低学习率导致的欠拟合。

### 3. SWA (Stochastic Weight Averaging) (USE_SWA=True)

**原理**：在训练后期（默认 75% epoch 后）启用权重平均，用恒定学习率继续训练并累积权重平均。SWA 找到更平坦的极小值，对噪声更鲁棒。

**参数**：
```text
SWA_START_EPOCH_RATIO = 0.75
SWA_LR = 5e-5
```

**实现**：训练结束后自动更新 BatchNorm 统计量，保存 `swa_2dcnn_model.pth`。

**注意**：SWA 阶段自动禁用 C-Mixup、R-Drop、EMA（它们与 SWA 的目标冲突）。

**参考**：Averaging Weights Leads to Wider Optima and Better Generalization (UAI 2018)

### 4. Log-Cosh Loss (USE_LOG_COSH_LOSS=True)

**原理**：`log(cosh(pred - true))` 是全光滑函数（C-infinity），不需要 beta 超参数。小误差时近似 MSE，大误差时近似 MAE，梯度行为比 SmoothL1 更好。

**实现**：`loss = diff + softplus(-2 * diff) - ln(2)`（数值稳定版本）

**预期效果**：更平滑的梯度有助于优化器更好地处理极端值和小值混合分布。

**参考**：Statistical properties of the log-cosh loss function used in machine learning (arXiv: 2208.04564)

### 5. C-Mixup (USE_CMIXUP=True, CMIXUP_ALPHA=0.4, CMIXUP_SIGMA=2.0)

**原理**：传统 Mixup 随机配对样本，可能产生不合理的插值标签。C-Mixup 按标签相似度配对——标签越接近的样本越可能被混合，这对回归任务尤为合理。

**关键**：数据集只有 167 张唯一图像，C-Mixup 可以在特征空间创造更多有效的训练样本变体。

**参数**：
```text
CMIXUP_ALPHA = 0.4   # Beta 分布参数
CMIXUP_SIGMA = 2.0   # 标签相似度带宽（label 已缩放 1000x）
```

**注意**：SWA 阶段自动禁用 C-Mixup。

**参考**：C-Mixup: Improving Generalization in Regression (NeurIPS 2022)

### 6. CBAM 注意力 (USE_CBAM=True, CBAM_REDUCTION=8)

**原理**：在每个 ConditionalResidualConvBlock2D 的 FiLM 调制后、池化前插入 CBAM（通道注意力 + 空间注意力），让 CNN 自适应聚焦于最重要的频率带和时间段。

**模块结构**：
```text
ChannelAttention: avg_pool + max_pool -> shared_mlp -> sigmoid gate
SpatialAttention: avg_pool + max_pool (channel-wise) -> conv7x7 -> sigmoid gate
```

**预期效果**：帮助 CNN 在仅 167 张图的约束下更有效地提取判别性特征。参数量增加很小（每个 block 约增加 2K 参数）。

**参考**：CBAM: Convolutional Block Attention Module (ECCV 2018)

### 7. 增强图像增广

**高斯模糊**（概率 20%，kernel 3-5，sigma 0.1-2.0）：
- 模拟不同分辨率/平滑程度的小波图变体

**随机擦除**（概率 15%，面积 2-15%，值 0.5）：
- 在 ToTensor 和 Normalize 之后应用
- 迫使模型不依赖局部特征

## 设计决策

### 为什么 SWA 阶段禁用 R-Drop 和 C-Mixup

SWA 的目标是在权重空间找到平坦极小值，它需要稳定的梯度信号。R-Drop 的随机性和 C-Mixup 的标签混合会干扰 SWA 的收敛。同理 EMA 在 SWA 阶段也被跳过。

### 为什么不用 LDS/FDS

LDS（Label Distribution Smoothing）和 FDS（Feature Distribution Smoothing）需要额外的 KDE 估计和特征平滑层，实现复杂度高。当前的密度加权采样 + 尾部低估惩罚已经提供了类似的功能，不值得引入额外的复杂性。

### 参数量变化

CBAM 每个 block 约增加 `C^2/R + C + 49` 个参数（R=reduction=8）。4 个 block 总增加约 12K 参数，从 1.955M 到约 1.967M，增幅 <1%。

## 运行方式

训练：

```powershell
uv run python "多模态-2DCNN-三阶段数据集-3到7层-强化标量融合-进一步优化.py"
```

测试：

```powershell
uv run python "2DCNN测试-三阶段数据集-3到7层-强化标量融合-进一步优化.py"
```

测试 SWA 权重：

```powershell
$env:SURMOD_2DCNN_WEIGHTS_NAME="swa_2dcnn_model.pth"
uv run python "2DCNN测试-三阶段数据集-3到7层-强化标量融合-进一步优化.py"
```

## 预期效果

- **Train-Val Gap**：R-Drop + C-Mixup + 增强增广应显著缩小过拟合差距
- **整体 MAE/R2**：期望 MAE < 0.00060，R2 > 0.90
- **尾部 MAE**：Log-Cosh 的平滑梯度 + C-Mixup 的标签感知混合应改善尾部精度
- **MAPE**：期望从 ~19% 降到 < 18%
- **SWA 权重**：可能比 EMA 权重有更好的泛化表现

## 关注指标对比

| 对比项 | 旧版 (182601) | 优化版预期 |
|---|---:|---:|
| R2 (seed mean) | 0.8806 | > 0.90 |
| MAE (seed mean) | 0.000632 | < 0.00060 |
| MAPE (seed mean) | 18.99% | < 18% |
| Train-Val Loss Gap | 3.2x | < 2.0x |
| Extreme Tail MAE | 0.00478 | < 0.0045 |
