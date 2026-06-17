# 最近两次训练对比总结

## 概览

| 项目 | 第一次训练 (v2) | 第二次训练 (v3) |
|---|---|---|
| **训练目录** | `train20260529-182601` | `train20260530-020242` |
| **训练时间** | 2026-05-29 18:26 | 2026-05-30 02:02 |
| **训练脚本** | [2dcnnv2.py](file:///x:/pyproject/Remote-Train/2dcnnv2/2dcnnv2.py) | [2dcnnv3.py](file:///x:/pyproject/Remote-Train/2dcnnv3/2dcnnv3.py) |
| **测试脚本** | [2dcnnv2test.py](file:///x:/pyproject/Remote-Train/2dcnnv2/2dcnnv2test.py) | [2dcnnv3test.py](file:///x:/pyproject/Remote-Train/2dcnnv3/2dcnnv3test.py) |
| **架构版本** | `regularized_fusion_tail_prob_gate_v2` | `v3_rdrop_swa_cmixup_cbam_logcosh` |
| **模型目录** | `model-2dcnn-3stage-rf-3to7-strong-scalar-fusion` | `model-2dcnn-3stage-rf-3to7-strong-scalar-fusion-v2` |
| **训练时长** | ~70.6 分钟 | ~212.6 分钟 |
| **参数量** | 1,955,097 | 1,970,081 (+14,984) |

---

## 测试性能对比

> [!IMPORTANT]
> v3 版本在所有关键指标上都出现了**明显回退**，没有达到预期的改善目标。

### Seed 平均测试指标

| 指标 | v2 (182601) | v3 (020242) | 变化 | 趋势 |
|---|---:|---:|---|---|
| **R²** | 0.8806 | 0.8179 | -0.0627 | ⬇️ 大幅下降 |
| **MAE** | 0.000632 | 0.000811 | +0.000179 | ⬇️ 恶化 28% |
| **RMSE** | 0.001235 | 0.001525 | +0.000290 | ⬇️ 恶化 23% |
| **MAPE** | 18.99% | 36.58% | +17.59pp | ⬇️ 恶化近一倍 |

### 验证集最佳 Epoch 指标

| 指标 | v2 (182601) | v3 (020242) | 变化 |
|---|---:|---:|---|
| **Best Epoch** | 33 | 38 | +5 |
| **Val MAE (raw)** | 0.000648 | 0.000679 | ⬇️ +4.8% |
| **Val Loss** | 0.4224 | 0.4254 | ⬇️ +0.7% |
| **Val Focus Score** | 0.00494 | 0.00677 | ⬇️ +37% |
| **Val Tail MAE (≥0.010)** | 0.00310 | 0.00349 | ⬇️ +12.8% |
| **Val Extreme Tail MAE (≥0.020)** | 0.00478 | 0.00715 | ⬇️ +49.8% |
| **Val Selection Score** | 0.000787 | 0.000847 | ⬇️ +7.6% |

> [!WARNING]
> 极端尾部 MAE（≥0.020）恶化近50%，MAPE 从 ~19% 暴涨到 ~37%，说明 v3 新增优化措施可能引入了训练不稳定性或损失函数冲突。

---

## 架构差异

### 共同点（保持不变）

两次训练共享相同的底层架构基础：

- **CNN 骨干**: `scalar_film_residual`，4 层通道 `[32, 64, 128, 192]`
- **标量编码器**: `residual`，嵌入维度 160，3 个残差块
- **融合方式**: `gated_bilinear`，双线性维度 64，输出 320
- **预测头**: `[320, 96]`
- **数据集**: 完全相同（75,009 训练 / 4,564 验证，167 唯一波形图像）
- **优化器**: AdamW，lr=1e-4，weight_decay=1e-4
- **批次大小**: 96
- **EMA**: 开启，decay=0.998
- **尾部修正头、尾部分类辅助头**: 结构和参数相同

### v3 新增内容

v3 在 v2 基础上新增了 **7 项优化技术**：

#### 1. R-Drop 正则化
```
R_DROP_ALPHA = 1.0
```
每个 batch 做两次前向传播（不同 dropout mask），用 MSE 惩罚两次输出的不一致性，对抗过拟合。

#### 2. CosineAnnealingWarmRestarts 学习率调度
```
v2: ReduceLROnPlateau (patience=4, factor=0.5, min_lr=5e-7)
v3: CosineAnnealingWarmRestarts (T_0=15, T_mult=2, eta_min=1e-6)
```
从自适应调度改为确定性周期调度，周期 15→30→60 epochs。

#### 3. SWA（随机权重平均）
```
SWA_START_RATIO = 0.75
SWA_LR = 5e-5
```
训练后期启用权重平均，额外保存 `swa_2dcnn_model.pth`。SWA 阶段自动禁用 C-Mixup、R-Drop、EMA。

#### 4. LogCosh 损失函数
```
v2: SmoothL1Loss (beta=1.0)
v3: LogCoshLoss
```
全光滑函数，小误差近似 MSE、大误差近似 MAE，无 beta 超参。

#### 5. C-Mixup 数据增强
```
CMIXUP_ALPHA = 0.4
CMIXUP_SIGMA = 2.0
```
按标签相似度配对样本做混合插值，为回归任务设计的 Mixup 变体。

#### 6. CBAM 注意力模块
```
CBAM_REDUCTION = 8
```
在每个 CNN 残差块的 FiLM 调制后插入通道+空间注意力。新增约 15K 参数。

#### 7. 增强图像增广
```
高斯模糊: prob=0.2
随机擦除: prob=0.15
```
在原有的时频 Mask 增广基础上追加。

---

## 训练配置差异汇总

| 配置项 | v2 (182601) | v3 (020242) |
|---|---|---|
| **损失函数** | SmoothL1Loss | LogCoshLoss |
| **学习率调度** | ReduceLROnPlateau | CosineAnnealingWarmRestarts |
| **R-Drop** | ❌ | ✅ (alpha=1.0) |
| **SWA** | ❌ | ✅ (75%, lr=5e-5) |
| **C-Mixup** | ❌ | ✅ (alpha=0.4, sigma=2.0) |
| **CBAM** | ❌ | ✅ (reduction=8) |
| **高斯模糊** | ❌ | ✅ (prob=0.2) |
| **随机擦除** | ❌ | ✅ (prob=0.15) |
| **总参数量** | 1,955,097 | 1,970,081 |
| **总训练时长** | ~71 分钟 | ~213 分钟 |

---

## 修改记录时间线

根据 [history](file:///x:/pyproject/Remote-Train/history) 目录和 Git 提交记录：

| 时间 | 事件 | 文档 |
|---|---|---|
| 5/28 晚 | v1 初始版本部署，首次训练 | — |
| 5/29 11:25 | v2 优化：强化标量融合 | [改动说明](file:///x:/pyproject/Remote-Train/history/2dcnnv2-强化标量融合改动说明.md) |
| 5/29 11:35 | v2 调整 checkpoint 选择策略 | Git commit `d1c5b44` |
| 5/29 午后 | v2 继续优化：收缩容量 + EMA + 概率门控 correction | [优化记录](file:///x:/pyproject/Remote-Train/history/2dcnnv2-强化标量融合继续优化记录.md) |
| 5/29 18:26 | **v2 最终训练 (182601)** — 当前最佳结果 | — |
| 5/30 01:55 | v3 新脚本：+R-Drop/SWA/C-Mixup/CBAM/LogCosh | [v3 说明](file:///x:/pyproject/Remote-Train/history/2dcnnv3-进一步优化改动说明.md) |
| 5/30 02:02 | **v3 训练 (020242)** — 性能回退 | — |

---

## 分析与结论

> [!CAUTION]
> v3 同时引入了 7 项新技术，且训练时间从 71 分钟暴增到 213 分钟（R-Drop 双前向使计算量翻倍），但性能全面恶化。

### 可能的原因

1. **优化措施冲突**：R-Drop + C-Mixup + SWA 三者同时存在可能相互干扰。R-Drop 期望一致性，C-Mixup 引入标签混合噪声，SWA 需要稳定梯度——三者目标矛盾。

2. **损失函数变化**：LogCoshLoss 替代 SmoothL1Loss 改变了梯度分布特性。在尾部加权损失和尾部低估惩罚的复合损失体系下，这种改变可能打破了 v2 中精心调校的平衡。

3. **学习率调度**：CosineAnnealingWarmRestarts 的确定性周期（15→30→60）可能不适合当前复杂的多任务损失结构。ReduceLROnPlateau 虽然有噪声敏感问题，但至少能自适应调整。

4. **过多增广叠加**：原有的时频 Mask + 新增的高斯模糊 + 随机擦除 + C-Mixup，对仅 167 张唯一图像的数据集可能造成过度正则化，使模型欠拟合。

5. **MAPE 暴涨**：MAPE 从 19% → 37% 暴涨说明模型在小漂移值样本上的相对误差剧增，可能是 LogCosh 损失在小值区域的梯度行为与 SmoothL1 显著不同。

### 建议

1. **回退到 v2 (182601) 作为基线**，该版本目前是最优结果。
2. **逐项消融测试**：不要同时引入多项优化，应逐一实验：
   - 先只试 LogCosh 损失（保持其他不变）
   - 再单独试 R-Drop（保持 SmoothL1）
   - 再单独试 CosineAnnealing
3. **特别关注 MAPE**：37% 的 MAPE 说明大量低值样本被严重偏估，需要检查 LogCosh 在 label_scale=1000 下的梯度行为是否合理。
4. **SWA 模型还未测试**：v3 保存了 `swa_2dcnn_model.pth`，建议单独用此权重跑一次测试对比，SWA 权重可能优于默认 EMA 权重。
