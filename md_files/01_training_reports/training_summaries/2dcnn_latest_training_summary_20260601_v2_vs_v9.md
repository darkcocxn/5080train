# 2D-CNN 最新训练结果总结（2dcnnv2 vs 2dcnnv9）

生成日期：2026-06-01  
项目目录：`X:\pyproject\Remote-Train`

## 1. 本次检查对象

### 1.1 训练成果目录

| 版本 | 目录 | 说明 |
|---|---|---|
| 2dcnnv2 | `output/2dcnnv2/model-nostamp-6f475c7d-train20260601-014156` | 当前测试集上综合指标更优，建议作为当前主结果 |
| 2dcnnv9 | `output/2dcnnv9/model-nostamp-6f475c7d-train20260601-024158` | 更偏保守 tail 选择，验证集 tail 目标略优，但测试集整体指标略差 |

### 1.2 数据集文件

用户给定 wave split 文件：

`newdata/olddata/opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258_wave_split.csv`

训练 metadata 实际指向的数据集基名：

`opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260601-001319-drop-split-rank-top5drift`

两份 wave split 文件内容一致，SHA256 均为：

`29CC46BEBEC068B4C097584C2B32111D3F994D6D4DE54B418328393D5DB4BB8D`

因此，用户给定的 `20260531-181258_wave_split.csv` 可作为本次 wave 划分依据；训练实际使用的是在此划分基础上生成的 `drop-split-rank-top5drift` train/val/test 数据表。

## 2. 测试集结果对比

### 2.1 全量测试集指标

| 模型 | 样本数 | R2 | MAE | RMSE | MAPE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| 2dcnnv2 | 3116 | 0.907476 | 0.000288331 | 0.000412856 | 17.16% | -0.000061087 |
| 2dcnnv9 | 3116 | 0.900106 | 0.000298733 | 0.000428985 | 18.06% | -0.000047841 |

去除重复 `sample_id` 后：

| 模型 | 样本数 | R2 | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|---:|
| 2dcnnv2 | 3112 | 0.908706 | 0.000286858 | 0.000409895 | -0.000063009 |
| 2dcnnv9 | 3112 | 0.902002 | 0.000296874 | 0.000424677 | -0.000050145 |

结论：去重不改变排序，`2dcnnv2` 仍优于 `2dcnnv9`。

### 2.2 多 seed 80% 重采样指标

| 模型 | R2 mean | R2 std | MAE mean | MAE std | RMSE mean | RMSE std | MAPE mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2dcnnv2 | 0.908377 | 0.000842 | 0.000287159 | 0.000001821 | 0.000411001 | 0.000001734 | 16.917% |
| 2dcnnv9 | 0.900878 | 0.001223 | 0.000297984 | 0.000002175 | 0.000427487 | 0.000002820 | 17.819% |

## 3. 按 drift 区间拆分

| True drift 区间 | 测试样本数 | 2dcnnv2 MAE | 2dcnnv2 Bias | 2dcnnv2 欠预测率 | 2dcnnv9 MAE | 2dcnnv9 Bias | 2dcnnv9 欠预测率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `<0.001` | 665 | 0.0001071 | 0.0000066 | 42.4% | 0.0001055 | 0.0000150 | 41.1% |
| `0.001-0.002` | 1309 | 0.0002203 | -0.0000262 | 54.0% | 0.0002311 | -0.0000316 | 54.2% |
| `0.002-0.003` | 621 | 0.0003581 | 0.0000051 | 51.2% | 0.0003830 | 0.0000235 | 46.5% |
| `0.003-0.005` | 378 | 0.0005049 | -0.0001530 | 63.0% | 0.0005155 | -0.0000808 | 57.4% |
| `0.005-0.010` | 143 | 0.0008787 | -0.0007399 | 81.8% | 0.0008771 | -0.0007112 | 81.8% |
| `>=0.010` | 0 | - | - | - | - | - | - |

高 drift 区间仍是主要误差来源。`0.005-0.010` 区间只有 143 条，但 MAE 接近 `8.8e-4`，且两个模型欠预测率均为 81.8%。这说明模型对中高响应段仍存在明显均值回归。

## 4. 数据分布与 tail 问题

### 4.1 train/val/test 分布

| split | 样本数 | unique waves | drift mean | drift p95 | drift max | `>=0.005` | `>=0.010` | `steel01_yielded=1` | `steel02_yielded=1` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 67260 | 163 | 0.0028423 | 0.0064711 | 0.0082426 | 9268 | 0 | 0 | 44840 |
| val | 3420 | 17 | 0.0025890 | 0.0095487 | 0.0144653 | 422 | 152 | 152 | 2111 |
| test | 3116 | 14 | 0.0019703 | 0.0049115 | 0.0073622 | 143 | 0 | 0 | 2048 |

关键现象：

1. 训练集没有 `>=0.010` 样本，测试集也没有，但验证集有 152 条。
2. 验证集的 `>=0.010` 样本正好对应 `steel01_yielded=1`，而 train/test 中 `steel01_yielded=1` 均为 0。
3. `2dcnnv9` 设计了 `0.01/0.02/0.03` 的 tail 分类阈值，但训练正样本数为 0，分类头没有实际正样本可学。
4. 因为验证集中存在 train 未覆盖的 `>=0.010`/`steel01_yielded=1` 组合，`2dcnnv9` 的保守 tail 选择会在验证目标上更受激励，但该策略没有在测试集上转化为整体收益。

### 4.2 重复样本

| split | 总行数 | unique sample_id | 重复行数 |
|---|---:|---:|---:|
| train | 67260 | 66755 | 1010 |
| val | 3420 | 3280 | 221 |
| test | 3116 | 3112 | 6 |

重复样本不会改变当前两个模型的排序，但会影响采样权重、验证选择与 tail 统计。后续数据生成建议在训练前按 `sample_id` 去重，或明确保留重复的物理含义。

## 5. 训练过程解读

### 5.1 2dcnnv2

| 项 | 数值 |
|---|---:|
| epoch 数 | 50 |
| best epoch | 30 |
| best val selection score | 0.000622614 |
| best val MAE | 0.000497610 |
| best val loss | 0.269193 |
| best val focus score | 0.004714 |
| 总训练时间 | 59.52 min |

训练曲线显示：前 5 个 epoch 快速下降，约 20-35 epoch 后验证指标基本平台化。训练 loss 后续继续下降，但验证 loss 与 MAE 改善很小，说明主要瓶颈不是训练轮数不足，而是数据分布与 tail 覆盖不足。

### 5.2 2dcnnv9

| 项 | 数值 |
|---|---:|
| epoch 数 | 35 |
| best epoch | 19 |
| 默认选择 checkpoint | `best_2dcnn_focus_model.pth`，epoch 24 |
| best val selection score | 0.002470534 |
| best val MAE | 0.000489083 |
| best val loss | 0.261455 |
| best val focus score | 0.005736 |
| 总训练时间 | 47.03 min |

`2dcnnv9` 在验证集 MAE/RMSE/R2 上优于 `2dcnnv2`，但测试集变差。原因是 v9 强化了保守 tail 选择：

- sampler multiplier 从 1.10 增至 1.25；
- tail loss 权重与上限更高；
- validation selection 对 tail MAE、extreme tail underprediction 的惩罚更重；
- 增加 `0.03` severe tail 设定；
- tail correction gate 更容易放大高响应预测。

这些改动理论上有利于极端响应，但当前训练集中没有 `>=0.01` 正样本，测试集中也没有 `>=0.01` 工况。因此 v9 的改动主要表现为普通区间预测略扰动，未形成明确收益。

## 6. 结构与材料机理解释

本数据集使用 OpenSees 非线性时程分析生成响应，目标为最大层间位移角。结果误差集中在屈服和高响应区间，与结构非线性机制一致：

1. `Steel01` 是双线性钢材模型，包含运动硬化和可选各向同性硬化。屈服后刚度发生分段变化，最大 drift 对刚度、强度、质量、楼层数和阻尼器布置更敏感。
2. `Steel02` 是 Giuffre-Menegotto-Pinto 滞回钢材模型，包含从弹性到塑性分支的平滑转变，并可考虑各向同性硬化。滞回路径和历史变量会增强响应的路径依赖。
3. 测试集中 `steel02_yielded=True` 有 2048 条，占 65.7%，说明大量样本已进入阻尼器或附加钢材非线性状态。
4. wave 特征中 PGA、RMS、CAV、Arias proxy、duration、predominant period 与结构周期比共同控制输入能量和共振风险。模型加入了这些派生特征是合理的，但高响应样本稀疏时，纯监督回归仍会偏向主密度区域。

## 7. 模型设计评价

当前模型采用 scalogram 图像 + 标量特征融合：

- 图像分支：2D CNN 处理时频图，适合提取地震波时频局部特征；
- 标量分支：楼层数、质量、高度、刚度、强度、周期、wave intensity、周期比、阻尼器布置等；
- 融合方式：FiLM/residual scalar fusion + gated bilinear fusion；
- 损失：SmoothL1Loss + target weighting + tail underprediction penalty；
- 优化：AdamW + ReduceLROnPlateau + EMA。

这种设计方向是合理的。文献中也有将地震动转为时频图并用 2D CNN 预测非线性结构响应的做法；FiLM 的思想也适合让结构标量条件调制图像特征。但当前性能瓶颈主要不在网络表达能力，而在数据划分与 tail 覆盖。

## 8. 当前结论

1. 当前测试集上建议采用 `2dcnnv2` 作为主模型结果。
2. `2dcnnv9` 不是无效版本，而是目标函数更偏“保守 tail”。它在验证集 tail 目标上更积极，但由于 train/test 没有 `>=0.01` 样本，无法在当前测试集上体现优势。
3. 两个模型在 `0.005-0.010` drift 区间都存在明显欠预测，说明 tail 学习仍不足。
4. 验证集包含 train/test 未覆盖的 `steel01_yielded=1` 和 `>=0.01` 工况，导致验证选择与测试目标不完全一致。
5. 后续提升优先级应是数据层面高于模型层面。

## 9. 后续建议

### 9.1 数据划分

建议重新分层划分 train/val/test，分层变量至少包括：

- `wave_cluster`
- `max_drift_ratio_raw` bin：`<0.001`、`0.001-0.002`、`0.002-0.003`、`0.003-0.005`、`0.005-0.010`、`>=0.010`
- `steel01_yielded`
- `steel02_yielded`
- `num_floors`

目标是让 train/val/test 都覆盖中高 drift 与屈服状态，避免验证集出现训练集完全没有的正类。

### 9.2 tail 阈值

在训练集最大 drift 只有 0.00824 的情况下，不建议继续使用 `0.01/0.02/0.03` 作为主要 tail 分类阈值。更合理的当前阈值是：

- mid tail：`>=0.003`
- tail：`>=0.005`
- high tail：`>=0.007`

等数据集中真实出现 `>=0.01` 工况后，再恢复 `0.01/0.02/0.03` 的工程阈值。

### 9.3 loss 与指标

建议同时报告：

- full test MAE/RMSE/R2/MAPE；
- 按 drift bin 的 MAE/Bias/欠预测率；
- `steel02_yielded=True/False` 分组；
- `num_floors` 分组；
- worst wave top-N；
- `>=0.005` tail recall-like 指标：例如高响应样本中预测低估超过 20% 的比例。

只看全局 MAE 会掩盖高响应区间风险。

### 9.4 数据清理

建议在进入训练前执行：

- 按 `sample_id` 去重；
- 检查 `round_idx` 与重复样本是否代表真实重复试验；
- 确认 `image_path` 在 train/val/test 中一致可解析；
- 保留 wave split 的 hash 和生成脚本参数。

## 10. 参考文献与技术手册

1. OpenSees Documentation. **Steel01 Material**.  
   https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/Steel01.html

2. OpenSees Documentation. **Steel02 Material**.  
   https://opensees.github.io/OpenSeesDocumentation/user/manual/material/uniaxialMaterials/Steel02.html

3. OpenSees Documentation. **Uniform Excitation Pattern**.  
   https://opensees.github.io/OpenSeesDocumentation/user/manual/model/pattern/uniformExcitationPattern.html

4. PyTorch Documentation. **SmoothL1Loss**.  
   https://docs.pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html

5. PyTorch Documentation. **AdamW**.  
   https://docs.pytorch.org/docs/stable/generated/torch.optim.adamw.AdamW_class.html

6. PyTorch Documentation. **ReduceLROnPlateau**.  
   https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html

7. Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. **FiLM: Visual Reasoning with a General Conditioning Layer**. arXiv:1709.07871, 2017.  
   https://arxiv.org/abs/1709.07871

8. Ning, C., Xie, Y., & Sun, L. **LSTM, WaveNet, and 2D CNN for nonlinear time history prediction of seismic responses**. Engineering Structures, 286, 116083, 2023.  
   https://doi.org/10.1016/j.engstruct.2023.116083

9. Campbell, K. W., & Bozorgnia, Y. **A Comparison of Ground Motion Prediction Equations for Arias Intensity and Cumulative Absolute Velocity Developed Using a Consistent Database and Functional Form**. Earthquake Spectra, 2012.  
   https://doi.org/10.1193/1.4000067

10. Campbell, K. W., & Bozorgnia, Y. **Ground Motion Models for the Horizontal Components of Arias Intensity and Cumulative Absolute Velocity Using the NGA-West2 Database**. Earthquake Spectra, 2019.  
    https://doi.org/10.1193/090818EQS212M

