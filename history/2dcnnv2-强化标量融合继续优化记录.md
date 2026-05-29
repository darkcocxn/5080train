# 强化标量融合继续优化记录 2026-05-29

## 复查结论

最新已完成的正则化强化融合训练目录：

```text
model-2dcnn-3stage-rf-3to7-strong-scalar-fusion/model-20260523-171351-541b9ed7-train20260529-155503
```

这轮结果说明上一版“收缩容量 + EMA + 轻尾部约束”的方向是对的：

- seed 平均：`R2=0.8659, MAE=0.0006868, RMSE=0.0013088, MAPE=20.97%`。
- 全量测试：`R2=0.8677, MAE=0.0006845, RMSE=0.0012971`，整体已经略优于旧版 `R2=0.8586, MAE=0.0006872, RMSE=0.0013411`。
- 极端尾部 `true >= 0.020` 的 MAE 从旧版 `0.012123` 降到 `0.009554`，低估率从 `100.0%` 降到 `91.30%`。
- 中高值段仍有优化空间：`true >= 0.005` 的 MAE 为 `0.001264`，略高于旧版 `0.001233`；`true >= 0.010` 的 MAE 为 `0.003370`，也略高于旧版 `0.003339`。

因此现在不是推翻上一版，而是继续做校准：保留极端尾部改善，同时减少 `0.005~0.010` 和 `0.010~0.020` 中高值段被 correction 误抬高或排序扰动的风险。

## 本次继续优化内容

修改文件：

```text
多模态-2DCNN-三阶段数据集-3到7层-强化标量融合.py
2DCNN测试-三阶段数据集-3到7层-强化标量融合.py
```

### 1. 进一步放缓训练

```text
LEARNING_RATE: 1.2e-4 -> 1.0e-4
WEIGHT_DECAY: 8e-5 -> 1.0e-4
SCHEDULER_PATIENCE: 5 -> 4
EMA_DECAY: 0.995 -> 0.998
```

目的：降低后期验证抖动，让 EMA 更像跨多个 epoch 的平滑权重，而不是只跟随最近小段 batch。

### 2. 降低尾部低估惩罚的早期干扰

```text
TAIL_UNDERPREDICTION_START_EPOCH: 5 -> 7
TAIL_UNDERPREDICTION_RAMP_EPOCHS: 8 -> 10
TAIL_UNDERPREDICTION_WEIGHT: 0.18 -> 0.16
EXTREME_TAIL_UNDERPREDICTION_WEIGHT: 0.50 -> 0.46
TAIL_UNDERPREDICTION_MAX_LOSS: 2.2 -> 2.0
```

目的：先让模型学全局主趋势，再逐步加入尾部低估约束，避免训练早期就把中等样本整体推高。

### 3. 继续收紧融合和预测头

```text
SCALAR_RES_DROPOUT: 0.16 -> 0.17
FUSION_DROPOUT: 0.16 -> 0.17
FUSION_INTERACTION_SCALE_INIT: 保持 0.35
HEAD_DROPOUT: 0.22 -> 0.23
```

目的：保留 `155503` 中已经有效的双线性交互强度，只略微提高 dropout，避免把有效增益一并砍掉。

### 4. 新增尾部概率门控 correction

新增配置：

```text
USE_TAIL_PROB_GATED_CORRECTION = True
TAIL_PROB_GATE_INDEX = 0
TAIL_PROB_GATE_DETACH = True
TAIL_PROB_GATE_POWER = 1.0
```

原逻辑：

```text
pred = base_pred + sigmoid(correction_gate) * softplus(correction)
```

新逻辑：

```text
tail_prob = sigmoid(tail_classifier_logits[:, 0])
pred = base_pred + sigmoid(correction_gate) * tail_prob * softplus(correction)
```

其中 `tail_prob` 默认 detach，不让回归损失直接把分类概率推高。这样 correction 主要在模型认为样本属于 `>=0.010` 尾部风险时才发挥作用。

测试脚本做了旧 checkpoint 兼容：旧 metadata 没有该字段时，默认保持旧 correction 逻辑，不启用概率门控。

### 5. 降低尾部分类辅助压力

```text
TAIL_CLASSIFICATION_LOSS_WEIGHTS: [0.025, 0.050] -> [0.022, 0.044]
TAIL_CLASSIFICATION_POS_WEIGHT_MAX: 30.0 -> 26.0
TAIL_CLASSIFICATION_RAMP_EPOCHS: 8 -> 10
```

目的：尾部分类继续作为 correction 门控信号，但避免极少数 `>=0.020` 样本让分类头过度激进。

### 6. 最佳模型选择更偏整体指标

```text
SELECTION_FOCUS_WEIGHT: 0.02 -> 0.018
```

最佳权重仍然按：

```text
selection_score = MAE + SELECTION_FOCUS_WEIGHT * focus_score
```

但尾部 focus 的占比更小，防止为了极端尾部牺牲整体 MAE。

## 预期效果

这版不是追求更强表达，而是在 `155503` 这轮成功结果基础上继续做校准：

- 整体 MAE/R2 应尽量维持或小幅优于 `155503`。
- `true >= 0.005` 和 `true >= 0.010` 的 MAE 应向旧版靠近或超过旧版。
- `true >= 0.020` 极端高值可能比 `155503` 略保守，但目标仍是明显优于旧版完全低估的状态。
- MAPE 应争取从 `155503` 的约 `20.97%` 往旧版 `19.33%` 靠近。
- 如果训练曲线显示最佳 epoch 仍很早出现，可以继续降低融合输出维度或进一步减弱 tail loss。

## 建议运行

训练：

```powershell
uv run python "多模态-2DCNN-三阶段数据集-3到7层-强化标量融合.py"
```

测试：

```powershell
uv run python "2DCNN测试-三阶段数据集-3到7层-强化标量融合.py"
```

重点查看：

- `test-*_seed_metrics.csv`
- `test-*_results.csv`
- `training_history.json`
- `training_curves.png`

分层对比时优先看：

- 全部样本 MAE / R2
- `true >= 0.005`
- `true >= 0.010`
- `true >= 0.020`
- 6 层、7 层样本 MAE

## 脚本验证

已完成：

- `py_compile` 语法检查通过。
- 新训练脚本可实例化并完成 2 条样本的 forward。
- 新结构参数量仍为 `1,955,097`。
- 测试脚本可严格加载 `155503` 旧 checkpoint；由于旧 metadata 没有 `tail_prob_gated_correction` 字段，兼容逻辑会保持旧行为。

## 2026-05-29 继续优化：基于 165419 训练反馈

最新有效训练目录：

```text
model-2dcnn-3stage-rf-3to7-strong-scalar-fusion/model-20260523-171351-541b9ed7-train20260529-165419
```

该轮结果继续提升：

```text
seed mean: R2=0.8694, MAE=0.0006501, RMSE=0.0012917, MAPE=19.58%
full test: R2=0.8722, MAE=0.0006465, RMSE=0.0012753
```

分段表现：

```text
true >= 0.005: MAE=0.001192, 优于旧版 0.001233
true >= 0.010: MAE=0.003392, 略高于旧版 0.003339
true >= 0.020: MAE=0.009992, 明显优于旧版 0.012123
```

结论：整体方向有效，新的主要矛盾变成 `>=0.010` 中高值段与 7 层极端点的精细校准。

### 网络与训练脚本新增优化

本次在训练脚本中继续做小步修改：

```text
TAIL_PROB_GATE_POWER: 1.0 -> 1.20
USE_EXTREME_PROB_GATE_BLEND = True
TAIL_EXTREME_PROB_GATE_INDEX = 1
TAIL_EXTREME_PROB_GATE_FLOOR = 0.65
TAIL_CLASSIFICATION_LOSS_WEIGHTS: [0.022, 0.044] -> [0.018, 0.048]
SELECTION_MID_TAIL_WEIGHT = 0.04
VAL_MID_TAIL_LOW = 0.005
VAL_MID_TAIL_HIGH = 0.010
```

含义：

- correction 仍先看 `>=0.010` 尾部概率，但概率会做 `power=1.20` 压缩，降低中等样本误触发 correction 的概率。
- correction 再乘一个极端尾部概率混合门：`0.65 + 0.35 * P(y>=0.020)`。这样极端风险高的样本保留更多修正，普通中高值样本修正更克制。
- 第一档尾部分类 loss 降低，第二档极端尾部分类 loss 略升，减少 `0.005~0.010` 假阳性，同时保留极端尾部识别。
- validation history 新增 `val_mid_tail_mae_raw / bias / under_mae / count`。
- selection score 从

```text
MAE + 0.018 * focus_score
```

扩展为：

```text
MAE + 0.018 * focus_score + 0.04 * mid_tail_mae
```

目标是避免最佳 checkpoint 只对整体或极端尾部好，而忽略 `0.005~0.010` 的稳定性。

### 启动脚本与 Git 脚本简化

已修改：

```text
启动训练.bat
启动测试.bat
git-pull-merge.ps1
git-commit-push.ps1
拉取与合并.bat
提交并推送.bat
```

修改后：

- `启动训练.bat` 固定启动强化标量融合训练脚本，不再自动搜索/自定义选择脚本。
- `启动测试.bat` 固定启动强化标量融合测试脚本。
- `git-pull-merge.ps1` 只执行当前分支 `git pull`，不再提供 stash、fetch、merge 方式等菜单。
- `git-commit-push.ps1` 只执行当前分支 `git push`，不再 stage、commit 或选择 remote/branch。
- 两个 `.bat` Git 入口不再透传自定义参数。
