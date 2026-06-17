# 强化标量分支与融合结构改动说明

## 背景

上一轮模型的整体测试指标已经比较稳定：

| 指标 | 数值 |
|---|---:|
| R2 | 0.8586 |
| MAE | 0.0006872 |
| RMSE | 0.0013411 |
| MAPE | 19.33% |

但分层分析显示，模型在高漂移尾部仍然存在系统性低估：

| 真实漂移范围 | 样本数 | MAE | Bias | 低估率 |
|---|---:|---:|---:|---:|
| 全部 | 7332 | 0.000687 | +0.000043 | 46.4% |
| >= 0.010 | 271 | 0.003339 | -0.002535 | 76.8% |
| >= 0.020 | 23 | 0.012123 | -0.012123 | 100.0% |

当前数据中训练样本约 75009 条，但唯一小波图只有 167 张；大量样本共享同一地震波小波图，差异主要来自结构标量、阻尼布置和派生特征。因此本次优先强化标量分支和图像-标量融合，而不是直接加深 CNN。

## 新增文件

- `多模态-2DCNN-三阶段数据集-3到7层-强化标量融合.py`
- `2DCNN测试-三阶段数据集-3到7层-强化标量融合.py`
- `强化标量融合改动说明.md`

## 训练脚本改动

训练脚本基于原 `多模态-2DCNN-三阶段数据集-3到7层.py` 复制后修改，数据处理、采样、loss、AMP、早停、checkpoint 保存逻辑基本保持一致，主要改网络结构。

### 独立输出目录

新模型默认保存到：

```text
model-2dcnn-3stage-rf-3to7-strong-scalar-fusion
```

这样不会覆盖旧模型目录 `model-2dcnn-3stage-rf-3to7`。

### 标量分支增强

原标量分支：

```text
63 -> 64 -> 128
```

新标量分支：

```text
63 -> 192 -> 4 个 ScalarResidualBlock -> 192
```

每个 `ScalarResidualBlock` 使用：

```text
LayerNorm -> Linear(192 -> 384) -> GELU -> Dropout -> Linear(384 -> 192) -> Dropout -> residual add
```

并使用可学习的 `residual_scale`，初始值为 `0.20`，让新残差分支在训练早期更稳。

### 融合层增强

原融合方式：

```text
concat(image_embedding, scalar_embedding)
```

新融合方式：

```text
image_embedding: 256
scalar_embedding: 192

1. scalar -> image gate
2. image -> scalar gate
3. low-rank bilinear interaction: 128
4. concat(gated_image, gated_scalar, bilinear)
5. projection -> 384
```

目标是让模型更容易学习：

```text
地震波特征 x 结构周期 x 刚度 x 楼层 x 阻尼布置
```

这类乘性交互，而不仅仅依赖手工派生特征。

### Head 与尾部辅助头

原主回归 head：

```text
384 -> 256 -> 64 -> 1
```

新主回归 head：

```text
384 -> 384 -> 128 -> 1
```

激活函数从 ReLU 改为 GELU。

尾部修正头和尾部分类头 hidden dim 从 `64` 提到 `96`，保持原有逻辑：

- `tail_correction_head` 仍然输出正向修正项，减少高漂移低估风险。
- `tail_classifier_head` 仍然预测 `>=0.010` 和 `>=0.020` 两个尾部阈值。

## 测试脚本改动

新测试脚本基于原 `2DCNN测试-三阶段数据集-3到7层.py` 复制后修改。

主要差异：

- 默认模型根目录改为 `model-2dcnn-3stage-rf-3to7-strong-scalar-fusion`
- 默认网络结构与新训练脚本一致
- 新增 `ScalarResidualBlock`
- 新增 `ScalarFeatureEncoder`
- 新增 `GatedBilinearFusionBlock`
- `load_training_metadata()` 会读取新结构字段：
  - `scalar_encoder`
  - `scalar_embed_dim`
  - `scalar_res_blocks`
  - `scalar_res_hidden_mult`
  - `scalar_res_dropout`
  - `fusion_mode`
  - `fusion_bilinear_dim`
  - `fusion_output_dim`
  - `fusion_dropout`

测试脚本会优先使用训练目录中的 `training_metadata.json` 配置网络，因此后续如果微调维度，只要 metadata 正确，测试脚本也能同步构建模型。

## 运行方式

训练：

```powershell
uv run python ".\多模态-2DCNN-三阶段数据集-3到7层-强化标量融合.py"
```

测试：

```powershell
uv run python ".\2DCNN测试-三阶段数据集-3到7层-强化标量融合.py"
```

测试不同 checkpoint：

```powershell
$env:SURMOD_2DCNN_WEIGHTS_NAME="best_2dcnn_focus_model.pth"
uv run python ".\2DCNN测试-三阶段数据集-3到7层-强化标量融合.py"
```

可选权重名包括：

- `best_2dcnn_model.pth`
- `best_2dcnn_mae_model.pth`
- `best_2dcnn_focus_model.pth`
- `best_2dcnn_extreme_under_model.pth`

## 对比建议

建议至少比较以下指标：

| 对比项 | 关注点 |
|---|---|
| 全量 R2 / MAE / RMSE | 整体拟合是否提升 |
| `true >= 0.010` MAE / Bias / 低估率 | 中高尾部是否改善 |
| `true >= 0.020` MAE / Bias / 低估率 | 极端尾部是否仍全低估 |
| F7 分层误差 | 最大误差是否仍集中在 7 层 |
| `test_tail_topup` 分层误差 | 尾部补充测试集是否改善 |

如果新模型整体 MAE 小幅波动但尾部低估率明显下降，需要根据工程目标决定是否接受。若目标偏安全评估，尾部 Bias 和低估率应优先于全量 MAE。

## 当前验证

已完成静态和前向传播检查：

```text
python -m py_compile 通过
uv run python 前向传播通过
pred shape: (2, 1)
tail_logits shape: (2, 2)
params: 2,482,473
```

原模型约 1.32M 参数，新模型约 2.48M 参数，增量主要来自残差标量编码器和门控双线性融合层。
## 2026-05-29 二次优化记录

本次复查最新训练结果后，发现强化标量融合版本在极端高漂移样本上略有改善，但整体指标明显回退：测试集均值 R2 约 0.802，MAE 约 0.000845，RMSE 约 0.001591，MAPE 约 22.24%。训练记录显示最佳验证 MAE 出现在第 10 轮，之后训练损失继续下降而验证指标回退，说明新增标量分支和双线性交互存在过拟合与校准不稳。

因此训练脚本做了保守化调整：

- 标量分支从 `192 x 4` 个残差块收缩为 `160 x 3`，并提高输入/残差 dropout。
- 双线性交互维度从 `128` 降到 `64`，融合输出从 `384` 降到 `320`。
- 新增可学习的 `fusion.interaction_scale`，初始值 `0.35`，限制双线性交互早期不要压过主干特征。
- 预测头从 `[384, 128]` 调整为 `[320, 96]`，尾部校正/分类头隐藏维度从 `96` 降到 `80`。
- 启用 EMA，`EMA_DECAY=0.995`，最佳 checkpoint 默认保存 EMA 权重。
- 学习率从 `1.5e-4` 降到 `1.2e-4`，weight decay 从 `3e-5` 提高到 `8e-5`。
- 早停 patience 从 `28` 降到 `20`，scheduler patience 从 `6` 降到 `5`。
- 最佳权重选择由纯 MAE 改成 `MAE + 0.02 * focus_score`，保留整体误差优先，同时给高值尾部一点约束。

对应测试脚本已同步新结构参数，并对旧 checkpoint 兼容：旧权重没有 `fusion.interaction_scale` 时会使用默认初始值加载。
