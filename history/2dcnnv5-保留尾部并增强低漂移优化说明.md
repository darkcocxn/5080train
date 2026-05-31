# 2dcnnv5 保留尾部优化并增强低漂移准确性的改动说明

生成日期：2026-05-31

## 1. 改动目标

v5 从当前表现最好的 v2 版本派生，而不是从 v4 继续修改。

v2 最好测试结果：

```text
模型目录: output/model-20260523-171351-541b9ed7-train20260529-182601
R2   = 0.8806
MAE  = 0.000632
RMSE = 0.001235
MAPE = 18.99%
```

v4 测试结果：

```text
模型目录: output/2dcnnv4/model-20260523-171351-541b9ed7-train20260531-132505
R2   = 0.8520
MAE  = 0.000676
RMSE = 0.001375
MAPE = 17.72%
```

v4 的经验说明：单纯关闭尾部压力、改为核心区间选模后，低漂移相对误差有所改善，但整体 `R2/MAE/RMSE` 和高漂移尾部识别均倒退。因此 v5 的原则是：

1. 保留 v2 已验证有效的尾部优化机制。
2. 不再像 v4 那样取消 weighted sampler、tail loss、tail correction。
3. 只增加轻量低漂移约束，目标是降低 `true < 0.001` 区间误差和 MAPE。
4. 验证选模同时约束低漂移、整体误差、中尾部和极端尾部，避免顾此失彼。

## 2. 新增文件

```text
2dcnnv5/2dcnnv5.py
2dcnnv5/2dcnnv5test.py
2dcnnv5/启动训练.bat
2dcnnv5/启动测试.bat
```

模型输出目录：

```text
output/2dcnnv5
```

## 3. 保留的 v2 机制

v5 完整保留以下 v2 机制：

| 机制 | v5 状态 | 作用 |
|---|---|---|
| `WeightedRandomSampler` | 保留 | 提高中高漂移稀有样本采样概率 |
| density weighted loss | 保留 | 按标签密度提高稀有连续响应区间权重 |
| tail loss multiplier | 保留 | 对 `>=0.005`、`>=0.010`、`>=0.020` 样本加权 |
| tail underprediction loss | 保留 | 惩罚高漂移样本低估，降低危险样本漏报 |
| tail correction head | 保留 | 高风险样本输出校正 |
| tail probability gated correction | 保留 | 用尾部分类概率控制校正幅度 |
| extreme probability gate blend | 保留 | 保护极端尾部 |
| tail classification auxiliary head | 保留 | 辅助学习 `>=0.010` 和 `>=0.020` 尾部概率 |
| EMA | 保留 | 提高验证与测试稳定性 |
| SmoothL1Loss | 保留 | 对异常点较稳健 |

这些机制是 v2 相比 v4 仍保持更高 `R2/MAE/RMSE` 和更好尾部 F1 的核心原因。

## 4. v5 新增低漂移准确性机制

### 4.1 新增低漂移相对精度损失

新增配置：

```python
USE_LOW_DRIFT_ACCURACY_LOSS = True
LOW_DRIFT_THRESHOLD = 0.001
LOW_DRIFT_RELATIVE_EPS = 0.00035
LOW_DRIFT_ACCURACY_WEIGHT = 0.028
LOW_DRIFT_ACCURACY_START_EPOCH = 1
LOW_DRIFT_ACCURACY_RAMP_EPOCHS = 8
LOW_DRIFT_ACCURACY_MAX_LOSS = 0.45
```

损失形式：

```text
low_mask = y_true < 0.001
normalized_error = (y_pred - y_true) / (abs(y_true) + 0.00035)
L_low = 0.028 * SmoothL1(normalized_error, 0)
```

设计意图：

1. 只作用于 `true < 0.001` 的低漂移样本。
2. 用 `abs(y_true) + eps` 归一化误差，减少小分母导致的 MAPE 爆炸。
3. 使用较小权重 `0.028`，避免破坏 v2 的尾部优化。
4. 前 8 个 epoch 线性 ramp-up，避免训练初期低漂移约束过早主导梯度。
5. 使用 `SmoothL1` 而非 MSE，避免个别极小真实值样本产生过大梯度。

### 4.2 新增低漂移验证指标

新增验证历史字段：

```text
val_low_drift_mae_raw
val_low_drift_bias_raw
val_low_drift_under_mae_raw
val_low_drift_over_mae_raw
val_low_drift_norm_mae
val_low_drift_mape
val_low_drift_count
```

其中：

```text
val_low_drift_norm_mae = mean(abs(error) / (abs(y_true) + 0.00035))
```

它比普通 MAPE 更稳健，因为分母加入了工程容忍尺度 `0.00035`。

### 4.3 修改验证选模指标

v2 原选模：

```text
selection_score =
    global_mae
  + 0.018 * focus_score
  + 0.04  * mid_tail_mae
```

v5 新选模：

```text
selection_score =
    global_mae
  + 0.018 * focus_score
  + 0.14  * low_drift_mae
  + 0.12  * low_drift_norm_mae * 0.00035
  + 0.04  * mid_tail_mae
```

这样做的目的：

1. `global_mae` 仍是主指标，避免整体倒退。
2. `focus_score` 保留尾部 MAE、极端尾部 MAE、尾部低估误差。
3. `low_drift_mae` 专门约束低漂移绝对误差。
4. `low_drift_norm_mae` 约束低漂移相对误差，但乘以 `eps` 拉回 drift ratio 量纲，避免支配选模。
5. `mid_tail_mae` 保留 v2 对 `0.005 <= true < 0.010` 区间的保护。

### 4.4 新增低漂移最佳 checkpoint

v5 新增保存：

```text
best_2dcnn_low_drift_model.pth
```

该 checkpoint 按 `val_low_drift_norm_mae` 最优保存，用于后续对比：

1. `best_2dcnn_model.pth`：综合选模。
2. `best_2dcnn_mae_model.pth`：全局 MAE 最优。
3. `best_2dcnn_focus_model.pth`：尾部 focus 最优。
4. `best_2dcnn_extreme_under_model.pth`：极端尾部低估最优。
5. `best_2dcnn_low_drift_model.pth`：低漂移归一化误差最优。

## 5. 预期效果

v5 的目标不是大幅改变模型结构，而是对 v2 做低风险修正。

理想结果：

```text
R2 >= v2 - 0.005
MAE <= v2
RMSE <= v2 或基本持平
MAPE < v2
true < 0.001 区间 MAE/MAPE 下降
MIDR >= 0.010 的 F1 不低于 v2
```

如果结果出现以下情况，应回退或调小低漂移权重：

```text
R2 明显低于 0.875
MAE 高于 0.00065
MIDR >= 0.010 的 F1 继续下降
true >= 0.010 的 tail bias 更负
```

优先调参顺序：

1. 若尾部变差：降低 `LOW_DRIFT_ACCURACY_WEIGHT` 到 `0.015-0.020`。
2. 若低漂移仍无改善：提高 `SELECTION_LOW_DRIFT_MAE_WEIGHT`，不要先提高训练 loss 权重。
3. 若低漂移过预测明显：降低 `LOW_DRIFT_RELATIVE_EPS` 或单独加入 overprediction 监控，不建议直接加大惩罚。
4. 若整体 MAE 变差：降低 `SELECTION_LOW_DRIFT_NORM_WEIGHT`。

## 6. 运行方式

训练：

```powershell
uv run python .\2dcnnv5\2dcnnv5.py
```

测试：

```powershell
uv run python .\2dcnnv5\2dcnnv5test.py
```

也可以双击：

```text
2dcnnv5/启动训练.bat
2dcnnv5/启动测试.bat
```

## 7. 训练后重点对比

与 v2 对比：

```text
output/model-20260523-171351-541b9ed7-train20260529-182601
```

重点看：

```text
test-*_seed_metrics.csv
test-*_results.csv
training_history.json
training_curves.png
```

推荐比较表：

| 指标 | v2 | v5 |
|---|---:|---:|
| R2 | 0.8806 | 待训练 |
| MAE | 0.000632 | 待训练 |
| RMSE | 0.001235 | 待训练 |
| MAPE | 18.99% | 待训练 |
| true < 0.001 MAE | 0.0000886 | 待训练 |
| true < 0.001 MAPE | 29.09% | 待训练 |
| F1@0.005 | 0.916 | 待训练 |
| F1@0.010 | 0.541 | 待训练 |
| F1@0.020 | 0.421 | 待训练 |

## 8. 优化依据

### 8.1 实验依据

v2 与 v4 的对比说明：

1. v2 的尾部机制有效，不能轻易移除。
2. v4 在低漂移 MAPE 上有改善，但整体 `R2/MAE/RMSE` 下降。
3. 低漂移优化应以轻量正则形式加入，而不是替代尾部优化目标。

### 8.2 机器学习依据

当前任务是连续响应回归，并且标签分布明显不均衡：

1. 高漂移样本稀少，但工程风险高。
2. 低漂移样本数量多，但真实值接近 0，MAPE 对小分母敏感。
3. 因此 v5 同时使用 density weighted loss、tail loss 和低漂移归一化误差约束。

DenseLoss/DenseWeight 文献说明，在不均衡回归中，可以根据目标值密度对样本损失加权，使稀有响应区间获得更高训练影响。v2/v5 的 density weighted loss 与这个思想一致。

Huber/SmoothL1 类损失适合当前任务，因为它在小误差区间近似二次损失、在大误差区间近似一次损失，比纯 MSE 更不容易被少数极端误差支配。

MAPE 在真实值接近 0 时会失真，因此 v5 没有直接把 MAPE 放进训练损失，而是使用带 `eps` 的归一化误差：

```text
abs(error) / (abs(y_true) + eps)
```

### 8.3 工程规范依据

当前输出 `max_drift_ratio_raw` 是最大层间位移角，属于结构抗震性能评价中的核心响应量。

ASCE/SEI 7-22 第 12.12.1 节要求设计层间位移不得超过 Table 12.12-1 的允许层间位移限值。公开资料中可见，普通结构在不同风险类别下的允许层间位移常见为约 `0.020h_sx`、`0.015h_sx`、`0.010h_sx` 等水平。

GB 50011-2010/2016《建筑抗震设计规范》也以层间位移角作为抗震变形验算指标。公开资料中，表 5.5.1 给出了多类结构在多遇地震下的弹性层间位移角限值，其中多、高层钢结构常见限值为 `1/250 = 0.004`；表 5.5.5 给出了罕遇地震下弹塑性层间位移角限值，多、高层钢结构常见为 `1/50 = 0.020`。

因此，本项目中的阈值具有如下解释：

| 阈值 | 工程含义 |
|---|---|
| `true < 0.001` | 低响应区间，主要影响 MAPE 和小震/轻微响应预测稳定性 |
| `0.005` | 接近或略高于部分弹性位移角限值量级，可作为中等响应阈值 |
| `0.010` | 明显非低响应区间，可作为尾部风险识别阈值 |
| `0.020` | 与 ASCE 常见允许漂移上限和 GB 罕遇地震弹塑性限值量级接近，是极端尾部重点 |

注意：上述阈值用于模型训练和论文评价时的工程解释，最终论文中应根据具体结构体系、设防水准和采用规范版本重新核定。

## 9. 参考文献与规范

1. Huber, P. J. (1964). Robust Estimation of a Location Parameter. *The Annals of Mathematical Statistics*, 35(1), 73-101. https://doi.org/10.1214/aoms/1177703732

2. PyTorch Documentation. `torch.nn.SmoothL1Loss`. https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.loss.SmoothL1Loss.html

3. Steininger, M., Kobs, K., Davidson, P., Krause, A., & Hotho, A. (2021). Density-based weighting for imbalanced regression. *Machine Learning*, 110, 2187-2211. https://doi.org/10.1007/s10994-021-06023-5

4. Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting*, 22(4), 679-688. https://doi.org/10.1016/j.ijforecast.2006.03.001

5. ASCE/SEI 7-22. Minimum Design Loads and Associated Criteria for Buildings and Other Structures, Section 12.12.1 and Table 12.12-1. ASCE version history page: https://amplify.asce.org/popup-data?apath=%2Fasceworks%2Fstandard%2F9780784415788%2Fpart%2Fprovisions%2Fstandard-chapter%2Fs12%2Fstandard-sec-ver%2Fs12.12.1.ver.atom

6. GB 50011-2010/2016. 《建筑抗震设计规范》Code for seismic design of buildings. 标准信息页: https://www.antpedia.com/standard/6154527-1.html

7. GB 50011-2010 公开 PDF 资料，表 5.5.1 与表 5.5.5 层间位移角限值: https://sutlib2.sut.ac.th/sut_contents/H176050.pdf

