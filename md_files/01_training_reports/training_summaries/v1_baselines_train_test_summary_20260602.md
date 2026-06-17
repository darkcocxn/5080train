# v1基线算法训练、测试与结果总结

生成时间：2026-06-02

## 1. 本次执行范围

本轮实际启动并汇总了 7 个 `v1` 基线算法：

- `catboostv1`
- `lightgbmv1`
- `randomforestv1`
- `xgboostv1`
- `mlpv1`
- `lstmv1`
- `wavenetv1`

说明：

- 本报告不包含 `2dcnnv1/2/3/...` 等 2D-CNN 变体的重新训练结果。
- 本轮数据集统一使用：
  - `train`: `newdata/opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258_train.csv`
  - `val`: `newdata/opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258_val.csv`
  - `test`: `newdata/opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258_test.csv`
- 统一预测目标：`max_drift_ratio_raw`
- 统一标签缩放：训练内部按 `y * 1000` 处理，落盘结果已还原为原始漂移比尺度。

数据规模：

- 训练集：`67260`
- 验证集：`3420`
- 测试集：`3116`

## 2. 总体结果排序

按测试集 `R2` 从高到低排序如下：

| 排名 | 算法 | 测试R2 | 测试MAE | 测试RMSE | 测试MAPE |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `wavenetv1` | 0.8664 | 0.000355626 | 0.000496153 | 23.67% |
| 2 | `lstmv1` | 0.8586 | 0.000374142 | 0.000510350 | 23.64% |
| 3 | `lightgbmv1` | 0.7794 | 0.000421465 | 0.000637425 | 24.88% |
| 4 | `catboostv1` | 0.7754 | 0.000431681 | 0.000643270 | 25.56% |
| 5 | `xgboostv1` | 0.7513 | 0.000459898 | 0.000676930 | 27.83% |
| 6 | `randomforestv1` | 0.6850 | 0.000557378 | 0.000761807 | 42.11% |
| 7 | `mlpv1` | 0.5439 | 0.000673718 | 0.000916616 | 38.40% |

结论：

- 当前最强模型是 `wavenetv1`，测试集指标略高于 `lstmv1`。
- 两个序列模型明显优于 4 个树模型和 `mlpv1`。
- 表格树模型中，`lightgbmv1` 与 `catboostv1` 最稳，且二者最接近。
- `mlpv1` 在当前特征表达下明显落后于树模型和序列模型。

## 3. 统一输入输出说明

### 3.1 表格类模型：`catboostv1` / `lightgbmv1` / `randomforestv1` / `xgboostv1` / `mlpv1`

输入内容：

- 63 个标量/工程特征：
  - 结构参数
  - 地震动统计特征
  - 阻尼器布置展开特征
  - 若干 log、比值、交叉项、tail risk proxy 特征
- 256 个波形下采样特征：
  - 来源通道：`acc_scaled`
  - 下采样长度配置：`seq_len=4096`
- 总输入维数：`319`

输出内容：

- 验证集预测：`*_val-*.csv`
- 测试集预测：`*_test-*.csv`
- 指标文件：`*_metrics.json`
- 重采样统计：`*_seed_metrics.csv`
- 尾部统计：`*_tail_metrics.csv`
- 漂移分箱统计：`*_drift_bins.csv`
- 结果图：`*.svg`
- 训练元数据：`training_metadata.json`

额外输出：

- 树模型：`best_*.pkl`、`feature_importance.csv`
- MLP：`best_mlp_model.pth`、`mlp_model.pth`、`feature_preprocessor.pkl`、`training_history.csv`、`training_curves.svg`

### 3.2 序列类模型：`lstmv1` / `wavenetv1`

输入内容：

- 原始地震动序列，来自 `txt_path`
- 辅助标量特征：63 维
- 通道：`acc_scaled`、`abs_acc_scaled`

差异：

- `lstmv1`：`seq_len=2048`
- `wavenetv1`：`seq_len=4096`

输出内容：

- 最优权重：`best_*.pth`
- 辅助最优权重：`best_*_mae_model.pth`、`best_*_focus_model.pth`、`best_*_extreme_under_model.pth`
- 标量与序列 scaler：`scalar_scaler.pkl`、`sequence_scaler.pkl`
- 测试集预测：`*_test-*.csv`
- 重采样统计：`*_seed_metrics.csv`
- 尾部统计：`*_tail_metrics.csv`
- 漂移分箱统计：`*_drift_bins.csv`
- 可视化图：`*.svg`
- 训练元数据：`training_metadata.json`

额外输出：

- `lstmv1`：`training_history.csv`、`training_history.json`、`training_curves.png`、`ema_lstm_model.pth`、`lstm_model.pth`
- `wavenetv1`：本轮因训练超时中断，保留了当前最优权重与测试产物，但未完整生成最终训练历史文件

## 4. 各算法结果明细

### 4.1 `catboostv1`

- 运行目录：`output/catboostv1/model-20260531-181258-99dfe0c9-train20260602-131522`
- 输入：
  - 63 标量特征 + 256 波形下采样特征
  - 总维数 319
- 主要模型输出：
  - `best_catboost_model.pkl`
  - `feature_importance.csv`
  - `catboostv1_val-20260531-181258-99dfe0c9_metrics.json`
  - `catboostv1_test-20260531-181258-99dfe0c9_metrics.json`
- 验证集：
  - `R2=0.8030`
  - `MAE=0.000636612`
  - `RMSE=0.001193881`
  - `MAPE=24.28%`
- 测试集：
  - `R2=0.7754`
  - `MAE=0.000431681`
  - `RMSE=0.000643270`
  - `MAPE=25.56%`

### 4.2 `lightgbmv1`

- 运行目录：`output/lightgbmv1/model-20260531-181258-99dfe0c9-train20260602-131602`
- 输入：
  - 63 标量特征 + 256 波形下采样特征
  - 总维数 319
- 主要模型输出：
  - `best_lightgbm_model.pkl`
  - `feature_importance.csv`
  - `lightgbmv1_val-20260531-181258-99dfe0c9_metrics.json`
  - `lightgbmv1_test-20260531-181258-99dfe0c9_metrics.json`
- 验证集：
  - `R2=0.7941`
  - `MAE=0.000656871`
  - `RMSE=0.001220710`
  - `MAPE=25.75%`
- 测试集：
  - `R2=0.7794`
  - `MAE=0.000421465`
  - `RMSE=0.000637425`
  - `MAPE=24.88%`

### 4.3 `randomforestv1`

- 运行目录：`output/randomforestv1/model-20260531-181258-99dfe0c9-train20260602-131635`
- 输入：
  - 63 标量特征 + 256 波形下采样特征
  - 总维数 319
- 主要模型输出：
  - `best_randomforest_model.pkl`
  - `feature_importance.csv`
  - `randomforestv1_val-20260531-181258-99dfe0c9_metrics.json`
  - `randomforestv1_test-20260531-181258-99dfe0c9_metrics.json`
- 验证集：
  - `R2=0.6745`
  - `MAE=0.000871128`
  - `RMSE=0.001534871`
  - `MAPE=43.50%`
- 测试集：
  - `R2=0.6850`
  - `MAE=0.000557378`
  - `RMSE=0.000761807`
  - `MAPE=42.11%`

### 4.4 `xgboostv1`

- 运行目录：`output/xgboostv1/model-20260531-181258-99dfe0c9-train20260602-131658`
- 输入：
  - 63 标量特征 + 256 波形下采样特征
  - 总维数 319
- 主要模型输出：
  - `best_xgboost_model.pkl`
  - `feature_importance.csv`
  - `xgboostv1_val-20260531-181258-99dfe0c9_metrics.json`
  - `xgboostv1_test-20260531-181258-99dfe0c9_metrics.json`
- 验证集：
  - `R2=0.8071`
  - `MAE=0.000646181`
  - `RMSE=0.001181505`
  - `MAPE=26.90%`
- 测试集：
  - `R2=0.7513`
  - `MAE=0.000459898`
  - `RMSE=0.000676930`
  - `MAPE=27.83%`

### 4.5 `mlpv1`

- 运行目录：`output/mlpv1/model-20260531-181258-99dfe0c9-train20260602-131723`
- 输入：
  - 63 标量特征 + 256 波形下采样特征
  - 总维数 319
- 主要模型输出：
  - `best_mlp_model.pth`
  - `mlp_model.pth`
  - `feature_preprocessor.pkl`
  - `training_history.csv`
  - `training_curves.svg`
  - `mlpv1_val-20260531-181258-99dfe0c9_metrics.json`
  - `mlpv1_test-20260531-181258-99dfe0c9_metrics.json`
- 最优验证轮次：`epoch 22`
- 验证集：
  - `R2=0.7852`
  - `MAE=0.000730444`
  - `RMSE=0.001246681`
  - `MAPE=29.82%`
- 测试集：
  - `R2=0.5439`
  - `MAE=0.000673718`
  - `RMSE=0.000916616`
  - `MAPE=38.40%`

### 4.6 `lstmv1`

- 运行目录：`output/lstmv1/model-20260531-181258-99dfe0c9-train20260602-131843`
- 输入：
  - 原始序列 `txt_path`
  - `seq_len=2048`
  - 通道：`acc_scaled`、`abs_acc_scaled`
  - 63 维标量特征
- 主要模型输出：
  - `best_lstm_model.pth`
  - `best_lstm_mae_model.pth`
  - `best_lstm_focus_model.pth`
  - `best_lstm_extreme_under_model.pth`
  - `ema_lstm_model.pth`
  - `lstm_model.pth`
  - `scalar_scaler.pkl`
  - `sequence_scaler.pkl`
  - `training_history.csv`
  - `training_history.json`
  - `training_curves.png`
- 训练状态：
  - 训练完成并早停
  - 最优轮次：`epoch 17`
  - 总训练时间：约 `20.7 min`
- 最优验证结果：
  - `R2=0.8427`
  - `MAE=0.000558192`
  - `RMSE=0.001066808`
  - `MAPE=22.88%`
- 测试集：
  - `R2=0.8586`
  - `MAE=0.000374142`
  - `RMSE=0.000510350`
  - `MAPE=23.64%`

### 4.7 `wavenetv1`

- 运行目录：`output/wavenetv1/model-20260531-181258-99dfe0c9-train20260602-134015`
- 训练日志：`output/wavenetv1/wavenetv1_train_20260602-134013.log`
- 输入：
  - 原始序列 `txt_path`
  - `seq_len=4096`
  - 通道：`acc_scaled`、`abs_acc_scaled`
  - 63 维标量特征
- 主要模型输出：
  - `best_wavenet_model.pth`
  - `best_wavenet_mae_model.pth`
  - `best_wavenet_focus_model.pth`
  - `best_wavenet_extreme_under_model.pth`
  - `scalar_scaler.pkl`
  - `sequence_scaler.pkl`
  - `wavenetv1_test-20260531-181258-99dfe0c9.csv`
  - `wavenetv1_test-20260531-181258-99dfe0c9_seed_metrics.csv`
  - `wavenetv1_test-20260531-181258-99dfe0c9_tail_metrics.csv`
  - `wavenetv1_test-20260531-181258-99dfe0c9_drift_bins.csv`
  - `wavenetv1_test-20260531-181258-99dfe0c9.svg`
- 训练状态：
  - 已启动并持续训练到 `epoch 31` 途中
  - 因本轮终端执行时间上限，中断在默认完整训练结束之前
  - 当前最优权重已保存，可正常测试
- 当前最佳验证结果：
  - 最优轮次：`epoch 18`
  - `R2=0.8413`
  - `MAE=0.000559`
  - `RMSE=0.001072`
  - `MAPE=23.53%`
- 基于当前最佳权重的测试集结果：
  - `R2=0.8664`
  - `MAE=0.000355626`
  - `RMSE=0.000496153`
  - `MAPE=23.67%`

## 5. 本轮执行中的修复与异常说明

### 5.1 `catboostv1` 输出路径过长问题

问题：

- Windows 下验证/测试产物文件名包含完整数据集名，导致路径超长，触发 `FileNotFoundError`。

处理：

- 将表格模型与序列模型测试结果文件名改为紧凑格式：`模型名 + 时间戳 + hash`
- 同时让 `save_json()` 在写入前自动创建父目录

影响：

- `catboostv1`、`lightgbmv1`、`randomforestv1`、`xgboostv1`、`mlpv1`，以及序列模型测试输出都同步受益

### 5.2 `lstmv1` / `wavenetv1` 采样器只读数组问题

问题：

- `sequence_model_common.py` 中构造 `WeightedRandomSampler` 权重时，数组为只读，原地乘法报错：
  - `ValueError: output array is read-only`

处理：

- 将采样权重显式转为 `copy=True` 的可写 `numpy` 数组

影响：

- `lstmv1` 与 `wavenetv1` 训练恢复正常

## 6. 综合建议

如果目标是当前这批 `v1` 基线中优先选最强模型：

- 第一优先：`wavenetv1`
- 第二优先：`lstmv1`
- 第三优先：`lightgbmv1`

如果目标是论文中的传统 ML 强基线对比：

- 优先保留：`lightgbmv1`、`catboostv1`、`xgboostv1`
- `randomforestv1` 可作为较弱 bagging 基线保留
- `mlpv1` 目前更像“神经网络表格基线下界”

如果下一步希望继续完善：

- 继续让 `wavenetv1` 自然早停完成一轮完整训练，再和当前测试结果复核一次
- 对 `lightgbmv1` / `catboostv1` / `xgboostv1` 增加 `log1p target` 与 `sample_weight` 消融
- 将 `wavenetv1` 与 `lstmv1` 的 tail metrics 单独抽出来做论文级对比表
