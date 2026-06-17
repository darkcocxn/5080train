# LSTM 与 WaveNet 训练脚本实施方案

生成日期：2026-06-01  
项目目录：`X:\pyproject\Remote-Train`

## 1. 目标

当前主线模型是多模态 `2D-CNN`：用小波时频图 `image_path` 加结构与地震动标量特征预测最大层间位移角：

```text
label = max_drift_ratio_raw
```

新增 `LSTM` 与 `WaveNet/TCN` 的目的不是替代当前最优 `2D-CNN`，而是补足一个“直接使用原始地震波时程”的深度学习对照组：

1. 验证原始加速度序列是否能提供小波图之外的有效信息；
2. 检查序列模型对 `0.005-0.010` 中高漂移区间低估问题是否有改善；
3. 为论文提供“表格模型、2D-CNN、序列模型”的完整算法对比；
4. 保持训练、测试、输出目录与现有版本脚本一致，方便远程训练和复现实验。

## 2. 当前问题定义

沿用现有任务定义：

```text
输入 1：地震波原始时程，由 txt_path 或 h5://...|index 解析
输入 2：结构参数、地震动派生特征、阻尼器布置等标量特征
输出：max_drift_ratio_raw
```

现有训练痛点：

| 问题 | 现象 | 序列模型关注点 |
|---|---|---|
| 高响应样本稀疏 | `>=0.005`、`>=0.010` 样本比例低 | 继续复用 tail-weighted loss 与 weighted sampler |
| 中高漂移低估 | `0.005-0.010` 区间欠预测率高 | 用时程序列捕捉脉冲、持续时间、能量累积 |
| 验证与测试分布不一致 | 部分数据版本 train/test 缺少 `>=0.010` | 自适应 tail 阈值，避免无正样本分类头 |
| 波形来源不统一 | 旧数据为 txt，新数据可为 h5 URI | 统一实现 sequence resolver |

## 3. 建议目录结构

保持当前 `2dcnnv11/` 风格：每个模型一个版本目录，训练脚本、测试脚本、bat 启动脚本和说明文件放在同一目录；结果统一写入 `output/模型版本/`。

```text
X:\pyproject\Remote-Train
├─ lstmv1
│  ├─ lstmv1.py
│  ├─ lstmv1test.py
│  ├─ 启动训练.bat
│  ├─ 启动测试.bat
│  └─ v1实现说明.md
├─ wavenetv1
│  ├─ wavenetv1.py
│  ├─ wavenetv1test.py
│  ├─ 启动训练.bat
│  ├─ 启动测试.bat
│  └─ v1实现说明.md
├─ output
│  ├─ lstmv1
│  │  └─ model-<dataset_hash>-train<timestamp>
│  └─ wavenetv1
│     └─ model-<dataset_hash>-train<timestamp>
├─ 数据集
├─ newdata
├─ rawdata
├─ floors_3_to_7_utils.py
└─ floors_3_to_8_utils.py
```

不建议把两个模型混在同一个脚本中。这样做虽然能少写一些代码，但后续远程并行训练、调参、比较 checkpoint 时会不够清晰。

## 4. 文件命名与输出约定

### 4.1 LSTM

训练：

```powershell
uv run python lstmv1/lstmv1.py
```

测试：

```powershell
uv run python lstmv1/lstmv1test.py
```

输出：

```text
output/lstmv1/model-<dataset_hash>-train<timestamp>/
├─ best_lstm_model.pth
├─ best_lstm_mae_model.pth
├─ best_lstm_focus_model.pth
├─ best_lstm_extreme_under_model.pth
├─ scalar_scaler.pkl
├─ sequence_scaler.pkl
├─ training_metadata.json
├─ training_history.csv
├─ training_curves.png
└─ test_predictions.csv
```

### 4.2 WaveNet

训练：

```powershell
uv run python wavenetv1/wavenetv1.py
```

测试：

```powershell
uv run python wavenetv1/wavenetv1test.py
```

输出：

```text
output/wavenetv1/model-<dataset_hash>-train<timestamp>/
├─ best_wavenet_model.pth
├─ best_wavenet_mae_model.pth
├─ best_wavenet_focus_model.pth
├─ best_wavenet_extreme_under_model.pth
├─ scalar_scaler.pkl
├─ sequence_scaler.pkl
├─ training_metadata.json
├─ training_history.csv
├─ training_curves.png
└─ test_predictions.csv
```

## 5. 数据读取方案

### 5.1 CSV 搜索顺序

训练脚本建议沿用当前路径候选思想，并把 `newdata` 纳入优先搜索：

```text
newdata
CSV-dataset
数据集
```

默认匹配：

```text
opensees_surrogate_dataset_floors_3_to_7_*_train.csv
opensees_surrogate_dataset_floors_3_to_7_*_val.csv
opensees_surrogate_dataset_floors_3_to_7_*_test.csv
```

同时支持环境变量覆盖：

```text
SURMOD_DATASET_DIRS
SURMOD_TRAIN_CSV
SURMOD_VAL_CSV
SURMOD_TEST_CSV
SURMOD_MODEL_ROOT_DIR
```

### 5.2 样本过滤

与现有 `2dcnnv11` 保持一致：

```text
analysis_status == ok
max_drift_ratio_raw 为有限值
txt_path 或 h5 wave reference 可解析
```

不进入模型的列：

| 列 | 原因 |
|---|---|
| `sample_id` | 标识符，可能造成记忆 |
| `split` / `stage` | 数据划分信息 |
| `image_path` | LSTM/WaveNet 不直接使用小波图 |
| `steel01_yielded` / `steel02_yielded` | 响应结果派生量，作为输入会泄漏 |
| `max_drift_ratio_raw` | 标签 |

### 5.3 波形解析

需要新增统一函数：

```python
def load_wave_sequence(reference: str, seq_len: int, target_dt: float) -> np.ndarray:
    ...
```

支持三类来源：

| 来源 | 示例 | 处理方式 |
|---|---|---|
| 本地 txt | `rawdata/Origin-earthquake-file/.../EQ231.txt` | `np.loadtxt` |
| 新数据 txt | `newdata/Newdatabase/波形文件/*.txt` | 第一行若为 `dt`，其余行为加速度 |
| HDF5 URI | `h5://.../rare_waves_scaled.h5|168` | 用 `h5py` 读取指定 index |

当前 `pyproject.toml` 尚未包含 `h5py`。如果训练 `newdata` 中的 HDF5 波形，应新增依赖：

```toml
"h5py>=3.11.0"
```

如果只使用 `数据集` 中的旧 txt 路径，第一阶段可以不加 `h5py`。

### 5.4 序列长度与重采样

建议第一版使用固定长度序列，避免 batch 内动态 padding 影响吞吐：

| 参数 | LSTM v1 | WaveNet v1 |
|---|---:|---:|
| `TARGET_DT` | 0.01 | 0.01 |
| `SEQ_LEN` | 2048 或 4096 | 4096 |
| 短序列 | 右侧补 0 | 右侧补 0 |
| 长序列 | 线性重采样后截断或均匀采样 | 线性重采样后截断或均匀采样 |
| 归一化 | train 集全局 robust scaler | train 集全局 robust scaler |

不要默认按单条地震波做 z-score，因为这会抹掉 PGA、RMS、CAV 等强度信息。建议使用训练集全局统计量：

```text
sequence_scaled = clip((sequence - train_mean) / train_std, -8, 8)
```

地震动强度差异继续由原始序列幅值和标量特征共同表达。

## 6. 标量特征方案

复用当前 `2dcnnv11` 的 63 维标量构造逻辑：

```text
基础结构参数
地震动特征
log1p 地震动特征
结构-地震动交互特征
结构派生特征
damper_layout 展开特征
tail risk proxy 特征
```

实现上建议直接从 `2dcnnv11.py` 迁移以下函数，并保持函数名一致：

```text
parse_damper_layout_flags
get_available_wave_feature_cols
add_wave_derived_features
add_structure_derived_features
add_tail_risk_features
build_scalar_feature_frame
```

训练阶段：

```text
scalar_scaler.fit(train_scalar_features)
scalar_scaler.transform(train/val/test)
```

测试阶段从 `training_metadata.json` 读取：

```text
scalar_feature_names
num_scalar_features
label_scale
sequence_config
```

## 7. Dataset 设计

统一 Dataset 返回：

```python
{
    "sequence": FloatTensor[C, T],
    "scalar": FloatTensor[num_scalar_features],
    "label": FloatTensor[1],
    "label_raw": float,
    "sample_id": str,
}
```

建议序列通道：

| 通道 | 含义 | 是否首版启用 |
|---|---|---|
| `acc_scaled` | 全局归一化加速度 | 是 |
| `abs_acc_scaled` | 绝对加速度 | 是 |
| `energy_cumsum` | 归一化累积能量 | 可选 |

首版可以先用 2 个通道：

```text
[acc_scaled, abs_acc_scaled]
```

如果训练稳定，再加 `energy_cumsum` 做消融。

## 8. LSTM 模型方案

### 8.1 网络结构

建议 `lstmv1.py` 实现：

```text
SequenceConvStem
  Conv1d(C -> 32, kernel=7, stride=2)
  GroupNorm + SiLU
  Conv1d(32 -> 64, kernel=5, stride=2)
  GroupNorm + SiLU

BiLSTMEncoder
  input_dim = 64
  hidden_dim = 128
  num_layers = 2
  bidirectional = True
  dropout = 0.20

AttentionPooling
  Linear(256 -> 128)
  Tanh
  Linear(128 -> 1)
  softmax over time

ScalarFeatureEncoder
  复用 v11 residual scalar encoder

FusionHead
  concat 或 gated_bilinear fusion
  regression head
  tail classification aux head
  tail correction head
```

### 8.2 推荐超参数

| 参数 | 建议值 |
|---|---:|
| `BATCH_SIZE` | 64 |
| `NUM_EPOCHS` | 120 |
| `LEARNING_RATE` | `1.0e-4` |
| `WEIGHT_DECAY` | `1.0e-4` |
| `GRAD_CLIP_NORM` | 0.8 |
| `USE_AMP` | True |
| `USE_EMA` | True |
| `EARLY_STOPPING_PATIENCE` | 20 |

LSTM 对长序列显存和速度比较敏感，首版不建议直接吃 6000 或更长原序列。先通过 `SequenceConvStem` 做 4 倍下采样，再进入 LSTM。

## 9. WaveNet 模型方案

### 9.1 网络结构

建议 `wavenetv1.py` 实现非因果 WaveNet/TCN 回归模型。当前任务是整条地震波预测最大响应，不是在线预测，因此不需要严格 causal，只需要膨胀卷积覆盖足够长的时程范围。

```text
WaveInputProjection
  Conv1d(C -> residual_channels, kernel=1)

DilatedResidualBlocks x N
  tanh/filter conv
  sigmoid/gate conv
  gated activation
  residual 1x1 conv
  skip 1x1 conv

GlobalPooling
  mean pooling
  max pooling
  attention pooling

ScalarFeatureEncoder
  复用 v11 residual scalar encoder

FusionHead
  gated_bilinear fusion
  regression head
  tail classification aux head
  tail correction head
```

### 9.2 膨胀卷积配置

首版建议：

```text
residual_channels = 64
skip_channels = 128
kernel_size = 3
dilation_cycles = 3
dilations_per_cycle = [1, 2, 4, 8, 16, 32, 64, 128, 256]
dropout = 0.10
```

单个 cycle 的理论感受野：

```text
1 + (kernel_size - 1) * sum(dilations)
= 1 + 2 * 511
= 1023 steps
```

3 个 cycle 叠加后可覆盖约 3000 步以上的局部-全局模式。若 `SEQ_LEN=4096`，基本覆盖多数有效脉冲、持续时间和能量累积过程。

### 9.3 推荐超参数

| 参数 | 建议值 |
|---|---:|
| `BATCH_SIZE` | 96 |
| `NUM_EPOCHS` | 120 |
| `LEARNING_RATE` | `2.0e-4` |
| `WEIGHT_DECAY` | `1.0e-4` |
| `GRAD_CLIP_NORM` | 1.0 |
| `USE_AMP` | True |
| `USE_EMA` | True |
| `EARLY_STOPPING_PATIENCE` | 20 |

WaveNet 通常比 LSTM 更适合并行训练，也更容易处理长序列，建议作为序列模型主推版本。

## 10. Loss 与采样策略

首版直接复用 `2dcnnv11` 的 tail-aware 训练策略：

```text
SmoothL1Loss
density weighted loss
tail underprediction loss
relative underprediction loss
tail pinball loss
tail classification auxiliary loss
WeightedRandomSampler
adaptive tail thresholds
EMA
warmup + ReduceLROnPlateau
```

自适应 tail 阈值继续使用：

```text
工程阈值: [0.005, 0.010, 0.020]
fallback 阈值: [0.003, 0.005, 0.007]
```

当训练集中 `>=0.010` 样本数不足时，自动使用 fallback 阈值，避免分类头没有正样本。

## 11. 验证与 checkpoint 选择

与现有脚本保持一致，至少保存四类 checkpoint：

| 文件 | 选择标准 |
|---|---|
| `best_<model>_model.pth` | `val_mae + focus_score + mid_tail_mae` 综合分最低 |
| `best_<model>_mae_model.pth` | 全局 val MAE 最低 |
| `best_<model>_focus_model.pth` | tail focus score 最低 |
| `best_<model>_extreme_under_model.pth` | extreme tail 欠预测最低 |

测试脚本默认加载 `best_<model>_model.pth`，同时支持环境变量切换：

```text
SURMOD_LSTM_WEIGHTS_NAME
SURMOD_WAVENET_WEIGHTS_NAME
```

## 12. 测试脚本指标

`lstmv1test.py` 与 `wavenetv1test.py` 应输出和 `2dcnnv11test.py` 同级别指标：

```text
Samples
R2
MAE
RMSE
MAPE
Bias
多 seed 80% 重采样 mean/std
```

并增加论文需要的分组结果：

```text
drift bin MAE/Bias/欠预测率
>=0.003 / >=0.005 / >=0.007 / >=0.010 F1-like 指标
steel02_yielded=True/False 分组
num_floors 分组
worst wave top-N
```

预测明细保存：

```text
test_predictions.csv
```

建议列：

```text
sample_id
true
pred
abs_error
relative_error_pct
drift_bin
num_floors
wave_cluster
txt_path
```

## 13. 运行脚本内容

`lstmv1/启动训练.bat`：

```bat
@echo off
cd /d "%~dp0.."
uv run python lstmv1/lstmv1.py
pause
```

`lstmv1/启动测试.bat`：

```bat
@echo off
cd /d "%~dp0.."
uv run python lstmv1/lstmv1test.py
pause
```

`wavenetv1/启动训练.bat`：

```bat
@echo off
cd /d "%~dp0.."
uv run python wavenetv1/wavenetv1.py
pause
```

`wavenetv1/启动测试.bat`：

```bat
@echo off
cd /d "%~dp0.."
uv run python wavenetv1/wavenetv1test.py
pause
```

## 14. 实施顺序

| 顺序 | 内容 | 产物 |
|---:|---|---|
| 1 | 新增波形解析器，支持 txt 与可选 h5 | `load_wave_sequence` |
| 2 | 从 `2dcnnv11` 迁移标量特征、tail loss、metrics | LSTM/WaveNet 共用逻辑 |
| 3 | 实现 `lstmv1/lstmv1.py` 与 smoke test | LSTM 可训练 |
| 4 | 实现 `lstmv1/lstmv1test.py` | LSTM 可评估 |
| 5 | 实现 `wavenetv1/wavenetv1.py` 与 smoke test | WaveNet 可训练 |
| 6 | 实现 `wavenetv1/wavenetv1test.py` | WaveNet 可评估 |
| 7 | 分别跑短训练 `DATA_USE_RATIO=0.05` | 验证流程无误 |
| 8 | 全量训练 LSTM 与 WaveNet | 生成正式结果 |
| 9 | 与 `2dcnnv2`、`2dcnnv11`、传统 ML/MLP 对比 | 论文表格 |

## 15. 最小 smoke test

每个训练脚本完成后先跑：

```powershell
python -m py_compile lstmv1/lstmv1.py lstmv1/lstmv1test.py
python -m py_compile wavenetv1/wavenetv1.py wavenetv1/wavenetv1test.py
```

再跑小样本前向：

```powershell
$env:SURMOD_DATA_USE_RATIO="0.01"
uv run python lstmv1/lstmv1.py
uv run python wavenetv1/wavenetv1.py
```

需要确认：

```text
sequence batch shape 正确
scalar batch shape 正确
model output shape = (batch, 1)
tail logits shape = (batch, 3)
training_metadata.json 正常写入
测试脚本可自动找到对应 test.csv
```

## 16. 推荐实验矩阵

第一阶段只做自然分布版本：

| 实验编号 | 模型 | 输入 | 序列长度 | tail loss | 目标 |
|---|---|---|---:|---|---|
| S1 | LSTM v1 | wave sequence + scalar | 2048 | v11 策略 | 验证序列模型可行性 |
| S2 | WaveNet v1 | wave sequence + scalar | 4096 | v11 策略 | 主序列模型候选 |
| S3 | WaveNet v1 | wave sequence only | 4096 | v11 策略 | 验证标量特征贡献 |
| S4 | WaveNet v1 | scalar only ablation | - | v11 策略 | 与 MLP/表格模型对照 |

第二阶段再做增强版本：

| 实验编号 | 模型 | 改动 | 目的 |
|---|---|---|---|
| T1 | LSTM v1 | `SEQ_LEN=4096` | 检查长序列收益 |
| T2 | WaveNet v1 | 加 `energy_cumsum` 通道 | 强化能量累积表达 |
| T3 | WaveNet v1 | 更大 residual channels=96 | 检查容量瓶颈 |
| T4 | WaveNet v1 | 关闭 tail correction | 验证 tail head 是否真实贡献 |

## 17. 预期结果判断

序列模型是否值得保留，不只看全局 R2：

| 指标 | 判定方式 |
|---|---|
| 全局 R2/MAE/RMSE | 至少接近当前 2D-CNN 主结果 |
| `0.005-0.010` MAE | 若低于 2D-CNN，说明时程输入对中高响应有帮助 |
| `0.005-0.010` 欠预测率 | 重点观察是否从约 80% 降低 |
| `>=0.005` F1-like 指标 | 判断高响应识别能力 |
| 推理耗时 | LSTM 可能较慢，WaveNet 应更有优势 |

可能出现三种结论：

| 情况 | 论文解释 |
|---|---|
| WaveNet 优于 2D-CNN | 原始时程的长程时序特征比小波图更适合当前响应预测 |
| WaveNet 接近 2D-CNN | 两类地震动表示互补，可考虑 ensemble |
| LSTM/WaveNet 弱于 2D-CNN | 小波时频图更适合当前数据规模和尾部稀疏条件 |

## 18. 风险与注意事项

1. `h5://` 波形读取需要明确 HDF5 数据集键名，若文件结构未知，需先写探测函数列出 keys。
2. 长序列 LSTM 训练可能明显慢于 2D-CNN，首版必须有卷积下采样。
3. 若对每条波做 z-score，模型会失去强度幅值信息，不建议作为默认方案。
4. 如果当前训练集没有 `>=0.010`，不要强行用 `0.010/0.020` 做主 tail 分类阈值。
5. 序列模型可能比 2D-CNN 更容易过拟合 wave identity，因此 train/val/test 必须继续按 wave split 检查。
6. 测试脚本必须读取训练 metadata，不能重新推断特征列或 scaler，否则结果不可复现。

## 19. 最终建议

优先实现顺序：

```text
WaveNet v1 > LSTM v1
```

原因是 WaveNet/TCN 对长序列并行友好，训练速度和稳定性通常优于 LSTM；LSTM 仍值得保留为经典序列模型对照。若时间有限，先完成 `wavenetv1`，再补 `lstmv1`。

建议论文最终对比表至少包含：

```text
Random Forest
XGBoost 或 CatBoost
MLP
2D-CNN v2/v11
LSTM v1
WaveNet v1
```

报告指标：

```text
R2
MAE
RMSE
MAPE
Bias
F1@0.005
F1@0.010
0.005-0.010 欠预测率
训练耗时
单样本推理耗时
```
