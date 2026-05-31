# 传统 ML 与 MLP 代理模型对比实现方案

生成日期：2026-05-31

## 1. 目的

当前主模型是多模态 `2D-CNN` 代理模型，输入包括地震波小波时频图、结构标量参数、地震动派生特征和阻尼器布置，输出为最大层间位移角：

```text
label = max_drift_ratio_raw
```

为了满足代理模型论文中常见的“不同模型/算法对比”要求，建议新增一组表格代理模型作为基线：

```text
Random Forest
XGBoost
LightGBM
CatBoost
MLP
```

这组模型只使用结构参数、地震动特征和阻尼器布置特征，不使用小波图像。这样可以回答两个论文问题：

1. 仅靠表格特征，传统机器学习能达到什么水平？
2. 当前多模态 2D-CNN 相比表格代理模型是否真的有增益？

## 2. 当前数据与任务定义

使用现有三份数据划分：

```text
数据集/*_train.csv
数据集/*_val.csv
数据集/*_test.csv
```

现有数据规模：

| 划分 | 有效样本数 | 唯一地震波数 |
|---|---:|---:|
| Train | 75009 | 167 |
| Val | 4564 | 26 |
| Test | 7332 | 34 |

目标列：

```text
max_drift_ratio_raw
```

不建议作为输入的列：

| 列 | 原因 |
|---|---|
| image_path | 表格模型不使用图像 |
| txt_path | 原始地震波文件路径，不应直接作为数值特征 |
| sample_id | 标识符，可能造成记忆 |
| split/stage | 数据划分信息，不能进入模型 |
| steel01_yielded/steel02_yielded | 属于响应结果或响应派生量，作为输入会造成信息泄漏 |
| max_drift_ratio_raw | 目标列 |

## 3. 统一特征方案

建议所有表格模型使用同一套特征，保证公平对比。

### 3.1 基础结构参数

```text
num_floors
floor_mass
floor_height
k_base_1_4
Fy_add
```

### 3.2 地震动特征

```text
period_1_sec
wave_pga
wave_rms
wave_mean_abs
wave_cav
wave_arias_proxy
wave_duration_5_95
wave_zero_crossing_rate
wave_dominant_freq
wave_spectral_centroid
wave_predominant_period
wave_intensity_score
```

### 3.3 对数特征

对正值地震动特征增加 `log1p` 版本：

```text
log1p_period_1_sec
log1p_wave_pga
log1p_wave_rms
log1p_wave_mean_abs
log1p_wave_cav
log1p_wave_arias_proxy
log1p_wave_duration_5_95
log1p_wave_zero_crossing_rate
log1p_wave_dominant_freq
log1p_wave_spectral_centroid
log1p_wave_predominant_period
log1p_wave_intensity_score
```

### 3.4 结构-地震动交互特征

```text
wave_to_structure_period_ratio
structure_to_wave_period_ratio
wave_structure_period_log_gap
wave_dominant_freq_x_period
wave_spectral_centroid_x_period
wave_pga_x_period
wave_cav_x_period
wave_arias_proxy_x_period
wave_intensity_score_x_period
```

### 3.5 结构派生特征

```text
inv_k_base_1_4
inv_Fy_add
mass_to_stiffness
height_to_stiffness
period_squared
num_floors_x_period
floor_mass_x_period
floor_height_x_period
flexibility_x_period
strength_inverse_x_period
```

### 3.6 阻尼器布置特征

将 `damper_layout` 解析为 7 个二值特征：

```text
damper_story_1
damper_story_2
damper_story_3
damper_story_4
damper_story_5
damper_story_6
damper_story_7
```

再增加统计特征：

```text
damper_install_count
damper_install_ratio
damper_sparse_ratio
```

### 3.7 尾部风险代理特征

保留当前 2D-CNN 标量分支中已经使用的尾部风险代理特征：

```text
tail_risk_proxy
log1p_tail_risk_proxy
wave_intensity_tail_risk
wave_arias_tail_risk
wave_cav_tail_risk
```

### 3.8 最终特征维度

建议复用当前训练元数据中的 `scalar_feature_names`，共 63 个特征。这样表格模型与当前多模态模型的标量分支完全对齐。

## 4. 统一预处理方案

### 4.1 缺失值处理

树模型：

```text
Random Forest: 使用 SimpleImputer(strategy="median")
XGBoost: 可原生处理缺失，但建议统一 median 填充
LightGBM: 可原生处理缺失，但建议统一 median 填充
CatBoost: 可原生处理缺失，但本项目特征均为数值，建议统一 median 填充
```

MLP：

```text
SimpleImputer(strategy="median")
StandardScaler()
```

### 4.2 特征缩放

树模型通常不需要特征缩放，但为了统一 pipeline，可以只对 MLP 使用标准化。

```text
RF/XGBoost/LightGBM/CatBoost: 不标准化
MLP: 必须标准化
```

### 4.3 目标缩放

建议沿用当前 2D-CNN 的目标缩放：

```text
y_train_scaled = max_drift_ratio_raw * 1000
预测后：
y_pred_raw = y_pred_scaled / 1000
```

这样训练数值范围更稳定，也方便和现有 2D-CNN 指标对比。

可选附加实验：

```text
y_train_log = log1p(max_drift_ratio_raw * 1000)
y_pred_raw = expm1(y_pred_log) / 1000
```

如果直接预测对尾部误差较大，可以对 XGBoost、LightGBM、CatBoost 追加一个 `log1p target` 版本，作为消融对比。

### 4.4 样本权重

当前任务存在高漂移尾部样本稀少的问题。建议设置两套实验：

| 实验 | 样本权重 | 目的 |
|---|---|---|
| Natural | 不加权 | 观察模型对整体分布的自然拟合能力 |
| Tail-weighted | 对 `>=0.005`、`>=0.010`、`>=0.020` 加权 | 提升高漂移尾部预测和危险阈值识别能力 |

建议默认权重：

```text
base_weight = 1.0
if y >= 0.005: weight *= 1.5
if y >= 0.010: weight *= 3.0
if y >= 0.020: weight *= 8.0
```

是否采用加权模型，应由验证集综合指标决定，而不是只看训练集误差。

## 5. 统一评价指标

所有模型统一输出以下指标：

```text
R2
MAE
RMSE
MAPE
Bias
```

并按漂移区间输出分段指标：

```text
true < 0.001
0.001 <= true < 0.005
0.005 <= true < 0.010
true >= 0.010
```

并按工程阈值输出分类指标：

```text
MIDR >= 0.005: Precision / Recall / F1
MIDR >= 0.010: Precision / Recall / F1
MIDR >= 0.015: Precision / Recall / F1
MIDR >= 0.020: Precision / Recall / F1
```

建议模型选择指标：

```text
selection_score =
    1.00 * global_mae
  + 0.20 * global_rmse
  + 0.30 * tail_mae_ge_0p010
  + 0.20 * under_mae_ge_0p010
```

如果论文重点是平均响应预测，可以降低尾部权重。如果论文重点是损伤预警，应提高 tail 和 under-prediction 权重。

## 6. Random Forest 实现方案

### 6.1 定位

Random Forest 是稳定、可解释、难调参程度低的传统基线。它通常不是最终最优模型，但适合作为论文中最基本的集成学习对比项。

### 6.2 推荐实现

使用：

```python
sklearn.ensemble.RandomForestRegressor
```

依赖当前项目已经具备：

```text
scikit-learn
```

### 6.3 推荐参数搜索范围

| 参数 | 搜索范围 |
|---|---|
| n_estimators | 500, 800, 1200 |
| max_depth | None, 12, 18, 24 |
| min_samples_leaf | 1, 2, 4, 8 |
| min_samples_split | 2, 5, 10 |
| max_features | "sqrt", 0.5, 0.7, 1.0 |
| bootstrap | True |
| n_jobs | -1 |

### 6.4 训练策略

Random Forest 没有原生 early stopping。建议：

1. 用 train 训练多组参数。
2. 用 val 计算统一评价指标。
3. 选 validation selection_score 最优的一组。
4. 用 train + val 重新训练最终模型。
5. 在 test 上一次性报告最终结果。

### 6.5 优点与风险

优点：

```text
训练稳定
不需要特征标准化
可以输出 feature_importances_
适合作为论文中最基础基线
```

风险：

```text
外推能力弱
对稀有尾部样本可能不敏感
模型文件较大
R2 可能低于 boosting 模型
```

## 7. XGBoost 实现方案

### 7.1 定位

XGBoost 是结构工程代理模型论文中最常见的强基线之一。很多同类文献中，XGBoost、CatBoost、Extra Trees、Random Forest 通常位于前几名。

### 7.2 推荐实现

使用：

```python
xgboost.XGBRegressor
```

当前项目尚未包含该依赖，需要新增：

```text
xgboost
```

### 7.3 推荐参数搜索范围

| 参数 | 搜索范围 |
|---|---|
| objective | "reg:squarederror", "reg:pseudohubererror" |
| n_estimators | 1000, 2000, 4000 |
| learning_rate | 0.01, 0.03, 0.05, 0.08 |
| max_depth | 3, 4, 5, 6, 8 |
| min_child_weight | 1, 3, 5, 10 |
| subsample | 0.7, 0.85, 1.0 |
| colsample_bytree | 0.7, 0.85, 1.0 |
| reg_lambda | 1, 3, 10 |
| reg_alpha | 0, 0.1, 1 |
| tree_method | "hist" |

如果使用 GPU，可尝试：

```text
device = "cuda"
tree_method = "hist"
```

### 7.4 训练策略

1. 使用 train 训练。
2. 使用 val 做 early stopping。
3. 记录最佳迭代轮数。
4. 用最佳参数在 train + val 上重训，或直接使用 train 训练并以 val early stopping 得到的模型作为最终模型。
5. 在 test 上输出全局指标、分段指标和阈值分类指标。

### 7.5 优点与风险

优点：

```text
通常精度高
对非线性和特征交互拟合强
训练速度快
支持 sample_weight
支持特征重要性分析
```

风险：

```text
调参敏感
尾部加权过强时可能牺牲核心区间 MAE
对未见参数组合的外推能力仍有限
```

## 8. LightGBM 实现方案

### 8.1 定位

LightGBM 是高效梯度提升树模型，适合样本量较大、特征数中等的表格数据。当前数据约 7.5 万训练样本、63 个标量特征，适合 LightGBM。

### 8.2 推荐实现

使用：

```python
lightgbm.LGBMRegressor
```

当前项目尚未包含该依赖，需要新增：

```text
lightgbm
```

### 8.3 推荐参数搜索范围

| 参数 | 搜索范围 |
|---|---|
| objective | "regression", "huber" |
| n_estimators | 1000, 3000, 5000 |
| learning_rate | 0.01, 0.03, 0.05 |
| num_leaves | 31, 63, 127, 255 |
| max_depth | -1, 6, 8, 12 |
| min_child_samples | 20, 50, 100, 200 |
| subsample | 0.7, 0.85, 1.0 |
| colsample_bytree | 0.7, 0.85, 1.0 |
| reg_lambda | 0, 1, 5, 10 |
| reg_alpha | 0, 0.1, 1 |

### 8.4 训练策略

1. 使用 train 训练。
2. 使用 val 做 early stopping。
3. 对比 `regression` 与 `huber` objective。
4. 如果尾部样本低估明显，加入 sample_weight 或单独训练 tail-weighted 版本。
5. 在 test 上报告最终指标。

### 8.5 优点与风险

优点：

```text
训练速度快
适合中大规模表格数据
通常比 Random Forest 精度高
支持 sample_weight
```

风险：

```text
leaf-wise 生长容易过拟合
num_leaves 和 min_child_samples 必须控制
对尾部稀有样本可能需要额外权重
```

## 9. CatBoost 实现方案

### 9.1 定位

CatBoost 在结构工程代理模型论文中经常表现很好，尤其适合非线性表格数据。当前数据特征多为数值特征，但 CatBoost 仍然可以作为强基线。

### 9.2 推荐实现

使用：

```python
catboost.CatBoostRegressor
```

当前项目尚未包含该依赖，需要新增：

```text
catboost
```

### 9.3 推荐参数搜索范围

| 参数 | 搜索范围 |
|---|---|
| loss_function | "RMSE", "MAE", "Huber:delta=1.0" |
| iterations | 2000, 4000, 6000 |
| learning_rate | 0.01, 0.03, 0.05 |
| depth | 4, 6, 8, 10 |
| l2_leaf_reg | 1, 3, 10, 30 |
| subsample | 0.7, 0.85, 1.0 |
| random_strength | 0.5, 1, 2 |
| bootstrap_type | "Bernoulli", "Bayesian" |

如果使用 GPU：

```text
task_type = "GPU"
devices = "0"
```

### 9.4 训练策略

1. 使用 train 训练。
2. 使用 val 做 early stopping。
3. 优先尝试 `RMSE` 与 `Huber`。
4. 若 MAPE 和小漂移区间误差较高，可尝试 `MAE`。
5. 保存最佳模型、特征重要性和测试预测结果。

### 9.5 优点与风险

优点：

```text
表格数据强基线
默认参数通常已经较稳
对非线性特征交互拟合能力强
支持 sample_weight
支持特征重要性
```

风险：

```text
训练时间可能较长
GPU 环境安装有时比 sklearn 复杂
若目标尾部极稀有，仍可能低估高漂移响应
```

## 10. MLP 实现方案

### 10.1 定位

MLP 用来回答一个关键问题：

```text
如果不使用小波图像，只使用和 2D-CNN 标量分支相同的 63 个特征，普通神经网络能达到什么水平？
```

它是当前多模态 2D-CNN 的重要消融基线。

### 10.2 推荐实现

使用 PyTorch 实现，而不是 `sklearn.neural_network.MLPRegressor`。原因是当前项目已经使用 PyTorch，且 PyTorch 更方便实现：

```text
SmoothL1Loss
AdamW
early stopping
sample_weight
tail under-prediction penalty
EMA
```

当前项目已经具备依赖：

```text
torch
scikit-learn
```

### 10.3 推荐网络结构

基础版本：

```text
Input(63)
Linear(63, 256) + LayerNorm + SiLU + Dropout(0.10)
Linear(256, 256) + LayerNorm + SiLU + Dropout(0.15)
Linear(256, 128) + LayerNorm + SiLU + Dropout(0.15)
Linear(128, 64) + LayerNorm + SiLU + Dropout(0.10)
Linear(64, 1)
```

加强版本：

```text
Input(63)
Residual MLP Block x 3
Hidden dim = 256
Activation = SiLU
Norm = LayerNorm
Dropout = 0.10-0.20
Output head = Linear(256, 96) + SiLU + Linear(96, 1)
```

### 10.4 推荐训练参数

| 参数 | 推荐值 |
|---|---:|
| batch_size | 256 或 512 |
| optimizer | AdamW |
| learning_rate | 1e-3, 5e-4, 2e-4 |
| weight_decay | 1e-4 |
| loss | SmoothL1Loss(beta=1.0) |
| epochs | 200 |
| early_stopping_patience | 25 |
| grad_clip_norm | 1.0 |
| target_scale | 1000 |

### 10.5 训练策略

1. 使用 train 训练。
2. 使用 val 做 early stopping。
3. 使用 `StandardScaler` 仅在 train 上 fit，再 transform val/test。
4. 默认使用 `SmoothL1Loss`。
5. 可选加入尾部低估惩罚，但应作为独立实验。
6. 每个配置至少跑 3 个随机种子，报告 mean/std。

### 10.6 优点与风险

优点：

```text
与当前 2D-CNN 的标量分支最接近
可以验证图像输入是否带来额外收益
训练速度快
容易加入和 2D-CNN 相同的 loss 设计
```

风险：

```text
对特征缩放敏感
可能不如 boosting tree 稳定
需要多个随机种子
容易在尾部样本上低估
```

## 11. 推荐实验矩阵

第一阶段先跑自然分布版本：

| 实验编号 | 模型 | 输入 | 样本权重 | 目标变换 |
|---|---|---|---|---|
| B1 | Random Forest | 63 scalar features | No | y * 1000 |
| B2 | XGBoost | 63 scalar features | No | y * 1000 |
| B3 | LightGBM | 63 scalar features | No | y * 1000 |
| B4 | CatBoost | 63 scalar features | No | y * 1000 |
| B5 | MLP | 63 scalar features | No | y * 1000 |

第二阶段只对表现最好的 2-3 个模型跑增强版本：

| 实验编号 | 模型 | 输入 | 样本权重 | 目标变换 |
|---|---|---|---|---|
| T1 | Best Tree Model | 63 scalar features | Tail-weighted | y * 1000 |
| T2 | Best Tree Model | 63 scalar features | No | log1p(y * 1000) |
| T3 | Best Tree Model | 63 scalar features | Tail-weighted | log1p(y * 1000) |
| T4 | MLP | 63 scalar features | Tail-weighted | y * 1000 |

最终论文表格建议包含：

| 模型 | R2 | MAE | RMSE | MAPE | F1@0.005 | F1@0.010 | F1@0.020 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |
| XGBoost | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |
| LightGBM | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |
| CatBoost | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |
| MLP | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 | 待训练 |
| Current 2D-CNN | 0.8806 | 0.000632 | 0.001235 | 18.99% | 0.916 | 0.541 | 0.421 |

## 12. 建议代码组织

建议新增目录：

```text
tabular_baselines/
```

建议新增文件：

```text
tabular_baselines/train_tabular_baselines.py
tabular_baselines/tabular_features.py
tabular_baselines/tabular_metrics.py
tabular_baselines/tabular_models.py
tabular_baselines/启动表格基线训练.bat
```

输出目录：

```text
output/tabular_baselines/
```

每个模型单独输出：

```text
output/tabular_baselines/{model_name}-{timestamp}/metadata.json
output/tabular_baselines/{model_name}-{timestamp}/best_model.pkl 或 best_model.pth
output/tabular_baselines/{model_name}-{timestamp}/val_metrics.json
output/tabular_baselines/{model_name}-{timestamp}/test_metrics.json
output/tabular_baselines/{model_name}-{timestamp}/test_results.csv
output/tabular_baselines/{model_name}-{timestamp}/feature_importance.csv
output/tabular_baselines/{model_name}-{timestamp}/prediction_scatter.png
output/tabular_baselines/{model_name}-{timestamp}/residual_hist.png
```

## 13. 推荐落地顺序

建议按下面顺序实现，先快后慢，先强基线后复杂模型：

| 顺序 | 内容 | 原因 |
|---:|---|---|
| 1 | 统一特征构建与指标函数 | 后续所有模型复用 |
| 2 | Random Forest | 依赖已存在，最快得到基线 |
| 3 | MLP | 依赖已存在，可验证标量神经网络能力 |
| 4 | XGBoost | 强基线，论文常见 |
| 5 | LightGBM | 快速 boosting 对比 |
| 6 | CatBoost | 表格强模型，可能成为最佳传统 ML |
| 7 | Tail-weighted 和 log target 消融 | 专门改善尾部预测 |

## 14. 预期结果与论文解释方式

可能出现三种情况。

第一种情况：传统 ML 明显弱于 2D-CNN。

```text
说明小波时频图提供了地震动形态信息，多模态输入有实质增益。
```

第二种情况：XGBoost/CatBoost 与 2D-CNN 接近。

```text
说明当前 63 个标量特征已经包含大量有效信息，论文需要强调 2D-CNN 在尾部、泛化、或可解释性上的优势。
```

第三种情况：XGBoost/CatBoost 优于 2D-CNN。

```text
这不是坏结果。论文可以转向“代理模型系统比较”，把多模态 2D-CNN 作为深度学习方案之一，同时报告表格 boosting 模型的优势。后续可以考虑 stacking，将 CatBoost/XGBoost 与 2D-CNN 预测结果融合。
```

## 15. 最小可发表对比配置

如果时间有限，建议至少完成：

```text
Random Forest
XGBoost
CatBoost
MLP
Current 2D-CNN
```

论文中最少需要报告：

```text
R2
MAE
RMSE
MAPE
F1@MIDR>=0.010
训练耗时
单样本推理耗时
```

这样基本可以满足“有无算法对比环节”的审稿要求。

