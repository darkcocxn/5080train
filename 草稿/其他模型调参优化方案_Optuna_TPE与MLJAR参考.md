# 其他对比模型调参优化方案

## 1. 方案目的

本文主模型多模态 2D-CNN 已经过多轮结构、输入表达和训练策略迭代，而 LSTM、WaveNet、MLP、Random Forest、XGBoost、LightGBM 和 CatBoost 等对比模型目前主要为初版实现。若直接将当前结果表述为“主模型优于全部模型”，容易引发比较公平性疑问。因此，建议在现有实验基础上补充一轮对比模型调参优化实验，并在论文中明确区分“初始基线结果”和“有限调参后的对比结果”。

该方案参考两类文献写法：第一类是采用 Optuna/TPE、贝叶斯优化、网格搜索或交叉验证对候选模型进行系统调参后再比较模型性能；第二类是采用 AutoML 框架对表格模型进行自动化模型选择、超参数搜索、特征处理和集成验证。本文建议以 Optuna/TPE 作为主要调参框架，以 MLJAR AutoML 作为表格模型的补充复核工具。

## 2. 调参原则

### 2.1 锁定测试集，避免测试集泄漏

所有模型调参均只允许使用训练集和验证集。测试集仅在最终模型确定后使用一次，用于报告泛化性能。该原则与地震工程机器学习研究中常见做法一致，即先在训练集或交叉验证中确定超参数，再在独立测试集上评价模型性能。

### 2.2 保持数据划分和输入口径一致

为保证对比口径一致，所有对比模型应继续使用当前论文中的训练集、验证集和测试集划分。若采用交叉验证，交叉验证应仅在训练集或训练集加验证集内部进行，并尽量以地震波编号作为分组依据，避免同一地震动相关样本同时进入训练折和验证折。

### 2.3 统一优化目标

建议以验证集 MAE 作为主要优化目标，因为本文预测目标为最大层间位移角，MAE 具有明确的平均绝对误差含义，且不易被少数极端样本完全主导。RMSE、$R^2$、Bias、分漂移区间误差和高漂移阈值识别指标用于最终评价，但不作为所有模型的唯一调参目标。

主优化目标可写为：

$$J_{\mathrm{overall}}=\mathrm{MAE}_{\mathrm{val}}$$

若需要补充尾部风险敏感性实验，可额外设置尾部惩罚目标，但该结果应作为补充实验，不应与整体精度主结果混用：

$$J_{\mathrm{tail}}=\mathrm{MAE}_{\mathrm{val}}+\lambda\max(0,-\mathrm{Bias}_{0.005\le y<0.010})$$

其中，$\lambda$ 为低估惩罚权重，建议仅在 Tail-aware 或风险识别实验中使用。

### 2.4 调参预算应公开报告

论文中应明确说明每类模型的调参方法、搜索空间、试验次数、随机种子、早停规则和最终模型选择标准。若出于计算成本原因，LSTM 和 WaveNet 的搜索次数少于树模型，应将其表述为“有限预算调参”，避免暗示所有模型均已达到全局最优。

## 3. 推荐实验流程

### 3.1 第一阶段：保留初版结果作为 Default baseline

保留当前已有的 LSTM、WaveNet、MLP、Random Forest、XGBoost、LightGBM 和 CatBoost 结果，作为初始基线。该结果用于说明在当前默认或人工设定参数下，不同输入表达与模型类别的初步差异。

建议表述：

> 首先训练各对比模型的初始版本，作为默认基线结果。该阶段主要用于检验各类输入表达和模型结构在统一数据划分下的基础预测能力。

### 3.2 第二阶段：Optuna/TPE 有限预算调参

对所有非主模型进行 Optuna/TPE 调参。Optuna 是一种自动超参数优化框架，官方推荐在科学论文中引用 Akiba 等提出的 Optuna 框架；其 TPESampler 使用 Tree-structured Parzen Estimator，通过对较优参数区域和其他参数区域分别建立概率模型，并选择使 $l(x)/g(x)$ 较大的参数候选点进行搜索。

建议采用如下预算：

| 模型 | 调参方法 | 推荐 trials | 最小 trials | 验证方式 | 说明 |
|---|---:|---:|---:|---|---|
| Random Forest | Optuna/TPE | 60 | 30 | GroupKFold 或固定验证集 | 训练较快，可适度扩大搜索 |
| XGBoost | Optuna/TPE | 80 | 40 | GroupKFold 或固定验证集 | 强表格基线，需重点调参 |
| LightGBM | Optuna/TPE | 80 | 40 | GroupKFold 或固定验证集 | 强表格基线，训练效率高 |
| CatBoost | Optuna/TPE | 80 | 40 | GroupKFold 或固定验证集 | 强表格基线，适合稳健比较 |
| MLP | Optuna/TPE + pruning | 40 | 20 | 固定验证集 | 使用早停和中间结果剪枝 |
| LSTM | Optuna/TPE + pruning | 25 | 12 | 固定验证集 | 计算成本高，有限预算 |
| WaveNet | Optuna/TPE + pruning | 25 | 12 | 固定验证集 | 计算成本高，有限预算 |

若计算资源有限，可以采用“先低保真、后高保真”的两阶段策略：先使用较少 epoch 或较低数据比例筛选前 5 组参数，再用完整训练轮次重新训练这些候选参数，并以验证集 MAE 选择最终模型。

### 3.3 第三阶段：MLJAR AutoML 复核表格模型

MLJAR AutoML 适合用于表格输入模型的补充复核。其文档说明，MLJAR 可训练表格机器学习 pipeline，并自动生成模型报告；其 AutoML 流程包括 default_algorithms、not_so_random、golden_features、features_selection、hill_climbing、ensemble 和 stack 等步骤。default_algorithms 会训练各算法默认参数模型；not_so_random 会在定义的超参数集合上进行随机搜索；hill_climbing 会对表现较好的模型进一步微调；ensemble 和 stack 可基于先前模型进行集成。

建议将 MLJAR AutoML 定位为“补充验证”，而不是替代 Optuna/TPE 的主调参结果。原因是本文数据划分按地震波编号控制泛化，若 MLJAR 的默认验证策略不能严格按地震波分组，则其结果可能与论文主实验划分不完全一致。因此，MLJAR 结果可用于确认表格模型是否存在明显未调参低估，但最终论文主表仍应优先报告按本文数据划分获得的 Optuna/TPE 结果。

MLJAR 建议配置：

| 项目 | 建议设置 |
|---|---|
| 任务类型 | regression |
| 输入特征 | 标量特征 + 256 点下采样波形特征 |
| 候选算法 | Random Forest、Xgboost、LightGBM、CatBoost、Neural Network |
| 模式 | Compete 或 Optuna |
| 评价指标 | MAE 或 RMSE |
| 时间预算 | 2-6 小时，视机器资源调整 |
| 集成 | 主比较可关闭 ensemble；补充分析可保留 ensemble |
| 输出 | leaderboard、best model、AutoML report、模型参数表 |

## 4. 各模型建议搜索空间

### 4.1 Random Forest

| 参数 | 当前初版值 | 建议搜索范围 |
|---|---:|---|
| n_estimators | 500 | 300-1500 |
| max_depth | None | None, 8-40 |
| min_samples_leaf | 2 | 1-20 |
| min_samples_split | 2 | 2-40 |
| max_features | sqrt | sqrt, log2, 0.3-1.0 |
| bootstrap | True | True, False |

Random Forest 的调参重点是树数量、树深度、叶节点最小样本数和特征子采样比例。若模型在中高漂移区间出现明显均值回归式低估，可尝试降低 min_samples_leaf 或增大 max_depth，但需通过验证集防止过拟合。

### 4.2 XGBoost

| 参数 | 当前初版值 | 建议搜索范围 |
|---|---:|---|
| n_estimators | 2000 | 500-4000 |
| learning_rate | 0.03 | 0.005-0.15, log scale |
| max_depth | 5 | 3-10 |
| min_child_weight | 3.0 | 0.5-20 |
| subsample | 0.85 | 0.6-1.0 |
| colsample_bytree | 0.85 | 0.6-1.0 |
| reg_lambda | 3.0 | 0.1-100, log scale |
| reg_alpha | 0.0 | 1e-8-10, log scale |
| gamma | 未显式设置 | 0-5 |

XGBoost 是本文表格模型中的强基线，应作为重点调参对象。若使用较大 n_estimators，应启用 early_stopping_rounds，并以验证集 MAE 或 RMSE 选择最佳迭代轮数。

### 4.3 LightGBM

| 参数 | 当前初版值 | 建议搜索范围 |
|---|---:|---|
| n_estimators | 3000 | 500-5000 |
| learning_rate | 0.03 | 0.005-0.15, log scale |
| num_leaves | 63 | 16-512 |
| max_depth | -1 | -1, 3-14 |
| min_child_samples | 50 | 5-150 |
| subsample | 0.85 | 0.6-1.0 |
| colsample_bytree | 0.85 | 0.6-1.0 |
| reg_lambda | 3.0 | 0.1-100, log scale |
| reg_alpha | 0.0 | 1e-8-10, log scale |
| min_split_gain | 未显式设置 | 0-2 |

LightGBM 的主要调参矛盾在于 num_leaves 与 min_child_samples。num_leaves 增大可增强非线性拟合能力，但若 min_child_samples 过小，可能在局部样本上过拟合。

### 4.4 CatBoost

| 参数 | 当前初版值 | 建议搜索范围 |
|---|---:|---|
| iterations | 4000 | 500-5000 |
| learning_rate | 0.03 | 0.005-0.15, log scale |
| depth | 6 | 4-10 |
| l2_leaf_reg | 3.0 | 1-50, log scale |
| subsample | 0.85 | 0.6-1.0 |
| bootstrap_type | Bernoulli | Bernoulli, Bayesian |
| bagging_temperature | 未显式设置 | 0-5, 当 bootstrap_type=Bayesian |
| random_strength | 未显式设置 | 0-5 |
| rsm | 未显式设置 | 0.6-1.0 |

CatBoost 对表格数据通常较稳健，但其 depth、l2_leaf_reg、learning_rate 和 bootstrap 设置会明显影响泛化能力。建议保留 allow_writing_files=False，以避免生成额外训练缓存文件。

### 4.5 MLP

| 参数 | 当前初版值 | 建议搜索范围 |
|---|---:|---|
| MLP_HIDDEN_DIM | 256 | 128, 256, 384, 512, 768, 1024 |
| MLP_BLOCK_COUNT | 3 | 1-5 |
| MLP_HIDDEN_MULT | 2 | 1-4 |
| MLP_DROPOUT | 0.15 | 0.05-0.40 |
| learning_rate | 1e-3 | 1e-4-3e-3, log scale |
| weight_decay | 1e-4 | 1e-8-1e-3, log scale |
| batch_size | 512 | 128, 256, 512, 1024 |
| SmoothL1 beta | 1.0 | 0.3, 0.5, 1.0, 2.0 |

MLP 调参应重点关注隐藏维度、残差块数量、dropout 和学习率。建议启用 Optuna pruning：若前 30-50 个 epoch 验证 MAE 明显劣于当前最优，可提前终止该 trial。

### 4.6 LSTM

| 参数 | 当前初版值 | 建议搜索范围 |
|---|---:|---|
| SEQ_LEN | 2048 | 1024, 2048, 4096 |
| BATCH_SIZE | 64 | 32, 64, 96, 128 |
| LEARNING_RATE | 1e-4 | 5e-5-8e-4, log scale |
| WEIGHT_DECAY | 1e-4 | 1e-8-1e-3, log scale |
| LSTM_HIDDEN_DIM | 128 | 64, 128, 192, 256, 384 |
| LSTM_NUM_LAYERS | 2 | 1-4 |
| LSTM_DROPOUT | 0.20 | 0.05-0.40 |
| LSTM_ATTENTION_DIM | 128 | 64, 128, 256 |
| SEQUENCE_PROJECTOR_DIM | 256 | 128, 256, 384, 512 |
| FUSION_DROPOUT | 0.18 | 0.05-0.35 |
| HEAD_DROPOUT | 0.25 | 0.05-0.40 |

LSTM 训练成本较高，不建议大规模网格搜索。可采用 12-25 个 TPE trials，并将每个 trial 的最大 epoch 降至 60-80；筛选前 3-5 组参数后，再用完整 epoch 重新训练。

### 4.7 WaveNet

| 参数 | 当前初版值 | 建议搜索范围 |
|---|---:|---|
| SEQ_LEN | 4096 | 2048, 4096 |
| BATCH_SIZE | 96 | 48, 64, 96, 128 |
| LEARNING_RATE | 2e-4 | 5e-5-1e-3, log scale |
| WEIGHT_DECAY | 1e-4 | 1e-8-1e-3, log scale |
| WAVENET_RESIDUAL_CHANNELS | 64 | 32, 64, 96, 128 |
| WAVENET_SKIP_CHANNELS | 128 | 64, 128, 192, 256 |
| WAVENET_KERNEL_SIZE | 3 | 2, 3, 5 |
| WAVENET_DILATION_CYCLES | 3 | 2-4 |
| WAVENET_DROPOUT | 0.10 | 0.05-0.35 |
| WAVENET_ATTENTION_DIM | 128 | 64, 128, 256 |
| SEQUENCE_PROJECTOR_DIM | 288 | 128, 256, 384, 512 |
| FUSION_DROPOUT | 0.18 | 0.05-0.35 |

WaveNet 的核心调参对象是残差通道数、skip 通道数、膨胀卷积循环数和 dropout。若显存紧张，应优先控制 SEQ_LEN、BATCH_SIZE 和通道数。

## 5. 建议保存的实验产物

每个模型调参后建议保存以下文件，方便论文复核和附录说明：

| 文件 | 说明 |
|---|---|
| optuna_study_{model}.db | Optuna SQLite study 数据库 |
| optuna_trials_{model}.csv | 每次 trial 的参数、验证指标、状态 |
| best_params_{model}.json | 最优超参数 |
| best_validation_metrics_{model}.json | 最优 trial 的验证集指标 |
| final_test_metrics_{model}.csv | 最终模型在测试集上的指标 |
| bin_metrics_{model}.csv | 分漂移区间 MAE、Bias、欠预测率 |
| threshold_metrics_{model}.csv | 高漂移阈值 precision、recall、F1 |
| training_metadata_{model}.json | 随机种子、数据版本、特征版本、运行环境 |

## 6. 论文中建议采用的表述

### 6.1 实验设计章写法

> 为提高模型比较的公平性，本文在初始基线模型之外，对 LSTM、WaveNet、MLP、Random Forest、XGBoost、LightGBM 和 CatBoost 进行了有限预算超参数优化。调参过程采用 Optuna 的 Tree-structured Parzen Estimator 采样器，以验证集 MAE 为主要目标函数，并在所有模型中保持相同的数据划分、输入特征定义、目标缩放方式和评价指标。测试集不参与任何超参数选择，仅用于最终泛化性能评估。

### 6.2 结果章写法

> 调参后模型结果用于评价各类对比模型在当前数据集和有限搜索空间下的最佳可达性能。需要指出的是，受计算成本限制，时序深度模型的搜索预算低于表格树模型，因此本文结论应解释为“在统一数据划分和有限调参预算下的相对表现”，而非各算法在无限搜索空间中的绝对最优排序。

### 6.3 讨论章写法

> 与初始基线相比，调参后的表格模型和时序模型能够更合理地代表各自算法类别的性能上限，从而降低由于主模型迭代更充分而导致的比较偏差。然而，超参数优化结果仍依赖于搜索空间、验证集分布和计算预算。后续研究可进一步采用嵌套交叉验证或更大规模 AutoML 框架，对不同模型类别进行更严格的统计比较。

## 7. 推荐引用文献

[1] Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019. https://optuna.org/

[2] Optuna Documentation. `optuna.samplers.TPESampler`. The documentation states that TPESampler uses the Tree-structured Parzen Estimator algorithm and selects parameter values by comparing distributions fitted to good and remaining trials. https://optuna.readthedocs.io/en/v3.3.0/reference/samplers/generated/optuna.samplers.TPESampler.html

[3] MLJAR. MLJAR AutoML documentation. The documentation describes MLJAR AutoML as a tool for training machine learning pipelines on tabular data and producing model reports. https://mljar.com/docs/mljar-automl/

[4] MLJAR-supervised Documentation. Steps of AutoML. The documentation describes default algorithms, random search, golden features, feature selection, hill climbing, ensemble and stacking steps. https://supervised.mljar.com/features/automl/

[5] MLJAR-supervised Documentation. Algorithms. The documentation describes stacking and ensemble behavior, including the reuse of top unstacked XGBoost, LightGBM and CatBoost models. https://supervised.mljar.com/features/algorithms/

[6] Ning, C., Xie, Y., and Sun, L. LSTM, WaveNet, and 2D CNN for nonlinear time history prediction of seismic responses. Engineering Structures, 286, 116083, 2023. https://doi.org/10.1016/j.engstruct.2023.116083

[7] Anand, T. P., Pandikkadavath, M. S., Mangalathu, S., and Sahoo, D. R. Machine learning models for seismic analysis of buckling-restrained braced frames. Journal of Building Engineering, 2024. The paper reports that nine supervised ML algorithms were optimized through hyperparameter tuning and validated for multiple EDP prediction. https://doi.org/10.1016/j.jobe.2024.111398

[8] Zhou, J., Qin, X., Hao, Y., Liu, J., Hou, R., and Li, P. Machine Learning-Based Rapid Assessment of Story-Level Seismic Damage in Steel Bundled-Tube Structures. Buildings, 15(20), 3758, 2025. The paper uses Bayesian optimization for RF and XGBoost and reports improved model accuracy after optimization. https://doi.org/10.3390/buildings15203758

[9] Explainable machine learning based seismic response prediction for eccentric bottom frame structures. Journal of Building Engineering, 2026. The article reports nested cross-validation with TPE hyperparameter optimization for LightGBM, XGBoost, CatBoost and related models. https://www.sciencedirect.com/science/article/pii/S2352012426007472

[10] An Explainable Machine Learning Model for Predicting Macroseismic Intensity for Emergency Management. Remote Sensing, 17(10), 1754, 2025. The article reports model comparison under an AutoML/MLJAR-style workflow and 10-fold averaged metrics for multiple machine learning models. https://doi.org/10.3390/rs17101754

