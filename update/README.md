# Optuna/TPE 自动化调优脚本说明

本目录用于对论文中参与比较的代理模型进行有限预算超参数优化。脚本首先采用 Optuna 的 Tree-structured Parzen Estimator（TPE）采样器，在保持原有数据划分、输入特征和评价指标体系不变的前提下，为各算法建立可重复运行、可追溯保存结果的调优入口。

该实现的直接目的不是替代正式训练脚本，而是为对比模型提供更公平的调参流程，降低“主模型经过多轮迭代优化，而其他模型仅为初版参数”的比较偏差。

## 一、本次脚本写作过程

### 1. 梳理现有训练脚本

脚本编写首先从项目已有模型入口开始，而不是另建一套训练框架。已检查并复用的主要文件包括：

- `randomforestv1/randomforestv1.py`
- `xgboostv1/xgboostv1.py`
- `lightgbmv1/lightgbmv1.py`
- `catboostv1/catboostv1.py`
- `mlpv1/mlpv1.py`
- `lstmv1/lstmv1.py`
- `wavenetv1/wavenetv1.py`
- `2dcnnv11/2dcnnv11.py`
- `tabular_model_common.py`
- `sequence_model_common.py`

其中，Random Forest、XGBoost、LightGBM、CatBoost 和 MLP 共用表格特征处理逻辑；LSTM 与 WaveNet 共用序列建模逻辑；2D-CNN 主模型为当前论文中的多模态时频图模型。因此，本次没有把 `2dcnnv1` 到 `2dcnnv10` 的历史迭代版本分别作为独立算法调参，而是只为当前论文主模型版本建立 `2dcnn/` 调优入口。

### 2. 确定目录结构

为了便于论文复核和后续扩展，每个算法单独建立子目录：

- `randomforest/`
- `xgboost/`
- `lightgbm/`
- `catboost/`
- `mlp/`
- `lstm/`
- `wavenet/`
- `2dcnn/`
- `common/`

各算法目录只保存一个 `tune_optuna_tpe.py`，负责定义该算法的搜索空间。公共逻辑统一放入 `common/optuna_tpe_common.py`，包括命令行参数、TPE study 创建、trial 配置生成、评价指标读取、结果保存和 dry-run 检查。

这种结构的优点是：算法搜索空间清楚，公共逻辑不重复；后续若增加 MLJAR、Bayesian Optimization、ASHA 或 Hyperband，只需新增公共模块和入口脚本，不需要重写现有训练代码。

### 3. 保持与原项目训练流程一致

本次实现没有重写数据加载、特征构造、标签缩放、样本权重或验证集评估逻辑，而是尽量复用现有项目函数。

表格模型调优复用：

- `prepare_tabular_training_data`
- `maybe_fit_with_sample_weight`
- `inverse_target`
- `calculate_regression_metrics`
- `calculate_selection_score`

MLP 调优复用：

- `train_torch_mlp_tabular_model`

序列模型调优复用：

- `train_sequence_model`

2D-CNN 调优复用：

- `2dcnnv11.py` 中的 `train`
- `2dcnnv11.py` 中的 `plot_results`

这样可以保证调优阶段与论文已有实验使用相同的数据处理和评价口径，避免由于重新实现训练流程而引入额外差异。

### 4. 设计统一优化目标

默认优化目标设为 `selection`，即使用项目现有的验证集综合选择指标；也可以通过命令行切换为 `mae` 或 `rmse`。

```powershell
--objective-metric selection
--objective-metric mae
--objective-metric rmse
```

保留 `selection` 的原因是：本文预测目标为最大层间位移角，单纯 MAE 能反映整体平均误差，但对中高漂移区间的低估风险不够敏感。项目中已有的 selection 指标已经融合整体误差、RMSE 和尾部误差，因此更符合当前论文对结构响应安全侧预测的关注。

若论文主表希望使用更常规的模型选择依据，也可以统一改为 `--objective-metric mae`，并在文中明确说明所有模型均以验证集 MAE 选择最优参数。

### 5. 设计 trial 输出方式

每个 trial 的训练结果默认保存到：

```text
update/<算法>/runs/trial_xxxx/
```

每个算法目录下还会保存：

- `best_params.json`: 最优 trial 的配置参数。
- `study_summary.json`: study 摘要、最优分数、最优 trial、模型目录。
- `optuna_trials.csv`: 所有 trial 的参数、状态和用户属性。
- `runs/trial_xxxx/optuna_trial.json`: 单次 trial 的参数、指标和目标值。

这种保存方式使论文中可以追溯每个模型的调参预算、搜索空间、最优验证集结果和最终参数来源。

### 6. 处理环境变量覆盖问题

原始训练脚本支持 `SURMOD_*` 环境变量覆盖数据路径、训练轮数、batch size、模型输出目录等。本次调优脚本保留这些机制，同时在 trial 内临时清理各模型专属环境变量，例如：

- `SURMOD_RF_*`
- `SURMOD_XGB_*`
- `SURMOD_LGB_*`
- `SURMOD_CAT_*`

这样做是为了防止外部环境变量覆盖 Optuna 在 trial 中采样得到的参数。数据路径类环境变量仍然可用，例如：

- `SURMOD_DATASET_DIRS`
- `SURMOD_TRAIN_CSV`
- `SURMOD_VAL_CSV`
- `SURMOD_TEST_CSV`

因此，调优时仍可以指定数据集位置，但模型超参数以 Optuna trial 为准。

### 7. 增加依赖并验证入口

本次在 `pyproject.toml` 中增加：

```toml
"optuna>=4.3.0"
```

并更新 `uv.lock`。完成后已进行轻量验证：

- 所有新增 Python 脚本通过 `py_compile`。
- 各算法入口均通过 `--dry-run` 检查。
- 验证过程只抽样参数，不启动正式训练。

示例：

```powershell
uv run python update/randomforest/tune_optuna_tpe.py --dry-run
uv run python update/lstm/tune_optuna_tpe.py --dry-run
uv run python update/2dcnn/tune_optuna_tpe.py --dry-run
```

## 二、调优脚本使用方法

### 0. 使用 BAT 脚本启动

本次已为每个算法添加 `启动参数优化.bat`。双击时会使用默认“预搜索”参数，适合先检查调优流程和初步筛选候选参数；若需要正式搜索，可在命令行中向 bat 传入自定义参数。

总入口：

```powershell
update\启动参数优化.bat
```

单算法入口：

```powershell
update\randomforest\启动参数优化.bat
update\xgboost\启动参数优化.bat
update\lightgbm\启动参数优化.bat
update\catboost\启动参数优化.bat
update\mlp\启动参数优化.bat
update\lstm\启动参数优化.bat
update\wavenet\启动参数优化.bat
update\2dcnn\启动参数优化.bat
```

命令行自定义参数示例：

```powershell
update\xgboost\启动参数优化.bat --trials 80 --data-use-ratio 1.0 --n-jobs 8 --storage sqlite:///update/xgboost/optuna.db --study-name xgboost_full --resume
update\lstm\启动参数优化.bat --trials 25 --num-epochs 80 --device cuda --num-workers 0
```

各 bat 的默认参数如下：

| 算法 | 默认参数 |
|---|---|
| Random Forest | `--trials 30 --data-use-ratio 0.30 --n-jobs 8` |
| XGBoost | `--trials 30 --data-use-ratio 0.30 --n-jobs 8` |
| LightGBM | `--trials 30 --data-use-ratio 0.30 --n-jobs 8` |
| CatBoost | `--trials 30 --data-use-ratio 0.30 --n-jobs 8` |
| MLP | `--trials 20 --data-use-ratio 0.30 --num-epochs 40 --device auto` |
| LSTM | `--trials 15 --data-use-ratio 0.30 --num-epochs 30 --num-workers 0 --device auto` |
| WaveNet | `--trials 15 --data-use-ratio 0.30 --num-epochs 30 --num-workers 0 --device auto` |
| 2D-CNN | `--trials 15 --data-use-ratio 0.30 --num-epochs 30 --num-workers 0 --device auto` |

### 1. 快速检查搜索空间

```powershell
uv run python update/randomforest/tune_optuna_tpe.py --dry-run
uv run python update/xgboost/tune_optuna_tpe.py --dry-run
uv run python update/lightgbm/tune_optuna_tpe.py --dry-run
uv run python update/catboost/tune_optuna_tpe.py --dry-run
uv run python update/mlp/tune_optuna_tpe.py --dry-run
uv run python update/lstm/tune_optuna_tpe.py --dry-run
uv run python update/wavenet/tune_optuna_tpe.py --dry-run
uv run python update/2dcnn/tune_optuna_tpe.py --dry-run
```

### 2. 预搜索

建议先使用较小数据比例和较少 epoch 做低成本预搜索：

```powershell
uv run python update/xgboost/tune_optuna_tpe.py --trials 30 --data-use-ratio 0.30 --n-jobs 8
uv run python update/lightgbm/tune_optuna_tpe.py --trials 30 --data-use-ratio 0.30 --n-jobs 8
uv run python update/catboost/tune_optuna_tpe.py --trials 30 --data-use-ratio 0.30 --n-jobs 8
uv run python update/mlp/tune_optuna_tpe.py --trials 20 --data-use-ratio 0.30 --num-epochs 40 --device cuda
uv run python update/lstm/tune_optuna_tpe.py --trials 15 --data-use-ratio 0.30 --num-epochs 30 --device cuda
uv run python update/wavenet/tune_optuna_tpe.py --trials 15 --data-use-ratio 0.30 --num-epochs 30 --device cuda
```

### 3. 正式搜索

正式搜索应使用完整训练集，并将神经网络训练轮数提高到论文实验设定。建议保存 SQLite study，便于断点续搜：

```powershell
uv run python update/lightgbm/tune_optuna_tpe.py --storage sqlite:///update/lightgbm/optuna.db --study-name lightgbm_full --resume --trials 80
```

### 4. 最终复核

完成调参后，应读取 `best_params.json`，将最优参数回填到正式训练脚本或新建最终训练配置中，再使用完整训练预算训练一次，并只在最终阶段评价测试集。

## 三、各模型搜索空间设计说明

### 1. Random Forest

调参重点为树数量、树深度、叶节点最小样本数、分裂最小样本数和特征子采样比例。该类模型训练相对稳定，但在连续响应预测中容易产生均值回归，因此搜索空间允许较深树和较小叶节点样本数。

主要参数：

- `N_ESTIMATORS`
- `MAX_DEPTH`
- `MIN_SAMPLES_LEAF`
- `MIN_SAMPLES_SPLIT`
- `MAX_FEATURES`
- `WAVEFORM_FEATURE_COUNT`

### 2. XGBoost

XGBoost 是强表格基线，因此搜索空间较完整，覆盖学习率、树数、树深、叶节点约束、样本采样、特征采样和正则化。学习率和正则项使用 log scale。

主要参数：

- `N_ESTIMATORS`
- `LEARNING_RATE`
- `MAX_DEPTH`
- `MIN_CHILD_WEIGHT`
- `SUBSAMPLE`
- `COLSAMPLE_BYTREE`
- `REG_LAMBDA`
- `REG_ALPHA`
- `WAVEFORM_FEATURE_COUNT`

### 3. LightGBM

LightGBM 的关键矛盾在于 `NUM_LEAVES`、`MAX_DEPTH` 和 `MIN_CHILD_SAMPLES`。搜索空间允许较大叶节点数，同时通过子节点样本数和正则化限制过拟合。

主要参数：

- `N_ESTIMATORS`
- `LEARNING_RATE`
- `NUM_LEAVES`
- `MAX_DEPTH`
- `MIN_CHILD_SAMPLES`
- `SUBSAMPLE`
- `COLSAMPLE_BYTREE`
- `REG_LAMBDA`
- `REG_ALPHA`
- `WAVEFORM_FEATURE_COUNT`

### 4. CatBoost

CatBoost 搜索空间主要围绕迭代次数、学习率、树深、L2 正则和样本采样比例展开。保留原脚本中的 `allow_writing_files=False`，避免生成额外训练缓存文件。

主要参数：

- `ITERATIONS`
- `LEARNING_RATE`
- `DEPTH`
- `L2_LEAF_REG`
- `SUBSAMPLE`
- `WAVEFORM_FEATURE_COUNT`

### 5. MLP

MLP 搜索空间覆盖 batch size、学习率、权重衰减、SmoothL1 beta、早停耐心、隐藏维度、残差块数量、隐藏层扩展倍率、dropout 和波形下采样特征数量。

主要参数：

- `BATCH_SIZE`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `SMOOTH_L1_BETA`
- `EARLY_STOPPING_PATIENCE`
- `MLP_HIDDEN_DIM`
- `MLP_BLOCK_COUNT`
- `MLP_HIDDEN_MULT`
- `MLP_DROPOUT`
- `WAVEFORM_FEATURE_COUNT`

### 6. LSTM

LSTM 训练成本高，因此搜索空间采用有限预算策略，重点搜索序列长度、batch size、学习率、权重衰减、卷积 stem 通道数、LSTM 隐藏维度、层数、dropout、注意力维度、序列投影维度、标量编码维度和融合层参数。

主要参数：

- `SEQ_LEN`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `SEQ_STEM_CHANNELS`
- `LSTM_HIDDEN_DIM`
- `LSTM_NUM_LAYERS`
- `LSTM_DROPOUT`
- `LSTM_ATTENTION_DIM`
- `SEQUENCE_PROJECTOR_DIM`
- `SCALAR_EMBED_DIM`
- `FUSION_OUTPUT_DIM`

### 7. WaveNet

WaveNet/TCN 的重点为序列长度、残差通道、skip 通道、卷积核大小、膨胀卷积循环数、每轮膨胀率序列、dropout 和融合层维度。若显存不足，应优先降低 `SEQ_LEN`、`BATCH_SIZE` 和通道数。

主要参数：

- `SEQ_LEN`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `WAVENET_RESIDUAL_CHANNELS`
- `WAVENET_SKIP_CHANNELS`
- `WAVENET_KERNEL_SIZE`
- `WAVENET_DILATION_CYCLES`
- `WAVENET_DILATIONS_PER_CYCLE`
- `WAVENET_DROPOUT`
- `SEQUENCE_PROJECTOR_DIM`
- `FUSION_OUTPUT_DIM`

### 8. 2D-CNN

2D-CNN 是当前论文主模型的继续调优入口，搜索空间只针对当前 `2dcnnv11` 版本，不包括历史迭代版本。搜索重点为图像尺寸、CNN 通道数、学习率、权重衰减、CNN dropout、投影维度、标量编码维度、融合维度、head dropout 和时频遮挡增强概率。

主要参数：

- `IMAGE_SIZE`
- `BATCH_SIZE`
- `LEARNING_RATE`
- `WEIGHT_DECAY`
- `CNN_CHANNELS`
- `CNN_DROPOUT`
- `CNN_PROJECTOR_DIM`
- `SCALAR_EMBED_DIM`
- `FUSION_OUTPUT_DIM`
- `HEAD_DROPOUT`
- `TIME_FREQ_MASK_PROB`

## 四、论文写作建议

### 1. 实验设计中的表述

可写为：

> 为提高模型比较的公平性，本文在初始基线模型之外，对 Random Forest、XGBoost、LightGBM、CatBoost、MLP、LSTM 和 WaveNet 等对比模型进行了有限预算超参数优化。调参过程采用 Optuna 的 TPE 采样器，并在所有模型中保持相同的数据划分、输入定义、目标缩放方式和验证集评价指标。测试集不参与任何超参数选择，仅用于最终泛化性能评估。

### 2. 结果分析中的表述

可写为：

> 调参后模型结果用于评价各类对比模型在当前数据集和有限搜索空间下的可达性能。需要指出的是，受计算成本限制，时序深度模型的搜索预算低于表格树模型，因此本文结论应解释为“在统一数据划分和有限调参预算下的相对表现”，而非各算法在无限搜索空间中的绝对最优排序。

### 3. 讨论中的表述

可写为：

> 与初始基线相比，调参后的对比模型能够更合理地代表各自算法类别的性能，从而降低主模型迭代更充分所导致的比较偏差。然而，超参数优化结果仍依赖于搜索空间、验证集分布和计算预算。后续研究可进一步采用嵌套交叉验证或更大规模 AutoML 框架，对不同模型类别进行更严格的统计比较。

## 五、参考文献与资料依据

[1] Akiba, T., Sano, S., Yanase, T., Ohta, T., and Koyama, M. Optuna: A Next-generation Hyperparameter Optimization Framework. Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2019: 2623-2631. https://arxiv.org/abs/1907.10902

[2] Bergstra, J., Bardenet, R., Bengio, Y., and Kégl, B. Algorithms for Hyper-Parameter Optimization. Advances in Neural Information Processing Systems 24, 2011. https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization

[3] Optuna Documentation. `optuna.samplers.TPESampler`. The documentation describes TPESampler as a sampler using the Tree-structured Parzen Estimator algorithm. https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.TPESampler.html

[4] MLJAR-supervised Documentation. Steps of AutoML. The documentation describes default algorithms, random search, feature selection, hill climbing, ensemble and stacking steps. https://supervised.mljar.com/features/automl/

[5] MLJAR-supervised Documentation. AutoML modes. The documentation describes Explain, Perform, Compete and Optuna-style operation modes. https://supervised.mljar.com/features/modes/

[6] MLJAR GitHub Repository. `mljar-supervised`: Python package for AutoML on tabular data with feature engineering, hyperparameter tuning, explanations and automatic documentation. https://github.com/mljar/mljar-supervised

[7] Breiman, L. Random Forests. Machine Learning, 45, 5-32, 2001. https://doi.org/10.1023/A:1010933404324

[8] Chen, T., and Guestrin, C. XGBoost: A Scalable Tree Boosting System. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2016: 785-794. https://doi.org/10.1145/2939672.2939785

[9] Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., and Liu, T.-Y. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. Advances in Neural Information Processing Systems 30, 2017. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

[10] Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., and Gulin, A. CatBoost: Unbiased Boosting with Categorical Features. Advances in Neural Information Processing Systems 31, 2018. https://papers.nips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features

[11] Hochreiter, S., and Schmidhuber, J. Long Short-Term Memory. Neural Computation, 9(8), 1735-1780, 1997. https://doi.org/10.1162/neco.1997.9.8.1735

[12] van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., Graves, A., Kalchbrenner, N., Senior, A., and Kavukcuoglu, K. WaveNet: A Generative Model for Raw Audio. arXiv preprint, 2016. https://arxiv.org/abs/1609.03499

[13] Bai, S., Kolter, J. Z., and Koltun, V. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. arXiv preprint, 2018. https://arxiv.org/abs/1803.01271

[14] Ning, C., Xie, Y., and Sun, L. LSTM, WaveNet, and 2D CNN for nonlinear time history prediction of seismic responses. Engineering Structures, 286, 116083, 2023. https://doi.org/10.1016/j.engstruct.2023.116083

[15] He, X., Zhao, K., and Chu, X. AutoML: A Survey of the State-of-the-Art. Knowledge-Based Systems, 212, 106622, 2021. https://doi.org/10.1016/j.knosys.2020.106622
