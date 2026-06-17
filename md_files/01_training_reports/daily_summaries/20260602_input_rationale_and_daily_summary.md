# 输入选择理由与今日工作总结

生成日期：2026-06-02

## 1. 结论先行

本项目把模型输入分成两条路线，不是遗漏地震波时程，而是刻意按模型家族来设计：

- `catboostv1`、`lightgbmv1`、`randomforestv1`、`xgboostv1`、`mlpv1`
  - 输入为固定长度表格特征
  - 具体是 `63` 个标量/工程特征 + `256` 个由地震波时程压缩得到的波形特征
- `lstmv1`、`wavenetv1`
  - 输入为原始地震波时程序列 + `63` 个标量特征
  - 其中序列来自 `txt_path`

这种区别的核心原因是：

- 树模型和表格 MLP 更适合固定长度特征向量
- LSTM / WaveNet 这类序列模型更适合直接建模时间依赖、局部模式和长程依赖

下面的解释结合了文献与本项目代码。关于“为什么项目最终选成这个具体形式”，其中有一部分是基于文献与模型性质做出的工程推断，我会明确标出来。

## 2. 本项目里各模型实际输入了什么

### 2.1 表格类模型

适用模型：

- `catboostv1`
- `lightgbmv1`
- `randomforestv1`
- `xgboostv1`
- `mlpv1`

代码依据：

- [tabular_model_common.py](C:/Users/jiaotong1/5080train/tabular_model_common.py:84)
- [tabular_model_common.py](C:/Users/jiaotong1/5080train/tabular_model_common.py:208)

输入组成：

- 结构参数：`num_floors`、`floor_mass`、`floor_height`、`k_base_1_4`、`Fy_add`
- 地震动统计特征：`period_1_sec`、`wave_pga`、`wave_rms`、`wave_mean_abs`、`wave_cav`、`wave_arias_proxy`、`wave_duration_5_95`、`wave_zero_crossing_rate`、`wave_dominant_freq`、`wave_spectral_centroid`、`wave_predominant_period`、`wave_intensity_score`
- 派生工程特征：`log1p_*`、周期比、交叉项、阻尼器布置展开特征、tail-risk proxy 等
- 来自地震波时程的波形压缩特征：
  - 先用 `txt_path` 读取时程
  - 再按 `seq_len=4096` 和 `acc_scaled` 通道构造序列
  - 最后等距抽成 `256` 维固定长度特征

因此，表格类模型并不是“没有地震波时程输入”，而是“输入的是时程的压缩表示”。

### 2.2 序列类模型

适用模型：

- `lstmv1`
- `wavenetv1`

代码依据：

- [sequence_model_common.py](C:/Users/jiaotong1/5080train/sequence_model_common.py:979)
- [lstmv1.py](C:/Users/jiaotong1/5080train/lstmv1/lstmv1.py:104)
- [wavenetv1.py](C:/Users/jiaotong1/5080train/wavenetv1/wavenetv1.py:101)

输入组成：

- 原始地震波时程序列，来自 `txt_path`
- 两个序列通道：`acc_scaled`、`abs_acc_scaled`
- `lstmv1` 使用 `seq_len=2048`
- `wavenetv1` 使用 `seq_len=4096`
- 再配合 `63` 个标量特征一起输入

因此，序列类模型是“直接利用原始时程结构”。

## 3. 为什么要这样选输入

## 3.1 树模型和表格 MLP 更适合固定长度特征向量

文献上，`XGBoost`、`LightGBM`、`CatBoost` 本质上都属于梯度提升树框架，学习对象是样本的固定维特征向量，而不是任意长度的时间序列。

- `XGBoost` 论文把它定义为 tree boosting system，本质是在特征空间上做分裂与增益优化。[Chen & Guestrin, 2016](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
- `LightGBM` 论文也是标准 GBDT 设定，关注的是高维特征、样本规模、特征扫描与直方图构建效率。[Ke et al., 2017](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
- `CatBoost` 论文同样把输入写成样本特征向量 `x` 与目标 `y` 的集合。[Prokhorenkova et al., 2018](https://papers.nips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf)

基于这些论文，可以做一个明确的工程推断：

- 这类模型天然假设输入是固定长度特征
- 它们没有像 RNN/TCN 那样针对“时间顺序、局部邻域、长程依赖”的结构偏置
- 如果把几千个时间点直接平铺成超长向量，理论上能训，但通常维数更高、冗余更强、效率更差，也更难稳定利用时序结构

所以，把时程先变成固定长度工程特征或压缩波形特征，再喂给树模型，是更符合这类算法工作方式的做法。

## 3.2 地震工程文献中，传统 surrogate / ML 也常走“结构参数 + 地震动特征”路线

这不是本项目独有的设计，结构工程文献本来就经常这么做。

- PBEE / fragility 文献长期把结构响应和地震动强度指标联系起来，例如把失效概率表示成地震动强度度量 `PGA`、谱加速度等的函数。[Mai et al., 2017](https://arxiv.org/abs/1704.03876)
- Samadian 等关于钢框架 surrogate model 的工作，明确使用“影响 EDP 的关键结构因素”来构建 surrogate，并比较多种 ML 方法，结论里指出 `CatBoost` 对 NLTHA 与 pushover 参数预测表现最好。[Samadian et al., 2024](https://research.tees.ac.uk/en/publications/surrogate-models-for-seismic-and-pushover-response-prediction-of-/)
- 同一研究团队关于 steel frame meta-database 的综述，也回顾了很多 surrogate 工作把 ground motion parameters、结构参数等作为输入特征来预测响应或 fragility 结果。[Samadian et al., 2024](https://research.tees.ac.uk/ws/portalfiles/portal/77114727/Meta_databases_of_steel_frame_buildings_for_surrogate_modelling_and_machine_learning-based_feature_importance_analysis.pdf)

所以，对树模型使用“结构参数 + 地震动工程特征 + 适度压缩后的波形特征”，是符合该领域已有做法的。

## 3.3 LSTM / WaveNet 这类序列模型更适合直接吃原始时程

这条路线对应另一类文献。

- `WaveNet` 原始论文就是直接对 raw audio waveforms 建模，核心优势来自对原始序列的局部模式和长程依赖进行建模。[van den Oord et al., 2016](https://arxiv.org/abs/1609.03499)
- `LSTM` 在结构地震响应预测里也常直接面向 response history / excitation history 工作。MIT 的工作明确提出了 deep LSTM 用于 data-driven structural seismic response modeling。[Zhang et al., 2019](https://energy.mit.edu/publication/deep-long-short-term-memory-networks-for-nonlinear-structural-seismic-response-prediction/)
- Xu 等则进一步提出 recursive LSTM 来预测任意长度和采样率下的 nonlinear structural seismic response，并强调它适用于不同频谱特征与幅值的地震动。[Xu et al., 2022](https://www.sciencedirect.com/science/article/pii/S0141029621015133)
- 更直接相关的是 Ning 等在 `Engineering Structures` 里把 `LSTM`、`WaveNet` 和 `2D CNN` 放在一起，目标就是预测 nonlinear seismic response time histories。[Ning et al., 2023](https://www.sciencedirect.com/science/article/pii/S0141029623004972)

这些工作背后的共同逻辑是：

- 原始地震波里有时序结构、持续时间、频率演化、非平稳性、能量累积等信息
- 这些信息并不一定能被少量 IM 或统计量完整保留
- 序列模型正好能直接对这种时序信息建模

因此，对 `lstmv1` 和 `wavenetv1` 来说，直接输入原始时程是更自然的。

## 3.4 本项目为什么不是“树模型只吃 IM”，而是额外加了 256 维波形特征

这部分是结合文献与代码后的工程解释。

如果完全照传统表格路线，树模型只吃 IM / 结构参数也可以；但本项目做了一个折中：

- 保留表格模型最擅长的固定长度输入形式
- 同时不给它们完全丢掉时程形状信息
- 所以增加了 `256` 维下采样波形特征

这个设计的好处是：

- 保留了树模型/MLP 对固定长度输入的适配性
- 比只用少量 IM 更可能保留部分波形形状和时域信息
- 训练和推理成本仍远低于直接把完整长序列交给表格模型

也就是说，表格模型这一支输入其实是“工程特征路线”和“有限波形信息保留”的折中方案。

## 4. 参考文献与其对本项目的意义

### 4.1 算法性质

1. [Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)
用途：说明 XGBoost 的本体是固定特征向量上的 tree boosting。

2. [Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.](https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)
用途：说明 LightGBM 面向高维固定特征、特征扫描和 GBDT 训练效率优化。

3. [Prokhorenkova, L., et al. (2018). CatBoost: unbiased boosting with categorical features. NeurIPS 2018.](https://papers.nips.cc/paper/2018/file/14491b756b3a51daac41c24863285549-Paper.pdf)
用途：说明 CatBoost 同样是在固定特征向量上学习。

4. [van den Oord, A., et al. (2016). WaveNet: A Generative Model for Raw Audio.](https://arxiv.org/abs/1609.03499)
用途：说明 WaveNet 类模型的设计初衷就是直接建模原始波形序列。

### 4.2 结构地震响应 / surrogate 相关

5. [Zhang, R., et al. (2019). Deep long short-term memory networks for nonlinear structural seismic response prediction.](https://energy.mit.edu/publication/deep-long-short-term-memory-networks-for-nonlinear-structural-seismic-response-prediction/)
用途：说明 LSTM 已被用于 data-driven structural seismic response modeling。

6. [Xu, Z., et al. (2022). Recursive long short-term memory network for predicting nonlinear structural seismic response. Engineering Structures.](https://www.sciencedirect.com/science/article/pii/S0141029621015133)
用途：说明 LSTM 可直接面向不同频谱特征与任意长度采样率的地震响应时程。

7. [Ning, X., et al. (2023). LSTM, WaveNet, and 2D CNN for nonlinear time history prediction of seismic responses. Engineering Structures.](https://www.sciencedirect.com/science/article/pii/S0141029623004972)
用途：说明 LSTM / WaveNet / CNN 直接用于非线性地震响应时程预测是合理路线。

8. [Samadian, D., et al. (2024). Surrogate models for seismic and pushover response prediction of steel special moment resisting frames.](https://research.tees.ac.uk/en/publications/surrogate-models-for-seismic-and-pushover-response-prediction-of-/)
用途：说明 surrogate 领域会用结构与响应相关特征建表格模型，且该研究中 `CatBoost` 表现突出。

9. [Samadian, D., et al. (2024). Meta databases of steel frame buildings for surrogate modelling and machine learning-based feature importance analysis.](https://research.tees.ac.uk/ws/portalfiles/portal/77114727/Meta_databases_of_steel_frame_buildings_for_surrogate_modelling_and_machine_learning-based_feature_importance_analysis.pdf)
用途：说明结构 surrogate 文献中常把结构参数、ground motion parameters 等作为输入。

10. [Mai, C., Konakli, K., & Sudret, B. (2017). Seismic fragility curves for structures using non-parametric representations.](https://arxiv.org/abs/1704.03876)
用途：说明 earthquake engineering 中长期存在以 IM 表征地震动并映射到响应/失效概率的传统。

## 5. 今日实际完成的工作

## 5.1 修复的问题

### 5.1.1 `catboostv1` 输出失败

问题：

- Windows 路径过长，验证/测试指标文件落盘时报错。

处理：

- 在 [tabular_model_common.py](C:/Users/jiaotong1/5080train/tabular_model_common.py:430) 和 [sequence_model_common.py](C:/Users/jiaotong1/5080train/sequence_model_common.py:1765) 中把结果文件名改成紧凑格式
- 在 [sequence_model_common.py](C:/Users/jiaotong1/5080train/sequence_model_common.py:250) 中让 `save_json()` 自动创建父目录

### 5.1.2 `lstmv1` / `wavenetv1` 采样器报错

问题：

- `WeightedRandomSampler` 权重数组只读，训练前报：
  - `ValueError: output array is read-only`

处理：

- 在 [sequence_model_common.py](C:/Users/jiaotong1/5080train/sequence_model_common.py:1085) 中把权重数组显式转为 `copy=True` 的可写 `numpy` 数组

## 5.2 今天跑完的模型

今天完成了以下 7 个 `v1` 模型的训练 / 测试或测试验证：

- `catboostv1`
- `lightgbmv1`
- `randomforestv1`
- `xgboostv1`
- `mlpv1`
- `lstmv1`
- `wavenetv1`

其中：

- `catboostv1`、`lightgbmv1`、`randomforestv1`、`xgboostv1`、`mlpv1`：训练和测试均完成
- `lstmv1`：训练和测试完成
- `wavenetv1`：默认长训练在终端时限前未自然结束，但最佳权重已保存，并完成了测试

## 5.3 当前结果概览

按测试集 `R2` 排序：

| 排名 | 模型 | 测试R2 | 测试MAE | 测试RMSE |
| --- | --- | ---: | ---: | ---: |
| 1 | `wavenetv1` | 0.8664 | 0.000355626 | 0.000496153 |
| 2 | `lstmv1` | 0.8586 | 0.000374142 | 0.000510350 |
| 3 | `lightgbmv1` | 0.7794 | 0.000421465 | 0.000637425 |
| 4 | `catboostv1` | 0.7754 | 0.000431681 | 0.000643270 |
| 5 | `xgboostv1` | 0.7513 | 0.000459898 | 0.000676930 |
| 6 | `randomforestv1` | 0.6850 | 0.000557378 | 0.000761807 |
| 7 | `mlpv1` | 0.5439 | 0.000673718 | 0.000916616 |

当前观察：

- 原始时程序列模型 `wavenetv1` 和 `lstmv1` 明显领先
- 表格模型里 `lightgbmv1` 与 `catboostv1` 最强
- `mlpv1` 在当前输入设计下明显弱于树模型和序列模型

## 5.4 相关产物

今天新增或更新的关键文档：

- [v1_baselines_train_test_summary_20260602.md](C:/Users/jiaotong1/5080train/output/v1_baselines_train_test_summary_20260602.md)
- [20260602_input_rationale_and_daily_summary.md](C:/Users/jiaotong1/5080train/output/20260602_input_rationale_and_daily_summary.md)

关键代码改动位置：

- [sequence_model_common.py](C:/Users/jiaotong1/5080train/sequence_model_common.py:250)
- [sequence_model_common.py](C:/Users/jiaotong1/5080train/sequence_model_common.py:1085)
- [sequence_model_common.py](C:/Users/jiaotong1/5080train/sequence_model_common.py:1765)
- [tabular_model_common.py](C:/Users/jiaotong1/5080train/tabular_model_common.py:430)

## 6. 对后续工作的建议

如果接下来继续推进，建议顺序是：

1. 让 `wavenetv1` 再完整跑一轮到自然早停
2. 对 `lightgbmv1`、`catboostv1`、`xgboostv1` 做 `log1p target` 与 sample-weight 消融
3. 补一份把 `2dcnnv*` 一起纳入的总对比文档

如果目标是论文叙事，目前最自然的说法是：

- 树模型采用“工程特征 + 压缩波形”的固定长度输入路线
- LSTM / WaveNet 采用“原始时程 + 标量特征”的时序建模路线
- 结果上，直接时序建模在当前数据集上取得了更高精度
