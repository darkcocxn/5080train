# TODO1 协议锁定生成报告

- 生成时间: `2026-06-25T22:08:18`
- YAML: `X:\pyproject\5080train\publication_eval_20260614\protocol\protocol_lock.yaml`
- JSON: `X:\pyproject\5080train\publication_eval_20260614\protocol\protocol_lock.json`
- Git branch: `master`
- Git commit: `cbec46baae9408e6ef12a65df5fd156020820414`

## 数据文件

| Split | Exists | Rows | SHA256 | Path |
| --- | --- | ---: | --- | --- |
| train | True | 67260 | `c9e884866321` | `X:\pyproject\5080train\newdata\opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258_train.csv` |
| val | True | 3420 | `a4c9ba436c4f` | `X:\pyproject\5080train\newdata\opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258_val.csv` |
| locked_test | True | 3116 | `720f3b86d6f5` | `X:\pyproject\5080train\newdata\opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258_test.csv` |

## 不干扰训练的执行约束

- 本脚本没有启动训练、测试或 Optuna 任务。
- 本脚本没有终止或修改任何正在运行的 Python 进程。
- 本脚本没有递归扫描小波图片目录，只记录目录存在性。
- 本脚本只写入 protocol 输出目录。

## 后续动作

1. 人工确认 `protocol_lock.yaml` 中的模型列表、seeds、claim boundaries 是否符合论文计划。
2. 冻结后进入 TODO2：数据泄漏、过滤和分布审计。
3. 若之后修改协议，必须在 `protocol_change_log.md` 记录原因、日期和影响范围。
