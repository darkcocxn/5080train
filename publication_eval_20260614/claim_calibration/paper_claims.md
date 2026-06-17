# Calibrated Paper Claims

## Supported Claims

| claim_id | claim | support_level | evidence |
| --- | --- | --- | --- |
| C1 | The 2D-CNN achieves the lowest locked-test MAE among completed models: 0.000314189. | strong | Table 2; TODO6 multi-seed final evaluation |
| C2 | 2D-CNN significantly improves MAE over LightGBM by 0.000145828 with positive bootstrap CI. | strong | Table 4 paired hierarchical bootstrap |
| C3 | Locked test is exact unseen-wave and exact unseen-structure relative to train+val; 2D-CNN remains the best model on this locked OOD benchmark. | moderate_strong_with_scope_limit | Table 6; TODO10 |
| C4 | Validation-calibrated adaptive conformal intervals give global locked-test 90% interval coverage of 0.958922. | moderate_with_calibration_split_caveat | Table 7; TODO11 |
| C5 | Tail p95 MAE is best for wavenet; 2D-CNN is competitive but not best on p95 tail MAE. | strong_for_cautious_tail_comparison | Table 3; TODO7 |

## Unsupported Or Must-Not-Write Claims

| claim_id | unsupported_claim | reason | needed_evidence |
| --- | --- | --- | --- |
| U1 | The model is proven reliable for high-drift, near-failure, or steel01-yielded regimes. | Locked test has no target >= 0.010 and no steel01_yielded=1 samples. | Run TODO10 high_drift_opensees_generation_plan and evaluate frozen models. |
| U2 | The surrogate has a quantified OpenSees speedup. | Controlled inference timing and OpenSees runtime are not measured. | Dedicated TODO13 latency benchmark plus OpenSees runtime baseline. |
| U3 | SHAP proves the model learned physically causal mechanisms. | SHAP/permutation was not run; TODO12 uses model-free association and ablation evidence. | Run SHAP/permutation after resolving h5py/model-loading path. |
| U4 | 2D-CNN has calibrated high-drift uncertainty intervals. | Tail coverage at target >= 0.005 drops and high-drift independent test rows are absent. | Independent high-drift stress set with conformal evaluation. |

## Recommended Abstract-Level Claim

Under a locked evaluation protocol with independent multi-seed retraining, the 2D-CNN surrogate achieves the lowest overall locked-test MAE and significantly improves over the strongest completed GBDT baseline on paired tests. The locked test is also exact unseen-wave and unseen-structure relative to train+val, supporting a bounded OOD generalization claim. Tail and uncertainty analyses reveal remaining limitations: high-drift/yielded regimes require new OpenSees stress labels, and global conformal coverage should not be overextended to high-response regimes.
