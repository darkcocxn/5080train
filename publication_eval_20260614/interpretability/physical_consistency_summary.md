# TODO12 Interpretability And Physical Consistency Summary

This is a non-training interpretability package. SHAP/permutation inference was not run because the available tabular feature builder attempts to load h5py-backed waveform sequences and the local DLL policy blocked h5py. Instead, this package provides reproducible model-free feature association, binned physical trend checks, TODO9 deep ablation evidence, and case studies with uncertainty.

## Top Feature Associations

| feature | importance_score | spearman_feature_vs_true_drift |
| --- | --- | --- |
| period_1_sec | 0.566959 | 0.616369 |
| mass_to_stiffness | 0.444149 | 0.481689 |
| wave_to_structure_period_ratio | 0.41429 | -0.431901 |
| num_floors | 0.357794 | 0.388568 |
| floor_mass | 0.330161 | 0.368938 |
| wave_intensity_tail_risk | 0.308668 | 0.325614 |
| wave_cav_tail_risk | 0.28848 | 0.305909 |
| k_base_1_4 | 0.282052 | -0.292101 |

## Claim Boundary

- Supported: engineering feature trends and ablation/case-study evidence are documented.
- Not supported: do not call `shap_summary.png` a true SHAP plot; it is a feature association summary saved under the expected TODO12 filename for artifact compatibility.
