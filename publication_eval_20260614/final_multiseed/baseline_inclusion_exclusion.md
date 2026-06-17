# TODO4 Baseline Inclusion/Exclusion Review

- Generated at: `2026-06-14T15:23:41`
- Scope: registry/audit only; no baseline training was run.

## Dataset Context

- Train rows: `67260`
- Validation rows: `3420`
- Locked test rows: `3116`
- Raw columns: `47`
- Raw feature count excluding target: `46`
- Task type: `tabular/regression plus waveform/image multimodal surrogate task`
- TODO2 limitation: test has no >=0.010 target samples and no steel01_yielded=1 samples per TODO2

## Package Availability

| Package | Available in current `uv run python` env |
| --- | --- |
| `sklearn` | `True` |
| `xgboost` | `True` |
| `lightgbm` | `True` |
| `catboost` | `True` |
| `optuna` | `True` |
| `autogluon` | `False` |
| `tabpfn` | `True` |
| `tabpfn_extensions` | `True` |
| `tabm` | `True` |
| `rtdl` | `False` |
| `rtdl_num_embeddings` | `True` |

## Registry Summary

| Baseline | Decision | Priority | Package | Action |
| --- | --- | --- | --- | --- |
| `dummy_mean` | include_must_add | P0 | available | Add to TODO5 final evaluator; no Optuna needed. |
| `dummy_median` | include_must_add | P0 | available | Add to TODO5 final evaluator; no Optuna needed. |
| `ridge` | include_must_add | P0 | available | Add RidgeCV or Optuna-tuned Ridge to TODO5. |
| `elasticnet` | include_must_add | P0 | available | Add ElasticNetCV or Optuna-tuned ElasticNet to TODO5. |
| `extratrees` | include_must_add | P0 | available | Add Optuna or fixed strong ExtraTrees baseline to TODO5. |
| `histgradientboosting` | include_must_add | P0 | available | Add HistGradientBoostingRegressor baseline to TODO5. |
| `randomforest` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `xgboost` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `lightgbm` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `catboost` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `mlp` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `lstm` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `wavenet` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `2dcnn_v11_fusion` | already_included_hpo_complete | P0 | available | Use frozen best_params in TODO5 multi-seed final evaluation. |
| `tabm_or_realmlp` | include_conditionally_available_not_integrated | P1_for_top_tier | available | Implement TabM/RealMLP runner for TODO5; if not feasible, document compute/task-fit exclusion. |
| `tabpfn_v2` | include_conditionally_subsample_available | P1_for_top_tier | available | Run as small-data/subsample baseline or document exclusion for full-data comparison. |
| `autogluon_or_tuned_weighted_ensemble` | include_conditionally_or_replace_with_in_repo_ensemble | P1_for_top_tier | missing | If AutoGluon install is allowed, run fixed-budget AutoGluon; otherwise build in-repo ensemble from existing tuned models. |

## Must Add Before Strong Publication Claims

- `dummy_mean`: Required sanity baseline to show the task is not solved by central tendency. Action: Add to TODO5 final evaluator; no Optuna needed.
- `dummy_median`: Robust central-tendency sanity baseline. Action: Add to TODO5 final evaluator; no Optuna needed.
- `ridge`: Required to quantify nonlinearity benefit. Action: Add RidgeCV or Optuna-tuned Ridge to TODO5.
- `elasticnet`: Required to quantify feature sparsity and linear baseline strength. Action: Add ElasticNetCV or Optuna-tuned ElasticNet to TODO5.
- `extratrees`: Complements RandomForest and is a strong, robust tabular baseline. Action: Add Optuna or fixed strong ExtraTrees baseline to TODO5.
- `histgradientboosting`: Useful reproducible GBDT baseline independent of external GBDT libraries. Action: Add HistGradientBoostingRegressor baseline to TODO5.

## Already Included, But Still Needs TODO5

- `randomforest`: 40/40 COMPLETE. Caveat: Validation best value is not final paper evidence.
- `xgboost`: 50/50 COMPLETE. Caveat: Validation best value is not final paper evidence.
- `lightgbm`: 50/50 COMPLETE. Caveat: Validation best value is not final paper evidence.
- `catboost`: 50/50 COMPLETE. Caveat: Validation best value is not final paper evidence.
- `mlp`: 30/30 COMPLETE. Caveat: Validation best value is not final paper evidence.
- `lstm`: 20/20 COMPLETE. Caveat: Validation best value is not final paper evidence.
- `wavenet`: 20/20 COMPLETE. Caveat: Validation best value is not final paper evidence.
- `2dcnn_v11_fusion`: 20/20 COMPLETE. Caveat: Validation best value is not final paper evidence.

## 2026 Conditional Baselines

- `tabm_or_realmlp`: TabM/RealMLP address the 2025+ expectation that tabular DL baselines should not stop at a plain MLP. Package: `available`. Action: Implement TabM/RealMLP runner for TODO5; if not feasible, document compute/task-fit exclusion. Caveat: Use only tabular/scalar/wave-derived features first; multimodal image comparison remains separate.
- `tabpfn_v2`: Nature 2025 TabPFN v2 makes foundation models a relevant small-data tabular baseline. Package: `available`. Action: Run as small-data/subsample baseline or document exclusion for full-data comparison. Caveat: Current train rows=67260, raw features excluding target=46; full-data feasibility and license must be verified.
- `autogluon_or_tuned_weighted_ensemble`: TabArena shows validation, HPO, and ensembling can materially change tabular rankings. Package: `missing`. Action: If AutoGluon install is allowed, run fixed-budget AutoGluon; otherwise build in-repo ensemble from existing tuned models. Caveat: Must report wall-clock budget; compare against individual models transparently.

## Paper-Ready Wording

Recommended:

```text
We compare against classical sanity, linear, tree, GBDT, neural-tabular, sequence,
and multimodal baselines. Recent 2025 tabular-learning baselines are handled explicitly:
TabM/RealMLP and TabPFN v2 are either included under the fixed protocol or excluded with
documented dependency, license, data-scale, or task-fit reasons; an AutoML/ensemble upper
baseline is included when feasible or approximated with in-repository tuned ensembles.
```

Avoid:

```text
The proposed model is state of the art because it beats XGBoost/LightGBM/CatBoost.
```

That wording is too weak for 2026 because modern tabular baselines and ensemble protocols must be addressed.

## References

1. Hollmann et al. (2025). Accurate predictions on small data with a tabular foundation model. Nature. https://www.nature.com/articles/s41586-024-08328-6
2. Gorishniy, Kotelnikov, & Babenko (2025). TabM: Advancing tabular deep learning with parameter-efficient ensembling. ICLR. https://openreview.net/forum?id=Sd4wYYOhmY
3. Erickson et al. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS Datasets and Benchmarks. https://openreview.net/forum?id=jZqCqpCLdU
4. Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. KDD. https://dl.acm.org/doi/10.1145/2939672.2939785
5. Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree
6. Prokhorenkova et al. (2018). CatBoost: Unbiased boosting with categorical features. NeurIPS. https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features
