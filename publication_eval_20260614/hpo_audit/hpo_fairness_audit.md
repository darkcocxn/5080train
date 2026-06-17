# TODO3 HPO Fairness Audit

- Generated at: `2026-06-14T14:03:28`
- Scope: existing Optuna artifacts only; no optimization or training was started.

## Budget Summary

| Model | Trials | Complete | Startup | Elapsed | sec/trial | Best trial | Best value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| randomforest | 40 | 40 | 8 | 48.1m | 72.159 | 17 | 0.0032879971261076434 |
| xgboost | 50 | 50 | 8 | 16.1m | 19.348 | 47 | 0.002740337777118779 |
| lightgbm | 50 | 50 | 8 | 11.4m | 13.734 | 38 | 0.0028032513552227315 |
| catboost | 50 | 50 | 8 | 38.6m | 46.284 | 39 | 0.0030691088306189383 |
| mlp | 30 | 30 | 8 | 15.9m | 31.757 | 19 | 0.0024903133360809518 |
| lstm | 20 | 20 | 8 | 10.35h | 1862.454 | 15 | 0.0006291551262668155 |
| wavenet | 20 | 20 | 8 | 18.41h | 3313.746 | 10 | 0.0006060673861426866 |
| 2dcnn | 20 | 20 | 8 | 38.73h | 6971.501 | 2 | 0.0005861382081615739 |

## Best Validation Metrics

| Model | Selection | Val MAE | Val RMSE | Val R2 |
| --- | ---: | ---: | ---: | ---: |
| randomforest | 0.0032879971261076434 | 0.0006680078404581216 | 0.001229154059820691 | 0.791233826005674 |
| xgboost | 0.002740337777118779 | 0.0006053183806563986 | 0.001061724596810041 | 0.8442345085649721 |
| lightgbm | 0.0028032513552227315 | 0.0006112504098021482 | 0.0010769958297983813 | 0.8397214017475975 |
| catboost | 0.0030691088306189383 | 0.0006307487813112616 | 0.0011630631839500505 | 0.8130807157133002 |
| mlp | 0.0024903133360809518 | 0.0006019679031399322 | 0.000998776380476354 | 0.8621572213724138 |
| lstm | 0.0006291551262668155 | 0.0005239700031289665 | None | None |
| wavenet | 0.0006060673861426866 | 0.0005004144677684176 | None | None |
| 2dcnn | 0.0005861382081615739 | 0.0004825879295822233 | None | None |

## Fairness Flags

| Severity | Issue | Details |
| --- | --- | --- |
| major | unequal_trial_budget | Trial counts range from 20 to 50; report or normalize budgets before strong model-ranking claims. |
| moderate | high_startup_fraction_for_low_trial_models | Startup/total ratios >=0.35 for lstm:0.40, wavenet:0.40, 2dcnn:0.40 |
| major | large_wall_clock_imbalance | Elapsed time ratio is 203.0x across models; report compute budgets separately. |
| methodological | validation_only_hpo_results | HPO best values are validation/model-selection results; final paper claims still require frozen-parameter multi-seed locked-test evaluation. |

## Interpretation

- All current HPO best values are validation/model-selection results, not final paper evidence.
- Unequal trial counts and very different wall-clock costs must be reported transparently.
- Final claims still require TODO5 multi-seed retraining and locked-test statistical comparison.
