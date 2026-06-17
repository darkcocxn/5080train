# Model Card

Primary model: `2dcnn` frozen-best-params multi-seed final evaluation.

## Main Performance

| rank_by_test_mae | model | seed_count | mae_mean | rmse_mean | r2_mean | fit_seconds_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2dcnn | 5 | 0.000314189 | 0.00044736 | 0.891259 | 15479.8 |
| 2 | wavenet | 5 | 0.000342498 | 0.000479019 | 0.875335 | 2151.6 |
| 3 | lstm | 5 | 0.000367785 | 0.000521445 | 0.851481 | 1125.22 |
| 4 | lightgbm | 5 | 0.000460017 | 0.000662132 | 0.761865 | 7.70253 |
| 5 | randomforest | 5 | 0.000479245 | 0.000679451 | 0.749405 | 76.9926 |
| 6 | catboost | 5 | 0.000485655 | 0.000730014 | 0.710648 | 29.8095 |
| 7 | xgboost | 5 | 0.000490517 | 0.000689166 | 0.742033 | 13.7081 |
| 8 | extratrees | 5 | 0.000538646 | 0.000780771 | 0.669096 | 50.947 |

## Intended Use

Fast surrogate screening of OpenSees-style seismic drift responses under the locked data distribution. Not validated for independent high-drift/yielded stress regimes until TODO10 generated OpenSees labels are evaluated.
