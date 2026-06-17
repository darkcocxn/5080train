# TODO13 Runtime And Complexity Summary

Controlled inference timing and OpenSees runtime were not available in current artifacts and were not re-run to avoid disturbing training tasks. This TODO therefore completes the auditable parts: parameter/file-size audit, final fit time, HPO time, and accuracy-cost Pareto front, while explicitly blocking speedup claims.

| model | test_mae_mean | fit_seconds_mean | hpo_elapsed_seconds | model_file_size_mb_mean | torch_parameter_count_mean | pareto_fit_time_mae |
| --- | --- | --- | --- | --- | --- | --- |
| 2dcnn | 0.000314189 | 15479.8 | 139430 | 37.7914 | 3.29059e+06 | 1 |
| wavenet | 0.000342498 | 2151.6 | 66274.9 | 39.8694 | 2.07121e+06 | 1 |
| lstm | 0.000367785 | 1125.22 | 37249.1 | 43.9255 | 2.29454e+06 | 1 |
| lightgbm | 0.000460017 | 7.70253 | 686.705 | 7.012 | NA | 1 |
| randomforest | 0.000479245 | 76.9926 | 2886.37 | 931.827 | NA | 0 |
| catboost | 0.000485655 | 29.8095 | 2314.21 | 1.90477 | NA | 0 |
| xgboost | 0.000490517 | 13.7081 | 967.422 | 8.59579 | NA | 0 |
| extratrees | 0.000538646 | 50.947 | NA | NA | NA | 0 |
| mlp | 0.000632584 | 22.7409 | 952.706 | 12.9266 | 3.38535e+06 | 0 |
| ridge | 0.000673302 | 0.281212 | NA | NA | NA | 1 |
| histgradientboosting | 0.000767541 | 1.48033 | NA | NA | NA | 0 |
| elasticnet | 0.000769079 | 12.5351 | NA | NA | NA | 0 |
| dummy_median | 0.0012104 | 0.169358 | NA | NA | NA | 1 |
| dummy_mean | 0.00140429 | 0.182074 | NA | NA | NA | 0 |
