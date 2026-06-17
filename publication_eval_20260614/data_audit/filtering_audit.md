# TODO2 Filtering and Distribution Audit

- Generated at: `2026-06-14T12:28:28`
- Dataset base: `opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258`
- Target column: `max_drift_ratio_raw`

## Filename Clues

- contains_drop: `False`
- contains_rank: `False`
- contains_top: `False`
- contains_tailfix: `True`

Interpretation: filename clues are not proof of filtering. They only flag what must be verified against the generation script and source manifest.

## Split Target/Tail Summary

| Split | Rows | p90 | p95 | p99 | max | >=0.005 | >=0.010 | >=0.015 | >=0.020 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 67260 | 0.0055119008 | 0.0064711423 | 0.0077800043 | 0.008242616 | 9268 | 0 | 0 | 0 |
| val | 3420 | 0.0056164295 | 0.0095486935 | 0.012896081 | 0.014465286 | 422 | 152 | 0 | 0 |
| test | 3116 | 0.0037563493 | 0.0049114533 | 0.0067342661 | 0.0073621561 | 143 | 0 | 0 | 0 |

## Group Overlap Summary

| Group | Train unique | Val unique | Test unique | Train-Test overlap | All-three overlap | Risk |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| sample_id | 66755 | 3280 | 3112 | 0 | 0 | low |
| txt_path | 163 | 17 | 14 | 0 | 0 | low |
| image_path | 0 | 0 | 0 | 0 | 0 | low |
| wave_cluster | 8 | 5 | 4 | 4 | 3 | review |
| structure_signature | 16331 | 881 | 802 | 0 | 0 | low |
| scenario_signature | 66755 | 3280 | 3112 | 0 | 0 | low |

## Duplicate Key Summary

| Key | Split | Unique keys | Duplicate key count | Duplicate extra rows |
| --- | --- | ---: | ---: | ---: |
| sample_id | train | 66755 | 505 | 505 |
| sample_id | val | 3280 | 81 | 140 |
| sample_id | test | 3112 | 2 | 4 |
| scenario_signature | train | 66755 | 505 | 505 |
| scenario_signature | val | 3280 | 81 | 140 |
| scenario_signature | test | 3112 | 2 | 4 |

## Key Missingness Summary

| Split | Column | Missing | Rows | Fraction |
| --- | --- | ---: | ---: | ---: |
| train | image_path | 67260 | 67260 | 1 |
| train | yield_margin | 2589 | 67260 | 0.038492417 |
| train | failure_reason | 67260 | 67260 | 1 |
| train | analysis_failed_time | 67260 | 67260 | 1 |
| val | image_path | 3420 | 3420 | 1 |
| val | yield_margin | 60 | 3420 | 0.01754386 |
| val | failure_reason | 3420 | 3420 | 1 |
| val | analysis_failed_time | 3420 | 3420 | 1 |
| test | image_path | 3116 | 3116 | 1 |
| test | yield_margin | 28 | 3116 | 0.0089858793 |
| test | failure_reason | 3116 | 3116 | 1 |
| test | analysis_failed_time | 3116 | 3116 | 1 |

## Required Human Review Before Safety Claims

- Verify whether any top-drift or failure samples were removed before this split.
- Verify whether `analysis_status != ok` samples were removed upstream.
- Verify whether high-drift and yielded samples are sufficient for claims about engineering safety.
- If train/val/test have strongly different tail coverage, do not make broad tail-reliability claims until a high-drift stress benchmark is built.

## Next TODO

Use this report to decide the exact grouping variables for TODO10 unseen-wave / unseen-structure / high-drift benchmarks.
