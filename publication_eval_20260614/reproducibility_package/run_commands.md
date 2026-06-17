# Reproduction Commands

```powershell
python update/lock_publication_protocol.py
```
```powershell
python update/audit_dataset_protocol.py
```
```powershell
python update/audit_hpo_fairness.py
```
```powershell
python update/audit_baseline_registry.py
```
```powershell
python update/run_final_multiseed_eval.py --models randomforest xgboost lightgbm catboost mlp
```
```powershell
python update/run_final_deep_multiseed_eval.py --models lstm wavenet 2dcnn
```
```powershell
python update/complete_todo6_publication_tables.py
```
```powershell
python update/complete_todo7_tail_safety.py
```
```powershell
python update/compare_models_statistics.py
```
```powershell
python update/summarize_completed_ablation_results.py
```
```powershell
python update/complete_todo10_ood_benchmark.py
```
```powershell
python update/complete_todo11_uncertainty.py
```
```powershell
python update/complete_remaining_todos.py
```
