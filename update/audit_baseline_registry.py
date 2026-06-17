"""Build a 2026-ready baseline registry without training models.

This script implements TODO 4 from the publication checklist. It reviews the
locked dataset, completed HPO artifacts, and package availability to decide
which baselines are already included, must be added, conditionally included, or
temporarily excluded with an explicit reason.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_JSON = PROJECT_ROOT / "publication_eval_20260614" / "protocol" / "protocol_lock.json"
DEFAULT_DATASET_MANIFEST = PROJECT_ROOT / "publication_eval_20260614" / "data_audit" / "dataset_manifest.json"
DEFAULT_HPO_BUDGET = PROJECT_ROOT / "publication_eval_20260614" / "hpo_audit" / "hpo_budget_table.csv"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def to_yaml(value: Any, indent: int = 0) -> str:
    prefix = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(value)}"


def package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def hpo_models(hpo_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["model"]: row for row in hpo_rows if row.get("model")}


def dataset_context(dataset_manifest: dict[str, Any]) -> dict[str, Any]:
    splits = dataset_manifest.get("split_manifests", {})
    train = splits.get("train", {})
    val = splits.get("val", {})
    test = splits.get("test", {})
    columns = train.get("columns", [])
    target = dataset_manifest.get("target_column", "max_drift_ratio_raw")
    raw_feature_count = max(len(columns) - 1, 0) if columns else None
    return {
        "train_rows": train.get("rows"),
        "val_rows": val.get("rows"),
        "test_rows": test.get("rows"),
        "raw_column_count": len(columns) if columns else None,
        "raw_feature_count_excluding_target": raw_feature_count,
        "target_column": target,
        "task_type": "tabular/regression plus waveform/image multimodal surrogate task",
        "locked_test_high_drift_limitation": "test has no >=0.010 target samples and no steel01_yielded=1 samples per TODO2",
    }


def registry_entry(
    name: str,
    family: str,
    decision: str,
    priority: str,
    role: str,
    implementation_status: str,
    action: str,
    rationale: str,
    package_status: str = "",
    script: str = "",
    hpo_status: str = "",
    caveat: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "family": family,
        "decision": decision,
        "priority": priority,
        "role": role,
        "implementation_status": implementation_status,
        "package_status": package_status,
        "hpo_status": hpo_status,
        "script": script,
        "todo5_action": action,
        "rationale": rationale,
        "caveat": caveat,
    }


def build_registry(context: dict[str, Any], packages: dict[str, bool], hpo_by_model: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sklearn_status = "available" if packages.get("sklearn") else "missing"

    rows.extend(
        [
            registry_entry(
                "dummy_mean",
                "sanity",
                "include_must_add",
                "P0",
                "Mean target predictor",
                "not_yet_implemented_for_publication_eval",
                "Add to TODO5 final evaluator; no Optuna needed.",
                "Required sanity baseline to show the task is not solved by central tendency.",
                sklearn_status,
                caveat="Report MAE/RMSE only; no HPO budget.",
            ),
            registry_entry(
                "dummy_median",
                "sanity",
                "include_must_add",
                "P0",
                "Median target predictor",
                "not_yet_implemented_for_publication_eval",
                "Add to TODO5 final evaluator; no Optuna needed.",
                "Robust central-tendency sanity baseline.",
                sklearn_status,
                caveat="Report MAE/RMSE only; no HPO budget.",
            ),
            registry_entry(
                "ridge",
                "linear_regularized",
                "include_must_add",
                "P0",
                "Regularized linear regression",
                "not_yet_implemented_for_publication_eval",
                "Add RidgeCV or Optuna-tuned Ridge to TODO5.",
                "Required to quantify nonlinearity benefit.",
                sklearn_status,
            ),
            registry_entry(
                "elasticnet",
                "linear_regularized",
                "include_must_add",
                "P0",
                "Sparse/regularized linear regression",
                "not_yet_implemented_for_publication_eval",
                "Add ElasticNetCV or Optuna-tuned ElasticNet to TODO5.",
                "Required to quantify feature sparsity and linear baseline strength.",
                sklearn_status,
            ),
            registry_entry(
                "extratrees",
                "classical_tree",
                "include_must_add",
                "P0",
                "Low-cost strong tree ensemble",
                "not_yet_implemented_for_publication_eval",
                "Add Optuna or fixed strong ExtraTrees baseline to TODO5.",
                "Complements RandomForest and is a strong, robust tabular baseline.",
                sklearn_status,
            ),
            registry_entry(
                "histgradientboosting",
                "classical_gbdt",
                "include_must_add",
                "P0",
                "Scikit-learn native gradient boosting",
                "not_yet_implemented_for_publication_eval",
                "Add HistGradientBoostingRegressor baseline to TODO5.",
                "Useful reproducible GBDT baseline independent of external GBDT libraries.",
                sklearn_status,
            ),
        ]
    )

    existing = [
        ("randomforest", "classical_tree", "RandomForest Optuna baseline"),
        ("xgboost", "gbdt", "XGBoost Optuna baseline"),
        ("lightgbm", "gbdt", "LightGBM Optuna baseline"),
        ("catboost", "gbdt", "CatBoost Optuna baseline"),
        ("mlp", "neural_tabular", "Current MLP Optuna baseline"),
        ("lstm", "sequence", "Waveform sequence baseline"),
        ("wavenet", "sequence", "WaveNet waveform baseline"),
        ("2dcnn_v11_fusion", "multimodal_time_frequency", "Proposed 2D CNN fusion candidate"),
    ]
    hpo_name_map = {"2dcnn_v11_fusion": "2dcnn"}
    for name, family, role in existing:
        hpo_name = hpo_name_map.get(name, name)
        hpo_row = hpo_by_model.get(hpo_name, {})
        rows.append(
            registry_entry(
                name,
                family,
                "already_included_hpo_complete",
                "P0",
                role,
                "implemented",
                "Use frozen best_params in TODO5 multi-seed final evaluation.",
                "Existing Optuna HPO completed; still requires locked-test multi-seed evaluation.",
                "available",
                script=f"update/{hpo_name}/tune_optuna_tpe.py",
                hpo_status=f"{hpo_row.get('complete_trials', '?')}/{hpo_row.get('actual_trials', '?')} COMPLETE",
                caveat="Validation best value is not final paper evidence.",
            )
        )

    tabm_available = packages.get("tabm") or packages.get("rtdl") or packages.get("rtdl_num_embeddings")
    rows.append(
        registry_entry(
            "tabm_or_realmlp",
            "modern_neural_tabular",
            "include_conditionally_available_not_integrated" if tabm_available else "include_conditionally_requires_dependency",
            "P1_for_top_tier",
            "Modern tabular DL strong baseline",
            "dependency_missing" if not tabm_available else "package_available_not_integrated",
            "Implement TabM/RealMLP runner for TODO5; if not feasible, document compute/task-fit exclusion." if tabm_available else "Install/implement TabM or RealMLP; if not feasible, document dependency/compute exclusion.",
            "TabM/RealMLP address the 2025+ expectation that tabular DL baselines should not stop at a plain MLP.",
            "available" if tabm_available else "missing",
            caveat="Use only tabular/scalar/wave-derived features first; multimodal image comparison remains separate.",
        )
    )

    tabpfn_available = packages.get("tabpfn") or packages.get("tabpfn_extensions")
    train_rows = int(context.get("train_rows") or 0)
    feature_count = int(context.get("raw_feature_count_excluding_target") or 0)
    if train_rows <= 10000 and feature_count <= 500:
        tabpfn_decision = "include_conditionally_full_data_if_license_allows"
        tabpfn_action = "Verify TabPFN v2 and run on full locked protocol if license and task settings allow." if tabpfn_available else "Install/verify TabPFN v2 and run on full locked protocol if license and task settings allow."
    else:
        tabpfn_decision = "include_conditionally_subsample_available" if tabpfn_available else "include_conditionally_subsample_or_exclusion_note"
        tabpfn_action = "Run as small-data/subsample baseline or document exclusion for full-data comparison."
    rows.append(
        registry_entry(
            "tabpfn_v2",
            "tabular_foundation_model",
            tabpfn_decision,
            "P1_for_top_tier",
            "Foundation model tabular baseline",
            "dependency_missing" if not tabpfn_available else "package_available_not_integrated",
            tabpfn_action,
            "Nature 2025 TabPFN v2 makes foundation models a relevant small-data tabular baseline.",
            "available" if tabpfn_available else "missing",
            caveat=f"Current train rows={train_rows}, raw features excluding target={feature_count}; full-data feasibility and license must be verified.",
        )
    )

    autogluon_available = packages.get("autogluon")
    rows.append(
        registry_entry(
            "autogluon_or_tuned_weighted_ensemble",
            "automl_ensemble",
            "include_conditionally_or_replace_with_in_repo_ensemble",
            "P1_for_top_tier",
            "AutoML / cross-model ensemble upper baseline",
            "dependency_missing" if not autogluon_available else "package_available_not_integrated",
            "If AutoGluon install is allowed, run fixed-budget AutoGluon; otherwise build in-repo ensemble from existing tuned models.",
            "TabArena shows validation, HPO, and ensembling can materially change tabular rankings.",
            "available" if autogluon_available else "missing",
            caveat="Must report wall-clock budget; compare against individual models transparently.",
        )
    )
    return rows


def build_markdown(registry: list[dict[str, Any]], context: dict[str, Any], packages: dict[str, bool]) -> str:
    by_decision: dict[str, list[dict[str, Any]]] = {}
    for row in registry:
        by_decision.setdefault(row["decision"], []).append(row)
    lines = [
        "# TODO4 Baseline Inclusion/Exclusion Review",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Scope: registry/audit only; no baseline training was run.",
        "",
        "## Dataset Context",
        "",
        f"- Train rows: `{context.get('train_rows')}`",
        f"- Validation rows: `{context.get('val_rows')}`",
        f"- Locked test rows: `{context.get('test_rows')}`",
        f"- Raw columns: `{context.get('raw_column_count')}`",
        f"- Raw feature count excluding target: `{context.get('raw_feature_count_excluding_target')}`",
        f"- Task type: `{context.get('task_type')}`",
        f"- TODO2 limitation: {context.get('locked_test_high_drift_limitation')}",
        "",
        "## Package Availability",
        "",
        "| Package | Available in current `uv run python` env |",
        "| --- | --- |",
    ]
    for package, available in packages.items():
        lines.append(f"| `{package}` | `{available}` |")
    lines.extend(
        [
            "",
            "## Registry Summary",
            "",
            "| Baseline | Decision | Priority | Package | Action |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in registry:
        lines.append(
            f"| `{row['name']}` | {row['decision']} | {row['priority']} | {row['package_status']} | {row['todo5_action']} |"
        )
    lines.extend(
        [
            "",
            "## Must Add Before Strong Publication Claims",
            "",
        ]
    )
    for row in by_decision.get("include_must_add", []):
        lines.append(f"- `{row['name']}`: {row['rationale']} Action: {row['todo5_action']}")
    lines.extend(
        [
            "",
            "## Already Included, But Still Needs TODO5",
            "",
        ]
    )
    for row in by_decision.get("already_included_hpo_complete", []):
        lines.append(f"- `{row['name']}`: {row['hpo_status']}. Caveat: {row['caveat']}")
    lines.extend(
        [
            "",
            "## 2026 Conditional Baselines",
            "",
        ]
    )
    for decision in (
        "include_conditionally_requires_dependency",
        "include_conditionally_available_not_integrated",
        "include_conditionally_subsample_or_exclusion_note",
        "include_conditionally_subsample_available",
        "include_conditionally_full_data_if_license_allows",
        "include_conditionally_or_replace_with_in_repo_ensemble",
    ):
        for row in by_decision.get(decision, []):
            lines.append(f"- `{row['name']}`: {row['rationale']} Package: `{row['package_status']}`. Action: {row['todo5_action']} Caveat: {row['caveat']}")
    lines.extend(
        [
            "",
            "## Paper-Ready Wording",
            "",
            "Recommended:",
            "",
            "```text",
            "We compare against classical sanity, linear, tree, GBDT, neural-tabular, sequence,",
            "and multimodal baselines. Recent 2025 tabular-learning baselines are handled explicitly:",
            "TabM/RealMLP and TabPFN v2 are either included under the fixed protocol or excluded with",
            "documented dependency, license, data-scale, or task-fit reasons; an AutoML/ensemble upper",
            "baseline is included when feasible or approximated with in-repository tuned ensembles.",
            "```",
            "",
            "Avoid:",
            "",
            "```text",
            "The proposed model is state of the art because it beats XGBoost/LightGBM/CatBoost.",
            "```",
            "",
            "That wording is too weak for 2026 because modern tabular baselines and ensemble protocols must be addressed.",
            "",
            "## References",
            "",
            "1. Hollmann et al. (2025). Accurate predictions on small data with a tabular foundation model. Nature. https://www.nature.com/articles/s41586-024-08328-6",
            "2. Gorishniy, Kotelnikov, & Babenko (2025). TabM: Advancing tabular deep learning with parameter-efficient ensembling. ICLR. https://openreview.net/forum?id=Sd4wYYOhmY",
            "3. Erickson et al. (2025). TabArena: A living benchmark for machine learning on tabular data. NeurIPS Datasets and Benchmarks. https://openreview.net/forum?id=jZqCqpCLdU",
            "4. Chen & Guestrin (2016). XGBoost: A scalable tree boosting system. KDD. https://dl.acm.org/doi/10.1145/2939672.2939785",
            "5. Ke et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS. https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree",
            "6. Prokhorenkova et al. (2018). CatBoost: Unbiased boosting with categorical features. NeurIPS. https://papers.neurips.cc/paper/7898-catboost-unbiased-boosting-with-categorical-features",
            "",
        ]
    )
    return "\n".join(lines)


def build_registry_bundle(protocol_path: Path, dataset_manifest_path: Path, hpo_budget_path: Path, output_root: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    dataset_manifest = load_json(dataset_manifest_path)
    hpo_rows = read_csv_rows(hpo_budget_path)
    packages = {
        "sklearn": package_available("sklearn"),
        "xgboost": package_available("xgboost"),
        "lightgbm": package_available("lightgbm"),
        "catboost": package_available("catboost"),
        "optuna": package_available("optuna"),
        "autogluon": package_available("autogluon"),
        "tabpfn": package_available("tabpfn"),
        "tabpfn_extensions": package_available("tabpfn_extensions"),
        "tabm": package_available("tabm"),
        "rtdl": package_available("rtdl"),
        "rtdl_num_embeddings": package_available("rtdl_num_embeddings"),
    }
    context = dataset_context(dataset_manifest)
    registry = build_registry(context, packages, hpo_models(hpo_rows))
    bundle = {
        "audit_name": "publication_baseline_registry_todo4",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_json": str(protocol_path),
        "dataset_manifest_json": str(dataset_manifest_path),
        "hpo_budget_table": str(hpo_budget_path),
        "non_interference_policy": [
            "No model training, testing, Optuna, or package installation was executed.",
            "Only existing audit artifacts and package availability metadata are read.",
            "Only publication_eval final_multiseed and paper_tables outputs are written.",
        ],
        "dataset_context": context,
        "package_availability": packages,
        "registry": registry,
        "protocol_status": protocol.get("protocol_status"),
    }
    final_dir = output_root / "final_multiseed"
    table_dir = output_root / "paper_tables"
    final_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    (final_dir / "model_registry.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (final_dir / "model_registry.yaml").write_text(to_yaml(bundle) + "\n", encoding="utf-8")
    (final_dir / "baseline_inclusion_exclusion.md").write_text(build_markdown(registry, context, packages), encoding="utf-8")
    write_csv(
        final_dir / "baseline_registry.csv",
        registry,
        [
            "name",
            "family",
            "decision",
            "priority",
            "role",
            "implementation_status",
            "package_status",
            "hpo_status",
            "script",
            "todo5_action",
            "rationale",
            "caveat",
        ],
    )
    write_csv(
        table_dir / "tableS_baseline_registry.csv",
        registry,
        ["name", "family", "decision", "priority", "role", "todo5_action", "rationale", "caveat"],
    )
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the TODO4 baseline inclusion/exclusion registry.")
    parser.add_argument("--protocol-json", default=str(DEFAULT_PROTOCOL_JSON))
    parser.add_argument("--dataset-manifest", default=str(DEFAULT_DATASET_MANIFEST))
    parser.add_argument("--hpo-budget", default=str(DEFAULT_HPO_BUDGET))
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol_json).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else protocol_path.parents[1]
    bundle = build_registry_bundle(
        protocol_path,
        Path(args.dataset_manifest).resolve(),
        Path(args.hpo_budget).resolve(),
        output_root,
    )
    print("TODO4 baseline registry generated without training or package installation.")
    print(f"Output root: {output_root}")
    decisions: dict[str, int] = {}
    for row in bundle["registry"]:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    for decision, count in sorted(decisions.items()):
        print(f"{decision}: {count}")


if __name__ == "__main__":
    main()
