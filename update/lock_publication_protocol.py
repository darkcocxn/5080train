"""Generate a locked publication protocol without starting any training.

This script implements TODO 1 from the publication checklist. It is intentionally
read-only with respect to datasets, model code, and Optuna artifacts: it only
computes lightweight metadata and writes protocol files under publication_eval_*.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
DEFAULT_DATASET_BASE = (
    "opensees_surrogate_dataset_floors_3_to_7_"
    "3stage-tailfix-steel01main-steel02damper-light-grid6-"
    "fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258"
)
DEFAULT_WAVELET_IMAGE_DIR = (
    PROJECT_ROOT
    / "newdata"
    / "Newdatabase"
    / "Scalogram"
    / "rare_waves-20260531-140353"
)
DEFAULT_FINAL_SEEDS = [20260614, 20260615, 20260616, 20260617, 20260618]
TARGET_COLUMN = "max_drift_ratio_raw"


@dataclass(frozen=True)
class HpoJob:
    name: str
    script: str
    trials: int
    startup_trials: int
    extra_args: tuple[str, ...] = ()
    notes: str = ""


HPO_JOBS = (
    HpoJob("randomforest", "update/randomforest/tune_optuna_tpe.py", 40, 8, ("--n-jobs", "8")),
    HpoJob("xgboost", "update/xgboost/tune_optuna_tpe.py", 50, 8, ("--n-jobs", "8")),
    HpoJob("lightgbm", "update/lightgbm/tune_optuna_tpe.py", 50, 8, ("--n-jobs", "8")),
    HpoJob("catboost", "update/catboost/tune_optuna_tpe.py", 50, 8, ("--n-jobs", "8")),
    HpoJob("mlp", "update/mlp/tune_optuna_tpe.py", 30, 8, ("--num-epochs", "60")),
    HpoJob(
        "lstm",
        "update/lstm/tune_optuna_tpe.py",
        20,
        8,
        ("--num-epochs", "40", "--batch-size", "128", "--num-workers", "0", "--device", "cuda"),
        "CUDA profile from update/run_publication_hpo.py.",
    ),
    HpoJob(
        "wavenet",
        "update/wavenet/tune_optuna_tpe.py",
        20,
        8,
        ("--num-epochs", "40", "--batch-size", "96", "--num-workers", "0", "--device", "cuda"),
        "CUDA profile from update/run_publication_hpo.py.",
    ),
    HpoJob(
        "2dcnn",
        "update/2dcnn/tune_optuna_tpe.py",
        20,
        8,
        ("--num-epochs", "40", "--batch-size", "32", "--num-workers", "0"),
        "Uses the 2dcnnv11 tuning entry with the unified newdata split.",
    ),
)


CODE_FILES = (
    "update/run_publication_hpo.py",
    "update/common/optuna_tpe_common.py",
    "tabular_model_common.py",
    "sequence_model_common.py",
    "2dcnnv11/2dcnnv11.py",
    "2dcnnv11/2dcnnv11test.py",
    "floors_3_to_7_utils.py",
)


MODEL_REGISTRY = (
    {
        "name": "dummy_mean_median",
        "status": "planned_p0_baseline",
        "role": "sanity baseline",
        "reason": "Detect whether the task is solved by target central tendency.",
    },
    {
        "name": "ridge_elasticnet",
        "status": "planned_p0_baseline",
        "role": "regularized linear baseline",
        "reason": "Quantify the gain from nonlinear models.",
    },
    {
        "name": "extratrees_histgradientboosting",
        "status": "planned_p0_baseline",
        "role": "classical tabular baseline",
        "reason": "Add robust, low-cost tree baselines.",
    },
    {
        "name": "randomforest",
        "status": "existing_hpo",
        "role": "classical tabular baseline",
        "script": "update/randomforest/tune_optuna_tpe.py",
    },
    {
        "name": "xgboost",
        "status": "existing_hpo",
        "role": "strong GBDT baseline",
        "script": "update/xgboost/tune_optuna_tpe.py",
    },
    {
        "name": "lightgbm",
        "status": "existing_hpo",
        "role": "strong GBDT baseline",
        "script": "update/lightgbm/tune_optuna_tpe.py",
    },
    {
        "name": "catboost",
        "status": "existing_hpo",
        "role": "strong GBDT baseline",
        "script": "update/catboost/tune_optuna_tpe.py",
    },
    {
        "name": "mlp",
        "status": "existing_hpo",
        "role": "neural tabular baseline",
        "script": "update/mlp/tune_optuna_tpe.py",
    },
    {
        "name": "realmlp_tabm",
        "status": "planned_2026_baseline",
        "role": "modern neural tabular baseline",
        "reason": "Align with 2025 tabular deep learning baselines.",
    },
    {
        "name": "tabpfn_v2",
        "status": "conditional_2026_baseline",
        "role": "tabular foundation model baseline",
        "reason": "Include if data size, feature count, task type, license, and compute allow it; otherwise document exclusion.",
    },
    {
        "name": "autogluon_or_ensembled_gbdt_mlp",
        "status": "planned_2026_baseline",
        "role": "AutoML/ensemble upper baseline",
        "reason": "Align with TabArena-style model comparison.",
    },
    {
        "name": "lstm",
        "status": "existing_hpo",
        "role": "sequence baseline",
        "script": "update/lstm/tune_optuna_tpe.py",
    },
    {
        "name": "wavenet",
        "status": "existing_hpo",
        "role": "sequence baseline",
        "script": "update/wavenet/tune_optuna_tpe.py",
    },
    {
        "name": "2dcnn_v11_fusion",
        "status": "existing_hpo_candidate_main",
        "role": "proposed multimodal/time-frequency fusion model",
        "script": "update/2dcnn/tune_optuna_tpe.py",
        "train_script": "2dcnnv11/2dcnnv11.py",
        "test_script": "2dcnnv11/2dcnnv11test.py",
    },
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int | None:
    if not path.exists() or not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def csv_columns(path: Path) -> list[str]:
    if not path.exists() or not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration:
            return []


def file_metadata(path: Path, *, include_rows: bool = False) -> dict[str, Any]:
    exists = path.exists()
    metadata: dict[str, Any] = {
        "path": str(path),
        "exists": exists,
        "type": "directory" if exists and path.is_dir() else "file",
    }
    if exists and path.is_file():
        stat = path.stat()
        metadata.update(
            {
                "bytes": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "sha256": sha256_file(path),
            }
        )
        if include_rows and path.suffix.lower() == ".csv":
            metadata["rows_excluding_header"] = count_csv_rows(path)
            metadata["columns"] = csv_columns(path)
            metadata["target_column_present"] = TARGET_COLUMN in metadata["columns"]
    elif exists and path.is_dir():
        stat = path.stat()
        metadata.update(
            {
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                "recursive_hash_skipped": True,
                "reason": "Directory hashing is intentionally skipped to avoid heavy IO during active training.",
            }
        )
    return metadata


def run_git(args: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


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
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            elif isinstance(item, list):
                lines.append(f"{prefix}-")
                lines.append(to_yaml(item, indent + 2))
            else:
                lines.append(f"{prefix}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{prefix}{yaml_scalar(value)}"


def write_text_target(path: Path, text: str, *, force: bool) -> Path:
    if not path.exists() or force:
        path.write_text(text, encoding="utf-8")
        return path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}_candidate_{stamp}{path.suffix}")
    candidate.write_text(text, encoding="utf-8")
    return candidate


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    data_dir = Path(args.data_dir).resolve()
    output_root = Path(args.output_root).resolve()
    dataset_base = args.dataset_base
    train_csv = data_dir / f"{dataset_base}_train.csv"
    val_csv = data_dir / f"{dataset_base}_val.csv"
    test_csv = data_dir / f"{dataset_base}_test.csv"
    wavelet_image_dir = Path(args.wavelet_image_dir).resolve()
    generated_at = datetime.now().isoformat(timespec="seconds")

    best_param_paths = {
        job.name: project_root / "update" / job.name / "best_params.json"
        for job in HPO_JOBS
    }

    return {
        "protocol_name": "publication_evaluation_protocol_lock",
        "protocol_status": "locked_candidate",
        "generated_at": generated_at,
        "generated_by": {
            "script": str((project_root / "update" / "lock_publication_protocol.py").resolve()),
            "project_root": str(project_root),
            "output_root": str(output_root),
            "non_interference_policy": [
                "No training entrypoints are executed.",
                "No running Python processes are stopped or modified.",
                "Dataset/image directories are read-only; directory recursive hashing is skipped.",
                "Only protocol files under the output protocol directory are written.",
            ],
        },
        "git": {
            "commit": run_git(["rev-parse", "HEAD"], project_root),
            "branch": run_git(["rev-parse", "--abbrev-ref", "HEAD"], project_root),
            "dirty_status_short": run_git(["status", "--short"], project_root),
        },
        "dataset": {
            "dataset_base": dataset_base,
            "target_column": TARGET_COLUMN,
            "data_dir": str(data_dir),
            "train_csv": file_metadata(train_csv, include_rows=not args.skip_row_counts),
            "val_csv": file_metadata(val_csv, include_rows=not args.skip_row_counts),
            "test_csv_reserved_locked": file_metadata(test_csv, include_rows=not args.skip_row_counts),
            "wavelet_image_dir": file_metadata(wavelet_image_dir),
            "filtering_rules_to_verify_in_todo2": [
                "Check whether top drift samples were removed.",
                "Check invalid analysis/status filtering rules.",
                "Check whether group overlap exists across train/val/test.",
                "Check high-drift/yield sample coverage before making safety claims.",
            ],
        },
        "splits_and_usage_policy": {
            "train": "Model fitting only.",
            "validation": "Optuna HPO, early stopping, checkpoint selection only.",
            "locked_test": "Final generalization evaluation only after models, metrics, seeds, and statistics are frozen.",
            "test_set_prohibitions": [
                "No hyperparameter selection.",
                "No early stopping.",
                "No model architecture selection.",
                "No ablation selection.",
                "No figure tuning based on test performance.",
            ],
        },
        "hpo_protocol": {
            "sampler": "Optuna TPE",
            "direction": "minimize",
            "objective_metric": "selection",
            "jobs": [
                {
                    "model": job.name,
                    "script": job.script,
                    "trials": job.trials,
                    "startup_trials": job.startup_trials,
                    "extra_args": list(job.extra_args),
                    "notes": job.notes,
                    "best_params_artifact": file_metadata(best_param_paths[job.name]),
                }
                for job in HPO_JOBS
            ],
        },
        "final_evaluation_protocol": {
            "final_training_seeds": args.final_seeds,
            "minimum_independent_training_seeds": 5,
            "preferred_independent_training_seeds": 10,
            "seed_policy": "Each final model must be retrained from scratch with frozen best hyperparameters.",
            "resampling_warning": "Test-set resampling metrics must not be reported as independent training seeds.",
            "primary_metrics": ["MAE", "RMSE", "R2"],
            "secondary_metrics": ["MAPE_or_SMAPE", "parameter_count", "training_time", "inference_latency"],
            "safety_metrics": [
                "tail_MAE",
                "tail_RMSE",
                "dangerous_underprediction_rate",
                "threshold_recall",
                "threshold_precision",
                "max_underprediction",
            ],
            "tail_thresholds": ["p90", "p95", "0.005", "0.010", "0.015", "0.020"],
            "statistical_tests": [
                "grouped paired bootstrap confidence intervals",
                "Wilcoxon signed-rank test",
                "Friedman test with post-hoc correction when multiple folds/tasks exist",
                "Holm correction for multiple comparisons",
            ],
        },
        "model_registry": list(MODEL_REGISTRY),
        "claim_boundaries": {
            "iid_interpolation": "Supported only by locked IID test metrics.",
            "unseen_wave_generalization": "Requires a test split whose earthquake waves are absent from train/val.",
            "unseen_structure_generalization": "Requires held-out structural configurations or parameter ranges.",
            "high_drift_safety": "Requires sufficient high-drift/yield samples and dangerous-underprediction metrics.",
            "uncertainty_claim": "Requires calibrated intervals or equivalent uncertainty evaluation.",
        },
        "code_artifacts": {
            "core_files": [
                file_metadata(project_root / relative_path)
                for relative_path in CODE_FILES
            ],
            "tuning_scripts": [
                file_metadata(project_root / job.script)
                for job in HPO_JOBS
            ],
        },
        "next_required_todos": [
            "TODO2: dataset manifest, leakage, filtering, and distribution audit.",
            "TODO3: HPO fairness audit.",
            "TODO4: 2026 baseline inclusion/exclusion review.",
            "TODO5: frozen-best-params multi-seed retraining.",
        ],
    }


def build_report(protocol: dict[str, Any], yaml_path: Path, json_path: Path) -> str:
    dataset = protocol["dataset"]
    git = protocol["git"]
    lines = [
        "# TODO1 协议锁定生成报告",
        "",
        f"- 生成时间: `{protocol['generated_at']}`",
        f"- YAML: `{yaml_path}`",
        f"- JSON: `{json_path}`",
        f"- Git branch: `{git.get('branch')}`",
        f"- Git commit: `{git.get('commit')}`",
        "",
        "## 数据文件",
        "",
        "| Split | Exists | Rows | SHA256 | Path |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for label, key in (("train", "train_csv"), ("val", "val_csv"), ("locked_test", "test_csv_reserved_locked")):
        item = dataset[key]
        rows = item.get("rows_excluding_header")
        sha = item.get("sha256") or ""
        lines.append(f"| {label} | {item.get('exists')} | {rows if rows is not None else '-'} | `{sha[:12]}` | `{item.get('path')}` |")
    lines.extend(
        [
            "",
            "## 不干扰训练的执行约束",
            "",
            "- 本脚本没有启动训练、测试或 Optuna 任务。",
            "- 本脚本没有终止或修改任何正在运行的 Python 进程。",
            "- 本脚本没有递归扫描小波图片目录，只记录目录存在性。",
            "- 本脚本只写入 protocol 输出目录。",
            "",
            "## 后续动作",
            "",
            "1. 人工确认 `protocol_lock.yaml` 中的模型列表、seeds、claim boundaries 是否符合论文计划。",
            "2. 冻结后进入 TODO2：数据泄漏、过滤和分布审计。",
            "3. 若之后修改协议，必须在 `protocol_change_log.md` 记录原因、日期和影响范围。",
            "",
        ]
    )
    return "\n".join(lines)


def ensure_change_log(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "\n".join(
            [
                "# Protocol Change Log",
                "",
                "Any change after protocol locking must be recorded here before running final evaluations.",
                "",
                "| Date | Change | Reason | Impact | Approved by |",
                "| --- | --- | --- | --- | --- |",
                "| 2026-06-14 | Initial TODO1 protocol lock generated. | Establish publication evaluation protocol. | No model training/evaluation executed. | pending |",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a publication protocol lock for TODO1.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Repository root.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Publication evaluation output root.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "newdata"), help="Directory containing train/val/test CSVs.")
    parser.add_argument("--dataset-base", default=DEFAULT_DATASET_BASE, help="Dataset file prefix without _train/_val/_test.csv.")
    parser.add_argument("--wavelet-image-dir", default=str(DEFAULT_WAVELET_IMAGE_DIR), help="Wavelet/scalogram image directory.")
    parser.add_argument("--final-seeds", nargs="+", type=int, default=DEFAULT_FINAL_SEEDS, help="Frozen final retraining seeds.")
    parser.add_argument("--skip-row-counts", action="store_true", help="Avoid reading CSVs for row counts/columns.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing protocol_lock files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    protocol_dir = output_root / "protocol"
    protocol_dir.mkdir(parents=True, exist_ok=True)

    protocol = build_protocol(args)
    yaml_text = to_yaml(protocol) + "\n"
    json_text = json.dumps(protocol, ensure_ascii=False, indent=2) + "\n"

    yaml_path = write_text_target(protocol_dir / "protocol_lock.yaml", yaml_text, force=args.force)
    json_path = write_text_target(protocol_dir / "protocol_lock.json", json_text, force=args.force)
    ensure_change_log(protocol_dir / "protocol_change_log.md")
    report_path = write_text_target(
        protocol_dir / "protocol_generation_report.md",
        build_report(protocol, yaml_path, json_path),
        force=args.force,
    )

    print("TODO1 protocol lock generated without running training.")
    print(f"YAML: {yaml_path}")
    print(f"JSON: {json_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
