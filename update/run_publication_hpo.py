# -*- coding: utf-8 -*-
"""Run a publication-oriented Optuna/TPE sweep for all surrogate networks.

The script is intentionally a thin orchestrator: it does not change any model
training code.  It fixes the dataset split, records the optimization budget,
captures logs, and writes a report that can be cited in the manuscript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPDATE_ROOT = PROJECT_ROOT / "update"
DATA_ROOT = PROJECT_ROOT / "newdata"
DATASET_BASE = (
    "opensees_surrogate_dataset_floors_3_to_7_"
    "3stage-tailfix-steel01main-steel02damper-light-grid6-"
    "fy5-fy500to2500-period09-m50to140-ydr1p0-20260531-181258"
)
TRAIN_CSV = DATA_ROOT / f"{DATASET_BASE}_train.csv"
VAL_CSV = DATA_ROOT / f"{DATASET_BASE}_val.csv"
TEST_CSV = DATA_ROOT / f"{DATASET_BASE}_test.csv"
WAVELET_IMAGE_DIR = DATA_ROOT / "Newdatabase" / "Scalogram" / "rare_waves-20260531-140353"


@dataclass(frozen=True)
class ModelJob:
    name: str
    script: str
    trials: int
    startup_trials: int
    extra_args: tuple[str, ...] = ()
    notes: str = ""


MODEL_JOBS: tuple[ModelJob, ...] = (
    ModelJob("randomforest", "update/randomforest/tune_optuna_tpe.py", 40, 8, ("--n-jobs", "8")),
    ModelJob("xgboost", "update/xgboost/tune_optuna_tpe.py", 50, 8, ("--n-jobs", "8")),
    ModelJob("lightgbm", "update/lightgbm/tune_optuna_tpe.py", 50, 8, ("--n-jobs", "8")),
    ModelJob("catboost", "update/catboost/tune_optuna_tpe.py", 50, 8, ("--n-jobs", "8")),
    ModelJob("mlp", "update/mlp/tune_optuna_tpe.py", 30, 8, ("--num-epochs", "60")),
    ModelJob(
        "lstm",
        "update/lstm/tune_optuna_tpe.py",
        20,
        8,
        ("--num-epochs", "40", "--batch-size", "128", "--num-workers", "0", "--device", "cuda"),
        "RTX 5080 run profile: AMP + CUDA, larger safe batch size; workers kept at 0 for Windows-safe Optuna configs.",
    ),
    ModelJob(
        "wavenet",
        "update/wavenet/tune_optuna_tpe.py",
        20,
        8,
        ("--num-epochs", "40", "--batch-size", "96", "--num-workers", "0", "--device", "cuda"),
        "RTX 5080 run profile: AMP + CUDA, default-safe WaveNet batch size; workers kept at 0 for Windows-safe Optuna configs.",
    ),
    ModelJob(
        "2dcnn",
        "update/2dcnn/tune_optuna_tpe.py",
        20,
        8,
        ("--num-epochs", "40", "--batch-size", "32", "--num-workers", "0"),
        "Uses the 2dcnnv11 tuning entry with the unified newdata split.",
    ),
)


def now_stamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.reader(file)
        next(reader, None)
        return sum(1 for _ in reader)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def snapshot_outputs(snapshot_dir: Path, model_names: tuple[str, ...]) -> None:
    for name in model_names:
        model_dir = UPDATE_ROOT / name
        for filename in ("study_summary.json", "best_params.json", "optuna_trials.csv"):
            copy_if_exists(model_dir / filename, snapshot_dir / name / filename)


def dataset_manifest() -> dict[str, Any]:
    required = (TRAIN_CSV, VAL_CSV, TEST_CSV)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing dataset files: " + ", ".join(missing))
    if not WAVELET_IMAGE_DIR.exists():
        raise FileNotFoundError(f"Missing wavelet image directory: {WAVELET_IMAGE_DIR}")
    return {
        "dataset_base": DATASET_BASE,
        "train_csv": str(TRAIN_CSV),
        "val_csv": str(VAL_CSV),
        "test_csv_reserved": str(TEST_CSV),
        "wavelet_image_dir": str(WAVELET_IMAGE_DIR),
        "train_rows": count_csv_rows(TRAIN_CSV),
        "val_rows": count_csv_rows(VAL_CSV),
        "test_rows": count_csv_rows(TEST_CSV),
        "sha256": {
            "train": sha256_file(TRAIN_CSV),
            "val": sha256_file(VAL_CSV),
            "test": sha256_file(TEST_CSV),
        },
    }


def command_for(job: ModelJob, batch_dir: Path, seed: int) -> list[str]:
    storage_path = batch_dir / "studies" / f"{job.name}.db"
    model_root = batch_dir / "artifacts" / job.name
    return [
        "uv",
        "run",
        "python",
        job.script,
        "--trials",
        str(job.trials),
        "--startup-trials",
        str(job.startup_trials),
        "--seed",
        str(seed),
        "--objective-metric",
        "selection",
        "--model-root",
        str(model_root),
        "--storage",
        f"sqlite:///{storage_path.as_posix()}",
        "--study-name",
        f"{job.name}_publication_hpo_{batch_dir.name}",
        *job.extra_args,
    ]


def run_job(job: ModelJob, batch_dir: Path, seed: int) -> dict[str, Any]:
    logs_dir = batch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"{job.name}.log"
    command = command_for(job, batch_dir, seed)
    summary_path = UPDATE_ROOT / job.name / "study_summary.json"
    before_summary_mtime = summary_path.stat().st_mtime if summary_path.exists() else None

    env = os.environ.copy()
    env.update(
        {
            "SURMOD_DATASET_DIR": str(DATA_ROOT),
            "SURMOD_TRAIN_CSV": str(TRAIN_CSV),
            "SURMOD_VAL_CSV": str(VAL_CSV),
            "SURMOD_TEST_CSV": str(TEST_CSV),
            "SURMOD_WAVELET_IMAGE_DIR": str(WAVELET_IMAGE_DIR),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
        }
    )

    started = time.time()
    status = {
        "model": job.name,
        "status": "running",
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "command": command,
        "log_path": str(log_path),
        "trials": job.trials,
        "startup_trials": job.startup_trials,
        "notes": job.notes,
    }
    write_json(batch_dir / "status" / f"{job.name}.json", status)

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("Command: " + " ".join(command) + "\n")
        log_file.write("Started: " + status["started_at"] + "\n\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    elapsed = time.time() - started
    after_summary_mtime = summary_path.stat().st_mtime if summary_path.exists() else None
    summary_is_current = (
        completed.returncode == 0
        or (
            before_summary_mtime is not None
            and after_summary_mtime is not None
            and after_summary_mtime > before_summary_mtime
        )
        or (before_summary_mtime is None and after_summary_mtime is not None)
    )
    summary = read_json(summary_path) if summary_is_current else None
    status.update(
        {
            "status": "success" if completed.returncode == 0 else "failed",
            "returncode": int(completed.returncode),
            "elapsed_seconds": round(elapsed, 3),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "study_summary": summary,
        }
    )
    write_json(batch_dir / "status" / f"{job.name}.json", status)
    snapshot_outputs(batch_dir / "after_each_model" / job.name, (job.name,))
    return status


def summarize_statuses(statuses: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for status in statuses:
        summary = status.get("study_summary") or {}
        rows.append(
            {
                "model": status["model"],
                "status": status["status"],
                "trials": status["trials"],
                "complete_trials": summary.get("complete_trial_count"),
                "best_trial": summary.get("best_trial_number"),
                "best_value": summary.get("best_value"),
                "best_metrics": summary.get("best_metrics"),
                "best_model_dir": summary.get("best_model_dir"),
                "elapsed_seconds": status.get("elapsed_seconds"),
                "log_path": status.get("log_path"),
            }
        )
    return {
        "success_count": sum(1 for item in statuses if item["status"] == "success"),
        "failed_count": sum(1 for item in statuses if item["status"] != "success"),
        "models": rows,
    }


def format_seconds(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(float(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"


def metric_text(metrics: dict[str, Any] | None) -> str:
    if not metrics:
        return "-"
    parts = []
    for key in ("best_val_mae_raw", "best_val_selection_score", "best_value", "MAE", "RMSE", "R2", "SelectionScore"):
        if key in metrics and metrics[key] is not None:
            value = metrics[key]
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.10g}")
            else:
                parts.append(f"{key}={value}")
    return "; ".join(parts) if parts else json.dumps(metrics, ensure_ascii=False)


def write_report(batch_dir: Path, manifest: dict[str, Any], statuses: list[dict[str, Any]]) -> Path:
    summary = summarize_statuses(statuses)
    report_path = batch_dir / "publication_hpo_report.md"
    lines: list[str] = []
    lines.append("# 论文级超参数优化报告")
    lines.append("")
    lines.append(f"- 批次目录: `{batch_dir}`")
    lines.append(f"- 生成时间: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append(f"- 优化器: Optuna TPE, `direction=minimize`, objective=`selection`")
    lines.append("- 测试集仅作为最终泛化评估保留，本轮 HPO 只使用训练集和验证集。")
    lines.append("")
    lines.append("## 数据协议")
    lines.append("")
    lines.append(f"- 训练集: `{manifest['train_csv']}` ({manifest['train_rows']} rows)")
    lines.append(f"- 验证集: `{manifest['val_csv']}` ({manifest['val_rows']} rows)")
    lines.append(f"- 保留测试集: `{manifest['test_csv_reserved']}` ({manifest['test_rows']} rows)")
    lines.append(f"- 2D-CNN 小波图目录: `{manifest['wavelet_image_dir']}`")
    lines.append("")
    lines.append("## 优化预算")
    lines.append("")
    lines.append("| 模型 | Trials | Startup trials | 额外约束 |")
    lines.append("| --- | ---: | ---: | --- |")
    for job in MODEL_JOBS:
        extra = " ".join(job.extra_args) if job.extra_args else "-"
        lines.append(f"| `{job.name}` | {job.trials} | {job.startup_trials} | `{extra}` |")
    lines.append("")
    lines.append("## 优化结果")
    lines.append("")
    lines.append("| 模型 | 状态 | 完成 trials | 最优 trial | 最优目标值 | 验证指标摘要 | 耗时 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: |")
    for row in summary["models"]:
        best_value = row["best_value"]
        best_value_text = f"{best_value:.10g}" if isinstance(best_value, (int, float)) else "-"
        lines.append(
            "| "
            f"`{row['model']}` | {row['status']} | {row['complete_trials'] or 0} | "
            f"{row['best_trial'] if row['best_trial'] is not None else '-'} | "
            f"{best_value_text} | {metric_text(row['best_metrics'])} | "
            f"{format_seconds(row['elapsed_seconds'])} |"
        )
    lines.append("")
    lines.append("## 论文写作说明")
    lines.append("")
    lines.append(
        "为保证主网络与对比网络的公平性，本文对所有候选模型采用同一训练/验证划分进行独立超参数优化。"
        "传统机器学习模型、MLP、序列模型和 2D-CNN 均使用相同的结构参数、地震波信息和阻尼器布置标签来源，"
        "并通过 Optuna 的 TPE 采样器在预定义搜索空间内最小化验证集 selection 指标。"
        "测试集在超参数搜索阶段保持封存，仅用于最终模型选择后的泛化性能评估。"
    )
    lines.append("")
    lines.append(
        "各模型的搜索空间、trial 数、随机种子、训练轮数上限和运行日志均随本批次保存。"
        "深度模型在 RTX 5080 环境下采用固定 `num_workers=0` 和显存安全的 batch-size 上限，以提高重复运行稳定性。"
        "最优超参数由验证集目标确定，未使用测试集反馈调整搜索空间或模型参数。"
    )
    lines.append("")
    lines.append("## 可复现文件")
    lines.append("")
    lines.append(f"- 机器可读汇总: `{batch_dir / 'publication_hpo_summary.json'}`")
    lines.append(f"- 批次 manifest: `{batch_dir / 'manifest.json'}`")
    lines.append(f"- 优化日志: `{batch_dir / 'logs'}`")
    lines.append(f"- 试验数据库: `{batch_dir / 'studies'}`")
    lines.append(f"- 模型 artifacts: `{batch_dir / 'artifacts'}`")
    lines.append(f"- 运行前快照: `{batch_dir / 'before_snapshot'}`")
    lines.append(f"- 运行后快照: `{batch_dir / 'after_snapshot'}`")
    lines.append("")
    lines.append("## 方法学参考")
    lines.append("")
    lines.append(
        "- Akiba et al. (2019), Optuna: A Next-generation Hyperparameter Optimization Framework, "
        "KDD. https://doi.org/10.1145/3292500.3330701"
    )
    lines.append(
        "- Bergstra et al. (2011), Algorithms for Hyper-Parameter Optimization, NeurIPS. "
        "https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization"
    )
    lines.append(
        "- Bergstra and Bengio (2012), Random Search for Hyper-Parameter Optimization, JMLR. "
        "https://jmlr.org/papers/v13/bergstra12a.html"
    )
    lines.append(
        "- Cawley and Talbot (2010), On over-fitting in model selection and subsequent selection bias "
        "in performance evaluation, JMLR. https://www.jmlr.org/papers/v11/cawley10a.html"
    )
    lines.append(
        "- Varma and Simon (2006), Bias in error estimation when using cross-validation for model selection, "
        "BMC Bioinformatics. https://doi.org/10.1186/1471-2105-7-91"
    )
    lines.append(
        "- Demsar (2006), Statistical Comparisons of Classifiers over Multiple Data Sets, JMLR. "
        "https://jmlr.org/papers/v7/demsar06a.html"
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_report(batch_dir: Path, manifest: dict[str, Any], statuses: list[dict[str, Any]]) -> Path:
    """Write a clean Chinese report; this definition overrides the legacy template above."""
    summary = summarize_statuses(statuses)
    report_path = batch_dir / "publication_hpo_report.md"
    lines: list[str] = []
    lines.append("# 论文级超参数优化报告")
    lines.append("")
    lines.append(f"- 批次目录: `{batch_dir}`")
    lines.append(f"- 生成时间: `{time.strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("- 优化器: Optuna TPE, `direction=minimize`, objective=`selection`")
    lines.append("- 测试集仅作为最终泛化评估保留，本轮 HPO 只使用训练集和验证集。")
    lines.append("")
    lines.append("## 数据协议")
    lines.append("")
    lines.append(f"- 训练集: `{manifest['train_csv']}` ({manifest['train_rows']} rows)")
    lines.append(f"- 验证集: `{manifest['val_csv']}` ({manifest['val_rows']} rows)")
    lines.append(f"- 保留测试集: `{manifest['test_csv_reserved']}` ({manifest['test_rows']} rows)")
    lines.append(f"- 2D-CNN 小波图目录: `{manifest['wavelet_image_dir']}`")
    lines.append("")
    lines.append("## 优化预算")
    lines.append("")
    lines.append("| 模型 | Trials | Startup trials | 额外约束 |")
    lines.append("| --- | ---: | ---: | --- |")
    for job in MODEL_JOBS:
        extra = " ".join(job.extra_args) if job.extra_args else "-"
        lines.append(f"| `{job.name}` | {job.trials} | {job.startup_trials} | `{extra}` |")
    lines.append("")
    lines.append("## 优化结果")
    lines.append("")
    lines.append("| 模型 | 状态 | 完成 trials | 最优 trial | 最优目标值 | 验证指标摘要 | 耗时 |")
    lines.append("| --- | --- | ---: | ---: | ---: | --- | ---: |")
    for row in summary["models"]:
        best_value = row["best_value"]
        best_value_text = f"{best_value:.10g}" if isinstance(best_value, (int, float)) else "-"
        lines.append(
            "| "
            f"`{row['model']}` | {row['status']} | {row['complete_trials'] or 0} | "
            f"{row['best_trial'] if row['best_trial'] is not None else '-'} | "
            f"{best_value_text} | {metric_text(row['best_metrics'])} | "
            f"{format_seconds(row['elapsed_seconds'])} |"
        )
    lines.append("")
    lines.append("## 论文写作说明")
    lines.append("")
    lines.append(
        "为保证主网络与对比网络的公平性，本文对所有候选模型采用同一训练/验证划分进行独立超参数优化。"
        "传统机器学习模型、MLP、序列模型和 2D-CNN 均使用相同的结构参数、地震波信息和阻尼器布置标签来源，"
        "并通过 Optuna 的 TPE 采样器在预定义搜索空间内最小化验证集 selection 指标。"
        "测试集在超参数搜索阶段保持封存，仅用于最终模型选择后的泛化性能评估。"
    )
    lines.append("")
    lines.append(
        "各模型的搜索空间、trial 数、随机种子、训练轮数上限和运行日志均随本批次保存。"
        "深度模型在 RTX 5080 环境下采用固定 `num_workers=0` 和显存安全的 batch-size 上限，以提高重复运行稳定性。"
        "最优超参数由验证集目标确定，未使用测试集反馈调整搜索空间或模型参数。"
    )
    lines.append("")
    lines.append("## 可复现文件")
    lines.append("")
    lines.append(f"- 机器可读汇总: `{batch_dir / 'publication_hpo_summary.json'}`")
    lines.append(f"- 批次 manifest: `{batch_dir / 'manifest.json'}`")
    lines.append(f"- 优化日志: `{batch_dir / 'logs'}`")
    lines.append(f"- 试验数据库: `{batch_dir / 'studies'}`")
    lines.append(f"- 模型 artifacts: `{batch_dir / 'artifacts'}`")
    lines.append(f"- 运行前快照: `{batch_dir / 'before_snapshot'}`")
    lines.append(f"- 运行后快照: `{batch_dir / 'after_snapshot'}`")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run publication-level HPO for all update models.")
    parser.add_argument("--batch-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="Skip models whose status file already records a successful run.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names to run, for example: lstm,wavenet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir or (UPDATE_ROOT / "run_logs" / f"publication-hpo-{now_stamp()}")
    batch_dir = batch_dir.resolve()
    batch_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("logs", "studies", "artifacts", "status"):
        (batch_dir / subdir).mkdir(parents=True, exist_ok=True)

    selected_names = None
    if args.models:
        selected_names = {name.strip().lower() for name in str(args.models).split(",") if name.strip()}
    selected_jobs = tuple(job for job in MODEL_JOBS if selected_names is None or job.name.lower() in selected_names)
    if not selected_jobs:
        raise ValueError(f"No jobs selected by --models={args.models!r}")

    manifest = {
        "batch_dir": str(batch_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": int(args.seed),
        "project_root": str(PROJECT_ROOT),
        "dataset": dataset_manifest(),
        "jobs": [job.__dict__ for job in selected_jobs],
    }
    write_json(batch_dir / "manifest.json", manifest)
    before_snapshot_dir = batch_dir / "before_snapshot"
    if not (args.resume_completed and before_snapshot_dir.exists()):
        snapshot_outputs(before_snapshot_dir, tuple(job.name for job in selected_jobs))

    statuses: list[dict[str, Any]] = []
    for job in selected_jobs:
        status_path = batch_dir / "status" / f"{job.name}.json"
        existing_status = read_json(status_path)
        if args.resume_completed and existing_status and existing_status.get("status") == "success":
            print(f">>> Skipping {job.name}; successful status already exists.")
            statuses.append(existing_status)
            write_json(batch_dir / "publication_hpo_summary.json", summarize_statuses(statuses))
            continue
        print(f">>> Running {job.name} ({job.trials} trials)")
        status = run_job(job, batch_dir, int(args.seed))
        statuses.append(status)
        write_json(batch_dir / "publication_hpo_summary.json", summarize_statuses(statuses))
        if status["status"] != "success":
            print(f">>> {job.name} failed; continuing to the next model. See {status['log_path']}")

    snapshot_outputs(batch_dir / "after_snapshot", tuple(job.name for job in selected_jobs))
    summary = summarize_statuses(statuses)
    summary.update(
        {
            "batch_dir": str(batch_dir),
            "started_at": manifest["started_at"],
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": manifest["dataset"],
        }
    )
    write_json(batch_dir / "publication_hpo_summary.json", summary)
    report_path = write_report(batch_dir, manifest["dataset"], statuses)
    print(f">>> Report: {report_path}")
    return 0 if summary["failed_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
