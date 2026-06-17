# -*- coding: utf-8 -*-
"""Run TODO5 frozen-parameter final multi-seed evaluation for deep models.

This runner fills the GPU-model gap left by ``run_final_multiseed_eval.py``:
LSTM, WaveNet, and the 2D-CNN multimodal model are retrained from scratch with
the protocol seeds and frozen ``update/<model>/best_params.json`` files.  It
writes the same stable ``metrics.json`` and ``predictions_test.csv`` layout used
by TODO6-TODO8 table generation.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import os
import shutil
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from update.common.optuna_tpe_common import TorchCheckpointIOGuard  # noqa: E402


PROTOCOL_PATH = PROJECT_ROOT / "publication_eval_20260614" / "protocol" / "protocol_lock.json"
OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
FINAL_DIR = OUTPUT_ROOT / "final_multiseed"


@dataclass(frozen=True)
class DeepModelSpec:
    name: str
    module_name: str
    config_name: str
    build_model_name: str
    train_kind: str
    best_params_path: Path


DEEP_MODELS: dict[str, DeepModelSpec] = {
    "lstm": DeepModelSpec(
        name="lstm",
        module_name="lstmv1.lstmv1",
        config_name="Config",
        build_model_name="build_model",
        train_kind="sequence",
        best_params_path=PROJECT_ROOT / "update" / "lstm" / "best_params.json",
    ),
    "wavenet": DeepModelSpec(
        name="wavenet",
        module_name="wavenetv1.wavenetv1",
        config_name="Config",
        build_model_name="build_model",
        train_kind="sequence",
        best_params_path=PROJECT_ROOT / "update" / "wavenet" / "best_params.json",
    ),
    "2dcnn": DeepModelSpec(
        name="2dcnn",
        module_name="update._ablation_2dcnnv11_train_importable",
        config_name="Config",
        build_model_name="",
        train_kind="2dcnn",
        best_params_path=PROJECT_ROOT / "update" / "2dcnn" / "best_params.json",
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def temporary_env(updates: dict[str, str | None]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    dataset = protocol["dataset"]
    required = [
        dataset["train_csv"]["path"],
        dataset["val_csv"]["path"],
        dataset["test_csv_reserved_locked"]["path"],
    ]
    wavelet = dataset.get("wavelet_image_dir", {}).get("path")
    if wavelet:
        required.append(wavelet)
    missing = [item for item in required if not Path(item).exists()]
    if missing:
        raise FileNotFoundError(f"Protocol references missing files/directories: {missing}")
    return protocol


def protocol_seeds(protocol: dict[str, Any], override: list[int] | None) -> list[int]:
    if override:
        return [int(seed) for seed in override]
    seeds = protocol.get("final_evaluation_protocol", {}).get("final_training_seeds", [])
    if not seeds:
        raise ValueError("No final_training_seeds found in protocol_lock.json.")
    return [int(seed) for seed in seeds]


def import_fresh(module_name: str):
    module = importlib.import_module(module_name)
    return importlib.reload(module)


def normalize_param_value(key: str, value: Any) -> Any:
    if key == "IMAGE_SIZE" and isinstance(value, list):
        return tuple(int(item) for item in value)
    return value


def apply_frozen_params(config: type, best_params: dict[str, Any]) -> None:
    for key, value in best_params.items():
        setattr(config, key, normalize_param_value(key, value))


def copy_missing_config_attrs(target_config: type, source_config: type) -> None:
    for name in dir(source_config):
        if name.isupper() and not hasattr(target_config, name):
            setattr(target_config, name, getattr(source_config, name))


def set_common_config(
    config: type,
    spec: DeepModelSpec,
    protocol: dict[str, Any],
    run_dir: Path,
    seed: int,
    best_params: dict[str, Any],
    num_epochs: int | None,
    early_stopping_patience: int | None,
    batch_size: int | None,
    train_num_workers: int,
    device: str,
) -> None:
    import torch

    dataset = protocol["dataset"]
    apply_frozen_params(config, best_params)
    if batch_size is not None:
        config.BATCH_SIZE = int(batch_size)
    if num_epochs is not None:
        config.NUM_EPOCHS = int(num_epochs)
    if early_stopping_patience is not None and hasattr(config, "EARLY_STOPPING_PATIENCE"):
        config.EARLY_STOPPING_PATIENCE = int(early_stopping_patience)

    base_tag = getattr(config, "MODEL_TAG", spec.name)
    config.MODEL_TAG = f"{base_tag}_final_seed_{seed}"
    config.SEED = int(seed)
    config.MODEL_ROOT_DIR = run_dir
    config.MODEL_DIR = run_dir
    config.SAVE_ROOT_DIR = run_dir
    config.SAVE_DIR = run_dir
    config.UNIQUE_MODEL_RUN_DIR = False
    config.TRAIN_CSV_PATH = Path(dataset["train_csv"]["path"])
    config.VAL_CSV_PATH = Path(dataset["val_csv"]["path"])
    config.TEST_CSV_PATH = Path(dataset["test_csv_reserved_locked"]["path"])
    wavelet = dataset.get("wavelet_image_dir", {}).get("path")
    if wavelet and hasattr(config, "WAVELET_IMAGE_DIR"):
        config.WAVELET_IMAGE_DIR = Path(wavelet)
        config.FORCE_WAVELET_IMAGE_DIR = True
    config.DATA_USE_RATIO = 1.0
    config.NUM_WORKERS = int(train_num_workers)
    config.USE_AMP = True
    config.PUBLICATION_EVAL_PROTOCOL = str(PROTOCOL_PATH)
    config.OPTUNA_FROZEN_BEST_PARAMS = dict(best_params)
    if hasattr(config, "SAVE_ALTERNATE_BEST_CHECKPOINTS"):
        config.SAVE_ALTERNATE_BEST_CHECKPOINTS = False

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device != "auto":
        config.DEVICE = torch.device(device)


def compute_metrics_from_predictions(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if {"y_true", "y_pred"}.issubset(df.columns):
        y_true = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    elif {"True_Drift", "Pred_Drift"}.issubset(df.columns):
        y_true = pd.to_numeric(df["True_Drift"], errors="coerce").to_numpy(dtype=float)
        y_pred = pd.to_numeric(df["Pred_Drift"], errors="coerce").to_numpy(dtype=float)
    else:
        raise ValueError(f"Cannot find prediction columns in {path}")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    error = y_pred - y_true
    abs_error = np.abs(error)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    metrics: dict[str, Any] = {
        "Samples": int(len(y_true)),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "MAE": float(np.mean(abs_error)),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAPE": float(np.mean(abs_error / np.maximum(np.abs(y_true), 1e-12)) * 100.0),
        "SMAPE": float(np.mean(2.0 * abs_error / np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-12)) * 100.0),
        "Bias": float(np.mean(error)),
    }
    for q in (0.90, 0.95):
        threshold = float(np.quantile(y_true, q))
        tail = y_true >= threshold
        label = f"tail_p{int(q * 100)}"
        metrics[f"{label}_threshold"] = threshold
        metrics[f"{label}_count"] = int(tail.sum())
        metrics[f"{label}_mae"] = float(np.mean(abs_error[tail])) if tail.any() else None
        metrics[f"{label}_rmse"] = float(np.sqrt(np.mean(error[tail] ** 2))) if tail.any() else None
    return metrics


def flatten_metrics(model: str, seed: int, phase: str, metrics: dict[str, Any], fit_seconds: float) -> dict[str, Any]:
    mapping = {
        "Samples": "samples",
        "R2": "r2",
        "MAE": "mae",
        "RMSE": "rmse",
        "MAPE": "mape",
        "SMAPE": "smape",
        "Bias": "bias",
    }
    row: dict[str, Any] = {
        "model": model,
        "seed": int(seed),
        "phase": phase,
        "fit_seconds": float(fit_seconds),
    }
    for key, value in metrics.items():
        out_key = mapping.get(key, str(key).lower())
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[out_key] = float(value)
    return row


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("phase") == "test":
            grouped.setdefault(str(row["model"]), []).append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        keys = sorted(
            key
            for key in {item for row in model_rows for item in row}
            if key not in {"model", "seed", "phase"} and isinstance(model_rows[0].get(key), (int, float))
        )
        out: dict[str, Any] = {"model": model, "seed_count": len(model_rows)}
        for key in keys:
            values = [float(row[key]) for row in model_rows if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ci95 = float(1.96 * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_ci95_normal"] = ci95
        output.append(out)
    output.sort(key=lambda row: row.get("mae_mean", float("inf")))
    return output


def copy_phase_artifacts(result_paths: tuple[Path, ...], run_dir: Path, phase: str) -> Path:
    predictions_path = Path(result_paths[0])
    if not predictions_path.exists():
        raise FileNotFoundError(f"Evaluation did not produce predictions: {predictions_path}")
    stable_predictions = run_dir / f"predictions_{phase}.csv"
    shutil.copy2(predictions_path, stable_predictions)
    suffixes = {
        1: f"seed_metrics_{phase}.csv",
        2: f"tail_metrics_{phase}.csv",
        3: f"drift_bins_{phase}.csv",
        4: f"plot_{phase}.svg",
    }
    for index, stable_name in suffixes.items():
        if index < len(result_paths) and Path(result_paths[index]).exists():
            shutil.copy2(result_paths[index], run_dir / stable_name)
    return stable_predictions


def set_eval_csv(config: type, csv_path: Path, eval_batch_size: int, eval_num_workers: int, device: str) -> None:
    import torch

    config.TEST_CSV_PATH = Path(csv_path)
    config.BATCH_SIZE = int(eval_batch_size)
    config.NUM_WORKERS = int(eval_num_workers)
    if device != "auto":
        config.DEVICE = torch.device(device)


def evaluate_sequence_phase(
    model_module: Any,
    config: type,
    build_model: Callable[[int, int], Any],
    csv_path: Path,
    run_dir: Path,
    phase: str,
    eval_batch_size: int,
    eval_num_workers: int,
    device: str,
) -> tuple[Path, dict[str, Any]]:
    import sequence_model_common as seq_common

    set_eval_csv(config, csv_path, eval_batch_size, eval_num_workers, device)
    model_module.Config = config
    seq_common.evaluate_sequence_model(config, build_model)
    result_paths = seq_common.build_result_paths(config, csv_path)
    stable_predictions = copy_phase_artifacts(result_paths, run_dir, phase)
    return stable_predictions, compute_metrics_from_predictions(stable_predictions)


def evaluate_2dcnn_phase(
    config: type,
    csv_path: Path,
    run_dir: Path,
    phase: str,
    eval_batch_size: int,
    eval_num_workers: int,
    device: str,
) -> tuple[Path, dict[str, Any]]:
    test_module = import_fresh("update._ablation_2dcnnv11_test_importable")
    copy_missing_config_attrs(config, test_module.Config)
    test_module.Config = config
    set_eval_csv(test_module.Config, csv_path, eval_batch_size, eval_num_workers, device)
    test_module.Config.MODEL_ROOT_DIR = run_dir
    test_module.Config.MODEL_DIR = run_dir
    test_module.Config.SCALER_PATH = run_dir / "scalar_scaler.pkl"
    test_module.Config.TRAINING_METADATA_PATH = run_dir / "training_metadata.json"
    test_module.Config.MODEL_WEIGHTS_PATH = run_dir / "best_2dcnn_model.pth"
    test_module.Config.MODEL_WEIGHTS_NAME_OVERRIDE = None
    test_module.evaluate()
    result_paths = test_module.build_result_paths(csv_path)
    stable_predictions = copy_phase_artifacts(result_paths, run_dir, phase)
    return stable_predictions, compute_metrics_from_predictions(stable_predictions)


def update_training_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    metadata_path = run_dir / "training_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    metadata.update(payload)
    save_json(metadata_path, metadata)


def find_2dcnn_artifact_dir(run_dir: Path) -> Path | None:
    candidates = [run_dir]
    if run_dir.exists():
        candidates.extend(path for path in run_dir.iterdir() if path.is_dir())
    valid = [
        path
        for path in candidates
        if (path / "best_2dcnn_model.pth").exists()
        and (path / "scalar_scaler.pkl").exists()
        and (path / "training_metadata.json").exists()
    ]
    if not valid:
        return None
    return max(valid, key=lambda path: (path / "training_metadata.json").stat().st_mtime)


def read_2dcnn_fit_seconds(model_dir: Path | None) -> float:
    if model_dir is None:
        return 0.0
    metadata_path = model_dir / "training_metadata.json"
    if not metadata_path.exists():
        return 0.0
    metadata = load_json(metadata_path)
    if "fit_seconds" in metadata:
        return float(metadata["fit_seconds"])
    if "total_train_minutes" in metadata:
        return float(metadata["total_train_minutes"]) * 60.0
    return 0.0


def update_2dcnn_training_metadata(run_dir: Path, payload: dict[str, Any]) -> None:
    model_dir = find_2dcnn_artifact_dir(run_dir)
    if model_dir is None:
        raise FileNotFoundError(f"Cannot update 2D-CNN metadata; no trained artifact dir under {run_dir}")
    metadata_path = model_dir / "training_metadata.json"
    metadata = load_json(metadata_path)
    metadata.update(payload)
    save_json(metadata_path, metadata)


def is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cudnn" in text or "gpu" in text)


def candidate_batches(best_batch: int, fallbacks: list[int]) -> list[int]:
    ordered = [int(best_batch)] + [int(value) for value in fallbacks]
    result: list[int] = []
    for value in ordered:
        if value > 0 and value not in result:
            result.append(value)
    return result


def run_sequence_final(
    spec: DeepModelSpec,
    seed: int,
    run_dir: Path,
    protocol: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if (run_dir / "metrics.json").exists() and (run_dir / "predictions_test.csv").exists() and not args.overwrite:
        existing = load_json(run_dir / "metrics.json")
        fit_seconds = float(existing.get("fit_seconds", 0.0))
        return [
            flatten_metrics(spec.name, seed, "val", existing["val_metrics"], fit_seconds),
            flatten_metrics(spec.name, seed, "test", existing["test_metrics"], fit_seconds),
        ]

    best_params = load_json(spec.best_params_path)
    batches = candidate_batches(int(best_params.get("BATCH_SIZE", 64)), args.batch_fallbacks)
    last_exc: BaseException | None = None
    for batch_size in batches:
        try:
            module = import_fresh(spec.module_name)
            base_config = getattr(module, spec.config_name)

            class FinalConfig(base_config):
                pass

            set_common_config(
                FinalConfig,
                spec,
                protocol,
                run_dir,
                seed,
                best_params,
                args.num_epochs,
                args.early_stopping_patience,
                batch_size,
                args.train_num_workers,
                args.device,
            )
            module.Config = FinalConfig
            build_model = getattr(module, spec.build_model_name)
            start = time.perf_counter()
            with TorchCheckpointIOGuard() as checkpoint_guard:
                module.train_sequence_model(FinalConfig, build_model)
                checkpoint_guard.finalize(run_dir)
            fit_seconds = time.perf_counter() - start
            val_csv = Path(protocol["dataset"]["val_csv"]["path"])
            test_csv = Path(protocol["dataset"]["test_csv_reserved_locked"]["path"])
            val_path, val_metrics = evaluate_sequence_phase(
                module,
                FinalConfig,
                build_model,
                val_csv,
                run_dir,
                "val",
                args.eval_batch_size,
                args.eval_num_workers,
                args.device,
            )
            test_path, test_metrics = evaluate_sequence_phase(
                module,
                FinalConfig,
                build_model,
                test_csv,
                run_dir,
                "test",
                args.eval_batch_size,
                args.eval_num_workers,
                args.device,
            )
            update_training_metadata(
                run_dir,
                {
                    "publication_todo": "TODO5",
                    "final_eval_model": spec.name,
                    "final_eval_seed": int(seed),
                    "frozen_best_params_path": str(spec.best_params_path),
                    "frozen_best_params": best_params,
                    "final_eval_batch_size": int(batch_size),
                    "fit_seconds": fit_seconds,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "predictions_val": str(val_path),
                    "predictions_test": str(test_path),
                    "locked_test_policy": "Test split is evaluated only after frozen-parameter retraining; no HPO or model selection is performed here.",
                },
            )
            save_json(
                run_dir / "metrics.json",
                {
                    "model": spec.name,
                    "seed": int(seed),
                    "fit_seconds": fit_seconds,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "best_params": best_params,
                    "batch_size": int(batch_size),
                },
            )
            return [
                flatten_metrics(spec.name, seed, "val", val_metrics, fit_seconds),
                flatten_metrics(spec.name, seed, "test", test_metrics, fit_seconds),
            ]
        except BaseException as exc:
            last_exc = exc
            if is_cuda_oom(exc) and batch_size != batches[-1]:
                print(f">>> {spec.name} seed={seed} batch={batch_size} hit CUDA OOM; retrying smaller batch.")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
    assert last_exc is not None
    raise last_exc


def run_2dcnn_final(
    spec: DeepModelSpec,
    seed: int,
    run_dir: Path,
    protocol: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if (run_dir / "metrics.json").exists() and (run_dir / "predictions_test.csv").exists() and not args.overwrite:
        existing = load_json(run_dir / "metrics.json")
        fit_seconds = float(existing.get("fit_seconds", 0.0))
        return [
            flatten_metrics(spec.name, seed, "val", existing["val_metrics"], fit_seconds),
            flatten_metrics(spec.name, seed, "test", existing["test_metrics"], fit_seconds),
        ]

    best_params = load_json(spec.best_params_path)
    batches = candidate_batches(int(best_params.get("BATCH_SIZE", 32)), args.batch_fallbacks)
    env_updates = {
        "SURMOD_TRAIN_CSV": protocol["dataset"]["train_csv"]["path"],
        "SURMOD_VAL_CSV": protocol["dataset"]["val_csv"]["path"],
        "SURMOD_WAVELET_IMAGE_DIR": protocol["dataset"].get("wavelet_image_dir", {}).get("path"),
        "SURMOD_MODEL_ROOT_DIR": str(run_dir),
    }
    existing_model_dir = find_2dcnn_artifact_dir(run_dir)
    if existing_model_dir is not None and not args.overwrite:
        print(f">>> Found completed 2D-CNN training artifacts for seed={seed}: {existing_model_dir}")
        print(">>> Skipping retraining and regenerating TODO5 val/test evaluation artifacts.")
        with temporary_env(env_updates):
            module = import_fresh(spec.module_name)
            base_config = getattr(module, spec.config_name)

            class FinalConfig(base_config):
                pass

            batch_size = int(best_params.get("BATCH_SIZE", 32))
            set_common_config(
                FinalConfig,
                spec,
                protocol,
                run_dir,
                seed,
                best_params,
                args.num_epochs,
                args.early_stopping_patience,
                batch_size,
                args.train_num_workers,
                args.device,
            )
            fit_seconds = read_2dcnn_fit_seconds(existing_model_dir)
            val_csv = Path(protocol["dataset"]["val_csv"]["path"])
            test_csv = Path(protocol["dataset"]["test_csv_reserved_locked"]["path"])
            val_path, val_metrics = evaluate_2dcnn_phase(
                FinalConfig,
                val_csv,
                run_dir,
                "val",
                args.eval_batch_size,
                args.eval_num_workers,
                args.device,
            )
            test_path, test_metrics = evaluate_2dcnn_phase(
                FinalConfig,
                test_csv,
                run_dir,
                "test",
                args.eval_batch_size,
                args.eval_num_workers,
                args.device,
            )
            update_2dcnn_training_metadata(
                run_dir,
                {
                    "publication_todo": "TODO5",
                    "final_eval_model": spec.name,
                    "final_eval_seed": int(seed),
                    "frozen_best_params_path": str(spec.best_params_path),
                    "frozen_best_params": best_params,
                    "final_eval_batch_size": int(batch_size),
                    "fit_seconds": fit_seconds,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "predictions_val": str(val_path),
                    "predictions_test": str(test_path),
                    "locked_test_policy": "Test split is evaluated only after frozen-parameter retraining; no HPO or model selection is performed here.",
                    "resume_note": "Training artifacts already existed; TODO5 runner regenerated final evaluation without retraining.",
                },
            )
            save_json(
                run_dir / "metrics.json",
                {
                    "model": spec.name,
                    "seed": int(seed),
                    "fit_seconds": fit_seconds,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                    "best_params": best_params,
                    "batch_size": int(batch_size),
                    "resume_from_artifacts": str(existing_model_dir),
                },
            )
            return [
                flatten_metrics(spec.name, seed, "val", val_metrics, fit_seconds),
                flatten_metrics(spec.name, seed, "test", test_metrics, fit_seconds),
            ]

    last_exc: BaseException | None = None
    for batch_size in batches:
        try:
            with temporary_env(env_updates):
                module = import_fresh(spec.module_name)
                base_config = getattr(module, spec.config_name)

                class FinalConfig(base_config):
                    pass

                set_common_config(
                    FinalConfig,
                    spec,
                    protocol,
                    run_dir,
                    seed,
                    best_params,
                    args.num_epochs,
                    args.early_stopping_patience,
                    batch_size,
                    args.train_num_workers,
                    args.device,
                )
                module.Config = FinalConfig
                start = time.perf_counter()
                with TorchCheckpointIOGuard() as checkpoint_guard:
                    history = module.train()
                    checkpoint_guard.finalize(run_dir)
                module.plot_results(history)
                fit_seconds = time.perf_counter() - start

                val_csv = Path(protocol["dataset"]["val_csv"]["path"])
                test_csv = Path(protocol["dataset"]["test_csv_reserved_locked"]["path"])
                val_path, val_metrics = evaluate_2dcnn_phase(
                    FinalConfig,
                    val_csv,
                    run_dir,
                    "val",
                    args.eval_batch_size,
                    args.eval_num_workers,
                    args.device,
                )
                test_path, test_metrics = evaluate_2dcnn_phase(
                    FinalConfig,
                    test_csv,
                    run_dir,
                    "test",
                    args.eval_batch_size,
                    args.eval_num_workers,
                    args.device,
                )
                update_2dcnn_training_metadata(
                    run_dir,
                    {
                        "publication_todo": "TODO5",
                        "final_eval_model": spec.name,
                        "final_eval_seed": int(seed),
                        "frozen_best_params_path": str(spec.best_params_path),
                        "frozen_best_params": best_params,
                        "final_eval_batch_size": int(batch_size),
                        "fit_seconds": fit_seconds,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "predictions_val": str(val_path),
                        "predictions_test": str(test_path),
                        "locked_test_policy": "Test split is evaluated only after frozen-parameter retraining; no HPO or model selection is performed here.",
                    },
                )
                save_json(
                    run_dir / "metrics.json",
                    {
                        "model": spec.name,
                        "seed": int(seed),
                        "fit_seconds": fit_seconds,
                        "val_metrics": val_metrics,
                        "test_metrics": test_metrics,
                        "best_params": best_params,
                        "batch_size": int(batch_size),
                    },
                )
                return [
                    flatten_metrics(spec.name, seed, "val", val_metrics, fit_seconds),
                    flatten_metrics(spec.name, seed, "test", test_metrics, fit_seconds),
                ]
        except BaseException as exc:
            last_exc = exc
            if is_cuda_oom(exc) and batch_size != batches[-1]:
                print(f">>> {spec.name} seed={seed} batch={batch_size} hit CUDA OOM; retrying smaller batch.")
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                continue
            raise
    assert last_exc is not None
    raise last_exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TODO5 final multi-seed evaluation for LSTM/WaveNet/2D-CNN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=["lstm", "wavenet", "2dcnn"], choices=sorted(DEEP_MODELS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--train-num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-fallbacks", nargs="+", type=int, default=[96, 64, 48, 32, 24, 16])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    seeds = protocol_seeds(protocol, args.seeds)
    final_dir = args.output_root / "final_multiseed"
    final_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    for model_name in args.models:
        spec = DEEP_MODELS[model_name]
        if not spec.best_params_path.exists():
            raise FileNotFoundError(f"Missing frozen best params: {spec.best_params_path}")
        for seed in seeds:
            run_dir = final_dir / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f">>> TODO5 deep final eval: model={model_name}, seed={seed}, output={run_dir}")
            if spec.train_kind == "sequence":
                rows = run_sequence_final(spec, seed, run_dir, protocol, args)
            elif spec.train_kind == "2dcnn":
                rows = run_2dcnn_final(spec, seed, run_dir, protocol, args)
            else:
                raise ValueError(f"Unsupported train kind: {spec.train_kind}")
            metric_rows.extend(rows)
            test_row = next(row for row in rows if row.get("phase") == "test")
            run_records.append(
                {
                    "model": model_name,
                    "seed": int(seed),
                    "run_dir": str(run_dir),
                    "test_mae": test_row.get("mae"),
                    "test_rmse": test_row.get("rmse"),
                    "test_r2": test_row.get("r2"),
                }
            )
            print(
                f">>> {model_name} seed={seed}: "
                f"test_mae={float(test_row.get('mae', float('nan'))):.9f}, "
                f"test_rmse={float(test_row.get('rmse', float('nan'))):.9f}"
            )

    by_seed_path = final_dir / "summary_metrics_deep_by_seed.csv"
    write_csv(by_seed_path, sorted(metric_rows, key=lambda row: (str(row["model"]), int(row["seed"]), str(row["phase"]))))
    aggregate_path = final_dir / "summary_metrics_deep_mean_std_ci.csv"
    write_csv(aggregate_path, aggregate_metric_rows(metric_rows))
    manifest = {
        "run_name": "todo5_deep_frozen_best_params_multiseed_eval",
        "protocol_path": str(args.protocol),
        "models": list(args.models),
        "seeds": seeds,
        "num_epochs": args.num_epochs,
        "early_stopping_patience": args.early_stopping_patience,
        "train_num_workers": args.train_num_workers,
        "eval_num_workers": args.eval_num_workers,
        "eval_batch_size": args.eval_batch_size,
        "device": args.device,
        "output_dir": str(final_dir),
        "summary_metrics_deep_by_seed": str(by_seed_path),
        "summary_metrics_deep_mean_std_ci": str(aggregate_path),
        "run_records": run_records,
        "notes": [
            "No Optuna search is performed in this runner.",
            "Each run retrains from scratch using frozen best_params.json and one final protocol seed.",
            "Validation and locked-test predictions are exported with stable filenames for TODO6-TODO8.",
        ],
    }
    save_json(final_dir / "final_deep_multiseed_manifest.json", manifest)
    print(f">>> Wrote {by_seed_path}")
    print(f">>> Wrote {aggregate_path}")
    print(f">>> Wrote {final_dir / 'final_deep_multiseed_manifest.json'}")


if __name__ == "__main__":
    main()
