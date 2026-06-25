# -*- coding: utf-8 -*-
"""Run TODO9 ablation experiments for the 2D-CNN multimodal surrogate.

The runner reuses the locked publication protocol, trains each ablation with the
same frozen Optuna hyperparameters, evaluates the locked test split, and writes
paper-ready ablation tables plus paired statistical tests.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PROTOCOL_PATH = PROJECT_ROOT / "publication_eval_20260614" / "protocol" / "protocol_lock.json"
BEST_PARAMS_PATH = PROJECT_ROOT / "update" / "2dcnn" / "best_params.json"
OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
ABLATION_ROOT = OUTPUT_ROOT / "ablations"
REPORT_DIR = PROJECT_ROOT / "md_files" / "02_research_and_paper" / "literature_reviews"
TRAIN_SCRIPT = PROJECT_ROOT / "2dcnnv1" / "2dcnnv1.py"
TEST_SCRIPT = PROJECT_ROOT / "2dcnnv1" / "2dcnnv1test.py"
THRESHOLDS = (0.005, 0.010, 0.015, 0.020)


@dataclass(frozen=True)
class AblationSpec:
    name: str
    description: str
    config_overrides: dict[str, Any]
    input_patch: str = "none"


ABLATIONS: dict[str, AblationSpec] = {
    "full_model": AblationSpec(
        name="full_model",
        description="完整模型：wavelet image + structural/wave scalar features + scalar-FiLM CNN + gated bilinear fusion + tail-aware training.",
        config_overrides={},
    ),
    "scalar_only_blank_image": AblationSpec(
        name="scalar_only_blank_image",
        description="标量分支保留，所有小波图替换为空白图，用于估计 scalar + waveform statistics 的贡献。",
        config_overrides={},
        input_patch="blank_images",
    ),
    "image_only_zero_scalar": AblationSpec(
        name="image_only_zero_scalar",
        description="小波图保留，所有标量特征置零，用于估计 wavelet image 分支的独立贡献。",
        config_overrides={},
        input_patch="zero_scalars",
    ),
    "no_wave_scalar_features": AblationSpec(
        name="no_wave_scalar_features",
        description="移除 CSV 中的地震波统计特征及派生波形特征，保留结构标量、布局特征和小波图。",
        config_overrides={
            "WAVE_FEATURE_COLS": [],
            "WAVE_LOG_FEATURE_COLS": [],
            "USE_WAVE_DERIVED_FEATURES": False,
        },
    ),
    "concat_fusion": AblationSpec(
        name="concat_fusion",
        description="将 gated bilinear fusion 替换为简单 concat fusion，其他模块保持一致。",
        config_overrides={"FUSION_MODE": "concat"},
    ),
    "no_tail_loss": AblationSpec(
        name="no_tail_loss",
        description="关闭尾部加权、低估惩罚、pinball/relative under loss、尾部辅助分类和尾部校正头。",
        config_overrides={
            "USE_TARGET_WEIGHTED_LOSS": False,
            "USE_DENSITY_WEIGHTED_LOSS": False,
            "USE_TAIL_UNDERPREDICTION_LOSS": False,
            "USE_TAIL_RELATIVE_UNDER_LOSS": False,
            "USE_TAIL_PINBALL_LOSS": False,
            "USE_TAIL_CLASSIFICATION_AUX": False,
            "USE_TAIL_CORRECTION_HEAD": False,
            "USE_TAIL_CORRECTION_GATE": False,
            "USE_TAIL_PROB_GATED_CORRECTION": False,
            "USE_EXTREME_PROB_GATE_BLEND": False,
        },
    ),
    "no_weighted_sampler": AblationSpec(
        name="no_weighted_sampler",
        description="关闭 target-aware weighted sampler，保持损失函数和模型结构不变。",
        config_overrides={"USE_WEIGHTED_SAMPLER": False},
    ),
    "no_image_augmentation": AblationSpec(
        name="no_image_augmentation",
        description="关闭训练期小波图 time-frequency masking 增强。",
        config_overrides={
            "USE_TRAIN_IMAGE_AUGMENTATION": False,
            "TIME_FREQ_MASK_PROB": 0.0,
            "TIME_MASK_COUNT": 0,
            "FREQ_MASK_COUNT": 0,
        },
    ),
    "no_ema": AblationSpec(
        name="no_ema",
        description="关闭 EMA checkpoint averaging，直接用在线模型权重选择 best checkpoint。",
        config_overrides={"USE_EMA": False},
    ),
    "legacy_scalar_encoder": AblationSpec(
        name="legacy_scalar_encoder",
        description="将 residual scalar encoder 替换为 legacy MLP scalar encoder。",
        config_overrides={"SCALAR_ENCODER": "legacy_mlp"},
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


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()


def load_module(script_path: Path, module_name: str):
    if script_path == TRAIN_SCRIPT:
        module = importlib.import_module("update._ablation_2dcnnv1_train_importable")
        return importlib.reload(module)
    if script_path == TEST_SCRIPT:
        module = importlib.import_module("update._ablation_2dcnnv1_test_importable")
        return importlib.reload(module)
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_protocol() -> dict[str, Any]:
    protocol = load_json(PROTOCOL_PATH)
    required = [
        protocol["dataset"]["train_csv"]["path"],
        protocol["dataset"]["val_csv"]["path"],
        protocol["dataset"]["test_csv_reserved_locked"]["path"],
        protocol["dataset"]["wavelet_image_dir"]["path"],
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Protocol references missing files/directories: {missing}")
    return protocol


def protocol_seeds(protocol: dict[str, Any], override: list[int] | None) -> list[int]:
    if override:
        return [int(seed) for seed in override]
    seeds = protocol.get("final_evaluation_protocol", {}).get("final_training_seeds", [])
    if not seeds:
        seeds = [20260614, 20260615, 20260616]
    return [int(seed) for seed in seeds[:3]]


def ensure_blank_image(image_size: tuple[int, int] | list[int]) -> Path:
    from PIL import Image

    width, height = int(image_size[0]), int(image_size[1])
    asset_dir = ABLATION_ROOT / "assets"
    asset_dir.mkdir(parents=True, exist_ok=True)
    blank_path = asset_dir / f"blank_wavelet_{width}x{height}.png"
    if not blank_path.exists():
        image = Image.new("RGB", (width, height), (0, 0, 0))
        image.save(blank_path)
    return blank_path


def apply_input_patch(module: Any, patch_name: str, image_size: tuple[int, int] | list[int]) -> dict[str, Any]:
    if patch_name == "none":
        return {"input_patch": "none"}
    if patch_name == "zero_scalars":
        original = module.build_scalar_feature_frame

        def zeroed_scalar_feature_frame(df, layout_width, *args, **kwargs):
            feature_df, feature_names = original(df, layout_width, *args, **kwargs)
            feature_df = feature_df.copy()
            for col in feature_df.columns:
                feature_df[col] = 0.0
            return feature_df, feature_names

        module.build_scalar_feature_frame = zeroed_scalar_feature_frame
        return {"input_patch": "zero_scalars", "details": "All scalar feature values are set to 0 after feature construction."}
    if patch_name == "blank_images":
        blank_path = ensure_blank_image(image_size)

        def blank_image_paths(values):
            return [str(blank_path)] * len(values)

        module.build_image_paths = blank_image_paths
        return {"input_patch": "blank_images", "blank_image_path": str(blank_path)}
    raise ValueError(f"Unknown input patch: {patch_name}")


def set_common_config(
    config: type,
    protocol: dict[str, Any],
    run_root: Path,
    seed: int,
    best_params: dict[str, Any],
    batch_size: int,
    num_epochs: int,
    early_stopping_patience: int | None,
    num_workers: int,
    device: str,
) -> None:
    dataset = protocol["dataset"]
    for key, value in best_params.items():
        setattr(config, key, tuple(value) if key == "IMAGE_SIZE" else value)

    config.SEED = int(seed)
    config.MODEL_ROOT_DIR = run_root
    config.MODEL_DIR = run_root
    config.SAVE_ROOT_DIR = run_root
    config.SAVE_DIR = run_root
    config.UNIQUE_MODEL_RUN_DIR = False
    config.TRAIN_CSV_PATH = Path(dataset["train_csv"]["path"])
    config.VAL_CSV_PATH = Path(dataset["val_csv"]["path"])
    config.TEST_CSV_PATH = Path(dataset["test_csv_reserved_locked"]["path"])
    config.WAVELET_IMAGE_DIR = Path(dataset["wavelet_image_dir"]["path"])
    config.FORCE_WAVELET_IMAGE_DIR = True
    config.DATA_USE_RATIO = 1.0
    config.BATCH_SIZE = int(batch_size)
    config.NUM_EPOCHS = int(num_epochs)
    if early_stopping_patience is not None:
        config.EARLY_STOPPING_PATIENCE = int(early_stopping_patience)
    config.NUM_WORKERS = int(num_workers)
    config.CACHE_IMAGES = True
    config.USE_AMP = True
    config.SAVE_ALTERNATE_BEST_CHECKPOINTS = False
    config.PUBLICATION_EVAL_PROTOCOL = str(PROTOCOL_PATH)
    config.OPTUNA_FROZEN_BEST_PARAMS = dict(best_params)

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device != "auto":
        config.DEVICE = torch.device(device)


def make_train_config(
    module: Any,
    spec: AblationSpec,
    protocol: dict[str, Any],
    run_root: Path,
    seed: int,
    best_params: dict[str, Any],
    batch_size: int,
    num_epochs: int,
    early_stopping_patience: int | None,
    num_workers: int,
    device: str,
) -> type:
    class AblationConfig(module.Config):
        pass

    set_common_config(
        AblationConfig,
        protocol,
        run_root,
        seed,
        best_params,
        batch_size,
        num_epochs,
        early_stopping_patience,
        num_workers,
        device,
    )
    for key, value in spec.config_overrides.items():
        setattr(AblationConfig, key, value)
    AblationConfig.ABLATION_NAME = spec.name
    AblationConfig.ABLATION_DESCRIPTION = spec.description
    AblationConfig.ABLATION_INPUT_PATCH = spec.input_patch
    return AblationConfig


def find_completed_model_dir(run_root: Path) -> Path | None:
    required = ("best_2dcnn_model.pth", "training_metadata.json", "training_history.json")
    if all((run_root / item).exists() for item in required):
        return run_root
    candidates = sorted(
        [path for path in run_root.glob("model*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if all((path / item).exists() for item in required):
            return path
    return None


def install_resilient_torch_save(torch_module: Any, retries: int = 8, base_delay: float = 0.75) -> None:
    """Patch torch.save for this ablation process to survive transient Windows file locks."""

    original_save = getattr(torch_module, "_ablation_original_save", None)
    if original_save is None:
        original_save = torch_module.save
        setattr(torch_module, "_ablation_original_save", original_save)
    if getattr(torch_module.save, "_ablation_resilient_save", False):
        return

    def resilient_save(obj: Any, f: Any, *args: Any, **kwargs: Any) -> None:
        if not isinstance(f, (str, os.PathLike)):
            return original_save(obj, f, *args, **kwargs)

        target = Path(f)
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: BaseException | None = None

        for attempt in range(retries):
            tmp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}-{time.time_ns()}-{attempt}")
            try:
                original_save(obj, tmp_path, *args, **kwargs)
                os.replace(tmp_path, target)
                return None
            except (OSError, RuntimeError) as exc:
                last_error = exc
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                message = str(exc).lower()
                transient = (
                    "error code: 1224" in message
                    or "open file failed" in message
                    or "used by another process" in message
                    or "being used by another process" in message
                    or "access is denied" in message
                    or getattr(exc, "winerror", None) in {5, 32, 33, 1224}
                )
                if not transient:
                    raise
                time.sleep(base_delay * (attempt + 1))

        raise RuntimeError(f"torch.save failed after {retries} retries for {target}") from last_error

    setattr(resilient_save, "_ablation_resilient_save", True)
    torch_module.save = resilient_save


def result_csv_for_model_dir(model_dir: Path) -> Path | None:
    candidates = sorted(model_dir.glob("test*_results.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def append_ablation_metadata(model_dir: Path, payload: dict[str, Any]) -> None:
    metadata_path = model_dir / "training_metadata.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    metadata["ablation"] = payload
    save_json(metadata_path, metadata)


def is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "cuda error: out of memory" in text


def gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not completed.stdout.strip():
            return {"available": False, "error": completed.stderr.strip()}
        parts = [part.strip() for part in completed.stdout.strip().splitlines()[0].split(",")]
        return {
            "available": True,
            "name": parts[0],
            "memory_total_mib": float(parts[1]),
            "memory_used_mib": float(parts[2]),
            "memory_free_mib": float(parts[3]),
            "utilization_gpu_pct": float(parts[4]),
            "temperature_c": float(parts[5]),
            "power_w": float(parts[6]),
        }
    except Exception as exc:  # pragma: no cover - diagnostic only
        return {"available": False, "error": str(exc)}


def train_one_run(
    spec: AblationSpec,
    protocol: dict[str, Any],
    best_params: dict[str, Any],
    seed: int,
    batch_candidates: list[int],
    num_epochs: int,
    early_stopping_patience: int | None,
    num_workers: int,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    run_root = ABLATION_ROOT / spec.name / f"seed_{seed}"
    run_root.mkdir(parents=True, exist_ok=True)

    existing_model_dir = find_completed_model_dir(run_root)
    if resume and existing_model_dir is not None:
        return {
            "ablation": spec.name,
            "seed": int(seed),
            "status": "trained_existing",
            "run_root": str(run_root),
            "model_dir": str(existing_model_dir),
            "batch_size": None,
            "train_seconds": None,
            "gpu_before": gpu_snapshot(),
            "gpu_after": gpu_snapshot(),
        }

    last_error = None
    for batch_size in batch_candidates:
        module_name = f"surmod_2dcnnv1_train_{slug(spec.name)}_{seed}_{batch_size}_{int(time.time())}"
        start = time.time()
        gpu_before = gpu_snapshot()
        try:
            train_module = load_module(TRAIN_SCRIPT, module_name)
            train_config = make_train_config(
                train_module,
                spec,
                protocol,
                run_root,
                seed,
                best_params,
                batch_size,
                num_epochs,
                early_stopping_patience,
                num_workers,
                device,
            )
            train_module.Config = train_config
            patch_info = apply_input_patch(train_module, spec.input_patch, train_config.IMAGE_SIZE)

            import torch

            install_resilient_torch_save(torch)
            if train_config.DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(train_config.DEVICE)
                torch.set_float32_matmul_precision("high")

            history = train_module.train()
            train_module.plot_results(history)
            model_dir = Path(train_module.Config.MODEL_DIR)
            peak_memory_mib = (
                float(torch.cuda.max_memory_allocated(train_config.DEVICE) / (1024**2))
                if train_config.DEVICE.type == "cuda"
                else None
            )
            elapsed = time.time() - start
            append_ablation_metadata(
                model_dir,
                {
                    "name": spec.name,
                    "description": spec.description,
                    "config_overrides": spec.config_overrides,
                    "input_patch": patch_info,
                    "seed": int(seed),
                    "batch_size": int(batch_size),
                    "num_epochs": int(num_epochs),
                    "early_stopping_patience": early_stopping_patience,
                    "num_workers": int(num_workers),
                    "device": str(train_config.DEVICE),
                    "peak_memory_mib": peak_memory_mib,
                },
            )
            return {
                "ablation": spec.name,
                "seed": int(seed),
                "status": "trained",
                "run_root": str(run_root),
                "model_dir": str(model_dir),
                "batch_size": int(batch_size),
                "train_seconds": float(elapsed),
                "peak_memory_mib": peak_memory_mib,
                "gpu_before": gpu_before,
                "gpu_after": gpu_snapshot(),
            }
        except Exception as exc:  # noqa: BLE001 - keep long runs moving
            last_error = exc
            if is_oom(exc):
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"[OOM] {spec.name} seed={seed} batch={batch_size}; retrying with a smaller batch.")
                continue
            raise

    return {
        "ablation": spec.name,
        "seed": int(seed),
        "status": "failed_oom",
        "run_root": str(run_root),
        "model_dir": None,
        "batch_size": None,
        "train_seconds": None,
        "error": repr(last_error),
        "traceback": traceback.format_exc(),
        "gpu_after": gpu_snapshot(),
    }


def evaluate_one_run(
    spec: AblationSpec,
    protocol: dict[str, Any],
    model_dir: Path,
    eval_batch_size: int,
    num_workers: int,
    device: str,
    resume: bool,
) -> dict[str, Any]:
    existing_result = result_csv_for_model_dir(model_dir)
    if resume and existing_result is not None:
        return {
            "ablation": spec.name,
            "status": "evaluated_existing",
            "model_dir": str(model_dir),
            "predictions_path": str(existing_result),
        }

    module_name = f"surmod_2dcnnv1_test_{slug(spec.name)}_{int(time.time())}"
    test_module = load_module(TEST_SCRIPT, module_name)

    class EvalConfig(test_module.Config):
        pass

    EvalConfig.MODEL_ROOT_DIR = model_dir.parent
    EvalConfig.MODEL_DIR = model_dir
    EvalConfig.TEST_CSV_PATH = Path(protocol["dataset"]["test_csv_reserved_locked"]["path"])
    EvalConfig.WAVELET_IMAGE_DIR = Path(protocol["dataset"]["wavelet_image_dir"]["path"])
    EvalConfig.FORCE_WAVELET_IMAGE_DIR = True
    EvalConfig.BATCH_SIZE = int(eval_batch_size)
    EvalConfig.NUM_WORKERS = int(num_workers)
    EvalConfig.MODEL_WEIGHTS_NAME_OVERRIDE = None

    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if device != "auto":
        EvalConfig.DEVICE = torch.device(device)

    test_module.Config = EvalConfig
    apply_input_patch(test_module, spec.input_patch, (192, 192))
    test_module.evaluate()

    result_path = result_csv_for_model_dir(model_dir)
    if result_path is None:
        raise FileNotFoundError(f"Evaluation did not produce a test results CSV in {model_dir}")
    return {
        "ablation": spec.name,
        "status": "evaluated",
        "model_dir": str(model_dir),
        "predictions_path": str(result_path),
    }


def read_predictions(path: Path, ablation: str, seed: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    if {"True_Drift", "Pred_Drift"}.issubset(df.columns):
        y_true_col, y_pred_col = "True_Drift", "Pred_Drift"
    elif {"y_true", "y_pred"}.issubset(df.columns):
        y_true_col, y_pred_col = "y_true", "y_pred"
    else:
        raise ValueError(f"Cannot find prediction columns in {path}")
    out = pd.DataFrame(
        {
            "ablation": ablation,
            "seed": int(seed),
            "sample_id": df["sample_id"].astype(str) if "sample_id" in df.columns else np.arange(len(df)).astype(str),
            "num_floors": pd.to_numeric(df.get("num_floors", np.nan), errors="coerce"),
            "wave_cluster": pd.to_numeric(df.get("wave_cluster", np.nan), errors="coerce"),
            "steel01_yielded": pd.to_numeric(df.get("steel01_yielded", np.nan), errors="coerce"),
            "steel02_yielded": pd.to_numeric(df.get("steel02_yielded", np.nan), errors="coerce"),
            "y_true": pd.to_numeric(df[y_true_col], errors="coerce"),
            "y_pred": pd.to_numeric(df[y_pred_col], errors="coerce"),
        }
    )
    out["error"] = out["y_pred"] - out["y_true"]
    out["abs_error"] = out["error"].abs()
    return out.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)


def compute_metrics(pred: pd.DataFrame, ablation: str, seed: int, train_seconds: float | None, peak_memory_mib: float | None) -> dict[str, Any]:
    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    denom = np.maximum(np.abs(y_true), 1.0e-12)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {
        "ablation": ablation,
        "seed": int(seed),
        "samples": int(len(pred)),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "mape": float(np.mean(abs_error / denom) * 100.0),
        "smape": float(np.mean(2.0 * abs_error / np.maximum(np.abs(y_true) + np.abs(y_pred), 1.0e-12)) * 100.0),
        "bias": float(np.mean(error)),
        "p95_abs_error": float(np.quantile(abs_error, 0.95)),
        "max_underprediction": float(np.max(np.maximum(y_true - y_pred, 0.0))),
        "train_seconds": train_seconds,
        "peak_memory_mib": peak_memory_mib,
    }


def compute_tail_metrics(pred: pd.DataFrame, ablation: str, seed: int) -> dict[str, Any]:
    y_true = pred["y_true"].to_numpy(dtype=float)
    y_pred = pred["y_pred"].to_numpy(dtype=float)
    error = y_pred - y_true
    abs_error = np.abs(error)
    under = np.maximum(y_true - y_pred, 0.0)
    row: dict[str, Any] = {
        "ablation": ablation,
        "seed": int(seed),
        "samples": int(len(pred)),
        "p95_underprediction": float(np.quantile(under, 0.95)),
        "max_underprediction": float(np.max(under)),
    }
    for q in (0.90, 0.95):
        threshold = float(np.quantile(y_true, q))
        mask = y_true >= threshold
        label = f"p{int(q * 100)}"
        row[f"{label}_threshold"] = threshold
        row[f"{label}_count"] = int(mask.sum())
        row[f"{label}_mae"] = float(abs_error[mask].mean()) if mask.any() else None
        row[f"{label}_rmse"] = float(np.sqrt(np.mean(error[mask] ** 2))) if mask.any() else None
    for threshold in THRESHOLDS:
        true_mask = y_true >= threshold
        pred_mask = y_pred >= threshold
        dangerous = true_mask & (y_pred < threshold)
        label = f"thr_{threshold:.3f}"
        row[f"{label}_true_count"] = int(true_mask.sum())
        row[f"{label}_pred_count"] = int(pred_mask.sum())
        row[f"{label}_recall"] = float((true_mask & pred_mask).sum() / true_mask.sum()) if true_mask.any() else None
        row[f"{label}_precision"] = float((true_mask & pred_mask).sum() / pred_mask.sum()) if pred_mask.any() else None
        row[f"{label}_dangerous_under_count"] = int(dangerous.sum())
        row[f"{label}_dangerous_under_rate"] = float(dangerous.sum() / true_mask.sum()) if true_mask.any() else None
    return row


def mean_std_ci(values: list[float]) -> tuple[float, float, float]:
    clean = [float(value) for value in values if value is not None and not pd.isna(value)]
    if not clean:
        return float("nan"), float("nan"), float("nan")
    arr = np.asarray(clean, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, std, ci95


def aggregate(rows: list[dict[str, Any]], id_key: str = "ablation", sort_key: str = "mae_mean") -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[id_key]), []).append(row)
    output: list[dict[str, Any]] = []
    for name, group_rows in grouped.items():
        numeric_keys = sorted(
            key
            for key in {field for row in group_rows for field in row}
            if key not in {id_key, "seed"} and any(isinstance(row.get(key), (int, float)) for row in group_rows)
        )
        out: dict[str, Any] = {id_key: name, "seed_count": len({row.get("seed") for row in group_rows})}
        for key in numeric_keys:
            mean, std, ci95 = mean_std_ci([row.get(key) for row in group_rows])
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_ci95_normal"] = ci95
        output.append(out)
    output.sort(key=lambda row: row.get(sort_key, float("inf")))
    return output


def paired_bootstrap_delta(reference: pd.DataFrame, candidate: pd.DataFrame, n_bootstrap: int, seed: int) -> dict[str, Any]:
    merged = reference[["sample_id", "y_true", "y_pred"]].merge(
        candidate[["sample_id", "y_true", "y_pred"]],
        on="sample_id",
        suffixes=("_reference", "_candidate"),
    )
    if merged.empty:
        raise ValueError("No overlapping sample_id values for paired bootstrap.")
    ref_abs = np.abs(merged["y_true_reference"].to_numpy(float) - merged["y_pred_reference"].to_numpy(float))
    cand_abs = np.abs(merged["y_true_candidate"].to_numpy(float) - merged["y_pred_candidate"].to_numpy(float))
    diff = cand_abs - ref_abs
    rng = np.random.default_rng(int(seed))
    n = len(diff)
    boot = np.empty(int(n_bootstrap), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample_idx = rng.integers(0, n, size=n)
        boot[idx] = float(np.mean(diff[sample_idx]))
    return {
        "n_pairs": int(n),
        "delta_mae_ablation_minus_full": float(np.mean(diff)),
        "ci95_low": float(np.quantile(boot, 0.025)),
        "ci95_high": float(np.quantile(boot, 0.975)),
        "ablation_worse_than_full": bool(np.quantile(boot, 0.025) > 0.0),
        "ablation_better_than_full": bool(np.quantile(boot, 0.975) < 0.0),
    }


def wilcoxon_delta(reference: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    merged = reference[["sample_id", "y_true", "y_pred"]].merge(
        candidate[["sample_id", "y_true", "y_pred"]],
        on="sample_id",
        suffixes=("_reference", "_candidate"),
    )
    ref_abs = np.abs(merged["y_true_reference"].to_numpy(float) - merged["y_pred_reference"].to_numpy(float))
    cand_abs = np.abs(merged["y_true_candidate"].to_numpy(float) - merged["y_pred_candidate"].to_numpy(float))
    diff = cand_abs - ref_abs
    try:
        result = wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except ValueError:
        statistic = float("nan")
        p_value = float("nan")
    return {
        "n_pairs": int(len(diff)),
        "median_delta_abs_error": float(np.median(diff)),
        "wilcoxon_statistic": statistic,
        "p_value_two_sided": p_value,
    }


def summarize_outputs(run_rows: list[dict[str, Any]], n_bootstrap: int) -> dict[str, Path]:
    prediction_frames: dict[tuple[str, int], pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    tail_rows: list[dict[str, Any]] = []

    train_lookup = {
        (str(row["ablation"]), int(row["seed"])): row
        for row in run_rows
        if row.get("status") in {"trained", "trained_existing", "evaluated", "evaluated_existing"}
    }
    for row in run_rows:
        if not row.get("predictions_path"):
            continue
        ablation = str(row["ablation"])
        seed = int(row["seed"])
        pred = read_predictions(Path(row["predictions_path"]), ablation, seed)
        prediction_frames[(ablation, seed)] = pred
        train_row = train_lookup.get((ablation, seed), {})
        train_seconds = train_row.get("train_seconds")
        peak_memory_mib = train_row.get("peak_memory_mib")
        metric_rows.append(compute_metrics(pred, ablation, seed, train_seconds, peak_memory_mib))
        tail_rows.append(compute_tail_metrics(pred, ablation, seed))

    metric_summary = aggregate(metric_rows)
    tail_summary = aggregate(tail_rows, sort_key="p95_mae_mean")

    bootstrap_rows: list[dict[str, Any]] = []
    wilcoxon_rows: list[dict[str, Any]] = []
    for (ablation, seed), candidate in sorted(prediction_frames.items()):
        if ablation == "full_model":
            continue
        reference = prediction_frames.get(("full_model", seed))
        if reference is None:
            continue
        boot = paired_bootstrap_delta(reference, candidate, n_bootstrap=n_bootstrap, seed=seed)
        boot.update({"ablation": ablation, "seed": int(seed), "reference": "full_model"})
        bootstrap_rows.append(boot)
        wil = wilcoxon_delta(reference, candidate)
        wil.update({"ablation": ablation, "seed": int(seed), "reference": "full_model"})
        wilcoxon_rows.append(wil)

    outputs = {
        "runs": ABLATION_ROOT / "ablation_runs.csv",
        "metrics_by_seed": ABLATION_ROOT / "ablation_test_metrics_by_seed.csv",
        "metrics_summary": ABLATION_ROOT / "ablation_test_metrics_summary.csv",
        "tail_by_seed": ABLATION_ROOT / "ablation_tail_metrics_by_seed.csv",
        "tail_summary": ABLATION_ROOT / "ablation_tail_metrics_summary.csv",
        "paired_bootstrap": ABLATION_ROOT / "ablation_paired_bootstrap_delta.csv",
        "wilcoxon": ABLATION_ROOT / "ablation_wilcoxon_tests.csv",
    }
    write_csv(outputs["runs"], run_rows)
    write_csv(outputs["metrics_by_seed"], metric_rows)
    write_csv(outputs["metrics_summary"], metric_summary)
    write_csv(outputs["tail_by_seed"], tail_rows)
    write_csv(outputs["tail_summary"], tail_summary)
    write_csv(outputs["paired_bootstrap"], bootstrap_rows)
    write_csv(outputs["wilcoxon"], wilcoxon_rows)
    return outputs


def write_markdown_report(outputs: dict[str, Path], run_rows: list[dict[str, Any]], protocol: dict[str, Any]) -> Path:
    summary_path = outputs["metrics_summary"]
    tail_path = outputs["tail_summary"]
    stats_path = outputs["paired_bootstrap"]
    metrics_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    tail_df = pd.read_csv(tail_path) if tail_path.exists() else pd.DataFrame()
    stats_df = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()

    def fmt(value: Any, digits: int = 6) -> str:
        try:
            if pd.isna(value):
                return ""
            return f"{float(value):.{digits}f}"
        except Exception:
            return str(value)

    top_metrics = []
    if not metrics_df.empty:
        cols = ["ablation", "mae_mean", "rmse_mean", "r2_mean", "seed_count", "peak_memory_mib_mean"]
        for _, row in metrics_df.head(12).iterrows():
            top_metrics.append(
                f"| {row['ablation']} | {fmt(row.get('mae_mean'))} | {fmt(row.get('rmse_mean'))} | "
                f"{fmt(row.get('r2_mean'), 4)} | {int(row.get('seed_count', 0))} | {fmt(row.get('peak_memory_mib_mean'), 1)} |"
            )
    tail_lines = []
    if not tail_df.empty:
        for _, row in tail_df.head(12).iterrows():
            tail_lines.append(
                f"| {row['ablation']} | {fmt(row.get('p95_mae_mean'))} | {fmt(row.get('p95_rmse_mean'))} | "
                f"{fmt(row.get('max_underprediction_mean'))} | {fmt(row.get('thr_0.005_recall_mean'), 4)} |"
            )

    significant = []
    if not stats_df.empty:
        for ablation, group in stats_df.groupby("ablation"):
            worse = int(group.get("ablation_worse_than_full", pd.Series(dtype=bool)).fillna(False).sum())
            better = int(group.get("ablation_better_than_full", pd.Series(dtype=bool)).fillna(False).sum())
            total = int(len(group))
            delta_mean = float(group["delta_mae_ablation_minus_full"].mean()) if total else float("nan")
            significant.append(f"- `{ablation}`: mean delta={delta_mean:.6g}, worse seeds={worse}/{total}, better seeds={better}/{total}.")

    completed = [row for row in run_rows if row.get("predictions_path")]
    failed = [row for row in run_rows if str(row.get("status", "")).startswith("failed")]
    date_text = time.strftime("%Y-%m-%d %H:%M:%S")
    dataset = protocol["dataset"]
    report = f"""# TODO9：2D-CNN 多模态消融实验工作总结

生成时间：{date_text}  
协议版本：`publication_eval_20260614/protocol/protocol_lock.json`

## 1. 本次完成内容

- 补齐 TODO9 消融实验脚本：`update/run_ablation_suite.py`。
- 使用 TODO1 锁定的 train/val/test 数据划分，不重新划分数据。
- 使用 `update/2dcnn/best_params.json` 中的 Optuna 冻结超参数，并在所有消融项上统一训练预算。
- 每个完成运行均训练后立即在 locked test split 上评估。
- 输出主指标、尾部安全指标、完整模型 vs 消融模型的 paired bootstrap 与 Wilcoxon 检验。

训练集：`{dataset['train_csv']['path']}`  
验证集：`{dataset['val_csv']['path']}`  
锁定测试集：`{dataset['test_csv_reserved_locked']['path']}`  
小波图目录：`{dataset['wavelet_image_dir']['path']}`

完成评估运行数：{len(completed)}  
失败运行数：{len(failed)}

## 2. 消融矩阵

| 消融项 | 解释 |
|---|---|
"""
    for name in sorted({row["ablation"] for row in run_rows} | set(ABLATIONS)):
        spec = ABLATIONS.get(name)
        if spec:
            report += f"| `{name}` | {spec.description} |\n"

    report += """
## 3. 主指标汇总

| ablation | MAE | RMSE | R2 | seeds | peak memory MiB |
|---|---:|---:|---:|---:|---:|
"""
    report += "\n".join(top_metrics) if top_metrics else "| | | | | | |\n"

    report += """

## 4. 尾部安全指标汇总

| ablation | p95 MAE | p95 RMSE | max underprediction | recall@0.005 |
|---|---:|---:|---:|---:|
"""
    report += "\n".join(tail_lines) if tail_lines else "| | | | | |\n"

    report += """

## 5. paired 统计检验

统计定义：`delta_mae_ablation_minus_full = MAE(ablation) - MAE(full_model)`。若 delta 为正，说明移除或替换该组件后误差更大，完整模型更优。

"""
    report += "\n".join(significant) if significant else "- 当前没有足够的完整模型/消融模型同 seed 配对结果。\n"

    report += f"""

## 6. 输出文件

- `{outputs['runs']}`
- `{outputs['metrics_by_seed']}`
- `{outputs['metrics_summary']}`
- `{outputs['tail_by_seed']}`
- `{outputs['tail_summary']}`
- `{outputs['paired_bootstrap']}`
- `{outputs['wilcoxon']}`

## 7. 论文表述规范

- 消融实验只解释“在当前锁定协议和训练预算下”的组件贡献。
- 若某个组件对整体 MAE 改善不显著，但对 `p95 MAE` 或 `dangerous underprediction` 有改善，应表述为安全性贡献。
- 若 paired bootstrap CI 跨 0，应避免写成显著优于，只能写方向一致或趋势性改善。
- 当前 locked test 没有 `target >= 0.010` 和屈服样本，强非线性安全结论仍需 TODO10 外推/压力测试补证。

## 8. 参考文献与规范依据

1. Rubachev, I., Kartashev, N., Gorishniy, Y., & Babenko, A. (2025). TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks. ICLR 2025. https://openreview.net/forum?id=L14sqcrUC3
2. Gorishniy, Y., Kotelnikov, A., & Babenko, A. (2025). TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling. ICLR 2025. https://openreview.net/forum?id=Sd4wYYOhmY
3. Hollmann, N., Müller, S., Purucker, L., et al. (2025). Accurate predictions on small data with a tabular foundation model. Nature, 637, 319-326. https://www.nature.com/articles/s41586-024-08328-6
4. NeurIPS. Paper Checklist Guidelines. https://neurips.cc/public/guides/PaperChecklist
5. Pineau, J., Vincent-Lamarre, P., Sinha, K., et al. (2021). Improving Reproducibility in Machine Learning Research. JMLR, 22(164), 1-20. https://jmlr.org/papers/v22/20-303.html
6. Efron, B., & Tibshirani, R. J. (1993). An Introduction to the Bootstrap. Chapman & Hall/CRC.
"""
    report_path = REPORT_DIR / "TODO9_2DCNN多模态消融实验工作总结_20260614.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", default=[
        "full_model",
        "scalar_only_blank_image",
        "image_only_zero_scalar",
        "no_wave_scalar_features",
        "concat_fusion",
        "no_tail_loss",
        "no_weighted_sampler",
        "no_image_augmentation",
        "no_ema",
    ])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=40)
    parser.add_argument("--early-stopping-patience", type=int, default=8)
    parser.add_argument("--batch-candidates", nargs="+", type=int, default=[96, 64, 48, 32])
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol()
    best_params = load_json(BEST_PARAMS_PATH)
    seeds = protocol_seeds(protocol, args.seeds)
    selected = []
    for name in args.configs:
        if name not in ABLATIONS:
            raise ValueError(f"Unknown ablation '{name}'. Available: {', '.join(sorted(ABLATIONS))}")
        selected.append(ABLATIONS[name])

    manifest_path = ABLATION_ROOT / "ablation_manifest.json"
    previous_rows: list[dict[str, Any]] = []
    runs_csv = ABLATION_ROOT / "ablation_runs.csv"
    if args.summarize_only and runs_csv.exists():
        previous_rows = pd.read_csv(runs_csv).to_dict("records")

    if args.dry_run:
        payload = {
            "selected_ablations": [spec.name for spec in selected],
            "seeds": seeds,
            "num_epochs": args.num_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_candidates": args.batch_candidates,
            "eval_batch_size": args.eval_batch_size,
            "num_workers": args.num_workers,
            "eval_num_workers": args.eval_num_workers,
            "device": args.device,
            "gpu_snapshot": gpu_snapshot(),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    run_rows: list[dict[str, Any]] = previous_rows
    if not args.summarize_only:
        for spec in selected:
            for seed in seeds:
                print(f"\n=== Ablation {spec.name} | seed={seed} ===")
                train_row = train_one_run(
                    spec=spec,
                    protocol=protocol,
                    best_params=best_params,
                    seed=seed,
                    batch_candidates=args.batch_candidates,
                    num_epochs=args.num_epochs,
                    early_stopping_patience=args.early_stopping_patience,
                    num_workers=args.num_workers,
                    device=args.device,
                    resume=args.resume,
                )
                model_dir_value = train_row.get("model_dir")
                if not model_dir_value:
                    run_rows.append(train_row)
                    write_csv(ABLATION_ROOT / "ablation_runs.csv", run_rows)
                    continue
                eval_row = evaluate_one_run(
                    spec=spec,
                    protocol=protocol,
                    model_dir=Path(str(model_dir_value)),
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.eval_num_workers,
                    device=args.device,
                    resume=args.resume,
                )
                merged_row = {**train_row, **eval_row, "seed": int(seed), "ablation": spec.name}
                run_rows.append(merged_row)
                write_csv(ABLATION_ROOT / "ablation_runs.csv", run_rows)
                save_json(
                    manifest_path,
                    {
                        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "protocol": str(PROTOCOL_PATH),
                        "best_params": str(BEST_PARAMS_PATH),
                        "selected_ablations": [item.name for item in selected],
                        "seeds": seeds,
                        "num_epochs": args.num_epochs,
                        "early_stopping_patience": args.early_stopping_patience,
                        "batch_candidates": args.batch_candidates,
                        "eval_batch_size": args.eval_batch_size,
                        "num_workers": args.num_workers,
                        "eval_num_workers": args.eval_num_workers,
                        "device": args.device,
                        "run_count": len(run_rows),
                        "gpu_snapshot": gpu_snapshot(),
                    },
                )

    outputs = summarize_outputs(run_rows, n_bootstrap=args.bootstrap)
    report_path = write_markdown_report(outputs, run_rows, protocol)
    save_json(
        manifest_path,
        {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "protocol": str(PROTOCOL_PATH),
            "best_params": str(BEST_PARAMS_PATH),
            "ablations": {name: spec.__dict__ for name, spec in ABLATIONS.items()},
            "selected_ablations": [item.name for item in selected],
            "seeds": seeds,
            "num_epochs": args.num_epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_candidates": args.batch_candidates,
            "eval_batch_size": args.eval_batch_size,
            "num_workers": args.num_workers,
            "eval_num_workers": args.eval_num_workers,
            "device": args.device,
            "run_count": len(run_rows),
            "outputs": {key: str(value) for key, value in outputs.items()},
            "report": str(report_path),
            "gpu_snapshot": gpu_snapshot(),
        },
    )
    print(f"\nAblation outputs written under: {ABLATION_ROOT}")
    print(f"Markdown report: {report_path}")


if __name__ == "__main__":
    main()
