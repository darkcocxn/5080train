# -*- coding: utf-8 -*-
"""Run TODO5 frozen-parameter multi-seed evaluation for tuned tabular models.

This script is intentionally separate from Optuna. It reads frozen
``update/<model>/best_params.json`` files, retrains each requested model from
scratch for the protocol seeds, evaluates the same locked test split, and writes
per-seed predictions that can be consumed by TODO6-TODO8 paper tables and
paired statistical tests.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tabular_model_common import (  # noqa: E402
    build_result_paths,
    build_test_features,
    calculate_selection_score,
    evaluate_torch_mlp_tabular_model,
    inverse_target,
    maybe_fit_with_sample_weight,
    prepare_tabular_training_data,
    read_valid_csv,
    train_torch_mlp_tabular_model,
    write_evaluation_outputs,
)
from update.common.optuna_tpe_common import TorchCheckpointIOGuard  # noqa: E402


PROTOCOL_PATH = PROJECT_ROOT / "publication_eval_20260614" / "protocol" / "protocol_lock.json"
OUTPUT_ROOT = PROJECT_ROOT / "publication_eval_20260614"
DEFAULT_MODELS = ("randomforest", "xgboost", "lightgbm", "catboost")
SUPPORTED_MODELS = {
    "randomforest": ("sklearn", "randomforestv1.randomforestv1", "Config", "build_model"),
    "xgboost": ("sklearn", "xgboostv1.xgboostv1", "Config", "build_model"),
    "lightgbm": ("sklearn", "lightgbmv1.lightgbmv1", "Config", "build_model"),
    "catboost": ("sklearn", "catboostv1.catboostv1", "Config", "build_model"),
    "mlp": ("torch_mlp", "mlpv1.mlpv1", "Config", None),
}
EXCLUDED_FROM_AGGREGATE = {"model", "seed", "phase"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    config_cls: type
    factory: Callable[[Any], Any] | None
    best_params_path: Path


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


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    required = [
        protocol["dataset"]["train_csv"]["path"],
        protocol["dataset"]["val_csv"]["path"],
        protocol["dataset"]["test_csv_reserved_locked"]["path"],
    ]
    missing = [item for item in required if not Path(item).exists()]
    if missing:
        raise FileNotFoundError(f"Protocol references missing CSV files: {missing}")
    return protocol


def protocol_seeds(protocol: dict[str, Any], override: list[int] | None = None) -> list[int]:
    if override:
        return [int(seed) for seed in override]
    seeds = protocol.get("final_evaluation_protocol", {}).get("final_training_seeds", [])
    if not seeds:
        seeds = protocol.get("final_evaluation", {}).get("seeds", [])
    if not seeds:
        raise ValueError("No final training seeds found in protocol lock.")
    return [int(seed) for seed in seeds]


def import_model_spec(name: str) -> ModelSpec:
    if name not in SUPPORTED_MODELS:
        known = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(f"Unsupported model '{name}'. Supported: {known}")
    kind, module_name, config_name, factory_name = SUPPORTED_MODELS[name]
    module = importlib.import_module(module_name)
    best_params_path = PROJECT_ROOT / "update" / name / "best_params.json"
    if not best_params_path.exists():
        raise FileNotFoundError(f"Missing frozen best params: {best_params_path}")
    return ModelSpec(
        name=name,
        kind=kind,
        config_cls=getattr(module, config_name),
        factory=getattr(module, factory_name) if factory_name else None,
        best_params_path=best_params_path,
    )


def make_final_config(
    spec: ModelSpec,
    best_params: dict[str, Any],
    seed: int,
    run_dir: Path,
    protocol: dict[str, Any],
) -> type:
    base_config = spec.config_cls

    class FinalConfig(base_config):
        pass

    for key, value in best_params.items():
        setattr(FinalConfig, key, value)

    dataset = protocol["dataset"]
    FinalConfig.SEED = int(seed)
    FinalConfig.MODEL_TAG = f"{getattr(base_config, 'MODEL_TAG', spec.name)}_final_seed_{seed}"
    FinalConfig.MODEL_ROOT_DIR = run_dir
    FinalConfig.MODEL_DIR = run_dir
    FinalConfig.SAVE_ROOT_DIR = run_dir
    FinalConfig.SAVE_DIR = run_dir
    FinalConfig.UNIQUE_MODEL_RUN_DIR = False
    FinalConfig.TRAIN_CSV_PATH = Path(dataset["train_csv"]["path"])
    FinalConfig.VAL_CSV_PATH = Path(dataset["val_csv"]["path"])
    FinalConfig.TEST_CSV_PATH = Path(dataset["test_csv_reserved_locked"]["path"])
    FinalConfig.DATA_USE_RATIO = 1.0
    FinalConfig.OPTUNA_FROZEN_BEST_PARAMS = dict(best_params)
    FinalConfig.PUBLICATION_EVAL_PROTOCOL = str(PROTOCOL_PATH)
    return FinalConfig


def set_torch_device(config: type, device: str) -> None:
    if device == "auto":
        return
    import torch

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    config.DEVICE = torch.device(device)


def flatten_metrics(model: str, seed: int, phase: str, metrics: dict[str, Any], fit_seconds: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "seed": int(seed),
        "phase": phase,
        "fit_seconds": float(fit_seconds),
    }
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[str(key).lower()] = float(value)
    return row


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import math
    from collections import defaultdict

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("phase") == "test":
            grouped[str(row["model"])].append(row)

    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        numeric_keys = sorted(
            key
            for key in {item for row in model_rows for item in row}
            if key not in EXCLUDED_FROM_AGGREGATE and isinstance(model_rows[0].get(key), (int, float))
        )
        out: dict[str, Any] = {"model": model, "seed_count": len(model_rows)}
        for key in numeric_keys:
            values = [float(row[key]) for row in model_rows if isinstance(row.get(key), (int, float))]
            if not values:
                continue
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / max(len(values) - 1, 1)
            std = math.sqrt(variance)
            ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_ci95_normal"] = ci95
        output.append(out)
    return output


def copy_common_prediction_names(run_dir: Path, config: type, val_csv: Path, test_csv: Path) -> None:
    """Create stable prediction filenames beside the verbose common artifacts."""
    import shutil

    for phase, csv_path in (("val", val_csv), ("test", test_csv)):
        paths = build_result_paths(config, csv_path, phase)
        stable = run_dir / f"predictions_{phase}.csv"
        if paths["predictions"].exists() and paths["predictions"].resolve() != stable.resolve():
            shutil.copy2(paths["predictions"], stable)


def run_sklearn_final(
    spec: ModelSpec,
    seed: int,
    run_dir: Path,
    protocol: dict[str, Any],
    n_jobs: int | None,
    overwrite: bool,
) -> list[dict[str, Any]]:
    import joblib
    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    done_marker = run_dir / "metrics.json"
    if done_marker.exists() and not overwrite:
        existing = load_json(done_marker)
        return [
            flatten_metrics(spec.name, seed, "val", existing["val_metrics"], float(existing.get("fit_seconds", 0.0))),
            flatten_metrics(spec.name, seed, "test", existing["test_metrics"], float(existing.get("fit_seconds", 0.0))),
        ]

    best_params = load_json(spec.best_params_path)
    config = make_final_config(spec, best_params, seed, run_dir, protocol)
    env_updates = {
        "SURMOD_N_JOBS": str(n_jobs) if n_jobs else None,
        "SURMOD_UNIQUE_MODEL_RUN_DIR": "0",
    }
    with temporary_env(env_updates):
        data = prepare_tabular_training_data(config)
        if spec.factory is None:
            raise RuntimeError(f"Model {spec.name} has no sklearn factory.")
        estimator = spec.factory(config)
        steps = [("imputer", SimpleImputer(strategy=getattr(config, "IMPUTER_STRATEGY", "median")))]
        if bool(getattr(config, "USE_STANDARD_SCALER", False)):
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)
        x_train = data["train_features"].to_numpy(dtype=np.float32)
        x_val = data["val_features"].to_numpy(dtype=np.float32)
        start = time.perf_counter()
        maybe_fit_with_sample_weight(
            pipeline,
            estimator,
            x_train,
            data["y_train"],
            data["sample_weights"] if bool(getattr(config, "USE_SAMPLE_WEIGHT", True)) else None,
        )
        fit_seconds = time.perf_counter() - start

        val_pred = inverse_target(config, pipeline.predict(x_val))
        val_paths = build_result_paths(config, Path(data["metadata"]["val_csv_path"]), "val")
        val_metrics = write_evaluation_outputs(config, data["val_df"], data["y_val_raw"], val_pred, val_paths, "val")

        test_csv = Path(protocol["dataset"]["test_csv_reserved_locked"]["path"])
        test_df = read_valid_csv(config, test_csv)
        x_test = build_test_features(config, data["metadata"], test_df).to_numpy(dtype=np.float32)
        y_test = test_df[config.LABEL_COL].to_numpy(dtype=np.float32)
        test_pred = inverse_target(config, pipeline.predict(x_test))
        test_paths = build_result_paths(config, test_csv, "test")
        test_metrics = write_evaluation_outputs(config, test_df, y_test, test_pred, test_paths, "test")

        model_path = run_dir / getattr(config, "BEST_MODEL_NAME", "best_model.pkl")
        joblib.dump(pipeline, model_path)
        metadata = dict(data["metadata"])
        metadata.update(
            {
                "publication_todo": "TODO5",
                "final_eval_model": spec.name,
                "final_eval_seed": int(seed),
                "frozen_best_params_path": str(spec.best_params_path),
                "frozen_best_params": best_params,
                "best_model_name": model_path.name,
                "fit_seconds": fit_seconds,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "test_csv_path": str(test_csv),
                "locked_test_policy": "Test split is evaluated after frozen-parameter retraining only; no HPO or model selection is performed here.",
            }
        )
        save_json(run_dir / "training_metadata.json", metadata)
        copy_common_prediction_names(run_dir, config, Path(data["metadata"]["val_csv_path"]), test_csv)
        save_json(
            done_marker,
            {
                "model": spec.name,
                "seed": int(seed),
                "fit_seconds": fit_seconds,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "best_params": best_params,
            },
        )

    return [
        flatten_metrics(spec.name, seed, "val", val_metrics, fit_seconds),
        flatten_metrics(spec.name, seed, "test", test_metrics, fit_seconds),
    ]


def run_torch_mlp_final(
    spec: ModelSpec,
    seed: int,
    run_dir: Path,
    protocol: dict[str, Any],
    device: str,
    num_epochs: int | None,
    overwrite: bool,
) -> list[dict[str, Any]]:
    done_marker = run_dir / "metrics.json"
    if done_marker.exists() and not overwrite:
        existing = load_json(done_marker)
        return [
            flatten_metrics(spec.name, seed, "val", existing["val_metrics"], float(existing.get("fit_seconds", 0.0))),
            flatten_metrics(spec.name, seed, "test", existing["test_metrics"], float(existing.get("fit_seconds", 0.0))),
        ]

    best_params = load_json(spec.best_params_path)
    config = make_final_config(spec, best_params, seed, run_dir, protocol)
    set_torch_device(config, device)
    env_updates = {
        "SURMOD_UNIQUE_MODEL_RUN_DIR": "0",
        "SURMOD_NUM_EPOCHS": str(num_epochs) if num_epochs else None,
    }
    with temporary_env(env_updates):
        start = time.perf_counter()
        with TorchCheckpointIOGuard() as checkpoint_guard:
            train_torch_mlp_tabular_model(config)
            checkpoint_guard.finalize(run_dir)
        fit_seconds = time.perf_counter() - start
        evaluate_torch_mlp_tabular_model(config)

        val_csv = Path(protocol["dataset"]["val_csv"]["path"])
        test_csv = Path(protocol["dataset"]["test_csv_reserved_locked"]["path"])
        val_paths = build_result_paths(config, val_csv, "val")
        test_paths = build_result_paths(config, test_csv, "test")
        val_metrics = load_json(val_paths["metrics"])
        test_metrics = load_json(test_paths["metrics"])
        copy_common_prediction_names(run_dir, config, val_csv, test_csv)

        metadata_path = run_dir / "training_metadata.json"
        metadata = load_json(metadata_path)
        metadata.update(
            {
                "publication_todo": "TODO5",
                "final_eval_model": spec.name,
                "final_eval_seed": int(seed),
                "frozen_best_params_path": str(spec.best_params_path),
                "frozen_best_params": best_params,
                "fit_seconds": fit_seconds,
                "test_metrics": test_metrics,
                "test_csv_path": str(test_csv),
                "locked_test_policy": "Test split is evaluated after frozen-parameter retraining only; no HPO or model selection is performed here.",
            }
        )
        save_json(metadata_path, metadata)
        save_json(
            done_marker,
            {
                "model": spec.name,
                "seed": int(seed),
                "fit_seconds": fit_seconds,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "best_params": best_params,
            },
        )

    return [
        flatten_metrics(spec.name, seed, "val", val_metrics, fit_seconds),
        flatten_metrics(spec.name, seed, "test", test_metrics, fit_seconds),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TODO5 frozen-best-parameter multi-seed final evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Override protocol seeds.")
    parser.add_argument("--n-jobs", type=int, default=8, help="CPU workers for sklearn-compatible estimators.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Device for torch MLP runs.")
    parser.add_argument("--mlp-num-epochs", type=int, default=None, help="Optional NUM_EPOCHS override for MLP final runs.")
    parser.add_argument("--overwrite", action="store_true", help="Re-run existing seed directories.")
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
        spec = import_model_spec(model_name)
        for seed in seeds:
            run_dir = final_dir / model_name / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            print(f">>> TODO5 final eval: model={model_name}, seed={seed}, output={run_dir}")
            if spec.kind == "sklearn":
                rows = run_sklearn_final(spec, seed, run_dir, protocol, args.n_jobs, args.overwrite)
            elif spec.kind == "torch_mlp":
                rows = run_torch_mlp_final(spec, seed, run_dir, protocol, args.device, args.mlp_num_epochs, args.overwrite)
            else:
                raise ValueError(f"Unsupported model kind for {model_name}: {spec.kind}")
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

    by_seed_path = final_dir / "summary_metrics_by_seed.csv"
    by_seed_rows = sorted(metric_rows, key=lambda row: (str(row["model"]), int(row["seed"]), str(row["phase"])))
    write_csv(by_seed_path, by_seed_rows)

    aggregate_rows = aggregate_metric_rows(metric_rows)
    aggregate_path = final_dir / "summary_metrics_mean_std_ci.csv"
    write_csv(aggregate_path, aggregate_rows)

    paper_table_path = args.output_root / "paper_tables" / "table2_overall_test_metrics.csv"
    write_csv(paper_table_path, aggregate_rows)

    manifest = {
        "run_name": "todo5_frozen_best_params_multiseed_eval",
        "protocol_path": str(args.protocol),
        "models": list(args.models),
        "seeds": seeds,
        "output_dir": str(final_dir),
        "summary_metrics_by_seed": str(by_seed_path),
        "summary_metrics_mean_std_ci": str(aggregate_path),
        "paper_table": str(paper_table_path),
        "run_records": run_records,
        "notes": [
            "This script performs no Optuna search.",
            "Each run retrains from scratch using frozen best_params.json and one final seed.",
            "The locked test split is evaluated only after fitting; per-seed prediction files are written for paired tests.",
        ],
    }
    save_json(final_dir / "final_multiseed_manifest.json", manifest)
    print(f">>> Wrote {by_seed_path}")
    print(f">>> Wrote {aggregate_path}")
    print(f">>> Wrote {paper_table_path}")


if __name__ == "__main__":
    main()
