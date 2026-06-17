# -*- coding: utf-8 -*-
"""Shared Optuna/TPE tuning helpers for the surrogate-model repository."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_arg_parser(algorithm_name: str, default_trials: int) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=f"Tune {algorithm_name} with Optuna/TPE.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--trials", type=int, default=default_trials, help="Number of Optuna trials.")
    parser.add_argument("--timeout", type=int, default=None, help="Maximum optimization time in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed for TPE and trial configs.")
    parser.add_argument("--startup-trials", type=int, default=8, help="Random startup trials before TPE.")
    parser.add_argument(
        "--objective-metric",
        choices=("selection", "mae", "rmse"),
        default="selection",
        help="Validation objective minimized by Optuna.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=None,
        help="Root directory for per-trial artifacts. Defaults to update/<algorithm>/runs.",
    )
    parser.add_argument("--study-name", default=None, help="Optuna study name.")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optional Optuna storage URI, e.g. sqlite:///update/randomforest/optuna.db.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume an existing named study when storage is set.")
    parser.add_argument("--data-use-ratio", type=float, default=None, help="Fraction of wave groups used for training.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Override NUM_EPOCHS for neural models.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override BATCH_SIZE.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override NUM_WORKERS when the config has it.")
    parser.add_argument("--n-jobs", type=int, default=None, help="Set SURMOD_N_JOBS for CPU tree models.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Override torch device for neural models.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and print one parameter set without training.",
    )
    return parser


def load_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Optuna is required for these tuning scripts. Install with: uv add optuna") from exc
    return optuna


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2, default=json_default)


def is_windows_user_mapped_section_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    winerror = getattr(exc, "winerror", None)
    return (
        winerror in {5, 1224}
        or "error code: 1224" in text
        or "user-mapped section open" in text
        or "access is denied" in text
    )


def replace_matching_filenames(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace_matching_filenames(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_matching_filenames(item, replacements) for item in value]
    if isinstance(value, str):
        return replacements.get(value, value)
    return value


class TorchCheckpointIOGuard:
    def __init__(self) -> None:
        self._torch = None
        self._original_save = None
        self._original_load = None
        self._redirected_targets: dict[Path, Path] = {}

    @staticmethod
    def _key(path_like: os.PathLike[str] | str) -> Path:
        return Path(path_like).expanduser().resolve(strict=False)

    @staticmethod
    def _needs_guard(path_like: Any) -> bool:
        if not isinstance(path_like, (str, os.PathLike)):
            return False
        return Path(path_like).suffix.lower() in {".pth", ".pt", ".ckpt"}

    def _promote_temp_path(self, temp_path: Path, target_path: Path) -> None:
        last_exc: OSError | None = None
        for attempt in range(8):
            try:
                os.replace(temp_path, target_path)
                return
            except OSError as exc:
                if not is_windows_user_mapped_section_error(exc):
                    raise
                last_exc = exc
                gc.collect()
                time.sleep(0.15 * (attempt + 1))
        if last_exc is not None:
            raise last_exc

    def _sync_redirects(self) -> dict[Path, Path]:
        unresolved: dict[Path, Path] = {}
        for target_key, redirect_path in list(self._redirected_targets.items()):
            if not redirect_path.exists():
                self._redirected_targets.pop(target_key, None)
                continue
            target_path = target_key
            try:
                self._promote_temp_path(redirect_path, target_path)
            except OSError as exc:
                if not is_windows_user_mapped_section_error(exc):
                    raise
                unresolved[target_path] = redirect_path
                continue
            self._redirected_targets.pop(target_key, None)
        return unresolved

    def finalize(self, model_dir: Path | None = None) -> None:
        unresolved = self._sync_redirects()
        if not unresolved or model_dir is None:
            return
        metadata_path = Path(model_dir) / "training_metadata.json"
        if not metadata_path.exists():
            return
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        replacements = {target.name: redirect.name for target, redirect in unresolved.items()}
        updated_metadata = replace_matching_filenames(metadata, replacements)
        save_json(metadata_path, updated_metadata)

    def __enter__(self):
        import torch

        self._torch = torch
        self._original_save = torch.save
        self._original_load = torch.load

        def guarded_save(obj, f, *args, **kwargs):
            if not self._needs_guard(f):
                return self._original_save(obj, f, *args, **kwargs)
            target_path = Path(f)
            target_key = self._key(target_path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = target_path.with_name(
                f".{target_path.stem}.save-tmp-{os.getpid()}-{time.time_ns()}{target_path.suffix}"
            )
            self._original_save(obj, temp_path, *args, **kwargs)
            try:
                self._promote_temp_path(temp_path, target_path)
            except OSError as exc:
                if not is_windows_user_mapped_section_error(exc):
                    if temp_path.exists():
                        temp_path.unlink(missing_ok=True)
                    raise
                redirect_path = target_path.with_name(
                    f"{target_path.stem}.redirect-{int(time.time())}-{time.time_ns() % 1_000_000:06d}{target_path.suffix}"
                )
                os.replace(temp_path, redirect_path)
                previous_redirect = self._redirected_targets.get(target_key)
                self._redirected_targets[target_key] = redirect_path
                if previous_redirect is not None and previous_redirect.exists():
                    previous_redirect.unlink(missing_ok=True)
                print(
                    f">>> Checkpoint replace blocked for {target_path.name}; "
                    f"temporarily using {redirect_path.name}"
                )
                return
            previous_redirect = self._redirected_targets.pop(target_key, None)
            if previous_redirect is not None and previous_redirect.exists():
                previous_redirect.unlink(missing_ok=True)

        def guarded_load(f, *args, **kwargs):
            if self._needs_guard(f):
                target_key = self._key(f)
                redirect_path = self._redirected_targets.get(target_key)
                if redirect_path is not None and redirect_path.exists():
                    return self._original_load(redirect_path, *args, **kwargs)
            return self._original_load(f, *args, **kwargs)

        torch.save = guarded_save
        torch.load = guarded_load
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._torch is not None:
            self._torch.save = self._original_save
            self._torch.load = self._original_load
        return False


def output_root_for(algorithm_name: str) -> Path:
    return PROJECT_ROOT / "update" / algorithm_name


def trial_artifact_root(args: argparse.Namespace, algorithm_name: str, trial_number: int) -> Path:
    root = Path(args.model_root) if args.model_root else output_root_for(algorithm_name) / "runs"
    return root / f"trial_{trial_number:04d}"


def make_trial_config(
    base_config: type,
    params: dict[str, Any],
    args: argparse.Namespace,
    algorithm_name: str,
    trial_number: int,
) -> type:
    class TunedConfig(base_config):
        pass

    for key, value in params.items():
        setattr(TunedConfig, key, value)

    base_tag = getattr(base_config, "MODEL_TAG", algorithm_name)
    TunedConfig.MODEL_TAG = f"{base_tag}_optuna_tpe_trial_{trial_number:04d}"
    TunedConfig.MODEL_FAMILY = getattr(base_config, "MODEL_FAMILY", base_tag)
    TunedConfig.ARCHITECTURE_REVISION = (
        f"{getattr(base_config, 'ARCHITECTURE_REVISION', base_tag)}_optuna_tpe"
    )
    TunedConfig.SEED = int(args.seed) + int(trial_number)
    TunedConfig.UNIQUE_MODEL_RUN_DIR = True

    root = trial_artifact_root(args, algorithm_name, trial_number)
    TunedConfig.MODEL_ROOT_DIR = root
    TunedConfig.MODEL_DIR = root
    TunedConfig.SAVE_ROOT_DIR = root
    TunedConfig.SAVE_DIR = root

    if args.data_use_ratio is not None:
        TunedConfig.DATA_USE_RATIO = float(args.data_use_ratio)
    if args.num_epochs is not None and hasattr(TunedConfig, "NUM_EPOCHS"):
        TunedConfig.NUM_EPOCHS = int(args.num_epochs)
    if args.batch_size is not None and hasattr(TunedConfig, "BATCH_SIZE"):
        TunedConfig.BATCH_SIZE = int(args.batch_size)
    if args.num_workers is not None and hasattr(TunedConfig, "NUM_WORKERS"):
        TunedConfig.NUM_WORKERS = int(args.num_workers)
    if args.device != "auto" and hasattr(TunedConfig, "DEVICE"):
        import torch

        TunedConfig.DEVICE = torch.device(args.device)

    TunedConfig.OPTUNA_CONFIG_PARAMS = {
        key: getattr(TunedConfig, key, value) for key, value in params.items()
    }
    return TunedConfig


@contextmanager
def temporary_model_env(prefixes: tuple[str, ...], args: argparse.Namespace):
    keys = [key for key in os.environ if any(key.startswith(prefix) for prefix in prefixes)]
    removed = {key: os.environ.pop(key) for key in keys}
    old_n_jobs = os.environ.get("SURMOD_N_JOBS")
    if args.n_jobs is not None:
        os.environ["SURMOD_N_JOBS"] = str(int(args.n_jobs))
    try:
        yield
    finally:
        for key in [key for key in os.environ if any(key.startswith(prefix) for prefix in prefixes)]:
            os.environ.pop(key, None)
        os.environ.update(removed)
        if args.n_jobs is None:
            if old_n_jobs is None:
                os.environ.pop("SURMOD_N_JOBS", None)
            else:
                os.environ["SURMOD_N_JOBS"] = old_n_jobs
        elif old_n_jobs is None:
            os.environ.pop("SURMOD_N_JOBS", None)
        else:
            os.environ["SURMOD_N_JOBS"] = old_n_jobs


def score_from_metrics(metrics: dict[str, Any], objective_metric: str) -> float:
    normalized = {str(key).lower(): value for key, value in metrics.items()}
    if objective_metric == "mae":
        return float(normalized.get("mae"))
    if objective_metric == "rmse":
        return float(normalized.get("rmse"))
    for key in ("selectionscore", "selection_score", "val_selection_score"):
        if key in normalized:
            return float(normalized[key])
    if "mae" in normalized:
        return float(normalized["mae"])
    raise KeyError(f"Cannot resolve objective metric from keys: {sorted(metrics)}")


def score_from_history(history: dict[str, list[Any]], objective_metric: str) -> float:
    key_map = {
        "selection": ("val_selection_score", "selection_score"),
        "mae": ("val_mae_raw", "mae"),
        "rmse": ("val_rmse_raw", "rmse"),
    }
    for key in key_map[objective_metric]:
        if key in history and history[key]:
            return float(min(float(value) for value in history[key]))
    raise KeyError(f"Cannot resolve {objective_metric!r} from history keys: {sorted(history)}")


def build_study(args: argparse.Namespace, algorithm_name: str):
    optuna = load_optuna()
    import warnings

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    sampler = optuna.samplers.TPESampler(
        seed=int(args.seed),
        n_startup_trials=int(args.startup_trials),
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(3, int(args.startup_trials)))
    study_name = args.study_name or f"{algorithm_name}_optuna_tpe"
    return optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        study_name=study_name,
        load_if_exists=bool(args.resume),
    )


def write_study_outputs(study, args: argparse.Namespace, algorithm_name: str) -> None:
    output_dir = output_root_for(algorithm_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = output_dir / "optuna_trials.csv"
    study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs")).to_csv(
        trials_csv,
        index=False,
        encoding="utf-8-sig",
    )

    complete_trials = [trial for trial in study.trials if str(trial.state) == "TrialState.COMPLETE"]
    summary: dict[str, Any] = {
        "algorithm": algorithm_name,
        "study_name": study.study_name,
        "objective_metric": args.objective_metric,
        "trial_count": len(study.trials),
        "complete_trial_count": len(complete_trials),
        "trials_csv": str(trials_csv),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if complete_trials:
        best_trial = study.best_trial
        best_config_params = best_trial.user_attrs.get("config_params", best_trial.params)
        summary.update(
            {
                "best_trial_number": int(best_trial.number),
                "best_value": float(best_trial.value),
                "best_optuna_params": dict(best_trial.params),
                "best_config_params": best_config_params,
                "best_artifact_root": best_trial.user_attrs.get("artifact_root"),
                "best_model_dir": best_trial.user_attrs.get("model_dir"),
                "best_metrics": best_trial.user_attrs.get("metrics"),
            }
        )
        save_json(output_dir / "best_params.json", best_config_params)
    save_json(output_dir / "study_summary.json", summary)


def dry_run_suggestion(
    args: argparse.Namespace,
    algorithm_name: str,
    suggest_params: Callable[[Any], dict[str, Any]],
) -> bool:
    if not args.dry_run:
        return False
    study = build_study(args, algorithm_name)
    trial = study.ask()
    params = suggest_params(trial)
    print(json.dumps(params, ensure_ascii=False, indent=2, default=json_default))
    return True


def run_study(
    args: argparse.Namespace,
    algorithm_name: str,
    objective: Callable[[Any], float],
) -> None:
    study = build_study(args, algorithm_name)
    study.optimize(objective, n_trials=int(args.trials), timeout=args.timeout, gc_after_trial=True)
    write_study_outputs(study, args, algorithm_name)
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None:
        print(f">>> Best trial: {best_trial.number}")
        print(f">>> Best value: {float(best_trial.value):.10f}")
    print(f">>> Outputs: {output_root_for(algorithm_name)}")


def parse_args_and_maybe_dry_run(
    algorithm_name: str,
    default_trials: int,
    suggest_params: Callable[[Any], dict[str, Any]],
) -> argparse.Namespace | None:
    args = build_arg_parser(algorithm_name, default_trials).parse_args()
    if dry_run_suggestion(args, algorithm_name, suggest_params):
        return None
    return args


def run_sklearn_tpe_tuning(
    algorithm_name: str,
    base_config: type,
    model_factory: Callable[[Any], Any],
    suggest_params: Callable[[Any], dict[str, Any]],
    *,
    model_env_prefixes: tuple[str, ...],
    default_trials: int,
) -> None:
    args = parse_args_and_maybe_dry_run(algorithm_name, default_trials, suggest_params)
    if args is None:
        return

    def objective(trial) -> float:
        import numpy as np
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        from tabular_model_common import (
            apply_tabular_environment_overrides,
            calculate_regression_metrics,
            calculate_selection_score,
            inverse_target,
            maybe_fit_with_sample_weight,
            prepare_tabular_training_data,
        )

        params = suggest_params(trial)
        trial.set_user_attr("config_params", params)
        config = make_trial_config(base_config, params, args, algorithm_name, trial.number)
        trial.set_user_attr("config_params", getattr(config, "OPTUNA_CONFIG_PARAMS", params))
        with temporary_model_env(model_env_prefixes, args):
            apply_tabular_environment_overrides(config)
            data = prepare_tabular_training_data(config)
            estimator = model_factory(config)
            steps = [("imputer", SimpleImputer(strategy=getattr(config, "IMPUTER_STRATEGY", "median")))]
            if bool(getattr(config, "USE_STANDARD_SCALER", False)):
                steps.append(("scaler", StandardScaler()))
            steps.append(("model", estimator))
            pipeline = Pipeline(steps)
            x_train = data["train_features"].to_numpy(dtype=np.float32)
            x_val = data["val_features"].to_numpy(dtype=np.float32)
            maybe_fit_with_sample_weight(
                pipeline,
                estimator,
                x_train,
                data["y_train"],
                data["sample_weights"] if bool(getattr(config, "USE_SAMPLE_WEIGHT", True)) else None,
            )
            y_pred = inverse_target(config, pipeline.predict(x_val))
            metrics = calculate_regression_metrics(data["y_val_raw"], y_pred)
            metrics["SelectionScore"] = calculate_selection_score(data["y_val_raw"], y_pred)
            value = score_from_metrics(metrics, args.objective_metric)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("artifact_root", str(config.MODEL_ROOT_DIR))
            trial.set_user_attr("model_dir", str(config.MODEL_DIR))
            save_json(config.MODEL_DIR / "optuna_trial.json", {"params": params, "metrics": metrics, "value": value})
            return value

    run_study(args, algorithm_name, objective)


def run_torch_mlp_tpe_tuning(
    algorithm_name: str,
    base_config: type,
    train_callable: Callable[[Any], Any],
    suggest_params: Callable[[Any], dict[str, Any]],
    *,
    default_trials: int,
) -> None:
    args = parse_args_and_maybe_dry_run(algorithm_name, default_trials, suggest_params)
    if args is None:
        return

    def objective(trial) -> float:
        params = suggest_params(trial)
        trial.set_user_attr("config_params", params)
        config = make_trial_config(base_config, params, args, algorithm_name, trial.number)
        trial.set_user_attr("config_params", getattr(config, "OPTUNA_CONFIG_PARAMS", params))
        with TorchCheckpointIOGuard() as checkpoint_guard:
            train_callable(config)
            checkpoint_guard.finalize(Path(config.MODEL_DIR))
        metadata_path = Path(config.MODEL_DIR) / "training_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        metrics = metadata.get("val_metrics", {})
        value = score_from_metrics(metrics, args.objective_metric)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("artifact_root", str(config.MODEL_ROOT_DIR))
        trial.set_user_attr("model_dir", str(config.MODEL_DIR))
        save_json(config.MODEL_DIR / "optuna_trial.json", {"params": params, "metrics": metrics, "value": value})
        return value

    run_study(args, algorithm_name, objective)


def run_history_module_tpe_tuning(
    algorithm_name: str,
    module: Any,
    base_config: type,
    train_callable: Callable[[Any], dict[str, list[Any]]],
    suggest_params: Callable[[Any], dict[str, Any]],
    *,
    default_trials: int,
) -> None:
    args = parse_args_and_maybe_dry_run(algorithm_name, default_trials, suggest_params)
    if args is None:
        return

    def objective(trial) -> float:
        params = suggest_params(trial)
        trial.set_user_attr("config_params", params)
        config = make_trial_config(base_config, params, args, algorithm_name, trial.number)
        trial.set_user_attr("config_params", getattr(config, "OPTUNA_CONFIG_PARAMS", params))
        module.Config = config
        with TorchCheckpointIOGuard() as checkpoint_guard:
            history = train_callable(config)
            checkpoint_guard.finalize(Path(getattr(config, "SAVE_DIR", config.MODEL_DIR)))
        value = score_from_history(history, args.objective_metric)
        metrics = {"best_value": value, "objective_metric": args.objective_metric}
        if "val_mae_raw" in history and history["val_mae_raw"]:
            metrics["best_val_mae_raw"] = float(min(history["val_mae_raw"]))
        if "val_selection_score" in history and history["val_selection_score"]:
            metrics["best_val_selection_score"] = float(min(history["val_selection_score"]))
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("artifact_root", str(config.MODEL_ROOT_DIR))
        trial.set_user_attr("model_dir", str(config.MODEL_DIR))
        save_json(config.MODEL_DIR / "optuna_trial.json", {"params": params, "metrics": metrics, "value": value})
        return value

    run_study(args, algorithm_name, objective)
