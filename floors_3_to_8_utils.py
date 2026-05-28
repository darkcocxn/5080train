# -*- coding: utf-8 -*-
"""
3 到 8 层新版数据集训练/测试脚本共用工具。
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def resolve_dataset_path(project_root, relative_dir, explicit_path, pattern):
    project_root = Path(project_root)
    dataset_dir = project_root / relative_dir

    if explicit_path:
        dataset_path = Path(explicit_path)
        if not dataset_path.is_absolute():
            dataset_path = project_root / dataset_path
        return dataset_path

    candidates = sorted(
        dataset_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"在 '{dataset_dir}' 中未找到匹配 '{pattern}' 的数据集文件。"
        )

    return candidates[0]


def validate_dataframe(df, required_cols, csv_path):
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"CSV 文件缺少必要列: {missing_cols}，源文件: {csv_path}")


def sample_dataframe_by_group(df, group_col, data_use_ratio, seed):
    if data_use_ratio >= 1.0:
        print(f">>> Using 100% of data: {len(df)} samples")
        return df.reset_index(drop=True)

    unique_groups = df[group_col].drop_duplicates().tolist()
    use_group_count = int(len(unique_groups) * data_use_ratio)
    min_group_count = 2 if len(unique_groups) >= 2 else 1
    use_group_count = max(min_group_count, min(use_group_count, len(unique_groups)))

    rng = np.random.RandomState(seed)
    selected_groups = rng.choice(unique_groups, size=use_group_count, replace=False)
    sampled_df = df[df[group_col].isin(selected_groups)].reset_index(drop=True)

    print(
        f">>> Using {data_use_ratio * 100:.1f}% of groups: "
        f"{use_group_count}/{len(unique_groups)} groups, {len(sampled_df)}/{len(df)} samples"
    )
    return sampled_df


def split_dataframe_by_group(df, group_col, val_split, seed):
    unique_groups = df[group_col].drop_duplicates().tolist()
    if len(unique_groups) < 2:
        raise ValueError(f"唯一分组数量不足 2，无法基于 '{group_col}' 划分训练集和验证集。")

    rng = np.random.RandomState(seed)
    rng.shuffle(unique_groups)

    val_group_count = int(round(len(unique_groups) * val_split))
    val_group_count = max(1, min(val_group_count, len(unique_groups) - 1))

    val_groups = set(unique_groups[:val_group_count])
    train_groups = set(unique_groups[val_group_count:])

    train_df = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val_df = df[df[group_col].isin(val_groups)].reset_index(drop=True)

    print(
        f">>> Train/Val split by {group_col}: "
        f"{len(train_groups)} train groups / {len(val_groups)} val groups"
    )
    print(
        f">>> Train/Val samples: {len(train_df)} / {len(val_df)} "
        f"({len(train_df) / len(df):.2%} / {len(val_df) / len(df):.2%})"
    )
    return train_df, val_df


def load_seismic_sequence(txt_path, seq_length):
    try:
        data = np.loadtxt(txt_path, dtype=np.float32)
        if data.ndim > 1:
            data = data.reshape(-1)

        if len(data) >= seq_length:
            return data[:seq_length]

        padded = np.zeros(seq_length, dtype=np.float32)
        padded[: len(data)] = data
        return padded
    except Exception as exc:
        print(f"Warning: Error loading {txt_path}: {exc}")
        return np.zeros(seq_length, dtype=np.float32)


def generate_windowed_sequence(seismic_data, window_size):
    usable_length = (len(seismic_data) // window_size) * window_size
    if usable_length == 0:
        return np.zeros((1, window_size, 1), dtype=np.float32)

    windowed = seismic_data[:usable_length].reshape(-1, window_size)
    return windowed[:, :, np.newaxis].astype(np.float32)


def resolve_image_path(raw_path, fallback_dir=None):
    raw_path = Path(str(raw_path))
    candidates = [raw_path]

    if fallback_dir is not None:
        fallback_dir = Path(fallback_dir)
        candidates.append(fallback_dir / raw_path.name)

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[-1])


def calculate_relative_errors(y_true, y_pred):
    denominator = np.abs(y_true) + 1e-9
    return np.abs(y_true - y_pred) / denominator * 100.0


def calculate_metrics(y_true, y_pred):
    relative_errors = calculate_relative_errors(y_true, y_pred)
    return {
        "Samples": len(y_true),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": float(np.mean(relative_errors)),
    }


def build_seed_metrics_dataframe(
    y_true,
    y_pred,
    seeds,
    sample_ratio=0.8,
    sample_with_replacement=False,
):
    sample_size = int(round(len(y_true) * sample_ratio))
    sample_size = max(2, min(sample_size, len(y_true)))

    rows = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        indices = rng.choice(
            len(y_true),
            size=sample_size,
            replace=sample_with_replacement,
        )
        metrics = calculate_metrics(y_true[indices], y_pred[indices])
        rows.append(
            {
                "Seed": seed,
                "Samples": metrics["Samples"],
                "R2": metrics["R2"],
                "MAE": metrics["MAE"],
                "RMSE": metrics["RMSE"],
                "MAPE": metrics["MAPE"],
            }
        )

    return pd.DataFrame(rows)


def build_seed_metrics_report(seed_metrics_df):
    report_df = seed_metrics_df.copy()
    numeric_cols = ["R2", "MAE", "RMSE", "MAPE"]

    mean_row = {"Seed": "mean", "Samples": seed_metrics_df["Samples"].mean()}
    std_row = {"Seed": "std", "Samples": seed_metrics_df["Samples"].std(ddof=0)}
    for col in numeric_cols:
        mean_row[col] = seed_metrics_df[col].mean()
        std_row[col] = seed_metrics_df[col].std(ddof=0)

    return pd.concat([report_df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def print_seed_summary(seed_metrics_df):
    summary = build_seed_metrics_report(seed_metrics_df).tail(2).reset_index(drop=True)
    mean_row = summary.iloc[0]
    std_row = summary.iloc[1]

    print("\n" + "-" * 60)
    print("Multi-seed resampling summary")
    print(
        f"R2   : {mean_row['R2']:.4f} ± {std_row['R2']:.4f} | "
        f"MAE  : {mean_row['MAE']:.6f} ± {std_row['MAE']:.6f}"
    )
    print(
        f"RMSE : {mean_row['RMSE']:.6f} ± {std_row['RMSE']:.6f} | "
        f"MAPE : {mean_row['MAPE']:.2f}% ± {std_row['MAPE']:.2f}%"
    )
    print("-" * 60)
