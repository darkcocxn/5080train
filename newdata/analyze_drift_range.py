from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DATASET_SUFFIXES = (
    "_train.csv",
    "_val.csv",
    "_test.csv",
    "_wave_split.csv",
    "_wave_features.csv",
    "_summary.json",
)

DEFAULT_DRIFT_COLUMN = "max_drift_ratio_raw"
DRIFT_FALLBACK_COLUMNS = ("max_drift_ratio_raw", "max_drift_ratio_q1e4")
THRESHOLDS = (0.005, 0.01, 0.015, 0.02)
SOURCE_FILE_COL = "__source_file__"

DEFAULT_INPUT = Path(
    r"X:\pyproject\Remote-Train\newdata\opensees_surrogate_dataset_floors_3_to_7_3stage-tailfix-steel01main-steel02damper-light-grid6-fy5-fy500to2500-period09-m50to140-ydr1p0-20260601-001319-drop-split-rank-top5drift_wave_split.csv"
)


def strip_dataset_suffix(name: str) -> str | None:
    for suffix in DATASET_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return None


def resolve_input_path(raw_path: str | None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return DEFAULT_INPUT.resolve()


def resolve_family_files(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        family_files = sorted(
            p
            for p in input_path.iterdir()
            if p.is_file() and strip_dataset_suffix(p.name) is not None
        )
        return family_files

    base_name = strip_dataset_suffix(input_path.name)
    if base_name is None:
        return [input_path]

    parent = input_path.parent
    family_order = ("_train.csv", "_val.csv", "_test.csv")
    family_files = [parent / f"{base_name}{suffix}" for suffix in family_order]
    existing = [path for path in family_files if path.exists()]

    # For wave split / summary inputs, keep the sibling split CSVs even if the
    # source file itself does not contain the drift column.
    if input_path.name.endswith(("_wave_split.csv", "_wave_features.csv", "_summary.json")):
        return existing or [input_path]

    # For train / val / test files, analyze the whole family if available.
    if input_path.name.endswith(family_order):
        if input_path not in existing and input_path.exists():
            existing.append(input_path)
        return existing or [input_path]

    if input_path.suffix.lower() == ".csv" and input_path not in existing:
        existing.append(input_path)

    return [input_path]


def infer_drift_column(df: pd.DataFrame, preferred_column: str | None) -> tuple[str, float]:
    candidates: list[tuple[str, float]] = []
    if preferred_column:
        if preferred_column == "max_drift_ratio_q1e4":
            candidates.append((preferred_column, 10000.0))
        else:
            candidates.append((preferred_column, 1.0))

    for column in DRIFT_FALLBACK_COLUMNS:
        if column == preferred_column:
            continue
        scale = 10000.0 if column == "max_drift_ratio_q1e4" else 1.0
        candidates.append((column, scale))

    for column, scale in candidates:
        if column in df.columns:
            return column, scale

    expected = list(DRIFT_FALLBACK_COLUMNS)
    if preferred_column:
        expected.insert(0, preferred_column)
    expected = list(dict.fromkeys(expected))
    raise KeyError("No drift column found. Expected one of: " + ", ".join(expected))


def make_numeric_series(df: pd.DataFrame, column: str, scale: float) -> pd.Series:
    series = pd.to_numeric(df[column], errors="coerce") / scale
    return series


def row_context(row: pd.Series, source_file: Path, raw_value: float, scale: float, column_name: str) -> dict[str, object]:
    preferred_fields = (
        "sample_id",
        "split",
        "stage",
        "round_idx",
        "num_floors",
        "peak_story_id",
        "analysis_status",
        "failure_category",
        "failure_reason",
        "nonconvergence_flag",
    )

    source_value = row.get(SOURCE_FILE_COL, source_file)
    if pd.isna(source_value):
        source_value = source_file

    payload: dict[str, object] = {"source_file": str(source_value)}
    for field in preferred_fields:
        if field in row.index and field != SOURCE_FILE_COL:
            value = row[field]
            if pd.isna(value):
                continue
            payload[field] = value.item() if hasattr(value, "item") else value

    payload[column_name] = float(raw_value * scale if column_name == "max_drift_ratio_q1e4" else raw_value)
    payload["max_drift_ratio_raw"] = float(raw_value)
    payload["max_drift_ratio_pct"] = float(raw_value * 100.0)
    payload["max_drift_ratio_per_1e4"] = float(raw_value * 10000.0)
    return payload


def summarize_frame(df: pd.DataFrame, source_file: Path, preferred_column: str | None = None) -> dict[str, object]:
    column_name, scale = infer_drift_column(df, preferred_column)
    raw_series = make_numeric_series(df, column_name, scale)
    valid_mask = raw_series.notna()
    valid_series = raw_series[valid_mask]

    if valid_series.empty:
        raise ValueError(f"{source_file} does not contain valid numeric values in {column_name}")

    min_idx = valid_series.idxmin()
    max_idx = valid_series.idxmax()
    min_value = float(valid_series.loc[min_idx])
    max_value = float(valid_series.loc[max_idx])
    span_value = float(max_value - min_value)

    threshold_counts = {
        f"ge_{threshold:.3f}": int((valid_series >= threshold).sum()) for threshold in THRESHOLDS
    }
    threshold_ratios = {
        key: count / float(valid_series.shape[0]) for key, count in threshold_counts.items()
    }

    quantiles = {
        "q05": float(valid_series.quantile(0.05)),
        "q25": float(valid_series.quantile(0.25)),
        "q50": float(valid_series.quantile(0.50)),
        "q75": float(valid_series.quantile(0.75)),
        "q90": float(valid_series.quantile(0.90)),
        "q95": float(valid_series.quantile(0.95)),
        "q99": float(valid_series.quantile(0.99)),
    }

    return {
        "source_file": str(source_file),
        "source_column": column_name,
        "source_scale": scale,
        "rows_total": int(df.shape[0]),
        "rows_valid": int(valid_series.shape[0]),
        "rows_invalid": int(df.shape[0] - valid_series.shape[0]),
        "drift_min_raw": min_value,
        "drift_max_raw": max_value,
        "drift_span_raw": span_value,
        "drift_min_pct": float(min_value * 100.0),
        "drift_max_pct": float(max_value * 100.0),
        "drift_span_pct": float(span_value * 100.0),
        "drift_min_per_1e4": float(min_value * 10000.0),
        "drift_max_per_1e4": float(max_value * 10000.0),
        "drift_span_per_1e4": float(span_value * 10000.0),
        "mean_raw": float(valid_series.mean()),
        "median_raw": float(valid_series.median()),
        "std_raw": float(valid_series.std(ddof=0)),
        "quantiles_raw": quantiles,
        "threshold_counts": threshold_counts,
        "threshold_ratios": threshold_ratios,
        "min_row": row_context(df.loc[min_idx], source_file, min_value, scale, column_name),
        "max_row": row_context(df.loc[max_idx], source_file, max_value, scale, column_name),
    }


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def build_report(input_path: Path, preferred_column: str | None = None, single_file: bool = False) -> dict[str, object]:
    family_files = [input_path] if single_file else resolve_family_files(input_path)
    csv_files = [path for path in family_files if path.is_file() and path.suffix.lower() == ".csv"]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found for {input_path}")

    per_file: list[dict[str, object]] = []
    frames: list[pd.DataFrame] = []

    for path in csv_files:
        df = load_csv(path)
        df = df.copy()
        df[SOURCE_FILE_COL] = str(path)
        frames.append(df)
        per_file.append(summarize_frame(df, path, preferred_column=preferred_column))

    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    overall_label = input_path if input_path.is_file() else csv_files[0]
    overall = summarize_frame(combined, overall_label, preferred_column=preferred_column)

    return {
        "input_path": str(input_path),
        "single_file": single_file,
        "family_files": [str(path) for path in csv_files],
        "overall": overall,
        "per_file": per_file,
    }


def print_report(report: dict[str, object]) -> None:
    overall = report["overall"]
    print(f"输入路径: {report['input_path']}")
    print(f"读取文件数: {len(report['family_files'])}")
    print(f"分析列: {overall['source_column']}")
    print(
        "整体结果: "
        f"最小值={overall['drift_min_raw']:.8f} "
        f"最大值={overall['drift_max_raw']:.8f} "
        f"区间={overall['drift_span_raw']:.8f} "
        f"({overall['drift_min_pct']:.4f}% ~ {overall['drift_max_pct']:.4f}%)"
    )
    print(f"样本数: 有效 {overall['rows_valid']} / 总计 {overall['rows_total']}")
    print("阈值计数:", overall["threshold_counts"])
    print("最小值样本:", overall["min_row"])
    print("最大值样本:", overall["max_row"])

    if len(report["per_file"]) > 1:
        print("\n分文件结果:")
        for item in report["per_file"]:
            print(
                f"  - {Path(item['source_file']).name}: "
                f"min={item['drift_min_raw']:.8f}, "
                f"max={item['drift_max_raw']:.8f}, "
                f"span={item['drift_span_raw']:.8f}, "
                f"valid={item['rows_valid']}"
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze drift ratio min/max range from the dataset CSV family.")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Input CSV, summary JSON, or dataset directory. Defaults to the provided olddata wave_split CSV.",
    )
    parser.add_argument(
        "--column",
        default=None,
        help="Drift column to analyze. Defaults to max_drift_ratio_raw, falling back to max_drift_ratio_q1e4.",
    )
    parser.add_argument(
        "--single-file",
        action="store_true",
        help="Analyze only the provided file instead of the sibling train/val/test family.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the report as JSON.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_path = resolve_input_path(args.path)
    report = build_report(input_path, preferred_column=args.column, single_file=args.single_file)
    print_report(report)

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写出 JSON 报告: {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
