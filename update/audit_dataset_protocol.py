"""Audit the locked publication dataset without running training.

This script implements TODO 2 from the publication checklist. It reads the
train/validation/locked-test CSV files from protocol_lock.json, computes split
manifests, target/tail distributions, filtering clues, and group-overlap
reports, then writes audit artifacts under publication_eval_*/data_audit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_JSON = PROJECT_ROOT / "publication_eval_20260614" / "protocol" / "protocol_lock.json"
TARGET_COLUMN = "max_drift_ratio_raw"
STATUS_COLUMN = "analysis_status"
WAVE_COLUMN = "txt_path"
IMAGE_COLUMN = "image_path"
SPLIT_COLUMN = "split"
TAIL_THRESHOLDS = (0.005, 0.010, 0.015, 0.020)
QUANTILES = (0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
KEY_CATEGORICAL_COLUMNS = (
    "split",
    "stage",
    "round_idx",
    "num_floors",
    "wave_cluster",
    "period_compliant",
    "period_check_status",
    "analysis_status",
    "failure_category",
    "steel01_yielded",
    "steel02_yielded",
    "nonconvergence_flag",
)
STRUCTURE_SIGNATURE_COLUMNS = (
    "num_floors",
    "floor_mass",
    "floor_height",
    "k_base_1_4",
    "k_add",
    "Fy_add",
    "damper_layout",
    "frame_b",
    "frame_yield_drift_ratio",
    "period_1_sec",
)
NUMERIC_RANGE_COLUMNS = (
    "num_floors",
    "floor_mass",
    "floor_height",
    "k_base_1_4",
    "k_add",
    "Fy_add",
    "frame_b",
    "frame_yield_drift_ratio",
    "period_1_sec",
    "wave_dt",
    "wave_length",
    "wave_pga",
    "wave_rms",
    "wave_cav",
    "wave_arias_proxy",
    "wave_duration_5_95",
    "wave_dominant_freq",
    "wave_spectral_centroid",
    "wave_predominant_period",
    "wave_intensity_score",
    TARGET_COLUMN,
)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def quantile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def summarize_numeric(values: list[float]) -> dict[str, Any]:
    values = [value for value in values if math.isfinite(value)]
    if not values:
        return {
            "count": 0,
            "missing_or_nonfinite": None,
            "min": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    sorted_values = sorted(values)
    mean = sum(sorted_values) / len(sorted_values)
    variance = sum((value - mean) ** 2 for value in sorted_values) / max(len(sorted_values) - 1, 1)
    return {
        "count": len(sorted_values),
        "min": sorted_values[0],
        "p50": quantile(sorted_values, 0.50),
        "p75": quantile(sorted_values, 0.75),
        "p90": quantile(sorted_values, 0.90),
        "p95": quantile(sorted_values, 0.95),
        "p99": quantile(sorted_values, 0.99),
        "max": sorted_values[-1],
        "mean": mean,
        "std": math.sqrt(variance),
    }


def split_signature(row: dict[str, str], columns: tuple[str, ...]) -> str:
    return "|".join(str(row.get(column, "")).strip() for column in columns)


def read_split_csv(split_name: str, path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    for row in rows:
        row["_audit_split"] = split_name
    return rows, fieldnames


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_protocol(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def split_paths_from_protocol(protocol: dict[str, Any]) -> dict[str, Path]:
    dataset = protocol["dataset"]
    return {
        "train": Path(dataset["train_csv"]["path"]).resolve(),
        "val": Path(dataset["val_csv"]["path"]).resolve(),
        "test": Path(dataset["test_csv_reserved_locked"]["path"]).resolve(),
    }


def build_split_manifest(split_name: str, path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> dict[str, Any]:
    target_values = [parse_float(row.get(TARGET_COLUMN)) for row in rows]
    valid_targets = [value for value in target_values if value is not None]
    status_counts = Counter(row.get(STATUS_COLUMN, "<missing>") or "<empty>" for row in rows)
    split_counts = Counter(row.get(SPLIT_COLUMN, "<missing>") or "<empty>" for row in rows)
    missing_by_column = {
        column: sum(1 for row in rows if str(row.get(column, "")).strip() == "")
        for column in fieldnames
    }
    duplicate_sample_ids = 0
    if "sample_id" in fieldnames:
        sample_counts = Counter(row.get("sample_id", "") for row in rows)
        duplicate_sample_ids = sum(count - 1 for value, count in sample_counts.items() if value and count > 1)
    return {
        "split": split_name,
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "sha256": sha256_file(path) if path.exists() else None,
        "rows": len(rows),
        "columns": fieldnames,
        "target_column_present": TARGET_COLUMN in fieldnames,
        "target_valid_count": len(valid_targets),
        "target_missing_or_nonfinite": len(rows) - len(valid_targets),
        "target_summary": summarize_numeric(valid_targets),
        "status_counts": dict(sorted(status_counts.items())),
        "split_column_counts": dict(sorted(split_counts.items())),
        "duplicate_sample_id_rows": duplicate_sample_ids,
        "missing_by_column": missing_by_column,
        "unique_counts": {
            "sample_id": len({row.get("sample_id", "") for row in rows if row.get("sample_id", "")}) if "sample_id" in fieldnames else None,
            "txt_path": len({row.get(WAVE_COLUMN, "") for row in rows if row.get(WAVE_COLUMN, "")}) if WAVE_COLUMN in fieldnames else None,
            "image_path": len({row.get(IMAGE_COLUMN, "") for row in rows if row.get(IMAGE_COLUMN, "")}) if IMAGE_COLUMN in fieldnames else None,
            "structure_signature": len({split_signature(row, STRUCTURE_SIGNATURE_COLUMNS) for row in rows}),
        },
    }


def build_target_tail_rows(split_rows: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, data in split_rows.items():
        target_values = [parse_float(row.get(TARGET_COLUMN)) for row in data]
        valid_values = [value for value in target_values if value is not None]
        sorted_values = sorted(valid_values)
        summary = summarize_numeric(valid_values)
        out: dict[str, Any] = {
            "split": split_name,
            "rows": len(data),
            "valid_target_count": len(valid_values),
            "missing_or_nonfinite_target": len(data) - len(valid_values),
            "min": summary["min"],
            "p50": summary["p50"],
            "p75": summary["p75"],
            "p90": summary["p90"],
            "p95": summary["p95"],
            "p99": summary["p99"],
            "max": summary["max"],
            "mean": summary["mean"],
            "std": summary["std"],
        }
        for threshold in TAIL_THRESHOLDS:
            count = sum(1 for value in valid_values if value >= threshold)
            out[f"count_ge_{threshold:.3f}"] = count
            out[f"frac_ge_{threshold:.3f}"] = count / len(valid_values) if valid_values else None
        for q in (0.90, 0.95):
            threshold = quantile(sorted_values, q)
            count = sum(1 for value in valid_values if threshold is not None and value >= threshold)
            out[f"threshold_p{int(q * 100)}"] = threshold
            out[f"count_ge_p{int(q * 100)}"] = count
        rows.append(out)
    return rows


def build_distribution_rows(split_rows: dict[str, list[dict[str, str]]], fieldnames: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name, rows in split_rows.items():
        total = len(rows)
        for column in KEY_CATEGORICAL_COLUMNS:
            if column not in fieldnames:
                continue
            counts = Counter((row.get(column, "") or "<empty>") for row in rows)
            for value, count in sorted(counts.items(), key=lambda item: (str(item[0]), item[1])):
                output.append(
                    {
                        "variable": column,
                        "split": split_name,
                        "value": value,
                        "count": count,
                        "fraction": count / total if total else None,
                    }
                )
    return output


def build_numeric_range_rows(split_rows: dict[str, list[dict[str, str]]], fieldnames: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name, rows in split_rows.items():
        for column in NUMERIC_RANGE_COLUMNS:
            if column not in fieldnames:
                continue
            values = [parse_float(row.get(column)) for row in rows]
            valid = [value for value in values if value is not None]
            summary = summarize_numeric(valid)
            output.append(
                {
                    "variable": column,
                    "split": split_name,
                    "valid_count": len(valid),
                    "missing_or_nonfinite": len(rows) - len(valid),
                    "min": summary["min"],
                    "p50": summary["p50"],
                    "p90": summary["p90"],
                    "p95": summary["p95"],
                    "p99": summary["p99"],
                    "max": summary["max"],
                    "mean": summary["mean"],
                    "std": summary["std"],
                }
            )
    return output


def build_missingness_rows(split_rows: dict[str, list[dict[str, str]]], fieldnames: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for split_name, rows in split_rows.items():
        total = len(rows)
        for column in fieldnames:
            missing = sum(1 for row in rows if str(row.get(column, "")).strip() == "")
            output.append(
                {
                    "split": split_name,
                    "column": column,
                    "rows": total,
                    "missing_count": missing,
                    "missing_fraction": missing / total if total else None,
                }
            )
    return output


def build_duplicate_key_rows(split_rows: dict[str, list[dict[str, str]]], max_examples: int) -> list[dict[str, Any]]:
    builders = {
        "sample_id": lambda row: row.get("sample_id", ""),
        "scenario_signature": lambda row: row.get(WAVE_COLUMN, "") + "||" + split_signature(row, STRUCTURE_SIGNATURE_COLUMNS),
        "image_path": lambda row: row.get(IMAGE_COLUMN, ""),
    }
    output: list[dict[str, Any]] = []
    for key_name, builder in builders.items():
        for split_name, rows in split_rows.items():
            values = [builder(row) for row in rows]
            values = [value for value in values if value]
            counts = Counter(values)
            duplicate_items = [(value, count) for value, count in counts.items() if count > 1]
            duplicate_extra_rows = sum(count - 1 for _, count in duplicate_items)
            examples = [f"{value} (n={count})" for value, count in sorted(duplicate_items)[:max_examples]]
            output.append(
                {
                    "key": key_name,
                    "split": split_name,
                    "rows": len(rows),
                    "nonempty_key_rows": len(values),
                    "unique_keys": len(counts),
                    "duplicate_key_count": len(duplicate_items),
                    "duplicate_extra_rows": duplicate_extra_rows,
                    "examples": " ; ".join(examples),
                }
            )
    return output


def risk_level(group_name: str, train_test_overlap: int, all_three_overlap: int) -> str:
    if train_test_overlap == 0 and all_three_overlap == 0:
        return "low"
    if group_name in {"sample_id", "scenario_signature"}:
        return "high"
    if group_name in {"txt_path", "image_path", "structure_signature"}:
        return "requires_task_definition"
    return "review"


def build_group_sets(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, set[str]]]:
    builders = {
        "sample_id": lambda row: row.get("sample_id", ""),
        "txt_path": lambda row: row.get(WAVE_COLUMN, ""),
        "image_path": lambda row: row.get(IMAGE_COLUMN, ""),
        "wave_cluster": lambda row: row.get("wave_cluster", ""),
        "structure_signature": lambda row: split_signature(row, STRUCTURE_SIGNATURE_COLUMNS),
        "scenario_signature": lambda row: row.get(WAVE_COLUMN, "") + "||" + split_signature(row, STRUCTURE_SIGNATURE_COLUMNS),
    }
    group_sets: dict[str, dict[str, set[str]]] = {name: {} for name in builders}
    for group_name, builder in builders.items():
        for split_name, rows in split_rows.items():
            values = {builder(row) for row in rows}
            values.discard("")
            group_sets[group_name][split_name] = values
    return group_sets


def build_overlap_rows(split_rows: dict[str, list[dict[str, str]]], max_examples: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group_name, sets in build_group_sets(split_rows).items():
        train = sets.get("train", set())
        val = sets.get("val", set())
        test = sets.get("test", set())
        train_val = train & val
        train_test = train & test
        val_test = val & test
        all_three = train & val & test
        examples = sorted(train_test or all_three or train_val or val_test)[:max_examples]
        output.append(
            {
                "group": group_name,
                "train_unique": len(train),
                "val_unique": len(val),
                "test_unique": len(test),
                "train_val_overlap": len(train_val),
                "train_test_overlap": len(train_test),
                "val_test_overlap": len(val_test),
                "all_three_overlap": len(all_three),
                "risk_level": risk_level(group_name, len(train_test), len(all_three)),
                "examples": " ; ".join(examples),
            }
        )
    return output


def build_filtering_audit(protocol: dict[str, Any], manifest: dict[str, Any], tail_rows: list[dict[str, Any]]) -> str:
    dataset_base = protocol["dataset"]["dataset_base"]
    filename_flags = {
        "contains_drop": "drop" in dataset_base.lower(),
        "contains_rank": "rank" in dataset_base.lower(),
        "contains_top": "top" in dataset_base.lower(),
        "contains_tailfix": "tailfix" in dataset_base.lower(),
    }
    lines = [
        "# TODO2 Filtering and Distribution Audit",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Dataset base: `{dataset_base}`",
        f"- Target column: `{TARGET_COLUMN}`",
        "",
        "## Filename Clues",
        "",
    ]
    for key, value in filename_flags.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "Interpretation: filename clues are not proof of filtering. They only flag what must be verified against the generation script and source manifest.",
            "",
            "## Split Target/Tail Summary",
            "",
            "| Split | Rows | p90 | p95 | p99 | max | >=0.005 | >=0.010 | >=0.015 | >=0.020 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in tail_rows:
        lines.append(
            "| {split} | {rows} | {p90} | {p95} | {p99} | {maxv} | {ge005} | {ge010} | {ge015} | {ge020} |".format(
                split=row["split"],
                rows=row["rows"],
                p90=format_float(row["p90"]),
                p95=format_float(row["p95"]),
                p99=format_float(row["p99"]),
                maxv=format_float(row["max"]),
                ge005=row["count_ge_0.005"],
                ge010=row["count_ge_0.010"],
                ge015=row["count_ge_0.015"],
                ge020=row["count_ge_0.020"],
            )
        )
    lines.extend(
        [
            "",
            "## Group Overlap Summary",
            "",
            "| Group | Train unique | Val unique | Test unique | Train-Test overlap | All-three overlap | Risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in manifest.get("group_overlap_summary", []):
        lines.append(
            f"| {row['group']} | {row['train_unique']} | {row['val_unique']} | {row['test_unique']} | "
            f"{row['train_test_overlap']} | {row['all_three_overlap']} | {row['risk_level']} |"
        )
    lines.extend(
        [
            "",
            "## Duplicate Key Summary",
            "",
            "| Key | Split | Unique keys | Duplicate key count | Duplicate extra rows |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in manifest.get("duplicate_key_summary", []):
        if row["duplicate_extra_rows"] or row["key"] in {"sample_id", "scenario_signature"}:
            lines.append(
                f"| {row['key']} | {row['split']} | {row['unique_keys']} | "
                f"{row['duplicate_key_count']} | {row['duplicate_extra_rows']} |"
            )
    lines.extend(
        [
            "",
            "## Key Missingness Summary",
            "",
            "| Split | Column | Missing | Rows | Fraction |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for split_name, split_manifest in manifest.get("split_manifests", {}).items():
        rows = split_manifest["rows"]
        missing_by_column = split_manifest.get("missing_by_column", {})
        for column in (IMAGE_COLUMN, "yield_margin", "failure_reason", "analysis_failed_time"):
            if column in missing_by_column and missing_by_column[column]:
                fraction = missing_by_column[column] / rows if rows else None
                lines.append(f"| {split_name} | {column} | {missing_by_column[column]} | {rows} | {format_float(fraction)} |")
    lines.extend(
        [
            "",
            "## Required Human Review Before Safety Claims",
            "",
            "- Verify whether any top-drift or failure samples were removed before this split.",
            "- Verify whether `analysis_status != ok` samples were removed upstream.",
            "- Verify whether high-drift and yielded samples are sufficient for claims about engineering safety.",
            "- If train/val/test have strongly different tail coverage, do not make broad tail-reliability claims until a high-drift stress benchmark is built.",
            "",
            "## Next TODO",
            "",
            "Use this report to decide the exact grouping variables for TODO10 unseen-wave / unseen-structure / high-drift benchmarks.",
            "",
        ]
    )
    return "\n".join(lines)


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return str(value)


def make_target_svg(path: Path, split_rows: dict[str, list[dict[str, str]]], bins: int = 30) -> None:
    values_by_split: dict[str, list[float]] = {}
    all_values: list[float] = []
    for split_name, rows in split_rows.items():
        values = [parse_float(row.get(TARGET_COLUMN)) for row in rows]
        valid = [value for value in values if value is not None]
        values_by_split[split_name] = valid
        all_values.extend(valid)
    if not all_values:
        return
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        max_value = min_value + 1.0
    width = 980
    height = 520
    margin_left = 70
    margin_bottom = 70
    plot_width = width - margin_left - 30
    plot_height = height - 60 - margin_bottom
    colors = {"train": "#4477aa", "val": "#cc6677", "test": "#228833"}
    split_order = ["train", "val", "test"]
    histograms: dict[str, list[int]] = {}
    max_count = 1
    for split_name, values in values_by_split.items():
        counts = [0] * bins
        for value in values:
            index = int((value - min_value) / (max_value - min_value) * bins)
            index = max(0, min(bins - 1, index))
            counts[index] += 1
        histograms[split_name] = counts
        max_count = max(max_count, max(counts) if counts else 1)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="32" font-family="Arial" font-size="20" font-weight="700">Target distribution by split</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 30}" y2="{height - margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="60" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222"/>',
    ]
    bar_width = plot_width / bins / 4.0
    for split_idx, split_name in enumerate(split_order):
        counts = histograms.get(split_name, [])
        color = colors.get(split_name, "#666666")
        for idx, count in enumerate(counts):
            x = margin_left + idx * (plot_width / bins) + split_idx * bar_width
            bar_h = count / max_count * plot_height
            y = height - margin_bottom - bar_h
            lines.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_h:.2f}" fill="{color}" opacity="0.78"/>'
            )
    for split_idx, split_name in enumerate(split_order):
        x = margin_left + 20 + split_idx * 120
        y = height - 28
        color = colors.get(split_name, "#666666")
        lines.append(f'<rect x="{x}" y="{y - 12}" width="16" height="12" fill="{color}"/>')
        lines.append(f'<text x="{x + 24}" y="{y}" font-family="Arial" font-size="14">{split_name}</text>')
    lines.extend(
        [
            f'<text x="{margin_left}" y="{height - 45}" font-family="Arial" font-size="12">{min_value:.4g}</text>',
            f'<text x="{width - 120}" y="{height - 45}" font-family="Arial" font-size="12">{max_value:.4g}</text>',
            f'<text x="{width / 2 - 80:.1f}" y="{height - 20}" font-family="Arial" font-size="14">max_drift_ratio_raw</text>',
            '<text x="14" y="210" transform="rotate(-90 14,210)" font-family="Arial" font-size="14">count</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_audit(protocol_path: Path, output_root: Path, max_examples: int) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    output_root.mkdir(parents=True, exist_ok=True)
    data_audit_dir = output_root / "data_audit"
    figure_dir = output_root / "paper_figures"
    data_audit_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    split_paths = split_paths_from_protocol(protocol)
    split_rows: dict[str, list[dict[str, str]]] = {}
    fieldnames_by_split: dict[str, list[str]] = {}
    for split_name, path in split_paths.items():
        rows, fieldnames = read_split_csv(split_name, path)
        split_rows[split_name] = rows
        fieldnames_by_split[split_name] = fieldnames

    common_fieldnames = list(fieldnames_by_split.get("train") or next(iter(fieldnames_by_split.values())))
    split_manifests = {
        split_name: build_split_manifest(split_name, split_paths[split_name], rows, fieldnames_by_split[split_name])
        for split_name, rows in split_rows.items()
    }
    tail_rows = build_target_tail_rows(split_rows)
    distribution_rows = build_distribution_rows(split_rows, common_fieldnames)
    numeric_range_rows = build_numeric_range_rows(split_rows, common_fieldnames)
    missingness_rows = build_missingness_rows(split_rows, common_fieldnames)
    duplicate_key_rows = build_duplicate_key_rows(split_rows, max_examples=max_examples)
    overlap_rows = build_overlap_rows(split_rows, max_examples=max_examples)

    manifest = {
        "audit_name": "publication_dataset_audit_todo2",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_json": str(protocol_path),
        "output_root": str(output_root),
        "non_interference_policy": [
            "No training, testing, or Optuna entrypoints are executed.",
            "No running Python processes are stopped or modified.",
            "Only train/val/test CSV files from protocol_lock.json are read.",
            "Only publication_eval data_audit and paper_figures outputs are written.",
        ],
        "dataset_base": protocol["dataset"]["dataset_base"],
        "target_column": TARGET_COLUMN,
        "split_manifests": split_manifests,
        "target_tail_summary": tail_rows,
        "group_overlap_summary": overlap_rows,
        "duplicate_key_summary": duplicate_key_rows,
    }

    (data_audit_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        data_audit_dir / "target_tail_table.csv",
        tail_rows,
        [
            "split",
            "rows",
            "valid_target_count",
            "missing_or_nonfinite_target",
            "min",
            "p50",
            "p75",
            "p90",
            "p95",
            "p99",
            "max",
            "mean",
            "std",
            "count_ge_0.005",
            "frac_ge_0.005",
            "count_ge_0.010",
            "frac_ge_0.010",
            "count_ge_0.015",
            "frac_ge_0.015",
            "count_ge_0.020",
            "frac_ge_0.020",
            "threshold_p90",
            "count_ge_p90",
            "threshold_p95",
            "count_ge_p95",
        ],
    )
    write_csv(
        data_audit_dir / "split_distribution.csv",
        distribution_rows,
        ["variable", "split", "value", "count", "fraction"],
    )
    write_csv(
        data_audit_dir / "numeric_range_summary.csv",
        numeric_range_rows,
        ["variable", "split", "valid_count", "missing_or_nonfinite", "min", "p50", "p90", "p95", "p99", "max", "mean", "std"],
    )
    write_csv(
        data_audit_dir / "missingness_summary.csv",
        missingness_rows,
        ["split", "column", "rows", "missing_count", "missing_fraction"],
    )
    write_csv(
        data_audit_dir / "duplicate_key_report.csv",
        duplicate_key_rows,
        ["key", "split", "rows", "nonempty_key_rows", "unique_keys", "duplicate_key_count", "duplicate_extra_rows", "examples"],
    )
    write_csv(
        data_audit_dir / "group_overlap_report.csv",
        overlap_rows,
        [
            "group",
            "train_unique",
            "val_unique",
            "test_unique",
            "train_val_overlap",
            "train_test_overlap",
            "val_test_overlap",
            "all_three_overlap",
            "risk_level",
            "examples",
        ],
    )
    (data_audit_dir / "filtering_audit.md").write_text(
        build_filtering_audit(protocol, manifest, tail_rows),
        encoding="utf-8",
    )
    make_target_svg(figure_dir / "target_distribution_by_split.svg", split_rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the locked publication dataset for TODO2.")
    parser.add_argument("--protocol-json", default=str(DEFAULT_PROTOCOL_JSON), help="Path to protocol_lock.json.")
    parser.add_argument("--output-root", default=None, help="Publication evaluation output root. Defaults to protocol parent parent.")
    parser.add_argument("--max-examples", type=int, default=5, help="Maximum overlap examples per group.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol_json).resolve()
    if args.output_root is None:
        output_root = protocol_path.parents[1]
    else:
        output_root = Path(args.output_root).resolve()
    manifest = build_audit(protocol_path, output_root, max_examples=int(args.max_examples))
    print("TODO2 dataset audit generated without running training.")
    print(f"Output root: {output_root}")
    for split_name, split_manifest in manifest["split_manifests"].items():
        rows = split_manifest["rows"]
        sha = split_manifest["sha256"]
        unique_waves = split_manifest["unique_counts"]["txt_path"]
        print(f"{split_name}: rows={rows}, unique_waves={unique_waves}, sha256={sha[:12] if sha else '-'}")


if __name__ == "__main__":
    main()
