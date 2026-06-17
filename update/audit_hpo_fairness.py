"""Audit Optuna HPO fairness without starting any optimization/training.

This script implements TODO 3 from the publication checklist. It reads existing
Optuna artifacts and run status JSON files, then writes publication-ready HPO
audit tables under publication_eval_*/hpo_audit.
"""

from __future__ import annotations

import ast
import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL_JSON = PROJECT_ROOT / "publication_eval_20260614" / "protocol" / "protocol_lock.json"
MODEL_ORDER = ("randomforest", "xgboost", "lightgbm", "catboost", "mlp", "lstm", "wavenet", "2dcnn")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_float(value: Any) -> float | None:
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


def safe_literal_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def protocol_jobs(protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        job["model"]: job
        for job in protocol.get("hpo_protocol", {}).get("jobs", [])
    }


def find_status(project_root: Path, model: str, study_name: str | None) -> tuple[dict[str, Any] | None, Path | None]:
    status_paths = sorted((project_root / "update" / "run_logs").glob(f"**/status/{model}.json"))
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in status_paths:
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        nested_study = (data.get("study_summary") or {}).get("study_name")
        if study_name and nested_study == study_name:
            return data, path
        candidates.append((path.stat().st_mtime, path, data))
    if not candidates:
        return None, None
    _, path, data = max(candidates, key=lambda item: item[0])
    return data, path


def summarize_trials(model: str, rows: list[dict[str, str]]) -> dict[str, Any]:
    state_counts = Counter(row.get("state", "<missing>") or "<empty>" for row in rows)
    complete_values: list[tuple[int, float]] = []
    convergence_rows: list[dict[str, Any]] = []
    best_so_far: float | None = None
    best_trial_so_far: int | None = None
    for fallback_number, row in enumerate(rows):
        number = int(parse_float(row.get("number")) or fallback_number)
        value = parse_float(row.get("value"))
        state = row.get("state", "")
        if state == "COMPLETE" and value is not None:
            complete_values.append((number, value))
            if best_so_far is None or value < best_so_far:
                best_so_far = value
                best_trial_so_far = number
        convergence_rows.append(
            {
                "model": model,
                "trial_number": number,
                "state": state,
                "value": value,
                "best_value_so_far": best_so_far,
                "best_trial_so_far": best_trial_so_far,
            }
        )
    best_number = None
    best_value = None
    if complete_values:
        best_number, best_value = min(complete_values, key=lambda item: item[1])
    param_columns = [column for column in rows[0].keys() if column.startswith("params_")] if rows else []
    return {
        "row_count": len(rows),
        "state_counts": dict(sorted(state_counts.items())),
        "complete_values": complete_values,
        "best_trial_number_from_csv": best_number,
        "best_value_from_csv": best_value,
        "param_columns": param_columns,
        "convergence_rows": convergence_rows,
    }


def observed_search_space(model: str, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    param_columns = [column for column in rows[0].keys() if column.startswith("params_")]
    output: list[dict[str, Any]] = []
    for column in param_columns:
        param_name = column.removeprefix("params_")
        raw_values = [row.get(column, "") for row in rows if row.get(column, "") != ""]
        numeric_values = [parse_float(value) for value in raw_values]
        numeric_values = [value for value in numeric_values if value is not None]
        unique_values = sorted({str(value) for value in raw_values})
        if len(numeric_values) == len(raw_values) and raw_values:
            value_type = "numeric"
            observed_min = min(numeric_values)
            observed_max = max(numeric_values)
            examples = ""
        else:
            value_type = "categorical_or_mixed"
            observed_min = None
            observed_max = None
            examples = " ; ".join(unique_values[:8])
        output.append(
            {
                "model": model,
                "parameter": param_name,
                "observed_type": value_type,
                "observed_min": observed_min,
                "observed_max": observed_max,
                "unique_count": len(unique_values),
                "example_values": examples,
            }
        )
    return output


def extract_metric(summary: dict[str, Any], *names: str) -> Any:
    metrics = summary.get("best_metrics", {}) or {}
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def status_elapsed(status: dict[str, Any] | None) -> float | None:
    if not status:
        return None
    value = parse_float(status.get("elapsed_seconds"))
    return value


def format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


def fairness_flags(budget_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []
    trials = [int(row["actual_trials"]) for row in budget_rows if row.get("actual_trials") is not None]
    elapsed_values = [parse_float(row.get("elapsed_seconds")) for row in budget_rows]
    elapsed_values = [value for value in elapsed_values if value is not None and value > 0]
    if trials and max(trials) != min(trials):
        flags.append(
            {
                "severity": "major",
                "issue": "unequal_trial_budget",
                "details": f"Trial counts range from {min(trials)} to {max(trials)}; report or normalize budgets before strong model-ranking claims.",
            }
        )
    startup_ratios = [
        (row["model"], parse_float(row.get("actual_startup_trials")), parse_float(row.get("actual_trials")))
        for row in budget_rows
    ]
    high_startup = [
        f"{model}:{startup / trials:.2f}"
        for model, startup, trials in startup_ratios
        if startup is not None and trials and trials > 0 and startup / trials >= 0.35
    ]
    if high_startup:
        flags.append(
            {
                "severity": "moderate",
                "issue": "high_startup_fraction_for_low_trial_models",
                "details": "Startup/total ratios >=0.35 for " + ", ".join(high_startup),
            }
        )
    if elapsed_values and max(elapsed_values) / min(elapsed_values) > 20:
        flags.append(
            {
                "severity": "major",
                "issue": "large_wall_clock_imbalance",
                "details": f"Elapsed time ratio is {max(elapsed_values) / min(elapsed_values):.1f}x across models; report compute budgets separately.",
            }
        )
    mismatches = [
        row["model"]
        for row in budget_rows
        if row.get("protocol_trials") not in (None, "", row.get("actual_trials"))
        or row.get("protocol_startup_trials") not in (None, "", row.get("actual_startup_trials"))
    ]
    if mismatches:
        flags.append(
            {
                "severity": "major",
                "issue": "protocol_vs_status_budget_mismatch",
                "details": "Protocol and actual status disagree for: " + ", ".join(mismatches),
            }
        )
    flags.append(
        {
            "severity": "methodological",
            "issue": "validation_only_hpo_results",
            "details": "HPO best values are validation/model-selection results; final paper claims still require frozen-parameter multi-seed locked-test evaluation.",
        }
    )
    return flags


def make_convergence_svg(path: Path, convergence_rows: list[dict[str, Any]]) -> None:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in convergence_rows:
        if row.get("best_value_so_far") is not None:
            by_model.setdefault(row["model"], []).append(row)
    if not by_model:
        return
    width = 1040
    height = 560
    margin_left = 80
    margin_bottom = 70
    plot_width = width - margin_left - 40
    plot_height = height - 80 - margin_bottom
    colors = {
        "randomforest": "#4477aa",
        "xgboost": "#228833",
        "lightgbm": "#66ccee",
        "catboost": "#cc6677",
        "mlp": "#aa3377",
        "lstm": "#bbbb44",
        "wavenet": "#ee7733",
        "2dcnn": "#000000",
    }
    all_trials = [row["trial_number"] for rows in by_model.values() for row in rows]
    all_values = [float(row["best_value_so_far"]) for rows in by_model.values() for row in rows]
    max_trial = max(all_trials) if all_trials else 1
    min_value = min(all_values)
    max_value = max(all_values)
    if min_value == max_value:
        max_value = min_value + 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="80" y="34" font-family="Arial" font-size="20" font-weight="700">Optuna best-so-far validation objective</text>',
        f'<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - 40}" y2="{height - margin_bottom}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="70" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#222"/>',
    ]
    for model in MODEL_ORDER:
        rows = by_model.get(model, [])
        if not rows:
            continue
        points = []
        for row in rows:
            x = margin_left + (float(row["trial_number"]) / max(max_trial, 1)) * plot_width
            y = height - margin_bottom - ((float(row["best_value_so_far"]) - min_value) / (max_value - min_value)) * plot_height
            points.append(f"{x:.2f},{y:.2f}")
        lines.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors.get(model, "#666")}" stroke-width="2"/>'
        )
    legend_x = margin_left
    legend_y = height - 34
    for idx, model in enumerate(MODEL_ORDER):
        x = legend_x + idx * 118
        color = colors.get(model, "#666")
        lines.append(f'<rect x="{x}" y="{legend_y - 12}" width="14" height="10" fill="{color}"/>')
        lines.append(f'<text x="{x + 19}" y="{legend_y}" font-family="Arial" font-size="12">{model}</text>')
    lines.extend(
        [
            f'<text x="{margin_left}" y="{height - 45}" font-family="Arial" font-size="12">0</text>',
            f'<text x="{width - 72}" y="{height - 45}" font-family="Arial" font-size="12">{max_trial}</text>',
            f'<text x="{width / 2 - 45:.1f}" y="{height - 18}" font-family="Arial" font-size="14">trial</text>',
            '<text x="18" y="230" transform="rotate(-90 18,230)" font-family="Arial" font-size="14">best value so far</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_search_space_md(rows: list[dict[str, Any]]) -> str:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    lines = [
        "# Observed Optuna Search Space",
        "",
        "This file reports parameter values observed in completed Optuna trials. It is an audit artifact, not a substitute for the exact search-space definitions in each tuning script.",
        "",
    ]
    for model in MODEL_ORDER:
        model_rows = by_model.get(model, [])
        if not model_rows:
            continue
        lines.extend(
            [
                f"## {model}",
                "",
                "| Parameter | Type | Observed min | Observed max | Unique count | Example values |",
                "| --- | --- | ---: | ---: | ---: | --- |",
            ]
        )
        for row in model_rows:
            lines.append(
                f"| `{row['parameter']}` | {row['observed_type']} | {row.get('observed_min') if row.get('observed_min') is not None else '-'} | "
                f"{row.get('observed_max') if row.get('observed_max') is not None else '-'} | {row['unique_count']} | {row.get('example_values') or '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_report(
    budget_rows: list[dict[str, Any]],
    best_rows: list[dict[str, Any]],
    flags: list[dict[str, Any]],
) -> str:
    lines = [
        "# TODO3 HPO Fairness Audit",
        "",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Scope: existing Optuna artifacts only; no optimization or training was started.",
        "",
        "## Budget Summary",
        "",
        "| Model | Trials | Complete | Startup | Elapsed | sec/trial | Best trial | Best value |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in budget_rows:
        lines.append(
            f"| {row['model']} | {row['actual_trials']} | {row['complete_trials']} | {row['actual_startup_trials']} | "
            f"{format_seconds(parse_float(row.get('elapsed_seconds')))} | {row.get('seconds_per_complete_trial') or '-'} | "
            f"{row.get('best_trial_number')} | {row.get('best_value')} |"
        )
    lines.extend(["", "## Best Validation Metrics", "", "| Model | Selection | Val MAE | Val RMSE | Val R2 |", "| --- | ---: | ---: | ---: | ---: |"])
    for row in best_rows:
        lines.append(
            f"| {row['model']} | {row.get('best_value')} | {row.get('val_mae')} | {row.get('val_rmse')} | {row.get('val_r2')} |"
        )
    lines.extend(["", "## Fairness Flags", "", "| Severity | Issue | Details |", "| --- | --- | --- |"])
    for flag in flags:
        lines.append(f"| {flag['severity']} | {flag['issue']} | {flag['details']} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- All current HPO best values are validation/model-selection results, not final paper evidence.",
            "- Unequal trial counts and very different wall-clock costs must be reported transparently.",
            "- Final claims still require TODO5 multi-seed retraining and locked-test statistical comparison.",
            "",
        ]
    )
    return "\n".join(lines)


def build_audit(protocol_path: Path, output_root: Path, project_root: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    jobs = protocol_jobs(protocol)
    hpo_dir = output_root / "hpo_audit"
    figure_dir = output_root / "paper_figures"
    hpo_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    budget_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    time_rows: list[dict[str, Any]] = []
    search_rows: list[dict[str, Any]] = []
    convergence_rows: list[dict[str, Any]] = []

    for model in MODEL_ORDER:
        model_dir = project_root / "update" / model
        summary_path = model_dir / "study_summary.json"
        trials_path = model_dir / "optuna_trials.csv"
        best_params_path = model_dir / "best_params.json"
        summary = load_json(summary_path) if summary_path.exists() else {}
        trial_rows = read_csv_rows(trials_path)
        trial_summary = summarize_trials(model, trial_rows)
        status, status_path = find_status(project_root, model, summary.get("study_name"))
        elapsed = status_elapsed(status)
        complete_trials = int(summary.get("complete_trial_count") or trial_summary["state_counts"].get("COMPLETE", 0) or 0)
        actual_trials = int(summary.get("trial_count") or len(trial_rows))
        actual_startup = status.get("startup_trials") if status else None
        protocol_job = jobs.get(model, {})
        seconds_per_complete = elapsed / complete_trials if elapsed and complete_trials else None
        best_value = summary.get("best_value", trial_summary.get("best_value_from_csv"))
        best_trial_number = summary.get("best_trial_number", trial_summary.get("best_trial_number_from_csv"))

        budget_rows.append(
            {
                "model": model,
                "objective_metric": summary.get("objective_metric"),
                "protocol_trials": protocol_job.get("trials"),
                "actual_trials": actual_trials,
                "complete_trials": complete_trials,
                "trial_csv_rows": len(trial_rows),
                "protocol_startup_trials": protocol_job.get("startup_trials"),
                "actual_startup_trials": actual_startup,
                "status": status.get("status") if status else None,
                "returncode": status.get("returncode") if status else None,
                "elapsed_seconds": elapsed,
                "elapsed_human": format_seconds(elapsed),
                "seconds_per_complete_trial": round(seconds_per_complete, 3) if seconds_per_complete else None,
                "best_trial_number": best_trial_number,
                "best_value": best_value,
                "study_summary_path": str(summary_path),
                "trials_csv_path": str(trials_path),
                "best_params_path": str(best_params_path),
                "status_path": str(status_path) if status_path else None,
                "extra_args": " ".join(protocol_job.get("extra_args", [])),
            }
        )

        metrics = summary.get("best_metrics", {}) or {}
        best_rows.append(
            {
                "model": model,
                "best_trial_number": best_trial_number,
                "best_value": best_value,
                "val_mae": extract_metric(summary, "MAE", "best_val_mae_raw"),
                "val_rmse": extract_metric(summary, "RMSE"),
                "val_r2": extract_metric(summary, "R2"),
                "selection_score": extract_metric(summary, "SelectionScore", "best_val_selection_score"),
                "metrics_json": json.dumps(metrics, ensure_ascii=False, sort_keys=True),
                "best_params_json": json.dumps(summary.get("best_optuna_params", {}), ensure_ascii=False, sort_keys=True),
            }
        )

        total_rows = len(trial_rows)
        for state, count in trial_summary["state_counts"].items():
            outcome_rows.append(
                {
                    "model": model,
                    "state": state,
                    "count": count,
                    "fraction": count / total_rows if total_rows else None,
                }
            )
        time_rows.append(
            {
                "model": model,
                "started_at": status.get("started_at") if status else None,
                "finished_at": status.get("finished_at") if status else summary.get("updated_at"),
                "elapsed_seconds": elapsed,
                "elapsed_human": format_seconds(elapsed),
                "complete_trials": complete_trials,
                "seconds_per_complete_trial": round(seconds_per_complete, 3) if seconds_per_complete else None,
                "command": " ".join(status.get("command", [])) if status else None,
            }
        )
        search_rows.extend(observed_search_space(model, trial_rows))
        convergence_rows.extend(trial_summary["convergence_rows"])

    flags = fairness_flags(budget_rows)
    manifest = {
        "audit_name": "publication_hpo_fairness_audit_todo3",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "protocol_json": str(protocol_path),
        "non_interference_policy": [
            "No training, testing, or Optuna entrypoints are executed.",
            "Only existing study_summary.json, optuna_trials.csv, best_params.json, and status JSON files are read.",
            "Only publication_eval hpo_audit and paper_figures outputs are written.",
        ],
        "budget_summary": budget_rows,
        "best_trials": best_rows,
        "fairness_flags": flags,
    }

    (hpo_dir / "hpo_audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(
        hpo_dir / "hpo_budget_table.csv",
        budget_rows,
        [
            "model",
            "objective_metric",
            "protocol_trials",
            "actual_trials",
            "complete_trials",
            "trial_csv_rows",
            "protocol_startup_trials",
            "actual_startup_trials",
            "status",
            "returncode",
            "elapsed_seconds",
            "elapsed_human",
            "seconds_per_complete_trial",
            "best_trial_number",
            "best_value",
            "extra_args",
            "study_summary_path",
            "trials_csv_path",
            "best_params_path",
            "status_path",
        ],
    )
    write_csv(hpo_dir / "optuna_best_trials.csv", best_rows, ["model", "best_trial_number", "best_value", "val_mae", "val_rmse", "val_r2", "selection_score", "metrics_json", "best_params_json"])
    write_csv(hpo_dir / "hpo_trial_outcomes.csv", outcome_rows, ["model", "state", "count", "fraction"])
    write_csv(hpo_dir / "hpo_time_cost.csv", time_rows, ["model", "started_at", "finished_at", "elapsed_seconds", "elapsed_human", "complete_trials", "seconds_per_complete_trial", "command"])
    write_csv(hpo_dir / "observed_search_space.csv", search_rows, ["model", "parameter", "observed_type", "observed_min", "observed_max", "unique_count", "example_values"])
    write_csv(hpo_dir / "hpo_convergence_points.csv", convergence_rows, ["model", "trial_number", "state", "value", "best_value_so_far", "best_trial_so_far"])
    write_csv(hpo_dir / "hpo_fairness_flags.csv", flags, ["severity", "issue", "details"])
    (hpo_dir / "search_spaces.md").write_text(build_search_space_md(search_rows), encoding="utf-8")
    (hpo_dir / "hpo_fairness_audit.md").write_text(build_report(budget_rows, best_rows, flags), encoding="utf-8")
    make_convergence_svg(figure_dir / "hpo_convergence_by_model.svg", convergence_rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit existing Optuna HPO artifacts for TODO3.")
    parser.add_argument("--protocol-json", default=str(DEFAULT_PROTOCOL_JSON), help="Path to protocol_lock.json.")
    parser.add_argument("--output-root", default=None, help="Publication output root. Defaults to protocol parent parent.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Repository root.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol_path = Path(args.protocol_json).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else protocol_path.parents[1]
    project_root = Path(args.project_root).resolve()
    manifest = build_audit(protocol_path, output_root, project_root)
    print("TODO3 HPO fairness audit generated without running training or Optuna.")
    print(f"Output root: {output_root}")
    for row in manifest["budget_summary"]:
        print(
            f"{row['model']}: complete={row['complete_trials']}/{row['actual_trials']}, "
            f"best={row['best_value']}, elapsed={row['elapsed_human']}"
        )


if __name__ == "__main__":
    main()
