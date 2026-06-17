# -*- coding: utf-8 -*-
"""Regenerate the publication HPO summary/report from saved status files."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import run_publication_hpo as runner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize a publication HPO batch report.")
    parser.add_argument("--batch-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    batch_dir = args.batch_dir.resolve()
    manifest = runner.read_json(batch_dir / "manifest.json") or {}
    dataset = manifest.get("dataset") or runner.dataset_manifest()

    statuses = []
    for job in runner.MODEL_JOBS:
        status = runner.read_json(batch_dir / "status" / f"{job.name}.json")
        if status is not None:
            statuses.append(status)

    summary = runner.summarize_statuses(statuses)
    summary.update(
        {
            "batch_dir": str(batch_dir),
            "started_at": manifest.get("started_at"),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": dataset,
        }
    )
    runner.write_json(batch_dir / "publication_hpo_summary.json", summary)
    runner.snapshot_outputs(batch_dir / "after_snapshot", tuple(job.name for job in runner.MODEL_JOBS))
    report_path = runner.write_report(batch_dir, dataset, statuses)
    print(f">>> Finalized report: {report_path}")
    return 0 if summary["failed_count"] == 0 and len(statuses) == len(runner.MODEL_JOBS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
