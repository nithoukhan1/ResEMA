#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--repo-a2", type=Path, required=True)
    parser.add_argument("--repo-a3", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, default=Path("/kaggle/working/BASELINE_FREEZE_01_PERCLASS"))
    args = parser.parse_args()

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    if args.output_root.exists():
        raise RuntimeError(f"Refusing pre-existing batch output root: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=False)

    child_results: list[dict[str, Any]] = []
    for exp in registry["experiments"]:
        repo = args.repo_a2 if exp["runtime_group"] == "A2" else args.repo_a3
        cmd = [
            sys.executable,
            str(args.evaluator),
            "--experiment-id", exp["experiment_id"],
            "--repo-root", str(repo),
            "--registry", str(args.registry),
            "--input-root", str(args.input_root),
            "--output-root", str(args.output_root),
        ]
        print("\n" + "#" * 110)
        print("RUNNING", exp["experiment_id"], "WITH", exp["runtime_group"])
        print("#" * 110, flush=True)
        proc = subprocess.run(cmd)
        child_results.append(
            {
                "experiment_id": exp["experiment_id"],
                "runtime_group": exp["runtime_group"],
                "return_code": proc.returncode,
            }
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Validation-only child failed: {exp['experiment_id']} rc={proc.returncode}"
            )

    combined_rows: list[dict[str, Any]] = []
    aggregate_rows: list[dict[str, Any]] = []
    for exp in registry["experiments"]:
        exp_dir = args.output_root / exp["experiment_id"]
        summary_path = exp_dir / "VALIDATION_ONLY_SUMMARY.json"
        per_class_path = exp_dir / "PER_CLASS_METRICS.csv"
        if not summary_path.is_file() or not per_class_path.is_file():
            raise RuntimeError(f"Missing standardized output for {exp['experiment_id']}")

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        agg = summary["aggregate"]
        aggregate_rows.append(
            {
                "experiment_id": exp["experiment_id"],
                "runtime_group": exp["runtime_group"],
                "source_commit": exp["source_commit"],
                "execution_commit": exp["execution_commit"],
                "data_binding": exp["data_binding"],
                "initialization": exp["initialization"],
                "checkpoint_sha256": exp["checkpoint_sha256"],
                "precision": agg["precision"],
                "recall": agg["recall"],
                "f1_from_mean_pr": agg["f1_from_mean_pr"],
                "mean_class_f1": agg["mean_class_f1"],
                "map50": agg["map50"],
                "map50_95": agg["map50_95"],
                "evaluation_class_count": agg["evaluation_class_count"],
                "summary_sha256": sha256_file(summary_path),
                "per_class_csv_sha256": sha256_file(per_class_path),
            }
        )

        with per_class_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                combined_rows.append(
                    {
                        "experiment_id": exp["experiment_id"],
                        **row,
                    }
                )

    aggregate_path = args.output_root / "STANDARDIZED_VALIDATION_AGGREGATE.csv"
    with aggregate_path.open("w", encoding="utf-8", newline="") as f:
        fields = list(aggregate_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(aggregate_rows)

    per_class_combined = args.output_root / "STANDARDIZED_PER_CLASS_ALL_EXPERIMENTS.csv"
    with per_class_combined.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "experiment_id","class_id","class_name","support_images",
            "support_instances","precision","recall","f1","ap50","ap50_95","status",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(combined_rows)

    batch_summary = {
        "schema_version":"BASELINE-FREEZE-01-standardized-val-batch-v1.0",
        "experiment_count":len(registry["experiments"]),
        "successful_children":sum(x["return_code"] == 0 for x in child_results),
        "training_started":False,
        "test_access":"NONE",
        "child_results":child_results,
        "aggregate_csv":{
            "path":str(aggregate_path),
            "sha256":sha256_file(aggregate_path),
        },
        "per_class_csv":{
            "path":str(per_class_combined),
            "sha256":sha256_file(per_class_combined),
        },
    }
    batch_summary_path = args.output_root / "STANDARDIZED_VALIDATION_BATCH_SUMMARY.json"
    batch_summary_path.write_text(
        json.dumps(batch_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 110)
    print("BASELINE-FREEZE-01 STANDARDIZED VALIDATION BATCH COMPLETE")
    print("=" * 110)
    print(f"EXPERIMENT_COUNT={len(registry['experiments'])}")
    print(f"SUCCESSFUL_CHILDREN={batch_summary['successful_children']}")
    print(f"AGGREGATE={aggregate_path}")
    print(f"PER_CLASS_ALL={per_class_combined}")
    print(f"BATCH_SUMMARY={batch_summary_path}")
    print("TRAINING_STARTED=FALSE")
    print("TEST_ACCESS=NONE")
    print("=" * 110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
