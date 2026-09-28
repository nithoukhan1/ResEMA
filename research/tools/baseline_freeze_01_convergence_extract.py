#!/usr/bin/env python3
"""Generate full-trajectory convergence evidence for BASELINE-FREEZE-01A.

Reads only canonical final results.csv files from the ignored local archive.
No model, dataset, checkpoint, training, validation, or test access occurs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

MAP95 = "metrics/mAP50-95(B)"
MAP50 = "metrics/mAP50(B)"
PREC = "metrics/precision(B)"
REC = "metrics/recall(B)"


def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    denom = sum((x - xm) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denom


def load_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"epoch", MAP95, MAP50, PREC, REC}
    if not rows or not required.issubset(rows[0]):
        raise RuntimeError(f"Missing required columns in {path}")
    return rows


def find_run_dir(root: Path, experiment_id: str) -> Path:
    matches = [
        p for p in root.rglob(experiment_id)
        if p.is_dir() and p.parent.name == "ResEMA_baseline_runs"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one run dir for {experiment_id}, found {len(matches)}"
        )
    return matches[0]


def summarize(rows: list[dict[str, str]], experiment_id: str) -> dict[str, Any]:
    epochs = [int(float(r["epoch"])) for r in rows]
    map95 = [float(r[MAP95]) for r in rows]
    map50 = [float(r[MAP50]) for r in rows]
    precision = [float(r[PREC]) for r in rows]
    recall = [float(r[REC]) for r in rows]

    best_i = max(range(len(rows)), key=lambda i: map95[i])
    out: dict[str, Any] = {
        "experiment_id": experiment_id,
        "epochs": len(rows),
        "first_epoch": epochs[0],
        "final_epoch": epochs[-1],
        "best_epoch": epochs[best_i],
        "best_map50_95": map95[best_i],
        "best_map50": map50[best_i],
        "best_precision": precision[best_i],
        "best_recall": recall[best_i],
        "final_map50_95": map95[-1],
        "best_minus_final_map50_95": map95[best_i] - map95[-1],
    }

    for window in (5, 10, 20, 30, 50):
        n = min(window, len(rows))
        xs = [float(x) for x in epochs[-n:]]
        ys = map95[-n:]
        out[f"last{window}_count"] = n
        out[f"last{window}_mean_map50_95"] = sum(ys) / n
        out[f"last{window}_min_map50_95"] = min(ys)
        out[f"last{window}_max_map50_95"] = max(ys)
        out[f"last{window}_slope_per_epoch"] = linear_slope(xs, ys)

    if len(rows) >= 2:
        out["final_minus_previous_map50_95"] = map95[-1] - map95[-2]
    if len(rows) >= 10:
        xs = [float(x) for x in epochs[-10:-1]]
        ys = map95[-10:-1]
        out["last9_excluding_final_slope_per_epoch"] = linear_slope(xs, ys)
        out["final_minus_last9_mean_map50_95"] = map95[-1] - (sum(ys) / len(ys))

    # Diagnostic only; not an automatic experimental decision.
    near_boundary = epochs[best_i] >= epochs[-1] - 5
    out["best_epoch_within_last_5"] = near_boundary
    out["requires_manual_budget_review"] = near_boundary
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    registry_path = repo / "research" / "05_experiments" / "BASELINE_FREEZE_01_INPUTS.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))

    archive_root = repo / "artifacts_external" / "baseline_freeze_01"
    manifests = archive_root / "manifests"
    report_dir = archive_root / "freeze_report"
    manifests.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    canonical = [x for x in registry["sessions"] if x["canonical_final"]]
    epoch_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for entry in canonical:
        session_root = (
            archive_root / "sessions" / Path(entry["archive"]).stem / "extracted"
        )
        run_dir = find_run_dir(session_root, entry["experiment_id"])
        results_path = run_dir / "results.csv"
        rows = load_results(results_path)

        if len(rows) != 100 or int(float(rows[-1]["epoch"])) != 100:
            raise RuntimeError(
                f"Canonical run is not 100 epochs: {entry['experiment_id']}"
            )

        summaries.append(summarize(rows, entry["experiment_id"]))

        for r in rows:
            p = float(r[PREC])
            rec = float(r[REC])
            f1 = 0.0 if p + rec == 0 else 2.0 * p * rec / (p + rec)
            epoch_rows.append(
                {
                    "experiment_id": entry["experiment_id"],
                    "epoch": int(float(r["epoch"])),
                    "precision": p,
                    "recall": rec,
                    "f1": f1,
                    "mAP50": float(r[MAP50]),
                    "mAP50_95": float(r[MAP95]),
                    "train_box_loss": r.get("train/box_loss"),
                    "train_cls_loss": r.get("train/cls_loss"),
                    "train_dfl_loss": r.get("train/dfl_loss"),
                    "val_box_loss": r.get("val/box_loss"),
                    "val_cls_loss": r.get("val/cls_loss"),
                    "val_dfl_loss": r.get("val/dfl_loss"),
                }
            )

    epoch_csv = manifests / "CANONICAL_EPOCH_METRICS.csv"
    with epoch_csv.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "experiment_id", "epoch", "precision", "recall", "f1",
            "mAP50", "mAP50_95", "train_box_loss", "train_cls_loss",
            "train_dfl_loss", "val_box_loss", "val_cls_loss", "val_dfl_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(epoch_rows)

    diag_csv = manifests / "CONVERGENCE_DIAGNOSTICS.csv"
    fields = sorted({key for row in summaries for key in row})
    with diag_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)

    report_path = report_dir / "BASELINE_FREEZE_01_CONVERGENCE.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "BASELINE-FREEZE-01-convergence-v1.0",
                "canonical_experiment_count": len(canonical),
                "epoch_row_count": len(epoch_rows),
                "training_started": False,
                "validation_started": False,
                "test_access": False,
                "summaries": summaries,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("=" * 100)
    print("BASELINE-FREEZE-01 CONVERGENCE EXTRACTION")
    print("=" * 100)
    print(f"CANONICAL_EXPERIMENTS={len(canonical)}")
    print(f"EPOCH_ROWS={len(epoch_rows)}")
    print(f"EPOCH_METRICS={epoch_csv}")
    print(f"DIAGNOSTICS={diag_csv}")
    print(f"REPORT={report_path}")
    print("TRAINING_STARTED=FALSE")
    print("VALIDATION_STARTED=FALSE")
    print("TEST_ACCESS=FALSE")
    print("=" * 100)

    return 0 if len(canonical) == 6 and len(epoch_rows) == 600 else 2


if __name__ == "__main__":
    raise SystemExit(main())
