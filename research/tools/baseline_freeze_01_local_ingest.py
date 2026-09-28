#!/usr/bin/env python3
"""BASELINE-FREEZE-01 local artifact ingestion.

Read-only with respect to source ZIPs and scientific run contents.
It copies source ZIPs into an ignored local archive, safely extracts them,
hashes all files, and produces compact manifests for review.

It does NOT:
- train or validate a model;
- access dataset test partitions;
- modify scientific run files;
- update EXPERIMENTS.csv;
- commit or push Git changes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any


METRIC_MAP95 = "metrics/mAP50-95(B)"
METRIC_MAP50 = "metrics/mAP50(B)"
METRIC_P = "metrics/precision(B)"
METRIC_R = "metrics/recall(B)"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_extract(zf: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in zf.infolist():
        target = (destination / member.filename).resolve()
        if destination != target and destination not in target.parents:
            raise RuntimeError(f"Unsafe ZIP member path: {member.filename}")
    zf.extractall(destination)


def find_run_dir(root: Path, experiment_id: str) -> Path | None:
    candidates = [
        p for p in root.rglob(experiment_id)
        if p.is_dir() and p.parent.name == "ResEMA_baseline_runs"
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise RuntimeError(
        f"Ambiguous run directory for {experiment_id}: "
        + ", ".join(str(p) for p in candidates)
    )


def find_summary(root: Path, experiment_id: str) -> Path | None:
    matches = list(root.rglob(f"{experiment_id}_FINAL_VALIDATION_SUMMARY.json"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    raise RuntimeError(
        f"Ambiguous final validation summary for {experiment_id}: "
        + ", ".join(str(p) for p in matches)
    )


def read_results(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    state: dict[str, Any] = {
        "exists": True,
        "rows": len(rows),
        "sha256": sha256_file(path),
    }
    if not rows:
        return state

    try:
        state["last_epoch"] = int(float(rows[-1]["epoch"]))
    except Exception:
        state["last_epoch"] = rows[-1].get("epoch")

    required = {METRIC_MAP95, METRIC_MAP50, METRIC_P, METRIC_R}
    if required.issubset(rows[0]):
        values = [float(row[METRIC_MAP95]) for row in rows]
        best_index = max(range(len(values)), key=values.__getitem__)
        best = rows[best_index]
        p = float(best[METRIC_P])
        r = float(best[METRIC_R])
        f1 = 0.0 if p + r == 0 else 2.0 * p * r / (p + r)
        state["best_index_zero_based"] = best_index
        state["best_epoch"] = int(float(best["epoch"]))
        state["best_precision"] = p
        state["best_recall"] = r
        state["best_f1"] = f1
        state["best_map50"] = float(best[METRIC_MAP50])
        state["best_map50_95"] = float(best[METRIC_MAP95])
        state["best_row"] = best

        tail = rows[max(0, len(rows) - 10):]
        state["last_10_epochs"] = [
            {
                "epoch": int(float(row["epoch"])),
                "map50_95": float(row[METRIC_MAP95]),
                "map50": float(row[METRIC_MAP50]),
                "precision": float(row[METRIC_P]),
                "recall": float(row[METRIC_R]),
            }
            for row in tail
        ]

    return state


def full_file_manifest(extracted_root: Path, out_csv: Path) -> tuple[int, int, str]:
    rows: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(p for p in extracted_root.rglob("*") if p.is_file()):
        rel = path.relative_to(extracted_root).as_posix()
        size = path.stat().st_size
        digest = sha256_file(path)
        rows.append({"relative_path": rel, "bytes": size, "sha256": digest})
        total_bytes += size

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["relative_path", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows), total_bytes, sha256_file(out_csv)


def governance_state(run_dir: Path) -> dict[str, Any]:
    gov = run_dir / "governance"
    pf_path = gov / "PRETRAIN_PREFLIGHT.json"
    rt_path = gov / "RUNTIME_MANIFEST.json"

    state: dict[str, Any] = {
        "preflight_exists": pf_path.is_file(),
        "runtime_exists": rt_path.is_file(),
    }

    if pf_path.is_file():
        pf = read_json(pf_path)
        state["preflight_sha256"] = sha256_file(pf_path)
        state["training_source_commit"] = pf.get("training_source_commit")
        state["execution_commit"] = pf.get("execution_commit")
        state["data_binding"] = pf.get("data_binding")
        state["initialization"] = pf.get("initialization")
        state["test_access"] = pf.get("test_access")

    if rt_path.is_file():
        rt = read_json(rt_path)
        state["runtime_sha256"] = sha256_file(rt_path)
        state["runtime_training_source_commit"] = rt.get("training_source_commit")
        state["runtime_execution_commit"] = rt.get("execution_commit")
        state["runtime_test_split_present"] = rt.get("test_split_present")

    sessions_dir = gov / "resume_sessions"
    preflights = sorted(sessions_dir.glob("RESUME_PREFLIGHT_*.json")) if sessions_dir.is_dir() else []
    runtimes = sorted(sessions_dir.glob("RESUME_RUNTIME_*.json")) if sessions_dir.is_dir() else []
    state["resume_preflight_count"] = len(preflights)
    state["resume_runtime_count"] = len(runtimes)
    state["resume_preflight_files"] = [p.name for p in preflights]
    state["resume_runtime_files"] = [p.name for p in runtimes]
    state["resume_session_pairing_ok"] = len(preflights) == len(runtimes)

    return state


def audit_session(
    entry: dict[str, Any],
    source_dir: Path,
    archive_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    archive_name = entry["archive"]
    source_zip = source_dir / archive_name
    if not source_zip.is_file():
        return {
            **entry,
            "status": "MISSING_SOURCE_ZIP",
            "source_zip": str(source_zip),
        }

    raw_dir = archive_root / "raw_zips"
    raw_dir.mkdir(parents=True, exist_ok=True)
    copied_zip = raw_dir / archive_name

    if copied_zip.exists():
        source_sha = sha256_file(source_zip)
        copied_sha = sha256_file(copied_zip)
        if source_sha != copied_sha:
            if not overwrite:
                raise RuntimeError(
                    f"Existing archived ZIP differs from source: {copied_zip}"
                )
            shutil.copy2(source_zip, copied_zip)
    else:
        shutil.copy2(source_zip, copied_zip)

    source_sha = sha256_file(source_zip)
    archived_sha = sha256_file(copied_zip)
    if source_sha != archived_sha:
        raise RuntimeError(f"ZIP copy SHA mismatch: {archive_name}")

    session_root = archive_root / "sessions" / source_zip.stem
    extracted_root = session_root / "extracted"
    manifest_path = session_root / "file_manifest.csv"

    if extracted_root.exists():
        if not overwrite:
            raise RuntimeError(
                f"Extraction target exists: {extracted_root}. "
                "Use --overwrite only if you intentionally want to rebuild the local mirror."
            )
        shutil.rmtree(extracted_root)

    extracted_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(copied_zip, "r") as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC failure in {archive_name}: {bad}")
        zip_member_count = len(zf.infolist())
        zip_uncompressed_bytes = sum(x.file_size for x in zf.infolist())
        safe_extract(zf, extracted_root)

    file_count, extracted_bytes, manifest_sha = full_file_manifest(
        extracted_root,
        manifest_path,
    )

    experiment_id = entry["experiment_id"]
    run_dir = find_run_dir(extracted_root, experiment_id)
    summary_path = find_summary(extracted_root, experiment_id)

    report: dict[str, Any] = {
        **entry,
        "status": "INGESTED",
        "source_zip": str(source_zip),
        "archived_zip": str(copied_zip),
        "zip_sha256": source_sha,
        "zip_bytes": source_zip.stat().st_size,
        "zip_member_count": zip_member_count,
        "zip_uncompressed_bytes": zip_uncompressed_bytes,
        "extracted_file_count": file_count,
        "extracted_bytes": extracted_bytes,
        "file_manifest": str(manifest_path),
        "file_manifest_sha256": manifest_sha,
        "run_dir_found": run_dir is not None,
        "final_validation_summary_found": summary_path is not None,
    }

    if summary_path is not None:
        report["final_validation_summary_path"] = str(summary_path)
        report["final_validation_summary_sha256"] = sha256_file(summary_path)

    if run_dir is None:
        report["classification"] = "RUN_DIRECTORY_NOT_FOUND"
        return report

    report["run_dir"] = str(run_dir)

    required = {
        "args_yaml": run_dir / "args.yaml",
        "results_csv": run_dir / "results.csv",
        "best_pt": run_dir / "weights" / "best.pt",
        "last_pt": run_dir / "weights" / "last.pt",
        "pretrain_preflight": run_dir / "governance" / "PRETRAIN_PREFLIGHT.json",
        "runtime_manifest": run_dir / "governance" / "RUNTIME_MANIFEST.json",
    }
    report["required_artifacts"] = {
        name: path.is_file() for name, path in required.items()
    }

    for name in ("args_yaml", "best_pt", "last_pt"):
        path = required[name]
        if path.is_file():
            report[f"{name}_sha256"] = sha256_file(path)
            report[f"{name}_bytes"] = path.stat().st_size

    report["results"] = read_results(required["results_csv"])
    report["governance"] = governance_state(run_dir)

    g = report["governance"]
    lineage_ok = (
        g.get("training_source_commit") == entry["expected_source_commit"]
        and g.get("execution_commit") == entry["expected_execution_commit"]
        and g.get("runtime_training_source_commit") == entry["expected_source_commit"]
        and g.get("runtime_execution_commit") == entry["expected_execution_commit"]
    )
    binding_ok = g.get("data_binding") == entry["expected_data_binding"]
    init_ok = g.get("initialization") == entry["expected_initialization"]

    test = g.get("test_access") or {}
    firewall_ok = (
        test.get("runtime_yaml_contains_test") is False
        and test.get("test_predictions") is False
        and test.get("test_metrics") is False
        and g.get("runtime_test_split_present") is False
    )

    results = report["results"]
    complete_100 = (
        results.get("rows") == 100
        and results.get("last_epoch") == 100
    )

    report["checks"] = {
        "required_artifacts_ok": all(report["required_artifacts"].values()),
        "lineage_ok": lineage_ok,
        "data_binding_ok": binding_ok,
        "initialization_ok": init_ok,
        "test_firewall_ok": firewall_ok,
        "resume_session_pairing_ok": g.get("resume_session_pairing_ok"),
        "complete_100_epochs": complete_100,
    }

    if entry["canonical_final"]:
        report["classification"] = (
            "CANONICAL_FINAL_PASS"
            if all(report["checks"].values())
            else "CANONICAL_FINAL_REVIEW_REQUIRED"
        )
    else:
        report["classification"] = "LINEAGE_SESSION_PRESERVED"

    return report


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the 11 downloaded Kaggle ZIP archives.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Local ResEMA repository root.",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Override BASELINE_FREEZE_01_INPUTS.json path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild an existing local extracted mirror.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    registry_path = (
        args.registry.resolve()
        if args.registry
        else repo_root / "research" / "05_experiments" / "BASELINE_FREEZE_01_INPUTS.json"
    )
    registry = read_json(registry_path)

    source_dir = args.source_dir.resolve()
    archive_root = repo_root / "artifacts_external" / "baseline_freeze_01"
    manifests_dir = archive_root / "manifests"
    report_dir = archive_root / "freeze_report"
    archive_root.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    reports = [
        audit_session(entry, source_dir, archive_root, args.overwrite)
        for entry in registry["sessions"]
    ]

    zip_rows = [
        {
            "archive": r.get("archive"),
            "experiment_id": r.get("experiment_id"),
            "session_role": r.get("session_role"),
            "canonical_final": r.get("canonical_final"),
            "kaggle_ref": r.get("kaggle_ref"),
            "zip_sha256": r.get("zip_sha256"),
            "zip_bytes": r.get("zip_bytes"),
            "zip_member_count": r.get("zip_member_count"),
            "file_manifest_sha256": r.get("file_manifest_sha256"),
            "classification": r.get("classification"),
        }
        for r in reports
    ]
    write_csv(
        manifests_dir / "ZIP_ARCHIVES.csv",
        zip_rows,
        [
            "archive", "experiment_id", "session_role", "canonical_final",
            "kaggle_ref", "zip_sha256", "zip_bytes", "zip_member_count",
            "file_manifest_sha256", "classification",
        ],
    )

    session_rows: list[dict[str, Any]] = []
    canonical_rows: list[dict[str, Any]] = []
    for r in reports:
        results = r.get("results") or {}
        gov = r.get("governance") or {}
        row = {
            "archive": r.get("archive"),
            "experiment_id": r.get("experiment_id"),
            "session_role": r.get("session_role"),
            "canonical_final": r.get("canonical_final"),
            "kaggle_ref": r.get("kaggle_ref"),
            "classification": r.get("classification"),
            "completed_epochs": results.get("rows"),
            "last_epoch": results.get("last_epoch"),
            "best_epoch": results.get("best_epoch"),
            "precision": results.get("best_precision"),
            "recall": results.get("best_recall"),
            "f1": results.get("best_f1"),
            "mAP50": results.get("best_map50"),
            "mAP50_95": results.get("best_map50_95"),
            "source_commit": gov.get("training_source_commit"),
            "execution_commit": gov.get("execution_commit"),
            "data_binding": gov.get("data_binding"),
            "initialization": gov.get("initialization"),
            "resume_preflight_count": gov.get("resume_preflight_count"),
            "resume_runtime_count": gov.get("resume_runtime_count"),
            "best_pt_sha256": r.get("best_pt_sha256"),
            "last_pt_sha256": r.get("last_pt_sha256"),
            "results_csv_sha256": results.get("sha256"),
            "args_yaml_sha256": r.get("args_yaml_sha256"),
            "zip_sha256": r.get("zip_sha256"),
            "full_file_manifest_sha256": r.get("file_manifest_sha256"),
        }
        session_rows.append(row)
        if r.get("canonical_final"):
            canonical_rows.append(row)

    fields = [
        "archive", "experiment_id", "session_role", "canonical_final",
        "kaggle_ref", "classification", "completed_epochs", "last_epoch",
        "best_epoch", "precision", "recall", "f1", "mAP50", "mAP50_95",
        "source_commit", "execution_commit", "data_binding", "initialization",
        "resume_preflight_count", "resume_runtime_count", "best_pt_sha256",
        "last_pt_sha256", "results_csv_sha256", "args_yaml_sha256",
        "zip_sha256", "full_file_manifest_sha256",
    ]
    write_csv(manifests_dir / "SESSION_SUMMARY.csv", session_rows, fields)
    write_csv(manifests_dir / "CANONICAL_FINAL_SUMMARY.csv", canonical_rows, fields)

    sha_lines = []
    for r in reports:
        if r.get("zip_sha256"):
            sha_lines.append(f'{r["zip_sha256"]}  raw_zips/{r["archive"]}')
    (manifests_dir / "SHA256SUMS.txt").write_text(
        "\n".join(sha_lines) + ("\n" if sha_lines else ""),
        encoding="utf-8",
    )

    overall = {
        "schema_version": "BASELINE-FREEZE-01-local-ingest-v1.0",
        "registry": str(registry_path),
        "source_dir": str(source_dir),
        "archive_root": str(archive_root),
        "registered_sessions": len(registry["sessions"]),
        "ingested_sessions": sum(r.get("status") == "INGESTED" for r in reports),
        "canonical_final_expected": registry["expected_canonical_final_runs"],
        "canonical_final_pass": sum(
            r.get("classification") == "CANONICAL_FINAL_PASS" for r in reports
        ),
        "test_predictions_generated": False,
        "test_metrics_generated": False,
        "training_started": False,
        "repository_scientific_records_modified": False,
        "reports": reports,
    }
    report_path = report_dir / "BASELINE_FREEZE_01_LOCAL_INGEST.json"
    report_path.write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")

    print("=" * 100)
    print("BASELINE-FREEZE-01 LOCAL INGEST")
    print("=" * 100)
    print(f"REGISTERED_SESSIONS={overall['registered_sessions']}")
    print(f"INGESTED_SESSIONS={overall['ingested_sessions']}")
    print(f"CANONICAL_FINAL_EXPECTED={overall['canonical_final_expected']}")
    print(f"CANONICAL_FINAL_PASS={overall['canonical_final_pass']}")
    print(f"REPORT={report_path}")
    print(f"CANONICAL_SUMMARY={manifests_dir / 'CANONICAL_FINAL_SUMMARY.csv'}")
    print("TRAINING_STARTED=FALSE")
    print("TEST_PREDICTIONS_GENERATED=FALSE")
    print("TEST_METRICS_GENERATED=FALSE")
    print("=" * 100)

    if overall["ingested_sessions"] != registry["expected_total_zip_archives"]:
        return 2
    if overall["canonical_final_pass"] != registry["expected_canonical_final_runs"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
