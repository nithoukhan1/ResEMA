from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"

ALLOWED_STATUS = {
    "NOT_STARTED", "IN_PROGRESS", "BLOCKED", "COMPLETE",
    "REJECTED", "SUPERSEDED", "LOCKED"
}
ALLOWED_PATH_STATUS = {"VERIFIED", "VERIFIED_DIRECTORY", "PATH_NOT_RECONSTRUCTED"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")

def read_csv(rel):
    with (RESEARCH / rel).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def test_required_history_counts():
    experiments = read_csv("02_history/EXPERIMENT_REGISTRY.csv")
    results = read_csv("02_history/RESULTS_MASTER.csv")
    literature = read_csv("03_literature/Detection-Grazpedwri_RAW.csv")

    assert len(experiments) >= 10
    assert len(results) >= 10
    assert len(literature) >= 27

def test_experiment_ids_are_unique():
    rows = read_csv("02_history/EXPERIMENT_REGISTRY.csv")
    ids = [r["experiment_id"] for r in rows]
    assert len(ids) == len(set(ids))

def test_results_ids_exist_in_experiment_registry():
    exp_ids = {r["experiment_id"] for r in read_csv("02_history/EXPERIMENT_REGISTRY.csv")}
    for row in read_csv("02_history/RESULTS_MASTER.csv"):
        assert row["experiment_id"] in exp_ids

def test_project_tracker_status_vocabulary():
    for row in read_csv("02_history/PROJECT_TRACKER.csv"):
        assert row["status"] in ALLOWED_STATUS

def test_artifact_registry_hashes_and_statuses():
    rows = read_csv("02_history/ARTIFACT_REGISTRY.csv")
    assert len(rows) >= 8
    for row in rows:
        assert row["path_status"] in ALLOWED_PATH_STATUS
        assert HEX64.fullmatch(row["sha256"]), row

def test_no_large_model_artifacts_in_research_tree():
    forbidden = {".pt", ".pth", ".onnx", ".engine", ".tflite"}
    bad = [p for p in RESEARCH.rglob("*") if p.is_file() and p.suffix.lower() in forbidden]
    assert not bad, bad

def test_test_policy_is_documented():
    text = (RESEARCH / "01_provenance/DATASET_PROVENANCE.md").read_text(encoding="utf-8")
    assert "Test remains sealed" in text or "test remains sealed" in text.lower()

def test_runtime_manifest_template_is_valid_json():
    path = RESEARCH / "tools/ARTIFACT_MANIFEST_TEMPLATE.json"
    json.loads(path.read_text(encoding="utf-8"))

def test_research_text_files_are_utf8():
    """All publication-governed text files must be strict UTF-8."""
    extensions = {
        ".md",
        ".csv",
        ".py",
        ".yml",
        ".yaml",
        ".json",
        ".txt",
        ".sh",
    }

    paths = [
        p
        for p in RESEARCH.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    workflows = ROOT / ".github" / "workflows"

    if workflows.exists():
        paths.extend(
            p
            for p in workflows.glob("research-*.yml")
            if p.is_file()
        )
        paths.extend(
            p
            for p in workflows.glob("research-*.yaml")
            if p.is_file()
        )

    bad = []

    for path in sorted(set(paths)):
        try:
            path.read_text(
                encoding="utf-8",
                errors="strict",
            )
        except UnicodeDecodeError as exc:
            bad.append(
                f"{path.relative_to(ROOT)}: {exc}"
            )

    assert not bad, (
        "Non-UTF-8 publication-governed text files:\n"
        + "\n".join(bad)
    )
