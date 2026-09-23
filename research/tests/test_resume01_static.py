from __future__ import annotations

import ast
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESEARCH = ROOT / "research"


def _read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8", errors="strict")


def _function_source(source: str, *, class_name: str, function_name: str) -> str:
    tree = ast.parse(source)
    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    ]
    assert len(classes) == 1
    functions = [
        node for node in classes[0].body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    ]
    assert len(functions) == 1
    segment = ast.get_source_segment(source, functions[0])
    assert segment
    return segment


def test_resume01_contract_is_static_ci_verifiable():
    contract = json.loads(
        _read("research/05_experiments/RESUME01_RESUME_CONTRACT.json")
    )
    assert contract["schema_version"] == "RESUME01-governed-resume-v1.0"
    assert contract["status"] == "IMPLEMENTED"
    assert contract["training_authorization"] == (
        "DERIVED_FROM_NONEMPTY_EXPERIMENT_SOURCE_COMMIT"
    )
    assert contract["source_freeze_protocol"][
        "resume_sessions_require_exact_original_execution_commit"
    ]
    assert contract["dataset_firewall"]["runtime_yaml_contains_train"]
    assert contract["dataset_firewall"]["runtime_yaml_contains_val"]
    assert not contract["dataset_firewall"]["runtime_yaml_contains_test"]
    assert contract["copy_and_lineage"][
        "original_runtime_manifest_preserved_unmodified"
    ]


def test_resume01_sources_exist_and_compile():
    paths = [
        ROOT / "ultralytics/research/baseline_trainer.py",
        RESEARCH / "runtime/baseline_runner.py",
        RESEARCH / "runtime/resume_runner.py",
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8", errors="strict")
        compile(source, str(path), "exec")


def test_governed_trainer_has_distinct_resume_model_contract():
    source = _read("ultralytics/research/baseline_trainer.py")
    get_model = _function_source(
        source,
        class_name="GovernedDetectionTrainer",
        function_name="get_model",
    )
    assert 'is_resume = bool(getattr(self, "resume", False))' in get_model
    assert "if is_resume:" in get_model
    assert "model.load_state_dict(weights.state_dict(), strict=True)" in get_model
    assert '"resume_checkpoint_all_tensors_loaded_exactly": True' in get_model
    assert '"fresh_init_transfer_contract_reapplied": False' in get_model
    assert "EXPECTED_TRANSFERABLE_STATE_ITEMS" in get_model


def test_governed_trainer_preserves_original_manifest_on_resume():
    source = _read("ultralytics/research/baseline_trainer.py")
    setup = _function_source(
        source,
        class_name="GovernedDetectionTrainer",
        function_name="_setup_train",
    )
    assert 'if is_resume:' in setup
    assert 'original_manifest = governance_dir / "RUNTIME_MANIFEST.json"' in setup
    assert 'RESUME_RUNTIME_' in setup
    assert "refuses to overwrite an existing session runtime manifest" in setup
    assert "return" in setup


def test_resume_runner_has_no_scientific_cli_override_surface():
    source = _read("research/runtime/resume_runner.py")
    assert 'choices=["preflight-only", "execute"]' in source
    for forbidden in (
        '--epochs',
        '--imgsz',
        '--batch',
        '--optimizer',
        '--lr0',
        '--seed',
        '--box',
        '--cls',
        '--dfl',
    ):
        assert forbidden not in source
    assert "model = YOLO(str(last_pt), task=\"detect\")" in source
    assert "resume=True" in source
    assert "trainer=GovernedDetectionTrainer" in source


def test_resume_runner_requires_complete_prior_run_and_exact_copy():
    source = _read("research/runtime/resume_runner.py")
    for rel in (
        "args.yaml",
        "results.csv",
        "weights/last.pt",
        "weights/best.pt",
        "governance/PRETRAIN_PREFLIGHT.json",
        "governance/RUNTIME_MANIFEST.json",
    ):
        assert f'"{rel}"' in source
    assert "destination_inventory != source_inventory_before" in source
    assert "copied last.pt hash mismatch" in source
    assert "ARGS_BEFORE_RESUME_" in source


def test_source_guard_includes_resume_and_upstream_resume_semantics():
    source = _read("research/runtime/baseline_runner.py")
    required = [
        "research/runtime/resume_runner.py",
        "research/05_experiments/RESUME01_RESUME_CONTRACT.json",
        "ultralytics/engine/model.py",
        "ultralytics/engine/trainer.py",
        "ultralytics/nn/tasks.py",
        "ultralytics/utils/dist.py",
        "ultralytics/models/yolo/detect/train.py",
        "ultralytics/cfg/models/11/yolo11.yaml",
    ]
    for path in required:
        assert f'"{path}"' in source


def test_resume01_source_binding_is_blank_or_atomically_bound():
    import re

    with (RESEARCH / "05_experiments/EXPERIMENTS.csv").open(
        newline="", encoding="utf-8-sig"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 6
    assert all(row["status"] == "NOT_STARTED" for row in rows)
    bindings = {row["source_commit"].strip() for row in rows}
    assert bindings == {""} or (
        len(bindings) == 1
        and re.fullmatch(r"[0-9a-f]{40}", next(iter(bindings)))
    )

def test_experiment_matrix_is_frozen_except_atomic_source_binding():
    source = _read("research/runtime/baseline_runner.py")
    assert "def verify_experiment_matrix_binding" in source
    assert 'f"{source_commit}:research/05_experiments/EXPERIMENTS.csv"' in source
    assert 'if field == "source_commit":' in source
    assert "EXPERIMENTS.csv changed outside the permitted source_commit " in source
    assert "binding: experiment=" in source
    assert "verify_experiment_matrix_binding(source_commit)" in source


def test_preflight_and_runtime_yaml_are_reverified_inside_trainer():
    source = _read("ultralytics/research/baseline_trainer.py")
    setup = _function_source(
        source,
        class_name="GovernedDetectionTrainer",
        function_name="_setup_train",
    )
    assert "RESEMA_RESUME_PREFLIGHT_SHA256" in setup
    assert "RESEMA_PREFLIGHT_SHA256" in setup
    assert "Governed preflight manifest changed before trainer setup." in setup
    assert "Governed runtime data YAML hash mismatch." in setup
    assert "Governed train dataset path differs from preflight." in setup
    assert "Governed validation dataset path differs from preflight." in setup


def test_resume_runner_crosschecks_preserved_evidence():
    source = _read("research/runtime/resume_runner.py")
    assert "_verify_args_yaml_matches_checkpoint" in source
    assert "_verify_results_against_checkpoint" in source
    assert "checkpoint train_results" in source
    assert "best.pt Git commit mismatch" in source
    assert "RESEMA_RESUME_PREFLIGHT_SHA256" in source
    assert "runtime data YAML changed before launch" in source

def test_dataset_firewall_runs_before_ultralytics_dataset_loading():
    source = _read("ultralytics/research/baseline_trainer.py")
    get_dataset = _function_source(
        source,
        class_name="GovernedDetectionTrainer",
        function_name="get_dataset",
    )
    assert "self._verify_preflight_before_dataset_access()" in get_dataset
    assert "super().get_dataset()" in get_dataset
    assert "forbidden test split" in get_dataset

    verify = _function_source(
        source,
        class_name="GovernedDetectionTrainer",
        function_name="_verify_preflight_before_dataset_access",
    )
    assert "changed before dataset access" in verify
    assert '"test" in runtime_yaml' in verify
    assert "runtime train path differs from preflight before dataset access" in verify
    assert "runtime validation path differs from preflight before dataset access" in verify
