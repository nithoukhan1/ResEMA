"""D4-C6 reproducible, metadata-only VGRA implementation closure audit.

Reads committed repository source, docs and gate records only. Never opens raw
images/YOLO labels, frozen B-TEST, or experiment outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

DESIGN_SHA = "a5d260ca83236c774e1920ce729e08eaacfcf5d4"
C5C_SHA = "fc991546d6d360905b9b1af33e736612d0379e31"
BRANCH = "research/vgra-impl-01"

STEPS = {
    "D4-C0": ("research/07_method/D4_C0_GATE_RECORD.json", None),
    "D4-C1": ("research/07_method/D4_C1_GATE_RECORD.json", "COMPLETE_CPU_UNIT_VERIFIED"),
    "D4-C2": ("research/07_method/D4_C2_GATE_RECORD.json", "COMPLETE_DETERMINISTIC_METADATA_ONLY"),
    "D4-C3": ("research/07_method/D4_C3_GATE_RECORD.json", "COMPLETE_CPU_VERIFIED"),
    "D4-C4": ("research/07_method/D4_C4_GATE_RECORD.json", "COMPLETE_CPU_VERIFIED_MANIFEST_AUDITED"),
    "D4-C5A": ("research/07_method/D4_C5A_GATE_RECORD.json", "COMPLETE_SYNTHETIC_CPU_VERIFIED"),
    "D4-C5B": ("research/07_method/D4_C5B_GATE_RECORD.json", "COMPLETE_CPU_SYNTHETIC_VERIFIED"),
    "D4-C5C": ("research/07_method/D4_C5C_GATE_RECORD.json", "COMPLETE_SYNTHETIC_CPU_VERIFIED"),
}

SOURCES = [
    "ultralytics/nn/modules/vgra.py",
    "ultralytics/nn/modules/vgra_head.py",
    "ultralytics/nn/modules/__init__.py",
    "ultralytics/nn/tasks.py",
    "ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml",
    "ultralytics/data/vgra.py",
    "ultralytics/models/yolo/detect/vgra_targets.py",
    "ultralytics/models/yolo/detect/vgra_runtime.py",
    "ultralytics/models/yolo/detect/vgra_trainer.py",
    "ultralytics/models/yolo/detect/vgra_validator.py",
    "research/tools/d4_vgra_pair_manifest.py",
    "research/tools/d4_vgra_batching_audit.py",
]

TESTS = [
    "research/tests/test_d4_c1_vgra_core.py",
    "research/tests/test_d4_c2_vgra_pairing.py",
    "research/tests/test_d4_c3_vgra_head.py",
    "research/tests/test_d4_c4_vgra_batching.py",
    "research/tests/test_d4_c5a_vgra_targets.py",
    "research/tests/test_d4_c5b_vgra_runtime_loss.py",
    "research/tests/test_d4_c5c_vgra_trainer_validator.py",
    "research/tests/test_arch_corr_01b_tpsc.py",
]

MANIFESTS = [
    "research/04_data/manifests/D4_C2_VGRA_PAIR_MANIFEST.csv",
    "research/04_data/manifests/D4_C2_VGRA_IMAGE_ASSIGNMENTS.csv",
    "research/04_data/manifests/D4_C2_VGRA_VISIBILITY_SCHEMA.json",
    "research/04_data/manifests/D4_C2_VGRA_PAIRING_SUMMARY.json",
    "research/04_data/manifests/D4_C4_VGRA_BATCHING_AUDIT.json",
]

RECORDS = [
    "research/07_method/D4_C1_VGRA_CORE_IMPLEMENTATION.md",
    "research/07_method/D4_C2_VGRA_PAIRING_MANIFEST_RECORD.md",
    "research/07_method/D4_C3_VGRA_HEAD_INTEGRATION.md",
    "research/07_method/D4_C4_VGRA_DATA_BATCHING.md",
    "research/07_method/D4_C5A_VISIBILITY_SUPERVISION_CONTRACT.md",
    "research/07_method/D4_C5B_RUNTIME_NATIVE_LOSS_CONTRACT.md",
    "research/07_method/D4_C5C_TRAINER_VALIDATOR_IMPLEMENTATION.md",
    "research/07_method/D4_C0_VGRA_EXECUTION_TRACKER.md",
    "research/07_method/D4_LPQ_MASTER_TRACKER.md",
    "research/07_method/D4_C0_VGRA_IMPLEMENTATION_MASTER_PLAN.md",
    "research/07_method/D4_B2C2_VGRA_MATHEMATICAL_SPEC.md",
]

PROTECTED_STOCK = [
    "ultralytics/nn/modules/head.py",
    "ultralytics/utils/loss.py",
    "ultralytics/data/dataset.py",
    "ultralytics/data/build.py",
    "ultralytics/engine/trainer.py",
    "ultralytics/engine/validator.py",
    "ultralytics/models/yolo/detect/train.py",
    "ultralytics/models/yolo/detect/val.py",
]

UNRESOLVED = [
    {
        "id": "VGRA_SIGNED_BETA_POLARITY",
        "status": "OPEN",
        "resolution_gate": "D4-D mathematical/polarity review before D4-E",
        "detail": "beta=2*tanh(rho) can invert the intended assist/suppress narrative"
    },
    {
        "id": "VGRA_ALL_ZERO_INPUT_GRADIENT",
        "status": "OPEN",
        "resolution_gate": "D4-D3 full numerical diagnosis",
        "detail": "C5C R1 all-zero synthetic pixels yielded non-finite gradient for an unidentified parameter"
    },
    {
        "id": "VGRA_PRETRAINED_CHECKPOINT_TRANSFER",
        "status": "UNVERIFIED",
        "resolution_gate": "D4-D2 exact Early weights/EMA transfer and zero-rho identity"
    },
    {
        "id": "VGRA_REAL_BTRAIN_CLASS_STATE_COUNTS",
        "status": "NOT_COMPUTED",
        "resolution_gate": "governed TRAIN label binding; never from VAL/TEST"
    },
    {
        "id": "VGRA_FULL_TRAINER_VALIDATOR_EPOCH",
        "status": "NOT_EXECUTED",
        "resolution_gate": "D4-D3 synthetic epoch, persistent loader/EMA/save-load behavior"
    },
    {
        "id": "VGRA_STANDALONE_PAIR_AWARE_FINAL_EVAL",
        "status": "UNAUTHORIZED",
        "resolution_gate": "separately governed D7/D8 B-TEST final-evaluation interface"
    },
]


def run_git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def audit(root: Path) -> dict:
    root = root.resolve()
    if run_git(root, "branch", "--show-current") != BRANCH:
        raise ValueError("VGRA implementation branch mismatch")
    if run_git(root, "rev-parse", "HEAD") != C5C_SHA:
        raise ValueError("Expected exact C5C parent for implementation closure")

    gates = {}
    for step, (rel, expected_status) in STEPS.items():
        file = root / rel
        gate = json.loads(file.read_text(encoding="utf-8"))
        if gate.get("branch", gate.get("implementation_branch")) != BRANCH:
            raise ValueError(f"{step}: branch/governance drift")
        if expected_status is not None and gate.get("status") != expected_status:
            raise ValueError(f"{step}: status drift {gate.get('status')}")
        if gate.get("gpu_training_authorized") is not False:
            raise ValueError(f"{step}: unexpected GPU training authorization")
        if gate.get("attention_authorized") is not False:
            raise ValueError(f"{step}: unexpected attention authorization")
        test = gate.get("b_test_access", gate.get("test_access"))
        if test != "NONE":
            raise ValueError(f"{step}: B-TEST firewall drift")
        gates[step] = {
            "status": gate.get("status", "GOVERNANCE_AUTHORIZED_NO_TRAINING"),
            "gate_record_sha256": checksum(file),
        }

    c0 = json.loads((root / STEPS["D4-C0"][0]).read_text(encoding="utf-8"))
    c2 = json.loads((root / STEPS["D4-C2"][0]).read_text(encoding="utf-8"))
    c3 = json.loads((root / STEPS["D4-C3"][0]).read_text(encoding="utf-8"))
    c4 = json.loads((root / STEPS["D4-C4"][0]).read_text(encoding="utf-8"))
    c5a = json.loads((root / STEPS["D4-C5A"][0]).read_text(encoding="utf-8"))
    c5b = json.loads((root / STEPS["D4-C5B"][0]).read_text(encoding="utf-8"))
    c5c = json.loads((root / STEPS["D4-C5C"][0]).read_text(encoding="utf-8"))
    if c0["design_commit"] != DESIGN_SHA:
        raise ValueError("Design authority mismatch")
    if c2["train_exact_pairs"] != 6496 or c2["val_operational_paired_images"] != 2802:
        raise ValueError("Frozen C2 pair count mismatch")
    if c3["vgra_parameters"] != 9_672_660 or c3["parameter_cap_pass"] is not True:
        raise ValueError("VGRA parameter cap drift")
    if c4["train_pair_units"] != 6496 or c4["val_pair_units"] != 1401:
        raise ValueError("Pair batch inventory drift")
    if c5a["signed_beta_semantic_inversion_gate"].startswith("OPEN") is not True:
        raise ValueError("Signed beta risk disappeared from C5A")
    if c5b["signed_beta_gate"] != "OPEN" or c5c["signed_beta_interpretation_gate"] != "OPEN":
        raise ValueError("Signed beta risk disappeared from later gate")
    if c5c["all_zero_input_gradient_edge_gate"] != "OPEN_D4_D3_DIAGNOSIS_REQUIRED":
        raise ValueError("Zero-gradient edge case not preserved")
    if c5c["r1_safe_stop"]["failed"] != 2 or c5c["r1_safe_stop"]["passed"] != 66:
        raise ValueError("C5C R1 failure evidence drift")
    if c5c["source_frozen"] or c5c["standalone_validation_authorized"]:
        raise ValueError("VGRA source/test evaluation prematurely authorized")

    assignments = root / "research/04_data/manifests/D4_C2_VGRA_IMAGE_ASSIGNMENTS.csv"
    if checksum(assignments) != c2["image_assignments_sha256"]:
        raise ValueError("C2 assignment hash drift")
    if checksum(assignments) != c4["assignment_manifest_sha256"]:
        raise ValueError("C4 assignment hash drift")

    protected = run_git(root, "diff", "--name-only", DESIGN_SHA, C5C_SHA).splitlines()
    touched_protected = sorted(set(protected) & set(PROTECTED_STOCK))
    if touched_protected:
        raise ValueError(f"Unexpected changes to stock code since design: {touched_protected}")

    audit_paths = sorted(set(SOURCES + TESTS + MANIFESTS + RECORDS + [rel for rel, _ in STEPS.values()]))
    files = {}
    for rel in audit_paths:
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(f"Required implementation/evidence missing: {rel}")
        files[rel] = {"sha256": checksum(path), "bytes": path.stat().st_size}

    return {
        "schema": "D4_C6_VGRA_IMPLEMENTATION_INVENTORY_V1",
        "audit_parent": C5C_SHA,
        "design_authority": DESIGN_SHA,
        "implementation_branch": BRANCH,
        "implementation_complete": True,
        "candidate_v1_source_frozen": False,
        "global_final_architecture_frozen": False,
        "gpu_training_authorized": False,
        "b_test_access": "NONE",
        "real_images_or_labels_opened_by_this_audit": False,
        "actual_train_state_counts_computed_by_this_audit": False,
        "model_trainable_parameters": 9_672_660,
        "parameter_cap": 9_870_093,
        "exact_train_pairs": 6496,
        "usable_val_pairs": 1401,
        "stage_gate_evidence": gates,
        "source_and_evidence_file_count": len(files),
        "source_and_evidence_hashes": files,
        "stock_ultralytics_protected_file_changes_since_design": touched_protected,
        "documented_safe_stops": {
            "D4-C1_R1": "documentation tracker mismatch; 17 tests passed; clean rollback",
            "D4-C4_R1": "audit import path; 38 tests passed; clean rollback",
            "D4-C5C_R1": "2 failed, 66 passed; clean rollback",
        },
        "unresolved_gates": UNRESOLVED,
        "next_action": "D4-D1_VGRA_INDEPENDENT_STATIC_UNIT_VERIFICATION",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    record = audit(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    print("D4_C6_IMPLEMENTATION_INVENTORY_AUDIT=PASS")
    print(f"INVENTORY_FILE_COUNT={record['source_and_evidence_file_count']}")
    print(f"INVENTORY_OUTPUT_SHA256={checksum(output)}")
    print("STOCK_PROTECTED_FILES_CHANGED=NONE")
    print("UNRESOLVED_GATES_PRESERVED=TRUE")


if __name__ == "__main__":
    main()
