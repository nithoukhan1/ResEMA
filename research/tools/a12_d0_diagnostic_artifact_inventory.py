from __future__ import annotations

import csv
import hashlib
import io
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REPO = Path(
    r"E:\PhD\Admitted\Research\Project 1\ResEMA-Github Repo\ResEMA-ARCH-CORR"
)

EXPECTED_BRANCH = "research/combination-screen-01"
EXPECTED_HEAD = "18579a62c68d1a56d8ccae6b47ab890ec6eb7d7e"

SEARCH_ROOTS = [
    Path(r"C:\Users\ik_pa\Downloads\Detection Project\Model output"),
    Path(
        r"E:\PhD\Admitted\Research\Project 1\ResEMA-Github Repo"
        r"\ResEMA\artifacts_external"
    ),
    Path(
        r"E:\PhD\Admitted\Research\Project 1\ResEMA-Github Repo"
        r"\ResEMA-ARCH-CORR\artifacts_external"
    ),
]

# Authoritative artifact identities from frozen baseline/SMS closure and
# COMB-01 final evidence. These are DEVELOPMENT-validation artifacts only.
MODELS = [
    {
        "experiment_id": "BASE-B-ORG-PT-S42",
        "paper_name": "YOLO11s baseline",
        "role": "BASELINE_REFERENCE",
        "best_sha256": "65e3c59901e0429f70a7b368dc29ebb698c427a799cfcf66354a882ec48bc4c7",
        "last_sha256": "1b1dfee37aa38ef0983f66799e6948c765cb6360366e9cfa5a041774eac818f6",
        "results_sha256": "eb1d95da732c5beeff09b585b7be51216aed666de1ef8f97c8ea7e610045b758",
        "args_sha256": "b777e925afcfe630847d6bcee2cc7da0c60df672a815e65ad644ba011820f3aa",
        "expected_rows": 100,
        "expected_archive_name": None,
    },
    {
        "experiment_id": "BORG-PT-S42-SCCONV-EARLY-E100",
        "paper_name": "SCConv-Early",
        "role": "SUCCESSFUL_BACKBONE_REFERENCE",
        "best_sha256": "620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1",
        "last_sha256": "c118eaaa37977153211f1d90d69355144f0efecb741c0423fe12a0fa2412530a",
        "results_sha256": "474533cc41f5531a1b78369c312f217c9ee7ec1f52c2942f8c1c519f834eee84",
        "args_sha256": "09ff3678be480df67eeb71613b5e365b56b0818b0b90b806b6ecf55b4adae667",
        "expected_rows": 100,
        "expected_archive_name": "BORG-PT-S42-SCCONV-EARLY-E100-2.zip",
    },
    {
        "experiment_id": "BORG-PT-S42-SCCONV-4STAGE-E100",
        "paper_name": "SCConv-4Stage",
        "role": "SCCONV_PLACEMENT_INTENSITY_CONTROL",
        "best_sha256": "5485ec2fd9cc6421bdff7b4f0cd8b724db3ed8b540d4d11c77519cdbbcbfb7c4",
        "last_sha256": "e5c25083a78c44d0ba815075ce0901ec1efd76a7d44e8c177f059a13e02d8089",
        "results_sha256": "8a1f406ad6346d9c10720c7ac0d00406e1102f97cdcde14eb1b3c70083037d51",
        "args_sha256": "eb69ca997357c142b4d993bff6262e698603a2c0858f5ae5e9748028a06449ce",
        "expected_rows": 100,
        "expected_archive_name": "BORG-PT-S42-SCCONV-4STAGE-E100-2.zip",
    },
    {
        "experiment_id": "BORG-PT-S42-DYSAMPLE-E100",
        "paper_name": "DySample",
        "role": "UPSAMPLING_FAILURE_DIAGNOSTIC",
        "best_sha256": "61f3a99e4d29f20fddddd103ef3af049861d0b2278f384a044897be93a5fd556",
        "last_sha256": "eb565a95d745f2f42c6ea6134b66c7a86053c6f018f6a4a0f122929ce5d28782",
        "results_sha256": "26176bd454e492a84a8a23864ef5fb6c2beed9fcf36f77852d609c06f4f8be8a",
        "args_sha256": "233af65dbae323b11ed6c5df0247a64342890056eda981420971f4e39dc3ca29",
        "expected_rows": 100,
        "expected_archive_name": "BORG-PT-S42-DYSAMPLE-E100-1.zip",
    },
    {
        "experiment_id": "BORG-PT-S42-CANONICAL-EMA-E100",
        "paper_name": "Canonical EMA",
        "role": "POSITIVE_ATTENTION_ABLATION",
        "best_sha256": "f9d262c22c2bcb075cafe8eac368c0d4083a99f765f64e8ed6781c337ff0b6ec",
        "last_sha256": "4a64caf2b89ff59a327fdfa05208c802721af3663c6fa9686c2649091c00ce51",
        "results_sha256": "21dbd6f1074267b59534b78e1d2db3eb559fcdcbca77c7271d16f081feddb6b7",
        "args_sha256": "f24add684540f72d012c369519acb4ff8a7589b72ec9f926defe39de789a7126",
        "expected_rows": 100,
        "expected_archive_name": "BORG-PT-S42-CANONICAL-EMA-E100-2.zip",
    },
    {
        "experiment_id": "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100",
        "paper_name": "SCConv-Early + Canonical EMA",
        "role": "FAILED_COMPLEMENTARITY_INTERACTION",
        "best_sha256": "531c9927592781e7e19b6af1a35ce910113758d7ad5842677a5b7065487c897d",
        "last_sha256": "a560a2a87fa577cb83b1dd63c461270153304c3e245df81d66e8f80b0037f3ee",
        "results_sha256": "63258d1a04c4629d08725d9ef189aaa71f915c0c4e280d5a14c5fe55c207c1a4",
        "args_sha256": None,
        "expected_rows": 100,
        "expected_archive_name": "BORG-PT-S42-SCCONV-EARLY-CANONICAL-EMA-E100-2.zip",
        "expected_zip_sha256": "d0de1f91d0a914c2b0b334fc622c17266fa65c310dd10d01e004d495dfa76f01",
        "expected_zip_bytes": 41571976,
    },
]


def fail(message: str) -> None:
    raise RuntimeError(message)


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout.rstrip())
        if result.stderr:
            print(result.stderr.rstrip())
        fail("Git command failed: git " + " ".join(args))
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalized(s: str) -> str:
    return s.replace("\\", "/").lower()


def count_csv_rows(data: bytes) -> Optional[int]:
    try:
        text = data.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)
        if not rows:
            return 0
        return max(len(rows) - 1, 0)
    except Exception:
        return None


def experiment_name_match(experiment_id: str, path_text: str) -> bool:
    # Match exact experiment identity in path text. This keeps, for example,
    # SCConv-Early separate from SCConv-Early + Canonical EMA.
    target = experiment_id.lower()
    p = normalized(path_text)
    return target in p


def find_zip_candidates(experiment_id: str) -> list[Path]:
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        try:
            for p in root.rglob("*.zip"):
                if experiment_name_match(experiment_id, p.name):
                    found.append(p)
        except PermissionError:
            continue
    return sorted(set(found), key=lambda p: str(p).lower())


def find_directory_candidates(experiment_id: str) -> list[Path]:
    # Search only for best.pt, then infer the run directory.
    found: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.is_dir():
            continue
        try:
            for best in root.rglob("best.pt"):
                if experiment_name_match(experiment_id, str(best)):
                    # Usually .../<EXP>/weights/best.pt -> experiment dir
                    parent = best.parent.parent if best.parent.name.lower() == "weights" else best.parent
                    found.append(parent)
        except PermissionError:
            continue
    return sorted(set(found), key=lambda p: str(p).lower())


def zip_member_by_suffix(names: list[str], suffix: str) -> list[str]:
    suffix_n = suffix.lower().replace("\\", "/")
    return [
        n for n in names
        if normalized(n).endswith(suffix_n)
    ]


def inspect_zip(path: Path, model: dict) -> dict:
    rec: dict = {
        "kind": "ZIP",
        "path": str(path),
        "zip_bytes": path.stat().st_size,
        "zip_sha256": sha256_file(path),
        "zip_ok": False,
        "entries": None,
        "best_member": None,
        "best_sha256": None,
        "last_member": None,
        "last_sha256": None,
        "results_member": None,
        "results_sha256": None,
        "results_rows": None,
        "args_member": None,
        "args_sha256": None,
        "confusion_matrix": False,
        "confusion_matrix_normalized": False,
        "box_curve_count": 0,
        "val_pred_image_count": 0,
        "image_plot_count": 0,
        "canonical_member_match": False,
        "expected_filename_match": None,
        "expected_zip_hash_match": None,
        "expected_zip_bytes_match": None,
    }

    try:
        with zipfile.ZipFile(path, "r") as zf:
            bad = zf.testzip()
            if bad is not None:
                rec["zip_error"] = f"CRC failure at {bad}"
                return rec

            rec["zip_ok"] = True
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
            rec["entries"] = len(names)

            bests = zip_member_by_suffix(names, "/weights/best.pt")
            lasts = zip_member_by_suffix(names, "/weights/last.pt")
            results = zip_member_by_suffix(names, "/results.csv")
            args = zip_member_by_suffix(names, "/args.yaml")

            if len(bests) == 1:
                rec["best_member"] = bests[0]
                rec["best_sha256"] = sha256_bytes(zf.read(bests[0]))

            if len(lasts) == 1:
                rec["last_member"] = lasts[0]
                rec["last_sha256"] = sha256_bytes(zf.read(lasts[0]))

            if len(results) == 1:
                rec["results_member"] = results[0]
                result_bytes = zf.read(results[0])
                rec["results_sha256"] = sha256_bytes(result_bytes)
                rec["results_rows"] = count_csv_rows(result_bytes)

            if len(args) == 1:
                rec["args_member"] = args[0]
                rec["args_sha256"] = sha256_bytes(zf.read(args[0]))

            lower_names = [normalized(n) for n in names]
            rec["confusion_matrix"] = any(
                n.endswith("/confusion_matrix.png") for n in lower_names
            )
            rec["confusion_matrix_normalized"] = any(
                n.endswith("/confusion_matrix_normalized.png") for n in lower_names
            )
            rec["box_curve_count"] = sum(
                (
                    n.endswith("/boxp_curve.png")
                    or n.endswith("/boxr_curve.png")
                    or n.endswith("/boxf1_curve.png")
                    or n.endswith("/boxpr_curve.png")
                )
                for n in lower_names
            )
            rec["val_pred_image_count"] = sum(
                "/val_batch" in n and n.endswith("_pred.jpg")
                for n in lower_names
            )
            rec["image_plot_count"] = sum(
                n.endswith(".png") or n.endswith(".jpg") or n.endswith(".jpeg")
                for n in lower_names
            )

    except Exception as exc:
        rec["zip_error"] = repr(exc)
        return rec

    checks = [
        rec["best_sha256"] == model["best_sha256"],
        rec["last_sha256"] == model["last_sha256"],
        rec["results_sha256"] == model["results_sha256"],
        rec["results_rows"] == model["expected_rows"],
    ]

    if model.get("args_sha256") is not None:
        checks.append(rec["args_sha256"] == model["args_sha256"])

    rec["canonical_member_match"] = all(checks)

    expected_name = model.get("expected_archive_name")
    if expected_name:
        rec["expected_filename_match"] = path.name.lower() == expected_name.lower()

    if model.get("expected_zip_sha256"):
        rec["expected_zip_hash_match"] = (
            rec["zip_sha256"] == model["expected_zip_sha256"]
        )

    if model.get("expected_zip_bytes"):
        rec["expected_zip_bytes_match"] = (
            rec["zip_bytes"] == model["expected_zip_bytes"]
        )

    return rec


def inspect_directory(path: Path, model: dict) -> dict:
    weights = path / "weights"
    best = weights / "best.pt"
    last = weights / "last.pt"
    results = path / "results.csv"
    args = path / "args.yaml"

    rec = {
        "kind": "DIRECTORY",
        "path": str(path),
        "best_sha256": sha256_file(best) if best.is_file() else None,
        "last_sha256": sha256_file(last) if last.is_file() else None,
        "results_sha256": sha256_file(results) if results.is_file() else None,
        "args_sha256": sha256_file(args) if args.is_file() else None,
        "results_rows": None,
        "confusion_matrix": (path / "confusion_matrix.png").is_file(),
        "confusion_matrix_normalized": (path / "confusion_matrix_normalized.png").is_file(),
        "box_curve_count": 0,
        "val_pred_image_count": 0,
        "image_plot_count": 0,
        "canonical_member_match": False,
    }

    if results.is_file():
        try:
            rec["results_rows"] = count_csv_rows(results.read_bytes())
        except Exception:
            pass

    rec["box_curve_count"] = sum(
        (path / n).is_file()
        for n in [
            "BoxP_curve.png",
            "BoxR_curve.png",
            "BoxF1_curve.png",
            "BoxPR_curve.png",
        ]
    )
    rec["val_pred_image_count"] = len(list(path.glob("val_batch*_pred.jpg")))
    rec["image_plot_count"] = (
        len(list(path.glob("*.png")))
        + len(list(path.glob("*.jpg")))
        + len(list(path.glob("*.jpeg")))
    )

    checks = [
        rec["best_sha256"] == model["best_sha256"],
        rec["last_sha256"] == model["last_sha256"],
        rec["results_sha256"] == model["results_sha256"],
        rec["results_rows"] == model["expected_rows"],
    ]
    if model.get("args_sha256") is not None:
        checks.append(rec["args_sha256"] == model["args_sha256"])

    rec["canonical_member_match"] = all(checks)
    return rec


print("=" * 78)
print("A12-D0 - SIX-MODEL DIAGNOSTIC ARTIFACT INVENTORY")
print("READ-ONLY - NO TRAINING - NO DATASET ACCESS - NO TEST-SPLIT ACCESS")
print("=" * 78)

if not REPO.is_dir():
    fail(f"Repository missing: {REPO}")

os.chdir(REPO)

print()
print("--- REPOSITORY BINDING ---")

branch = git_value("branch", "--show-current")
head = git_value("rev-parse", "HEAD")
status = git_value("status", "--porcelain")

print(f"BRANCH={branch}")
print(f"HEAD={head}")
print(f"WORKTREE_CLEAN={str(not bool(status)).upper()}")

if branch != EXPECTED_BRANCH:
    fail(f"Unexpected branch: {branch}")

if head != EXPECTED_HEAD:
    fail(f"Unexpected HEAD: {head}")

if status:
    print(status)
    fail("Worktree must be clean for read-only D0 inventory")

print("REPOSITORY_BINDING=PASS")

print()
print("--- SEARCH ROOTS ---")

existing_roots = []
for root in SEARCH_ROOTS:
    exists = root.is_dir()
    print(f"ROOT={root}")
    print(f"EXISTS={str(exists).upper()}")
    if exists:
        existing_roots.append(root)

if not existing_roots:
    fail("None of the governed artifact roots exists")

print(f"EXISTING_SEARCH_ROOT_COUNT={len(existing_roots)}")

print()
print("--- SIX-MODEL INVENTORY ---")

overall_canonical_found = 0
inventory_rows: list[dict] = []

for idx, model in enumerate(MODELS, start=1):
    exp = model["experiment_id"]
    print()
    print("=" * 78)
    print(f"MODEL_INDEX={idx}")
    print(f"EXPERIMENT_ID={exp}")
    print(f"PAPER_NAME={model['paper_name']}")
    print(f"ROLE={model['role']}")
    print("=" * 78)

    zip_candidates = find_zip_candidates(exp)
    dir_candidates = find_directory_candidates(exp)

    print(f"ZIP_CANDIDATE_COUNT={len(zip_candidates)}")
    print(f"DIRECTORY_CANDIDATE_COUNT={len(dir_candidates)}")

    records = []

    for z in zip_candidates:
        rec = inspect_zip(z, model)
        records.append(rec)

        print()
        print(f"ARTIFACT_KIND=ZIP")
        print(f"PATH={rec['path']}")
        print(f"ZIP_BYTES={rec['zip_bytes']}")
        print(f"ZIP_SHA256={rec['zip_sha256']}")
        print(f"ZIP_CRC_OK={str(rec['zip_ok']).upper()}")
        print(f"ZIP_ENTRIES={rec.get('entries')}")
        print(f"BEST_SHA256={rec.get('best_sha256')}")
        print(f"LAST_SHA256={rec.get('last_sha256')}")
        print(f"RESULTS_SHA256={rec.get('results_sha256')}")
        print(f"ARGS_SHA256={rec.get('args_sha256')}")
        print(f"RESULTS_ROWS={rec.get('results_rows')}")
        print(f"CONFUSION_MATRIX={str(rec.get('confusion_matrix')).upper()}")
        print(
            "CONFUSION_MATRIX_NORMALIZED="
            + str(rec.get("confusion_matrix_normalized")).upper()
        )
        print(f"BOX_CURVE_COUNT={rec.get('box_curve_count')}")
        print(f"VAL_PRED_IMAGE_COUNT={rec.get('val_pred_image_count')}")
        print(f"IMAGE_PLOT_COUNT={rec.get('image_plot_count')}")
        print(
            "CANONICAL_MEMBER_MATCH="
            + str(rec.get("canonical_member_match")).upper()
        )
        if rec.get("expected_filename_match") is not None:
            print(
                "EXPECTED_FILENAME_MATCH="
                + str(rec["expected_filename_match"]).upper()
            )
        if rec.get("expected_zip_hash_match") is not None:
            print(
                "EXPECTED_ZIP_HASH_MATCH="
                + str(rec["expected_zip_hash_match"]).upper()
            )
        if rec.get("expected_zip_bytes_match") is not None:
            print(
                "EXPECTED_ZIP_BYTES_MATCH="
                + str(rec["expected_zip_bytes_match"]).upper()
            )
        if rec.get("zip_error"):
            print(f"ZIP_ERROR={rec['zip_error']}")

    for d in dir_candidates:
        rec = inspect_directory(d, model)
        records.append(rec)

        print()
        print("ARTIFACT_KIND=DIRECTORY")
        print(f"PATH={rec['path']}")
        print(f"BEST_SHA256={rec.get('best_sha256')}")
        print(f"LAST_SHA256={rec.get('last_sha256')}")
        print(f"RESULTS_SHA256={rec.get('results_sha256')}")
        print(f"ARGS_SHA256={rec.get('args_sha256')}")
        print(f"RESULTS_ROWS={rec.get('results_rows')}")
        print(f"CONFUSION_MATRIX={str(rec.get('confusion_matrix')).upper()}")
        print(
            "CONFUSION_MATRIX_NORMALIZED="
            + str(rec.get("confusion_matrix_normalized")).upper()
        )
        print(f"BOX_CURVE_COUNT={rec.get('box_curve_count')}")
        print(f"VAL_PRED_IMAGE_COUNT={rec.get('val_pred_image_count')}")
        print(f"IMAGE_PLOT_COUNT={rec.get('image_plot_count')}")
        print(
            "CANONICAL_MEMBER_MATCH="
            + str(rec.get("canonical_member_match")).upper()
        )

    canonical_records = [
        rec for rec in records
        if rec.get("canonical_member_match") is True
    ]

    canonical_zip_records = [
        rec for rec in canonical_records
        if rec["kind"] == "ZIP"
    ]

    if canonical_records:
        overall_canonical_found += 1
        model_status = "CANONICAL_ARTIFACT_FOUND"
    elif records:
        model_status = "CANDIDATES_FOUND_BUT_NO_CANONICAL_HASH_MATCH"
    else:
        model_status = "NO_LOCAL_ARTIFACT_FOUND"

    print()
    print(f"MODEL_STATUS={model_status}")
    print(f"CANONICAL_RECORD_COUNT={len(canonical_records)}")
    print(f"CANONICAL_ZIP_COUNT={len(canonical_zip_records)}")

    selected = canonical_zip_records[0] if canonical_zip_records else (
        canonical_records[0] if canonical_records else None
    )

    inventory_rows.append(
        {
            "experiment_id": exp,
            "paper_name": model["paper_name"],
            "role": model["role"],
            "status": model_status,
            "selected_kind": selected["kind"] if selected else "",
            "selected_path": selected["path"] if selected else "",
            "best_sha256": selected.get("best_sha256", "") if selected else "",
            "results_sha256": selected.get("results_sha256", "") if selected else "",
            "results_rows": selected.get("results_rows", "") if selected else "",
            "confusion_matrix": selected.get("confusion_matrix", "") if selected else "",
            "confusion_matrix_normalized": (
                selected.get("confusion_matrix_normalized", "") if selected else ""
            ),
            "box_curve_count": selected.get("box_curve_count", "") if selected else "",
            "val_pred_image_count": (
                selected.get("val_pred_image_count", "") if selected else ""
            ),
            "image_plot_count": selected.get("image_plot_count", "") if selected else "",
        }
    )

print()
print("=" * 78)
print("A12-D0 SUMMARY")
print("=" * 78)

print(f"EXPECTED_MODEL_COUNT={len(MODELS)}")
print(f"CANONICAL_MODEL_ARTIFACTS_FOUND={overall_canonical_found}")

for row in inventory_rows:
    print(
        "SUMMARY="
        + row["experiment_id"]
        + "|"
        + row["status"]
        + "|"
        + row["selected_kind"]
        + "|"
        + row["selected_path"]
    )

# Diagnostic-readiness classification. D0 itself never accesses validation
# images or any test images.
if overall_canonical_found == len(MODELS):
    readiness = "READY_FOR_A12_D1_STANDARDIZED_VALIDATION_DIAGNOSTIC_PREFLIGHT"
else:
    readiness = "LOCAL_ARTIFACT_GAPS_REQUIRE_RECOVERY_BEFORE_A12_D1"

print()
print(f"A12_D0_INVENTORY_COMPLETE=TRUE")
print(f"DIAGNOSTIC_ARTIFACT_READINESS={readiness}")
print("FILES_MODIFIED=FALSE")
print("TRAINING_OCCURRED=FALSE")
print("DATASET_ACCESS=NONE")
print("TEST_ACCESS=NONE")
print("NEW_GPU_TRAINING_AUTHORIZED=FALSE")
print("=" * 78)
