#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REQUIRED_COLUMNS = {
    "filestem",
    "patient_id",
    "study_number",
    "timehash",
    "laterality",
    "projection",
    "split",
}

KNOWN_UNREADABLE_VAL = {"1502_0635264266_05_WRI-R2_M015"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(path: Path, expected_split: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"missing CSV header: {path}")
        missing = REQUIRED_COLUMNS - set(reader.fieldnames)
        if missing:
            raise ValueError(f"missing required columns in {path}: {sorted(missing)}")
        rows = list(reader)

    for r in rows:
        if r["split"].strip().lower() != expected_split:
            raise ValueError(
                f"split column drift in {path}: expected {expected_split!r}, got {r['split']!r}"
            )
    return rows


def _int_text(value: str) -> str:
    return str(int(float(str(value).strip())))


def _projection(value: str) -> int:
    return int(float(str(value).strip()))


def _pair_id(split: str, patient_id: str, study_number: str, laterality: str) -> str:
    return (
        f"{split.upper()}-P{int(float(patient_id)):04d}"
        f"-S{int(float(study_number)):02d}-{laterality.strip().upper()}"
    )


def build_manifests(
    rows: list[dict[str, str]],
    split: str,
    known_unreadable: set[str] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    known_unreadable = set(known_unreadable or set())
    split = split.strip().lower()

    filestems = [r["filestem"].strip() for r in rows]
    if len(filestems) != len(set(filestems)):
        raise ValueError(f"duplicate filestem(s) in {split}")

    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        key = (
            _int_text(r["patient_id"]),
            _int_text(r["study_number"]),
            r["laterality"].strip().upper(),
        )
        groups[key].append(r)

    pair_rows: list[dict[str, object]] = []
    assignment_rows: list[dict[str, object]] = []

    pair_capable_groups = 0
    exact_pair_groups = 0
    exact_pair_patients: set[str] = set()
    ambiguous_ap_lat_groups = 0

    def group_sort_key(item):
        (patient, study, lat), _ = item
        return (int(patient), int(study), lat)

    for (patient, study, lat), group in sorted(groups.items(), key=group_sort_key):
        ordered = sorted(group, key=lambda r: (_projection(r["projection"]), r["filestem"]))
        p1 = [r for r in ordered if _projection(r["projection"]) == 1]
        p2 = [r for r in ordered if _projection(r["projection"]) == 2]
        p3 = [r for r in ordered if _projection(r["projection"]) == 3]

        pair_capable = bool(p1 and p2)
        if pair_capable:
            pair_capable_groups += 1

        exact = len(ordered) == 2 and len(p1) == 1 and len(p2) == 1 and len(p3) == 0

        if exact:
            exact_pair_groups += 1
            exact_pair_patients.add(patient)
            ap_row, lat_row = p1[0], p2[0]
            pid = _pair_id(split, patient, study, lat)
            ap_stem = ap_row["filestem"].strip()
            lat_stem = lat_row["filestem"].strip()
            unreadable_members = sorted({ap_stem, lat_stem} & known_unreadable)

            if not unreadable_members:
                operational_pair_status = "USABLE_PAIR" if split == "val" else "READABILITY_NOT_CHECKED"
            else:
                operational_pair_status = "COMPANION_UNREADABLE_FALLBACK"

            pair_rows.append(
                {
                    "split": split,
                    "pair_id": pid,
                    "patient_id": patient,
                    "study_number": study,
                    "laterality": lat,
                    "ap_filestem": ap_stem,
                    "lat_filestem": lat_stem,
                    "ap_timehash": ap_row["timehash"].strip(),
                    "lat_timehash": lat_row["timehash"].strip(),
                    "group_size": len(ordered),
                    "projection_1_count": len(p1),
                    "projection_2_count": len(p2),
                    "projection_3_count": len(p3),
                    "structural_pair_valid": 1,
                    "operational_pair_status": operational_pair_status,
                }
            )

            for role, row, companion in (
                ("AP", ap_row, lat_row),
                ("LAT", lat_row, ap_row),
            ):
                stem = row["filestem"].strip()
                companion_stem = companion["filestem"].strip()

                if split == "val" and stem in known_unreadable:
                    operational_role = "EXCLUDED_UNREADABLE"
                    readability = "KNOWN_UNREADABLE"
                elif split == "val" and companion_stem in known_unreadable:
                    operational_role = "SINGLE_FALLBACK_COMPANION_UNREADABLE"
                    readability = "NOT_KNOWN_UNREADABLE"
                elif split == "val":
                    operational_role = "PAIRED"
                    readability = "NOT_KNOWN_UNREADABLE"
                else:
                    operational_role = "PAIR_CANDIDATE"
                    readability = "NOT_CHECKED"

                assignment_rows.append(
                    {
                        "split": split,
                        "filestem": stem,
                        "patient_id": patient,
                        "study_number": study,
                        "laterality": lat,
                        "projection": _projection(row["projection"]),
                        "structural_role": "PAIRED",
                        "view_code": role,
                        "pair_id": pid,
                        "companion_filestem": companion_stem,
                        "operational_role": operational_role,
                        "runtime_readability_status": readability,
                        "group_size": len(ordered),
                        "single_reason": "",
                    }
                )
        else:
            if pair_capable:
                ambiguous_ap_lat_groups += 1
                reason = "AMBIGUOUS_NONEXACT_AP_LAT_GROUP"
            else:
                reason = "NO_EXACT_AP_LAT_PAIR"

            for row in ordered:
                stem = row["filestem"].strip()
                if split == "val" and stem in known_unreadable:
                    operational_role = "EXCLUDED_UNREADABLE"
                    readability = "KNOWN_UNREADABLE"
                elif split == "val":
                    operational_role = "SINGLE"
                    readability = "NOT_KNOWN_UNREADABLE"
                else:
                    operational_role = "SINGLE_CANDIDATE"
                    readability = "NOT_CHECKED"

                projection = _projection(row["projection"])
                view_code = "AP" if projection == 1 else "LAT" if projection == 2 else "OTHER"
                assignment_rows.append(
                    {
                        "split": split,
                        "filestem": stem,
                        "patient_id": patient,
                        "study_number": study,
                        "laterality": lat,
                        "projection": projection,
                        "structural_role": "SINGLE",
                        "view_code": view_code,
                        "pair_id": "",
                        "companion_filestem": "",
                        "operational_role": operational_role,
                        "runtime_readability_status": readability,
                        "group_size": len(ordered),
                        "single_reason": reason,
                    }
                )

    assignment_rows.sort(
        key=lambda r: (
            int(r["patient_id"]),
            int(r["study_number"]),
            str(r["laterality"]),
            int(r["projection"]),
            str(r["filestem"]),
        )
    )

    if len(assignment_rows) != len(rows):
        raise ValueError(
            f"{split} accounting mismatch: assignments={len(assignment_rows)} rows={len(rows)}"
        )
    if len({str(r["filestem"]) for r in assignment_rows}) != len(rows):
        raise ValueError(f"{split} assignment filestems are not unique")

    paired_images = sum(r["structural_role"] == "PAIRED" for r in assignment_rows)
    singles = len(assignment_rows) - paired_images
    operational_excluded = sum(r["operational_role"] == "EXCLUDED_UNREADABLE" for r in assignment_rows)
    operational_paired = sum(r["operational_role"] == "PAIRED" for r in assignment_rows)
    fallback = sum(
        r["operational_role"] == "SINGLE_FALLBACK_COMPANION_UNREADABLE"
        for r in assignment_rows
    )

    summary = {
        "split": split,
        "images": len(rows),
        "patients": len({_int_text(r["patient_id"]) for r in rows}),
        "side_specific_study_groups": len(groups),
        "pair_capable_groups": pair_capable_groups,
        "exact_pair_groups": exact_pair_groups,
        "exact_pair_patients": len(exact_pair_patients),
        "ambiguous_ap_lat_groups": ambiguous_ap_lat_groups,
        "structurally_paired_images": paired_images,
        "structural_single_images": singles,
        "structural_pair_image_fraction": paired_images / len(rows) if rows else 0.0,
        "operational_paired_images": operational_paired if split == "val" else None,
        "operational_single_fallback_images": fallback if split == "val" else None,
        "operational_excluded_unreadable": operational_excluded if split == "val" else None,
        "all_images_accounted_exactly_once": True,
    }
    return pair_rows, assignment_rows, summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def visibility_schema() -> dict[str, object]:
    return {
        "schema": "VGRA_VISIBILITY_TARGET_SCHEMA_V1",
        "classes": {
            "0": "boneanomaly",
            "1": "bonelesion",
            "2": "foreignbody",
            "3": "fracture",
            "4": "metal",
            "5": "periostealreaction",
            "6": "pronatorsign",
            "7": "softtissue",
            "8": "text",
        },
        "states": {
            "0": {"code": "00", "meaning": "absent in AP and LAT"},
            "1": {"code": "10", "meaning": "AP only"},
            "2": {"code": "01", "meaning": "LAT only"},
            "3": {"code": "11", "meaning": "present in AP and LAT"},
        },
        "target_formula": "state = ap_present + 2 * lat_present",
        "presence_definition": "1 iff the image has at least one YOLO GT object of that class",
        "training_target_source": "B-TRAIN YOLO label files only",
        "training_state_weight_source": "B-TRAIN exact-pair targets only",
        "training_state_weight_formula": "raw=1/sqrt(n+1); normalize four weights per class to mean 1",
        "validation_policy": "VAL targets may be used for validation loss/diagnostics only; never to derive training weights",
        "test_policy": "B-TEST labels/visibility states remain forbidden before the final sealed-test transaction",
        "metadata_columns_are_not_detection_targets": True,
        "actual_label_read_in_d4_c2": False,
        "actual_state_values_generated_in_d4_c2": False,
        "actual_state_frequency_generation": "DEFERRED_TO_D4-C5_USING_B-TRAIN_LABELS_ONLY",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    train = load_rows(args.train_csv.resolve(), "train")
    val = load_rows(args.val_csv.resolve(), "val")

    train_pairs, train_assign, train_summary = build_manifests(train, "train")
    val_pairs, val_assign, val_summary = build_manifests(
        val, "val", known_unreadable=KNOWN_UNREADABLE_VAL
    )

    train_patients = {_int_text(r["patient_id"]) for r in train}
    val_patients = {_int_text(r["patient_id"]) for r in val}
    overlap = sorted(train_patients & val_patients, key=int)
    if overlap:
        raise ValueError(f"TRAIN/VAL patient overlap detected: {overlap[:10]}")

    pair_rows = train_pairs + val_pairs
    assignment_rows = train_assign + val_assign

    pair_path = out / "D4_C2_VGRA_PAIR_MANIFEST.csv"
    assign_path = out / "D4_C2_VGRA_IMAGE_ASSIGNMENTS.csv"
    schema_path = out / "D4_C2_VGRA_VISIBILITY_SCHEMA.json"
    summary_path = out / "D4_C2_VGRA_PAIRING_SUMMARY.json"

    write_csv(pair_path, pair_rows)
    write_csv(assign_path, assignment_rows)
    schema_path.write_text(
        json.dumps(visibility_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary = {
        "schema": "D4_C2_VGRA_PAIRING_SUMMARY_V1",
        "train_csv_sha256": sha256_file(args.train_csv.resolve()),
        "val_csv_sha256": sha256_file(args.val_csv.resolve()),
        "train": train_summary,
        "val": val_summary,
        "train_val_patient_overlap": 0,
        "test_file_opened": False,
        "raw_images_opened": False,
        "raw_yolo_labels_opened": False,
        "visibility_state_values_generated": False,
        "visibility_schema_frozen": True,
        "pair_manifest_sha256": sha256_file(pair_path),
        "image_assignments_sha256": sha256_file(assign_path),
        "visibility_schema_sha256": sha256_file(schema_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"PAIR_MANIFEST={pair_path}")
    print(f"PAIR_MANIFEST_SHA256={summary['pair_manifest_sha256']}")
    print(f"IMAGE_ASSIGNMENTS={assign_path}")
    print(f"IMAGE_ASSIGNMENTS_SHA256={summary['image_assignments_sha256']}")
    print(f"VISIBILITY_SCHEMA={schema_path}")
    print(f"VISIBILITY_SCHEMA_SHA256={summary['visibility_schema_sha256']}")
    print(f"SUMMARY={summary_path}")
    print(f"SUMMARY_SHA256={sha256_file(summary_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
