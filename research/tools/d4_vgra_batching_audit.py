#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from ultralytics.data.vgra import VGRAPairBatchSampler, bind_vgra_assignments


def sha256_text(value: object) -> str:
    raw = json.dumps(value, separators=(",", ":"), sort_keys=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load(path: Path, split: str):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"].strip().lower() == split]


def audit_split(rows, split: str, batch_size: int, seed: int):
    if split == "val":
        operational = [r for r in rows if r["operational_role"].strip().upper() != "EXCLUDED_UNREADABLE"]
    else:
        operational = list(rows)

    fake_files = [f"{r['filestem'].strip()}.png" for r in operational]
    metadata, units = bind_vgra_assignments(fake_files, rows, split)

    pair_units = [u for u in units if len(u) == 2]
    single_units = [u for u in units if len(u) == 1]
    sampler = VGRAPairBatchSampler(units, batch_size=batch_size, shuffle=True, seed=seed)
    epoch0 = list(iter(sampler))
    epoch1 = list(iter(sampler))

    def check(batches):
        flat = [i for b in batches for i in b]
        if sorted(flat) != list(range(len(fake_files))):
            raise RuntimeError(f"{split}: batch coverage mismatch")
        batch_of = {idx: bi for bi, batch in enumerate(batches) for idx in batch}
        for pair in pair_units:
            if batch_of[pair[0]] != batch_of[pair[1]]:
                raise RuntimeError(f"{split}: pair split detected")
        if any(len(b) > batch_size or len(b) == 0 for b in batches):
            raise RuntimeError(f"{split}: invalid batch size")

    check(epoch0)
    check(epoch1)

    return {
        "images": len(fake_files),
        "pair_units": len(pair_units),
        "single_units": len(single_units),
        "total_units": len(units),
        "batch_size": batch_size,
        "epoch0_batches": len(epoch0),
        "epoch1_batches": len(epoch1),
        "epoch0_min_batch": min(len(b) for b in epoch0),
        "epoch0_max_batch": max(len(b) for b in epoch0),
        "epoch0_order_sha256": sha256_text(epoch0),
        "epoch1_order_sha256": sha256_text(epoch1),
        "epochs_differ": epoch0 != epoch1,
        "all_images_accounted_once": True,
        "all_pairs_cobatched": True,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assignments", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows_train = load(args.assignments, "train")
    rows_val = load(args.assignments, "val")
    out = {
        "schema": "D4_C4_VGRA_BATCHING_AUDIT_V1",
        "source_assignment_manifest": str(args.assignments),
        "train": audit_split(rows_train, "train", args.batch_size, args.seed),
        "val": audit_split(rows_val, "val", args.batch_size, args.seed),
        "raw_images_opened": False,
        "raw_yolo_labels_opened": False,
        "test_access": "NONE",
        "ddp_supported_v1": False,
        "rectangular_pair_validation_enabled": False,
        "cross_study_composition_enabled": False,
    }
    args.output.write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
