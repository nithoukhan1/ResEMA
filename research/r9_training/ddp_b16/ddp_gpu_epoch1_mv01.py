"""Bounded MV01 two-T4 FRESH GPU acceptance (one epoch, NOT official training).

Uses the independently verified private C1-derived source stage and fixed DDP-B16
recipe. The worker calls native trainer.train(), then exits immediately after
a successfully published rank-paired G4 epoch boundary. It does not use the
historical epoch-1 G4 from a different source identity. No B-TEST access.

This is a development acceptance executable, not a 100-epoch session runner.
Its receipt must be checked on a separate Kaggle session before promoting it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

RUN_SCHEMA = "DDP_B16_FRESH_ONE_EPOCH_ACCEPTANCE_V1"
OUTPUT_SCHEMA = "DDP_B16_EPOCH1_G4_CANDIDATE_V1"


class AcceptanceFailure(RuntimeError):
    pass


class EpochBoundaryStop(Exception):
    pass


def need(ok, reason):
    if not ok:
        raise AcceptanceFailure("DDP_B16_ACCEPTANCE_" + reason)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_output_path(raw, *, label):
    """Only fresh, non-aliased /kaggle/working paths; never sealed B-TEST."""
    need(type(raw) is str and raw.startswith("/kaggle/working/"),
         label + "_ROOT")
    p = Path(raw)
    need(".." not in p.parts and "\\" not in raw and
         p.as_posix() == raw and
         not any(part.casefold() in {"test", "b-test", "b_test"}
                 for part in p.parts), label + "_CANONICAL")
    need(not p.exists() and not p.is_symlink() and
         p.parent.is_dir() and
         not any(parent.is_symlink() for parent in p.parents),
         label + "_MUST_BE_NEW")
    return p


def run_configuration(plan, project, g4_root, artifacts):
    """Review-only shape check; executed before any writes or CUDA use."""
    from research.r9_training.ddp_b16.ddp_worker import validate_plan
    validate_plan(plan)
    need(plan["arm"] == "MV01" and plan["mode"] == "FRESH_FROM_EARLY",
         "MV01_FRESH_ONLY")
    need(plan["b_test_access"] == "NONE", "B_TEST_SEALED")
    need(not Path(plan["stage_root"]).exists(), "PRIVATE_STAGE_NOT_NEW")
    roots = [
        safe_output_path(project, label="PROJECT"),
        safe_output_path(g4_root, label="G4"),
        safe_output_path(artifacts, label="ARTIFACTS"),
        Path(plan["stage_root"]),
    ]
    for i, root in enumerate(roots):
        for other in roots[i + 1:]:
            need(root != other and not root.is_relative_to(other) and
                 not other.is_relative_to(root), "OVERLAPPING_OUTPUT_ROOTS")
    checkout = Path(plan["checkout_root"]).resolve(strict=True)
    for root in roots:
        need(not root.is_relative_to(checkout) and
             not checkout.is_relative_to(root), "OUTPUT_OVERLAPS_CHECKOUT")
    return roots


def read_plan(path):
    from research.r9_training.ddp_b16.ddp_worker import _unique_json
    return _unique_json(Path(path))


def _g4_output(trainer, root, artifact_root, capsule, identity, run_id):
    """Rank 0 exports a complete, locally verified next-epoch G4 set."""
    next_epoch = 1
    name = f"{run_id}_epoch{next_epoch:03d}_two_rank_manifest.json"
    source_manifest = root / name
    need(source_manifest.is_file(), "NEW_G4_MANIFEST_MISSING")
    need(sha256(source_manifest) == capsule["manifest_sha256"],
         "NEW_G4_MANIFEST_SHA")
    payload = json.loads(source_manifest.read_text("utf-8"))
    need(payload.get("complete") is True and
         payload.get("identity") == identity and
         payload.get("cursor", {}).get("next_epoch") == 1 and
         payload["cursor"]["optimizer_updates"] > 0,
         "NEW_G4_IDENTITY_OR_CURSOR")

    output = artifact_root / "g4"
    need(not output.exists(), "OUTPUT_ALREADY_EXISTS")
    output.mkdir()
    files = {name: capsule["manifest_sha256"],
             "best.pt": capsule["selected_best_sha256"]}
    for rank in (0, 1):
        record = payload["ranks"][str(rank)]
        files[record["filename"]] = record["sha256"]
        files[record["receipt_filename"]] = record["receipt_sha256"]
    for filename, digest in files.items():
        need(Path(filename).name == filename, "BAD_G4_MEMBER_NAME")
        origin = Path(trainer.best) if filename == "best.pt" else root / filename
        need(origin.is_file() and not origin.is_symlink() and
             sha256(origin) == digest, "G4_MEMBER_SHA_MISMATCH")
        target = output / filename
        shutil.copyfile(origin, target)
        need(sha256(target) == digest, "G4_COPY_DRIFT")
    result = {
        "schema": OUTPUT_SCHEMA,
        "status": "NONOFFICIAL_EPOCH1_G4_SAVED_NOT_OFFSESSION_VERIFIED",
        "source_release_sha": identity["ddp_release_sha"],
        "run_id": run_id,
        "next_epoch": 1,
        "optimizer_updates": payload["cursor"]["optimizer_updates"],
        "manifest_sha256": capsule["manifest_sha256"],
        "selected_best_sha256": capsule["selected_best_sha256"],
        "files_sha256": files,
        "official_training_authorized": False,
        "offsession_resume_verified": False,
        "b_test_access": "NONE",
    }
    receipt = output / "ACCEPTANCE.json"
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")
    return result


def child(plan, *, project, g4_root, artifacts):
    import torch
    import torch.distributed as dist

    from research.r9_training.ddp_b16.ddp_worker import review_worker_inputs
    from research.r9_training.ddp_b16.ddp_verified_admission import verify_staged_admission
    from research.r9_training.ddp_b16.ddp_rank_admission import require_review_consensus
    from research.r9_training.ddp_b16.ddp_native_binding import make_native_binding
    from research.r9_training.ddp_b16.ddp_native_binding_consensus import require_native_binding_consensus
    from research.r9_training.ddp_b16.ddp_native_bootstrap_candidate import (
        verify_two_t4_runtime, load_pinned_native_class, bind_candidate_fresh_state,
    )
    from research.r9_training.ddp_b16 import ddp_native_g4_hook as hook

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1" and
         (world, rank, local) in {(2, 0, 0), (2, 1, 1)}, "T4_RANK_MAPPING")
    need(Path(plan["stage_root"]).is_dir() and
         Path(g4_root).is_dir(), "PREPARED_STAGE_OR_G4_MISSING")
    need(not Path(project).exists(), "NATIVE_PROJECT_ALREADY_EXISTS")

    dist.init_process_group("gloo", init_method="env://",
                            rank=rank, world_size=2,
                            timeout=timedelta(seconds=600))
    try:
        review = review_worker_inputs(plan, stage_source=False)
        admission = verify_staged_admission(plan, review)
        require_review_consensus(admission)
        config = read_plan(plan["r0_input_config"])
        binding = make_native_binding(plan, admission, config,
                                      project=project)
        require_native_binding_consensus(binding)
        need(binding["identity"]["ddp_release_sha"] ==
             plan["expected_git_head"], "RELEASE_IDENTITY")
        need(binding["arm"] == "MV01" and
             binding["overrides"]["epochs"] == 100 and
             binding["overrides"]["batch"] == 16 and
             binding["overrides"]["resume"] is False, "FROZEN_RECIPE")
        verify_two_t4_runtime(torch.cuda)
        native = load_pinned_native_class(binding)
        need(native.__name__ == "VGRADetectionTrainer", "NATIVE_CLASS")
        trainer = native(**binding["constructor"],
                         overrides=binding["overrides"])
        need(trainer.world_size == 2 and trainer.ddp is False,
             "NATIVE_WORLD_MAPPING")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

    # This separate candidate binding is not inferred from historical batch-8 G4.
    bind_candidate_fresh_state(trainer, binding, Path(g4_root))
    original = hook.ddp_b16_g4_epoch_end
    capsule = {}

    def one_boundary(obj, *, epoch, nb, last_opt_step):
        need(epoch == 0, "EXPECTED_FRESH_EPOCH1")
        result = original(obj, epoch=epoch, nb=nb,
                          last_opt_step=last_opt_step)
        need(result.get("status") ==
             "LOCAL_SYNCHRONIZED_DDP_CAPSULE_EXTERNAL_AUTHENTICATION_PENDING",
             "NO_COMPLETE_NONTERMINAL_G4")
        capsule.update(result)
        dist.barrier()
        raise EpochBoundaryStop("EPOCH1_G4_BOUNDARY_COMPLETE")

    hook.ddp_b16_g4_epoch_end = one_boundary
    try:
        try:
            trainer.train()
            raise AcceptanceFailure("UNEXPECTED_FULL_TRAINING_RETURN")
        except EpochBoundaryStop:
            need(bool(capsule.get("manifest_sha256")), "G4_NOT_PUBLISHED")
            record = {
                "rank": rank,
                "manifest_sha256": capsule["manifest_sha256"],
                "best_sha256": capsule["selected_best_sha256"],
            }
            peers = [None, None]
            dist.all_gather_object(peers, record)
            need([p["rank"] for p in peers] == [0, 1] and
                 peers[0]["manifest_sha256"] == peers[1]["manifest_sha256"] and
                 peers[0]["best_sha256"] == peers[1]["best_sha256"],
                 "G4_RANK_CONSENSUS")
            if rank == 0:
                result = _g4_output(trainer, Path(g4_root),
                                    Path(artifacts), capsule,
                                    binding["identity"], plan["run_id"])
                print("DDP_B16_EPOCH1_G4_RESULT=" +
                      json.dumps(result, sort_keys=True), flush=True)
            dist.barrier()
            receipt = Path(artifacts) / "g4" / "ACCEPTANCE.json"
            need(receipt.is_file(), "ACCEPTANCE_RECEIPT_NOT_SHARED")
            dist.barrier()
    finally:
        hook.ddp_b16_g4_epoch_end = original
        if dist.is_initialized():
            dist.destroy_process_group()


def parent(plan_path, *, project, g4_root, artifacts):
    from research.r9_training.ddp_b16.ddp_worker import review_worker_inputs
    from research.r9_training.ddp_b16.ddp_verified_admission import verify_staged_admission

    plan = read_plan(plan_path)
    run_configuration(plan, project, g4_root, artifacts)
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "0,1",
         "CUDA_0_1_REQUIRED")
    need(not any(name in os.environ for name in
                 ("RANK", "LOCAL_RANK", "WORLD_SIZE")),
         "PARENT_INSIDE_TORCHRUN")

    # One-time source/asset admission before launching two GPU ranks.
    review = review_worker_inputs(plan, stage_source=True)
    admission = verify_staged_admission(plan, review)
    need(admission["identity"]["ddp_release_sha"] ==
         plan["expected_git_head"], "STAGE_RELEASE_DRIFT")
    Path(g4_root).mkdir()
    Path(artifacts).mkdir()
    command = [
        sys.executable, "-m", "torch.distributed.run",
        "--standalone", "--nnodes=1", "--nproc_per_node=2",
        "--max_restarts=0", str(Path(__file__).resolve()),
        "--plan", str(plan_path), "--project", project,
        "--g4-root", g4_root, "--artifacts", artifacts, "--child",
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    checkout = str(Path(plan["checkout_root"]).resolve(strict=True))
    env["PYTHONPATH"] = checkout + os.pathsep + env.get("PYTHONPATH", "")
    log_path = Path(artifacts) / "TWO_RANK_EPOCH1.log"
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=checkout, env=env,
                                 stdout=log, stderr=subprocess.STDOUT,
                                 check=False)
    need(process.returncode == 0,
         "TWO_RANK_FAILED_READ_" + str(log_path))
    receipt_path = Path(artifacts) / "g4" / "ACCEPTANCE.json"
    need(receipt_path.is_file(), "MISSING_EPOCH1_ACCEPTANCE_RECEIPT")
    receipt = json.loads(receipt_path.read_text("utf-8"))
    need(receipt.get("status") ==
         "NONOFFICIAL_EPOCH1_G4_SAVED_NOT_OFFSESSION_VERIFIED" and
         receipt.get("source_release_sha") == plan["expected_git_head"] and
         receipt.get("b_test_access") == "NONE", "ACCEPTANCE_RECEIPT_INVALID")
    for name, digest in receipt["files_sha256"].items():
        candidate = Path(artifacts) / "g4" / name
        need(candidate.is_file() and sha256(candidate) == digest,
             "OUTPUT_CHECKSUM_FAILED")
    print("DDP_B16_GPU_ACCEPTANCE_CANDIDATE_EPOCH1_SAVED")
    print("OUTPUT_ROOT=" + str(Path(artifacts) / "g4"))
    print("MANIFEST_SHA256=" + receipt["manifest_sha256"])
    print("B_TEST=SEALED; OFFSESSION_RESUME=UNVERIFIED")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--g4-root", required=True)
    parser.add_argument("--artifacts", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--accept-one-epoch-gpu", action="store_true")
    mode.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    plan = read_plan(args.plan)
    if args.child:
        child(plan, project=args.project, g4_root=args.g4_root,
              artifacts=args.artifacts)
    else:
        parent(args.plan, project=args.project, g4_root=args.g4_root,
               artifacts=args.artifacts)


if __name__ == "__main__":
    main()
