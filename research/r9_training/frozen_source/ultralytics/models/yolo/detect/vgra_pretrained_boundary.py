# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""R5-C2 REVIEW-ONLY: exact-byte initial-pretrained boundary for both guarded arms.

THIS FILE IS NOT DEPLOYED. Never deserialize untrusted checkpoints. Even a pinned
legacy pickle checkpoint is not an OS sandbox. Native load requires a later,
separately approved protocol and compatibility audit.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Freeze inherited from authenticated D4-E2 R2 source candidate / C1 evidence.
EARLY_ARCHIVE_SHA256 = "4e3a811be5b8126e788ca4b7854e2df2b070a08e2ab8f0c6bbde8b6e1abef396"
EARLY_BEST_SHA256 = "620594d3560310b5d24243e1b321ce4463b0795b295891c47f6ec177e5272ac1"
EARLY_BEST_BYTE_COUNT = 19507164
EARLY_ZIP_MEMBER = "ResEMA_single_module_runs/BORG-PT-S42-SCCONV-EARLY-E100/weights/best.pt"
ARM_YAMLS = {
    "ultralytics/cfg/models/11/yolo11s-tpsc-early-v1.yaml": (
        "01caddd418e280684380c4629cb2b594a8ce628c37666216b37bef4226b11882", "Detect"
    ),
    "ultralytics/cfg/models/11/yolo11s-tpsc-early-vgra-v1.yaml": (
        "cee2646cca95d337e36f3ad1a92914da2610e184e136e23596a5013b4566faed", "VGRADetect"
    ),
}


@dataclass(frozen=True, repr=False)
class ApprovedCheckpointSnapshot:
    data: bytes
    canonical_path: Path
    digest: str
    size: int


@dataclass(frozen=True, repr=False)
class ApprovedYamlSnapshot:
    data: bytes
    identity_path: str
    digest: str
    head_name: str

    def model_config(self) -> dict[str, Any]:
        """Rebind fixed arm/byte identity BEFORE importing the YAML parser."""
        if not isinstance(self.data, bytes) or not isinstance(self.identity_path, str):
            raise TypeError("D4 YAML snapshot must contain bytes and an arm identity path")
        approved = ARM_YAMLS.get(self.identity_path)
        if approved is None or (self.digest, self.head_name) != approved:
            raise ValueError("D4 YAML snapshot arm/digest/head differs from fixed approval")
        if _sha256(self.data) != approved[0]:
            raise ValueError("D4 YAML snapshot bytes differ from fixed approved digest")
        import yaml  # installed environment dependency; no filesystem lookup

        config = yaml.safe_load(self.data.decode("utf-8", errors="strict"))
        if not isinstance(config, dict) or config.get("nc") != 9:
            raise ValueError("D4 pinned model YAML must define nine detection classes")
        if "backbone" not in config or not isinstance(config.get("head"), list):
            raise ValueError("D4 pinned model YAML structure invalid")
        if not isinstance(config["head"][-1], list) or config["head"][-1][2] != self.head_name:
            raise ValueError("D4 pinned model YAML detection head differs from the arm")
        # Native tasks.yaml_model_load() derives these two metadata fields from
        # the pinned yolo11s... path; pass the same values with dict-based cfg.
        config["scale"] = "s"
        config["yaml_file"] = self.identity_path
        return config


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_no_symlink(path: str | Path, name: str) -> tuple[Path, bytes]:
    p = Path(path)
    # Basic name/ancestor-symlink guard. Windows junction/reparse/hardlink
    # handling remains an OS provenance requirement (NOT claimed here).
    if any(v.is_symlink() for v in (p, *p.parents)):
        raise ValueError(f"D4 {name}: symbolic links are not authorized")
    if not p.is_file():
        raise FileNotFoundError(f"D4 {name}: local regular file missing")
    return p.resolve(strict=True), p.read_bytes()


def _authenticate_initial_bytes(
    archive: str | Path,
    best: str | Path,
    *,
    expected_archive_sha: str,
    expected_best_sha: str,
    expected_best_size: int,
    member_name: str,
) -> ApprovedCheckpointSnapshot:
    """Pure stdlib; overrides only for synthetic tests, NOT production entry."""
    source_path, raw_best = _read_no_symlink(best, "canonical best.pt")
    if len(raw_best) != expected_best_size or _sha256(raw_best) != expected_best_sha:
        raise ValueError("D4 canonical pretrained exact snapshot size/SHA256 mismatch")
    _, raw_zip = _read_no_symlink(archive, "canonical ZIP")
    if _sha256(raw_zip) != expected_archive_sha:
        raise ValueError("D4 canonical pretrained ZIP SHA256 mismatch")
    with zipfile.ZipFile(io.BytesIO(raw_zip)) as container:
        members = [m for m in container.infolist() if m.filename == member_name]
        if len(members) != 1 or members[0].is_dir():
            raise ValueError("D4 canonical pretrained ZIP member absent or ambiguous")
        inner = container.read(members[0])  # zipfile validates member CRC
    if inner != raw_best:
        raise ValueError("D4 canonical ZIP member differs from approved checkpoint snapshot")
    return ApprovedCheckpointSnapshot(raw_best, source_path, expected_best_sha, expected_best_size)


def authenticate_initial_checkpoint(archive: str | Path, best: str | Path) -> ApprovedCheckpointSnapshot:
    """Production API: fixed published digest/size/member; no caller overrides."""
    return _authenticate_initial_bytes(
        archive,
        best,
        expected_archive_sha=EARLY_ARCHIVE_SHA256,
        expected_best_sha=EARLY_BEST_SHA256,
        expected_best_size=EARLY_BEST_BYTE_COUNT,
        member_name=EARLY_ZIP_MEMBER,
    )


def _authenticate_yaml_bytes(path: str | Path, *, expected_sha: str, head: str) -> ApprovedYamlSnapshot:
    _, data = _read_no_symlink(path, "model YAML")
    if _sha256(data) != expected_sha:
        raise ValueError("D4 model YAML bytes differ from pinned arm content SHA256")
    return ApprovedYamlSnapshot(data, str(path), expected_sha, head)


def authenticate_arm_yaml(identity_path: str) -> ApprovedYamlSnapshot:
    """Fixed two-arm path/content policy, not caller-configurable."""
    if identity_path not in ARM_YAMLS:
        raise ValueError("D4 unrecognized model YAML arm path")
    expected_sha, head = ARM_YAMLS[identity_path]
    return _authenticate_yaml_bytes(identity_path, expected_sha=expected_sha, head=head)


def _validate_canonical_checkpoint_snapshot(snapshot: ApprovedCheckpointSnapshot) -> None:
    """Rebind the immutable checkpoint identity, not just caller-declared fields."""
    if not isinstance(snapshot, ApprovedCheckpointSnapshot):
        raise TypeError("D4 unapproved checkpoint snapshot type")
    if not isinstance(snapshot.data, bytes) or not isinstance(snapshot.canonical_path, Path):
        raise TypeError("D4 canonical checkpoint snapshot types are invalid")
    if snapshot.digest != EARLY_BEST_SHA256 or snapshot.size != EARLY_BEST_BYTE_COUNT:
        raise ValueError("D4 checkpoint snapshot differs from fixed canonical SHA256/size")
    if len(snapshot.data) != EARLY_BEST_BYTE_COUNT or _sha256(snapshot.data) != EARLY_BEST_SHA256:
        raise ValueError("D4 checkpoint memory bytes differ from fixed canonical content")


def _consume_approved_bytes(snapshot: ApprovedCheckpointSnapshot, consumer):
    """Revalidate the canonical bytes before passing an in-memory stream."""
    _validate_canonical_checkpoint_snapshot(snapshot)
    return consumer(io.BytesIO(snapshot.data))


def deserialize_initial_pretrained(snapshot: ApprovedCheckpointSnapshot, *, device=None):
    """No native torch_safe_load/load_checkpoint; no download/install/retry path.

    WARNING: torch_load(...weights_only=False) performs legacy pickle loading.
    Approved byte identity is necessary but not a pickle-execution sandbox.
    Real use is HELD pending separate C2 compatibility and execution approval.
    """
    _validate_canonical_checkpoint_snapshot(snapshot)  # must precede all native imports
    import torch
    from ultralytics.nn.tasks import guess_model_task, temporary_modules
    from ultralytics.utils import DEFAULT_CFG_DICT
    from ultralytics.utils.patches import torch_load

    def single_attempt(stream):
        with temporary_modules(
            modules={
                "ultralytics.yolo.utils": "ultralytics.utils",
                "ultralytics.yolo.v8": "ultralytics.models.yolo",
                "ultralytics.yolo.data": "ultralytics.data",
            },
            attributes={
                "ultralytics.nn.modules.block.Silence": "torch.nn.Identity",
                "ultralytics.nn.tasks.YOLOv10DetectionModel": "ultralytics.nn.tasks.DetectionModel",
                "ultralytics.utils.loss.v10DetectLoss": "ultralytics.utils.loss.E2EDetectLoss",
            },
        ):
            return torch_load(stream, map_location="cpu", weights_only=False)

    ckpt = _consume_approved_bytes(snapshot, single_attempt)  # EXACT bytes, no path reopening
    if not isinstance(ckpt, dict):
        raise TypeError("D4 approved canonical checkpoint must deserialize to native dictionary")
    source = ckpt.get("ema") or ckpt.get("model")
    if not isinstance(source, torch.nn.Module):
        raise TypeError("D4 approved checkpoint has no native model/EMA")
    args = ckpt.get("train_args", {})
    if not isinstance(args, dict):
        raise TypeError("D4 approved checkpoint train_args not a dictionary")
    model = source.float()
    model.args = {**DEFAULT_CFG_DICT, **args}
    model.pt_path = str(snapshot.canonical_path)  # metadata only, never reopened here
    model.task = getattr(model, "task", guess_model_task(model))
    if not hasattr(model, "stride"):
        model.stride = torch.tensor([32.0])
    model = model.eval().to(device)
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = True  # native load_checkpoint(inplace=True)
        elif isinstance(module, torch.nn.Upsample) and not hasattr(module, "recompute_scale_factor"):
            module.recompute_scale_factor = None
    return model, ckpt
