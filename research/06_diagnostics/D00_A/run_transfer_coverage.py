from __future__ import annotations

from pathlib import Path
import argparse
import csv
import gc
import hashlib
import json
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_REGISTRY = (
    Path(__file__).resolve().with_name(
        "ARCHITECTURE_REGISTRY.json"
    )
)

DEFAULT_BINDING = (
    Path(__file__).resolve().with_name(
        "D00_A0_RUNTIME_BINDING.json"
    )
)

OUTPUT_FILENAMES = (
    "transfer_coverage.csv",
    "transfer_unmatched_tensors.csv",
    "transfer_unmatched_layers.csv",
    "transfer_type_mismatch_matches.csv",
    "D00_A_RUN_PROVENANCE.json",
    "D00_A_TRANSFER_COVERAGE_REPORT.md",
)

TOP_LEVEL_KEY = re.compile(
    r"^model\.(\d+)\."
)


def stop(message: str) -> None:
    raise SystemExit(
        "ERROR: " + message
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    if result.returncode != 0:
        print(result.stdout)

        if result.stderr:
            print(
                result.stderr,
                file=sys.stderr,
            )

        stop(
            "Git command failed: "
            + " ".join(args)
        )

    return result.stdout.strip()


def require_clean_exact_repo(
    expected_sha: str,
) -> dict:
    head = run_git(
        "rev-parse",
        "HEAD",
    )

    branch = run_git(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )

    status = run_git(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )

    if head != expected_sha:
        stop(
            "Repository HEAD does not match "
            "--expected-repo-sha."
        )

    if status:
        stop(
            "Repository must be clean before "
            "D00-A execution."
        )

    return {
        "head": head,
        "branch": branch,
        "clean": True,
    }


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8",
            errors="strict",
        )
    )


def load_runtime_dependencies():
    import torch
    import ultralytics
    from ultralytics import YOLO
    from ultralytics.nn.tasks import (
        DetectionModel,
    )
    from ultralytics.utils import YAML
    from ultralytics.utils.torch_utils import (
        intersect_dicts,
    )

    package_path = Path(
        ultralytics.__file__
    ).resolve()

    repo_path = ROOT.resolve()

    try:
        package_path.relative_to(
            repo_path
        )
    except ValueError:
        stop(
            "Imported ultralytics is not from "
            "the checked-out repository. "
            f"Observed: {package_path}"
        )

    return {
        "torch": torch,
        "ultralytics":
            ultralytics,
        "YOLO": YOLO,
        "DetectionModel":
            DetectionModel,
        "YAML": YAML,
        "intersect_dicts":
            intersect_dicts,
        "package_path":
            package_path,
    }


def build_model(
    yaml_path: Path,
    *,
    scale: str,
    nc: int,
    deps: dict,
):
    if not yaml_path.is_file():
        stop(
            "Model YAML missing: "
            + str(yaml_path)
        )

    cfg = deps["YAML"].load(
        yaml_path
    )

    cfg["scale"] = scale
    cfg["yaml_file"] = str(
        yaml_path
    )
    cfg["nc"] = nc

    model = deps[
        "DetectionModel"
    ](
        cfg=cfg,
        ch=3,
        nc=nc,
        verbose=False,
    )

    return model.float()


def top_level_types(
    model,
) -> dict[int, str]:
    return {
        index:
            layer.__class__.__name__
        for index, layer
        in enumerate(model.model)
    }


def key_top_index(
    key: str,
):
    match = TOP_LEVEL_KEY.match(
        key
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def shape_string(
    tensor,
) -> str:
    return "x".join(
        str(int(x))
        for x in tensor.shape
    )


def pct(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 100.0

    return (
        100.0
        * numerator
        / denominator
    )


def validate_source_checkpoint(
    weights: Path,
    baseline_arch: dict,
    deps: dict,
):
    if not weights.is_file():
        stop(
            "Official checkpoint file does "
            "not exist: "
            + str(weights)
        )

    source_wrapper = deps[
        "YOLO"
    ](
        str(weights)
    )

    source_model = (
        source_wrapper
        .model
        .float()
    )

    baseline_yaml = (
        ROOT
        / baseline_arch[
            "yaml_path"
        ]
    )

    reference_model = build_model(
        baseline_yaml,
        scale=baseline_arch[
            "scale"
        ],
        nc=80,
        deps=deps,
    )

    source_state = (
        source_model.state_dict()
    )

    reference_state = (
        reference_model.state_dict()
    )

    if (
        set(source_state)
        != set(reference_state)
    ):
        stop(
            "Checkpoint topology does not "
            "match repository YOLO11s "
            "80-class reference keys."
        )

    bad_shapes = [
        key
        for key
        in source_state
        if (
            source_state[key].shape
            != reference_state[key].shape
        )
    ]

    if bad_shapes:
        stop(
            "Checkpoint topology shape "
            "mismatch for "
            f"{len(bad_shapes)} tensors."
        )

    source_types = (
        top_level_types(
            source_model
        )
    )

    reference_types = (
        top_level_types(
            reference_model
        )
    )

    if source_types != reference_types:
        stop(
            "Checkpoint top-level module "
            "types differ from repository "
            "YOLO11s reference."
        )

    del reference_model
    gc.collect()

    return (
        source_wrapper,
        source_model,
    )


def audit_architecture(
    arch: dict,
    source_model,
    deps: dict,
):
    target = build_model(
        ROOT / arch["yaml_path"],
        scale=arch["scale"],
        nc=int(
            arch["target_nc"]
        ),
        deps=deps,
    )

    source_state = (
        source_model.state_dict()
    )

    target_state = (
        target.state_dict()
    )

    loader_matches = deps[
        "intersect_dicts"
    ](
        source_state,
        target_state,
    )

    target_params = dict(
        target.named_parameters()
    )

    source_types = (
        top_level_types(
            source_model
        )
    )

    target_types = (
        top_level_types(
            target
        )
    )

    detect_index = int(
        target.model[-1].i
    )

    detect_prefix = (
        f"model.{detect_index}."
    )

    tensor_rows = []

    type_mismatch_rows = []

    for key, target_tensor in (
        target_state.items()
    ):
        source_tensor = (
            source_state.get(key)
        )

        target_index = (
            key_top_index(key)
        )

        source_type = (
            source_types.get(
                target_index,
                ""
            )
            if target_index
            is not None
            else ""
        )

        target_type = (
            target_types.get(
                target_index,
                ""
            )
            if target_index
            is not None
            else ""
        )

        is_parameter = (
            key in target_params
        )

        is_detect = (
            key.startswith(
                detect_prefix
            )
        )

        if key in loader_matches:
            status = (
                "MATCHED_KEY_AND_SHAPE"
            )

            same_type = (
                (
                    target_index
                    is None
                )
                or (
                    source_type
                    == target_type
                )
            )

        elif source_tensor is None:
            status = (
                "MISSING_SOURCE_KEY"
            )
            same_type = False

        else:
            status = (
                "SHAPE_MISMATCH"
            )
            same_type = False

        row = {
            "architecture_id":
                arch["id"],

            "key":
                key,

            "status":
                status,

            "is_parameter":
                int(is_parameter),

            "is_target_detect":
                int(is_detect),

            "target_numel":
                int(
                    target_tensor.numel()
                ),

            "target_shape":
                shape_string(
                    target_tensor
                ),

            "source_shape":
                (
                    shape_string(
                        source_tensor
                    )
                    if source_tensor
                    is not None
                    else ""
                ),

            "top_level_index":
                (
                    target_index
                    if target_index
                    is not None
                    else ""
                ),

            "source_top_level_type":
                source_type,

            "target_top_level_type":
                target_type,

            "same_top_level_type":
                (
                    int(same_type)
                    if status
                    == "MATCHED_KEY_AND_SHAPE"
                    else 0
                ),
        }

        tensor_rows.append(row)

        if (
            status
            == "MATCHED_KEY_AND_SHAPE"
            and not same_type
        ):
            type_mismatch_rows.append(
                row.copy()
            )

    unmatched_rows = [
        row
        for row in tensor_rows
        if row["status"]
        != "MATCHED_KEY_AND_SHAPE"
    ]

    target_state_tensors = len(
        target_state
    )

    matched_state_tensors = len(
        loader_matches
    )

    target_state_elements = sum(
        int(t.numel())
        for t in target_state.values()
    )

    matched_state_elements = sum(
        int(
            target_state[key]
            .numel()
        )
        for key in loader_matches
    )

    target_parameter_names = set(
        target_params
    )

    matched_parameter_names = {
        key
        for key in loader_matches
        if key
        in target_parameter_names
    }

    total_parameters = sum(
        int(p.numel())
        for p in target_params.values()
    )

    matched_parameters = sum(
        int(
            target_params[key]
            .numel()
        )
        for key
        in matched_parameter_names
    )

    non_detect_parameter_names = {
        key
        for key
        in target_parameter_names
        if not key.startswith(
            detect_prefix
        )
    }

    matched_non_detect_names = (
        matched_parameter_names
        & non_detect_parameter_names
    )

    non_detect_parameters = sum(
        int(
            target_params[key]
            .numel()
        )
        for key
        in non_detect_parameter_names
    )

    matched_non_detect_parameters = (
        sum(
            int(
                target_params[key]
                .numel()
            )
            for key
            in matched_non_detect_names
        )
    )

    same_type_parameter_names = {
        row["key"]
        for row in tensor_rows
        if (
            row[
                "status"
            ]
            == "MATCHED_KEY_AND_SHAPE"
            and row[
                "same_top_level_type"
            ]
            == 1
            and row[
                "is_parameter"
            ]
            == 1
        )
    }

    same_type_parameters = sum(
        int(
            target_params[key]
            .numel()
        )
        for key
        in same_type_parameter_names
    )

    same_type_non_detect_names = (
        same_type_parameter_names
        & non_detect_parameter_names
    )

    same_type_non_detect_parameters = (
        sum(
            int(
                target_params[key]
                .numel()
            )
            for key
            in same_type_non_detect_names
        )
    )

    shape_mismatch_count = sum(
        1
        for row in tensor_rows
        if row["status"]
        == "SHAPE_MISMATCH"
    )

    missing_source_count = sum(
        1
        for row in tensor_rows
        if row["status"]
        == "MISSING_SOURCE_KEY"
    )

    unexpected_source_count = sum(
        1
        for key in source_state
        if key not in target_state
    )

    unmatched_top_level_indices = {
        row["top_level_index"]
        for row in unmatched_rows
        if row["top_level_index"] != ""
    }

    summary = {
        "architecture_id":
            arch["id"],

        "label":
            arch["label"],

        "yaml_path":
            arch["yaml_path"],

        "modules":
            "+".join(
                arch["modules"]
            )
            if arch["modules"]
            else "BASELINE",

        "role":
            arch["role"],

        "target_nc":
            int(
                arch["target_nc"]
            ),

        "target_detect_index":
            detect_index,

        "target_state_tensors":
            target_state_tensors,

        "loader_matched_state_tensors":
            matched_state_tensors,

        "loader_unmatched_state_tensors":
            (
                target_state_tensors
                - matched_state_tensors
            ),

        "loader_tensor_coverage_pct":
            pct(
                matched_state_tensors,
                target_state_tensors,
            ),

        "target_state_elements":
            target_state_elements,

        "loader_matched_state_elements":
            matched_state_elements,

        "loader_state_element_coverage_pct":
            pct(
                matched_state_elements,
                target_state_elements,
            ),

        "target_parameter_tensors":
            len(
                target_parameter_names
            ),

        "loader_matched_parameter_tensors":
            len(
                matched_parameter_names
            ),

        "loader_parameter_tensor_coverage_pct":
            pct(
                len(
                    matched_parameter_names
                ),
                len(
                    target_parameter_names
                ),
            ),

        "total_parameters":
            total_parameters,

        "loader_matched_parameters":
            matched_parameters,

        "loader_parameter_coverage_pct":
            pct(
                matched_parameters,
                total_parameters,
            ),

        "non_detect_parameter_tensors":
            len(
                non_detect_parameter_names
            ),

        "loader_matched_non_detect_parameter_tensors":
            len(
                matched_non_detect_names
            ),

        "non_detect_parameters":
            non_detect_parameters,

        "loader_matched_non_detect_parameters":
            matched_non_detect_parameters,

        "loader_non_detect_parameter_coverage_pct":
            pct(
                matched_non_detect_parameters,
                non_detect_parameters,
            ),

        "same_type_matched_parameters":
            same_type_parameters,

        "same_type_parameter_coverage_pct":
            pct(
                same_type_parameters,
                total_parameters,
            ),

        "same_type_non_detect_matched_parameters":
            same_type_non_detect_parameters,

        "same_type_non_detect_parameter_coverage_pct":
            pct(
                same_type_non_detect_parameters,
                non_detect_parameters,
            ),

        "unmatched_top_level_layer_count":
            len(
                unmatched_top_level_indices
            ),

        "shape_mismatch_target_count":
            shape_mismatch_count,

        "missing_source_target_count":
            missing_source_count,

        "unexpected_source_count":
            unexpected_source_count,

        "matched_but_top_level_type_diff_count":
            len(
                type_mismatch_rows
            ),

        "build_status":
            "PASS",
    }

    del target
    gc.collect()

    return (
        summary,
        unmatched_rows,
        type_mismatch_rows,
    )



def aggregate_unmatched_layers(
    rows: list[dict],
) -> list[dict]:
    """Aggregate unmatched state-dict tensors by target top-level module."""

    grouped: dict[
        tuple[str, object, str, str],
        dict,
    ] = {}

    for row in rows:
        key = (
            row["architecture_id"],
            row["top_level_index"],
            row["source_top_level_type"],
            row["target_top_level_type"],
        )

        if key not in grouped:
            grouped[key] = {
                "architecture_id":
                    row["architecture_id"],

                "top_level_index":
                    row["top_level_index"],

                "source_top_level_type_at_same_index":
                    row["source_top_level_type"],

                "target_top_level_type":
                    row["target_top_level_type"],

                "unmatched_tensor_count":
                    0,

                "shape_mismatch_count":
                    0,

                "missing_source_key_count":
                    0,

                "unmatched_parameter_tensor_count":
                    0,

                "unmatched_parameter_numel":
                    0,

                "unmatched_state_numel":
                    0,
            }

        group = grouped[key]

        group[
            "unmatched_tensor_count"
        ] += 1

        group[
            "unmatched_state_numel"
        ] += int(
            row["target_numel"]
        )

        if (
            row["status"]
            == "SHAPE_MISMATCH"
        ):
            group[
                "shape_mismatch_count"
            ] += 1

        elif (
            row["status"]
            == "MISSING_SOURCE_KEY"
        ):
            group[
                "missing_source_key_count"
            ] += 1

        if int(
            row["is_parameter"]
        ) == 1:
            group[
                "unmatched_parameter_tensor_count"
            ] += 1

            group[
                "unmatched_parameter_numel"
            ] += int(
                row["target_numel"]
            )

    return list(
        grouped.values()
    )


def write_csv(
    path: Path,
    rows: list[dict],
    fieldnames: list[str],
) -> None:
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_report(
    path: Path,
    *,
    summaries: list[dict],
    checkpoint_sha: str,
    checkpoint_hash: str,
    repo_sha: str,
    runtime_binding_sha: str,
) -> None:
    lines = [
        "# D00-A Transfer Coverage Report",
        "",
        "Status: deterministic transfer audit output",
        "",
        "## Provenance",
        "",
        f"- Repository SHA: `{repo_sha}`",
        (
            "- Source checkpoint: "
            f"`{checkpoint_sha}`"
        ),
        (
            "- Source checkpoint SHA256: "
            f"`{checkpoint_hash}`"
        ),
        (
            "- D00-A0 runtime binding SHA256: "
            f"`{runtime_binding_sha}`"
        ),
        "- Training: NONE",
        "- Dataset inference/evaluation: NONE",
        "- Split-B test outcomes: SEALED",
        "",
        "## Coverage",
        "",
        (
            "| Architecture | Loader parameter % | "
            "Loader non-Detect % | Same-type parameter % | "
            "Same-type non-Detect % | Unmatched layers | "
            "Shape mismatch | Missing source |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in summaries:
        lines.append(
            "| "
            + str(row["label"])
            + " | "
            + f'{row["loader_parameter_coverage_pct"]:.4f}'
            + " | "
            + f'{row["loader_non_detect_parameter_coverage_pct"]:.4f}'
            + " | "
            + f'{row["same_type_parameter_coverage_pct"]:.4f}'
            + " | "
            + f'{row["same_type_non_detect_parameter_coverage_pct"]:.4f}'
            + " | "
            + str(
                row[
                    "unmatched_top_level_layer_count"
                ]
            )
            + " | "
            + str(
                row[
                    "shape_mismatch_target_count"
                ]
            )
            + " | "
            + str(
                row[
                    "missing_source_target_count"
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Definitions",
            "",
            (
                "`loader_parameter_coverage_pct` reproduces "
                "the repository's exact key + shape loading "
                "contract."
            ),
            "",
            (
                "`loader_non_detect_parameter_coverage_pct` "
                "excludes the target Detect module so the "
                "expected COCO-80 to GRAZPEDWRI-9 head "
                "difference is not misclassified as backbone/"
                "neck transfer disruption."
            ),
            "",
            (
                "`same_type_*` additionally requires the "
                "source and target tensor to belong to the "
                "same top-level module class at the same "
                "model index. It is intentionally stricter "
                "than the actual loader."
            ),
            "",
            (
                "This report is descriptive evidence only. "
                "Mechanism selection is deferred until the "
                "full D00 diagnostic gate."
            ),
            "",
        ]
    )

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "D00-A deterministic YOLO11 "
            "pretrained-transfer coverage audit."
        )
    )

    parser.add_argument(
        "--weights",
        required=True,
        type=Path,
        help=(
            "Explicit local path to the official "
            "YOLO11s pretrained checkpoint. "
            "No automatic download is performed."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--expected-repo-sha",
        required=True,
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY,
    )

    args = parser.parse_args()

    repo_state = (
        require_clean_exact_repo(
            args.expected_repo_sha
        )
    )

    registry = load_json(
        args.registry
    )

    binding = load_json(
        DEFAULT_BINDING
    )

    if (
        registry[
            "runtime_binding_sha256"
        ]
        != binding[
            "runtime_binding_sha256"
        ]
    ):
        stop(
            "Runtime-binding SHA differs "
            "between D00-A registry and "
            "D00-A0 binding record."
        )

    deps = (
        load_runtime_dependencies()
    )

    architectures = (
        registry["architectures"]
    )

    if len(architectures) != 8:
        stop(
            "Expected exactly 8 "
            "registered architectures."
        )

    baseline = next(
        architecture
        for architecture
        in architectures
        if architecture["id"]
        == "baseline_yolo11s"
    )

    weights = (
        args.weights
        .expanduser()
        .resolve()
    )

    checkpoint_hash = (
        sha256_file(weights)
    )

    (
        source_wrapper,
        source_model,
    ) = validate_source_checkpoint(
        weights,
        baseline,
        deps,
    )

    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if any(
        output_dir.iterdir()
    ):
        stop(
            "Output directory must be empty "
            "for a fresh D00-A run."
        )

    summaries = []
    unmatched = []
    type_mismatches = []

    for index, arch in enumerate(
        architectures,
        start=1,
    ):
        print(
            f"[{index}/"
            f"{len(architectures)}] "
            f"{arch['id']}"
        )

        (
            summary,
            arch_unmatched,
            arch_type_mismatches,
        ) = audit_architecture(
            arch,
            source_model,
            deps,
        )

        summaries.append(
            summary
        )

        unmatched.extend(
            arch_unmatched
        )

        type_mismatches.extend(
            arch_type_mismatches
        )

    baseline_summary = next(
        row
        for row in summaries
        if row[
            "architecture_id"
        ]
        == "baseline_yolo11s"
    )

    if abs(
        baseline_summary[
            "loader_non_detect_parameter_coverage_pct"
        ]
        - 100.0
    ) > 1e-9:
        stop(
            "Baseline non-Detect transfer "
            "coverage is not 100%; audit "
            "setup is inconsistent."
        )

    coverage_fields = list(
        summaries[0].keys()
    )

    detail_fields = [
        "architecture_id",
        "key",
        "status",
        "is_parameter",
        "is_target_detect",
        "target_numel",
        "target_shape",
        "source_shape",
        "top_level_index",
        "source_top_level_type",
        "target_top_level_type",
        "same_top_level_type",
    ]

    unmatched_layers = (
        aggregate_unmatched_layers(
            unmatched
        )
    )

    layer_fields = [
        "architecture_id",
        "top_level_index",
        "source_top_level_type_at_same_index",
        "target_top_level_type",
        "unmatched_tensor_count",
        "shape_mismatch_count",
        "missing_source_key_count",
        "unmatched_parameter_tensor_count",
        "unmatched_parameter_numel",
        "unmatched_state_numel",
    ]

    write_csv(
        output_dir
        / "transfer_coverage.csv",
        summaries,
        coverage_fields,
    )

    write_csv(
        output_dir
        / "transfer_unmatched_tensors.csv",
        unmatched,
        detail_fields,
    )

    write_csv(
        output_dir
        / "transfer_unmatched_layers.csv",
        unmatched_layers,
        layer_fields,
    )

    write_csv(
        output_dir
        / "transfer_type_mismatch_matches.csv",
        type_mismatches,
        detail_fields,
    )

    provenance = {
        "schema_version":
            "D00-A-run-provenance-v1.0",

        "repository_sha":
            repo_state["head"],

        "repository_branch":
            repo_state["branch"],

        "repository_clean":
            repo_state["clean"],

        "ultralytics_import_path":
            str(
                deps["package_path"]
            ),

        "python_version":
            sys.version,

        "torch_version":
            deps[
                "torch"
            ].__version__,

        "ultralytics_version":
            deps[
                "ultralytics"
            ].__version__,

        "source_checkpoint":
            str(weights),

        "source_checkpoint_sha256":
            checkpoint_hash,

        "source_checkpoint_policy":
            (
                "EXPLICIT_EXTERNAL_NO_AUTO_DOWNLOAD"
            ),

        "source_topology_validation":
            "PASS",

        "target_nc":
            registry["target_nc"],

        "architecture_count":
            len(architectures),

        "runtime_binding_sha256":
            binding[
                "runtime_binding_sha256"
            ],

        "training":
            "NONE",

        "dataset_model_evaluation":
            "NONE",

        "test_seal":
            "SEALED_UNTIL_P10",

        "output_files":
            list(
                OUTPUT_FILENAMES
            ),
    }

    (
        output_dir
        / "D00_A_RUN_PROVENANCE.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )

    write_report(
        output_dir
        / "D00_A_TRANSFER_COVERAGE_REPORT.md",
        summaries=summaries,
        checkpoint_sha=str(
            weights
        ),
        checkpoint_hash=checkpoint_hash,
        repo_sha=repo_state[
            "head"
        ],
        runtime_binding_sha=binding[
            "runtime_binding_sha256"
        ],
    )

    print()
    print("=" * 88)
    print(
        "D00-A TRANSFER COVERAGE AUDIT = PASS"
    )
    print(
        "ARCHITECTURES =",
        len(summaries),
    )
    print(
        "SOURCE_CHECKPOINT_SHA256 =",
        checkpoint_hash,
    )
    print(
        "BASELINE_NON_DETECT_TRANSFER_PCT =",
        (
            baseline_summary[
                "loader_non_detect_parameter_coverage_pct"
            ]
        ),
    )
    print(
        "TRAINING = NONE"
    )
    print(
        "DATASET_MODEL_EVALUATION = NONE"
    )
    print(
        "TEST = SEALED"
    )
    print(
        "OUTPUT_DIR =",
        output_dir,
    )
    print("=" * 88)

    del source_model
    del source_wrapper
    gc.collect()


if __name__ == "__main__":
    main()
