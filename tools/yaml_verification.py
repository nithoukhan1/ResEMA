from __future__ import annotations

import sys
from pathlib import Path


# Import Ultralytics from this repository, not site-packages.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


import ultralytics
from ultralytics import YOLO


MODEL_DIR = (
    REPOSITORY_ROOT
    / "ultralytics"
    / "cfg"
    / "models"
    / "11"
)


EXPECTED = {
    "yolo11s-sc-v2.yaml": (4, 0, 0),
    "yolo11s-dysample-v2.yaml": (0, 2, 0),
    "yolo11s-resema-only-v2.yaml": (0, 0, 4),
    "yolo11s-sc-dysample-v2.yaml": (4, 2, 0),
    "yolo11s-sc-resema-v2.yaml": (4, 0, 4),
    "yolo11s-dysample-resema-v2.yaml": (0, 2, 4),
}


print("Imported Ultralytics from:")
print(Path(ultralytics.__file__).resolve())

assert Path(ultralytics.__file__).resolve().is_relative_to(
    REPOSITORY_ROOT
), "Ultralytics was not imported from the local repository."


for filename, expected_counts in EXPECTED.items():
    yaml_path = MODEL_DIR / filename

    assert yaml_path.exists(), (
        f"YAML file not found: {yaml_path}"
    )

    model = YOLO(
        str(yaml_path),
        task="detect",
        verbose=False,
    )

    layers = list(model.model.model)

    actual_counts = (
        sum(
            layer.__class__.__name__ == "C3k2_SC"
            for layer in layers
        ),
        sum(
            layer.__class__.__name__ == "DySample"
            for layer in layers
        ),
        sum(
            layer.__class__.__name__ == "ResEMA"
            for layer in layers
        ),
    )

    assert actual_counts == expected_counts, (
        f"{filename}\n"
        f"Expected counts: {expected_counts}\n"
        f"Actual counts:   {actual_counts}"
    )

    assert model.model.yaml["scale"] == "s", (
        f"{filename}: model scale is not 's'."
    )

    assert model.model.yaml["nc"] == 9, (
        f"{filename}: expected nc=9."
    )

    assert model.model.stride.tolist() == [
        8.0,
        16.0,
        32.0,
    ], (
        f"{filename}: unexpected detection strides "
        f"{model.model.stride.tolist()}."
    )

    parameters = sum(
        parameter.numel()
        for parameter in model.model.parameters()
    )

    print(
        f"{filename}: "
        f"SC={actual_counts[0]}, "
        f"DySample={actual_counts[1]}, "
        f"ResEMA={actual_counts[2]}, "
        f"parameters={parameters:,} — PASSED"
    )


print("\nAll six corrected ablation YAMLs passed.")
