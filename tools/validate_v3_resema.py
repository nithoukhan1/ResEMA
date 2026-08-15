from __future__ import annotations

import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules.custom import (
    EMAOfficial,
    ResEMA_V3,
)


MODEL_DIR = (
    REPOSITORY_ROOT
    / "ultralytics"
    / "cfg"
    / "models"
    / "11"
)


def assert_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:
    assert torch.isfinite(tensor).all(), (
        f"{name} contains NaN or Inf."
    )


def test_local_import() -> None:
    imported = Path(
        ultralytics.__file__
    ).resolve()

    print("Imported Ultralytics from:")
    print(imported)

    assert imported.is_relative_to(
        REPOSITORY_ROOT
    ), (
        "Ultralytics was not imported from this repository."
    )


def test_ema_core() -> None:
    print("\nTesting EMAOfficial...")

    torch.manual_seed(42)

    for channels, expected_params in (
        (128, 176),
        (256, 672),
        (512, 2624),
    ):

        module = EMAOfficial(
            c1=channels,
            factor=32,
        )

        x = torch.randn(
            2,
            channels,
            16,
            20,
            requires_grad=True,
        )

        y = module(x)

        assert y.shape == x.shape
        assert_finite(
            y,
            f"EMAOfficial c={channels}",
        )

        params = sum(
            p.numel()
            for p in module.parameters()
        )

        assert params == expected_params, (
            f"EMAOfficial({channels}) expected "
            f"{expected_params} parameters, got {params}."
        )

        loss = y.mean()
        loss.backward()

        assert x.grad is not None
        assert_finite(
            x.grad,
            f"EMAOfficial gradient c={channels}",
        )

        print(
            f"EMAOfficial c={channels}: "
            f"params={params}, "
            f"shape={tuple(y.shape)} — PASSED"
        )


def test_resema_v3() -> None:
    print("\nTesting ResEMA_V3...")

    torch.manual_seed(42)

    module = ResEMA_V3(
        c1=128,
        factor=32,
    )

    x = torch.randn(
        2,
        128,
        16,
        20,
    )

    module.eval()

    with torch.no_grad():

        core = module.ema(x)
        output = module(x)

    assert output.shape == x.shape

    assert_finite(
        core,
        "EMA core output",
    )

    assert_finite(
        output,
        "ResEMA_V3 output",
    )

    assert torch.allclose(
        output,
        x + core,
        atol=1e-6,
        rtol=1e-5,
    ), (
        "ResEMA_V3 residual mapping is incorrect."
    )

    print(
        "ResEMA_V3 residual mapping: PASSED"
    )


def test_invalid_factor() -> None:

    try:
        EMAOfficial(
            c1=130,
            factor=32,
        )
    except ValueError:
        print(
            "Invalid channel/factor check: PASSED"
        )
    else:
        raise AssertionError(
            "Expected invalid channel/factor combination "
            "to raise ValueError."
        )


def test_models() -> None:

    configurations = {
        "yolo11s-resema-v3-only.yaml": {
            "counts": (0, 0, 4),
            "params": 9_435_419,
            "positions": [14, 18, 22, 26],
        },

        "yolo11s-resema-v3-e50.yaml": {
            "counts": (4, 2, 4),
            "params": 11_155_259,
            "positions": [14, 18, 22, 26],
        },
    }

    for filename, expected in configurations.items():

        print(
            "\n" + "=" * 70
        )
        print("Testing:", filename)
        print("=" * 70)

        yaml_path = (
            MODEL_DIR
            / filename
        )

        assert yaml_path.exists(), yaml_path

        model = YOLO(
            str(yaml_path),
            task="detect",
            verbose=False,
        )

        network = list(
            model.model.model
        )

        sc_count = sum(
            layer.__class__.__name__
            == "C3k2_SC"
            for layer in network
        )

        dy_count = sum(
            layer.__class__.__name__
            == "DySample"
            for layer in network
        )

        v3_count = sum(
            layer.__class__.__name__
            == "ResEMA_V3"
            for layer in network
        )

        legacy_resema_count = sum(
            layer.__class__.__name__
            == "ResEMA"
            for layer in network
        )

        counts = (
            sc_count,
            dy_count,
            v3_count,
        )

        assert counts == expected["counts"], (
            f"{filename}: expected {expected['counts']}, "
            f"got {counts}."
        )

        assert legacy_resema_count == 0, (
            f"{filename} unexpectedly contains V2 ResEMA."
        )

        positions = [
            index
            for index, layer in enumerate(network)
            if layer.__class__.__name__
            == "ResEMA_V3"
        ]

        assert positions == expected["positions"], (
            f"{filename}: unexpected ResEMA_V3 "
            f"positions {positions}."
        )

        assert model.model.yaml["scale"] == "s"

        assert model.model.yaml["nc"] == 9

        assert model.model.stride.tolist() == [
            8.0,
            16.0,
            32.0,
        ]

        params = sum(
            p.numel()
            for p in model.model.parameters()
        )

        assert params == expected["params"], (
            f"{filename}: expected "
            f"{expected['params']:,} params, "
            f"got {params:,}."
        )

        print(
            f"SC={sc_count}, "
            f"DySample={dy_count}, "
            f"ResEMA_V3={v3_count}"
        )

        print(
            f"Parameters={params:,}"
        )

        print(
            f"Strides={model.model.stride.tolist()}"
        )

        print("MODEL BUILD: PASSED")


if __name__ == "__main__":

    test_local_import()
    test_ema_core()
    test_resema_v3()
    test_invalid_factor()
    test_models()

    print(
        "\nALL RESEMA-V3 VALIDATION TESTS PASSED."
    )