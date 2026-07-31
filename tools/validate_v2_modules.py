from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


# ---------------------------------------------------------------------
# Ensure that Python imports Ultralytics from this repository rather
# than from a separately installed pip package.
# ---------------------------------------------------------------------
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


import ultralytics
from ultralytics import YOLO
from ultralytics.nn.modules.block import C3k2_SC
from ultralytics.nn.modules.custom import DySample, ResEMA


MODEL_YAML = (
    REPOSITORY_ROOT
    / "ultralytics"
    / "cfg"
    / "models"
    / "11"
    / "yolo11s-resema-v2.yaml"
)


def assert_tensor_finite(
    tensor: torch.Tensor,
    description: str,
) -> None:
    """Verify that a tensor contains no NaN or infinite values."""
    assert torch.isfinite(tensor).all(), (
        f"{description} contains NaN or infinite values."
    )


def collect_tensors(
    value: Any,
    output: list[torch.Tensor],
) -> None:
    """Recursively collect tensors from nested model outputs."""
    if torch.is_tensor(value):
        output.append(value)

    elif isinstance(value, dict):
        for item in value.values():
            collect_tensors(item, output)

    elif isinstance(value, (list, tuple)):
        for item in value:
            collect_tensors(item, output)


def test_repository_import() -> None:
    """Confirm that the local repository is being imported."""
    imported_path = Path(
        ultralytics.__file__
    ).resolve()

    print(
        "Imported Ultralytics from:",
        imported_path,
    )

    assert imported_path.is_relative_to(REPOSITORY_ROOT), (
        "Ultralytics was not imported from the current repository.\n"
        f"Repository root: {REPOSITORY_ROOT}\n"
        f"Imported path: {imported_path}"
    )

    print("Local repository import: PASSED")


def test_dysample() -> None:
    """Test both DySample execution styles and their initialization."""
    print("\nTesting DySample...")

    torch.manual_seed(42)

    x = torch.randn(
        2,
        64,
        16,
        20,
    )

    for style in ("lp", "pl"):
        module = DySample(
            c1=64,
            scale=2,
            style=style,
            groups=4,
            dyscope=False,
        )

        module.eval()

        with torch.no_grad():
            y = module(x)

        assert y.shape == (
            2,
            64,
            32,
            40,
        ), (
            f"Unexpected DySample {style!r} output shape: "
            f"{tuple(y.shape)}"
        )

        assert_tensor_finite(
            y,
            f"DySample {style!r} output",
        )

        # The offset layer should be initialized near std=0.001.
        offset_std = (
            module.offset.weight
            .detach()
            .float()
            .std()
            .item()
        )

        assert 0.0005 < offset_std < 0.0015, (
            f"Unexpected DySample {style!r} offset initialization. "
            f"Expected approximately 0.001 but received {offset_std:.8f}."
        )

        assert module.offset.bias is not None, (
            f"DySample {style!r} offset convolution should contain bias."
        )

        assert torch.allclose(
            module.offset.bias.detach(),
            torch.zeros_like(
                module.offset.bias.detach()
            ),
        ), (
            f"DySample {style!r} offset bias is not initialized to zero."
        )

        expected_init_channels = (
            2
            * module.groups
            * module.scale**2
        )

        assert module.init_pos.shape == (
            1,
            expected_init_channels,
            1,
            1,
        ), (
            f"Unexpected init_pos shape for DySample {style!r}: "
            f"{tuple(module.init_pos.shape)}"
        )

        print(
            f"DySample {style}: "
            f"shape={tuple(y.shape)}, "
            f"offset_std={offset_std:.6f}, "
            f"bias_zero=True"
        )

    print("DySample tests: PASSED")


def test_c3k2_sc() -> None:
    """Test expansion, repetition, grouping and shortcut configuration."""
    print("\nTesting C3k2_SC...")

    torch.manual_seed(42)

    module = C3k2_SC(
        c1=64,
        c2=128,
        n=2,
        shortcut=False,
        e=0.25,
        g=1,
        pooling_r=4,
    )

    assert module.hidden_channels == 32, (
        "C3k2_SC expansion mapping is incorrect. "
        f"Expected 32 hidden channels but received "
        f"{module.hidden_channels}."
    )

    assert module.expansion == 0.25
    assert module.groups == 1
    assert module.pooling_r == 4
    assert module.use_shortcut is False

    assert len(module.blocks) == 2, (
        f"Expected two internal SC blocks but found "
        f"{len(module.blocks)}."
    )

    assert all(
        block.use_shortcut is False
        for block in module.blocks
    )

    # Verify that pooling and group arguments reached SCConv.
    first_scconv = module.blocks[0].scconv

    assert first_scconv.k2[0].kernel_size == 4
    assert first_scconv.k2[0].stride == 4

    assert first_scconv.k2[1].groups == 1
    assert first_scconv.k3[0].groups == 1
    assert first_scconv.k4[0].groups == 1

    x = torch.randn(
        2,
        64,
        32,
        32,
    )

    module.eval()

    with torch.no_grad():
        y = module(x)

    assert y.shape == (
        2,
        128,
        32,
        32,
    ), (
        f"Unexpected C3k2_SC output shape: "
        f"{tuple(y.shape)}"
    )

    assert_tensor_finite(
        y,
        "C3k2_SC output",
    )

    # Test a shortcut-enabled block.
    residual_module = C3k2_SC(
        c1=128,
        c2=128,
        n=1,
        shortcut=True,
        e=0.5,
        g=1,
        pooling_r=4,
    )

    assert residual_module.hidden_channels == 64
    assert residual_module.use_shortcut is True
    assert residual_module.blocks[0].use_shortcut is True

    residual_module.eval()

    residual_input = torch.randn(
        2,
        128,
        16,
        16,
    )

    with torch.no_grad():
        residual_output = residual_module(
            residual_input
        )

    assert residual_output.shape == residual_input.shape

    assert_tensor_finite(
        residual_output,
        "Shortcut-enabled C3k2_SC output",
    )

    # Confirm that grouped convolutions are now operational.
    grouped_module = C3k2_SC(
        c1=64,
        c2=128,
        n=1,
        shortcut=False,
        e=0.25,
        g=4,
        pooling_r=4,
    )

    assert grouped_module.groups == 4

    grouped_scconv = (
        grouped_module
        .blocks[0]
        .scconv
    )

    assert grouped_scconv.k2[1].groups == 4
    assert grouped_scconv.k3[0].groups == 4
    assert grouped_scconv.k4[0].groups == 4

    grouped_module.eval()

    with torch.no_grad():
        grouped_output = grouped_module(x)

    assert grouped_output.shape == (
        2,
        128,
        32,
        32,
    )

    assert_tensor_finite(
        grouped_output,
        "Grouped C3k2_SC output",
    )

    print(
        "C3k2_SC: "
        f"hidden={module.hidden_channels}, "
        f"repeats={len(module.blocks)}, "
        f"output={tuple(y.shape)}"
    )

    print(
        "C3k2_SC grouped test: "
        f"groups={grouped_module.groups}, "
        f"output={tuple(grouped_output.shape)}"
    )

    print("C3k2_SC tests: PASSED")


def test_resema() -> None:
    """Test ResEMA output shape, grouping and numerical stability."""
    print("\nTesting ResEMA...")

    torch.manual_seed(42)

    module = ResEMA(
        c1=128,
        groups=8,
        reduction=2,
    )

    assert module.groups == 8
    assert module.group_channels == 16
    assert module.reduction == 2

    assert (
        module.pre_transform[0].in_channels
        == 128
    )

    assert (
        module.pre_transform[0].out_channels
        == 64
    )

    assert (
        module.pre_transform[3].in_channels
        == 64
    )

    assert (
        module.pre_transform[3].out_channels
        == 128
    )

    x = torch.randn(
        2,
        128,
        16,
        16,
    )

    module.eval()

    with torch.no_grad():
        y = module(x)

    assert y.shape == x.shape, (
        f"Unexpected ResEMA output shape: "
        f"{tuple(y.shape)}"
    )

    assert_tensor_finite(
        y,
        "ResEMA output",
    )

    print(
        "ResEMA: "
        f"groups={module.groups}, "
        f"group_channels={module.group_channels}, "
        f"output={tuple(y.shape)}"
    )

    print("ResEMA tests: PASSED")


def test_complete_model() -> YOLO:
    """Build and inspect the complete corrected YOLO11-S model."""
    print("\nTesting complete YOLO11-S ResEMA-v2 model...")

    assert MODEL_YAML.exists(), (
        "Corrected model YAML was not found:\n"
        f"{MODEL_YAML}\n"
        "Confirm that the file is named "
        "'yolo11s-resema-v2.yaml'."
    )

    model = YOLO(
        str(MODEL_YAML),
        task="detect",
    )

    detection_model = model.model
    network = detection_model.model

    detection_model.info(
        detailed=False,
        verbose=True,
        imgsz=1024,
    )

    # -------------------------------------------------------------
    # Validate filename-derived scale and number of classes.
    # -------------------------------------------------------------
    actual_scale = detection_model.yaml.get(
        "scale"
    )

    assert actual_scale == "s", (
        f"Expected model scale 's' but received "
        f"{actual_scale!r}.\n"
        "The YAML filename must contain 'yolo11s'."
    )

    assert detection_model.yaml.get("nc") == 9, (
        f"Expected nc=9 but received "
        f"{detection_model.yaml.get('nc')}."
    )

    # -------------------------------------------------------------
    # Validate the custom layer positions.
    # -------------------------------------------------------------
    custom_layers: list[tuple[int, str]] = []

    for index, layer in enumerate(network):
        layer_name = (
            layer.__class__.__name__
        )

        if layer_name in {
            "C3k2_SC",
            "DySample",
            "ResEMA",
        }:
            custom_layers.append(
                (
                    index,
                    layer_name,
                )
            )

    print("\nCustom layers:")

    for index, layer_name in custom_layers:
        print(
            f"  {index}: {layer_name}"
        )

    expected_layers = [
        (2, "C3k2_SC"),
        (4, "C3k2_SC"),
        (6, "C3k2_SC"),
        (8, "C3k2_SC"),
        (11, "DySample"),
        (14, "ResEMA"),
        (15, "DySample"),
        (18, "ResEMA"),
        (22, "ResEMA"),
        (26, "ResEMA"),
    ]

    assert custom_layers == expected_layers, (
        "The custom layer arrangement does not match "
        "the intended architecture.\n"
        f"Expected: {expected_layers}\n"
        f"Received: {custom_layers}"
    )

    # -------------------------------------------------------------
    # Validate the corrected C3k2_SC argument mapping.
    # -------------------------------------------------------------
    expected_hidden_channels = {
        2: 32,
        4: 64,
        6: 128,
        8: 256,
    }

    expected_expansions = {
        2: 0.25,
        4: 0.25,
        6: 0.50,
        8: 0.50,
    }

    expected_shortcuts = {
        2: False,
        4: False,
        6: True,
        8: True,
    }

    for layer_index in (
        2,
        4,
        6,
        8,
    ):
        layer = network[layer_index]

        assert isinstance(
            layer,
            C3k2_SC,
        )

        assert (
            layer.hidden_channels
            == expected_hidden_channels[
                layer_index
            ]
        ), (
            f"Layer {layer_index} hidden-channel error. "
            f"Expected "
            f"{expected_hidden_channels[layer_index]} "
            f"but received "
            f"{layer.hidden_channels}."
        )

        assert (
            layer.expansion
            == expected_expansions[
                layer_index
            ]
        ), (
            f"Layer {layer_index} expansion error. "
            f"Expected "
            f"{expected_expansions[layer_index]} "
            f"but received "
            f"{layer.expansion}."
        )

        assert (
            layer.use_shortcut
            is expected_shortcuts[
                layer_index
            ]
        ), (
            f"Layer {layer_index} shortcut error. "
            f"Expected "
            f"{expected_shortcuts[layer_index]} "
            f"but received "
            f"{layer.use_shortcut}."
        )

        assert layer.groups == 1
        assert layer.pooling_r == 4

        assert all(
            block.use_shortcut
            is expected_shortcuts[
                layer_index
            ]
            for block in layer.blocks
        )

        print(
            f"Layer {layer_index}: "
            f"hidden={layer.hidden_channels}, "
            f"e={layer.expansion}, "
            f"shortcut={layer.use_shortcut}, "
            f"groups={layer.groups}, "
            f"pooling_r={layer.pooling_r}"
        )

    # -------------------------------------------------------------
    # Validate the two DySample configurations.
    # -------------------------------------------------------------
    for layer_index in (
        11,
        15,
    ):
        layer = network[layer_index]

        assert isinstance(
            layer,
            DySample,
        )

        assert layer.scale == 2
        assert layer.style == "lp"
        assert layer.groups == 4
        assert layer.dyscope is False

        offset_std = (
            layer.offset.weight
            .detach()
            .float()
            .std()
            .item()
        )

        assert 0.0005 < offset_std < 0.0015

        assert layer.offset.bias is not None

        assert torch.allclose(
            layer.offset.bias.detach(),
            torch.zeros_like(
                layer.offset.bias.detach()
            ),
        )

        print(
            f"Layer {layer_index}: "
            f"DySample scale={layer.scale}, "
            f"style={layer.style}, "
            f"groups={layer.groups}, "
            f"offset_std={offset_std:.6f}"
        )

    # -------------------------------------------------------------
    # Validate the four ResEMA configurations.
    # -------------------------------------------------------------
    for layer_index in (
        14,
        18,
        22,
        26,
    ):
        layer = network[layer_index]

        assert isinstance(
            layer,
            ResEMA,
        )

        assert layer.groups == 8
        assert layer.reduction == 2

        print(
            f"Layer {layer_index}: "
            f"ResEMA groups={layer.groups}, "
            f"group_channels={layer.group_channels}, "
            f"reduction={layer.reduction}"
        )

    # -------------------------------------------------------------
    # Validate output strides.
    # -------------------------------------------------------------
    actual_strides = (
        detection_model.stride
        .detach()
        .cpu()
        .float()
    )

    expected_strides = torch.tensor(
        [
            8.0,
            16.0,
            32.0,
        ]
    )

    assert actual_strides.shape == expected_strides.shape

    assert torch.allclose(
        actual_strides,
        expected_strides,
    ), (
        f"Unexpected detection strides. "
        f"Expected {expected_strides.tolist()} "
        f"but received "
        f"{actual_strides.tolist()}."
    )

    total_parameters = sum(
        parameter.numel()
        for parameter
        in detection_model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter
        in detection_model.parameters()
        if parameter.requires_grad
    )

    print(
        "\nComplete model summary:"
    )

    print(
        f"  Scale: {actual_scale}"
    )

    print(
        f"  Classes: "
        f"{detection_model.yaml['nc']}"
    )

    print(
        f"  Strides: "
        f"{actual_strides.tolist()}"
    )

    print(
        f"  Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"  Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    print("Complete architecture checks: PASSED")

    return model


def test_complete_model_forward(
    model: YOLO,
) -> None:
    """Run a complete finite-value forward pass."""
    print("\nTesting complete-model forward pass...")

    detection_model = model.model
    detection_model.eval()

    torch.manual_seed(42)

    x = torch.randn(
        1,
        3,
        256,
        256,
    )

    with torch.no_grad():
        outputs = detection_model(x)

    tensors: list[torch.Tensor] = []

    collect_tensors(
        outputs,
        tensors,
    )

    assert tensors, (
        "No tensors were found in the complete model output."
    )

    for index, tensor in enumerate(tensors):
        assert_tensor_finite(
            tensor,
            f"Complete model output tensor {index}",
        )

    print(
        "Complete-model forward pass: "
        f"PASSED ({len(tensors)} output tensors checked)"
    )


def test_custom_gradients(
    model: YOLO,
) -> None:
    """
    Confirm that all custom modules receive finite gradients.

    A small synthetic input is used only to validate backpropagation;
    it is not a training or performance test.
    """
    print("\nTesting custom-module gradients...")

    detection_model = model.model
    detection_model.train()
    detection_model.zero_grad(
        set_to_none=True
    )

    torch.manual_seed(42)

    x = torch.randn(
        1,
        3,
        256,
        256,
        requires_grad=True,
    )

    outputs = detection_model(x)

    output_tensors: list[torch.Tensor] = []

    collect_tensors(
        outputs,
        output_tensors,
    )

    differentiable_tensors = [
        tensor
        for tensor in output_tensors
        if tensor.is_floating_point()
        and tensor.requires_grad
    ]

    assert differentiable_tensors, (
        "No differentiable tensors were found "
        "in the model output."
    )

    # Squared means produce a stable synthetic scalar objective.
    loss_terms = [
        tensor.float().square().mean()
        for tensor in differentiable_tensors
    ]

    synthetic_loss = torch.stack(
        loss_terms
    ).sum()

    assert torch.isfinite(
        synthetic_loss
    ), (
        "The synthetic gradient-test loss "
        "is NaN or infinite."
    )

    synthetic_loss.backward()

    checked_modules = 0

    for module_index, module in enumerate(
        detection_model.model
    ):
        if not isinstance(
            module,
            (
                C3k2_SC,
                DySample,
                ResEMA,
            ),
        ):
            continue

        trainable_parameters = [
            parameter
            for parameter
            in module.parameters()
            if parameter.requires_grad
        ]

        assert trainable_parameters, (
            f"Custom module at layer {module_index} "
            "has no trainable parameters."
        )

        parameters_with_gradients = [
            parameter
            for parameter
            in trainable_parameters
            if parameter.grad is not None
        ]

        assert parameters_with_gradients, (
            f"No gradients reached "
            f"{module.__class__.__name__} "
            f"at layer {module_index}."
        )

        assert all(
            torch.isfinite(
                parameter.grad
            ).all()
            for parameter
            in parameters_with_gradients
        ), (
            f"Non-finite gradients were detected in "
            f"{module.__class__.__name__} "
            f"at layer {module_index}."
        )

        gradient_norm = sum(
            parameter.grad
            .detach()
            .float()
            .norm()
            .item()
            for parameter
            in parameters_with_gradients
        )

        assert gradient_norm > 0.0, (
            f"Zero total gradient norm for "
            f"{module.__class__.__name__} "
            f"at layer {module_index}."
        )

        print(
            f"Layer {module_index}: "
            f"{module.__class__.__name__}, "
            f"gradient_norm={gradient_norm:.6e}, "
            f"parameters_with_grad="
            f"{len(parameters_with_gradients)}/"
            f"{len(trainable_parameters)}"
        )

        checked_modules += 1

    assert checked_modules == 10, (
        f"Expected to check 10 custom modules "
        f"but checked {checked_modules}."
    )

    assert x.grad is not None, (
        "No gradient reached the synthetic input."
    )

    assert_tensor_finite(
        x.grad,
        "Synthetic input gradient",
    )

    print(
        "Custom gradient test: "
        f"PASSED ({checked_modules} modules checked)"
    )

    detection_model.zero_grad(
        set_to_none=True
    )

    detection_model.eval()


def main() -> None:
    """Run all corrected-version validation checks."""
    print(
        "=" * 72
    )

    print(
        "C3k2-DySample-ResEMA Version-2 Validation"
    )

    print(
        "=" * 72
    )

    print(
        "PyTorch version:",
        torch.__version__,
    )

    print(
        "Python executable:",
        sys.executable,
    )

    test_repository_import()
    test_dysample()
    test_c3k2_sc()
    test_resema()

    model = test_complete_model()

    test_complete_model_forward(
        model
    )

    test_custom_gradients(
        model
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "ALL CORRECTED VERSION-2 CHECKS PASSED"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()