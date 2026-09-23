from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import torch

from ultralytics.research.baseline_trainer import (
    EXPECTED_NONTRANSFERABLE_KEYS,
    EXPECTED_TARGET_PARAMETERS,
    EXPECTED_TARGET_STATE_ITEMS,
    EXPECTED_TRANSFERABLE_STATE_ITEMS,
    isolated_model_init_seed,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = ROOT / "research/runtime/baseline_runner.py"


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "train01_baseline_runner_unit",
        RUNNER_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train01_isolated_model_seed_restores_outer_rng():
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)

    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()

    with isolated_model_init_seed(42):
        first = torch.rand(8)

    assert random.getstate() == py_state

    observed_np = np.random.get_state()
    assert observed_np[0] == np_state[0]
    assert np.array_equal(observed_np[1], np_state[1])
    assert observed_np[2:] == np_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)

    with isolated_model_init_seed(42):
        second = torch.rand(8)

    assert torch.equal(first, second)


def test_train01_init_contract_constants_match_init01():
    assert EXPECTED_TARGET_PARAMETERS == 9431275
    assert EXPECTED_TARGET_STATE_ITEMS == 499
    assert EXPECTED_TRANSFERABLE_STATE_ITEMS == 493
    assert EXPECTED_NONTRANSFERABLE_KEYS == [
        "model.23.cv3.0.2.weight",
        "model.23.cv3.0.2.bias",
        "model.23.cv3.1.2.weight",
        "model.23.cv3.1.2.bias",
        "model.23.cv3.2.2.weight",
        "model.23.cv3.2.2.bias",
    ]


def test_train01_build_training_args_is_frozen_and_val_only():
    runner = load_runner_module()
    training = runner.load_yaml(runner.TRAINING_YAML)

    args = runner.build_training_args(
        training,
        experiment_id="BASE-B-ORG-PT-S42",
        runtime_data_yaml=Path("/tmp/runtime_data_train_val_only.yaml"),
    )

    assert args["epochs"] == 100
    assert args["patience"] == 100
    assert args["imgsz"] == 1024
    assert args["batch"] == 16
    assert args["optimizer"] == "SGD"
    assert args["seed"] == 42
    assert args["device"] == [0, 1]
    assert args["workers"] == 4
    assert args["split"] == "val"
    assert args["exist_ok"] is False
    assert args["save_period"] == -1
