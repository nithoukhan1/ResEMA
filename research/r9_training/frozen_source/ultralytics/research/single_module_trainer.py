from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from .baseline_trainer import (
    GovernedDetectionTrainer,
    _normalize_names,
    _sha256_file,
    _state_clone,
    architecture_fingerprint,
    isolated_model_init_seed,
)


EXPECTED_NAMES = [
    "boneanomaly",
    "bonelesion",
    "foreignbody",
    "fracture",
    "metal",
    "periostealreaction",
    "pronatorsign",
    "softtissue",
    "text",
]


class GovernedSingleModuleTrainer(GovernedDetectionTrainer):
    """Governed trainer for SINGLE-MODULE-SCREEN-01."""

    def _sms_preflight(self) -> dict:
        is_resume = bool(getattr(self, "resume", False))

        path_env = (
            "RESEMA_RESUME_PREFLIGHT_MANIFEST"
            if is_resume
            else "RESEMA_PREFLIGHT_MANIFEST"
        )
        sha_env = (
            "RESEMA_RESUME_PREFLIGHT_SHA256"
            if is_resume
            else "RESEMA_PREFLIGHT_SHA256"
        )

        path = Path(os.environ.get(path_env, ""))

        if not path.is_file():
            raise RuntimeError(
                "SMS-01 governed preflight manifest is missing."
            )

        expected_sha = os.environ.get(sha_env, "")

        if not expected_sha:
            raise RuntimeError(
                "SMS-01 governed preflight SHA binding is missing."
            )

        if _sha256_file(path) != expected_sha:
            raise RuntimeError(
                "SMS-01 preflight changed after runtime binding."
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="strict",
            )
        )

        if data.get("experiment_family") != "SINGLE_MODULE_SCREEN_01":
            raise RuntimeError(
                "Wrong governed experiment family."
            )

        return data

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ):
        preflight = self._sms_preflight()
        model_contract = preflight["model_contract"]

        expected_initialization = os.environ.get(
            "RESEMA_INITIALIZATION",
            "",
        )

        configured_seed = int(self.args.seed)
        is_resume = bool(getattr(self, "resume", False))

        with isolated_model_init_seed(configured_seed):
            model = DetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=verbose and RANK == -1,
            )
            initial_state = _state_clone(model)

        dataset_nc = int(self.data["nc"])
        model_yaml_nc = int(model.yaml["nc"])
        detect_head_nc = int(
            getattr(model.model[-1], "nc", -1)
        )

        if {
            dataset_nc,
            model_yaml_nc,
            detect_head_nc,
        } != {9}:
            raise RuntimeError(
                "SMS-01 construction-time class-count mismatch."
            )

        if _normalize_names(self.data["names"]) != EXPECTED_NAMES:
            raise RuntimeError(
                "SMS-01 class-name/order contract mismatch."
            )

        parameter_count = sum(
            int(p.numel())
            for p in model.parameters()
        )

        expected_parameters = int(
            model_contract["expected_parameters"]
        )

        if parameter_count != expected_parameters:
            raise RuntimeError(
                "SMS-01 target parameter-count mismatch: "
                f"{parameter_count} != {expected_parameters}"
            )

        expected_state_items = int(
            model_contract["expected_state_items"]
        )

        if len(initial_state) != expected_state_items:
            raise RuntimeError(
                "SMS-01 target state-item mismatch: "
                f"{len(initial_state)} != {expected_state_items}"
            )

        audit = {
            "configured_seed": configured_seed,
            "model_construction_seed": configured_seed,
            "target_parameters": parameter_count,
            "target_state_items": len(initial_state),
            "target_architecture_fingerprint_sha256":
                architecture_fingerprint(initial_state),
            "dataset_nc": dataset_nc,
            "model_yaml_nc": model_yaml_nc,
            "detect_head_nc": detect_head_nc,
            "expected_initialization": expected_initialization,
            "resume": is_resume,
            "screen_family": "SINGLE_MODULE_SCREEN_01",
            "model_yaml": model_contract["model_yaml"],
        }

        # ------------------------------------------------------------
        # Resume path — exact trained candidate checkpoint restoration.
        # ------------------------------------------------------------

        if is_resume:
            if weights is None:
                raise RuntimeError(
                    "SMS-01 resume expected candidate checkpoint weights."
                )

            source_state = {
                key: value.detach().cpu()
                for key, value
                in weights.state_dict().items()
            }

            if list(source_state) != list(initial_state):
                raise RuntimeError(
                    "SMS-01 resume state-key contract mismatch."
                )

            mismatched_shapes = [
                key
                for key in initial_state
                if source_state[key].shape
                != initial_state[key].shape
            ]

            if mismatched_shapes:
                raise RuntimeError(
                    "SMS-01 resume tensor-shape mismatch: "
                    f"{mismatched_shapes}"
                )

            source_parameters = sum(
                int(p.numel())
                for p in weights.parameters()
            )

            if source_parameters != expected_parameters:
                raise RuntimeError(
                    "SMS-01 resume checkpoint parameter-count mismatch."
                )

            model.load_state_dict(
                weights.state_dict(),
                strict=True,
            )

            restored = _state_clone(model)

            if not all(
                torch.equal(
                    restored[key],
                    source_state[key],
                )
                for key in initial_state
            ):
                raise RuntimeError(
                    "SMS-01 resume checkpoint was not restored exactly."
                )

            audit.update(
                {
                    "initialization": "resume",
                    "original_initialization": expected_initialization,
                    "resume_checkpoint_loaded": True,
                    "resume_checkpoint_state_items":
                        len(source_state),
                    "resume_checkpoint_parameter_count":
                        source_parameters,
                    "resume_checkpoint_all_tensors_loaded_exactly":
                        True,
                }
            )

            self.train01_initialization_audit = audit
            return model

        # ------------------------------------------------------------
        # Fresh screen runs are pretrained-only.
        # ------------------------------------------------------------

        if expected_initialization != "pretrained":
            raise RuntimeError(
                "SINGLE-MODULE-SCREEN-01 fresh runs must be pretrained."
            )

        if weights is None:
            raise RuntimeError(
                "SMS-01 expected official pretrained weights."
            )

        source_state = {
            key: value.detach().cpu()
            for key, value
            in weights.state_dict().items()
        }

        transferable_keys = [
            key
            for key, tensor in source_state.items()
            if (
                key in initial_state
                and tensor.shape == initial_state[key].shape
            )
        ]

        expected_transferable = int(
            model_contract[
                "expected_transferable_source_items"
            ]
        )

        if len(transferable_keys) != expected_transferable:
            raise RuntimeError(
                "SMS-01 pretrained transfer-count mismatch: "
                f"{len(transferable_keys)} "
                f"!= {expected_transferable}"
            )

        nontransferable_target = {
            key
            for key, tensor in initial_state.items()
            if (
                key not in source_state
                or source_state[key].shape != tensor.shape
            )
        }

        expected_nontransferable = set(
            model_contract[
                "expected_nontransferable_target_keys"
            ]
        )

        if nontransferable_target != expected_nontransferable:
            missing = sorted(
                expected_nontransferable
                - nontransferable_target
            )
            extra = sorted(
                nontransferable_target
                - expected_nontransferable
            )

            raise RuntimeError(
                "SMS-01 nontransferable target-state drift: "
                f"missing={missing}, extra={extra}"
            )

        model.load(
            weights,
            verbose=False,
        )

        loaded_state = _state_clone(model)

        transferred_exact = all(
            torch.equal(
                loaded_state[key],
                source_state[key],
            )
            for key in transferable_keys
        )

        unmatched_preserved = all(
            torch.equal(
                loaded_state[key],
                initial_state[key],
            )
            for key in nontransferable_target
        )

        if not transferred_exact:
            raise RuntimeError(
                "SMS-01 pretrained tensors were not loaded exactly."
            )

        if not unmatched_preserved:
            raise RuntimeError(
                "SMS-01 unmatched candidate tensors did not preserve "
                "seeded initialization."
            )

        new_target_keys = set(
            model_contract["new_target_state_keys"]
        )

        baseline_nontransfer = set(
            model_contract[
                "baseline_native_nontransfer_keys"
            ]
        )

        if nontransferable_target != (
            new_target_keys | baseline_nontransfer
        ):
            raise RuntimeError(
                "SMS-01 target nontransfer partition mismatch."
            )

        audit.update(
            {
                "initialization": "pretrained",
                "official_checkpoint_loaded": True,
                "transferable_state_items":
                    len(transferable_keys),
                "baseline_native_nontransfer_items":
                    len(baseline_nontransfer),
                "new_candidate_state_items":
                    len(new_target_keys),
                "nontransferable_target_state_items":
                    len(nontransferable_target),
                "transferred_tensors_exact": True,
                "unmatched_tensors_preserved_from_seeded_initialization":
                    True,
                "new_target_state_keys":
                    sorted(new_target_keys),
            }
        )

        self.train01_initialization_audit = audit
        return model

    def _setup_train(self):
        # Retain the proven dataset/runtime/test-firewall logic.
        super()._setup_train()

        if RANK not in {-1, 0}:
            return

        preflight = self._sms_preflight()

        governance_dir = (
            self.save_dir / "governance"
        )

        governance_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        if bool(getattr(self, "resume", False)):
            session_index = int(
                preflight[
                    "resume_session_index"
                ]
            )

            runtime_dir = (
                governance_dir
                / "resume_sessions"
            )

            runtime_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            filename = (
                "SMS01_RESUME_MODEL_RUNTIME_"
                f"{session_index:03d}.json"
            )
        else:
            runtime_dir = governance_dir

            filename = (
                "SMS01_MODEL_RUNTIME.json"
            )

        path = runtime_dir / filename

        if path.exists():
            raise RuntimeError(
                f"SMS-01 refuses to overwrite {filename}."
            )

        payload = {
            "schema_version":
                "SMS01-model-runtime-v1.0",
            "experiment_family":
                "SINGLE_MODULE_SCREEN_01",
            "experiment_id":
                preflight["experiment_id"],
            "training_source_commit":
                preflight["training_source_commit"],
            "execution_commit":
                preflight["execution_commit"],
            "model_contract":
                preflight["model_contract"],
            "initialization_audit":
                self.train01_initialization_audit,
            "test_access": "NONE",
        }

        if bool(getattr(self, "resume", False)):
            payload["resume_session_index"] = (
                session_index
            )

        tmp = path.with_suffix(".json.tmp")

        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                default=str,
            ) + "\n",
            encoding="utf-8",
            newline="\n",
        )

        tmp.replace(path)