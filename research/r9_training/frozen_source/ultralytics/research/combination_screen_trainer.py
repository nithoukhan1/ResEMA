from __future__ import annotations

import json
import os
from pathlib import Path

import torch

from ultralytics.nn.tasks import (
    DetectionModel,
)

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


class GovernedCombinationScreenTrainer(
    GovernedDetectionTrainer
):
    """
    Governed fresh-run and multi-session resume
    trainer for COMBINATION-SCREEN-01.
    """

    def _comb_preflight(self) -> dict:
        is_resume = bool(
            getattr(
                self,
                "resume",
                False,
            )
        )

        manifest_env = (
            "RESEMA_RESUME_PREFLIGHT_MANIFEST"
            if is_resume
            else "RESEMA_PREFLIGHT_MANIFEST"
        )

        sha_env = (
            "RESEMA_RESUME_PREFLIGHT_SHA256"
            if is_resume
            else "RESEMA_PREFLIGHT_SHA256"
        )

        path = Path(
            os.environ.get(
                manifest_env,
                "",
            )
        )

        if not path.is_file():

            raise RuntimeError(
                "COMB-01 governed preflight "
                "manifest is missing."
            )

        expected_sha = os.environ.get(
            sha_env,
            "",
        )

        if not expected_sha:

            raise RuntimeError(
                "COMB-01 governed preflight "
                "SHA binding is missing."
            )

        if (
            _sha256_file(path)
            != expected_sha
        ):

            raise RuntimeError(
                "COMB-01 preflight changed "
                "after runtime binding."
            )

        data = json.loads(
            path.read_text(
                encoding="utf-8",
                errors="strict",
            )
        )

        expected_schema = (
            "COMB01-resume-preflight-v1.0"
            if is_resume
            else "COMB01-preflight-v1.0"
        )

        if (
            data.get(
                "schema_version"
            )
            != expected_schema
        ):

            raise RuntimeError(
                "Wrong COMB-01 preflight schema."
            )

        if (
            data.get(
                "experiment_family"
            )
            != "COMBINATION_SCREEN_01"
        ):

            raise RuntimeError(
                "Wrong governed COMB-01 "
                "experiment family."
            )

        model_contract = data.get(
            "model_contract",
            {},
        )

        if (
            model_contract.get(
                "technical_verification_sha256"
            )
            !=
            "0295a6e90bbe6856517c5e723fbe58901f87d405743a8b6ecf334f335191fe29"
        ):

            raise RuntimeError(
                "COMB-01 model contract lost "
                "technical-verification binding."
            )

        technical = data.get(
            "technical_verification",
            {},
        )

        if (
            technical.get("sha256")
            !=
            "0295a6e90bbe6856517c5e723fbe58901f87d405743a8b6ecf334f335191fe29"
        ):

            raise RuntimeError(
                "COMB-01 technical-verification "
                "SHA drift."
            )

        if (
            technical.get("status")
            != "PASS"
        ):

            raise RuntimeError(
                "COMB-01 technical verification "
                "is not PASS."
            )

        test_access = data.get(
            "test_access",
            {},
        )

        for key in (
            "runtime_yaml_contains_test",
            "test_predictions",
            "test_metrics",
            "test_error_analysis",
        ):

            if (
                test_access.get(key)
                is not False
            ):

                raise RuntimeError(
                    "COMB-01 preflight test "
                    f"firewall drift: {key}"
                )

        return data


    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ):
        preflight = (
            self._comb_preflight()
        )

        model_contract = (
            preflight[
                "model_contract"
            ]
        )

        expected_initialization = (
            os.environ.get(
                "RESEMA_INITIALIZATION",
                "",
            )
        )

        configured_seed = int(
            self.args.seed
        )

        is_resume = bool(
            getattr(
                self,
                "resume",
                False,
            )
        )

        with isolated_model_init_seed(
            configured_seed
        ):

            model = DetectionModel(
                cfg,
                nc=self.data["nc"],
                ch=self.data["channels"],
                verbose=(
                    verbose
                    and RANK == -1
                ),
            )

            initial_state = (
                _state_clone(
                    model
                )
            )


        dataset_nc = int(
            self.data["nc"]
        )

        model_yaml_nc = int(
            model.yaml["nc"]
        )

        detect_head_nc = int(
            getattr(
                model.model[-1],
                "nc",
                -1,
            )
        )


        if {
            dataset_nc,
            model_yaml_nc,
            detect_head_nc,
        } != {9}:

            raise RuntimeError(
                "COMB-01 class-count mismatch."
            )


        if (
            _normalize_names(
                self.data["names"]
            )
            != EXPECTED_NAMES
        ):

            raise RuntimeError(
                "COMB-01 class-name/order "
                "contract mismatch."
            )


        parameter_count = sum(
            int(p.numel())
            for p in model.parameters()
        )

        expected_parameters = int(
            model_contract[
                "expected_parameters"
            ]
        )

        if (
            parameter_count
            != expected_parameters
        ):

            raise RuntimeError(
                "COMB-01 target parameter-count "
                f"mismatch: {parameter_count} "
                f"!= {expected_parameters}"
            )


        expected_state_items = int(
            model_contract[
                "expected_state_items"
            ]
        )

        if (
            len(initial_state)
            != expected_state_items
        ):

            raise RuntimeError(
                "COMB-01 target state-item "
                f"mismatch: {len(initial_state)} "
                f"!= {expected_state_items}"
            )


        # ------------------------------------------------------------
        # Governed resume:
        # restore the trained COMB checkpoint exactly.
        # ------------------------------------------------------------

        if is_resume:

            if (
                expected_initialization
                != "pretrained"
            ):

                raise RuntimeError(
                    "COMB-01 resume original "
                    "initialization must be pretrained."
                )

            if weights is None:

                raise RuntimeError(
                    "COMB-01 resume expected "
                    "candidate checkpoint weights."
                )

            source_state = {
                key:
                    value.detach().cpu()
                for key, value
                in weights.state_dict().items()
            }

            target_keys = list(
                initial_state
            )

            source_keys = list(
                source_state
            )

            if (
                source_keys
                != target_keys
            ):

                missing = [
                    key
                    for key in target_keys
                    if key not in source_state
                ]

                extra = [
                    key
                    for key in source_keys
                    if key not in initial_state
                ]

                raise RuntimeError(
                    "COMB-01 resume state-key "
                    "contract mismatch: "
                    f"missing={missing}, "
                    f"extra={extra}"
                )

            shape_mismatches = [
                key
                for key in target_keys
                if (
                    source_state[key].shape
                    != initial_state[key].shape
                )
            ]

            if shape_mismatches:

                raise RuntimeError(
                    "COMB-01 resume tensor-shape "
                    "mismatch: "
                    f"{shape_mismatches[:20]}"
                )

            source_parameters = sum(
                int(p.numel())
                for p in weights.parameters()
            )

            if (
                source_parameters
                != expected_parameters
            ):

                raise RuntimeError(
                    "COMB-01 resume checkpoint "
                    "parameter-count mismatch."
                )

            if (
                len(source_state)
                != expected_state_items
            ):

                raise RuntimeError(
                    "COMB-01 resume checkpoint "
                    "state-item mismatch."
                )

            model.load_state_dict(
                weights.state_dict(),
                strict=True,
            )

            restored_state = (
                _state_clone(
                    model
                )
            )

            restored_exact = all(
                torch.equal(
                    restored_state[key],
                    source_state[key],
                )
                for key in target_keys
            )

            if not restored_exact:

                raise RuntimeError(
                    "COMB-01 resume checkpoint "
                    "was not restored exactly."
                )

            audit = {
                "configured_seed":
                    configured_seed,

                "model_construction_seed":
                    configured_seed,

                "target_parameters":
                    parameter_count,

                "target_state_items":
                    len(initial_state),

                "target_architecture_fingerprint_sha256":
                    architecture_fingerprint(
                        initial_state
                    ),

                "dataset_nc":
                    dataset_nc,

                "model_yaml_nc":
                    model_yaml_nc,

                "detect_head_nc":
                    detect_head_nc,

                "expected_initialization":
                    expected_initialization,

                "initialization":
                    "resume",

                "original_initialization":
                    "pretrained",

                "resume":
                    True,

                "screen_family":
                    "COMBINATION_SCREEN_01",

                "model_yaml":
                    model_contract[
                        "model_yaml"
                    ],

                "technical_verification_sha256":
                    model_contract[
                        "technical_verification_sha256"
                    ],

                "resume_checkpoint_loaded":
                    True,

                "resume_checkpoint_state_items":
                    len(source_state),

                "resume_checkpoint_parameter_count":
                    source_parameters,

                "resume_checkpoint_all_tensors_loaded_exactly":
                    True,
            }

            self.train01_initialization_audit = (
                audit
            )

            return model


        if (
            expected_initialization
            != "pretrained"
        ):

            raise RuntimeError(
                "COMB-01 fresh runs must be "
                "pretrained."
            )


        if weights is None:

            raise RuntimeError(
                "COMB-01 expected official "
                "pretrained weights."
            )


        source_state = {
            key:
                value.detach().cpu()
            for key, value
            in weights.state_dict().items()
        }


        transferable_keys = [
            key
            for key, tensor
            in source_state.items()
            if (
                key in initial_state
                and
                tensor.shape
                == initial_state[key].shape
            )
        ]


        expected_transferable = int(
            model_contract[
                "expected_transferable_source_items"
            ]
        )


        if (
            len(transferable_keys)
            != expected_transferable
        ):

            raise RuntimeError(
                "COMB-01 pretrained transfer "
                "count mismatch: "
                f"{len(transferable_keys)} "
                f"!= {expected_transferable}"
            )


        nontransferable_target = {
            key
            for key, tensor
            in initial_state.items()
            if (
                key not in source_state
                or
                source_state[key].shape
                != tensor.shape
            )
        }


        expected_nontransferable = set(
            model_contract[
                "expected_nontransferable_target_keys"
            ]
        )


        if (
            nontransferable_target
            != expected_nontransferable
        ):

            missing = sorted(
                expected_nontransferable
                - nontransferable_target
            )

            extra = sorted(
                nontransferable_target
                - expected_nontransferable
            )

            raise RuntimeError(
                "COMB-01 nontransferable "
                "target-state drift: "
                f"missing={missing}, "
                f"extra={extra}"
            )


        model.load(
            weights,
            verbose=False,
        )


        loaded_state = (
            _state_clone(
                model
            )
        )


        transferred_exact = all(
            torch.equal(
                loaded_state[key],
                source_state[key],
            )
            for key
            in transferable_keys
        )


        unmatched_preserved = all(
            torch.equal(
                loaded_state[key],
                initial_state[key],
            )
            for key
            in nontransferable_target
        )


        if not transferred_exact:

            raise RuntimeError(
                "COMB-01 pretrained tensors "
                "were not loaded exactly."
            )


        if not unmatched_preserved:

            raise RuntimeError(
                "COMB-01 unmatched tensors did "
                "not preserve seeded initialization."
            )


        new_target_keys = set(
            model_contract[
                "new_target_state_keys"
            ]
        )

        baseline_nontransfer = set(
            model_contract[
                "baseline_native_nontransfer_keys"
            ]
        )


        if (
            nontransferable_target
            != (
                new_target_keys
                | baseline_nontransfer
            )
        ):

            raise RuntimeError(
                "COMB-01 target nontransfer "
                "partition mismatch."
            )


        gates = [
            (
                name,
                parameter,
            )
            for name, parameter
            in model.named_parameters()
            if (
                name.endswith(".alpha")
                and (
                    ".sc_adapters." in name
                    or
                    ".ema_adapter." in name
                )
            )
        ]


        if (
            len(gates) != 6
            or not all(
                torch.equal(
                    parameter.detach(),
                    torch.zeros_like(
                        parameter.detach()
                    ),
                )
                for _, parameter
                in gates
            )
        ):

            raise RuntimeError(
                "COMB-01 zero-gate "
                "initialization drift."
            )


        audit = {
            "configured_seed":
                configured_seed,

            "model_construction_seed":
                configured_seed,

            "target_parameters":
                parameter_count,

            "target_state_items":
                len(initial_state),

            "target_architecture_fingerprint_sha256":
                architecture_fingerprint(
                    initial_state
                ),

            "dataset_nc":
                dataset_nc,

            "model_yaml_nc":
                model_yaml_nc,

            "detect_head_nc":
                detect_head_nc,

            "expected_initialization":
                expected_initialization,

            "initialization":
                "pretrained",

            "resume":
                False,

            "screen_family":
                "COMBINATION_SCREEN_01",

            "model_yaml":
                model_contract[
                    "model_yaml"
                ],

            "technical_verification_sha256":
                model_contract[
                    "technical_verification_sha256"
                ],

            "official_checkpoint_loaded":
                True,

            "transferable_state_items":
                len(
                    transferable_keys
                ),

            "baseline_native_nontransfer_items":
                len(
                    baseline_nontransfer
                ),

            "new_candidate_state_items":
                len(
                    new_target_keys
                ),

            "nontransferable_target_state_items":
                len(
                    nontransferable_target
                ),

            "transferred_tensors_exact":
                True,

            "unmatched_tensors_preserved_from_seeded_initialization":
                True,

            "zero_gate_count":
                len(gates),

            "all_zero_gates_exact":
                True,

            "new_target_state_keys":
                sorted(
                    new_target_keys
                ),
        }


        self.train01_initialization_audit = (
            audit
        )

        return model


    def _setup_train(self):
        # Reuse the already-audited baseline
        # dataset/runtime/test-firewall and generic
        # RESUME-01 mechanics.
        super()._setup_train()

        if RANK not in {
            -1,
            0,
        }:
            return

        preflight = (
            self._comb_preflight()
        )

        governance_dir = (
            self.save_dir
            / "governance"
        )

        governance_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        is_resume = bool(
            getattr(
                self,
                "resume",
                False,
            )
        )

        if is_resume:

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
                "COMB01_RESUME_MODEL_RUNTIME_"
                f"{session_index:03d}.json"
            )

        else:

            session_index = None

            runtime_dir = (
                governance_dir
            )

            filename = (
                "COMB01_MODEL_RUNTIME.json"
            )

        path = (
            runtime_dir
            / filename
        )

        if path.exists():

            raise RuntimeError(
                "COMB-01 refuses to overwrite "
                f"{filename}."
            )

        payload = {
            "schema_version":
                "COMB01-model-runtime-v1.0",

            "experiment_family":
                "COMBINATION_SCREEN_01",

            "experiment_id":
                preflight[
                    "experiment_id"
                ],

            "training_source_commit":
                preflight[
                    "training_source_commit"
                ],

            "execution_commit":
                preflight[
                    "execution_commit"
                ],

            "model_contract":
                preflight[
                    "model_contract"
                ],

            "initialization_audit":
                self.train01_initialization_audit,

            "technical_verification":
                preflight[
                    "technical_verification"
                ],

            "test_access":
                "NONE",
        }

        if is_resume:

            payload[
                "resume_session_index"
            ] = session_index

        tmp = path.with_suffix(
            ".json.tmp"
        )

        tmp.write_text(
            json.dumps(
                payload,
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
            newline="\n",
        )

        tmp.replace(
            path
        )
