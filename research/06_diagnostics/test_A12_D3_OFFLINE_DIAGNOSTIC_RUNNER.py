from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
ENGINE_PATH = HERE / "A12_D3_OFFLINE_DIAGNOSTIC_ENGINE.py"
RUNNER_PATH = HERE / "A12_D3_OFFLINE_DIAGNOSTIC_RUNNER.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


engine = load_module("A12_D3_OFFLINE_DIAGNOSTIC_ENGINE", ENGINE_PATH)
runner = load_module("A12_D3_OFFLINE_DIAGNOSTIC_RUNNER", RUNNER_PATH)


def gt(stem: str, patient: str, box: int, cls: int, x: float, conf_shift: float = 0.0):
    w = 0.20
    h = 0.20
    return engine.GroundTruth(
        filestem=stem,
        patient_id=patient,
        box_index=box,
        class_id=cls,
        class_name=runner.CLASS_NAMES[cls],
        x_center=x,
        y_center=0.5,
        width=w,
        height=h,
        area_norm=w*h,
        size_bin=engine.size_bin(w*h),
    )


def pred(stem: str, index: int, cls: int, confidence: float, x: float):
    w = 0.20
    h = 0.20
    return engine.Prediction(
        filestem=stem,
        prediction_index=index,
        class_id=cls,
        class_name=runner.CLASS_NAMES[cls],
        x_center=x,
        y_center=0.5,
        width=w,
        height=h,
        area_norm=w*h,
        confidence=confidence,
    )


def synthetic_events():
    mapping = {"i1": "P1", "i2": "P2"}
    gt_rows = [gt("i1", "P1", 0, 3, 0.50), gt("i2", "P2", 0, 3, 0.50)]
    predictions = [
        pred("i1", 0, 3, 0.90, 0.50),  # TP
        pred("i1", 1, 3, 0.80, 0.50),  # duplicate
        pred("i2", 0, 3, 0.90, 0.80),  # background; leaves FN
    ]
    return mapping, gt_rows, engine.analyze_dataset(gt_rows, predictions, mapping, 0.05)


class TestA12D3Runner(unittest.TestCase):
    def test_class_map_is_frozen(self):
        self.assertEqual(len(runner.CLASS_NAMES), 9)
        self.assertEqual(runner.CLASS_NAMES[2], "foreignbody")
        self.assertEqual(runner.CLASS_NAMES[3], "fracture")

    def test_global_summary_conservation(self):
        _mapping, _gt, (pe, ge) = synthetic_events()
        row = runner.global_summary("M", "Model", 0.05, pe, ge)
        self.assertEqual((row["tp"], row["fp"], row["fn"]), (1, 2, 1))
        self.assertEqual(row["active_predictions"], 3)
        self.assertEqual(row["gt_support"], 2)
        self.assertEqual(row["duplicate_fp"], 1)
        self.assertEqual(row["background_fp"], 1)

    def test_per_class_foreignbody_is_na(self):
        _mapping, _gt, (pe, ge) = synthetic_events()
        rows = runner.per_class_summary("M", "Model", 0.05, pe, ge)
        foreign = rows[2]
        self.assertEqual(foreign["status"], "NO_VALIDATION_SUPPORT")
        self.assertIsNone(foreign["precision"])
        self.assertIsNone(foreign["recall"])
        self.assertIsNone(foreign["f1"])

    def test_per_class_fracture_counts(self):
        _mapping, _gt, (pe, ge) = synthetic_events()
        rows = runner.per_class_summary("M", "Model", 0.05, pe, ge)
        fracture = rows[3]
        self.assertEqual(fracture["gt_support"], 2)
        self.assertEqual((fracture["tp"], fracture["fp"], fracture["fn"]), (1, 2, 1))

    def test_gt_size_summary_uses_gt_size(self):
        _mapping, _gt, (pe, ge) = synthetic_events()
        rows = runner.gt_size_summary("M", "Model", 0.05, ge)
        large = next(row for row in rows if row["gt_size_bin"] == "medium")
        self.assertEqual(large["gt_support"], 2)
        self.assertEqual(large["tp"], 1)
        self.assertEqual(large["fn"], 1)

    def test_taxonomy_conservation(self):
        _mapping, _gt, (pe, ge) = synthetic_events()
        rows = runner.fp_taxonomy_summary("M", "Model", 0.05, pe)
        self.assertEqual(sum(row["count"] for row in rows), 2)

    def test_patient_metrics_include_all_patients(self):
        mapping, _gt, (pe, ge) = synthetic_events()
        rows = runner.patient_metrics("M", "Model", 0.05, pe, ge, sorted(set(mapping.values())))
        self.assertEqual([row["patient_id"] for row in rows], ["P1", "P2"])
        p1 = rows[0]
        p2 = rows[1]
        self.assertEqual((p1["tp"], p1["fp"], p1["fn"]), (1, 1, 0))
        self.assertEqual((p2["tp"], p2["fp"], p2["fn"]), (0, 1, 1))

    def test_prediction_identity_rejects_duplicate_index(self):
        rows = [pred("i1", 0, 3, 0.9, 0.5), pred("i1", 0, 3, 0.8, 0.6)]
        with self.assertRaises(runner.RunnerContractError):
            runner.validate_prediction_identity(rows, "M")

    def test_class_pair_rejects_wrong_name(self):
        with self.assertRaises(runner.RunnerContractError):
            runner.validate_class_pair(3, "metal", "test")

    def _bootstrap_lattice(self, candidate_improves: bool):
        rows = []
        patients = ["P1", "P2", "P3"]
        model_ids = [str(spec["experiment_id"]) for spec in runner.MODEL_SPECS]
        display = {str(spec["experiment_id"]): str(spec["display_name"]) for spec in runner.MODEL_SPECS}
        for threshold in engine.CONFIDENCE_THRESHOLDS:
            for mid in model_ids:
                for i, pid in enumerate(patients):
                    if mid == runner.BASELINE_ID:
                        tp, fp, fn, tp75 = (1, 1, 1, 1)
                    elif candidate_improves and mid == model_ids[1]:
                        tp, fp, fn, tp75 = (2, 0, 0, 2)
                    else:
                        tp, fp, fn, tp75 = (1, 1, 1, 1)
                    rows.append({
                        "experiment_id": mid,
                        "display_name": display[mid],
                        "confidence_threshold": threshold,
                        "patient_id": pid,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "tp_iou75": tp75,
                    })
        return rows

    def test_frozen_comparison_lattice_is_exact(self):
        expected = (
            (runner.BASELINE_ID, runner.EARLY_ID),
            (runner.BASELINE_ID, runner.STAGE4_ID),
            (runner.BASELINE_ID, runner.DYSAMPLE_ID),
            (runner.BASELINE_ID, runner.EMA_ID),
            (runner.BASELINE_ID, runner.COMB_ID),
            (runner.EARLY_ID, runner.EMA_ID),
            (runner.EARLY_ID, runner.STAGE4_ID),
            (runner.EARLY_ID, runner.COMB_ID),
        )
        self.assertEqual(runner.COMPARISON_SPECS, expected)

    def test_paired_bootstrap_identical_models_have_zero_delta(self):
        rows = self._bootstrap_lattice(candidate_improves=False)
        out = runner.paired_patient_bootstrap(rows, replicates=200, seed=42, chunk_size=32)
        self.assertEqual(len(out), len(runner.COMPARISON_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * 4)
        for row in out:
            self.assertAlmostEqual(row["point_delta_comparison_minus_reference"], 0.0, places=12)
            self.assertAlmostEqual(row["ci_low"], 0.0, places=12)
            self.assertAlmostEqual(row["ci_high"], 0.0, places=12)

    def test_paired_bootstrap_detects_constructed_baseline_to_early_improvement(self):
        rows = self._bootstrap_lattice(candidate_improves=True)
        out = runner.paired_patient_bootstrap(rows, replicates=200, seed=42, chunk_size=32)
        f1 = next(
            r for r in out
            if r["reference_experiment_id"] == runner.BASELINE_ID
            and r["comparison_experiment_id"] == runner.EARLY_ID
            and r["confidence_threshold"] == 0.25
            and r["metric"] == "f1"
        )
        self.assertGreater(f1["point_delta_comparison_minus_reference"], 0.0)
        self.assertGreater(f1["ci_low"], 0.0)

    def test_paired_bootstrap_includes_early_nonbaseline_contrasts(self):
        rows = self._bootstrap_lattice(candidate_improves=False)
        out = runner.paired_patient_bootstrap(rows, replicates=20, seed=42, chunk_size=7)
        pairs = {(r["reference_experiment_id"], r["comparison_experiment_id"]) for r in out}
        self.assertIn((runner.EARLY_ID, runner.EMA_ID), pairs)
        self.assertIn((runner.EARLY_ID, runner.STAGE4_ID), pairs)
        self.assertIn((runner.EARLY_ID, runner.COMB_ID), pairs)

    def test_bootstrap_is_reproducible_for_fixed_seed(self):
        rows = self._bootstrap_lattice(candidate_improves=True)
        a = runner.paired_patient_bootstrap(rows, replicates=100, seed=42, chunk_size=17)
        b = runner.paired_patient_bootstrap(rows, replicates=100, seed=42, chunk_size=17)
        self.assertEqual(a, b)

    def test_confidence_summary_is_predeclared_and_deterministic(self):
        _mapping, _gt, (pe, _ge) = synthetic_events()
        rows = runner.confidence_summary("M", "Model", 0.05, pe)
        self.assertEqual(
            [row["event_group"] for row in rows],
            ["all_predictions", "tp", "all_fp", "duplicate", "class_confusion", "localization", "background", "other_overlap"],
        )
        all_row = rows[0]
        self.assertEqual(all_row["count"], 3)
        self.assertAlmostEqual(all_row["confidence_min"], 0.8)
        self.assertAlmostEqual(all_row["confidence_max"], 0.9)
        self.assertIsNone(next(row for row in rows if row["event_group"] == "class_confusion")["confidence_mean"])

    def test_paired_patient_changes_cover_exact_lattice(self):
        rows = self._bootstrap_lattice(candidate_improves=True)
        out = runner.paired_patient_changes(rows)
        patient_count = len({row["patient_id"] for row in rows})
        self.assertEqual(len(out), len(runner.COMPARISON_SPECS) * len(engine.CONFIDENCE_THRESHOLDS) * patient_count)
        sample = next(
            r for r in out
            if r["reference_experiment_id"] == runner.BASELINE_ID
            and r["comparison_experiment_id"] == runner.EARLY_ID
            and r["confidence_threshold"] == 0.25
            and r["patient_id"] == "P1"
        )
        self.assertEqual(sample["delta_tp"], 1)
        self.assertEqual(sample["delta_fp"], -1)
        self.assertEqual(sample["delta_fn"], -1)
        self.assertGreater(sample["delta_f1"], 0.0)

    def test_write_execution_outputs_emits_complete_predeclared_schema(self):
        mapping = {"i1": "P1", "i2": "P2"}
        gt_rows = [gt("i1", "P1", 0, 3, 0.50), gt("i2", "P2", 0, 3, 0.50)]
        prediction_rows = [
            pred("i1", 0, 3, 0.90, 0.50),
            pred("i1", 1, 3, 0.80, 0.50),
            pred("i2", 0, 3, 0.90, 0.80),
        ]
        predictions = {str(spec["experiment_id"]): list(prediction_rows) for spec in runner.MODEL_SPECS}
        original_expected_patients = runner.EXPECTED_PATIENTS
        runner.EXPECTED_PATIENTS = 2
        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "out"
                manifest = runner.write_execution_outputs(
                    out, mapping, gt_rows, predictions, {"archive_sha256": "synthetic"}
                )
                expected = {
                    "A12_D3_GLOBAL_SUMMARY.csv",
                    "A12_D3_PER_CLASS_SUMMARY.csv",
                    "A12_D3_GT_SIZE_SUMMARY.csv",
                    "A12_D3_FP_TAXONOMY_SUMMARY.csv",
                    "A12_D3_FP_PREDICTION_SIZE_SUMMARY.csv",
                    "A12_D3_CONFIDENCE_SUMMARY.csv",
                    "A12_D3_PATIENT_METRICS.csv",
                    "A12_D3_PAIRED_PATIENT_CHANGES.csv",
                    "A12_D3_PAIRED_PATIENT_BOOTSTRAP.csv",
                    "A12_D3_EXECUTION_MANIFEST.json",
                }
                self.assertTrue(expected.issubset({path.name for path in out.iterdir()}))
                self.assertEqual(len(manifest["output_files"]), 57)
                self.assertEqual(
                    len(manifest["bootstrap"]["comparisons"]),
                    len(runner.COMPARISON_SPECS),
                )
        finally:
            runner.EXPECTED_PATIENTS = original_expected_patients

    def test_csv_writer_is_deterministic(self):
        rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
        with tempfile.TemporaryDirectory() as d:
            p1 = Path(d) / "a.csv"
            p2 = Path(d) / "b.csv"
            runner.write_csv(p1, rows)
            runner.write_csv(p2, rows)
            self.assertEqual(p1.read_bytes(), p2.read_bytes())

    def test_event_filename_threshold_codes(self):
        self.assertEqual(runner.event_filename("M", 0.05, "GT_EVENTS"), "M__CONF_05__GT_EVENTS.csv")
        self.assertEqual(runner.event_filename("M", 0.50, "GT_EVENTS"), "M__CONF_50__GT_EVENTS.csv")

    def test_numpy_is_available(self):
        self.assertTrue(hasattr(np.random, "default_rng"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
