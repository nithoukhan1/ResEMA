import importlib.util
from pathlib import Path
import unittest

ENGINE_PATH = Path(__file__).with_name("A12_D3_OFFLINE_DIAGNOSTIC_ENGINE.py")
spec = importlib.util.spec_from_file_location("a12_d3_engine", ENGINE_PATH)
engine = importlib.util.module_from_spec(spec)
assert spec.loader is not None
import sys
sys.modules[spec.name] = engine
spec.loader.exec_module(engine)


def gt(box_index, class_id, x, y, w, h, patient="P1", stem="img"):
    area = w * h
    return engine.GroundTruth(
        filestem=stem,
        patient_id=patient,
        box_index=box_index,
        class_id=class_id,
        class_name=f"c{class_id}",
        x_center=x,
        y_center=y,
        width=w,
        height=h,
        area_norm=area,
        size_bin=engine.size_bin(area),
    )


def pred(index, class_id, conf, x, y, w, h, stem="img"):
    return engine.Prediction(
        filestem=stem,
        prediction_index=index,
        class_id=class_id,
        class_name=f"c{class_id}",
        x_center=x,
        y_center=y,
        width=w,
        height=h,
        area_norm=w*h,
        confidence=conf,
    )


class TestA12D3Engine(unittest.TestCase):
    def test_size_bin_boundaries(self):
        self.assertEqual(engine.size_bin(0.009999), "small")
        self.assertEqual(engine.size_bin(0.01), "medium")
        self.assertEqual(engine.size_bin(0.049999), "medium")
        self.assertEqual(engine.size_bin(0.05), "large")

    def test_iou_identity(self):
        a = gt(0, 3, 0.5, 0.5, 0.2, 0.2)
        b = pred(0, 3, 0.9, 0.5, 0.5, 0.2, 0.2)
        self.assertAlmostEqual(engine.iou_xywh(a, b), 1.0, places=12)

    def test_equal_iou_tie_break_uses_prediction_index_not_confidence(self):
        # Both predictions have IoU=1.0 to the same GT. The repository protocol
        # does not rank by confidence; exact-IoU ties use frozen prediction_index.
        g = [gt(0, 3, 0.5, 0.5, 0.2, 0.2)]
        p = [
            pred(0, 3, 0.60, 0.5, 0.5, 0.2, 0.2),
            pred(1, 3, 0.99, 0.5, 0.5, 0.2, 0.2),
        ]
        pe, ge = engine.match_image(g, p, 0.05)
        by_index = {row["prediction_index"]: row["event_type"] for row in pe}
        self.assertEqual(by_index[0], "tp")
        self.assertEqual(by_index[1], "duplicate")
        self.assertEqual(ge[0]["matched_prediction_index"], 0)


    def test_global_pair_iou_matching_is_normative_not_confidence_first(self):
        # High-confidence prediction overlaps both GTs above 0.50, while the
        # lower-confidence prediction exactly matches GT0. Global pair-IoU
        # sorting first assigns the exact pair, then preserves a second TP on GT1.
        # A confidence-first matcher would incorrectly produce TP+duplicate+FN.
        g = [
            gt(0, 3, 0.40, 0.5, 0.25, 0.20),
            gt(1, 3, 0.55, 0.5, 0.25, 0.20),
        ]
        p = [
            pred(0, 3, 0.90, 0.475, 0.5, 0.25, 0.20),
            pred(1, 3, 0.80, 0.40, 0.5, 0.25, 0.20),
        ]
        pe, ge = engine.match_image(g, p, 0.05)
        self.assertEqual(sum(row["event_type"] == "tp" for row in pe), 2)
        self.assertEqual(sum(row["event_type"] == "tp" for row in ge), 2)
        self.assertEqual(sum(row["event_type"] == "fn" for row in ge), 0)
        gt_to_pred = {row["box_index"]: row["matched_prediction_index"] for row in ge}
        self.assertEqual(gt_to_pred[0], 1)
        self.assertEqual(gt_to_pred[1], 0)

    def test_class_confusion_precedes_localization(self):
        # Prediction class 1 overlaps class-2 GT perfectly and a class-1 GT only weakly.
        g = [
            gt(0, 2, 0.5, 0.5, 0.2, 0.2),
            gt(1, 1, 0.65, 0.5, 0.2, 0.2),
        ]
        p = [pred(0, 1, 0.9, 0.5, 0.5, 0.2, 0.2)]
        pe, _ = engine.match_image(g, p, 0.05)
        self.assertEqual(pe[0]["event_type"], "class_confusion")
        self.assertEqual(pe[0]["reference_gt_class_id"], 2)

    def test_localization(self):
        g = [gt(0, 3, 0.5, 0.5, 0.2, 0.2)]
        p = [pred(0, 3, 0.9, 0.62, 0.5, 0.2, 0.2)]
        pe, ge = engine.match_image(g, p, 0.05)
        self.assertEqual(pe[0]["event_type"], "localization")
        self.assertEqual(ge[0]["event_type"], "fn")
        self.assertGreaterEqual(pe[0]["max_iou_same_class"], 0.10)
        self.assertLess(pe[0]["max_iou_same_class"], 0.50)

    def test_background(self):
        g = [gt(0, 3, 0.2, 0.2, 0.1, 0.1)]
        p = [pred(0, 3, 0.9, 0.8, 0.8, 0.1, 0.1)]
        pe, _ = engine.match_image(g, p, 0.05)
        self.assertEqual(pe[0]["event_type"], "background")

    def test_other_overlap(self):
        # Different-class overlap below 0.50 but above 0.10 and no same-class GT.
        g = [gt(0, 2, 0.5, 0.5, 0.2, 0.2)]
        p = [pred(0, 1, 0.9, 0.62, 0.5, 0.2, 0.2)]
        pe, _ = engine.match_image(g, p, 0.05)
        self.assertEqual(pe[0]["event_type"], "other_overlap")

    def test_confidence_threshold_filter(self):
        g = [gt(0, 3, 0.5, 0.5, 0.2, 0.2)]
        p = [pred(0, 3, 0.049, 0.5, 0.5, 0.2, 0.2)]
        pe, ge = engine.match_image(g, p, 0.05)
        self.assertEqual(pe, [])
        self.assertEqual(ge[0]["event_type"], "fn")

    def test_iou75_flag_is_reporting_only(self):
        g = [gt(0, 3, 0.5, 0.5, 0.2, 0.2)]
        p = [pred(0, 3, 0.9, 0.52, 0.5, 0.2, 0.2)]
        pe, ge = engine.match_image(g, p, 0.05)
        self.assertEqual(pe[0]["event_type"], "tp")
        self.assertTrue(pe[0]["strict_iou75"])
        self.assertTrue(ge[0]["strict_iou75"])

    def test_prediction_tie_break_uses_prediction_index(self):
        g = [gt(0, 3, 0.5, 0.5, 0.2, 0.2)]
        p = [
            pred(9, 3, 0.9, 0.5, 0.5, 0.2, 0.2),
            pred(2, 3, 0.9, 0.5, 0.5, 0.2, 0.2),
        ]
        pe, ge = engine.match_image(g, p, 0.05)
        self.assertEqual(pe[0]["prediction_index"], 2)
        self.assertEqual(pe[0]["event_type"], "tp")
        self.assertEqual(ge[0]["matched_prediction_index"], 2)

    def test_dataset_patient_binding_and_conservation(self):
        g = [gt(0, 3, 0.5, 0.5, 0.2, 0.2, patient="PX", stem="im1")]
        p = [pred(0, 3, 0.9, 0.5, 0.5, 0.2, 0.2, stem="im1")]
        pe, ge = engine.analyze_dataset(g, p, {"im1": "PX", "im2": "PY"}, 0.05)
        self.assertEqual(pe[0]["patient_id"], "PX")
        self.assertEqual(len(ge), 1)
        metrics = engine.precision_recall_f1(pe, ge)
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["fn"]), (1, 0, 0))

    def test_empty_gt_image_prediction_is_background(self):
        p = [pred(0, 3, 0.9, 0.5, 0.5, 0.2, 0.2, stem="im2")]
        pe, ge = engine.analyze_dataset([], p, {"im2": "PY"}, 0.05)
        self.assertEqual(ge, [])
        self.assertEqual(pe[0]["event_type"], "background")


if __name__ == "__main__":
    unittest.main(verbosity=2)
