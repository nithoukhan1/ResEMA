"""D4-E R5 REVIEW CANDIDATE: fixed-conf B-VAL metrics from already-decoded native batches.

DO NOT DEPLOY UNREVIEWED. No checkpoint/data loading; calls the pinned native
DetectionValidator _prepare_batch/_prepare_pred/_process_batch for each image.
Crucially re-runs IoU=.50 class-aware matching AFTER confidence > .25 filter.
Must be incorporated IDENTICALLY into both paired validators, and only after
training=True plus explicit B-VAL dataset binding. This module does not itself
start validation or authorize B-TEST access.
"""
from __future__ import annotations

import math

import numpy as np
import torch

CLASS_ORDER = (
    "boneanomaly", "bonelesion", "foreignbody", "fracture", "metal",
    "periostealreaction", "pronatorsign", "softtissue", "text",
)
FIXED_CONF = 0.25
MATCH_IOU = 0.50


class FixedThresholdBValMixin:
    """Source-review candidate; caller must enforce governed pair-aware B-VAL loader."""

    def _d4_init_fixed_bval(self):
        if not bool(getattr(self, 'training', False)):
            raise RuntimeError('D4_FIXED_CONF_REQUIRES_DEVELOPMENT_TRAINER')
        if int(getattr(self, 'nc', -1)) != 9:
            raise ValueError('D4_FIXED_CONF_NINE_CLASSES_REQUIRED')
        if tuple(self.names[i] for i in range(9)) != CLASS_ORDER:
            raise ValueError('D4_FIXED_CONF_CLASS_ORDER_DRIFT')
        if not hasattr(self, 'iouv') or not math.isclose(float(self.iouv[0]), MATCH_IOU, abs_tol=1e-6):
            raise ValueError('D4_FIXED_CONF_NATIVE_IOU_GRID_DRIFT')
        # For fixed 0.25 evaluation NMS must not have pruned >0.25 predictions.
        low_conf = getattr(self.args, 'conf', None)
        if low_conf is not None and float(low_conf) > FIXED_CONF:
            raise ValueError('D4_FIXED_CONF_NMS_FLOOR_TOO_HIGH')
        self._d4_fixed_counters = torch.zeros((9, 3), dtype=torch.long)  # TP FP FN
        self._d4_fixed_seen = 0

    @torch.no_grad()
    def _d4_update_fixed_bval(self, detections, batch):
        if not hasattr(self, '_d4_fixed_counters'):
            raise RuntimeError('D4_FIXED_CONF_NOT_INITIALIZED')
        if len(detections) != len(batch['im_file']):
            raise RuntimeError('D4_FIXED_CONF_BATCH_CARDINALITY')
        for si, pred in enumerate(detections):
            ground = self._prepare_batch(si, batch)
            # Never mutate native predictions or native metric state.
            prepared = self._prepare_pred({k: v.clone() if isinstance(v, torch.Tensor) else v
                                          for k, v in pred.items()})
            conf = prepared['conf']
            keep = conf > FIXED_CONF  # matches pinned ConfusionMatrix's strict > convention
            selected = {k: v[keep] for k, v in prepared.items()}
            gt_classes = ground['cls'].long().detach().cpu()
            pred_classes = selected['cls'].long().detach().cpu()
            if any(int(x) < 0 or int(x) >= 9 for x in gt_classes.tolist() + pred_classes.tolist()):
                raise ValueError('D4_FIXED_CONF_CLASS_ID_DRIFT')
            # Native _process_batch executes IoU/class matching on thresholded predictions.
            # NEVER slice the precomputed low-conf TP flags: low-conf matches can steal a GT.
            flags = self._process_batch(selected, ground)['tp']
            if flags.shape != (len(pred_classes), len(self.iouv)):
                raise RuntimeError('D4_FIXED_CONF_NATIVE_MATCH_SHAPE')
            tp50 = flags[:, 0]
            for cls in range(9):
                is_pred = pred_classes.numpy() == cls
                cls_tp = int(np.count_nonzero(tp50[is_pred]))
                cls_pred = int(np.count_nonzero(is_pred))
                cls_gt = int((gt_classes == cls).sum().item())
                if cls_tp > cls_pred or cls_tp > cls_gt:
                    raise RuntimeError('D4_FIXED_CONF_ACCOUNTING_MISMATCH')
                self._d4_fixed_counters[cls, 0] += cls_tp
                self._d4_fixed_counters[cls, 1] += cls_pred - cls_tp
                self._d4_fixed_counters[cls, 2] += cls_gt - cls_tp
            self._d4_fixed_seen += 1

    def _d4_fixed_class_counts_for_private_report(self):
        """Aggregate-only per-class TP/FP/FN; NEVER insert lists into native metrics."""
        return {name: tuple(int(v) for v in self._d4_fixed_counters[i].tolist())
                for i, name in enumerate(CLASS_ORDER)}

    def _d4_finalize_fixed_bval(self):
        if not hasattr(self, '_d4_fixed_counters') or self._d4_fixed_seen <= 0:
            raise RuntimeError('D4_FIXED_CONF_EMPTY_EVALUATION')
        counts = self._d4_fixed_counters.tolist()
        tp, fp, fn = (sum(row[j] for row in counts) for j in range(3))
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        return {
            'd4_fixed_conf_025_iou_050_tp': tp,
            'd4_fixed_conf_025_iou_050_fp': fp,
            'd4_fixed_conf_025_iou_050_fn': fn,
            'd4_fixed_conf_025_iou_050_precision_micro': p,
            'd4_fixed_conf_025_iou_050_recall_micro': r,
            'd4_fixed_conf_025_iou_050_f1_micro': f1,
            'd4_fixed_conf_025_eval_images': self._d4_fixed_seen,
        }


if __name__ == '__main__':
    raise SystemExit('REVIEW_ONLY_NO_STANDALONE_VALIDATION')
