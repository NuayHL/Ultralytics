"""
Custom validation backend that collects GT area and prediction score pairs
for each detection during validation.  Works for both YOLO and RT-DETR models.

Usage:
    from area_score_val import AreaScoreValidator
    model.val(validator=AreaScoreValidator, data="...", ...)

    # Or let val_area_score() in myval_area_statics auto-select:
    from area_score_val import get_area_score_validator
    ValidCls = get_area_score_validator(model)
"""

from pathlib import Path

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import ops
from ultralytics.utils.metrics import box_iou


# ---------------------------------------------------------------------------
#  Mixin: area-score data collection logic shared by all model flavours
# ---------------------------------------------------------------------------

class _AreaScoreMixin:
    """
    Mixin that adds per-detection (area, pred_score, iou) collection to any
    DetectionValidator subclass.  Must be placed *before* the base validator in
    the MRO so that super() resolves correctly.

    Collected data fields (per record):
        area_original / area_input / area_pct       – GT box area × 3
        pred_area_original / pred_area_input / pred_area_pct – pred box area × 3
        pred_score, iou, status (TP|FP|FN), gt_class, pred_class
    """

    last_instance = None   # class-level ref because model.val() hides the instance

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.area_score_data = []
        self._iou_threshold = 0.5
        _AreaScoreMixin.last_instance = self

    def init_metrics(self, model):
        super().init_metrics(model)
        self.area_score_data = []

    # ------------------------------------------------------------------
    #  Override update_metrics to additionally capture area–score pairs
    # ------------------------------------------------------------------

    def update_metrics(self, preds, batch):
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)

            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            no_gt = len(pbatch["cls"]) == 0

            self._collect_area_score_batch(predn, pbatch)

            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls) if len(cls) > 0 else np.array([]),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )

            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)

            if no_pred:
                if not no_gt:
                    ori_h, ori_w = pbatch["ori_shape"]
                    im_area = float(ori_h * ori_w)
                    gt_boxes_scaled = self._scale_gt_boxes(pbatch)
                    for gt_idx in range(len(pbatch["cls"])):
                        area_original = float(
                            (gt_boxes_scaled[gt_idx][2] - gt_boxes_scaled[gt_idx][0])
                            * (gt_boxes_scaled[gt_idx][3] - gt_boxes_scaled[gt_idx][1])
                        )
                        area_input = float(
                            (pbatch["bboxes"][gt_idx][2] - pbatch["bboxes"][gt_idx][0])
                            * (pbatch["bboxes"][gt_idx][3] - pbatch["bboxes"][gt_idx][1])
                        )
                        area_pct = float(area_original / im_area * 100) if im_area > 0 else 0.0
                        self.area_score_data.append({
                            "area_original": area_original,
                            "area_input": area_input,
                            "area_pct": area_pct,
                            "pred_area_original": 0.0,
                            "pred_area_input": 0.0,
                            "pred_area_pct": 0.0,
                            "pred_score": 0.0,
                            "iou": 0.0,
                            "status": "FN",
                            "gt_class": int(pbatch["cls"][gt_idx]),
                            "pred_class": -1,
                        })
                continue

            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    # ------------------------------------------------------------------
    #  Box scaling helpers
    # ------------------------------------------------------------------

    def _scale_gt_boxes(self, pbatch):
        """Scale GT boxes from imgsz space to original image space."""
        return ops.scale_boxes(
            pbatch["imgsz"],
            pbatch["bboxes"].clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )

    def _scale_pred_boxes(self, pred_bboxes, pbatch):
        """Scale prediction boxes from imgsz space to original image space."""
        return ops.scale_boxes(
            pbatch["imgsz"],
            pred_bboxes.clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )

    # ------------------------------------------------------------------
    #  Core area–score collection (matching logic)
    # ------------------------------------------------------------------

    def _collect_area_score_batch(self, predn, pbatch):
        """
        Collect (area, pred_score, iou) records for one image using greedy
        matching at IoU ≥ 0.5, exactly mirroring match_predictions().
        """
        gt_cls = pbatch["cls"]
        gt_bboxes = pbatch["bboxes"]          # imgsz-space xyxy
        pred_cls = predn["cls"]
        pred_conf = predn["conf"]
        pred_bboxes = predn["bboxes"]

        n_gt = len(gt_cls)
        n_pred = len(pred_cls)

        # --- pred box areas (all three variants) ---
        ori_h, ori_w = pbatch["ori_shape"]
        im_area = float(ori_h * ori_w)

        pred_areas_input = (pred_bboxes[:, 2] - pred_bboxes[:, 0]) * (pred_bboxes[:, 3] - pred_bboxes[:, 1])
        if n_pred > 0:
            pred_boxes_orig = self._scale_pred_boxes(pred_bboxes, pbatch)
            pred_areas_original = (pred_boxes_orig[:, 2] - pred_boxes_orig[:, 0]) * (pred_boxes_orig[:, 3] - pred_boxes_orig[:, 1])
            pred_areas_pct = pred_areas_original / im_area * 100 if im_area > 0 else torch.zeros_like(pred_areas_original)
        else:
            pred_areas_original = pred_areas_input
            pred_areas_pct = pred_areas_input

        # --- No GT: everything is FP ---
        if n_gt == 0:
            for p_idx in range(n_pred):
                self.area_score_data.append({
                    "area_original": 0.0, "area_input": 0.0, "area_pct": 0.0,
                    "pred_area_original": float(pred_areas_original[p_idx]),
                    "pred_area_input": float(pred_areas_input[p_idx]),
                    "pred_area_pct": float(pred_areas_pct[p_idx]),
                    "pred_score": float(pred_conf[p_idx]),
                    "iou": 0.0, "status": "FP",
                    "gt_class": -1, "pred_class": int(pred_cls[p_idx]),
                })
            return

        # --- No predictions: handle in caller (FN) ---
        if n_pred == 0:
            return

        # --- IoU matrix + class-filtered matching ---
        iou_matrix = box_iou(gt_bboxes, pred_bboxes)
        correct_class = gt_cls[:, None] == pred_cls
        iou_class = iou_matrix * correct_class
        iou_np = iou_class.cpu().numpy()

        threshold = self._iou_threshold
        matches = np.nonzero(iou_np >= threshold)
        matches = np.array(matches).T

        matched_gt = set()
        matched_pred = set()

        if matches.shape[0] > 0:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            matched_gt = set(matches[:, 0].astype(int))
            matched_pred = set(matches[:, 1].astype(int))

        # --- GT box areas (all three variants) ---
        gt_boxes_scaled = self._scale_gt_boxes(pbatch)
        gt_areas_input = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        gt_areas_original = (gt_boxes_scaled[:, 2] - gt_boxes_scaled[:, 0]) * (gt_boxes_scaled[:, 3] - gt_boxes_scaled[:, 1])
        gt_areas_pct = gt_areas_original / im_area * 100 if im_area > 0 else torch.zeros_like(gt_areas_original)

        # --- True positives ---
        for gt_idx, pred_idx in matches:
            gt_idx, pred_idx = int(gt_idx), int(pred_idx)
            self.area_score_data.append({
                "area_original": float(gt_areas_original[gt_idx]),
                "area_input": float(gt_areas_input[gt_idx]),
                "area_pct": float(gt_areas_pct[gt_idx]),
                "pred_area_original": float(pred_areas_original[pred_idx]),
                "pred_area_input": float(pred_areas_input[pred_idx]),
                "pred_area_pct": float(pred_areas_pct[pred_idx]),
                "pred_score": float(pred_conf[pred_idx]),
                "iou": float(iou_np[gt_idx, pred_idx]),
                "status": "TP",
                "gt_class": int(gt_cls[gt_idx]),
                "pred_class": int(pred_cls[pred_idx]),
            })

        # --- False positives ---
        for p_idx in range(n_pred):
            if p_idx not in matched_pred:
                same_cls_ious = iou_np[:, p_idx]
                best_iou = float(same_cls_ious.max()) if same_cls_ious.size > 0 else 0.0
                self.area_score_data.append({
                    "area_original": 0.0, "area_input": 0.0, "area_pct": 0.0,
                    "pred_area_original": float(pred_areas_original[p_idx]),
                    "pred_area_input": float(pred_areas_input[p_idx]),
                    "pred_area_pct": float(pred_areas_pct[p_idx]),
                    "pred_score": float(pred_conf[p_idx]),
                    "iou": best_iou, "status": "FP",
                    "gt_class": -1, "pred_class": int(pred_cls[p_idx]),
                })

        # --- False negatives ---
        for gt_idx in range(n_gt):
            if gt_idx not in matched_gt:
                self.area_score_data.append({
                    "area_original": float(gt_areas_original[gt_idx]),
                    "area_input": float(gt_areas_input[gt_idx]),
                    "area_pct": float(gt_areas_pct[gt_idx]),
                    "pred_area_original": 0.0, "pred_area_input": 0.0, "pred_area_pct": 0.0,
                    "pred_score": 0.0, "iou": 0.0, "status": "FN",
                    "gt_class": int(gt_cls[gt_idx]), "pred_class": -1,
                })


# ---------------------------------------------------------------------------
#  Concrete validators  (mixin *before* base so overrides win)
# ---------------------------------------------------------------------------

class AreaScoreValidator(_AreaScoreMixin, DetectionValidator):
    """Area-score validator for YOLO detect models."""
    pass


try:
    from ultralytics.models.rtdetr.val import RTDETRValidator

    class RTDETRAreaScoreValidator(_AreaScoreMixin, RTDETRValidator):
        """Area-score validator for RT-DETR models."""
        pass

except ImportError:
    RTDETRValidator = None
    RTDETRAreaScoreValidator = None


# ---------------------------------------------------------------------------
#  Auto-detect helper
# ---------------------------------------------------------------------------

def get_area_score_validator(model):
    """
    Return the appropriate AreaScoreValidator *class* for the given model.

    Usage:
        ValidCls = get_area_score_validator(model)
        model.val(validator=ValidCls, ...)
    """
    try:
        default_cls = model._smart_load("validator")
    except Exception:
        return AreaScoreValidator

    if RTDETRValidator is not None and default_cls is RTDETRValidator:
        return RTDETRAreaScoreValidator
    return AreaScoreValidator
