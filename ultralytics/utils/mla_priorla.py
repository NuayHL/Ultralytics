"""
mla_priorla — faithful one-stage reproductions of prior-work label assigners.

A label-assignment algorithm is NOT just a box-similarity metric. It is three
*independent* design axes, and the classic tiny-object assigners (NWD-RKA,
RFLA, DotD) differ from YOLO's TaskAlignedAssigner (TAL) on ALL THREE:

    ┌───────────────────┬─────────────────────────┬──────────────────────────┐
    │ axis              │ TAL (YOLO default)      │ NWD-RKA / RFLA (original)│
    ├───────────────────┼─────────────────────────┼──────────────────────────┤
    │ ① match score     │ score^α · metric^β      │ metric ONLY (ignore cls) │
    │   (cls × reg)     │ (multiplicative)        │                          │
    │ ② pos selection   │ top-k by align score    │ top-k by metric (RKA),   │
    │                   │                         │ (+ optional neg floor)   │
    │ ③ supervision     │ SOFT (target = norm.    │ HARD (target = 1/0,      │
    │   label           │ metric, GFL-style)      │ plain CrossEntropy/BCE)  │
    └───────────────────┴─────────────────────────┴──────────────────────────┘

This module exposes all three axes as knobs so each prior method can be
reproduced *as a whole*, not just its metric:

    align_type   : 'tal'    -> score^α · metric^β        (cls × reg, TAL)
                   'metric'  -> metric only               (NWD-RKA / RFLA)
    select_type  : 'topk'    -> top-k per GT by the align score
    label_type   : 'soft'    -> target_scores = normalized metric (TAL/GFL)
                   'hard'     -> target_scores = 1 for the positive class
    metric_type  : 'CIoU' | 'NWD' | 'KLD' | 'WD' | 'DotD' | 'SimD'
    neg_thr      : optional float — positives must also have metric ≥ neg_thr
                   (the "NWD < 0.3 can't be positive" quality floor in RKA).

Faithful compositions (see cfg/mla_priorla/):
    TAL (control)  : align=tal,    label=soft, metric=CIoU
    NWD-RKA        : align=metric,  label=hard, metric=NWD
    RFLA-KLD       : align=metric,  label=hard, metric=KLD
    RFLA-WD        : align=metric,  label=hard, metric=WD
    DotD           : align=metric,  label=hard, metric=DotD

Not portable: RFLA's RFGenerator (receptive-field ANCHORS) and its two-stage
hierarchical anchor-RESCALE are anchor-box specific and have no analogue in
anchor-free YOLO (points, not boxes). So on YOLO, "RFLA" = its KLD/WD metric
with top-k selection; the hierarchical rescale is intentionally NOT emulated
(it would be a no-op without anchor boxes). ``select_type`` therefore only
supports ``'topk'`` for now.
"""

from __future__ import annotations

import torch

from .metrics import bbox_iou_ext
from .tal import TaskAlignedAssigner

PRIOR_METRICS = ("IoU", "CIoU", "NWD", "DotD", "KLD", "WD", "SimD")


class TaskAlignedAssigner_PriorLA(TaskAlignedAssigner):
    """TAL-based assigner with decoupled (combination, selection, label) axes.

    Args:
        topk (int): top-k candidates per GT (RKA's "k").
        num_classes (int): number of classes.
        alpha, beta (float): TAL exponents — only used when ``align_type='tal'``.
        metric_type (str): one of :data:`PRIOR_METRICS`.
        metric_kwargs (dict): extra kwargs for the metric (e.g. ``nwd_c``).
        align_type (str): ``'tal'`` (score^α·metric^β) or ``'metric'`` (metric
            only — ignore the classification term, as NWD-RKA / RFLA do).
        select_type (str): ``'topk'`` (only supported strategy).
        label_type (str): ``'soft'`` (normalized-metric target, TAL/GFL) or
            ``'hard'`` (target score 1 for the positive class).
        neg_thr (float|None): quality floor from the original RKA paper.
            **Caveat**: in TAL the metric is on predicted (post-regression)
            boxes, which are random early in training — a non-None value
            often kills all positives. Prefer None in this setting.
    """

    def __init__(self, topk: int = 10, num_classes: int = 80,
                 alpha: float = 0.5, beta: float = 6.0, eps: float = 1e-9,
                 metric_type: str = "NWD", metric_kwargs: dict | None = None,
                 align_type: str = "metric", select_type: str = "topk",
                 label_type: str = "hard", neg_thr: float | None = None):
        super().__init__(topk=topk, num_classes=num_classes,
                         alpha=alpha, beta=beta, eps=eps)
        assert metric_type in PRIOR_METRICS, \
            f"metric_type must be one of {PRIOR_METRICS}, got {metric_type!r}"
        assert align_type in ("tal", "metric"), align_type
        assert select_type in ("topk",), \
            f"select_type {select_type!r} not supported (RFLA hierarchical " \
            f"rescale is anchor-box specific and does not port to anchor-free)"
        assert label_type in ("soft", "hard"), label_type
        self.metric_type = metric_type
        self.metric_kwargs = metric_kwargs or {}
        self.align_type = align_type
        self.select_type = select_type
        self.label_type = label_type
        self.neg_thr = neg_thr

    # ── axis 0: metric ─────────────────────────────────────────────────────
    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """Localization affinity using the configured prior-work metric.

        ``gt_bboxes`` is box1 so asymmetric metrics (KLD) use the GT as the
        reference distribution (matches the two-stage RFLA convention).
        """
        return bbox_iou_ext(
            gt_bboxes, pd_bboxes, xywh=False,
            iou_type=self.metric_type, iou_kargs=self.metric_kwargs,
        ).squeeze(-1).clamp_(0)

    # ── axis ①: how cls and reg combine into the ranking score ─────────────
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na],
                               dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na],
                                  dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        if self.align_type == "tal":
            align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        else:  # 'metric' — rank by the localization metric ONLY (NWD-RKA/RFLA)
            align_metric = overlaps
        return align_metric, overlaps

    # ── axis ②: positive selection (+ optional quality floor) ──────────────
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(
            align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt
        if self.neg_thr is not None:  # RKA negative threshold (quality floor)
            mask_pos = mask_pos * (overlaps >= self.neg_thr).to(mask_pos.dtype)
        return mask_pos, align_metric, overlaps

    # ── axis ③: soft (normalized metric) vs hard (1/0) supervision ─────────
    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        if self.label_type == "soft":
            # TAL/GFL soft label: target = per-GT normalized metric.
            align_metric = align_metric * mask_pos
            pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
            pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
            norm_align_metric = (align_metric * pos_overlaps /
                                 (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
            target_scores = target_scores * norm_align_metric
        # 'hard': leave target_scores as the one-hot 1 from get_targets.

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx
