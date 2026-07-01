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

Two regimes live in this module:

1. :class:`TaskAlignedAssigner_PriorLA` — a TAL-framework assigner (assigns on
   the network's *predicted* boxes, gates candidates to anchor-centres inside
   the GT). Use it for the ``align='tal'`` ABLATION bridges (e.g. NWD inside
   TAL's dynamic ``s^α·m^β``), NOT as a faithful prior-work reproduction: TAL's
   predicted-box + in-GT machinery is not what NWD-RKA / RFLA do.

2. :class:`RankingAssigner_OneStage` (NWD-RKA) and :class:`HieAssigner_OneStage`
   (RFLA) — FAITHFUL one-stage ports. NWD-RKA and RFLA are *static* assigners:
   they assign GT to fixed priors (RPN anchors / receptive-field priors) by a
   box-similarity metric, ignoring the predictions, with metric-only ranking
   (no cls term) and a hard one-hot label. Anchor-free YOLO has no anchor box,
   only one point per cell, so we stand in a square prior box of side
   ``rf_scale * stride`` centred on each cell (a proxy for the cell's receptive
   field, mirroring RFLA's RFGenerator which sizes the prior to the level RF).
   The metric is computed GT-vs-prior exactly as the two-stage assigners compute
   it GT-vs-anchor. HieAssigner_OneStage additionally reproduces RFLA's
   *hierarchical* two-stage selection: top-k1 on the prior, then top-k2 on the
   prior rescaled by ``ratio``, unioned. Hyper-params mirror this repo's mmdet
   configs: NWD-RKA ``topk=2, C=12.7``; RFLA ``topk=[3,1], ratio=0.9``.

The receptive-field-prior STAND-IN (square cell box) is the one unavoidable
one-stage adaptation; everything else (metric-only, static assignment, hard
label, hierarchical two-stage for RFLA) is faithful. ``label_type='soft'`` is an
opt-in extension (GFL-style normalized-metric target) for the soft/hard sweep,
not part of the original hard-label methods.
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


class _StaticPriorAssigner(TaskAlignedAssigner):
    """Base for faithful one-stage ports of static-anchor tiny-object assigners.

    NWD-RKA (``RankingAssigner``) and RFLA (``HieAssigner``) assign GT to a fixed
    geometric prior by a box-similarity metric, ignoring the network output and
    the classification score (metric-only ranking), and emit a hard one-hot
    label. In anchor-free YOLO there is no anchor box, so we build a square prior
    box of side ``rf_scale * stride`` centred on every cell point and compute the
    metric GT-vs-prior. Subclasses implement the positive selection.

    Args:
        topk (int): per-GT top-k.
        num_classes (int): number of classes.
        metric_type (str): one of :data:`PRIOR_METRICS`.
        metric_kwargs (dict): metric extras, e.g. ``{'nwd_c': 12.7}``.
        label_type (str): ``'hard'`` (one-hot, the original methods) or
            ``'soft'`` (GFL-style normalized-metric target; opt-in extension).
        rf_scale (float): prior box side as a multiple of the cell stride
            (the receptive-field stand-in). ``1.0`` = one cell.
    """

    def __init__(self, topk: int = 2, num_classes: int = 80,
                 alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 metric_type: str = "NWD", metric_kwargs: dict | None = None,
                 label_type: str = "hard", rf_scale: float = 1.0):
        super().__init__(topk=topk, num_classes=num_classes,
                         alpha=alpha, beta=beta, eps=eps)
        assert metric_type in PRIOR_METRICS, \
            f"metric_type must be one of {PRIOR_METRICS}, got {metric_type!r}"
        assert label_type in ("soft", "hard"), label_type
        self.metric_type = metric_type
        self.metric_kwargs = metric_kwargs or {}
        self.label_type = label_type
        self.rf_scale = rf_scale

    def iou_calculation(self, gt_bboxes, pr_bboxes):
        """Affinity between GT and the static prior boxes (GT is box1)."""
        return bbox_iou_ext(
            gt_bboxes, pr_bboxes, xywh=False,
            iou_type=self.metric_type, iou_kargs=self.metric_kwargs,
        ).squeeze(-1).clamp_(0)

    def build_prior_boxes(self, anc_points, stride):
        """Square prior box per cell: side ``rf_scale * stride`` at the centre.

        Args:
            anc_points (Tensor): (na, 2) image-space cell centres.
            stride (Tensor): (bs, n_max, na) per-anchor stride, passed NEGATIVE
                by the loss (see mla_scale convention).

        Returns:
            Tensor: (bs, na, 4) xyxy prior boxes.
        """
        s = stride.abs().amax(dim=1)              # (bs, na) — stride per anchor
        half = (self.rf_scale * s) / 2.0          # (bs, na)
        cx = anc_points[:, 0].unsqueeze(0)        # (1, na)
        cy = anc_points[:, 1].unsqueeze(0)        # (1, na)
        return torch.stack((cx - half, cy - half, cx + half, cy + half), dim=-1)

    def compute_overlaps(self, gt_bboxes, prior_boxes, mask_gt):
        """Metric matrix (bs, n_max, na) between GTs and static priors."""
        na = prior_boxes.shape[1]
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na],
                               dtype=prior_boxes.dtype, device=prior_boxes.device)
        mask = mask_gt.expand(-1, -1, na).bool()
        gt_exp = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask]
        pr_exp = prior_boxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask]
        overlaps[mask] = self.iou_calculation(gt_exp, pr_exp)
        return overlaps

    def topk_mask(self, metrics, k, mask_gt):
        """Per-GT top-k selection mask (bs, n_max, na), like TAL but with arbitrary k."""
        k = min(int(k), metrics.shape[-1])
        topk_metrics, topk_idxs = torch.topk(metrics, k, dim=-1, largest=True)
        valid = mask_gt.expand(-1, -1, k).bool()
        topk_idxs = topk_idxs.masked_fill(~valid, 0)
        count = torch.zeros(metrics.shape, dtype=torch.int8, device=metrics.device)
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8)
        for j in range(k):
            count.scatter_add_(-1, topk_idxs[:, :, j:j + 1], ones)
        count.masked_fill_(count > 1, 0)
        return count.to(metrics.dtype)

    def finalize(self, mask_pos, overlaps, gt_labels, gt_bboxes):
        """Conflict resolution + targets + hard/soft label. ``overlaps`` is the
        prior metric (used for both highest-overlap tie-breaking and the soft
        normalized target)."""
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        if self.label_type == "soft":
            align_metric = overlaps * mask_pos
            pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
            pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
            norm_align_metric = (align_metric * pos_overlaps /
                                 (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
            target_scores = target_scores * norm_align_metric
        # 'hard': one-hot target from get_targets (the original NWD-RKA/RFLA label).

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def _empty_assignment(self, pd_scores, pd_bboxes):
        """Return the no-GT default (matches TaskAlignedAssigner)."""
        return (
            torch.full_like(pd_scores[..., 0], self.num_classes),
            torch.zeros_like(pd_bboxes),
            torch.zeros_like(pd_scores),
            torch.zeros_like(pd_scores[..., 0]),
            torch.zeros_like(pd_scores[..., 0]),
        )


class RankingAssigner_OneStage(_StaticPriorAssigner):
    """NWD-RKA's RKA ported to one-stage: single-stage per-GT top-k on static
    priors, ranked by the box-similarity metric only (no cls term, no in-GT
    gate). Faithful defaults: ``metric_type='NWD', nwd_c=12.7, topk=2``.

    Also used as the shared single-stage protocol for the DotD / SimD metric
    rows (same selection, only the metric differs).
    """

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                mask_gt, stride=None, **kwargs):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            return self._empty_assignment(pd_scores, pd_bboxes)
        assert stride is not None, \
            "RankingAssigner_OneStage needs `stride`; add it to ASSIGN_USE_STRIDE"

        prior_boxes = self.build_prior_boxes(anc_points, stride)
        overlaps = self.compute_overlaps(gt_bboxes, prior_boxes, mask_gt)
        mask_topk = self.topk_mask(overlaps, self.topk, mask_gt)
        mask_pos = mask_topk * mask_gt          # no in-GT gate (faithful to RKA)
        return self.finalize(mask_pos, overlaps, gt_labels, gt_bboxes)


class HieAssigner_OneStage(_StaticPriorAssigner):
    """RFLA's HieAssigner (Hierarchical Label Assignment) ported to one-stage.

    Two-stage selection on static priors: stage 1 takes per-GT top-``topk1`` on
    the prior metric; stage 2 takes per-GT top-``topk2`` on the prior rescaled by
    ``ratio`` (RFLA's ``anchor_rescale``); the two positive sets are unioned
    (stage-1 positives preserved, tie-broken by the original-scale metric).
    Faithful defaults mirror this repo's mmdet config: ``topk=[3,1], ratio=0.9``,
    ``metric_type='KLD'`` (use ``'WD'`` for RFLA-WD).

    Note: RFLA's per-stage 0.8 negative floor sets the background/ignore split,
    which in anchor-free YOLO is a no-op (non-positives are background anyway and
    there is no ignore region), so it is intentionally not applied — top-k
    positives are identical with or without it.
    """

    def __init__(self, topk: int = 3, num_classes: int = 80,
                 alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 metric_type: str = "KLD", metric_kwargs: dict | None = None,
                 label_type: str = "hard", rf_scale: float = 1.0,
                 topk1: int = 3, topk2: int = 1, ratio: float = 0.9):
        super().__init__(topk=topk1, num_classes=num_classes,
                         alpha=alpha, beta=beta, eps=eps, metric_type=metric_type,
                         metric_kwargs=metric_kwargs, label_type=label_type,
                         rf_scale=rf_scale)
        self.topk1 = topk1
        self.topk2 = topk2
        self.ratio = ratio

    @staticmethod
    def rescale(boxes, ratio):
        """Rescale xyxy boxes by ``ratio`` about their centres (RFLA stage 2)."""
        cx = (boxes[..., 0] + boxes[..., 2]) / 2
        cy = (boxes[..., 1] + boxes[..., 3]) / 2
        w = (boxes[..., 2] - boxes[..., 0]) * ratio
        h = (boxes[..., 3] - boxes[..., 1]) * ratio
        return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes,
                mask_gt, stride=None, **kwargs):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        if self.n_max_boxes == 0:
            return self._empty_assignment(pd_scores, pd_bboxes)
        assert stride is not None, \
            "HieAssigner_OneStage needs `stride`; add it to ASSIGN_USE_STRIDE"

        prior_boxes = self.build_prior_boxes(anc_points, stride)
        prior_boxes2 = self.rescale(prior_boxes, self.ratio)
        overlaps1 = self.compute_overlaps(gt_bboxes, prior_boxes, mask_gt)
        overlaps2 = self.compute_overlaps(gt_bboxes, prior_boxes2, mask_gt)

        mask1 = self.topk_mask(overlaps1, self.topk1, mask_gt)
        mask2 = self.topk_mask(overlaps2, self.topk2, mask_gt)
        mask_pos = ((mask1 + mask2) > 0).to(overlaps1.dtype) * mask_gt
        # Conflict resolution + soft target use the original-scale metric.
        return self.finalize(mask_pos, overlaps1, gt_labels, gt_bboxes)
