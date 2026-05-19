"""
USAA-style HungarianMatcher variant for RT-DETR.

Contains:
    HungarianMatcher_ScaleAware — per-GT dynamic cost weights based on object
    size.  The per-GT scale factor ρ directly modulates the cost_gain of the
    Hungarian matcher, without introducing YOLO-specific α/β exponent concepts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from ultralytics.utils.metrics import bbox_iou

from .ops import HungarianMatcher


class HungarianMatcher_ScaleAware(HungarianMatcher):
    """
    Scale-aware HungarianMatcher with per-GT dynamic cost weights.

    When soft-label calibration compresses cls-channel discrimination on small
    objects, the matching cost should favour spatial over classification cues.
    For each GT a modulation strength ``mod = 1 − ρ`` is computed, then applied
    directly to the cost_gain coefficients:

        ρ_i          = r_i² / (r_i² + r_ref_ab²)           ∈ (0, 1]
        mod_i        = 1 − ρ_i                              ∈ [0, 1)   (1 = hardest)
        cost_gain_cls     = gain_cls     × (1 − mod_i · cls_reduction)
        cost_gain_spatial = gain_spatial × (1 + mod_i · spatial_boost)

    This is the RT-DETR analogue of ``DyabCalibrationAware`` in mla_usaa.py,
    expressed directly on the cost coefficients that the Hungarian matcher uses.
    """

    def __init__(
        self,
        cost_gain: dict[str, float] | None = None,
        use_fl: bool = True,
        with_mask: bool = False,
        num_sample_points: int = 12544,
        alpha: float = 0.25,
        gamma: float = 2.0,
        # ── USAA scale-aware params (direct cost_gain modulation) ──
        r_ref_ab: float = 64.0,
        cls_reduction: float = 0.5,
        spatial_boost: float = 0.5,
    ):
        super().__init__(
            cost_gain=cost_gain,
            use_fl=use_fl,
            with_mask=with_mask,
            num_sample_points=num_sample_points,
            alpha=alpha,
            gamma=gamma,
        )
        self.r_ref_ab = r_ref_ab
        self.cls_reduction = cls_reduction
        self.spatial_boost = spatial_boost

    # ─────────────────────────────────────────────────────────────────────
    # Per-GT cost_gain modulation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_per_gt_scale_factors(self, gt_bboxes: torch.Tensor):
        """
        Compute per-GT dynamic scale factors for cls and spatial cost components.

        ρ → 1 (large object, r ≫ r_ref): factors → 1.0  (baseline gains)
        ρ → 0 (small object, r ≪ r_ref): cls_factor → 1−cls_reduction,
                                          spatial_factor → 1+spatial_boost

        Args:
            gt_bboxes: (num_gts, 4) in xywh format.

        Returns:
            cls_factor:     (1, num_gts) — multiplier on cost_gain["class"].
            spatial_factor: (1, num_gts) — multiplier on cost_gain["bbox"]
                                          and cost_gain["giou"].
        """
        gt_w = gt_bboxes[:, 2]
        gt_h = gt_bboxes[:, 3]
        r_sq = (gt_w * gt_h).clamp(min=1.0)

        # ρ_i = r_i² / (r_i² + r_ref_ab²)   — same as DyabCalibrationAware
        rho_ab = r_sq / (r_sq + self.r_ref_ab ** 2)          # (num_gts,)

        # Modulation strength: 1 for smallest objects, 0 for largest
        mod = 1.0 - rho_ab

        # Direct ρ-based modulation of cost_gain
        # cls:  reduce for small objects (cls channel compressed by calibration)
        # bbox + giou: boost for small objects (spatial cue more reliable)
        cls_factor     = (1.0 - mod * self.cls_reduction).unsqueeze(0)   # (1, num_gts)
        spatial_factor = (1.0 + mod * self.spatial_boost).unsqueeze(0)   # (1, num_gts)

        return cls_factor, spatial_factor

    # ─────────────────────────────────────────────────────────────────────
    # Forward with scale-aware cost reweighting
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        pred_bboxes: torch.Tensor,
        pred_scores: torch.Tensor,
        gt_bboxes: torch.Tensor,
        gt_cls: torch.Tensor,
        gt_groups: list[int],
        masks: torch.Tensor | None = None,
        gt_mask: list[torch.Tensor] | None = None,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        bs, nq, nc = pred_scores.shape

        if sum(gt_groups) == 0:
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in range(bs)]

        # ── Per-GT dynamic cost_gain modulation ───────────────────────────
        cls_factor, spatial_factor = self._compute_per_gt_scale_factors(gt_bboxes)

        # ── Flatten and compute costs (same as parent) ──────────────────
        pred_scores = pred_scores.detach().view(-1, nc)
        pred_scores = F.sigmoid(pred_scores) if self.use_fl else F.softmax(pred_scores, dim=-1)
        pred_bboxes = pred_bboxes.detach().view(-1, 4)

        # Classification cost
        pred_scores = pred_scores[:, gt_cls]
        if self.use_fl:
            neg_cost_class = (1 - self.alpha) * (pred_scores ** self.gamma) * (-(1 - pred_scores + 1e-8).log())
            pos_cost_class = self.alpha * ((1 - pred_scores) ** self.gamma) * (-(pred_scores + 1e-8).log())
            cost_class = pos_cost_class - neg_cost_class
        else:
            cost_class = -pred_scores

        # L1 bbox cost
        cost_bbox = (pred_bboxes.unsqueeze(1) - gt_bboxes.unsqueeze(0)).abs().sum(-1)

        # GIoU cost
        cost_giou = 1.0 - bbox_iou(
            pred_bboxes.unsqueeze(1), gt_bboxes.unsqueeze(0), xywh=True, GIoU=True
        ).squeeze(-1)

        # ── ★ Scale-aware cost combination ──────────────────────────────
        C = (
            self.cost_gain["class"] * cls_factor * cost_class
            + self.cost_gain["bbox"] * spatial_factor * cost_bbox
            + self.cost_gain["giou"] * spatial_factor * cost_giou
        )

        if self.with_mask:
            C += self._cost_mask(bs, gt_groups, masks, gt_mask)

        C[C.isnan() | C.isinf()] = 0.0

        # ── Hungarian solve (same as parent) ────────────────────────────
        C = C.view(bs, nq, -1).cpu()
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(gt_groups, -1))]
        gt_groups_cum = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        return [
            (torch.tensor(i, dtype=torch.long), torch.tensor(j, dtype=torch.long) + gt_groups_cum[k])
            for k, (i, j) in enumerate(indices)
        ]
