"""
USAA-style HungarianMatcher variant for RT-DETR.

Contains:
    HungarianMatcher_ScaleAware — per-GT dynamic cost weights based on object
    size.  The per-GT scale factor ρ directly modulates the cost_gain of the
    Hungarian matcher, without introducing YOLO-specific α/β exponent concepts.
"""

from __future__ import annotations

import torch

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
    # Scale-aware cost matrix (only method that needs to differ from parent)
    # ─────────────────────────────────────────────────────────────────────

    def _get_cost_matrix(self, cost_class, cost_bbox, cost_giou, gt_bboxes):
        """
        Combine costs with per-GT dynamic scale-aware weights.

        For small objects the cls cost weight is reduced and the spatial
        cost weight is increased.  The modulation strength mod ∈ [0,1)
        controls how far from baseline the per-GT gains deviate.
        """
        cls_factor, spatial_factor = self._compute_per_gt_scale_factors(gt_bboxes)
        return (
            self.cost_gain["class"] * cls_factor * cost_class
            + self.cost_gain["bbox"] * spatial_factor * cost_bbox
            + self.cost_gain["giou"] * spatial_factor * cost_giou
        )
