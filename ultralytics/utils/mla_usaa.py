"""
USAA — Uncertainty-Scale-Adaptive Assignment components.

Contains the three decoupled building blocks of TaskAlignedAssigner_dyab_dmetric_dscale:

    DScaleFunctions            — pure-static stride/size → scale-ratio mapping library (dscale)
    DyabBase / DyabLinearFusion — pluggable dynamic α/β computation                   (dyab)
    DYAB_REGISTRY              — YAML string → DyabBase subclass registry
    TaskAlignedAssigner_dyab_dmetric_dscale — three-way decoupled TAL assigner
"""

from __future__ import annotations

import math

import torch

from .metrics import bbox_iou_ext
from .mla_scale import TaskAlignedAssigner_Scale


# =============================================================================
# DScaleFunctions — stride/size → scale-ratio mapping library (dscale)
# =============================================================================

class DScaleFunctions:
    """
    Pure-static library of stride/object-size → scale-ratio mapping functions.

    All functions share the same signature:
        f(r, scale_ratio) -> Tensor  (same shape as r)
    where r = |stride| / object_size  (≥ 0).

    Use DScaleFunctions.compute(func_name, r, scale_ratio) for name-based dispatch.

    Available functions
    -------------------
    static             : constant scale_ratio (scalar)
    func_1             : piecewise-linear bell, peak at r∈[1,2)
    func_2             : linear ramp clamped at r_max
    func_smooth_1      : sigmoid rise × soft fall
    func_gaussian_dip  : Gaussian dip at r_ideal, plateau at r_max
    func_exp_saturate  : exponential saturation toward r_max
    func_inverse_smooth: monotone decreasing sigmoid (small obj → r_max)
    func_scale_adaptive: tanh ramp, scale_ratio=(base, boost, gamma)
    """

    @classmethod
    def compute(cls, func_name: str, r, scale_ratio):
        """Dispatch r → scale-ratio tensor by function name."""
        if func_name == 'static':
            return scale_ratio
        elif func_name == 'func_1':
            return cls.func_1(r, scale_ratio)
        elif func_name == 'func_2':
            return cls.func_2(r, scale_ratio)
        elif func_name == 'func_smooth_1':
            return cls.func_smooth_1(r, scale_ratio)
        elif func_name == 'func_gaussian_dip':
            return cls.func_gaussian_dip(r, scale_ratio[0], scale_ratio[1])
        elif func_name == 'func_exp_saturate':
            return cls.func_exp_saturate(r, scale_ratio)
        elif func_name == 'func_inverse_smooth':
            return cls.func_inverse_smooth(r, scale_ratio[0], scale_ratio[1], scale_ratio[2])
        elif func_name == 'func_scale_adaptive':
            base  = scale_ratio[0] if isinstance(scale_ratio, (list, tuple)) else 1.0
            boost = scale_ratio[1] if isinstance(scale_ratio, (list, tuple)) else 0.5
            gamma = scale_ratio[2] if isinstance(scale_ratio, (list, tuple)) else 1.5
            return cls.func_scale_adaptive(r, base, boost, gamma)
        else:
            raise NotImplementedError(f'DScaleFunctions: unknown func_name={func_name!r}')

    @staticmethod
    def func_1(r, r_max):
        """Piecewise-linear bell: ramp up on [0.25,1), flat on [1,2), ramp down on [2,2.5)."""
        s = torch.zeros_like(r)
        s[(r >= 0.25) & (r < 1)]   = r_max * (r[(r >= 0.25) & (r < 1)]   - 0.25) / 0.75
        s[(r >= 1)    & (r < 2)]   = r_max
        s[(r >= 2)    & (r < 2.5)] = r_max * (2.5 - r[(r >= 2) & (r < 2.5)])
        return s

    @staticmethod
    def func_2(r, r_max):
        """Linear ramp clamped at r_max."""
        s = torch.zeros_like(r)
        s[r < 1]  = r_max * r[r < 1]
        s[r >= 1] = r_max
        return s

    @staticmethod
    def func_smooth_1(r, r_max, a=5.0, b=1.0):
        """Sigmoid rise × soft fall."""
        rise = 1 / (1 + torch.exp(-a * (r - 0.5)))
        fall = 1 - 0.25 / (1 + torch.exp(-b * (r - 2.0)))
        return r_max * rise * fall

    @staticmethod
    def func_gaussian_dip(r, r_max, r_ideal, sigma=0.1):
        """Gaussian dip at r_ideal; plateau near r_max away from the ideal ratio."""
        return r_max * (1.0 - 0.5 * torch.exp(-torch.pow(r - r_ideal, 2) / (2 * sigma ** 2 + 1e-9)))

    @staticmethod
    def func_exp_saturate(r, r_max, a=1.0):
        """Exponential saturation: s = r_max * (1 - exp(-a*r))."""
        return r_max * (1.0 - torch.exp(-a * r))

    @staticmethod
    def func_inverse_smooth(r, r_min, r_max, k):
        """Monotone-decreasing sigmoid: large objects → r_min, small objects → r_max."""
        sig = 1 / (1 + torch.exp(-k * (r - 1)))
        return r_min + (r_max - r_min) * (1 - sig)

    @staticmethod
    def func_scale_adaptive(r, scale_base, scale_boost, gamma=2.0):
        """Smooth tanh ramp: s = scale_base + scale_boost * tanh(gamma * r)."""
        return scale_base + scale_boost * torch.tanh(gamma * r)


# =============================================================================
# DyabBase / concrete implementations — dynamic α/β computation (dyab)
# =============================================================================

class DyabBase:
    """
    Abstract base for dynamic (α, β) computation.

    Subclasses receive per-anchor uncertainty and GT geometry and return
    per-anchor α and β tensors used in the TAL align metric s^α × IoU^β.

    Implement:
        compute(uncertainty, gt_bboxes, n_max_boxes, na, device)
            -> (alpha: Tensor[bs, n_max, na], beta: Tensor[bs, n_max, na])
    """
    def compute(self, uncertainty, gt_bboxes, n_max_boxes, na, device):
        raise NotImplementedError


class DyabLinearFusion(DyabBase):
    """
    Linear fusion of per-anchor uncertainty and GT area → dynamic α/β.

    Difficulty score:
        k = lambda_fusion * u_norm + (1 - lambda_fusion) * (1 - area_score)

    k = 1 → hardest (small + uncertain) → alpha_hard, beta_hard
    k = 0 → easiest (large + confident) → alpha_easy, beta_easy

    Two independent normalisations (each overridable in a subclass):
        _norm_uncertainty : raw DFL variance → [0, 1]  via tanh(u / tau)
        _norm_area        : GT side length   → [0, 1]  via log-scale linear clip

    YAML example
    ------------
    dyab_type: DyabLinearFusion
    dyab_kwargs:
      lambda_fusion:    0.6
      uncertainty_tau:  2.0   # tanh temperature for uncertainty norm
      scale_min:        16    # GT side length lower bound (pixels)
      scale_max:        64    # GT side length upper bound (pixels)
      alpha_easy:       0.5
      alpha_hard:       1.2
      beta_easy:        6.0
      beta_hard:        3.0
    """

    def __init__(self, lambda_fusion=0.6, uncertainty_tau=2.0,
                 scale_min=16, scale_max=64,
                 alpha_easy=0.5, alpha_hard=1.2, beta_easy=6.0, beta_hard=3.0):
        self.lambda_fusion    = lambda_fusion
        self.uncertainty_tau  = uncertainty_tau   # controls tanh saturation speed
        self.log_scale_min    = math.log(scale_min)
        self.log_scale_max    = math.log(scale_max)
        self.alpha_easy = alpha_easy
        self.alpha_hard = alpha_hard
        self.beta_easy  = beta_easy
        self.beta_hard  = beta_hard

    def _norm_uncertainty(self, uncertainty, n_max_boxes, na):
        """
        Map raw per-anchor uncertainty to [0, 1] via tanh.

        Args:
            uncertainty (Tensor): (bs, num_anchors) — raw DFL variance from the loss.
        Returns:
            u_norm (Tensor): (bs, n_max_boxes, na) — higher = more uncertain.
        """
        return torch.tanh(uncertainty / self.uncertainty_tau).unsqueeze(1).expand(-1, n_max_boxes, -1)

    def _norm_area(self, gt_bboxes, na):
        """
        Map GT object side length to [0, 1] on a log scale.

        scale_min → 0.0  (small / hard),  scale_max → 1.0  (large / easy).

        Args:
            gt_bboxes (Tensor): (bs, n_max_boxes, 4) xyxy.
        Returns:
            area_score (Tensor): (bs, n_max_boxes, na) — higher = larger object.
        """
        gt_areas   = (gt_bboxes[..., 2] - gt_bboxes[..., 0]) * (gt_bboxes[..., 3] - gt_bboxes[..., 1])
        gt_scale   = torch.sqrt(torch.clamp(gt_areas, min=1.0))
        area_score = ((gt_scale.log() - self.log_scale_min) /
                      (self.log_scale_max - self.log_scale_min)).clamp(0.0, 1.0)
        return area_score.unsqueeze(-1).expand(-1, -1, na)

    def compute(self, uncertainty, gt_bboxes, n_max_boxes, na, device):
        u_norm     = self._norm_uncertainty(uncertainty, n_max_boxes, na)  # (bs, n_max, na) ∈ [0,1]
        area_score = self._norm_area(gt_bboxes, na)                        # (bs, n_max, na) ∈ [0,1]

        # k = 1 → hardest (small & uncertain), k = 0 → easiest (large & confident)
        k             = self.lambda_fusion * u_norm + (1 - self.lambda_fusion) * (1 - area_score)
        dynamic_alpha = self.alpha_easy + (self.alpha_hard - self.alpha_easy) * k
        dynamic_beta  = self.beta_easy  + (self.beta_hard  - self.beta_easy)  * k
        return dynamic_alpha, dynamic_beta

class DyabInverseVariance(DyabBase):
    """
    Inverse-variance fusion for dynamic α/β.

    Core idea: log(t) = α·log(s) + β·log(m) is a weighted sum in log-space.
    Optimal weighting under heteroscedastic noise → inverse-variance weighting.

    Localization variance estimate:
        σ²_loc(i,j) ≈ q_j / r_i²
    where q_j = DFL variance, r_i = sqrt(w_i * h_i).

    Reliability ratio:
        ρ_ij = 1 / (1 + q_j / (σ₀² · r_i²))

    Dynamic exponents (inheriting standard TAL baseline at ρ=1):
        α_ij = α₀ · (2 - ρ_ij)
        β_ij = β₀ · ρ_ij

    Hyperparameters: α₀ (default 1.0), β₀ (default 6.0), σ₀ (default 1.0).

    YAML example
    ------------
    dyab_type: DyabInverseVariance
    dyab_kwargs:
      alpha_base: 1.0
      beta_base:  6.0
      sigma_0:    1.0
    """

    def __init__(self, alpha_base=1.0, beta_base=6.0, sigma_0=1.0):
        self.alpha_base = alpha_base
        self.beta_base  = beta_base
        self.sigma_0_sq = sigma_0 ** 2

    def compute(self, uncertainty, gt_bboxes, n_max_boxes, na, device):
        """
        Args:
            uncertainty: (bs, num_anchors) — raw DFL variance per anchor.
            gt_bboxes:   (bs, n_max_boxes, 4) — xyxy format.
            n_max_boxes: int
            na:          int — number of anchors
            device:      torch.device

        Returns:
            alpha: (bs, n_max_boxes, na)
            beta:  (bs, n_max_boxes, na)
        """
        # ── Object scale: r_i = sqrt(w * h) ─────────────────────────────
        gt_w = (gt_bboxes[..., 2] - gt_bboxes[..., 0]).clamp(min=1.0)
        gt_h = (gt_bboxes[..., 3] - gt_bboxes[..., 1]).clamp(min=1.0)
        r_sq = gt_w * gt_h                                  # (bs, n_max_boxes)

        # ── Localization variance: v_ij = q_j / r_i² ────────────────────
        # uncertainty: (bs, na) → (bs, 1, na)
        q = uncertainty.unsqueeze(1)
        # r_sq: (bs, n_max_boxes) → (bs, n_max_boxes, 1)
        r_sq = r_sq.unsqueeze(-1)

        v = q / (r_sq + 1e-9)                               # (bs, n_max_boxes, na)

        # ── Reliability ratio: ρ = σ₀² / (σ₀² + v) ─────────────────────
        rho = self.sigma_0_sq / (self.sigma_0_sq + v)        # ∈ (0, 1]

        # ── Dynamic exponents ────────────────────────────────────────────
        alpha = self.alpha_base * (2.0 - rho)
        beta  = self.beta_base  * rho

        return alpha, beta

# Registry: map YAML string → DyabBase subclass
DYAB_REGISTRY = {
    'DyabLinearFusion': DyabLinearFusion,
    'DyabInverseVariance':   DyabInverseVariance,
}


# =============================================================================
# TaskAlignedAssigner_dyab_dmetric_dscale — three-way decoupled TAL assigner
# =============================================================================

class TaskAlignedAssigner_dyab_dmetric_dscale(TaskAlignedAssigner_Scale):
    """
    Three-way decoupled TAL assigner for small-object detection.

    dyab    — Pluggable dynamic α/β computation. Configured via `dyab_type` (string
              key in DYAB_REGISTRY) and `dyab_kwargs` (passed to its __init__).
              Swap the algorithm entirely by changing two YAML keys.

    dmetric — SimD: separate IoU metric for each assignment role.
                * overlap_iou  : GT ownership (select_highest_overlaps)
                * align_iou    : Top-k candidate ranking
                * score_iou    : Soft target-score normalization

    dscale  — Adaptive candidate region via DScaleFunctions. The GT-containment
              threshold scales with stride/object-size so small objects admit more
              candidate anchors. Set `dscale_func='static'` for fixed-threshold.

    Minimal YAML example
    --------------------
    assigner:
      type: TaskAlignedAssigner_dyab_dmetric_dscale
      topk: 13
      dyab_type: DyabLinearFusion
      dyab_kwargs: {lambda_fusion: 0.6, scale_min: 16, scale_max: 64,
                    alpha_easy: 0.5, alpha_hard: 1.2, beta_easy: 6.0, beta_hard: 3.0}
      overlap_iou_type: CIoU
      align_iou_type:   Hausdorff
      score_iou_type:   CIoU
      dscale_func: func_1
      scale_ratio: 1.0
    """

    def __init__(self, topk: int = 13, num_classes: int = 80,
                 alpha=None, beta=None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk        = topk
        self.num_classes = num_classes
        self.alpha       = alpha if alpha else 1.0
        self.beta        = beta  if beta  else 4.0
        self.eps         = eps

        # ── dyab ─────────────────────────────────────────────────────────────
        dyab_type   = kwargs.get('dyab_type',   'DyabLinearFusion')
        dyab_kwargs = kwargs.get('dyab_kwargs', {})
        dyab_cls    = DYAB_REGISTRY[dyab_type] if isinstance(dyab_type, str) else dyab_type
        self.dyab   = dyab_cls(**dyab_kwargs)

        # ── dmetric ──────────────────────────────────────────────────────────
        self.overlap_iou_type   = kwargs.get('overlap_iou_type',   'CIoU')
        self.overlap_iou_kwargs = kwargs.get('overlap_iou_kwargs',  {})
        self.align_iou_type     = kwargs.get('align_iou_type',     'Hausdorff')
        self.align_iou_kwargs   = kwargs.get('align_iou_kwargs',    {})
        self.score_iou_type     = kwargs.get('score_iou_type',      'CIoU')
        self.score_iou_kwargs   = kwargs.get('score_iou_kwargs',    {})

        # ── dscale ───────────────────────────────────────────────────────────
        # func_name: see DScaleFunctions.compute() for available options.
        # scale_ratio: scalar or (base, boost, ...) tuple — interpreted by func.
        self.dscale_func = kwargs.get('dscale_func', 'static')
        self.scale_ratio = kwargs.get('scale_ratio', 1.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Entry points
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, uncertainty,
                anc_points, gt_labels, gt_bboxes, mask_gt,
                stride=None, **kwargs):
        self.bs          = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )
        return self._forward(pd_scores, pd_bboxes, uncertainty,
                             anc_points, gt_labels, gt_bboxes, mask_gt, stride)

    def _forward(self, pd_scores, pd_bboxes, uncertainty,
                 anc_points, gt_labels, gt_bboxes, mask_gt, stride):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes,
            anc_points, mask_gt, stride,
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )
        align_metric      = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)
        pos_overlaps      = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores     = target_scores * norm_align_metric
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline stages
    # ─────────────────────────────────────────────────────────────────────────

    def get_pos_mask(self, pd_scores, pd_bboxes, uncertainty,
                     gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        # dscale: adaptive candidate region
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride)

        # dyab + dmetric
        align_metric_for_align, overlaps, align_metric_for_score = self.get_box_metrics(
            pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes,
            mask_in_gts * mask_gt,
        )
        mask_topk = self.select_topk_candidates(
            align_metric_for_align,
            topk_mask=mask_gt.expand(-1, -1, self.topk).bool(),
        )
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric_for_score, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, uncertainty,
                        gt_labels, gt_bboxes, mask_gt):
        na      = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()
        device  = pd_bboxes.device
        dtype   = pd_bboxes.dtype

        overlaps           = torch.zeros([self.bs, self.n_max_boxes, na], dtype=dtype, device=device)
        overlaps_for_align = torch.zeros_like(overlaps)
        overlaps_for_score = torch.zeros_like(overlaps)
        bbox_scores        = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=device)

        # ── dyab ─────────────────────────────────────────────────────────────
        dynamic_alpha, dynamic_beta = self.dyab.compute(
            uncertainty, gt_bboxes, self.n_max_boxes, na, device
        )

        # Classification scores for each anchor × GT class
        ind    = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]

        # ── dmetric ──────────────────────────────────────────────────────────
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt]           = self.iou_calculation(gt_boxes, pd_boxes, self.overlap_iou_type, self.overlap_iou_kwargs)
        overlaps_for_align[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes, self.align_iou_type,   self.align_iou_kwargs)
        overlaps_for_score[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes, self.score_iou_type,   self.score_iou_kwargs)

        bbox_scores_part       = bbox_scores.pow(dynamic_alpha)
        align_metric_for_align = bbox_scores_part * overlaps_for_align.pow(dynamic_beta)
        align_metric_for_score = bbox_scores_part * overlaps_for_score.pow(dynamic_beta)
        return align_metric_for_align, overlaps, align_metric_for_score

    # ─────────────────────────────────────────────────────────────────────────
    # dscale: adaptive GT-containment threshold
    # ─────────────────────────────────────────────────────────────────────────

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9, stride=None):
        """
        Positive anchor selection with optionally dynamic containment threshold.
        Falls back to parent (fixed margin) when stride is None or dscale_func='static'.
        """
        if stride is None or self.dscale_func == 'static':
            return super().select_candidates_in_gts(xy_centers, gt_bboxes, eps=eps, stride=stride)

        n_anchors      = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape

        x_min, y_min = gt_bboxes[..., 0], gt_bboxes[..., 1]
        x_max, y_max = gt_bboxes[..., 2], gt_bboxes[..., 3]
        w = (x_max - x_min).clamp(min=eps).unsqueeze(-1)  # (bs, n_boxes, 1)
        h = (y_max - y_min).clamp(min=eps).unsqueeze(-1)

        r_w = stride / (-w + self.eps)                                      # (bs, n_boxes, n_anchors)
        r_h = stride / (-h + self.eps)
        thresh_w = stride * DScaleFunctions.compute(self.dscale_func, r_w, self.scale_ratio)
        thresh_h = stride * DScaleFunctions.compute(self.dscale_func, r_h, self.scale_ratio)

        lt      = torch.stack((x_min, y_min), dim=-1)
        rb      = torch.stack((x_max, y_max), dim=-1)
        lt_exp  = lt[..., None, :].expand(bs, n_boxes, n_anchors, 2)
        rb_exp  = rb[..., None, :].expand(bs, n_boxes, n_anchors, 2)
        centers = xy_centers[None, None, :, :].expand(bs, n_boxes, n_anchors, 2)

        ltrb    = torch.cat([centers - lt_exp, rb_exp - centers], dim=-1)  # (bs, n_boxes, n_anchors, 4)
        mask_in = (ltrb[..., 0] > thresh_w) & (ltrb[..., 2] > thresh_w) \
                & (ltrb[..., 1] > thresh_h) & (ltrb[..., 3] > thresh_h)
        return mask_in

    # ─────────────────────────────────────────────────────────────────────────
    # dmetric: IoU dispatch
    # ─────────────────────────────────────────────────────────────────────────

    def iou_calculation(self, gt_bboxes, pd_bboxes, iou_type=None, iou_kwargs=None):
        return bbox_iou_ext(pd_bboxes, gt_bboxes, xywh=False,
                            iou_type=iou_type, iou_kargs=iou_kwargs).squeeze(-1).clamp_(0)
