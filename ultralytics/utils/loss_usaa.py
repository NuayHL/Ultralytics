"""
Detection loss for USAA — Uncertainty-Scale-Adaptive Assignment.

Pairs with TaskAlignedAssigner_dyab_dmetric_dscale (mla_sub.py).

Uncertainty pipeline (two explicit, decoupled steps):
    Step 1  _compute_dfl_uncertainty()
            DFL distribution → raw variance u_raw  (bs, na)
            This is a geometric/statistical quantity from the regression head;
            no normalization is applied here.

    Step 2  DyabLinearFusion._norm_uncertainty()   [inside the assigner]
            raw variance → u_norm ∈ [0, 1]  via tanh(u / tau)
            Controlled by dyab_kwargs.uncertainty_tau in the YAML config.

Area normalization lives entirely in DyabLinearFusion._norm_area() and is
independent of the uncertainty pipeline; modify scale_min / scale_max there.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

from ultralytics.utils.tal import make_anchors

from .loss import v8DetectionLoss


class DetectionLoss_USAA(v8DetectionLoss):
    """
    Detection loss that feeds per-anchor DFL uncertainty into the USAA assigner.

    Inherits all standard detection loss infrastructure from v8DetectionLoss
    (preprocess, bbox_decode, BboxLoss, BCE).  The only additions are:

        _compute_dfl_uncertainty()  — explicit DFL-variance computation
        __call__()                  — passes uncertainty as third assigner arg

    YAML config keys (in addition to standard ones):
        assigner_type: TaskAlignedAssigner_dyab_dmetric_dscale
        dyab_type:     DyabLinearFusion
        dyab_kwargs:
          uncertainty_tau: 2.0    # ← controls uncertainty normalisation
          lambda_fusion:   0.6
          scale_min:       16     # ← controls area normalisation lower bound
          scale_max:       64     # ← controls area normalisation upper bound
          alpha_easy: 0.5
          alpha_hard: 1.2
          beta_easy:  6.0
          beta_hard:  3.0
        overlap_iou_type:  CIoU
        align_iou_type:    Hausdorff
        score_iou_type:    CIoU
        dscale_func:       static
        scale_ratio:       1.0
    """

    def __init__(self, model, cfg: dict, **kwargs):
        super().__init__(model=model, cfg=cfg)

    # ─────────────────────────────────────────────────────────────────────────
    # Uncertainty computation
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_dfl_uncertainty(self, pred_distri: torch.Tensor) -> torch.Tensor:
        """
        Compute per-anchor prediction uncertainty from the DFL distribution.

        Method: for each anchor compute the variance of the DFL distribution over
        the reg_max bins, then average across the 4 coordinate dimensions (l,t,r,b).

            Var(X) = E[X²] − (E[X])²

        A high variance means the model's box distribution is spread out, indicating
        low confidence (hard or ambiguous sample).

        Args:
            pred_distri (Tensor): raw DFL logits, shape (bs, num_anchors, reg_max * 4).
                                  Provided BEFORE permute so shape is (bs, na, 4*reg_max).

        Returns:
            u_raw (Tensor): (bs, num_anchors), raw DFL variance — NOT normalised.
                            Normalisation happens inside DyabLinearFusion._norm_uncertainty().
        """
        bs = pred_distri.shape[0]
        # (bs, na, 4, reg_max) probability distributions
        dfl_probs = pred_distri.view(bs, -1, 4, self.reg_max).detach().softmax(-1)
        bins      = self.proj.view(1, 1, 1, -1)         # bin centres

        e_x   = (dfl_probs * bins).sum(-1)              # E[X],   (bs, na, 4)
        e_x2  = (dfl_probs * bins.pow(2)).sum(-1)        # E[X²],  (bs, na, 4)
        var   = e_x2 - e_x.pow(2)                        # Var(X), (bs, na, 4)

        return var.mean(-1)                               # (bs, na)  — average over 4 coords

    # ─────────────────────────────────────────────────────────────────────────
    # Forward pass
    # ─────────────────────────────────────────────────────────────────────────

    def __call__(
        self,
        preds: Any,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute box + cls + dfl loss with uncertainty-driven label assignment."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl

        if isinstance(preds, tuple):
            if isinstance(preds[1], tuple):
                feats = preds[1][1]
                pd_coefs = preds[1][0]
            else:
                feats = preds[1]
                pd_coefs = preds[0]
        else:
            feats = preds
            pd_coefs = None
        bs = feats[0].shape[0]

        pred_distri, pred_scores = (
            torch.cat([xi.view(bs, self.no, -1) for xi in feats], dim=2)
            .split((self.reg_max * 4, self.nc), dim=1)
        )

        # ── Step 1: compute raw DFL uncertainty (no normalisation here) ───────
        u_raw = self._compute_dfl_uncertainty(pred_distri)   # (bs, na)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()   # (bs, na, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()   # (bs, na, 4*reg_max)

        dtype      = pred_scores.dtype
        imgsz      = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), dim=1
        )
        targets    = self.preprocess(targets.to(self.device), bs, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), dim=2)   # (bs, n_max, 1), (bs, n_max, 4)
        mask_gt    = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (bs, na, 4)

        # ── Stride tensor for dscale (shape: bs × n_max × na) ────────────────
        if self.assigner_use_stride_input:
            n_max   = gt_bboxes.shape[1]
            _stride = (
                stride_tensor.clone().squeeze()
                .unsqueeze(0).unsqueeze(0)
                .expand(bs, n_max, -1)
                .to(dtype=gt_bboxes.dtype, device=gt_bboxes.device)
            )
        else:
            _stride = None

        # ── Label assignment (uncertainty is the 3rd positional argument) ─────
        # Step 2 of the uncertainty pipeline (normalisation) happens inside the
        # assigner's DyabLinearFusion._norm_uncertainty() — see mla_sub.py.
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid() if self.assigner_sigmoid_input else pred_scores.detach(),
            (pred_bboxes.detach() * stride_tensor).to(gt_bboxes.dtype),
            u_raw,                                       # ← raw DFL variance
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
            stride=-_stride if _stride is not None else None,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # ── Classification loss (BCE) ─────────────────────────────────────────
        loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        # ── Box + DFL loss ────────────────────────────────────────────────────
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask,
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss * bs, loss.detach()
