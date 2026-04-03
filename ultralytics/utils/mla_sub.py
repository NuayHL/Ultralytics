import torch
import torch.nn as nn
from .metrics import bbox_iou, bbox_iou_ext

from .mla_scale import TaskAlignedAssigner_Scale

class TaskAlignedAssigner_Subnet_Scale(TaskAlignedAssigner_Scale):
    """
    Task-aligned assigner with per-anchor learned alpha/beta from a subnet.

    Uses pd_coef (bs, num_anchors, 2) as learned (alpha, beta) for the align metric
    s^alpha * iou^beta. Forward keeps gradients enabled for target_scores so the
    subnet can be trained end-to-end with the assignment. Falls back to fixed
    alpha/beta when pd_coef is None.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80,
                 alpha: float = 0.5, beta: float = 6.0,
                 eps: float = 1e-9, scale_ratio: float = 1.0, **kwargs):
        super().__init__(topk, num_classes, alpha, beta, eps, scale_ratio=scale_ratio)
        self.warned = False
    
    def forward(self, pd_scores, pd_bboxes, pd_coef, 
                anc_points, gt_labels, gt_bboxes, mask_gt, 
                stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            pd_coef (torch.Tensor): Predicted coefficients with shape (bs, num_total_anchors, 2).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            stride (torch.Tensor): Stride of anchors with shape (bs, num_total_anchors, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        num_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        with torch.no_grad():
            if self.n_max_boxes == 0:
                return (
                    torch.full_like(pd_scores[..., 0], self.num_classes),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]),
                    torch.zeros_like(pd_scores[..., 0]),
                )
        if pd_coef is None:
            if not self.warned:
                print(f"Warning: No learned alpha beta found, using default values: alpha = {self.alpha}, beta = {self.beta}")
                self.warned = True
            pd_coef = torch.ones([self.bs, num_anchors, 2], dtype=torch.float32, device=device).requires_grad_(False)
            pd_coef[:, :, 0] = self.alpha
            pd_coef[:, :, 1] = self.beta
        return self._forward(pd_scores, pd_bboxes, pd_coef, anc_points, gt_labels, gt_bboxes, mask_gt, stride)
    
    def _forward(self, pd_scores, pd_bboxes, pd_coef, anc_points, gt_labels, gt_bboxes, mask_gt, stride):
        with torch.no_grad():
            mask_pos, align_metric, overlaps = self.get_pos_mask(
                pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, anc_points, mask_gt, stride
            )

            target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

            target_labels, target_bboxes, target_scores_hard = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        with torch.enable_grad():
            _pd_scores = pd_scores.detach()
            _overlaps = overlaps.detach()
            
            alpha = pd_coef[..., 0].unsqueeze(-2)
            beta = pd_coef[..., 1].unsqueeze(-2)

            ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long, device=_pd_scores.device)
            ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)
            ind[1] = gt_labels.squeeze(-1)
            
            bbox_scores = _pd_scores[ind[0], :, ind[1]]
            
            align_metric_grad = bbox_scores.pow(alpha) * _overlaps.pow(beta)
            
            align_metric_grad = align_metric_grad * mask_pos

            pos_align_metrics = align_metric_grad.amax(dim=-1, keepdim=True)  # b, max_num_obj
            pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
            norm_align_metric = (align_metric_grad * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
            target_scores = target_scores_hard * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2] # na = num_anchors
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        alpha = pd_coef[..., 0].unsqueeze(-2)
        beta = pd_coef[..., 1].unsqueeze(-2)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # print("overlaps.shape:", overlaps.shape)
        # print("bbox_scores.shape:", bbox_scores.shape)
        # print("alpha.shape:", alpha.shape)
        # print("beta.shape:", beta.shape)

        align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)
        return align_metric, overlaps


class TaskAlignedAssigner_Subnet_NoGrad_Scale(TaskAlignedAssigner_Scale):
    """
    Same as Subnet_Scale but with fully no-grad forward.

    Uses pd_coef when available, else static alpha/beta. No gradient flow through
    assignment; suitable for inference-style or non-differentiable pipelines.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80,
                 alpha: float = 0.5, beta: float = 6.0,
                 eps: float = 1e-9, scale_ratio: float = 1.0, **kwargs):
        super().__init__(topk, num_classes, alpha, beta, eps, scale_ratio=scale_ratio)
        self.warned = False
        self.using_subnet = True
    
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, pd_coef, 
                anc_points, gt_labels, gt_bboxes, mask_gt, 
                stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            pd_coef (torch.Tensor): Predicted coefficients with shape (bs, num_total_anchors, 2).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            stride (torch.Tensor): Stride of anchors with shape (bs, num_total_anchors, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        num_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        with torch.no_grad():
            if self.n_max_boxes == 0:
                return (
                    torch.full_like(pd_scores[..., 0], self.num_classes),
                    torch.zeros_like(pd_bboxes),
                    torch.zeros_like(pd_scores),
                    torch.zeros_like(pd_scores[..., 0]),
                    torch.zeros_like(pd_scores[..., 0]),
                )
        if pd_coef is None:
            if not self.warned:
                print(f"Warning: No learned alpha beta found, using default values: alpha = {self.alpha}, beta = {self.beta}")
                self.warned = True
                self.using_subnet = False
            pd_coef = torch.ones([self.bs, num_anchors, 2], dtype=torch.float32, device=device).requires_grad_(False)
            pd_coef = [self.alpha, self.beta]
        return self._forward(pd_scores, pd_bboxes, pd_coef, anc_points, gt_labels, gt_bboxes, mask_gt, stride)
    
    def _forward(self, pd_scores, pd_bboxes, pd_coef, anc_points, gt_labels, gt_bboxes, mask_gt, stride):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, anc_points, mask_gt, stride
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
    
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, pd_coef, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2] # na = num_anchors
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        if self.using_subnet:
            alpha = pd_coef[..., 0].unsqueeze(-2)
            beta = pd_coef[..., 1].unsqueeze(-2)
        else:
            alpha = self.alpha
            beta = self.beta

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta)
        return align_metric, overlaps

class TaskAlignedAssigner_SubN_NG_Uncertainty_S(TaskAlignedAssigner_Scale):
    """
    Task-aligned assigner with uncertainty-modulated alpha/beta.

    Maps per-anchor uncertainty to alpha (0.5->1.5) and beta (6->2) via tanh.
    Uses align_metric * exp(-uncertainty) for topk ranking: high-uncertainty
    anchors are downweighted when selecting positive candidates. Helps adapt
    assignment to prediction confidence, especially for small objects.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80,
                 alpha: float = 0.5, beta: float = 6.0,
                 eps: float = 1e-9, scale_ratio: float = 1.0, **kwargs):
        super().__init__(topk, num_classes, alpha, beta, eps, scale_ratio=scale_ratio, **kwargs)
    
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, uncertainty, 
                anc_points, gt_labels, gt_bboxes, mask_gt, 
                stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            uncertainty (torch.Tensor): Uncertainty with shape (bs, num_total_anchors).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            stride (torch.Tensor): Stride of anchors with shape (bs, num_total_anchors, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).

        References:
            https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py
        """
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        num_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )
        return self._forward(pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride)
    
    def _forward(self, pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, anc_points, mask_gt, stride
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
    
        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    # def _forward(self, pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride):
    #     mask_pos, align_metric, overlaps = self.get_pos_mask(
    #         pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, anc_points, mask_gt, stride
    #     )

    #     target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

    #     target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)
    

    #     align_metric = align_metric * mask_pos
        
    #     u_raw = uncertainty.unsqueeze(1).expand(-1, self.n_max_boxes, -1) * mask_pos

    #     pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # [b, max_obj, 1]
        
    #     best_anchor_idx = align_metric.argmax(dim=-1, keepdim=True) # [b, max_obj, 1]
    #     u_best = torch.gather(u_raw, dim=-1, index=best_anchor_idx) # [b, max_obj, 1]

    #     relative_uncertainty = (u_raw - u_best).clamp(min=0.0)
    #     decay_factor = torch.exp(-1.0 * relative_uncertainty) 

    #     ratio = align_metric / (pos_align_metrics + self.eps)
        
    #     pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        
    #     norm_align_metric = (ratio * decay_factor * pos_overlaps)
        
    #     norm_align_metric = norm_align_metric.amax(dim=-2).unsqueeze(-1)
        
    #     target_scores = target_scores * norm_align_metric

    #     return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        align_metric, overlaps, align_metric_for_rank = self.get_box_metrics(pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self.select_topk_candidates(align_metric_for_rank, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2] # na = num_anchors
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        u_raw = torch.zeros([self.bs, self.n_max_boxes, na], dtype=uncertainty.dtype, device=uncertainty.device)

        alpha, beta = self.uncertainty_ab(uncertainty)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        u_raw[mask_gt] = uncertainty.unsqueeze(-2).expand(-1, self.n_max_boxes, -1)[mask_gt]

        # print("bbox_scores.shape", bbox_scores.shape)
        # print("overlaps.shape", overlaps.shape)
        # print("u_raw.shape", u_raw.shape)

        align_metric = bbox_scores.pow(alpha) * overlaps.pow(beta) 
        align_metric_for_rank = align_metric * torch.exp(- u_raw)

        return align_metric, overlaps, align_metric_for_rank
    
    def uncertainty_ab(self, uncertainty):
        """
        Map per-anchor uncertainty to dynamic alpha and beta.

        Args:
            uncertainty (torch.Tensor): Shape (bs, num_anchors). Higher -> harder sample.

        Returns:
            u_alpha, u_beta: alpha in [base_alpha, max_alpha], beta in [min_beta, base_beta].
        """
        base_alpha = 0.5
        max_alpha = 1.5
        base_beta = 6.0
        min_beta = 2.0
        temp_tau = 2.0
        u_norm = torch.tanh(uncertainty / temp_tau)
        u_alpha = base_alpha + (max_alpha - base_alpha) * u_norm
        u_beta = base_beta + (min_beta - base_beta) * u_norm
        return u_alpha.unsqueeze(-2), u_beta.unsqueeze(-2)


import math


class TaskAlignedAssigner_ab_uncertainty_joint(TaskAlignedAssigner_Scale):
    """
    Joint uncertainty + object-scale for dynamic alpha/beta.

    Fuses per-anchor uncertainty and GT area: k = lambda_fusion * u_norm +
    (1 - lambda_fusion) * (1 - area_score). Small objects and high uncertainty
    yield higher k, which increases alpha (more cls weight) and decreases beta
    (less IoU weight). Uses standard IoU for overlap computation.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.eps = eps
        self.lambda_fusion = kwargs.get('lambda_fusion', 0.6)
        self.scale_min = math.log(kwargs.get('scale_min', 16))
        self.scale_max = math.log(kwargs.get('scale_max', 64))
        self.alpha_easy = kwargs.get('alpha_easy', 0.5)
        self.alpha_hard = kwargs.get('alpha_hard', 1.2)
        self.beta_easy = kwargs.get('beta_easy', 6.0)
        self.beta_hard = kwargs.get('beta_hard', 3.0)

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, uncertainty,
                anc_points, gt_labels, gt_bboxes, mask_gt,
                stride=None, **kwargs):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        num_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )
        return self._forward(pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride)

    def _forward(self, pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, anc_points, mask_gt, stride
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def dynamic_alpha(self, x):
        a, b, c, d = self.score_alpha
        x_safe = x + 1e-6
        return a + (b - a) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def dynamic_beta(self, x):
        a, b, c, d = self.score_beta
        x_safe = x + 1e-6
        return b + (a - b) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def get_pos_mask(self, pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, align_metric_for_score = self.get_box_metrics(pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_for_score, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        u_norm = torch.tanh(uncertainty / 2.0)
        u_norm = u_norm.unsqueeze(1).expand(-1, self.n_max_boxes, -1)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_scale = torch.sqrt(torch.clamp(gt_areas, min=1.0))
        area_score = ((gt_scale.log() - self.scale_min)/ (self.scale_max - self.scale_min)).clamp(min=0.0, max=1.0)
        area_score = area_score.unsqueeze(-1).expand(-1, -1, na)

        k = self.lambda_fusion * u_norm + (1 - self.lambda_fusion) * (1 - area_score)

        dynamic_alpha = self.alpha_easy + (self.alpha_hard - self.alpha_easy) * k
        dynamic_beta = self.beta_easy + (self.beta_hard - self.beta_easy) * k


        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(dynamic_alpha) * overlaps.pow(dynamic_beta)
        # align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps, align_metric


class TaskAlignedAssigner_ab_uncertainty_simd_joint(TaskAlignedAssigner_Scale):
    """
    Joint uncertainty + scale with SimD (Separate IoU for Different roles).

    Same alpha/beta fusion as ab_uncertainty_joint, but uses different IoU types:
    - overlap_iou_type: for select_highest_overlaps / GT assignment
    - align_iou_type: for topk candidate ranking (e.g. Hausdorff for small objects)
    - score_iou_type: for soft target score normalization
    Configurable via overlap_iou_type, align_iou_type, score_iou_type.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.eps = eps
        self.lambda_fusion = kwargs.get('lambda_fusion', 0.6)
        self.scale_min = math.log(kwargs.get('scale_min', 16))
        self.scale_max = math.log(kwargs.get('scale_max', 64))
        self.alpha_easy = kwargs.get('alpha_easy', 0.5)
        self.alpha_hard = kwargs.get('alpha_hard', 1.2)
        self.beta_easy = kwargs.get('beta_easy', 6.0)
        self.beta_hard = kwargs.get('beta_hard', 3.0)

        self.overlap_iou_type = kwargs.get("overlap_iou_type", "CIoU")
        self.overlap_iou_kwargs = kwargs.get("overlap_iou_kwargs", {})
        self.align_iou_type = kwargs.get("align_iou_type", "Hausdorff")
        self.align_iou_kwargs = kwargs.get("align_iou_kwargs", {})
        self.score_iou_type = kwargs.get("score_iou_type", "CIoU")
        self.score_iou_kwargs = kwargs.get("score_iou_kwargs", {})

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, uncertainty,
                anc_points, gt_labels, gt_bboxes, mask_gt,
                stride=None, **kwargs):
        self.bs = pd_scores.shape[0]
        self.n_max_boxes = gt_bboxes.shape[1]
        num_anchors = anc_points.shape[0]
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )
        return self._forward(pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride)

    def _forward(self, pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride):
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, anc_points, mask_gt, stride
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        align_metric = align_metric * mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def dynamic_alpha(self, x):
        a, b, c, d = self.score_alpha
        x_safe = x + 1e-6
        return a + (b - a) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def dynamic_beta(self, x):
        a, b, c, d = self.score_beta
        x_safe = x + 1e-6
        return b + (a - b) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def get_pos_mask(self, pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric_for_align, overlaps, align_metric_for_score = self.get_box_metrics(pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric_for_align, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_for_score, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, uncertainty, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        overlaps_for_align = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        overlaps_for_score = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        u_norm = torch.tanh(uncertainty / 2.0)
        u_norm = u_norm.unsqueeze(1).expand(-1, self.n_max_boxes, -1)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_scale = torch.sqrt(torch.clamp(gt_areas, min=1.0))
        area_score = ((gt_scale.log() - self.scale_min)/ (self.scale_max - self.scale_min)).clamp(min=0.0, max=1.0)
        area_score = area_score.unsqueeze(-1).expand(-1, -1, na)

        k = self.lambda_fusion * u_norm + (1 - self.lambda_fusion) * (1 - area_score)

        dynamic_alpha = self.alpha_easy + (self.alpha_hard - self.alpha_easy) * k
        dynamic_beta = self.beta_easy + (self.beta_hard - self.beta_easy) * k

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes, self.overlap_iou_type, self.overlap_iou_kwargs)
        overlaps_for_align[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes, self.align_iou_type, self.align_iou_kwargs)
        overlaps_for_score[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes, self.score_iou_type, self.score_iou_kwargs)

        bbox_scores_part = bbox_scores.pow(dynamic_alpha)
        align_metric_for_align = bbox_scores_part * overlaps_for_align.pow(dynamic_beta)
        align_metric_for_score = bbox_scores_part * overlaps_for_score.pow(dynamic_beta)
        return align_metric_for_align, overlaps, align_metric_for_score

    def iou_calculation(self, gt_bboxes, pd_bboxes, iou_type=None, iou_kwargs=None):
        return bbox_iou_ext(pd_bboxes, gt_bboxes, xywh=False,
                            iou_type=iou_type, iou_kargs=iou_kwargs).squeeze(-1).clamp_(0)


