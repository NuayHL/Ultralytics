import torch
import torch.nn as nn

from .mla_scale import TaskAlignedAssigner_Scale

class TaskAlignedAssigner_Subnet_Scale(TaskAlignedAssigner_Scale):
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
