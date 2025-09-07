import os, pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import bbox_iou
from . import LOGGER, colorstr
from .tal import TaskAlignedAssigner

_STEP = 0
_SAVE_DIR = "assign_record"
os.makedirs(_SAVE_DIR, exist_ok=True)

def save_assign_info(dir_name, **kwargs):
    save_path = os.path.join(dir_name, f"{_STEP}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(kwargs, f)

class TaskAlignedAssigner_Record(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 dir_name: str = 'test', save_step: int = 10):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.dir_name = os.path.join(_SAVE_DIR, dir_name)
        os.makedirs(self.dir_name, exist_ok=True)
        self.save_step = save_step

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

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
        device = gt_bboxes.device
        global _STEP
        _STEP += 1

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        """
        mask_pos: (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        align_metric: (b, max_num_obj, h*w) alignment metric for positive anchor points.
        overlaps: (b, max_num_obj, h*w) IoU_based overlaps between predicted and ground truth boxes for positive anchor points.
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # dealing with more than one assigned anchor,
        """
        target_gt_idx : (b, h*w) indicator of the assigned ground truth object for positive anchor points, with shape (b, h*w)
        fg_mask : (b, h*w) boolean tensor indicating the positive (foreground) anchor points.
        mask_pos : (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        """
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        """
        target_labels : (b, h*w) target labels for positive anchor points, with shape (b, h*w).
        target_bboxes : (b, h*w, 4) target bounding boxes for positive anchor points, with shape (b, h*w, 4).
        target_scores : (b, h*w, num_classes) target scores for positive anchor points, with shape (b, h*w, num_classes).
        """
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric_ = align_metric * mask_pos
        pos_align_metrics = align_metric_.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric_ * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        if _STEP % self.save_step == 0:
            save_assign_info(
                dir_name=self.dir_name,
                pd_scores=pd_scores.detach().cpu(),
                pd_bboxes=pd_bboxes.detach().cpu(),
                anc_points=anc_points.detach().cpu(),
                gt_labels=gt_labels.detach().cpu(),
                gt_bboxes=gt_bboxes.detach().cpu(),
                mask_gt=mask_gt.detach().cpu(),
                fg_mask=fg_mask.detach().cpu(),
                target_gt_idx=target_gt_idx.detach().cpu(),
                target_scores=target_scores.detach().cpu(),
                align_metric=align_metric.detach().cpu(),
                mask_pos=mask_pos.detach().cpu(),
                overlaps=overlaps.detach().cpu(),
            )

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where
                topk is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device) # b, max_num_obj, num_anch
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes (due to the unstable behavior of topk)
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        """
        Compute target labels, target bounding boxes, and target scores for the positive anchor points.

        Args:
            gt_labels (torch.Tensor): Ground truth labels of shape (b, max_num_obj, 1), where b is the
                                batch size and max_num_obj is the maximum number of objects.
            gt_bboxes (torch.Tensor): Ground truth bounding boxes of shape (b, max_num_obj, 4).
            target_gt_idx (torch.Tensor): Indices of the assigned ground truth objects for positive
                                    anchor points, with shape (b, h*w), where h*w is the total
                                    number of anchor points.
            fg_mask (torch.Tensor): A boolean tensor of shape (b, h*w) indicating the positive
                              (foreground) anchor points.

        Returns:
            target_labels (torch.Tensor): Target labels for positive anchor points with shape (b, h*w).
            target_bboxes (torch.Tensor): Target bounding boxes for positive anchor points with shape (b, h*w, 4).
            target_scores (torch.Tensor): Target scores for positive anchor points with shape (b, h*w, num_classes).
        """
        # Assigned target labels, (b, 1)
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes  # (b, h*w)
        target_labels = gt_labels.long().flatten()[target_gt_idx]  # (b, h*w)

        # Assigned target boxes, (b, max_num_obj, 4) -> (b, h*w, 4)
        target_bboxes = gt_bboxes.view(-1, gt_bboxes.shape[-1])[target_gt_idx]

        # Assigned target scores
        target_labels.clamp_(0)

        # 10x faster than F.one_hot()
        target_scores = torch.zeros(
            (target_labels.shape[0], target_labels.shape[1], self.num_classes),
            dtype=torch.int64,
            device=target_labels.device,
        )  # (b, h*w, 80)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)

        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)  # (b, h*w, 80)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, 0)

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        """
        Select anchor boxes with highest IoU when assigned to multiple ground truths.

        Args:
            mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
            overlaps (torch.Tensor): IoU overlaps, shape (b, n_max_boxes, h*w).
            n_max_boxes (int): Maximum number of ground truth boxes.

        Returns:
            target_gt_idx (torch.Tensor): Indices of assigned ground truths, shape (b, h*w).
            fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
            mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
        """
        # Convert (b, n_max_boxes, h*w) -> (b, h*w)
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            max_overlaps_idx = overlaps.argmax(1)  # (b, h*w)

            is_max_overlaps = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype, device=mask_pos.device)
            is_max_overlaps.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        # Find each grid serve which gt(index)
        target_gt_idx = mask_pos.argmax(-2)  # (b, h*w)
        return target_gt_idx, fg_mask, mask_pos


class TaskAlignedAssigner_BCE(TaskAlignedAssigner):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.
        Using BCE loss as the classification quality score.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores (logits) with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, max_num_obj, h*w)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)

        # --- Start of Modification ---

        # 1. Prepare ground-truth scores (one-hot)
        # Shape: (b, max_num_obj, num_classes)
        gt_scores = torch.zeros(
            (self.bs, self.n_max_boxes, self.num_classes),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        gt_scores.scatter_(2, gt_labels.to(torch.int64), 1)

        # 2. Expand tensors to a common shape for broadcasting
        # pd_scores: (b, na, nc) -> (b, 1, na, nc) -> (b, max_num_obj, na, nc)
        # gt_scores: (b, max_num_obj, nc) -> (b, max_num_obj, 1, nc) -> (b, max_num_obj, na, nc)
        pd_scores_expanded = pd_scores.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
        gt_scores_expanded = gt_scores.unsqueeze(2).expand(-1, -1, na, -1)

        # 3. Calculate BCE loss for all (gt, anchor) pairs
        # NOTE: It's better to use BCEWithLogitsLoss for numerical stability,
        # assuming pd_scores are logits (pre-sigmoid).
        # The output shape of bce_loss will be (b, max_num_obj, na, nc)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pd_scores_expanded, gt_scores_expanded)

        # 4. Sum loss over classes and convert loss to a similarity score
        # A common way is to use exp(-loss). A lower loss gives a higher score.
        # Shape of bbox_scores: (b, max_num_obj, na)
        bbox_scores = torch.exp(-bce_loss.sum(dim=-1))

        # --- End of Modification ---

        # Calculate IoU
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # Note: Only calculate IoU for the anchors that are inside GTs (masked by mask_gt)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Calculate final alignment metric
        # mask_gt is applied to bbox_scores to filter out anchors not in the preliminary candidates
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta) * mask_gt

        return align_metric, overlaps

class TaskAlignedAssigner_BCE1(TaskAlignedAssigner):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """
    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, align_metric_p = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_p, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.
        Using BCE loss as the classification quality score.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores (logits) with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, max_num_obj, h*w)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)

        # --- Start of Modification ---

        # 1. Prepare ground-truth scores (one-hot)
        # Shape: (b, max_num_obj, num_classes)
        gt_scores = torch.zeros(
            (self.bs, self.n_max_boxes, self.num_classes),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        gt_scores.scatter_(2, gt_labels.to(torch.int64), 1)

        # 2. Expand tensors to a common shape for broadcasting
        # pd_scores: (b, na, nc) -> (b, 1, na, nc) -> (b, max_num_obj, na, nc)
        # gt_scores: (b, max_num_obj, nc) -> (b, max_num_obj, 1, nc) -> (b, max_num_obj, na, nc)
        pd_scores_expanded = pd_scores.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
        gt_scores_expanded = gt_scores.unsqueeze(2).expand(-1, -1, na, -1)

        # 3. Calculate BCE loss for all (gt, anchor) pairs
        # NOTE: It's better to use BCEWithLogitsLoss for numerical stability,
        # assuming pd_scores are logits (pre-sigmoid).
        # The output shape of bce_loss will be (b, max_num_obj, na, nc)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pd_scores_expanded, gt_scores_expanded)

        # 4. Sum loss over classes and convert loss to a similarity score
        # A common way is to use exp(-loss). A lower loss gives a higher score.
        # Shape of bbox_scores: (b, max_num_obj, na)
        bbox_scores = torch.exp(-bce_loss.sum(dim=-1))

        # --- End of Modification ---

        # Calculate IoU
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # Note: Only calculate IoU for the anchors that are inside GTs (masked by mask_gt)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Previous bbox_scores cal
        bbox_scores_p = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores_p[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # Calculate final alignment metric
        # mask_gt is applied to bbox_scores to filter out anchors not in the preliminary candidates
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        align_metric_p = bbox_scores_p.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps, align_metric_p

class TaskAlignedAssigner_BCE2(TaskAlignedAssigner_BCE1):
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.
        Using BCE loss as the classification quality score.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores (logits) with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, max_num_obj, h*w)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)

        # --- Start of Modification ---

        # 1. Prepare ground-truth scores (one-hot)
        # Shape: (b, max_num_obj, num_classes)
        gt_scores = torch.zeros(
            (self.bs, self.n_max_boxes, self.num_classes),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        gt_scores.scatter_(2, gt_labels.to(torch.int64), 1)

        # 2. Expand tensors to a common shape for broadcasting
        # pd_scores: (b, na, nc) -> (b, 1, na, nc) -> (b, max_num_obj, na, nc)
        # gt_scores: (b, max_num_obj, nc) -> (b, max_num_obj, 1, nc) -> (b, max_num_obj, na, nc)
        pd_scores_expanded = pd_scores.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
        gt_scores_expanded = gt_scores.unsqueeze(2).expand(-1, -1, na, -1)

        # 3. Calculate BCE loss for all (gt, anchor) pairs
        # NOTE: It's better to use BCEWithLogitsLoss for numerical stability,
        # assuming pd_scores are logits (pre-sigmoid).
        # The output shape of bce_loss will be (b, max_num_obj, na, nc)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pd_scores_expanded, gt_scores_expanded)

        # 4. Sum loss over classes and convert loss to a similarity score
        # A common way is to use exp(-loss). A lower loss gives a higher score.
        # Shape of bbox_scores: (b, max_num_obj, na)
        bbox_scores = torch.exp(-bce_loss.sum(dim=-1))

        # --- End of Modification ---

        # Calculate IoU
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # Note: Only calculate IoU for the anchors that are inside GTs (masked by mask_gt)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Previous bbox_scores cal
        bbox_scores_p = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        pd_scores_p = pd_scores.sigmoid()
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores_p[mask_gt] = pd_scores_p[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # Calculate final alignment metric
        # mask_gt is applied to bbox_scores to filter out anchors not in the preliminary candidates
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        align_metric_p = bbox_scores_p.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps, align_metric_p

class TaskAlignedAssigner_General(TaskAlignedAssigner_BCE1):
    def __init__(self, topk=10, num_classes=100, alpha=1.0, beta=1.0, eps=1e-8, **kwargs):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.align_cost = self.__getattr__(f"_{kwargs['align_type']}")

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
            pd_scores (torch.Tensor): Predicted classification scores (logits) with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, max_num_obj, h*w)

        overlaps, align_metric_p = self.score_guide_metric(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt)

        # Shape: (b, max_num_obj, num_classes) one-hot
        gt_scores = torch.zeros(
            (self.bs, self.n_max_boxes, self.num_classes),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        gt_scores.scatter_(2, gt_labels.to(torch.int64), 1)

        # pd_scores: (b, na, nc) -> (b, 1, na, nc) -> (b, max_num_obj, na, nc)
        # gt_scores: (b, max_num_obj, nc) -> (b, max_num_obj, 1, nc) -> (b, max_num_obj, na, nc)
        pd_scores_expanded = pd_scores.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
        gt_scores_expanded = gt_scores.unsqueeze(2).expand(-1, -1, na, -1)

        # The output shape of bce_loss will be (b, max_num_obj, na, nc)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pd_scores_expanded, gt_scores_expanded)

        # Shape of bbox_scores: (b, max_num_obj, na)
        bbox_scores = torch.exp(-bce_loss.sum(dim=-1))
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps, align_metric_p

    def _gaussian_kernel(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        bbox_scores_p = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores_p[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, na


    def score_guide_metric(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        # Calculate final alignment metric
        # mask_gt is applied to bbox_scores to filter out anchors not in the preliminary candidates

        # Previous bbox_scores cal
        bbox_scores_p = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores_p[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w
        # Calculate Score for Supervision
        align_metric_p = bbox_scores_p.pow(self.alpha) * overlaps.pow(self.beta)
        return overlaps, align_metric_p


class TaskAlignedAssigner_Scale(TaskAlignedAssigner):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 scale_ratio=1.0):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.scale_ratio = scale_ratio


    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

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
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        """
        mask_pos: (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        align_metric: (b, max_num_obj, h*w) alignment metric for positive anchor points.
        overlaps: (b, max_num_obj, h*w) IoU_based overlaps between predicted and ground truth boxes for positive anchor points.
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, stride
        )

        # dealing with more than one assigned anchor,
        """
        target_gt_idx : (b, h*w) indicator of the assigned ground truth object for positive anchor points, with shape (b, h*w)
        fg_mask : (b, h*w) boolean tensor indicating the positive (foreground) anchor points.
        mask_pos : (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        """
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        """
        target_labels : (b, h*w) target labels for positive anchor points, with shape (b, h*w).
        target_bboxes : (b, h*w, 4) target bounding boxes for positive anchor points, with shape (b, h*w, 4).
        target_scores : (b, h*w, num_classes) target scores for positive anchor points, with shape (b, h*w, num_classes).
        """
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9, stride=None):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(stride * self.scale_ratio)


class TaskAlignedAssigner_Scale_BCE1(TaskAlignedAssigner):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 scale_ratio=1.0):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.scale_ratio = scale_ratio


    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

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
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, stride
        )

        # dealing with more than one assigned anchor,
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx


    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, align_metric_p = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_p, overlaps

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9, stride=None):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(stride * self.scale_ratio)

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.
        Using BCE loss as the classification quality score.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores (logits) with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, max_num_obj, h*w)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)

        # --- Start of Modification ---

        # 1. Prepare ground-truth scores (one-hot)
        # Shape: (b, max_num_obj, num_classes)
        gt_scores = torch.zeros(
            (self.bs, self.n_max_boxes, self.num_classes),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        gt_scores.scatter_(2, gt_labels.to(torch.int64), 1)

        # 2. Expand tensors to a common shape for broadcasting
        # pd_scores: (b, na, nc) -> (b, 1, na, nc) -> (b, max_num_obj, na, nc)
        # gt_scores: (b, max_num_obj, nc) -> (b, max_num_obj, 1, nc) -> (b, max_num_obj, na, nc)
        pd_scores_expanded = pd_scores.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
        gt_scores_expanded = gt_scores.unsqueeze(2).expand(-1, -1, na, -1)

        # 3. Calculate BCE loss for all (gt, anchor) pairs
        # NOTE: It's better to use BCEWithLogitsLoss for numerical stability,
        # assuming pd_scores are logits (pre-sigmoid).
        # The output shape of bce_loss will be (b, max_num_obj, na, nc)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pd_scores_expanded, gt_scores_expanded)

        # 4. Sum loss over classes and convert loss to a similarity score
        # A common way is to use exp(-loss). A lower loss gives a higher score.
        # Shape of bbox_scores: (b, max_num_obj, na)
        bbox_scores = torch.exp(-bce_loss.sum(dim=-1))

        # --- End of Modification ---

        # Calculate IoU
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # Note: Only calculate IoU for the anchors that are inside GTs (masked by mask_gt)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Previous bbox_scores cal
        bbox_scores_p = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores_p[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # Calculate final alignment metric
        # mask_gt is applied to bbox_scores to filter out anchors not in the preliminary candidates
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        align_metric_p = bbox_scores_p.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps, align_metric_p

class TaskAlignedAssigner_Scale_BCE2(TaskAlignedAssigner_Scale_BCE1):
    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.
        Using BCE loss as the classification quality score.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores (logits) with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # (b, max_num_obj, h*w)
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)

        # --- Start of Modification ---

        # 1. Prepare ground-truth scores (one-hot)
        # Shape: (b, max_num_obj, num_classes)
        gt_scores = torch.zeros(
            (self.bs, self.n_max_boxes, self.num_classes),
            dtype=pd_scores.dtype,
            device=pd_scores.device,
        )
        gt_scores.scatter_(2, gt_labels.to(torch.int64), 1)

        # 2. Expand tensors to a common shape for broadcasting
        # pd_scores: (b, na, nc) -> (b, 1, na, nc) -> (b, max_num_obj, na, nc)
        # gt_scores: (b, max_num_obj, nc) -> (b, max_num_obj, 1, nc) -> (b, max_num_obj, na, nc)
        pd_scores_expanded = pd_scores.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
        gt_scores_expanded = gt_scores.unsqueeze(2).expand(-1, -1, na, -1)

        # 3. Calculate BCE loss for all (gt, anchor) pairs
        # NOTE: It's better to use BCEWithLogitsLoss for numerical stability,
        # assuming pd_scores are logits (pre-sigmoid).
        # The output shape of bce_loss will be (b, max_num_obj, na, nc)
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(pd_scores_expanded, gt_scores_expanded)

        # 4. Sum loss over classes and convert loss to a similarity score
        # A common way is to use exp(-loss). A lower loss gives a higher score.
        # Shape of bbox_scores: (b, max_num_obj, na)
        bbox_scores = torch.exp(-bce_loss.sum(dim=-1))

        # --- End of Modification ---

        # Calculate IoU
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        # Note: Only calculate IoU for the anchors that are inside GTs (masked by mask_gt)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        # Previous bbox_scores cal
        bbox_scores_p = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)
        pd_scores_p = pd_scores.sigmoid()
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores_p[mask_gt] = pd_scores_p[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # Calculate final alignment metric
        # mask_gt is applied to bbox_scores to filter out anchors not in the preliminary candidates
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        align_metric_p = bbox_scores_p.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps, align_metric_p

class TaskAlignedAssigner_MixAssign(nn.Module):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).
            stride (torch.Tensor, optional): The stride of the anchor. (bs, num_total_anchors, 1))

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
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        """
        mask_pos: (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        align_metric: (b, max_num_obj, h*w) alignment metric for positive anchor points.
        overlaps: (b, max_num_obj, h*w) IoU_based overlaps between predicted and ground truth boxes for positive anchor points.
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        # dealing with more than one assigned anchor,
        """
        target_gt_idx : (b, max_num_obj, h*w) indicator of the assigned ground truth object for positive anchor points, with shape (b, h*w)
        fg_mask : (b, h*w) boolean tensor indicating the positive (foreground) anchor points.
        mask_pos : (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        """
        assign_pos_fin, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes, top_k=2)

        # Assigned target
        """
        target_labels : (b, h*w) target labels for positive anchor points, with shape (b, h*w).
        target_bboxes : (b, h*w, 4) target bounding boxes for positive anchor points, with shape (b, h*w, 4).
        target_scores : (b, h*w, num_classes) target scores for positive anchor points, with shape (b, h*w, num_classes).
        """
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, mask_pos, fg_mask)

        max_indices = mask_pos.argmax(dim=1, keepdim=True)  # 保持维度方便后续广播
        mask = torch.zeros_like(mask_pos)
        mask.scatter_(1, max_indices, 1)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric *= mask
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), assign_pos_fin

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        """
        Calculate IoU for horizontal bounding boxes.

        Args:
            gt_bboxes (torch.Tensor): Ground truth boxes.
            pd_bboxes (torch.Tensor): Predicted boxes.

        Returns:
            (torch.Tensor): IoU values between each pair of boxes.
        """
        return bbox_iou(gt_bboxes, pd_bboxes, xywh=False, CIoU=True).squeeze(-1).clamp_(0)

    def select_topk_candidates(self, metrics, topk_mask=None):
        """
        Select the top-k candidates based on the given metrics.

        Args:
            metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
                the maximum number of objects, and h*w represents the total number of anchor points.
            topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where
                topk is the number of top candidates to consider. If not provided, the top-k values are automatically
                computed based on the given metrics.

        Returns:
            (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
        """
        # (b, max_num_obj, topk)
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1, largest=True)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)
        # (b, max_num_obj, topk)
        topk_idxs.masked_fill_(~topk_mask, 0)

        # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
        count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device) # b, max_num_obj, num_anch
        ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
        for k in range(self.topk):
            # Expand topk_idxs for each value of k and add 1 at the specified positions
            count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
        # Filter invalid bboxes (due to the unstable behavior of topk)
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, target_weights, fg_mask):
        """
        Compute soft target labels, boxes, and scores using a weight matrix.

        Args:
            gt_labels (Tensor): (b, n_max_boxes, 1) long
            gt_bboxes (Tensor): (b, n_max_boxes, 4) float
            target_weights (Tensor): (b, n_max_boxes, h*w) soft weights from select_highest_overlaps
            fg_mask (Tensor): (b, h*w) bool mask for positive anchors

        Returns:
            target_labels (Tensor): (b, h*w, num_classes) one-hot/prob
            target_bboxes (Tensor): (b, h*w, 4) weighted boxes
            target_scores (Tensor): same as target_labels (alias for loss calc)
        """
        b, n_max_boxes, hw = target_weights.shape

        # one-hot（b, n_max_boxes, num_classes）
        one_hot_labels = torch.zeros(
            (b, n_max_boxes, self.num_classes),
            device=gt_labels.device,
            dtype=torch.float
        )
        one_hot_labels.scatter_(2, gt_labels.long(), 1.0)

        # mix for score
        target_labels = torch.bmm(
            target_weights.permute(0, 2, 1),  # (b, hw, n_max_boxes)
            one_hot_labels  # (b, n_max_boxes, num_classes)
        )  # (b, hw, num_classes)

        # mix for bbox
        target_bboxes = torch.bmm(
            target_weights.permute(0, 2, 1),  # (b, hw, n_max_boxes)
            gt_bboxes  # (b, n_max_boxes, 4)
        )  # (b, hw, 4)

        # 4. filter using fg_mask
        fg_mask_exp = fg_mask.unsqueeze(-1).float()  # (b, hw, 1)
        target_labels = target_labels * fg_mask_exp
        target_bboxes = target_bboxes * fg_mask_exp

        # target_scores 可以直接等于 target_labels（soft label）
        target_scores = target_labels

        return target_labels, target_bboxes, target_scores

    @staticmethod
    def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes, top_k=1):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:  # one anchor is assigned to multiple gt_bboxes
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes, -1)  # (b, n_max_boxes, h*w)
            topk_overlaps, topk_idxs = torch.topk(overlaps, k=top_k, dim=1)
            is_max_overlaps = torch.full_like(overlaps, float('-inf'))
            is_max_overlaps.scatter_(1, topk_idxs, topk_overlaps)
            is_max_overlaps = is_max_overlaps.softmax(dim=1)

            mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos).float()  # (b, n_max_boxes, h*w)
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    # @staticmethod
    # def select_highest_overlaps(mask_pos, overlaps, n_max_boxes, top_k=1):
    #     """
    #     Select anchor boxes with highest IoU when assigned to multiple ground truths.
    #
    #     Args:
    #         mask_pos (torch.Tensor): Positive mask, shape (b, n_max_boxes, h*w).
    #         overlaps (torch.Tensor): Here we change to the align metric, shape (b, n_max_boxes, h*w).
    #         n_max_boxes (int): Maximum number of ground truth boxes.
    #
    #     Returns:
    #         assign_pos_fin (torch.Tensor): output of a weight matrix, shape (b, n_max_boxes, h*w).
    #         fg_mask (torch.Tensor): Foreground mask, shape (b, h*w).
    #         mask_pos (torch.Tensor): Updated positive mask, shape (b, n_max_boxes, h*w).
    #     """
    #     # Convert (b, n_max_boxes, h*w) -> (b, h*w)
    #     overlaps_masked = overlaps * mask_pos.float()
    #     fg_mask = mask_pos.sum(-2).gt(0)
    #     assign_pos_fin = overlaps_masked.clone()
    #     multi_gt_mask = mask_pos.sum(-2).gt(0)
    #     if multi_gt_mask.any():
    #         topk_overlaps, topk_idxs = torch.topk(overlaps_masked, k=top_k, dim=1)
    #         weights = topk_overlaps / (topk_overlaps.sum(-1, keepdim=True) + 1e-9)
    #         assign_pos_multi = torch.zeros_like(assign_pos_fin)
    #         assign_pos_multi.scatter(1, topk_idxs, weights)
    #         multi_gt_mask = multi_gt_mask.unsqueeze(1).expand(-1, n_max_boxes, -1)
    #         assign_pos_fin[multi_gt_mask] = assign_pos_multi[multi_gt_mask]
    #     return assign_pos_fin, fg_mask.clamp_max(1), mask_pos

class TaskAlignedAssigner_dynamicK(TaskAlignedAssigner):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 **kwargs):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.max_topk = kwargs['max_topk'] if 'max_topk' in kwargs else 10
        self.min_topk = kwargs['min_topk'] if 'min_topk' in kwargs else 3
        self.metric_sum_thr = kwargs['metric_sum_thr'] if 'metric_sum_thr' in kwargs else 4.0
        self.topk = self.max_topk

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_dynamic_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def select_dynamic_topk_candidates(self, metrics, topk_mask=None,):
        """
        Vectorized dynamic top-k selection.

        Args:
            metrics (torch.Tensor): Shape (b, max_num_obj, h*w)
            min_topk (int): Minimum number of candidates
            max_topk (int): Maximum number of candidates
            metric_sum_thr (float): Threshold for cumulative sum of metrics
            topk_mask (torch.Tensor, optional): Boolean mask for valid top-k
            eps (float, optional): Small value for numerical stability. Default: 1e-9.
        Returns:
            torch.Tensor: Shape (b, max_num_obj, h*w)
        """
        b, max_num_obj, num_anch = metrics.shape

        # 先取最大可能需要的 topk
        topk_metrics, topk_idxs = torch.topk(metrics, self.max_topk, dim=-1, largest=True)

        # 默认 mask
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)

        # 计算累积和并找到动态 topk
        cumsum_metrics = torch.cumsum(topk_metrics, dim=-1)  # (b, max_num_obj, max_topk)
        over_thr = (cumsum_metrics > self.metric_sum_thr)

        # 第一次超过阈值的位置
        first_over_idx = over_thr.float().argmax(dim=-1)  # (b, max_num_obj)
        # 如果从未超过，则设为 max_topk
        no_exceed = over_thr.sum(dim=-1) == 0
        first_over_idx[no_exceed] = self.max_topk

        # clamp 到 [min_topk, max_topk]
        dynamic_topk = torch.clamp(first_over_idx, min=self.min_topk, max=self.max_topk)  # (b, max_num_obj)

        # 构造每个位置的选择 mask
        arange_k = torch.arange(self.max_topk, device=metrics.device).view(1, 1, -1)  # (1, 1, max_topk)
        k_mask = arange_k < dynamic_topk.unsqueeze(-1)  # (b, max_num_obj, max_topk)

        # 同时满足 topk_mask 和 dynamic_k_mask
        final_mask = k_mask & topk_mask.bool()  # (b, max_num_obj, max_topk)

        # 生成 count_tensor
        count_tensor = torch.zeros_like(metrics, dtype=torch.int8)  # (b, max_num_obj, num_anch)
        count_tensor.scatter_add_(
            -1,
            topk_idxs * final_mask,  # mask 后的索引（未选中的会是 0，但因为 mask=0 不会加）
            final_mask.to(torch.int8)
        )

        # 过滤重复（理论上不会出现，但保险起见）
        count_tensor.masked_fill_(count_tensor > 1, 0)
        return count_tensor.to(metrics.dtype)

class TaskAlignedAssigner_Scale_dynamicK(TaskAlignedAssigner):
    """
    A task-aligned assigner for object detection.

    This class assigns ground-truth (gt) objects to anchors based on the task-aligned metric, which combines both
    classification and localization information.

    Attributes:
        topk (int): The number of top candidates to consider.
        num_classes (int): The number of object classes.
        alpha (float): The alpha parameter for the classification component of the task-aligned metric.
        beta (float): The beta parameter for the localization component of the task-aligned metric.
        eps (float): A small value to prevent division by zero.
    """
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 scale_ratio=1.0, **kwargs):
        """
        Initialize a TaskAlignedAssigner object with customizable hyperparameters.

        Args:
            topk (int, optional): The number of top candidates to consider.
            num_classes (int, optional): The number of object classes.
            alpha (float, optional): The alpha parameter for the classification component of the task-aligned metric.
            beta (float, optional): The beta parameter for the localization component of the task-aligned metric.
            eps (float, optional): A small value to prevent division by zero.
        """
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps)
        self.scale_ratio = scale_ratio
        self.max_topk = kwargs['max_topk'] if 'max_topk' in kwargs else 10
        self.min_topk = kwargs['min_topk'] if 'min_topk' in kwargs else 3
        self.metric_sum_thr = kwargs['metric_sum_thr'] if 'metric_sum_thr' in kwargs else 4.0
        self.topk = self.max_topk

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

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
        device = gt_bboxes.device

        if self.n_max_boxes == 0:
            return (
                torch.full_like(pd_scores[..., 0], self.num_classes),
                torch.zeros_like(pd_bboxes),
                torch.zeros_like(pd_scores),
                torch.zeros_like(pd_scores[..., 0]),
                torch.zeros_like(pd_scores[..., 0]),
            )

        try:
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride)
        except torch.cuda.OutOfMemoryError:
            # Move tensors to CPU, compute, then move back to original device
            LOGGER.warning("CUDA OutOfMemoryError in TaskAlignedAssigner, using CPU")
            cpu_tensors = [t.cpu() for t in (pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride)]
            result = self._forward(*cpu_tensors)
            return tuple(t.to(device) for t in result)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        """
        mask_pos: (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        align_metric: (b, max_num_obj, h*w) alignment metric for positive anchor points.
        overlaps: (b, max_num_obj, h*w) IoU_based overlaps between predicted and ground truth boxes for positive anchor points.
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, stride
        )

        # dealing with more than one assigned anchor,
        """
        target_gt_idx : (b, h*w) indicator of the assigned ground truth object for positive anchor points, with shape (b, h*w)
        fg_mask : (b, h*w) boolean tensor indicating the positive (foreground) anchor points.
        mask_pos : (b, max_num_obj, h*w) boolean tensor indicating the positive (foreground) anchor points.
        """
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

        # Assigned target
        """
        target_labels : (b, h*w) target labels for positive anchor points, with shape (b, h*w).
        target_bboxes : (b, h*w, 4) target bounding boxes for positive anchor points, with shape (b, h*w, 4).
        target_scores : (b, h*w, num_classes) target scores for positive anchor points, with shape (b, h*w, num_classes).
        """
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        """
        Get positive mask for each ground truth box.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            mask_pos (torch.Tensor): Positive mask with shape (bs, max_num_obj, h*w).
            align_metric (torch.Tensor): Alignment metric with shape (bs, max_num_obj, h*w).
            overlaps (torch.Tensor): Overlaps between predicted and ground truth boxes with shape (bs, max_num_obj, h*w).
        """
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_dynamic_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps

    def select_dynamic_topk_candidates(self, metrics, topk_mask=None,):
        """
        Vectorized dynamic top-k selection.

        Args:
            metrics (torch.Tensor): Shape (b, max_num_obj, h*w)
            min_topk (int): Minimum number of candidates
            max_topk (int): Maximum number of candidates
            metric_sum_thr (float): Threshold for cumulative sum of metrics
            topk_mask (torch.Tensor, optional): Boolean mask for valid top-k
            eps (float, optional): Small value for numerical stability. Default: 1e-9.
        Returns:
            torch.Tensor: Shape (b, max_num_obj, h*w)
        """
        b, max_num_obj, num_anch = metrics.shape

        # 先取最大可能需要的 topk
        topk_metrics, topk_idxs = torch.topk(metrics, self.max_topk, dim=-1, largest=True)

        # 默认 mask
        if topk_mask is None:
            topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > self.eps).expand_as(topk_idxs)

        # 计算累积和并找到动态 topk
        cumsum_metrics = torch.cumsum(topk_metrics, dim=-1)  # (b, max_num_obj, max_topk)
        over_thr = (cumsum_metrics > self.metric_sum_thr)

        # 第一次超过阈值的位置
        first_over_idx = over_thr.float().argmax(dim=-1)  # (b, max_num_obj)
        # 如果从未超过，则设为 max_topk
        no_exceed = over_thr.sum(dim=-1) == 0
        first_over_idx[no_exceed] = self.max_topk

        # clamp 到 [min_topk, max_topk]
        dynamic_topk = torch.clamp(first_over_idx, min=self.min_topk, max=self.max_topk)  # (b, max_num_obj)

        # 构造每个位置的选择 mask
        arange_k = torch.arange(self.max_topk, device=metrics.device).view(1, 1, -1)  # (1, 1, max_topk)
        k_mask = arange_k < dynamic_topk.unsqueeze(-1)  # (b, max_num_obj, max_topk)

        # 同时满足 topk_mask 和 dynamic_k_mask
        final_mask = k_mask & topk_mask.bool()  # (b, max_num_obj, max_topk)

        # 生成 count_tensor
        count_tensor = torch.zeros_like(metrics, dtype=torch.int8)  # (b, max_num_obj, num_anch)
        count_tensor.scatter_add_(
            -1,
            topk_idxs * final_mask,  # mask 后的索引（未选中的会是 0，但因为 mask=0 不会加）
            final_mask.to(torch.int8)
        )

        # 过滤重复（理论上不会出现，但保险起见）
        count_tensor.masked_fill_(count_tensor > 1, 0)

        return count_tensor.to(metrics.dtype)

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9, stride=None):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.

        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # left-top, right-bottom
        bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(bs, n_boxes, n_anchors, -1)
        return bbox_deltas.amin(3).gt_(stride * self.scale_ratio)

class SimOTA(nn.Module):
    def __init__(self):
        super().__init__()
        pass


