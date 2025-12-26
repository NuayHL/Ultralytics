import os, pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import bbox_iou, bbox_iou_ext
from . import LOGGER, colorstr
from .tal import TaskAlignedAssigner

_STEP = 0
_SAVE_DIR = "assign_record"
os.makedirs(_SAVE_DIR, exist_ok=True)

def save_assign_info(dir_name, **kwargs):
    save_path = os.path.join(dir_name, f"{_STEP}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(kwargs, f)

class TaskAlignedAssigner_dab(TaskAlignedAssigner):
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

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9):
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
        self.alpha = alpha if alpha else [0.3, 2.5, 2.0, 1.7]
        self.beta = beta if beta else [0.3, 5.0, 2.0, 1.7]
        self.eps = eps

    def dynamic_alpha(self, x):
        a, b, c, d = self.alpha
        x_safe = x + 1e-6
        return a + (b - a) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def dynamic_beta(self, x):
        a, b, c, d = self.beta
        x_safe = x + 1e-6
        return b + (a - b) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

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
                                        (bs, max_num_obj, h*w)
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

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_areas_safe = torch.clamp(gt_areas, min=1.0)
        dynamic_alpha = self.dynamic_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_beta = self.dynamic_beta(gt_areas_safe).unsqueeze(-1)
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(dynamic_alpha) * overlaps.pow(dynamic_beta)
        return align_metric, overlaps

class TaskAlignedAssigner_dabsep(TaskAlignedAssigner):
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

    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
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
        self.alpha = alpha if alpha else [0.3, 2.5, 2.0, 1.7]
        self.beta = beta if beta else [0.3, 5.0, 2.0, 1.7]
        self.eps = eps
        self.score_alpha = kwargs.get("score_alpha", 1.0)
        self.score_beta = kwargs.get("score_beta", 4.0)

    def dynamic_alpha(self, x):
        a, b, c, d = self.alpha
        x_safe = x + 1e-6
        return a + (b - a) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def dynamic_beta(self, x):
        a, b, c, d = self.beta
        x_safe = x + 1e-6
        return b + (a - b) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

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
        align_metric, overlaps, align_metric_for_score = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_for_score, overlaps

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
                                        (bs, max_num_obj, h*w)
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

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_areas_safe = torch.clamp(gt_areas, min=1.0)
        dynamic_alpha = self.dynamic_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_beta = self.dynamic_beta(gt_areas_safe).unsqueeze(-1)
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric = bbox_scores.pow(dynamic_alpha) * overlaps.pow(dynamic_beta)
        align_metric_for_score = bbox_scores.pow(self.score_alpha) * overlaps.pow(self.score_beta)
        return align_metric, overlaps, align_metric_for_score

class TaskAlignedAssigner_dabsepScore(TaskAlignedAssigner):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.eps = eps
        self.score_alpha = kwargs.get("score_alpha", [0.3, 2.5, 2.0, 1.7])
        self.score_beta = kwargs.get("score_beta", [0.3, 5.0, 2.0, 1.7])

    def dynamic_alpha(self, x):
        a, b, c, d = self.score_alpha
        x_safe = x + 1e-6
        return a + (b - a) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def dynamic_beta(self, x):
        a, b, c, d = self.score_beta
        x_safe = x + 1e-6
        return b + (a - b) / (1 + torch.exp(c * (torch.log10(x_safe) - d)))

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, align_metric_for_score = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_for_score, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_areas_safe = torch.clamp(gt_areas, min=1.0)
        dynamic_alpha = self.dynamic_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_beta = self.dynamic_beta(gt_areas_safe).unsqueeze(-1)
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric_for_score = bbox_scores.pow(dynamic_alpha) * overlaps.pow(dynamic_beta)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps, align_metric_for_score

class TaskAlignedAssigner_dabsepScore1(TaskAlignedAssigner):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.eps = eps
        self.score_alpha = kwargs.get("score_alpha", [-0.1, 1.3])
        self.score_beta = kwargs.get("score_beta", [0.33, 3.0])

    def dynamic_alpha(self, x):
        k, b = self.score_alpha
        return k * torch.log10(x) + b

    def dynamic_beta(self, x):
        k, b = self.score_beta
        return k * torch.log10(x) + b

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, align_metric_for_score = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_for_score, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_areas_safe = torch.clamp(gt_areas, min=1.0)
        dynamic_alpha = self.dynamic_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_beta = self.dynamic_beta(gt_areas_safe).unsqueeze(-1)
        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric_for_score = bbox_scores.pow(dynamic_alpha) * overlaps.pow(dynamic_beta)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps, align_metric_for_score

class CreateFunc:
    def __init__(self, func_name, *args, **kwargs):
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        return getattr(self, self.func_name)(x, *self.args, **self.kwargs)

    @staticmethod
    def log_linear(x , k, b):
        return k * torch.log10(x) + b

    @staticmethod
    def log_sigmoid(x, a, b ,c ,d):
        return a + (b - a) / (1 + torch.exp(c * (torch.log10(x) - d)))

    @staticmethod
    def static(x, a):
        return a

class TaskAlignedAssigner_dynaAB(TaskAlignedAssigner):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.eps = eps
        self.align_alpha = CreateFunc(kwargs.get("align_alpha", "log_linear"), *kwargs.get("align_alpha_args"))
        self.is_align_alpha_static = self.align_alpha.func_name == "static"
        self.align_beta = CreateFunc(kwargs.get("align_beta", "log_linear"), *kwargs.get("align_beta_args"))
        self.is_align_beta_static = self.align_beta.func_name == "static"
        self.score_alpha = CreateFunc(kwargs.get("score_alpha", "log_linear"), *kwargs.get("score_alpha_args"))
        self.is_score_alpha_static = self.score_alpha.func_name == "static"
        self.score_beta = CreateFunc(kwargs.get("score_beta", "log_linear"), *kwargs.get("score_beta_args"))
        self.is_score_beta_static = self.score_beta.func_name == "static"

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric_for_align, overlaps, align_metric_for_score = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric_for_align, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_for_score, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_areas_safe = torch.clamp(gt_areas, min=1.0)
        dynamic_score_alpha = self.score_alpha(gt_areas_safe) \
            if self.is_score_alpha_static else self.score_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_score_beta = self.score_beta(gt_areas_safe) \
            if self.is_score_beta_static else self.score_beta(gt_areas_safe).unsqueeze(-1)
        dynamic_align_alpha = self.align_alpha(gt_areas_safe) \
            if self.is_align_alpha_static else self.align_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_align_beta = self.align_beta(gt_areas_safe) \
            if self.is_align_beta_static else self.align_beta(gt_areas_safe).unsqueeze(-1)

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric_for_score = bbox_scores.pow(dynamic_score_alpha) * overlaps.pow(dynamic_score_beta)
        align_metric_for_align = bbox_scores.pow(dynamic_align_alpha) * overlaps.pow(dynamic_align_beta)
        return align_metric_for_align, overlaps, align_metric_for_score

from .mla_scale import TaskAlignedAssigner_dScale
class TaskAlignedAssigner_dynamicJoint_v1(TaskAlignedAssigner_dScale):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9,
                 scale_ratio=1.0, func_type='func_1', **kwargs):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps,
                         scale_ratio=scale_ratio, func_type=func_type)
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.align_alpha = CreateFunc(kwargs.get("align_alpha", "log_linear"), *kwargs.get("align_alpha_args"))
        self.is_align_alpha_static = self.align_alpha.func_name == "static"
        self.align_beta = CreateFunc(kwargs.get("align_beta", "log_linear"), *kwargs.get("align_beta_args"))
        self.is_align_beta_static = self.align_beta.func_name == "static"
        self.score_alpha = CreateFunc(kwargs.get("score_alpha", "log_linear"), *kwargs.get("score_alpha_args"))
        self.is_score_alpha_static = self.score_alpha.func_name == "static"
        self.score_beta = CreateFunc(kwargs.get("score_beta", "log_linear"), *kwargs.get("score_beta_args"))
        self.is_score_beta_static = self.score_beta.func_name == "static"

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, stride=None):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes, stride=stride) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric_for_align, overlaps, align_metric_for_score = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric_for_align, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric_for_score, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        gt_areas = (gt_bboxes[..., 2]-gt_bboxes[..., 0])*(gt_bboxes[..., 3]-gt_bboxes[..., 1])
        gt_areas_safe = torch.clamp(gt_areas, min=1.0)
        dynamic_score_alpha = self.score_alpha(gt_areas_safe) \
            if self.is_score_alpha_static else self.score_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_score_beta = self.score_beta(gt_areas_safe) \
            if self.is_score_beta_static else self.score_beta(gt_areas_safe).unsqueeze(-1)
        dynamic_align_alpha = self.align_alpha(gt_areas_safe) \
            if self.is_align_alpha_static else self.align_alpha(gt_areas_safe).unsqueeze(-1)
        dynamic_align_beta = self.align_beta(gt_areas_safe) \
            if self.is_align_beta_static else self.align_beta(gt_areas_safe).unsqueeze(-1)

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)

        align_metric_for_score = bbox_scores.pow(dynamic_score_alpha) * overlaps.pow(dynamic_score_beta)
        align_metric_for_align = bbox_scores.pow(dynamic_align_alpha) * overlaps.pow(dynamic_align_beta)
        return align_metric_for_align, overlaps, align_metric_for_score

class TaskAlignedAssigner_VaryingIoU(TaskAlignedAssigner):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.eps = eps
        self.iou_type = kwargs.get("iou_type", "CIoU")
        self.iou_kwargs = kwargs.get("iou_kwargs", {})

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        return bbox_iou_ext(gt_bboxes, pd_bboxes, xywh=False,
                            iou_type=self.iou_type, iou_kargs=self.iou_kwargs).squeeze(-1).clamp_(0)

class TaskAlignedAssigner_VaryingIoU_Sep(TaskAlignedAssigner):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.alpha = alpha if alpha else 1.0
        self.beta = beta if beta else 4.0
        self.eps = eps
        self.align_iou_type = kwargs.get("align_iou_type", "Hausdorff")
        self.align_iou_kwargs = kwargs.get("align_iou_kwargs", {})
        self.score_iou_type = kwargs.get("score_iou_type", "CIoU")
        self.score_iou_kwargs = kwargs.get("score_iou_kwargs", {})

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes) # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps, score_metric = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, score_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps_for_align = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        overlaps_for_score = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps_for_align[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes, self.align_iou_type, self.align_iou_kwargs)
        overlaps_for_score[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes, self.score_iou_type, self.score_iou_kwargs)

        align_metric = bbox_scores.pow(self.alpha) * overlaps_for_align.pow(self.beta)
        score_metric = bbox_scores.pow(self.alpha) * overlaps_for_score.pow(self.beta)
        return align_metric, overlaps_for_score, score_metric

    def iou_calculation(self, gt_bboxes, pd_bboxes, iou_type=None, iou_kwargs=None):
        return bbox_iou_ext(pd_bboxes, gt_bboxes, xywh=False,
                            iou_type=iou_type, iou_kargs=iou_kwargs).squeeze(-1).clamp_(0)

class TaskAlignedAssigner_VaryingIoU_Sep_Dynamic(TaskAlignedAssigner_VaryingIoU_Sep):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha = None, beta = None, eps: float = 1e-9, **kwargs):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps, **kwargs)
        self.align_iou_type = kwargs.get("align_iou_type", "Hausdorff")
        self.align_iou_kwargs = kwargs.get("align_iou_kwargs", {})
        self.score_iou_type = kwargs.get("score_iou_type", "CIoU")
        self.score_iou_kwargs = kwargs.get("score_iou_kwargs", {})

        self.initial_align_iou_kwargs = self.align_iou_kwargs.copy()
        self.final_align_iou_kwargs = kwargs.get("final_align_iou_kwargs", {})

    def get_align_iou_kwargs(self, epoch, total_epochs):
        progress = epoch / max(total_epochs - 1, 1)
        return {k: self.initial_align_iou_kwargs[k] + (v - self.initial_align_iou_kwargs[k]) * progress for k, v in self.final_align_iou_kwargs.items()}

    def update_align_iou_kwargs(self, new_kwargs):
        """
        更新 align_iou_kwargs 参数
        
        Args:
            new_kwargs (dict): 新的 align_iou_kwargs 字典，可以是部分更新
        """
        print(f"last align_iou_kwargs: {self.align_iou_kwargs}")
        if isinstance(new_kwargs, dict):
            self.align_iou_kwargs.update(new_kwargs)
        else:
            self.align_iou_kwargs = new_kwargs
        print(f"new align_iou_kwargs: {self.align_iou_kwargs}")
        
    def update_align_iou_kwargs_by_epoch(self, epoch, total_epochs, update_func=None):
        """
        根据训练 epoch 动态更新 align_iou_kwargs
        
        Args:
            epoch (int): 当前 epoch (从0开始)
            total_epochs (int): 总 epoch 数
            update_func (callable, optional): 自定义更新函数，接收 (epoch, total_epochs) 并返回新的 kwargs
                如果为 None，则使用默认的线性更新策略
                
        示例:
            # 自定义更新函数示例：线性更新 alpha 参数
            def custom_update(epoch, total_epochs):
                progress = epoch / max(total_epochs - 1, 1)
                return {'alpha': 0.5 + 0.5 * progress}
            
            assigner.update_align_iou_kwargs_by_epoch(epoch, total_epochs, custom_update)
        """
        if update_func is not None:
            new_kwargs = update_func(epoch, total_epochs)
        else:
            new_kwargs = self.get_align_iou_kwargs(epoch, total_epochs)
        self.update_align_iou_kwargs(new_kwargs)
