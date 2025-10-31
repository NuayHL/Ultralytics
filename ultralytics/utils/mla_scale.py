import os, pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import bbox_iou
from . import LOGGER, colorstr
from .tal import TaskAlignedAssigner
from .mla import TaskAlignedAssigner_Scale

_STEP = 0
_SAVE_DIR = "assign_record"
os.makedirs(_SAVE_DIR, exist_ok=True)

def save_assign_info(dir_name, **kwargs):
    save_path = os.path.join(dir_name, f"{_STEP}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(kwargs, f)


class TaskAlignedAssigner_dScale(TaskAlignedAssigner_Scale):
    def __init__(self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9,
                 scale_ratio=1.0, func_type='func_1'):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta, eps=eps, scale_ratio=scale_ratio)
        self.func_type = func_type

    def select_candidates_in_gts(self, xy_centers, gt_bboxes, eps=1e-9, stride=None):
        """
        Select positive anchor centers within ground truth bounding boxes.

        Args:
            xy_centers (torch.Tensor): Anchor center coordinates, shape (h*w, 2).
            gt_bboxes (torch.Tensor): Ground truth bounding boxes, shape (b, n_boxes, 4).
            eps (float, optional): Small value for numerical stability.
            stride (torch.Tensor): negative value of stride of each anchor (b, n_boxes, h*w)
        Returns:
            (torch.Tensor): Boolean mask of positive anchors, shape (b, n_boxes, h*w).

        Note:
            b: batch size, n_boxes: number of ground truth boxes, h: height, w: width.
            Bounding box format: [x_min, y_min, x_max, y_max].
        """
        n_anchors = xy_centers.shape[0]
        bs, n_boxes, _ = gt_bboxes.shape

        # unpack
        x_min = gt_bboxes[..., 0]
        y_min = gt_bboxes[..., 1]
        x_max = gt_bboxes[..., 2]
        y_max = gt_bboxes[..., 3]

        w = (x_max - x_min).clamp(min=eps)  # (bs, n_boxes)
        h = (y_max - y_min).clamp(min=eps)

        # compute ratio r = stride / w (and for h)
        # we need r shape = (bs, n_boxes, n_anchors)
        w_expand = w[..., None]  # (bs, n_boxes, 1)
        h_expand = h[..., None]

        # compute scale ratios, returns same shape as r_w/r_h
        scale_ratio_w = self.compute_scale_ratio_from_ratio(stride, -w_expand)
        scale_ratio_h = self.compute_scale_ratio_from_ratio(stride, -h_expand)

        # thresholds per-anchor
        thresh_w = stride * scale_ratio_w  # (bs, n_boxes, n_anchors)
        thresh_h = stride * scale_ratio_h

        # compute bbox deltas: left, top, right, bottom per anchor
        # xy_centers: (n_anchors, 2) -> (1,1,n_anchors,2) for broadcast
        lt = torch.stack((x_min, y_min), dim=-1)  # (bs, n_boxes, 2)
        rb = torch.stack((x_max, y_max), dim=-1)  # (bs, n_boxes, 2)
        # expand lt/rb to (bs, n_boxes, n_anchors, 2)
        lt_exp = lt[..., None, :].expand(bs, n_boxes, n_anchors, 2)
        rb_exp = rb[..., None, :].expand(bs, n_boxes, n_anchors, 2)
        centers = xy_centers[None, None, :, :].expand(bs, n_boxes, n_anchors, 2)

        # bbox_deltas: left=x - x_min, top=y - y_min, right=x_max - x, bottom=y_max - y
        left_top = centers - lt_exp  # (bs, n_boxes, n_anchors, 2)
        right_bottom = rb_exp - centers
        bbox_deltas = torch.cat([left_top, right_bottom], dim=-1)  # (..., 4) order: [l, t, r, b]

        # Now check per-direction thresholds:
        left_ok = bbox_deltas[..., 0] > thresh_w
        right_ok = bbox_deltas[..., 2] > thresh_w
        top_ok = bbox_deltas[..., 1] > thresh_h
        bot_ok = bbox_deltas[..., 3] > thresh_h

        mask_in = left_ok & right_ok & top_ok & bot_ok  # (bs, n_boxes, n_anchors) boolean
        return mask_in

    def compute_scale_ratio_from_ratio(self, stride, expand):
        """
        r: (bs, n_boxes, 1), stride / w 或 stride / h
        返回对应的 scale_ratio
        """
        if self.func_type == 'func_1':
            r = stride / (expand + self.eps)
            return self.func_1(r, self.scale_ratio)

        elif self.func_type == 'func_2':
            r = stride / (expand + self.eps)
            return self.func_2(r, self.scale_ratio)

        elif self.func_type == 'func_smooth_1':
            r = stride / (expand + self.eps)
            return self.func_smooth_1(r, self.scale_ratio)

        elif self.func_type == 'static':
            return self.scale_ratio

        else:
            raise NotImplementedError(f'{self.func_type} is not implemented')

    @staticmethod
    def func_1(r, r_max):
        """
        r: (bs, n_boxes, 1), stride / w 或 stride / h
        返回对应的 scale_ratio
        """
        s = torch.zeros_like(r)
        mask1 = (r >= 0.25) & (r < 1)
        s[mask1] = r_max * (r[mask1] - 0.25) / 0.75
        mask2 = (r >= 1) & (r < 2)
        s[mask2] = r_max
        mask3 = (r >= 2) & (r < 2.5)
        s[mask3] = r_max * (2.5 - r[mask3])
        return s

    @staticmethod
    def func_2(r, r_max):
        """
        r: (bs, n_boxes, 1), stride / w 或 stride / h
        返回对应的 scale_ratio
        """
        s = torch.zeros_like(r)
        mask1 = (r < 1)
        s[mask1] = r_max * r[mask1]
        mask2 = (r >= 1)
        s[mask2] = r_max
        return s

    @staticmethod
    def func_smooth_1(r, r_max, a=5.0, b=1.0):
        sigmoid = lambda x: 1 / (1 + torch.exp(-x))
        # 上升段 (控制上凸程度)
        rise = sigmoid(a * (r - 0.5))
        # 缓降段 (控制下降速度)
        fall = 1 - 0.25 * sigmoid(b * (r - 2.0))
        return r_max * rise * fall

