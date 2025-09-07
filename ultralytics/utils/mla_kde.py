import os, pickle
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from .metrics import bbox_iou
from . import LOGGER, colorstr
from .tal import TaskAlignedAssigner

# 兼容性处理
try:
    from torch.cuda import OutOfMemoryError
except ImportError:
    OutOfMemoryError = RuntimeError

_STEP = 0
_SAVE_DIR = "assign_record"
os.makedirs(_SAVE_DIR, exist_ok=True)

from ultralytics.utils.checks import check_version
TORCH_1_10 = check_version(torch.__version__, "1.10.0")

def save_assign_info(dir_name, **kwargs):
    save_path = os.path.join(dir_name, f"{_STEP}.pkl")
    with open(save_path, 'wb') as f:
        pickle.dump(kwargs, f)

def make_anchors(feats_size, strides, torch_dtype=torch.float, torch_device='cuda', grid_cell_offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats_size is not None
    for i, stride in enumerate(strides):
        h, w = feats_size[i]
        sx = torch.arange(end=w, device=torch_device, dtype=torch_dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=torch_device, dtype=torch_dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=torch_dtype, device=torch_device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

def select_topk_candidates(metrics, topk=10, topk_mask=None, eps=1e-9):
    """
    Select the top-k candidates based on the given metrics.

    Args:
        metrics (torch.Tensor): A tensor of shape (b, max_num_obj, h*w), where b is the batch size, max_num_obj is
            the maximum number of objects, and h*w represents the total number of anchor points.
        topk (int, optional): The number of top candidates to consider. Default: 10.
        topk_mask (torch.Tensor, optional): An optional boolean tensor of shape (b, max_num_obj, topk), where
            topk is the number of top candidates to consider. If not provided, the top-k values are automatically
            computed based on the given metrics.
        eps (float, optional): Small value for numerical stability. Default: 1e-9.

    Returns:
        (torch.Tensor): A tensor of shape (b, max_num_obj, h*w) containing the selected top-k candidates.
    """
    # (b, max_num_obj, topk)
    topk_metrics, topk_idxs = torch.topk(metrics, topk, dim=-1, largest=True)
    if topk_mask is None:
        topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > eps).expand_as(topk_idxs)
    # (b, max_num_obj, topk)
    topk_idxs.masked_fill_(~topk_mask, 0)

    # (b, max_num_obj, topk, h*w) -> (b, max_num_obj, h*w)
    count_tensor = torch.zeros(metrics.shape, dtype=torch.int8, device=topk_idxs.device) # b, max_num_obj, num_anch
    ones = torch.ones_like(topk_idxs[:, :, :1], dtype=torch.int8, device=topk_idxs.device)
    for k in range(topk):
        # Expand topk_idxs for each value of k and add 1 at the specified positions
        count_tensor.scatter_add_(-1, topk_idxs[:, :, k : k + 1], ones)
    # Filter invalid bboxes (due to the unstable behavior of topk)
    count_tensor.masked_fill_(count_tensor > 1, 0)

    return count_tensor.to(metrics.dtype)

def select_dynamic_topk_candidates(metrics, min_topk=4, max_topk=10, metric_sum_thr=3, topk_mask=None, eps=1e-9):
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
    topk_metrics, topk_idxs = torch.topk(metrics, max_topk, dim=-1, largest=True)

    # 默认 mask
    if topk_mask is None:
        topk_mask = (topk_metrics.max(-1, keepdim=True)[0] > eps).expand_as(topk_idxs)

    # 计算累积和并找到动态 topk
    cumsum_metrics = torch.cumsum(topk_metrics, dim=-1)  # (b, max_num_obj, max_topk)
    over_thr = (cumsum_metrics > metric_sum_thr)

    # 第一次超过阈值的位置
    first_over_idx = over_thr.float().argmax(dim=-1)  # (b, max_num_obj)
    # 如果从未超过，则设为 max_topk
    no_exceed = over_thr.sum(dim=-1) == 0
    first_over_idx[no_exceed] = max_topk

    # clamp 到 [min_topk, max_topk]
    dynamic_topk = torch.clamp(first_over_idx, min=min_topk, max=max_topk)  # (b, max_num_obj)

    # 构造每个位置的选择 mask
    arange_k = torch.arange(max_topk, device=metrics.device).view(1, 1, -1)  # (1, 1, max_topk)
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

class SpatialKDE(nn.Module):
    def __init__(self,
                 feature_map_sizes: Tuple[Tuple[int, int], ...],
                 feature_stride: Tuple[int] = (8, 16, 32),
                 bandwidth_scale_factor: float = 0.15,
                 max_topk: int = 10,
                 min_topk: int = 4,
                 metric_sum_thr = 3
                 ):
        """
        Args:
            anc_points (torch.Tensor): The (x, y) coordinates of all anchor points.
                Shape: (num_anchors, 2).
            anc_stride: (torch.Tensor): The stride of each anchor point. shape: (num_anchors, 1).
            feature_map_sizes (Tuple[Tuple[int, int], ...]): The (H, W) of each feature
                map level, e.g., ((80, 80), (40, 40), (20, 20)).
            bandwidth_scale_factor (float): A hyperparameter to scale the GT box size
                to get the final bandwidth. bandwidth = scale_factor * sqrt(w*h).
        """
        super().__init__()
        self.feature_map_sizes = feature_map_sizes
        self.feature_stride = feature_stride
        self.num_anchors_per_level = [h * w for h, w in feature_map_sizes]
        self.bandwidth_scale_factor = bandwidth_scale_factor

        anc_points, strides = make_anchors(self.feature_map_sizes, self.feature_stride)
        self._precompute_distances(anc_points * strides.view(-1, 1))

        # Pre-compute and register distance matrices to avoid re-computation
        self.max_topk = max_topk
        self.min_topk = min_topk
        self.metric_sum_thr = metric_sum_thr

    def forward(self,
                align_metric: torch.Tensor,
                gt_boxes: torch.Tensor,
                mask_gt: torch.Tensor,
                mask_in_gts: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the KDE-based assignment.

        Args:
            align_metric (torch.Tensor): The alignment values between predictions (anchors) and GTs.
                Shape: (batch_size, num_max_gts, num_anchors).
            gt_boxes (torch.Tensor): The ground truth boxes in (x1, y1, x2, y2) format.
                Shape: (batch_size, num_max_gts, 4).
            mask_gt (torch.Tensor): A boolean mask indicating which GT boxes are valid with shape (bs, n_max_boxes, 1).
            mask_in_gts (torch.Tensor): A boolean mask indicating which anchors are inside GT boxes with shape

        Returns:
            torch.Tensor: The estimated probability density at each anchor location.
                Shape: (batch_size, num_max_gts, num_anchors).
        """
        bs, n_max_gts, num_anchors = align_metric.shape
        torch_dtype, torch_device = align_metric.dtype, align_metric.device

        # all shape is (bs, n_max_gts, num_anchors)
        mask_topk = select_dynamic_topk_candidates(align_metric, min_topk=self.max_topk, max_topk=self.max_topk,
                                                   metric_sum_thr=self.metric_sum_thr,
                                                   topk_mask=mask_gt.expand(-1, -1, self.max_topk))

        output_prob_maps = torch.zeros_like(align_metric)

        for i in range(bs):
            # --- Slice all tensors for the current sample in the batch ---
            gt_boxes_i = gt_boxes[i]  # Shape: (n_max_gts, 4)
            align_metric_i = align_metric[i]  # Shape: (n_max_gts, num_anchors)
            mask_gt_i = mask_gt[i]  # Shape: (n_max_gts, 1)
            mask_in_gts_i = mask_in_gts[i]  # Shape: (n_max_gts, num_anchors)
            mask_topk_i = mask_topk[i]  # Shape: (n_max_gts, num_anchors)

            # --- Pre-calculate adaptive bandwidth for all valid GTs in this sample ---
            # Filter for valid GTs to avoid unnecessary computation
            valid_gt_mask_i = mask_gt_i.squeeze().bool()
            if not valid_gt_mask_i.any():
                continue  # No valid GTs in this sample, skip to the next

            gt_wh = gt_boxes_i[valid_gt_mask_i, 2:] - gt_boxes_i[valid_gt_mask_i, :2]
            # Add epsilon for numerical stability with zero-area boxes
            h = self.bandwidth_scale_factor * torch.sqrt(gt_wh[:, 0] * gt_wh[:, 1] + 1e-9)
            # Unsqueeze for broadcasting: (num_valid_gts, 1, 1)
            h_sq = (h ** 2).view(-1, 1, 1)

            prob_map_i = torch.zeros_like(align_metric_i)
            anchor_offset = 0

            # --- Loop over each FPN level to perform KDE calculation ---
            for level_idx, num_anchors_level in enumerate(self.num_anchors_per_level):
                # Define the anchor slice for the current level
                start_idx, end_idx = anchor_offset, anchor_offset + num_anchors_level

                # --- Slice masks and metrics for the current level and valid GTs ---
                mask_topk_level = mask_topk_i[valid_gt_mask_i, start_idx:end_idx]
                mask_in_gts_level = mask_in_gts_i[valid_gt_mask_i, start_idx:end_idx]
                align_metric_level = align_metric_i[valid_gt_mask_i, start_idx:end_idx]

                # --- Get pre-computed squared distances for this level ---
                # Shape: (num_anchors_level, num_anchors_level)
                dist_sq_level = getattr(self, f'dist_sq_level_{level_idx}').to(torch_device).to(torch_dtype)

                # --- Calculate normalized KDE weights ---
                topk_align_vals = align_metric_level * mask_topk_level

                # 检查当前层是否有有效的anchor
                if topk_align_vals.shape[0] == 0:
                    continue
                    
                # --- Perform the core KDE computation using broadcasting ---
                # kernel_matrix shape: (num_valid_gts, num_anchors_level, num_anchors_level)
                kernel_matrix = torch.exp(-dist_sq_level.unsqueeze(0) / (2 * h_sq))

                # Weighted sum via batch matrix multiplication
                # (G, 1, N_lvl) @ (G, N_lvl, N_lvl) -> (G, 1, N_lvl) -> (G, N_lvl)
                # where G is num_valid_gts and N_lvl is num_anchors_level
                prob_map_level_unmasked = torch.einsum('bk,bkn->bn', topk_align_vals.to(kernel_matrix.dtype), kernel_matrix)

                # Apply the in-GT mask to the final probabilities
                prob_map_level = prob_map_level_unmasked * mask_in_gts_level
                # --- Place the computed probabilities back into the full map for this sample ---
                prob_map_i[valid_gt_mask_i, start_idx:end_idx] = prob_map_level.to(align_metric.dtype)

                # Update the offset for the next level
                anchor_offset = end_idx

            # --- Store the result for the current sample ---
            prob_map_i = prob_map_i / (prob_map_i.max(dim=-1, keepdim=True)[0] + 1e-9)
            output_prob_maps[i] = prob_map_i
        return output_prob_maps

    def _precompute_distances(self, anc_points: torch.Tensor):
        """
        Computes pairwise squared distances for anchors at each FPN level.
        The results are stored as registered buffers.
        """
        anc_points_levels = torch.split(anc_points, self.num_anchors_per_level, dim=0)

        for i, points_level in enumerate(anc_points_levels):
            # Calculate pairwise squared distances
            diff = points_level.unsqueeze(1) - points_level.unsqueeze(0)
            dist_sq = torch.sum(diff ** 2, dim=-1)
            # Register as a buffer, not a parameter
            self.register_buffer(f'dist_sq_level_{i}', dist_sq)


class SpatialKDEConv(nn.Module):
    def __init__(self,
                 bandwidth_scale_factor: float = 0.01,
                 max_topk: int = 10,
                 min_topk: int = 4,
                 metric_sum_thr = 3,
                 kernel_size: int = 7
                 ):
        """
        基于高斯卷积核的SpatialKDE实现，避免计算大量pairwise距离
        
        Args:
            feature_map_sizes: 每个FPN层的特征图尺寸，如((80,80),(40,40),(20,20))
            feature_stride: 每个FPN层的步长，如(8, 16, 32)
            bandwidth_scale_factor: GT box尺寸到bandwidth的缩放因子
            max_topk: 最大top-k候选数
            min_topk: 最小top-k候选数
            metric_sum_thr: 累积和阈值
            kernel_size: 卷积核大小，必须是奇数
        """
        super().__init__()
        self.bandwidth_scale_factor = bandwidth_scale_factor
        self.max_topk = max_topk
        self.min_topk = min_topk
        self.metric_sum_thr = metric_sum_thr
        
        # 确保kernel_size是奇数
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        # 预计算不同bandwidth的高斯卷积核
        self._precompute_gaussian_kernels()

    def _precompute_gaussian_kernels(self):
        """预计算不同bandwidth的高斯卷积核"""
        # 创建坐标网格
        coords = torch.arange(self.kernel_size, dtype=torch.float32)
        coords = coords - self.padding  # 中心为0

        # 创建2D坐标网格
        y_coords, x_coords = torch.meshgrid(coords, coords, indexing='ij')

        # 计算到中心的距离平方
        dist_sq = x_coords ** 2 + y_coords ** 2

        # 预计算几个标准bandwidth的核
        # 这里我们预计算几个常用的bandwidth值，实际使用时进行插值
        self.register_buffer('dist_sq', dist_sq)

        # 预计算一些标准bandwidth值

        # 预计算对应的核
        kernel = torch.exp(-dist_sq / 2)
        kernel = kernel / kernel.sum()  # 归一化

        self.register_buffer('precomputed_kernels', kernel)

    def _get_gaussian_kernel(self, bandwidth: torch.Tensor) -> torch.Tensor:
        """
        根据bandwidth获取对应的高斯卷积核
        
        Args:
            bandwidth: 形状为(num_valid_gts,)的bandwidth值
            
        Returns:
            形状为(num_valid_gts, kernel_size, kernel_size)的卷积核
        """
        bw_expanded = bandwidth.unsqueeze(-1).unsqueeze(-1) # (num_gts, 1, 1, 1)
        self.precomputed_kernels = self.precomputed_kernels.to(bandwidth.device)
        weighted_kernels =  torch.pow(self.precomputed_kernels, torch.pow(bw_expanded, -1))  # (num_gts, num_std, kernel_size, kernel_size)
        return weighted_kernels

    def forward(self,
                align_metric: torch.Tensor,
                gt_boxes: torch.Tensor,
                mask_gt: torch.Tensor,
                mask_in_gts: torch.Tensor,
                feature_map_sizes: List[List[int]]) -> torch.Tensor:
        """
        使用高斯卷积核进行前向传播
        Args:
            align_metric: 形状为(batch_size, num_max_gts, num_anchors)的对齐度量
            gt_boxes: 形状为(batch_size, num_max_gts, 4)的GT框
            mask_gt: 形状为(batch_size, num_max_gts, 1)的GT有效掩码
            mask_in_gts: 形状为(batch_size, num_max_gts, num_anchors)的anchor在GT内的掩码
            feature_map_sizes:
        Returns:
            形状为(batch_size, num_max_gts, num_anchors)的概率密度图
        """
        bs, n_max_gts, num_anchors = align_metric.shape
        device = align_metric.device
        
        # 获取top-k候选
        mask_topk = select_dynamic_topk_candidates(
            align_metric, 
            min_topk=self.min_topk, 
            max_topk=self.max_topk,
            metric_sum_thr=self.metric_sum_thr,
            topk_mask=mask_gt.expand(-1, -1, self.max_topk)
        )
        
        output_prob_maps = torch.zeros_like(align_metric)
        
        for i in range(bs):
            # 获取当前样本的数据
            gt_boxes_i = gt_boxes[i]  # (n_max_gts, 4)
            align_metric_i = align_metric[i]  # (n_max_gts, num_anchors)
            mask_gt_i = mask_gt[i]  # (n_max_gts, 1)
            mask_in_gts_i = mask_in_gts[i]  # (n_max_gts, num_anchors)
            mask_topk_i = mask_topk[i]  # (n_max_gts, num_anchors)
            
            # 获取有效GT
            valid_gt_mask_i = mask_gt_i.squeeze().bool()
            if not valid_gt_mask_i.any():
                continue
                
            # 计算bandwidth
            gt_wh = gt_boxes_i[valid_gt_mask_i, 2:] - gt_boxes_i[valid_gt_mask_i, :2]
            h = self.bandwidth_scale_factor * torch.sqrt(gt_wh[:, 0] * gt_wh[:, 1] + 1e-9)
            
            # 获取对应的高斯卷积核
            kernels = self._get_gaussian_kernel(h)  # (num_valid_gts, kernel_size, kernel_size)
            
            # 对每个FPN层分别处理
            anchor_offset = 0
            prob_map_i = torch.zeros_like(align_metric_i)
            
            for level_idx, (h_level, w_level) in enumerate(feature_map_sizes):
                start_idx, end_idx = anchor_offset, anchor_offset + h_level * w_level
                
                # 获取当前层的数据
                mask_topk_level = mask_topk_i[valid_gt_mask_i, start_idx:end_idx]
                mask_in_gts_level = mask_in_gts_i[valid_gt_mask_i, start_idx:end_idx]
                align_metric_level = align_metric_i[valid_gt_mask_i, start_idx:end_idx]
                
                # 重塑为2D特征图格式 (num_valid_gts, h, w)
                align_metric_2d = align_metric_level.view(-1, h_level, w_level)
                mask_topk_2d = mask_topk_level.view(-1, h_level, w_level)
                mask_in_gts_2d = mask_in_gts_level.view(-1, h_level, w_level)

                # 获取有效GT的索引
                valid_gt_indices = torch.where(valid_gt_mask_i)[0]
                # 如果没有有效GT，跳过这一层
                if len(valid_gt_indices) == 0:
                    continue

                # 应用top-k掩码
                topk_align_vals = align_metric_2d * mask_topk_2d

                if topk_align_vals.shape[0] > 0:
                    # (1, num_valid_gts, h, w)
                    inputs = topk_align_vals.unsqueeze(0)

                    # (num_valid_gts, 1, k, k)
                    weight = kernels.unsqueeze(1).to(inputs.dtype)

                    # 分组卷积，一次性完成所有GT的卷积
                    conv_results = F.conv2d(
                        inputs,
                        weight,
                        stride=1,
                        padding=self.padding,
                        groups=topk_align_vals.shape[0]
                    )  # (num_valid_gts, 1, h, w)

                    # 去掉 channel 维度
                    prob_map_level = conv_results.squeeze(0)

                    # 避免全零输入的情况：保持原逻辑
                    zero_mask = (inputs.sum(dim=(2, 3)) == 0)  # (num_valid_gts, 1)
                    if zero_mask.any():
                        prob_map_level[zero_mask.squeeze()] = 0
                else:
                    prob_map_level = torch.zeros_like(topk_align_vals)
                
                # 应用GT内掩码
                prob_map_level = prob_map_level * mask_in_gts_2d
                
                # 重塑回1D并放回结果
                prob_map_level_1d = prob_map_level.view(-1, h_level * w_level).to(prob_map_i.dtype)
                prob_map_i[valid_gt_mask_i, start_idx:end_idx] = prob_map_level_1d
                
                anchor_offset = end_idx
            
            # 归一化，避免除零和NaN
            max_vals = prob_map_i.max(dim=-1, keepdim=True)[0]
            # 只对非零值进行归一化
            prob_map_i = torch.where(
                max_vals > 0,
                prob_map_i / (max_vals + 1e-9),
                prob_map_i
            )
            output_prob_maps[i] = prob_map_i
            
        return output_prob_maps


class TaskAlignedAssigner_kde_dynamicK(TaskAlignedAssigner):
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

        self.kde_max_topk = kwargs['kde_max_topk'] if 'kde_max_topk' in kwargs else 10
        self.kde_min_topk = kwargs['kde_min_topk'] if 'kde_min_topk' in kwargs else 3
        self.kde_metric_sum_thr = kwargs['kde_metric_sum_thr'] if 'kde_metric_sum_thr' in kwargs else 4.0
        self.bandwidth_scale_factor = kwargs['bandwidth_scale_factor'] if 'bandwidth_scale_factor' in kwargs else 0.15

        self.kde_weight = SpatialKDEConv(bandwidth_scale_factor=self.bandwidth_scale_factor,
                                         max_topk=self.kde_max_topk, min_topk=self.kde_min_topk,
                                         metric_sum_thr=self.kde_metric_sum_thr,
                                         kernel_size=kwargs.get('kernel_size', 7))

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
            return self._forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt,
                                 kwargs['feature_map_size'])
        except OutOfMemoryError:
            raise Exception('Label assignment out of memory. Try reducing the batch size or reducing the number of '
                            'ground truth boxes.')


    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, feature_map_size):
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
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, feature_map_size
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


    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, feature_map_size):
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
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)  # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                                                      mask_in_gts * mask_gt, mask_gt, mask_in_gts,
                                                      feature_map_size)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.max_topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps


    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                        mask_gt_and_in_gt, mask_gt, mask_in_gts, feature_map_size):
        """
        Compute alignment metric given predicted and ground truth bounding boxes.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt_and_in_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, h*w).

        Returns:
            align_metric (torch.Tensor): Alignment metric combining classification and localization.
            overlaps (torch.Tensor): IoU overlaps between predicted and ground truth boxes.
        """
        na = pd_bboxes.shape[-2]
        mask_gt_and_in_gt = mask_gt_and_in_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls (only for gt's cls)
        bbox_scores[mask_gt_and_in_gt] = pd_scores[ind[0], :, ind[1]][mask_gt_and_in_gt]  # b, max_num_obj, h*w

        bbox_scores_weight = self.kde_weight(bbox_scores, gt_bboxes, mask_gt, mask_in_gts, feature_map_size)
        bbox_scores = bbox_scores * bbox_scores_weight

        # (b, max_num_obj, 1, 4), (b, 1, h*w, 4)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt_and_in_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt_and_in_gt]
        overlaps[mask_gt_and_in_gt] = self.iou_calculation(gt_boxes, pd_boxes)
        overlaps_weight = self.kde_weight(overlaps, gt_bboxes, mask_gt, mask_in_gts, feature_map_size)
        overlaps = overlaps * overlaps_weight

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

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
        return select_dynamic_topk_candidates(metrics, min_topk=self.max_topk, max_topk=self.max_topk,
                                       metric_sum_thr=self.metric_sum_thr,
                                       topk_mask=topk_mask)


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

class TaskAlignedAssigner_kde(TaskAlignedAssigner_kde_dynamicK):
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
        self.kde_max_topk = kwargs['kde_max_topk'] if 'kde_max_topk' in kwargs else 10
        self.kde_min_topk = kwargs['kde_min_topk'] if 'kde_min_topk' in kwargs else 3
        self.kde_metric_sum_thr = kwargs['kde_metric_sum_thr'] if 'kde_metric_sum_thr' in kwargs else 4.0
        self.bandwidth_scale_factor = kwargs['bandwidth_scale_factor'] if 'bandwidth_scale_factor' in kwargs else 0.15

        self.kde_weight = SpatialKDEConv(bandwidth_scale_factor=self.bandwidth_scale_factor,
                                         max_topk=self.kde_max_topk, min_topk=self.kde_min_topk,
                                         metric_sum_thr=self.kde_metric_sum_thr,
                                         kernel_size=kwargs.get('kernel_size', 7))

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt, feature_map_size):
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
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)  # (b, max_num_obj, num_anchor=h*w)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes,
                                                      mask_in_gts * mask_gt, mask_gt, mask_in_gts,
                                                      feature_map_size)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        return select_topk_candidates(metrics, topk=self.topk, topk_mask=topk_mask,)

# 使用示例：
# 
# # 使用卷积版本（推荐，更快）
# assigner = TaskAlignedAssigner_kde_dynamicK(
#     max_topk=10,
#     min_topk=4,
#     metric_sum_thr=3.0,
#     feature_map_sizes=((80,80),(40,40),(20,20)),
#     feature_stride=(8, 16, 32),
#     bandwidth_scale_factor=0.15,
#     use_conv=True,  # 使用卷积版本
#     kernel_size=7   # 卷积核大小
# )
# 
# # 使用原始KDE版本（更准确但更慢）
# assigner = TaskAlignedAssigner_kde_dynamicK(
#     max_topk=10,
#     min_topk=4,
#     metric_sum_thr=3.0,
#     feature_map_sizes=((80,80),(40,40),(20,20)),
#     feature_stride=(8, 16, 32),
#     bandwidth_scale_factor=0.15,
#     use_conv=False  # 使用原始KDE版本
# )
# 
# 主要改进：
# 1. 使用预计算的高斯卷积核替代pairwise距离计算
# 2. 支持批量卷积操作，显著提升计算效率
# 3. 保持与原始KDE相同的接口和功能
# 4. 通过分组卷积处理不同GT的不同bandwidth
# 5. 内存占用更少，避免OOM问题
