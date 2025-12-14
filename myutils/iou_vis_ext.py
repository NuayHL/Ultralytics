import math
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from ultralytics.utils.metrics import bbox_iou_ext


def calculate_metrics(pred_bboxes: torch.Tensor, gt_bbox: torch.Tensor) -> Dict[str, torch.Tensor]:
    """计算多种相似度指标，并对异常值做裁剪以保证绘图稳定。"""
    with torch.no_grad():
        metrics = {
            "IoU": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="IoU", xywh=True),
            "GIoU": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="GIoU", xywh=True),
            "DIoU": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="DIoU", xywh=True),
            "CIoU": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="CIoU", xywh=True),
            "SIoU": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="SIoU", xywh=True),
            "PIoU": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="PIoU", xywh=True),
            "AlphaIoU": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="AlphaIoU", xywh=True, iou_kargs={"alpha": 0.3}),
            "L1": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="l1", xywh=True, iou_kargs={"lambda1": 0.8}),
            # "NWD": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="NWD", xywh=True, iou_kargs={"nwd_c": 12}),
            "NWD": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="Hausdorff_Ext_L2", 
                                xywh=True, iou_kargs={"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 7}),
            # "SimD": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="SimD", xywh=True),
            "SimD": bbox_iou_ext(pred_bboxes, gt_bbox, iou_type="Hausdorff_Ext_L2_fix",
                                xywh=True, iou_kargs={"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 12}),
            "Hausdorff": bbox_iou_ext(
                pred_bboxes,
                gt_bbox,
                iou_type="Hausdorff",
                xywh=True,
                iou_kargs={"lambda1": 2.5, "lambda2": 1.0},
            ),
        }

    # 防止个别指标出现 nan/inf 导致 contour 报错
    for k, v in metrics.items():
        metrics[k] = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)
    return metrics


def build_phase_space(
    grid_res: int = 100,
    angles_deg: Iterable[float] = (0.0, 45.0, 90.0, 135.0),
    dist_max: float = 1.5,
    shape_range: Tuple[float, float] = (0.2, 1.0),
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, Dict[float, Dict[str, torch.Tensor]]]:
    """
    生成相空间数据，并在多个偏移角度下计算指标。

    Args:
        grid_res: 网格分辨率。
        angles_deg: 需要测试的角度（度），0° 表示沿 x 轴正方向偏移。
        dist_max: 归一化距离最大值（相对 GT 对角线）。
        shape_range: 形态缩放比范围 (min, max)。
        device: 张量设备。
    """
    assert 10 <= grid_res <= 400, "grid_res 建议控制在 10~400 之间，避免内存过大或分辨率过低"

    norm_dists = np.linspace(0, dist_max, grid_res)
    shape_factors = np.linspace(shape_range[0], shape_range[1], grid_res)
    X, Y = np.meshgrid(norm_dists, shape_factors)

    # 基准 GT：正方形，使用 xywh
    gt_s = 100.0
    gt_bbox = torch.tensor([[0.0, 0.0, gt_s, gt_s]], device=device, dtype=torch.float32)
    gt_diag = math.sqrt(gt_s**2 + gt_s**2)

    results: Dict[float, Dict[str, torch.Tensor]] = {}

    for ang in angles_deg:
        rad = math.radians(ang)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        pred_list = []
        for d, s in zip(X.ravel(), Y.ravel()):
            if s <= 1:
                curr_w = gt_s * s
                curr_h = gt_s / s  # 长宽反比，模拟形态扰动
            else:
                curr_w = gt_s / (2 - s)
                curr_h = gt_s * (2 - s)

            offset_dist = d * gt_diag
            move_x = offset_dist * cos_a
            move_y = offset_dist * sin_a

            pred_list.append([move_x, move_y, curr_w, curr_h])

        pred_bboxes = torch.tensor(pred_list, device=device, dtype=torch.float32)
        gt_bboxes_exp = gt_bbox.expand(len(pred_list), -1)
        results[ang] = calculate_metrics(pred_bboxes, gt_bboxes_exp)

    return X, Y, results


def _auto_levels(Z: np.ndarray, num: int = 21) -> np.ndarray:
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))
    if math.isclose(zmin, zmax):
        zmin -= 1e-3
        zmax += 1e-3
    zmin, zmax = max(-1.0, zmin), min(1.0, zmax)
    return np.linspace(zmin, zmax, num)


def plot_phase_space_sensitivity(
    grid_res: int = 120,
    angles_deg: Iterable[float] = (0.0, 15.0, 30.0, 45.0, ),
    metrics_to_plot: Iterable[str] = ("IoU", "GIoU", "DIoU", "NWD", "SimD", "Hausdorff", "CIoU", "SIoU", "PIoU", "L1", "AlphaIoU"),
    dist_max: float = 1.5,
    shape_range: Tuple[float, float] = (0.2, 1.8),
    save_prefix: str = "similarity_phase_space",
    show: bool = False,
) -> None:
    """
    绘制相空间敏感度图，支持多角度偏移对比。

    Args:
        grid_res: 网格分辨率。
        angles_deg: 偏移角度列表（度）。
        metrics_to_plot: 需要输出的指标名称。
        dist_max: 归一化距离最大值。
        shape_range: 形态缩放比范围。
        save_prefix: 保存文件名前缀。
        show: 是否调用 plt.show()。
    """
    X, Y, all_results = build_phase_space(
        grid_res=grid_res, angles_deg=angles_deg, dist_max=dist_max, shape_range=shape_range, device='cuda'
    )

    for metric in metrics_to_plot:
        fig, axes = plt.subplots(1, len(angles_deg), figsize=(5 * len(angles_deg), 4), sharex=True, sharey=True)
        if len(angles_deg) == 1:
            axes = [axes]  # 兼容单图

        cf_last = None
        for ax, ang in zip(axes, angles_deg):
            val = all_results[ang][metric].view(grid_res, grid_res).cpu().numpy()
            levels = _auto_levels(val)

            cf = ax.contourf(X, Y, val, levels=levels, cmap="viridis", alpha=0.85)
            cs = ax.contour(X, Y, val, levels=[0.0, 0.5, 0.75, 0.9], colors="k", linewidths=1.0)
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.2f")
            ax.set_title(f"{metric} @ {ang:.0f}°")
            ax.set_xlabel(r"normalized distance $d / \mathrm{diag}_{gt}$")
            ax.set_ylabel(r"shape scaling ratio $S_{pred}/S_{gt}$")
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_xlim(0, dist_max)
            ax.set_ylim(shape_range[0], shape_range[1])
            cf_last = cf

        if cf_last is not None:
            cbar = fig.colorbar(cf_last, ax=axes, shrink=0.92, pad=0.02)
            cbar.set_label(metric)

        fig.suptitle(f"Similarity Phase Space ({metric})", fontsize=14)
        # fig.tight_layout()
        fig.savefig(f"{save_prefix}_{metric.lower()}.png", dpi=300, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


if __name__ == "__main__":
    plot_phase_space_sensitivity(show=False)