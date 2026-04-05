import math
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics.utils.metrics import bbox_iou_ext

# ================= 配置区域 =================
# 模拟小物体环境
GT_SIZE = 16.0  # 16x16 像素的小物体
STRIDE = 8.0    # 假设 stride 为 8

# 定义绘图范围
# X轴: 中心偏移量 (以 GT_SIZE 为单位)
# 我们关注 0 到 1.0 倍 GT 宽度的偏移 (对于小物体，偏一个身位很正常)
DIST_MAX_RATIO = 1.2 

# Y轴: 尺度缩放比例 (Scale Ratio = sqrt(Area_pred / Area_gt))
# 小物体容易被预测得过大或过小，我们看 0.4x 到 2.5x 的面积变化
SCALE_MIN = 0.4
SCALE_MAX = 2.5

GRID_RES = 100
# ===========================================

def get_metrics_map():
    # 1. 构建网格
    # x: 归一化中心距离 (dx / w_gt)
    x = np.linspace(0, DIST_MAX_RATIO, GRID_RES)
    # y: 尺度比例 (s / s_gt)
    y = np.linspace(SCALE_MIN, SCALE_MAX, GRID_RES)
    X, Y = np.meshgrid(x, y)
    
    # 2. 生成 Pred Bboxes
    # GT 固定在 (0,0, 16, 16)
    gt_bbox = torch.tensor([[0, 0, GT_SIZE, GT_SIZE]], dtype=torch.float32)
    
    pred_bboxes = []
    
    # 我们假设偏移是沿着对角线方向的 (最坏情况)，或者单纯沿X轴
    # 这里为了简单，沿 X 轴偏移即可，效果是一样的
    for dx_ratio, scale_ratio in zip(X.ravel(), Y.ravel()):
        # 计算偏移像素
        offset = dx_ratio * GT_SIZE
        
        # 计算预测框大小
        # 这里假设长宽比保持 1:1，只改变 Scale，因为你关注的是 Size 容忍度
        pred_w = GT_SIZE * scale_ratio
        pred_h = GT_SIZE * scale_ratio
        
        # 构造 pred [center_x, center_y, w, h]
        # GT center is (0,0) -> pred center is (offset, 0)
        # 格式为 xywh
        pred_bboxes.append([offset, 0.0, pred_w, pred_h])
        
    pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32)
    
    # 3. 计算指标
    # 扩展 GT 以匹配 pred 数量
    gt_bboxes_exp = gt_bbox.expand(len(pred_bboxes), -1)
    
    results = {}
    with torch.no_grad():
        # IoU (Baseline)
        results['IoU'] = bbox_iou_ext(pred_bboxes, gt_bboxes_exp, iou_type='IoU', xywh=True)
        
        # Your Metric (Simulate Hausdorff or your custom logic)
        # 这里用 Hausdorff 模拟你的方法，实际请替换为你的 metric 计算函数
        results['Hausdorff'] = bbox_iou_ext(
            pred_bboxes, gt_bboxes_exp, iou_type='Hausdorff', xywh=True,
            iou_kargs={"lambda1": 2.0, "lambda2": 0.5} # 模拟对距离宽容
        )
        
        # 也可以加入 SIoU 或 NWD 对比
        results['NWD'] = bbox_iou_ext(pred_bboxes, gt_bboxes_exp, iou_type='NWD', xywh=True, iou_kargs={"nwd_c": 12})
        results['SimD'] = bbox_iou_ext(pred_bboxes, gt_bboxes_exp, iou_type='SimD', xywh=True, iou_kargs={"sim_x": 6.13, "sim_y": 4.59})
        results['L1'] = bbox_iou_ext(pred_bboxes, gt_bboxes_exp, iou_type='l1', xywh=True, iou_kargs={"lambda1": 0.8})
        results['HPP'] = bbox_iou_ext(pred_bboxes, gt_bboxes_exp, iou_type='hausdorff_plateau_peak', xywh=True,
                                      iou_kargs={"lambda_h": 2.5, "lambda_c": 5.0, "tau": 4})

    # Reshape
    for k in results:
        results[k] = results[k].view(GRID_RES, GRID_RES).numpy()
        
    return X, Y, results

def plot_tolerance_landscape():
    X, Y, metrics = get_metrics_map()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # -------------------------------------------------------
    # 1. 绘制背景热力图 (使用你的 Metric)
    # -------------------------------------------------------
    # 我们希望背景展示你的 Metric 的平缓下降趋势
    our_metric = metrics['Hausdorff'] # 替换为你的 Metric key
    
    # 使用较浅的色调，避免干扰前景的等高线
    cf = ax.contourf(X, Y, our_metric, levels=30, cmap='GnBu', alpha=0.6, vmin=0, vmax=1)
    cbar = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('Your Metric Value (Smooth Gradient)', rotation=270, labelpad=15)

    # -------------------------------------------------------
    # 2. 绘制 IoU 的等高线 (Baseline) - 红色虚线
    # -------------------------------------------------------
    # IoU 通常下降非常快，我们画出 0.5 (匹配阈值) 和 0.1 (几乎无重叠)
    iou_levels = [0.1, 0.5]
    cs_iou = ax.contour(X, Y, metrics['IoU'], levels=iou_levels, 
                        colors='red', linestyles='--', linewidths=2)
    # 手动添加 Label
    ax.clabel(cs_iou, inline=True, fontsize=10, fmt='IoU=%.1f', colors='red')
    
    # -------------------------------------------------------
    # 3. 绘制 你的 Metric 的等高线 - 蓝色实线
    # -------------------------------------------------------
    # 你的 Metric 应该在同样的数值下，圈出更大的范围
    # 假设你的 Metric 归一化到了 0-1
    our_levels = [0.1, 0.5]
    cs_our = ax.contour(X, Y, our_metric, levels=our_levels, 
                        colors='blue', linestyles='-', linewidths=2.5)
    ax.clabel(cs_our, inline=True, fontsize=10, fmt='Ours=%.1f', colors='blue')

    # -------------------------------------------------------
    # 4. 关键区域标注 (The "Noise" Zone)
    # -------------------------------------------------------
    # 假设 Stride 带来的位置噪声约为 0.5 * GT_SIZE
    # 假设 尺度噪声在 0.8 - 1.2 之间
    rect = plt.Rectangle((0, 0.8), 0.5, 0.4, linewidth=2, edgecolor='green', facecolor='none', hatch='//', alpha=0.5)
    ax.add_patch(rect)
    ax.text(0.25, 1.25, "Intrinsic Noise Zone\n(Small Object)", color='green', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 装饰
    ax.set_title(f"Tolerance Landscape: Small Object (size={int(GT_SIZE)}px)", fontsize=14)
    ax.set_xlabel(r"Center Shift ($\Delta pixel / w_{gt}$)", fontsize=12)
    ax.set_ylabel(r"Scale Ratio ($Size_{pred} / Size_{gt}$)", fontsize=12)
    
    # 添加辅助线
    ax.axvline(0.5, color='gray', linestyle=':', alpha=0.5) # 半个身位偏移
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5) # 完美尺度

    # 限制显示范围
    ax.set_xlim(0, DIST_MAX_RATIO)
    ax.set_ylim(SCALE_MIN, SCALE_MAX)
    
    plt.tight_layout()
    plt.savefig('iou_illus/tolerance_landscape.png')
    plt.show() # savefig in real use

if __name__ == "__main__":
    plot_tolerance_landscape()
