from ultralytics.utils.metrics import bbox_iou_ext
import numpy as np
import matplotlib.pyplot as plt
import torch




def plot_iou_contour(iou_list: list, contour_levels: list = None,
                     gt_center: tuple = (0, 0), bbox_size: tuple = (60, 60),
                     grid_range: float = 150, grid_resolution: int = 200,
                     filled: bool = True, show_gt: bool = True):
    """
    绘制IoU度量的等高线图，展示预测框中心点在不同位置时的IoU值分布。
    
    这个可视化可以帮助理解不同IoU度量在各个方向上的"距离"特性：
    - 圆形等高线表示各向同性（所有方向的惩罚相同）
    - 椭圆形等高线表示各向异性（某些方向的惩罚更大）
    
    Args:
        iou_list: 列表，每个元素为 [name, iou_type, iou_kwargs]
                  例如: [["GIoU", "GIoU", {}], ["DIoU", "DIoU", {}]]
        contour_levels: 要绘制的等高线值列表，如 [0.1, 0.3, 0.5, 0.7, 0.9]
                        如果为None，则自动选择
        gt_center: GT框的中心点坐标 (x, y)
        bbox_size: bbox的尺寸 (w, h)，假设GT和预测框尺寸相同
        grid_range: 网格范围，从 gt_center 向外延伸的距离
        grid_resolution: 网格分辨率（每个轴的采样点数）
        filled: 是否绘制填充等高线图
        show_gt: 是否显示GT框
    """
    # 创建网格
    x = np.linspace(gt_center[0] - grid_range, gt_center[0] + grid_range, grid_resolution)
    y = np.linspace(gt_center[1] - grid_range, gt_center[1] + grid_range, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # 构建预测框tensor: (grid_resolution*grid_resolution, 4)
    pred_centers = np.stack([X.ravel(), Y.ravel()], axis=1)
    pred_bboxes = torch.tensor(np.concatenate([
        pred_centers,
        np.full((pred_centers.shape[0], 2), bbox_size)
    ], axis=1), dtype=torch.float32)
    
    # GT框
    gt_bbox = torch.tensor([gt_center[0], gt_center[1], bbox_size[0], bbox_size[1]], 
                           dtype=torch.float32).unsqueeze(0).expand(pred_bboxes.shape[0], -1)
    
    # 计算每种IoU度量的值
    n_metrics = len(iou_list)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows), squeeze=False)
    
    if contour_levels is None:
        contour_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    for idx, (name, iou_type, iou_kwargs) in enumerate(iou_list):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]
        
        # 计算IoU值
        iou_values = bbox_iou_ext(pred_bboxes, gt_bbox,
                                  iou_type=iou_type, iou_kargs=iou_kwargs,
                                  xywh=True).numpy()
        Z = iou_values.reshape(grid_resolution, grid_resolution)
        
        # 绘制等高线
        if filled:
            cf = ax.contourf(X, Y, Z, levels=20, cmap='RdYlGn')
            plt.colorbar(cf, ax=ax, label='Value')
        
        # 绘制等高线线条
        cs = ax.contour(X, Y, Z, levels=contour_levels, colors='black', linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt='%.2f')
        
        # 显示GT框
        if show_gt:
            gt_rect = plt.Rectangle(
                (gt_center[0] - bbox_size[0]/2, gt_center[1] - bbox_size[1]/2),
                bbox_size[0], bbox_size[1],
                fill=False, edgecolor='blue', linewidth=2, linestyle='--'
            )
            ax.add_patch(gt_rect)
            ax.plot(gt_center[0], gt_center[1], 'b+', markersize=10, markeredgewidth=2)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_title(f'{name}', fontsize=14)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('iou_contour.png')


def plot_iou_contour_overlay(iou_list: list, contour_value: float = 0.5,
                             gt_center: tuple = (0, 0), bbox_size: tuple = (60, 60),
                             grid_range: float = 150, grid_resolution: int = 300):
    """
    在同一张图上叠加绘制多种IoU度量的单一等高线，便于对比它们的形状差异。
    
    Args:
        iou_list: 列表，每个元素为 [name, iou_type, iou_kwargs]
        contour_value: 要绘制的等高线值（单一值）
        gt_center: GT框的中心点坐标 (x, y)
        bbox_size: bbox的尺寸 (w, h)
        grid_range: 网格范围
        grid_resolution: 网格分辨率
    """
    # 创建网格
    x = np.linspace(gt_center[0] - grid_range, gt_center[0] + grid_range, grid_resolution)
    y = np.linspace(gt_center[1] - grid_range, gt_center[1] + grid_range, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # 构建预测框tensor
    pred_centers = np.stack([X.ravel(), Y.ravel()], axis=1)
    pred_bboxes = torch.tensor(np.concatenate([
        pred_centers,
        np.full((pred_centers.shape[0], 2), bbox_size)
    ], axis=1), dtype=torch.float32)
    
    gt_bbox = torch.tensor([gt_center[0], gt_center[1], bbox_size[0], bbox_size[1]], 
                           dtype=torch.float32).unsqueeze(0).expand(pred_bboxes.shape[0], -1)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(iou_list)))
    
    for idx, (name, iou_type, iou_kwargs) in enumerate(iou_list):
        iou_values = bbox_iou_ext(gt_bbox, pred_bboxes,
                                  iou_type=iou_type, iou_kargs=iou_kwargs,
                                  xywh=True).numpy()
        Z = iou_values.reshape(grid_resolution, grid_resolution)
        
        # 绘制单一等高线
        cs = ax.contour(X, Y, Z, levels=[contour_value], colors=[colors[idx]], 
                        linewidths=2, linestyles='-')
        # 为图例添加标签
        cs.collections[0].set_label(f'{name} = {contour_value}')
    
    # 显示GT框
    gt_rect = plt.Rectangle(
        (gt_center[0] - bbox_size[0]/2, gt_center[1] - bbox_size[1]/2),
        bbox_size[0], bbox_size[1],
        fill=False, edgecolor='black', linewidth=2, linestyle='--', label='GT BBox'
    )
    ax.add_patch(gt_rect)
    ax.plot(gt_center[0], gt_center[1], 'k+', markersize=15, markeredgewidth=2)
    
    ax.set_xlabel('Pred Center X', fontsize=14)
    ax.set_ylabel('Pred Center Y', fontsize=14)
    ax.set_title(f'IoU Contour Comparison (value = {contour_value})', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_iou_contour_multi_values(iou_list: list, contour_values: list = None,
                                   gt_center: tuple = (0, 0), bbox_size: tuple = (60, 60),
                                   grid_range: float = 150, grid_resolution: int = 300):
    """
    在同一张图上绘制多种IoU度量的多条等高线，每种度量使用不同颜色。
    
    Args:
        iou_list: 列表，每个元素为 [name, iou_type, iou_kwargs]
        contour_values: 要绘制的等高线值列表，如 [0.3, 0.5, 0.7]
        gt_center: GT框的中心点坐标 (x, y)
        bbox_size: bbox的尺寸 (w, h)
        grid_range: 网格范围
        grid_resolution: 网格分辨率
    """
    if contour_values is None:
        contour_values = [0.3, 0.5, 0.7]
    
    # 创建网格
    x = np.linspace(gt_center[0] - grid_range, gt_center[0] + grid_range, grid_resolution)
    y = np.linspace(gt_center[1] - grid_range, gt_center[1] + grid_range, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    pred_centers = np.stack([X.ravel(), Y.ravel()], axis=1)
    pred_bboxes = torch.tensor(np.concatenate([
        pred_centers,
        np.full((pred_centers.shape[0], 2), bbox_size)
    ], axis=1), dtype=torch.float32)
    
    gt_bbox = torch.tensor([gt_center[0], gt_center[1], bbox_size[0], bbox_size[1]], 
                           dtype=torch.float32).unsqueeze(0).expand(pred_bboxes.shape[0], -1)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(iou_list)))
    linestyles = ['-', '--', '-.', ':']
    
    legend_handles = []
    
    for idx, (name, iou_type, iou_kwargs) in enumerate(iou_list):
        iou_values = bbox_iou_ext(gt_bbox, pred_bboxes,
                                  iou_type=iou_type, iou_kargs=iou_kwargs,
                                  xywh=True).numpy()
        Z = iou_values.reshape(grid_resolution, grid_resolution)
        
        # 为每个值绘制等高线
        for val_idx, val in enumerate(contour_values):
            ls = linestyles[val_idx % len(linestyles)]
            cs = ax.contour(X, Y, Z, levels=[val], colors=[colors[idx]], 
                            linewidths=1.5, linestyles=ls)
        
        # 创建图例项
        legend_handles.append(plt.Line2D([0], [0], color=colors[idx], linewidth=2, label=name))
    
    # 添加等高线值的图例
    for val_idx, val in enumerate(contour_values):
        ls = linestyles[val_idx % len(linestyles)]
        legend_handles.append(plt.Line2D([0], [0], color='gray', linewidth=1.5, 
                                         linestyle=ls, label=f'value={val}'))
    
    # 显示GT框
    gt_rect = plt.Rectangle(
        (gt_center[0] - bbox_size[0]/2, gt_center[1] - bbox_size[1]/2),
        bbox_size[0], bbox_size[1],
        fill=False, edgecolor='black', linewidth=2, linestyle='--'
    )
    ax.add_patch(gt_rect)
    ax.plot(gt_center[0], gt_center[1], 'k+', markersize=15, markeredgewidth=2)
    legend_handles.append(plt.Line2D([0], [0], color='black', linewidth=2, 
                                     linestyle='--', label='GT BBox'))
    
    ax.set_xlabel('Pred Center X', fontsize=14)
    ax.set_ylabel('Pred Center Y', fontsize=14)
    ax.set_title('IoU Contour Comparison (Multiple Values)', fontsize=16)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(handles=legend_handles, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.show()

STEP=100
# BBOX_SIZE1 = [5, 5]
# BBOX_SIZE2 = [5, 5]

BBOX_SIZE1 = [10, 10]
BBOX_SIZE2 = [10, 10]
GT_BBOX = torch.tensor([130, 130, BBOX_SIZE2[0], BBOX_SIZE2[1]]).expand(STEP+1, -1)
START_PRED_BBOX = torch.tensor([150, 150, BBOX_SIZE1[0], BBOX_SIZE1[1]]).expand(STEP+1, -1)
coe = torch.tensor([[float(i)/STEP] for i in range(STEP+1)])
INTERP_BBOX = GT_BBOX * coe + START_PRED_BBOX * (1 - coe)
X_RANGE = np.arange(STEP+1)

def plot_iou_curve(iou_list: list):
    iou_values = list()
    for _, iou_type, iou_kwargs in iou_list:
        iou_values.append(bbox_iou_ext(INTERP_BBOX, GT_BBOX,
                                          iou_type=iou_type, iou_kargs=iou_kwargs,
                                          xywh=True).numpy())

    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    for (name, _, _), iou_value in zip(iou_list, iou_values):
        ax.plot(X_RANGE, iou_value, label=name)

    ax.set_xlabel('Step', fontsize=14)
    ax.set_ylabel('Loss Value', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlim(0, STEP)
    ax.set_ylim(0, 2)
    plt.legend()
    plt.savefig('iou_curve.png')
    plt.show()


if __name__ == "__main__":
    # 原有的曲线图
    # plot_iou_curve([
    #                 # ["GIoU", "GIoU",{}],
    #                 # ["DIoU", "DIoU",{}],
    #                 # ["CIoU_with_alpha", "CIoU",{"alpha":0.5}],
    #                 # ["CIoU", "CIoU",{}],
    #                 # ["l1", "l1", {}],
    #                 ["l1", "l1", {"lambda1": 0.8}],
    #                 ["l1_ext", "l1_ext", {"lambda1": 7}],
    #                 # ["Hausdorff", "Hausdorff", {"lambda1": 2.5}],
    #                 # ["Hausdorff_Ext_IoU", "Hausdorff_Ext_IoU", {"lambda1": 2.5, "hybrid_pow": 5,}],
    #                 ["Hausdorff_Ext_L2", "Hausdorff_Ext_L2", {"lambda1": 2.5, "hybrid_pow": 5, "lambda3": 10}],
    #                 ["Hausdorff_Ext_L2_good", "Hausdorff_Ext_L2", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 7}],
    #                 ["Hausdorff_Ext_L2_fix", "Hausdorff_Ext_L2_fix", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 12}],
    #                 # ["Hausdorff_test", "Hausdorff_test", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 12}],
    #                 # ["Hausdorff1", "Hausdorff", {"lambda1": 5}],
    #                 # ["Hausdorff2", "Hausdorff", {"lambda1": 3.}],
    #                 # ["Hausdorff3", "Hausdorff", {"lambda1": 4.}],
    #                 # ["Hausdorff4", "Hausdorff", {"lambda1": 4.5}],
    #                 # ["NWD", "NWD", {"nwd_c": 12}],
    #                 # ["AlphaIoU", "AlphaIoU", {"alpha": 0.3}],
    #                 ["IoU", "IoU" ,{}],
    #                 # ["SimD1", "SimD",{"sim_x":6.13, "sim_y":4.59}],
    #                 ])
    LOSS_CONFIGS = [
        # ["CIoU", "CIoU", {}],
        ["Hausdorff in Gaussian Kernel", "Hausdorff", {"lambda1": 2.5}],
        ["L2 in Laplacian Kernel", "l1_ext", {"lambda1": 7.0}],
        ["HATS", "Hausdorff_Ext_L2", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 7}],
        ["HATS-fix", "Hausdorff_Ext_L2_fix", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 12}],
        ["HATS-rfix", "Hausdorff_Ext_L2_rfix", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 7}],
        # ["IoU", "IoU", {}],
    ]

    plot_iou_curve(LOSS_CONFIGS)

    # 示例：等高线子图（每种IoU一个子图）
    iou_metrics = [
        ["GIoU", "GIoU", {}],
        ["DIoU", "DIoU", {}],
        # ["CIoU_with_alpha", "CIoU",{"alpha":0.5}],
        ["CIoU", "CIoU", {}],
        # ["PIoU", "PIoU",{}],
        # ["InterpIoU", "InterpIoU",{"interp_coe": 0.98}],
        # ["D_InterpIoU", "D_InterpIoU", {"lv":0.9, "hv":0.98}],
        ["l1", "l1", {"lameda": 0.4}],
        ["Hausdorff_Ext_L2_good", "Hausdorff_Ext_L2", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 7}],
        ["Hausdorff1", "Hausdorff", {"lameda": 2.5}],
        # ["AlphaIoU", "AlphaIoU", {"alpha": 0.5}],
        ["IoU", "IoU", {}],
        ["SimD1", "SimD", {"sim_x": 6.13, "sim_y": 4.59}],
    ]
    # plot_iou_contour(iou_metrics, bbox_size=(60, 60), grid_range=120)
    #
    # 示例：叠加对比单一等高线值
    # plot_iou_contour_overlay(iou_metrics, contour_value=0.5, bbox_size=(60, 60))
    
    # 示例：叠加对比多个等高线值
    # plot_iou_contour_multi_values(iou_metrics, contour_values=[0.3, 0.5, 0.7], bbox_size=(60, 60))
