import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
# 假设你的环境中可以使用 ultralytics 的这个函数
from ultralytics.utils.metrics import bbox_iou_ext

# ===========================
# 1. 配置部分 (Config)
# ===========================

# 更加柔和、不突兀的科研色系 (参考 Matplotlib tab10 的柔和版或 Seaborn deep)
COLORS = [
    '#4E79A7',  # 蓝色
    '#F28E2B',  # 橙色
    '#59A14F',  # 绿色
    '#E15759',  # 红色
    '#76B7B2',  # 青色
    '#EDC948',  # 黄色 (暗)
    '#B07AA1',  # 紫色
]




# ===========================
# 2. 核心计算逻辑
# ===========================

def get_loss_values(loss_list, fixed_bbox, moving_bbox_sequence):
    """
    计算所有 Loss 在给定移动序列下的值
    """
    results = {}

    # 将 list 转为 tensor 方便批处理
    # moving_bbox_sequence shape: [Steps, 4]
    # fixed_bbox shape: [1, 4] -> 扩展到 [Steps, 4]
    steps = moving_bbox_sequence.shape[0]
    gt_bboxes = fixed_bbox.expand(steps, -1)

    for name, iou_type, kwargs in loss_list:
        # 调用 ultralytics 的接口
        # 注意：这里我们假设 bbox_iou_ext 返回的是 tensor
        value_tensor = bbox_iou_ext(moving_bbox_sequence, gt_bboxes, iou_type=iou_type, iou_kargs=kwargs, xywh=True)

        vals = value_tensor.numpy()

        # 展平数组
        if len(vals.shape) > 1:
            vals = vals.squeeze()

        results[name] = vals

    return results


def draw_schematic(ax, scenario='horizontal'):
    """绘制左侧的示意图"""
    ax.set_aspect('equal')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # 样式定义
    color_gt = '#D3D3D3'  # 浅灰 GT
    color_pred = '#4E79A7'  # 预测框主色

    if scenario == 'horizontal':
        # 场景1：水平移动 (Horizontal Deviation)
        # GT Box
        rect_gt = patches.Rectangle((30, 30), 40, 40, linewidth=2, edgecolor='gray', facecolor=color_gt, alpha=0.5)
        ax.add_patch(rect_gt)
        ax.text(32, 32, 'GT', fontsize=12, fontweight='bold', color='gray')

        # Pred Box (Center)
        rect_center = patches.Rectangle((30, 30), 40, 40, linewidth=2, edgecolor=color_pred, facecolor='none',
                                        linestyle=':')
        ax.add_patch(rect_center)

        # Pred Box (Shifted Right)
        rect_right = patches.Rectangle((50, 30), 40, 40, linewidth=2, edgecolor=color_pred, facecolor=color_pred,
                                       alpha=0.2)
        ax.add_patch(rect_right)

        # 箭头：表示移动方向
        ax.arrow(50, 50, 15, 0, head_width=3, head_length=4, fc=color_pred, ec=color_pred)
        ax.text(50, 75, 'Deviation', fontsize=11, color=color_pred, ha='center')

    elif scenario == 'diagonal':
        # 场景2：对角线移动 (Diagonal Deviation) - 模拟尺度或位置的综合变化
        # GT Box
        rect_gt = patches.Rectangle((30, 30), 40, 40, linewidth=2, edgecolor='gray', facecolor=color_gt, alpha=0.5)
        ax.add_patch(rect_gt)
        ax.text(32, 32, 'GT', fontsize=12, fontweight='bold', color='gray')

        # Pred Box (Shifted Up-Right)
        rect_diag = patches.Rectangle((45, 45), 40, 40, linewidth=2, edgecolor=color_pred, facecolor=color_pred,
                                      alpha=0.2)
        ax.add_patch(rect_diag)

        # 箭头
        ax.arrow(50, 50, 10, 10, head_width=3, head_length=4, fc=color_pred, ec=color_pred)


# ===========================
# 3. 主绘图程序
# ===========================

# 定义你要对比的 Loss 列表
# 格式: [Legend Name, iou_type string, kwargs]
LOSS_CONFIGS = [
    # ["CIoU", "CIoU", {}],
    ["Hausdorff in Gaussian Kernel", "Hausdorff", {"lambda1": 2.5}],
    ["L2 in Laplacian Kernel", "l1_ext", {"lambda1": 7.0}],
    ["L2 in Laplacian Kernel_ori", "l1", {"lambda1": 7.0}],
    ["HATS", "Hausdorff_Ext_L2", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 7}],
    ["HATS_rfix", "Hausdorff_Ext_L2_rfix", {"lambda1": 2.5, "hybrid_pow": 4, "lambda3": 7}],
    ["IoU", "IoU", {}],
]

_size = 40
RANGE = _size * 3
BBOX_SIZE = [_size, _size]
STRIDE = RANGE / 1000.0

# 定义 X 轴：像素偏差 (从 -30 到 30)
DEVIATION_STEPS = np.arange(-RANGE, RANGE+1, STRIDE)  # 步长为2，产生稀疏点
CENTER_IDX = len(DEVIATION_STEPS) // 2

# 创建 GT Box (中心点 100, 100, 宽高 40, 40)
GT_BOX_TENSOR = torch.tensor([[100.0, 100.0, BBOX_SIZE[0], BBOX_SIZE[1]]])

# --- 场景 A 数据生成: 纯水平移动 ---
bboxes_horizontal = []
for dev in DEVIATION_STEPS:
    # x 发生偏差，y 不变，wh 不变
    bboxes_horizontal.append([100.0 + dev, 100.0, BBOX_SIZE[0], BBOX_SIZE[1]])
BBOXES_HOR = torch.tensor(bboxes_horizontal)

# --- 场景 B 数据生成: 对角线移动 (XY同时偏差) ---
bboxes_diagonal = []
for dev in DEVIATION_STEPS:
    # x 和 y 同时发生偏差
    bboxes_diagonal.append([100.0 + dev, 100.0 + dev, BBOX_SIZE[0], BBOX_SIZE[1]])
BBOXES_DIAG = torch.tensor(bboxes_diagonal)

# 计算 Loss
results_hor = get_loss_values(LOSS_CONFIGS, GT_BOX_TENSOR, BBOXES_HOR)
results_diag = get_loss_values(LOSS_CONFIGS, GT_BOX_TENSOR, BBOXES_DIAG)

# --- 开始画图 ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={'width_ratios': [1, 2.5]})
plt.subplots_adjust(wspace=0.1, hspace=0.3)

# Row 1: Horizontal Shift
draw_schematic(axes[0, 0], scenario='horizontal')
ax1 = axes[0, 1]

for i, (name, val_array) in enumerate(results_hor.items()):
    color = COLORS[i % len(COLORS)]
    # 关键修改：markersize=4 (更小), linewidth=1.5 (细线)
    ax1.plot(DEVIATION_STEPS, val_array, linestyle='-', label=name, color=color, alpha=0.9)

ax1.set_title("Loss Landscape: Horizontal Deviation", fontsize=14)
ax1.set_ylabel("Distance", fontsize=11)
ax1.set_ylim(bottom=0)  # Loss 从 0 开始
ax1.grid(True, linestyle='--', alpha=0.4)  # 网格更淡
ax1.legend(fontsize=9, frameon=True, ncol=1)

# Row 2: Diagonal Shift
draw_schematic(axes[1, 0], scenario='diagonal')
ax2 = axes[1, 1]

for i, (name, val_array) in enumerate(results_diag.items()):
    color = COLORS[i % len(COLORS)]
    ax2.plot(DEVIATION_STEPS, val_array, linestyle='-', label=name, color=color, alpha=0.9)

ax2.set_title("Loss Landscape: Diagonal Deviation", fontsize=14)
ax2.set_xlabel("Deviation (Pixels)", fontsize=12)
ax2.set_ylabel("Loss Value", fontsize=11)
ax2.set_ylim(bottom=0)
ax2.grid(True, linestyle='--', alpha=0.4)
# ax2.legend() # 下面的图可以不加 Legend 保持简洁，或者也加上

# 统一设置 X 轴刻度
for ax in [ax1, ax2]:
    ax.set_xticks(np.arange(-RANGE, RANGE+1, 10))
    ax.set_xlim(-(RANGE+1), (RANGE+1))

plt.savefig('hausdorff_iou_loss_comparison.png', dpi=200, bbox_inches='tight')
plt.show()