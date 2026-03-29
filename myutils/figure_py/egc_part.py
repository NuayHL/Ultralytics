import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def create_grid_component(mode='traditional', save_name='component.png'):
    """
    生成 Label Assignment 的示意图组件
    mode: 'traditional' (左图) 或 'ours' (右图)
    """
    # ================= 配置区域 (配色参考原图) =================
    colors = {
        'grid_line': '#B0B0B0',  # 浅灰网格线
        'gt_edge': '#D62728',  # GT 红色边框
        'gt_face': '#F7C4C4',  # GT 浅红填充
        'anchor_dot': '#4879BA',  # Anchor 蓝色圆点
        'anchor_edge': '#30517E',  # Anchor 深蓝色边界
        'expand_edge': '#FF7F0E',  # 扩展区域 橙色边框
        'expand_fill': '#FFBB78',  # 扩展区域 浅橙色填充
        'bg_fill': '#F0F5FA'  # 画布背景淡蓝色 (可选，这里设为透明以便Draw.io使用)
    }

    # 几何设置
    grid_size = 5  # 5x5 网格
    gt_center = (2.7, 2.5)  # GT 中心坐标
    gt_width = (0.8, 1.1)  # GT 宽度 (对应1个Grid大小)
    corner_radius = 0.15  # 圆角半径
    line_width = 3

    # 初始化画布
    fig, ax = plt.subplots(figsize=(4, 4))
    # ax.set_facecolor(colors['bg_fill']) # 如果需要背景色可解开，建议透明

    # 1. 绘制网格 (Grid Lines)
    for i in range(grid_size + 1):
        # 横线
        ax.plot([0, grid_size], [i, i], color=colors['grid_line'], lw=line_width, zorder=1)
        # 竖线
        ax.plot([i, i], [0, grid_size], color=colors['grid_line'], lw=line_width, zorder=1)

    # 2. 绘制 Ground Truth (GT)
    # 计算左下角
    gt_xy = (gt_center[0] - gt_width[0] / 2, gt_center[1] - gt_width[1] / 2)
    rect_gt = patches.Rectangle(
        gt_xy, gt_width[0], gt_width[1],
        linewidth=3.5, edgecolor=colors['gt_edge'], facecolor='none', zorder=10
    )
    ax.add_patch(rect_gt)

    # 3. 根据模式绘制 Anchor 和 扩展区域

    # 生成所有潜在的 anchor 中心点
    centers = np.arange(0.5, grid_size, 1)
    grid_x, grid_y = np.meshgrid(centers, centers)
    all_anchors = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)

    active_anchors = []

    if mode == 'traditional':
        # Traditional: 只有中心点落在 GT 内的才是候选
        # 简单模拟：只取最中间的一个点
        active_anchors = [[2.5, 2.5]]

        rect_gt = patches.FancyBboxPatch(
            gt_xy, gt_width[0], gt_width[1],
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            linewidth=3.5, edgecolor='none', facecolor=colors['gt_face'], alpha=0.3, zorder=5
        )
        ax.add_patch(rect_gt)

    elif mode == 'ours':
        # Ours: 绘制扩展区域 (Expanded Region)
        expand_r = 1.0  # 扩展半径
        expand_width = [gt_width[0] + 2 * expand_r, gt_width[1] + 2 * expand_r]
        exp_xy = (gt_center[0] - expand_width[0] / 2, gt_center[1] - expand_width[1] / 2)

        # 绘制橙色虚线框
        rect_exp = patches.FancyBboxPatch(
            exp_xy, expand_width[0], expand_width[1],
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            linewidth=2.5, linestyle='--',
            edgecolor=colors['expand_edge'], facecolor=colors['expand_fill'],
            alpha=0.3, zorder=5
        )
        ax.add_patch(rect_exp)

        # 绘制橙色虚线边框 (为了更清晰，再叠一层不透明的边框)
        rect_exp_border = patches.FancyBboxPatch(
            exp_xy, expand_width[0], expand_width[1],
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            linewidth=2.5, linestyle='--',
            edgecolor=colors['expand_edge'], facecolor='none',
            zorder=6
        )
        ax.add_patch(rect_exp_border)

        # Ours: 扩展区域内的点都是候选
        # 简单模拟：取中心 3x3 区域的点
        for ax_x in centers:
            for ax_y in centers:
                if 1.0 < ax_x < 4.0 and 1.0 < ax_y < 4.0:
                    active_anchors.append([ax_x, ax_y])

    # 4. 绘制 Anchor 点
    active_anchors = np.array(active_anchors)
    if len(active_anchors) > 0:
        ax.scatter(
            active_anchors[:, 0], active_anchors[:, 1],
            s=120, c=colors['anchor_dot'], edgecolors=colors['anchor_edge'], linewidth=1.3, zorder=20
        )

    # 5. 调整显示
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)
    ax.axis('off')  # 关闭坐标轴

    # 保存为去背景的 PNG，方便 Draw.io 使用
    plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Generated: {save_name}")
    plt.close()


if __name__ == "__main__":
    # 生成左边的图 (Traditional)
    create_grid_component(mode='traditional', save_name='asset_traditional.png')

    # 生成右边的图 (Ours)
    create_grid_component(mode='ours', save_name='asset_ours_expanded.png')