import matplotlib.pyplot as plt
import numpy as np


def create_similarity_curve_icon(save_name='icon_similarity_curves.png'):
    """
    生成 BBox Similarity 对比曲线 Icon
    对比 IoU (尖峰) 和 Ours (平台)
    """
    # ================= 配置区域 =================
    colors = {
        'axis': 'black',  # 坐标轴颜色
        'iou_line': '#1F77B4',  # IoU 曲线颜色 (深蓝 - 对应上一张图的 Traditional/Anchor)
        'ours_line': '#FF7F0E',  # Ours 曲线颜色 (橙色 - 对应上一张图的 Expand/Ours)
        'bg': 'none'  # 背景透明
    }

    # 几何与数据设置
    x_range = 3.0  # X轴范围

    # 初始化画布
    fig, ax = plt.subplots(figsize=(4, 2))

    # -----------------------------------------------------------
    # 1. 生成数据
    # -----------------------------------------------------------
    x = np.linspace(-x_range, x_range, 300)

    # A. 模拟 IoU (Sharp Peak)
    # 使用标准高斯，sigma 较小，表现为尖锐
    sigma_iou = 0.6
    y_iou = np.exp(-0.5 * (x / sigma_iou) ** 2)

    # B. 模拟 Ours (Plateau/Stable)
    # 使用超高斯 (Super-Gaussian)，幂次(power)越大，顶部越平
    # 同时也稍微调大一点宽度 (scale)，体现对噪声的容忍
    scale_ours = 1.3
    power_ours = 4.0  # 幂次越高越平
    y_ours = np.exp(-0.5 * (abs(x) / scale_ours) ** power_ours)

    # -----------------------------------------------------------
    # 2. 绘制曲线
    # -----------------------------------------------------------

    # 绘制 IoU (蓝色，在下层)
    ax.plot(x, y_iou, color=colors['iou_line'], linewidth=3.5, label='IoU', zorder=5)

    # 绘制 Ours (橙色，在上层，因为它的顶部更宽，不会遮挡 IoU 的尖峰)
    ax.plot(x, y_ours, color=colors['ours_line'], linewidth=3.5, label='Ours', zorder=6)

    # -----------------------------------------------------------
    # 3. 绘制坐标轴 (Icon 风格)
    # -----------------------------------------------------------
    # 定义坐标轴的绘制位置
    x_lim = x_range + 0.2
    y_lim = 1.2

    # X 轴箭头 (从左到右)
    ax.arrow(-x_lim, 0, 2 * x_lim, 0,
             fc=colors['axis'], ec=colors['axis'], lw=2,
             head_width=0.06, head_length=0.2, zorder=10)

    # Y 轴箭头 (从下到上)
    # 起点稍微往下一点 (-0.1) 避免曲线贴底不好看
    ax.arrow(-x_lim, 0, 0, y_lim,
             fc=colors['axis'], ec=colors['axis'], lw=2,
             head_width=0.2, head_length=0.06, zorder=10)

    # -----------------------------------------------------------
    # 4. 调整布局与输出
    # -----------------------------------------------------------

    # 隐藏 Matplotlib 默认边框
    ax.set_axis_off()

    # 设置显示范围
    ax.set_xlim(-x_lim - 0.3, x_lim + 0.3)
    ax.set_ylim(-0.05, y_lim + 0.1)

    # 保存
    plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Generated Similarity Icon: {save_name}")
    plt.close()


if __name__ == "__main__":
    create_similarity_curve_icon('asset_similarity_curves.png')