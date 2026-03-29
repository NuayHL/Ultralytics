import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def create_dfl_distribution_icon(save_name='icon_dfl_dist.png'):
    """
    生成 DFL/GFL 概率分布示意图 (Icon style)
    """
    # ================= 配置区域 =================
    colors = {
        'axis': 'black',  # 坐标轴颜色
        'curve': '#1F77B4',  # 连续分布曲线颜色 (深蓝)
        'bar_fill': '#AEC7E8',  # 离散柱状图填充 (浅蓝)
        'bar_edge': '#4682B4',  # 离散柱状图边框 (中蓝)
        'bg': 'none'  # 背景透明
    }

    # 几何与数据设置
    mu, sigma = -1.0, 1.5  # 高斯分布参数
    x_range = 3.2  # X轴范围 (-x 到 x)
    num_bins = 7  # 柱状图的数量 (奇数最好，为了对称)

    # 初始化画布
    fig, ax = plt.subplots(figsize=(4, 2))  # 4:3 比例，像一个小插图

    # -----------------------------------------------------------
    # 1. 生成数据 (高斯分布)
    # -----------------------------------------------------------
    # 连续曲线数据
    x_smooth = np.linspace(-x_range, x_range, 200)
    y_smooth = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_smooth - mu) / sigma) ** 2)

    # 离散柱状图数据 (取样点)
    # 我们选择在 x_range 范围内均匀分布的 bin
    x_bins = np.linspace(-2.5, 2.5, num_bins)
    bin_width = (x_bins[1] - x_bins[0]) * 0.95  # 宽度稍微留一点缝隙，或者设为1.0紧贴
    # 计算每个 bin 高度 (直接取 PDF 值或者积分，这里为了icon好看直接取PDF中心值)
    y_bins = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_bins - mu) / sigma) ** 2)

    # -----------------------------------------------------------
    # 2. 绘制图形
    # -----------------------------------------------------------

    # A. 绘制离散柱状图 (Bars)
    # zorder=1: 在最底层
    ax.bar(x_bins, y_bins, width=bin_width,
           color=colors['bar_fill'], edgecolor=colors['bar_edge'],
           linewidth=1.5, alpha=0.9, zorder=1)

    # B. 绘制连续曲线 (Curve)
    # zorder=2: 在柱状图上面
    ax.plot(x_smooth, y_smooth, color=colors['curve'], linewidth=3.5, zorder=2)

    # -----------------------------------------------------------
    # 3. 绘制坐标轴 (自定义箭头)
    # -----------------------------------------------------------
    # Matplotlib 自带的 axis 比较丑，我们手动画带箭头的线

    # 定义坐标轴的极值
    x_lim = x_range + 0.5
    y_lim = max(y_smooth) * 1.2

    # X 轴箭头
    ax.arrow(-x_lim, 0, 2 * x_lim, 0,
             fc=colors['axis'], ec=colors['axis'], lw=2,
             head_width=0.03, head_length=0.2, zorder=10)

    # Y 轴箭头
    ax.arrow(-x_lim, 0, 0, y_lim,
             fc=colors['axis'], ec=colors['axis'], lw=2,
             head_width=0.15, head_length=0.03, zorder=10)

    # -----------------------------------------------------------
    # 4. 调整布局与去噪
    # -----------------------------------------------------------

    # 隐藏原本的边框和刻度
    ax.set_axis_off()

    # 限制画布范围，保证箭头不被切掉
    ax.set_xlim(-x_lim - 0.2, x_lim + 0.2)
    ax.set_ylim(-0.02, y_lim + 0.05)

    # 保存
    plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Generated DFL Icon: {save_name}")
    plt.close()


if __name__ == "__main__":
    create_dfl_distribution_icon('asset_dfl_distribution.png')