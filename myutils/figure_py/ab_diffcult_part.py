import matplotlib.pyplot as plt
import numpy as np


def create_mapping_curve_icon(mode='alpha', save_name='icon_mapping.png'):
    """
    生成参数映射曲线 Icon (Alpha 或 Beta)
    mode: 'alpha' (上升 S 曲线) 或 'beta' (下降 S 曲线)
    """
    # ================= 配置区域 =================
    colors = {
        'axis': 'black',  # 坐标轴颜色
        'line': '#000000',  # 曲线颜色 (纯黑，高对比度)
        # 如果想要彩色，可以换成: '#1F77B4' (蓝) 或 '#D62728' (红)
        'bg': 'none'  # 背景透明
    }

    # 画布大小 (比之前的 icon 稍高一点，为了展示 S 形的完整性)
    fig, ax = plt.subplots(figsize=(4.5, 3.5))

    # -----------------------------------------------------------
    # 1. 生成数据 (Sigmoid 及其变体)
    # -----------------------------------------------------------
    # x 范围 [-6, 6] 可以生成比较完美的 S 形
    x = np.linspace(-6, 6, 300)

    if mode == 'alpha':
        # Alpha: 上升曲线 (Sigmoid)
        # 含义: Difficulty 越大 -> 权重越高 (Trust more)
        y = 1 / (1 + np.exp(-x))
    else:
        # Beta: 下降曲线 (1 - Sigmoid)
        # 含义: Difficulty 越大 -> 权重越低 (Dampen noise)
        y = 1 - (1 / (1 + np.exp(-x)))

    # -----------------------------------------------------------
    # 2. 绘制曲线
    # -----------------------------------------------------------
    # 留出一点边距，不要贴边
    # 将 x 归一化到绘图空间，以便和坐标轴配合
    x_plot = (x - x.min()) / (x.max() - x.min()) * 3.0  # 映射到 [0, 3]
    y_plot = y * 2.0 + 0.2  # 映射高度，并稍微抬高一点以免碰底

    ax.plot(x_plot, y_plot, color=colors['line'], linewidth=4.5, zorder=5)

    # -----------------------------------------------------------
    # 3. 绘制坐标轴 (Icon 风格)
    # -----------------------------------------------------------
    # 定义坐标轴长度
    x_len = 3.5
    y_len = 2.8

    # X 轴箭头
    ax.arrow(-0.1, 0, x_len, 0,
             fc=colors['axis'], ec=colors['axis'], lw=2.5,
             head_width=0.12, head_length=0.2, zorder=10)

    # Y 轴箭头
    ax.arrow(0, -0.1, 0, y_len,
             fc=colors['axis'], ec=colors['axis'], lw=2.5,
             head_width=0.15, head_length=0.15, zorder=10)

    # -----------------------------------------------------------
    # 4. 调整布局与输出
    # -----------------------------------------------------------
    ax.set_axis_off()

    # 设置显示范围 (稍微留白，防止箭头出界)
    ax.set_xlim(-0.3, x_len + 0.3)
    ax.set_ylim(-0.3, y_len + 0.3)

    plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Generated Mapping Icon ({mode}): {save_name}")
    plt.close()


if __name__ == "__main__":
    # 生成 Alpha 曲线 (上升)
    create_mapping_curve_icon(mode='alpha', save_name='asset_mapping_alpha.png')

    # 生成 Beta 曲线 (下降)
    create_mapping_curve_icon(mode='beta', save_name='asset_mapping_beta.png')