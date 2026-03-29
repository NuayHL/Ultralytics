import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm


def create_difficulty_heatmap_icon(save_name='icon_difficulty_heatmap.png'):
    """
    生成难度融合热力图 (Difficulty Fusion Heatmap)
    X轴: Uncertainty (Low -> High)
    Y轴: Scale Indicator (Large Obj -> Small Obj) / Difficulty Factor
    Result: Color Gradient (Easy -> Hard)
    """
    # ================= 配置区域 =================
    colors = {
        'axis': 'black',  # 坐标轴颜色
        'crosshair': 'black',  # 准星颜色
        'bg': 'none'  # 背景透明
    }

    # 画布大小 (方形，紧凑)
    fig, ax = plt.subplots(figsize=(4, 4))

    # -----------------------------------------------------------
    # 1. 生成热力图数据
    # -----------------------------------------------------------
    # 分辨率
    res = 100
    x = np.linspace(0, 1, res)  # Uncertainty
    y = np.linspace(0, 1, res)  # Scale Factor (0=Large/Easy, 1=Small/Hard)
    X, Y = np.meshgrid(x, y)

    # 定义难度函数 Z (简单的线性融合或欧氏距离)
    # Z 越大，难度越高 (颜色越红)
    # 这里用简单的线性叠加: Difficulty = 0.5*Uncertainty + 0.5*ScaleFactor
    Z = 0.5 * X + 0.5 * Y

    # -----------------------------------------------------------
    # 2. 绘制热力图
    # -----------------------------------------------------------
    # 使用 'coolwarm' 或 'RdYlBu_r' (Red-Yellow-Blue reversed)
    # 这样 蓝色=Easy(0), 红色=Hard(1)
    im = ax.imshow(Z, extent=[0, 1, 0, 1], origin='lower',
                   cmap='RdYlBu_r', interpolation='bicubic', alpha=0.9)

    # -----------------------------------------------------------
    # 3. 绘制 "当前样本" 十字准星 (Operating Point)
    # -----------------------------------------------------------
    # 假设有一个样本，Uncertainty较高(0.7)，且是小物体(0.6) -> 比较难
    sample_x = 0.7
    sample_y = 0.6

    # 画准星 (Crosshair)
    # 线的长度
    d = 0.08
    # 横线
    ax.plot([sample_x - d, sample_x + d], [sample_y, sample_y],
            color=colors['crosshair'], lw=4, zorder=10, solid_capstyle='round')
    # 竖线
    ax.plot([sample_x, sample_x], [sample_y - d, sample_y + d],
            color=colors['crosshair'], lw=4, zorder=10, solid_capstyle='round')

    # 加一个中心圆点
    ax.scatter(sample_x, sample_y, s=150, c=colors['crosshair'], zorder=11)

    # -----------------------------------------------------------
    # 4. 绘制坐标轴 (Icon 风格 - 贴边)
    # -----------------------------------------------------------
    # 定义箭头长度 (稍微超出热力图一点点)
    arrow_len = 1.15

    # X 轴箭头 (Bottom)
    ax.arrow(0, 0, arrow_len, 0,
             fc=colors['axis'], ec=colors['axis'], lw=3,
             head_width=0.06, head_length=0.08, zorder=20, clip_on=False)

    # Y 轴箭头 (Left)
    ax.arrow(0, 0, 0, arrow_len,
             fc=colors['axis'], ec=colors['axis'], lw=3,
             head_width=0.06, head_length=0.08, zorder=20, clip_on=False)

    # -----------------------------------------------------------
    # 5. 调整布局与输出
    # -----------------------------------------------------------
    ax.set_axis_off()

    # 限制范围，留出箭头的位置
    ax.set_xlim(-0.05, 1.25)
    ax.set_ylim(-0.05, 1.25)

    plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Generated Heatmap Icon: {save_name}")
    plt.close()


if __name__ == "__main__":
    create_difficulty_heatmap_icon('asset_difficulty_heatmap.png')