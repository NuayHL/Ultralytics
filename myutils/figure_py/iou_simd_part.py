import matplotlib.pyplot as plt
import matplotlib.patches as patches


def create_bbox_geometry_icon(save_name='asset_bbox_geometry.png'):
    """
    生成 BBox 几何关系示意图 (Geometric Similarity)
    包含: IoU 重叠, Center Distance, Corner Connections
    """
    # ================= 配置区域 =================
    colors = {
        'gt_edge': '#D62728',  # GT 红
        'gt_fill': '#D62728',  # GT 淡红填充
        'gt_diag': '#D62728',  # GT 内部对角线

        'pred_edge': '#1F77B4',  # Pred 蓝
        'pred_fill': '#1F77B4',  # Pred 淡蓝填充
        'pred_diag': '#1F77B4',  # Pred 内部对角线

        'center_line': 'black',  # 中心连线
        'corner_line': 'gray',  # 角点连线
        'bg': 'none'  # 透明背景
    }

    # 画布设置
    fig, ax = plt.subplots(figsize=(4, 4))

    # ================= 数据定义 =================
    # 格式: [center_x, center_y, width, height]
    # 故意设置成部分重叠，且形状略有不同
    gt_box = [1.2, 2.2, 1.4, 1.4]  # GT 在左上
    pred_box = [2.0, 1.9, 1.3, 1.6]  # Pred 在右下

    def get_corners(box):
        cx, cy, w, h = box
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2
        # 返回: 左下, 左上, 右上, 右下 (逆时针或顺时针均可，这里按顺序)
        # Matplotlib Rectangle 是从左下角开始的
        return {
            'bl': (x1, y1), 'tl': (x1, y2),
            'tr': (x2, y2), 'br': (x2, y1),
            'center': (cx, cy),
            'w': w, 'h': h,
            'xy': (x1, y1)  # 左下角坐标用于画矩形
        }

    gt = get_corners(gt_box)
    pred = get_corners(pred_box)

    # ================= 绘图步骤 =================

    # 1. 绘制角点连接线 (Corner Distance Lines)
    # 放在最底层 (zorder=1)，用细虚线或实线
    # corners = ['tl', 'tr', 'bl', 'br']
    # for c in corners:
    #     ax.plot([gt[c][0], pred[c][0]], [gt[c][1], pred[c][1]],
    #             color=colors['corner_line'], linewidth=1.0, linestyle='--', alpha=0.6, zorder=1)

    # 2. 绘制中心连接线 (Center Distance Line)
    # 稍微粗一点，实线
    ax.plot([gt['center'][0], pred['center'][0]], [gt['center'][1], pred['center'][1]],
            color=colors['center_line'], linewidth=2.0, zorder=20, alpha=0.3)

    # 3. 绘制矩形填充 (Fills)
    # GT Fill
    ax.add_patch(patches.Rectangle(gt['xy'], gt['w'], gt['h'],
                                   linewidth=0, facecolor=colors['gt_fill'], alpha=0.1, zorder=3))
    # Pred Fill
    ax.add_patch(patches.Rectangle(pred['xy'], pred['w'], pred['h'],
                                   linewidth=0, facecolor=colors['pred_fill'], alpha=0.1, zorder=3))

    # 4. 绘制矩形内部对角线 (Diagonals - 强调中心感)
    # GT Diagonals
    # ax.plot([gt['tl'][0], gt['br'][0]], [gt['tl'][1], gt['br'][1]], color=colors['gt_diag'], lw=1, alpha=0.4, zorder=4)
    # ax.plot([gt['bl'][0], gt['tr'][0]], [gt['bl'][1], gt['tr'][1]], color=colors['gt_diag'], lw=1, alpha=0.4, zorder=4)
    # # Pred Diagonals
    # ax.plot([pred['tl'][0], pred['br'][0]], [pred['tl'][1], pred['br'][1]], color=colors['pred_diag'], lw=1, alpha=0.4,
    #         zorder=4)
    # ax.plot([pred['bl'][0], pred['tr'][0]], [pred['bl'][1], pred['tr'][1]], color=colors['pred_diag'], lw=1, alpha=0.4,
    #         zorder=4)

    # 5. 绘制矩形边框 (Edges)
    # GT Edge
    ax.add_patch(patches.Rectangle(gt['xy'], gt['w'], gt['h'],
                                   linewidth=3, edgecolor=colors['gt_edge'], facecolor='none', zorder=5))
    # Pred Edge
    ax.add_patch(patches.Rectangle(pred['xy'], pred['w'], pred['h'],
                                   linewidth=3, edgecolor=colors['pred_edge'], facecolor='none', zorder=5))

    # 6. 绘制中心点 (Dots)
    ax.scatter(*gt['center'], s=150, c=colors['gt_edge'], zorder=21, edgecolors='none', linewidth=1.5)
    ax.scatter(*pred['center'], s=150, c=colors['pred_edge'], zorder=21, edgecolors='none', linewidth=1.5)

    # ================= 调整与输出 =================
    ax.set_aspect('equal')
    ax.set_axis_off()

    # 留白控制
    all_x = [gt['xy'][0], gt['xy'][0] + gt['w'], pred['xy'][0], pred['xy'][0] + pred['w']]
    all_y = [gt['xy'][1], gt['xy'][1] + gt['h'], pred['xy'][1], pred['xy'][1] + pred['h']]
    margin = 0.2
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    plt.savefig(save_name, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Generated BBox Geometry Icon: {save_name}")
    plt.close()


if __name__ == "__main__":
    create_bbox_geometry_icon('asset_bbox_geometry.png')