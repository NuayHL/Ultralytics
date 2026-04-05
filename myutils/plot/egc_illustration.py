import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def plot_dynamic_egc_illustration():
    # Setup the figure with 3 subplots
    # fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5))  # 增加高度给图例留位置
    plt.subplots_adjust(bottom=0.25, top=0.85, wspace=0.15)

    # Common parameters
    stride = 1.0  # Normalize stride to 1 for easier visualization
    # Define grid range
    grid_x, grid_y = np.meshgrid(np.arange(-1, 5, stride), np.arange(-1, 5, stride))
    anchors = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)

    # Helper function to compute expansion based on user's logic (simplified version of func_1/smooth)
    def get_expansion(obj_size, stride, mode='dynamic'):
        if mode == 'none':
            return 0.0

        # Calculate ratio r = stride / size
        # Mimicking the logic: r = stride / w
        r = stride / obj_size

        # A simplified dynamic function mimicking your 'func_1' or 'func_scale_adaptive'
        # If object is large (r is small), expansion is small.
        # If object is small (r is large), expansion is large.

        scale_ratio = 0.0
        if r < 0.5:
            scale_ratio = 0.1  # Minimal expansion for large objects
        elif r >= 0.5 and r < 1.5:
            scale_ratio = 0.5  # Moderate
        else:
            scale_ratio = 0.85  # Aggressive expansion for tiny objects

        # The expansion amount (padding) is stride * scale_ratio
        return stride * scale_ratio

    # -------------------------------------------------------------------------
    # Scenario 1: Large/Normal Object (Low Stride/Size Ratio)
    # -------------------------------------------------------------------------
    ax = axes[0]
    gt_cx, gt_cy = 1.5, 1.5
    gt_w, gt_h = 2.2, 2.2  # Large object (approx 2x stride)

    expansion = get_expansion(gt_w, stride, mode='dynamic')  # Should be small

    _draw_scenario(ax, anchors, stride, gt_cx, gt_cy, gt_w, gt_h, expansion,
                   title="Case 1: Large Object (Ratio < 0.5)\nSufficient Anchors inside GT",
                   show_expansion=True)

    # -------------------------------------------------------------------------
    # Scenario 2: Tiny Object (High Stride/Size Ratio) - Without EGC
    # -------------------------------------------------------------------------
    ax = axes[1]
    gt_cx, gt_cy = 1.5, 1.5  # Placed exactly between grids to show "miss"
    gt_w, gt_h = 0.4, 0.4  # Tiny object (0.4x stride)

    expansion = 0.0  # No expansion

    _draw_scenario(ax, anchors, stride, gt_cx, gt_cy, gt_w, gt_h, expansion,
                   title="Case 2: Tiny Object (Standard TAL)\nZero/Few Anchors inside GT -> Loss NaN / Poor Training",
                   show_expansion=False)

    # -------------------------------------------------------------------------
    # Scenario 3: Tiny Object (High Stride/Size Ratio) - With Dynamic EGC
    # -------------------------------------------------------------------------
    ax = axes[2]
    # Same tiny object
    expansion = get_expansion(gt_w, stride, mode='dynamic')  # Should be large

    _draw_scenario(ax, anchors, stride, gt_cx, gt_cy, gt_w, gt_h, expansion,
                   title=f"Case 3: Tiny Object with EGC (Ratio > 1.0)\nDynamic Expansion Captures Neighbor Anchors",
                   show_expansion=True)

    # Legend (Custom)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='black', lw=1, alpha=0.3, label='Feature Grid (Stride)'),
        patches.Patch(edgecolor='red', facecolor='none', lw=2, label='Ground Truth (GT)'),
        patches.Patch(facecolor='blue', alpha=0.2, label='Original Candidate Region (Inside GT)'),
        patches.Patch(facecolor='green', alpha=0.2, linestyle='--', label='Expanded Region (EGC)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='Original Positive Anchors',
               markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='Added Positive Anchors', markersize=8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='gray',
               label='Negative Anchors', markersize=6),
    ]

    fig.legend(handles=legend_elements, loc='upper center', ncol=4,
               bbox_to_anchor=(0.5, 0.18), fontsize=14, frameon=False)

    # 调整主标题位置，避免太高
    plt.suptitle("Visualization of Dynamic Expand Geometry Constraint (EGC) for Label Assignment",
                 fontsize=18, y=0.95, fontweight='bold')

    # fig.legend(handles=legend_elements, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=12)
    # plt.suptitle("Visualization of Dynamic Expand Geometry Constraint (EGC) for Label Assignment", fontsize=16, y=1.05)
    plt.savefig('egc_illustration.pdf', dpi=300, format='pdf')
    plt.savefig('egc_illustration.png', dpi=300)
    plt.show()


def _draw_scenario(ax, anchors, stride, gt_cx, gt_cy, gt_w, gt_h, expansion, title, show_expansion):
    # Draw Grid
    ax.set_aspect('equal')
    ax.set_xticks(np.arange(-1, 5, stride))
    ax.set_yticks(np.arange(-1, 5, stride))
    ax.grid(True, linestyle=':', alpha=0.6, color='black')
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)

    # 1. Draw Ground Truth
    x1, y1 = gt_cx - gt_w / 2, gt_cy - gt_h / 2
    rect_gt = patches.Rectangle((x1, y1), gt_w, gt_h, linewidth=2, edgecolor='red', facecolor='none', zorder=10)
    ax.add_patch(rect_gt)

    # 2. Draw Original Region (Inside GT) visual fill
    rect_orig = patches.Rectangle((x1, y1), gt_w, gt_h, linewidth=0, facecolor='blue', alpha=0.15, zorder=5)
    ax.add_patch(rect_orig)

    # 3. Draw Expanded Region
    if show_expansion and expansion > 0:
        exp_w = gt_w + 2 * expansion
        exp_h = gt_h + 2 * expansion
        exp_x1 = gt_cx - exp_w / 2
        exp_y1 = gt_cy - exp_h / 2

        rect_exp = patches.Rectangle((exp_x1, exp_y1), exp_w, exp_h,
                                     linewidth=2, linestyle='--', edgecolor='green', facecolor='green', alpha=0.1,
                                     zorder=4)
        ax.add_patch(rect_exp)

        # Add annotation for 'padding'
        if expansion > 0.2:
            ax.annotate(r'$+\delta_{expand}$', xy=(gt_cx + gt_w / 2, gt_cy),
                        xytext=(gt_cx + gt_w / 2 + expansion, gt_cy),
                        arrowprops=dict(arrowstyle='<->', color='green', lw=1.5), color='green', fontsize=10, ha='left')

    # 4. Determine Anchor Status
    for ax_x, ax_y in anchors:
        # Check Inside GT
        in_gt_x = (ax_x > x1) and (ax_x < x1 + gt_w)
        in_gt_y = (ax_y > y1) and (ax_y < y1 + gt_h)
        is_orig_pos = in_gt_x and in_gt_y

        # Check Inside Expansion
        if show_expansion:
            # Logic: |center - anchor| < (size/2 + expansion)
            in_exp_x = abs(ax_x - gt_cx) < (gt_w / 2 + expansion)
            in_exp_y = abs(ax_y - gt_cy) < (gt_h / 2 + expansion)
            is_new_pos = in_exp_x and in_exp_y and not is_orig_pos
        else:
            is_new_pos = False

        # Plot Anchors
        if is_orig_pos:
            ax.scatter(ax_x, ax_y, c='blue', s=80, zorder=20, edgecolors='white')
        elif is_new_pos:
            ax.scatter(ax_x, ax_y, c='green', s=80, zorder=20, edgecolors='white')
        else:
            if 0 <= ax_x <= 3 and 0 <= ax_y <= 3:  # Only plot relevant background anchors
                ax.scatter(ax_x, ax_y, c='gray', s=30, alpha=0.5, zorder=1)

    ax.set_title(title, fontsize=11, pad=10)
    # Hide ticks for cleaner look
    ax.set_xticklabels([])
    ax.set_yticklabels([])


if __name__ == "__main__":
    plot_dynamic_egc_illustration()

