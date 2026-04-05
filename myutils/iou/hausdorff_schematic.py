import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_hausdorff_schematic():
    # Setup the figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # -------------------------------------------------------
    # 1. Define Box Coordinates (Axis-Aligned)
    # -------------------------------------------------------
    # Box 1: Anchor (B1) - Let's make it smaller and slightly shifted
    # Format: (x, y, width, height)
    b1_x, b1_y = 2.0, 2.0
    b1_w, b1_h = 4.0, 3.0

    # Box 2: Ground Truth (B2) - Larger and shifted to make Top-Right the furthest
    b2_x, b2_y = 3.5, 3.0
    b2_w, b2_h = 5.0, 4.5

    # -------------------------------------------------------
    # 2. Calculate Corner Points
    # -------------------------------------------------------
    # Helper to get corners: [TL, TR, BR, BL]
    # Note: In standard Cartesian, TL is (x, y+h), but for visual consistency with
    # image coordinates often used in CV, we can stick to standard math plot (y-up).
    # Here we treat it as standard geometry (y-up).

    # Corners for B1 (Anchor)
    p1 = {
        'tl': np.array([b1_x, b1_y + b1_h]),
        'tr': np.array([b1_x + b1_w, b1_y + b1_h]),
        'br': np.array([b1_x + b1_w, b1_y]),
        'bl': np.array([b1_x, b1_y])
    }

    # Corners for B2 (Ground Truth)
    p2 = {
        'tl': np.array([b2_x, b2_y + b2_h]),
        'tr': np.array([b2_x + b2_w, b2_y + b2_h]),
        'br': np.array([b2_x + b2_w, b2_y]),
        'bl': np.array([b2_x, b2_y])
    }

    # -------------------------------------------------------
    # 3. Draw Boxes
    # -------------------------------------------------------
    # Box 1 (Anchor) - Blue Dashed
    rect1 = patches.Rectangle((b1_x, b1_y), b1_w, b1_h, linewidth=2, edgecolor='#1f77b4', facecolor='none',
                              linestyle='--')
    ax.add_patch(rect1)
    ax.text(b1_x + 0.2, b1_y + 0.2, r'$B_1$ (Anchor)', color='#1f77b4', fontsize=12, fontweight='bold')

    # Box 2 (GT) - Green Solid
    rect2 = patches.Rectangle((b2_x, b2_y), b2_w, b2_h, linewidth=2, edgecolor='#2ca02c', facecolor='none',
                              linestyle='-')
    ax.add_patch(rect2)
    ax.text(b2_x + b2_w - 2.5, b2_y + 0.2, r'$B_2$ (Ground Truth)', color='#2ca02c', fontsize=12, fontweight='bold')

    # -------------------------------------------------------
    # 4. Calculate Distances and Draw Lines
    # -------------------------------------------------------
    corners = ['tl', 'tr', 'br', 'bl']
    distances = {}

    # Calculate all distances first to find the max
    for c in corners:
        dist = np.linalg.norm(p1[c] - p2[c])
        distances[c] = dist

    # Find which corner pair has the maximum distance (Hausdorff Distance)
    max_corner = max(distances, key=distances.get)

    # Draw connection lines
    for c in corners:
        start = p1[c]
        end = p2[c]

        # Determine style based on whether it is the max distance
        if c == max_corner:
            # Highlight the Hausdorff distance
            plt.plot([start[0], end[0]], [start[1], end[1]], color='#d62728', linewidth=2.5, zorder=10)  # Red

            # Label the distance
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x - 0.5, mid_y + 0.2, r'$d_H(B_1, B_2)$', color='#d62728', fontsize=14, fontweight='bold')
        else:
            # Normal corner distance
            plt.plot([start[0], end[0]], [start[1], end[1]], color='gray', linestyle=':', linewidth=1.5, alpha=0.7)

    # -------------------------------------------------------
    # 5. Annotate Points (Corners)
    # -------------------------------------------------------
    # Mapping for LaTeX labels
    labels = {'tl': 'tl', 'tr': 'tr', 'br': 'br', 'bl': 'bl'}

    for c in corners:
        # Points for B1
        ax.plot(p1[c][0], p1[c][1], 'o', color='#1f77b4', markersize=6)
        # Offset text slightly
        off_x = -0.6 if 'l' in c else 0.2
        off_y = 0.2 if 't' in c else -0.5
        ax.text(p1[c][0] + off_x, p1[c][1] + off_y, rf'$C_{{{labels[c]}}}^{(1)}$', fontsize=12, color='#1f77b4')

        # Points for B2
        ax.plot(p2[c][0], p2[c][1], 'o', color='#2ca02c', markersize=6)
        off_x = 0.2 if 'r' in c else -0.6
        off_y = 0.2 if 't' in c else -0.5
        ax.text(p2[c][0] + off_x, p2[c][1] + off_y, rf'$C_{{{labels[c]}}}^{(2)}$', fontsize=12, color='#2ca02c')

    # -------------------------------------------------------
    # 6. Final Formatting
    # -------------------------------------------------------
    # Set limits with some padding
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9)

    # Hide axes ticks but keep the box or make it look like a coordinate system
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Add arrows to axes
    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('hausdorff_schematic.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    draw_hausdorff_schematic()