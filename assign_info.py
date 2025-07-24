import os
import pickle
from pathlib import Path
import random

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 设置 Matplotlib 支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# --- 配置参数 ---
# 你保存记录的目录 (根据你的 TaskAlignedAssigner_Record 初始化参数)
# 例如，如果 dir_name='train'，这里就是 'assign_record/train'
RECORD_DIR = "assign_record/visdrone"
# Assigner 中的 topk 参数，需要与你训练时设置的一致
TOPK = 10


def load_record(file_path):
    """加载单个 pickle 文件"""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"成功加载文件: {file_path}")
    # 打印一些基本信息
    print(f"  - Batch size: {data['gt_bboxes'].shape[0]}")
    print(f"  - Num GTs in first image: {(data['gt_labels'][0] != -1).sum().item()}")
    return data


def visualize_topk_overlaps(data, image_idx=0, max_gts_to_plot=20):
    """
    可视化 Top-K anchors 的 IoU 分布。
    对于指定图片中的每一个 GT，绘制其 align_metric 最高的 top-k 个 anchors 的 IoU 分布。

    Args:
        data (dict): 加载的 assignment 数据。
        image_idx (int): 要分析的图片在 batch 中的索引。
        max_gts_to_plot (int): 为防止图像过大，最多为多少个GT绘制子图。
    """
    print(f"\n--- 正在生成图片 {image_idx} 的 Top-K IoU 分布图 ---")

    # 从数据中解包
    align_metric = data['align_metric']  # (b, n_max_boxes, num_anchors)
    overlaps = data['overlaps']  # (b, n_max_boxes, num_anchors)
    gt_labels = data['gt_labels']  # (b, n_max_boxes, 1)
    mask_pos = data['mask_pos']  # (b, n_max_boxes, num_anchors)

    # 选择指定图片的数据
    gt_labels_img = gt_labels[image_idx].squeeze(-1)  # (n_max_boxes,)
    valid_gt_mask = gt_labels_img != -1  # 过滤掉 padding 的 GT

    valid_gt_indices = torch.where(valid_gt_mask)[0]
    num_valid_gts = len(valid_gt_indices)

    if num_valid_gts == 0:
        print(f"图片 {image_idx} 中没有有效的 GT Box，跳过可视化。")
        return

    # --- 修改部分：限制要绘制的GT数量 ---
    if num_valid_gts > max_gts_to_plot:
        print(f"警告: 图片中有 {num_valid_gts} 个GT，超过了显示上限 {max_gts_to_plot}。将随机采样进行可视化。")
        # 从有效的GT中随机采样
        indices_to_plot = random.sample(valid_gt_indices.tolist(), max_gts_to_plot)
        indices_to_plot.sort()  # 保持索引顺序
    else:
        indices_to_plot = valid_gt_indices.tolist()

    num_gts_to_plot = len(indices_to_plot)
    # --- 修改结束 ---

    # 使用新的、有限的GT数量来创建figure
    fig, axes = plt.subplots(num_gts_to_plot, 1, figsize=(10, 4 * num_gts_to_plot), squeeze=False)
    fig.suptitle(f'图片 {image_idx} - Top-{TOPK} Anchor IoU 分布 (随机采样 {num_gts_to_plot}/{num_valid_gts} GTs)',
                 fontsize=16)

    for i, gt_idx in enumerate(indices_to_plot):
        ax = axes[i, 0]

        # 获取当前 GT 对应的 align_metric 和 overlaps
        gt_align_metric = align_metric[image_idx, gt_idx, :]
        gt_overlaps = overlaps[image_idx, gt_idx, :]

        # 找到 align_metric 最高的 top-k 个 anchors
        _, topk_indices = torch.topk(gt_align_metric, TOPK)
        topk_ious = gt_overlaps[topk_indices].numpy()

        # 找到最终被选为正样本的 anchors 的 IoU
        final_positive_mask = mask_pos[image_idx, gt_idx, :].bool()
        final_positive_ious = gt_overlaps[final_positive_mask].numpy()

        ax.hist(topk_ious, bins=20, range=(0, 1), alpha=0.7, label=f'Top-{TOPK} Metric Anchors (共 {len(topk_ious)}个)')
        if len(final_positive_ious) > 0:
            ax.hist(final_positive_ious, bins=20, range=(0, 1), alpha=0.9,
                    label=f'最终匹配的正样本 (共 {len(final_positive_ious)}个)', color='orange')

        ax.set_title(f'GT #{gt_idx} (类别: {gt_labels_img[gt_idx].item()})')
        ax.set_xlabel('IoU (Overlap)')
        ax.set_ylabel('Anchor 数量')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    # 仅在有子图时调用 tight_layout
    if num_gts_to_plot > 0:
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    plt.show()

# def visualize_topk_overlaps(data, image_idx=0):
#     """
#     可视化 Top-K anchors 的 IoU 分布。
#     对于指定图片中的每一个 GT，绘制其 align_metric 最高的 top-k 个 anchors 的 IoU 分布。
#     """
#     print(f"\n--- 正在生成图片 {image_idx} 的 Top-K IoU 分布图 ---")
#
#     # 从数据中解包
#     align_metric = data['align_metric']  # (b, n_max_boxes, num_anchors)
#     overlaps = data['overlaps']  # (b, n_max_boxes, num_anchors)
#     gt_labels = data['gt_labels']  # (b, n_max_boxes, 1)
#     mask_pos = data['mask_pos']  # (b, n_max_boxes, num_anchors)
#
#     # 选择指定图片的数据
#     gt_labels_img = gt_labels[image_idx].squeeze(-1)  # (n_max_boxes,)
#     valid_gt_mask = gt_labels_img != -1  # 过滤掉 padding 的 GT
#     num_valid_gts = valid_gt_mask.sum().item()
#
#     if num_valid_gts == 0:
#         print(f"图片 {image_idx} 中没有有效的 GT Box，跳过可视化。")
#         return
#
#     fig, axes = plt.subplots(num_valid_gts, 1, figsize=(10, 4 * num_valid_gts), squeeze=False)
#     fig.suptitle(f'图片 {image_idx} 中各 GT 的 Top-{TOPK} Anchor IoU 分布', fontsize=16)
#
#     valid_gt_indices = torch.where(valid_gt_mask)[0]
#
#     for i, gt_idx in enumerate(valid_gt_indices):
#         ax = axes[i, 0]
#
#         # 获取当前 GT 对应的 align_metric 和 overlaps
#         # Shape: (num_anchors,)
#         gt_align_metric = align_metric[image_idx, gt_idx, :]
#         gt_overlaps = overlaps[image_idx, gt_idx, :]
#
#         # 找到 align_metric 最高的 top-k 个 anchors
#         # 注意：这里我们不考虑 mask_in_gts，因为 align_metric 已经隐式处理了
#         # 值为0的区域（不在gt内的anchor，其align_metric为0）
#         _, topk_indices = torch.topk(gt_align_metric, TOPK)
#         topk_ious = gt_overlaps[topk_indices].numpy()
#
#         # 找到最终被选为正样本的 anchors 的 IoU
#         # mask_pos 是经过 topk 和 in_gts 筛选后的最终正样本 mask
#         final_positive_mask = mask_pos[image_idx, gt_idx, :].bool()
#         final_positive_ious = gt_overlaps[final_positive_mask].numpy()
#
#         ax.hist(topk_ious, bins=20, range=(0, 1), alpha=0.7, label=f'Top-{TOPK} Metric Anchors (共 {len(topk_ious)}个)')
#         if len(final_positive_ious) > 0:
#             ax.hist(final_positive_ious, bins=20, range=(0, 1), alpha=0.9,
#                     label=f'最终匹配的正样本 (共 {len(final_positive_ious)}个)', color='orange')
#
#         ax.set_title(f'GT #{gt_idx.item()} (类别: {gt_labels_img[gt_idx].item()})')
#         ax.set_xlabel('IoU (Overlap)')
#         ax.set_ylabel('Anchor 数量')
#         ax.set_xlim(0, 1)
#         ax.legend()
#         ax.grid(axis='y', linestyle='--', alpha=0.7)
#
#     plt.tight_layout(rect=[0, 0.03, 1, 0.96])
#     plt.show()


def visualize_matched_bboxes(data, image_idx=0, canvas_size=(640, 640)):
    """
    在空白画布上可视化匹配的 GT 和 Pred BBox。
    """
    print(f"\n--- 正在生成图片 {image_idx} 的 BBox 匹配可视化图 ---")

    # 从数据中解包
    gt_bboxes = data['gt_bboxes']  # (b, n_max_boxes, 4)
    pd_bboxes = data['pd_bboxes']  # (b, num_anchors, 4)
    fg_mask = data['fg_mask']  # (b, num_anchors)
    target_gt_idx = data['target_gt_idx']  # (b, num_anchors)
    gt_labels = data['gt_labels']  # (b, n_max_boxes, 1)

    # 选择指定图片的数据
    gt_bboxes_img = gt_bboxes[image_idx]
    pd_bboxes_img = pd_bboxes[image_idx]
    fg_mask_img = fg_mask[image_idx]
    target_gt_idx_img = target_gt_idx[image_idx]
    gt_labels_img = gt_labels[image_idx].squeeze(-1)

    # 找到所有前景 anchors (即被分配了任务的 anchors)
    fg_indices = fg_mask_img.nonzero().squeeze(-1)

    if len(fg_indices) == 0:
        print(f"图片 {image_idx} 中没有匹配到任何正样本，跳过可视化。")
        return

    # 获取这些前景 anchors 对应的 pred_bbox 和它们被分配到的 gt_bbox
    matched_pd_bboxes = pd_bboxes_img[fg_indices]
    assigned_gt_indices = target_gt_idx_img[fg_indices]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f'图片 {image_idx} - GT BBox (绿色) vs 匹配的 Pred BBox (红色)')
    ax.set_xlim(0, canvas_size[0])
    ax.set_ylim(canvas_size[1], 0)  # y轴反转，使左上角为(0,0)

    # 创建图例代理
    legend_patches = [
        patches.Patch(color='green', label='Ground Truth Box'),
        patches.Patch(color='red', linestyle='--', label='Matched Predicted Box')
    ]

    # 随机颜色映射，方便区分不同的 GT
    unique_gt_ids = torch.unique(assigned_gt_indices)
    colors = {gt_id.item(): (random.random(), random.random(), random.random()) for gt_id in unique_gt_ids}

    # 绘制 GT BBox
    for gt_id in unique_gt_ids:
        gt_id = gt_id.item()

        # 检查是否是有效的 GT
        if gt_labels_img[gt_id] == -1:
            continue

        color = colors[gt_id]
        gt_box = gt_bboxes_img[gt_id].numpy()
        x1, y1, x2, y2 = gt_box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2.5, edgecolor='green', facecolor='none',
            label=f'GT {gt_id}'
        )
        ax.add_patch(rect)
        ax.text(x1, y1 - 5, f'GT-{gt_id}', color='green', fontsize=12, weight='bold')

    # 绘制匹配的 Pred BBox
    for i in range(len(fg_indices)):
        pd_box = matched_pd_bboxes[i].numpy()
        gt_id = assigned_gt_indices[i].item()

        if gt_labels_img[gt_id] == -1:
            continue

        color = colors[gt_id]
        x1, y1, x2, y2 = pd_box
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--'
        )
        ax.add_patch(rect)

    ax.legend(handles=legend_patches)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


if __name__ == '__main__':
    if not os.path.exists(RECORD_DIR):
        print(f"错误: 记录目录 '{RECORD_DIR}' 不存在。请检查路径或运行训练以生成记录。")
    else:
        # 获取所有保存的 pkl 文件
        record_files = sorted(list(Path(RECORD_DIR).glob('*.pkl')), key=lambda p: int(p.stem))

        if not record_files:
            print(f"错误: 在 '{RECORD_DIR}' 中没有找到任何 .pkl 文件。")
        else:
            # --- 用户选择 ---
            # 这里默认分析第一个文件和batch中的第一张图
            # 你可以修改这里的逻辑来分析特定的文件或图片
            target_file = record_files[-2]
            image_to_analyze = 0

            # 加载数据
            assignment_data = load_record(target_file)

            # --- 执行可视化 ---
            # 1. 可视化 top-k overlaps 分布
            visualize_topk_overlaps(assignment_data, image_idx=image_to_analyze)

            # 2. 可视化匹配的 bboxes
            # 注意: 你的bbox坐标需要是绝对像素坐标才能正确显示。
            # 如果是归一化的(0-1)，画布尺寸可以设为(1,1)，但看起来会很小。
            # 这里假设是 640x640 像素坐标。
            visualize_matched_bboxes(assignment_data, image_idx=image_to_analyze, canvas_size=(640, 640))