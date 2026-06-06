#!/usr/bin/env python3
"""
Soft-label assignment overlay visualization for small object detection papers.

Compares baseline (stock TAL) vs ours (calibration assigner) on a single image.
Renders a three-panel figure: [image crop] | [baseline overlay] | [ours overlay]
showing the P3 (80x80) grid with soft-label assignment masks superimposed.

Usage example
-------------
python visualize_assign_overlay.py \
    --baseline_ckpt runs/train/baseline/weights/best.pt \
    --ours_ckpt     runs/train/ours/weights/best.pt \
    --image         data/coco/images/val2017/000000000139.jpg \
    --labels        data/coco/labels/val2017/000000000139.txt \
    --crop 100 100 500 500 \
    --class_colors '{"0":0,"1":60,"3":120}' \
    --out           output/assign_overlay.pdf
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")  # MUST come before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

# ── Ultralytics imports ───────────────────────────────────────────────────────
from ultralytics.utils.tal import TaskAlignedAssigner, make_anchors, dist2bbox
from ultralytics.utils.mla_usaa import (
    TaskAlignedAssigner_dyab_dmetric_dscale,
    TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea,
)

# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

# Hue angles (0–360) for up to 80 COCO classes
DEFAULT_CLASS_HUES: dict[str, float] = {
    "0": 0, "1": 30, "2": 60, "3": 90, "4": 120, "5": 150,
    "6": 180, "7": 210, "8": 240, "9": 270, "10": 300, "11": 330,
}
# Extend to 80 classes with a rotating palette
for _i in range(12, 80):
    DEFAULT_CLASS_HUES[str(_i)] = (_i * 37) % 360

# LetterBox / model input size
IMGSZ = 640
STRIDES = [8, 16, 32]
EXPECTED_ANCHOR_SPLITS = [6400, 1600, 400]  # P3, P4, P5

# Level → (index in feats list, grid size, cell size in pixels, stride)
LEVEL_CONFIG: dict[str, dict] = {
    "P3": {"idx": 0, "grid": 80, "cell": 8, "stride": 8},
    "P4": {"idx": 1, "grid": 40, "cell": 16, "stride": 16},
    "P5": {"idx": 2, "grid": 20, "cell": 32, "stride": 32},
}

# Default rendering style — adjust these to taste, or override via CLI
DEFAULT_GRID_STYLE: dict = {"color": "gray", "alpha": 0.15, "linewidth": 0.3}
DEFAULT_GT_STYLE: dict = {"edgecolor": "white", "linewidth": 1.2, "linestyle": "-"}

# ──────────────────────────────────────────────────────────────────────────────
# Helper: hue → RGBA
# ──────────────────────────────────────────────────────────────────────────────

def hsv_to_rgba(h: float, s: float = 0.85, v: float = 0.95, alpha: float = 1.0) -> tuple:
    """Convert HSV (h in [0,360], s,v in [0,1]) to RGBA tuple (0-1)."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return (r, g, b, alpha)


# ──────────────────────────────────────────────────────────────────────────────
# LetterBox
# ──────────────────────────────────────────────────────────────────────────────

def letterbox_image(
    img_bgr: np.ndarray,
    new_shape: tuple[int, int] = (IMGSZ, IMGSZ),
    padding_value: int = 114,
) -> tuple[np.ndarray, float, float, float]:
    """
    Resize + pad image to *new_shape* keeping aspect ratio (standard YOLO letterbox).

    Returns
    -------
    img_out : np.ndarray  (H,W,3) uint8, shape = new_shape
    r       : float        scale ratio (new / old)
    pad_left: float        left padding in pixels
    pad_top : float        top padding in pixels

    All GT boxes should be transformed as:
        x_lb = x_orig * r + pad_left
        y_lb = y_orig * r + pad_top
    """
    h0, w0 = img_bgr.shape[:2]
    r = min(new_shape[0] / h0, new_shape[1] / w0)  # scale ratio
    new_unpad_w = int(round(w0 * r))
    new_unpad_h = int(round(h0 * r))
    dw = new_shape[1] - new_unpad_w
    dh = new_shape[0] - new_unpad_h
    # center
    pad_left = dw / 2.0
    pad_top = dh / 2.0

    if (new_unpad_w, new_unpad_h) != (w0, h0):
        img_bgr = cv2.resize(img_bgr, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(pad_top - 0.1))
    bottom = int(round(pad_top + 0.1))
    left = int(round(pad_left - 0.1))
    right = int(round(pad_left + 0.1))
    img_out = cv2.copyMakeBorder(
        img_bgr, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(padding_value,) * 3,
    )
    return img_out, r, pad_left, pad_top


# ──────────────────────────────────────────────────────────────────────────────
# Label parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_yolo_labels(label_path: str, img_h: int, img_w: int) -> list[dict]:
    """
    Parse YOLO-format labels (class_id cx cy w h, all normalised 0–1).
    Returns list of {cls, x1, y1, x2, y2} in original image *pixel* coordinates.
    """
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx = float(parts[1]) * img_w
            cy = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            x1 = cx - w / 2.0
            y1 = cy - h / 2.0
            x2 = cx + w / 2.0
            y2 = cy + h / 2.0
            boxes.append({"cls": cls_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return boxes


# ──────────────────────────────────────────────────────────────────────────────
# Model helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_detection_model(ckpt_path: str, device: str):
    """
    Load a YOLO checkpoint and return the DetectionModel in training mode.
    We use the ultralytics YOLO API to load and extract the raw nn.Module.
    """
    from ultralytics import YOLO
    yolo_model = YOLO(ckpt_path).to(device)
    detection_model = yolo_model.model  # DetectionModel
    detection_model.train()  # set to training mode so Detect head returns raw feats
    # Freeze all params (we only do forward, no backward)
    for p in detection_model.parameters():
        p.requires_grad_(False)
    return detection_model


def get_head_info(detection_model) -> dict:
    """Extract head-level metadata from the Detect module."""
    head = detection_model.model[-1]  # Detect / DetectWithSubnet
    return {
        "nc": head.nc,
        "reg_max": head.reg_max,
        "no": head.no,
        "stride": head.stride,
        "nl": head.nl,
    }


def forward_features(detection_model, img_tensor: torch.Tensor):
    """
    Run the model in training mode and return the list of head feature maps
    (handles both standard Detect and DetectWithSubnet).
    """
    with torch.no_grad():
        preds = detection_model(img_tensor)
    if isinstance(preds, tuple):
        # DetectWithSubnet returns (param_outputs, feats)
        feats = preds[1]
    elif isinstance(preds, list):
        feats = preds
    else:
        raise TypeError(f"Unexpected model output type: {type(preds)}")
    return feats


# ──────────────────────────────────────────────────────────────────────────────
# DFL uncertainty (replicated from DetectionLoss_USAA)
# ──────────────────────────────────────────────────────────────────────────────

def compute_dfl_uncertainty(
    pred_distri: torch.Tensor,  # (bs, na, 4*reg_max)
    reg_max: int,
    proj: torch.Tensor,         # (reg_max,)
) -> torch.Tensor:
    """
    Per-anchor DFL variance, averaged over 4 coordinates.
    Returns (bs, na).
    """
    bs, na = pred_distri.shape[:2]
    dfl_probs = pred_distri.view(bs, na, 4, reg_max).detach().softmax(-1)
    bins = proj.view(1, 1, 1, -1)  # (1, 1, 1, reg_max)
    e_x = (dfl_probs * bins).sum(-1)     # (bs, na, 4)
    e_x2 = (dfl_probs * bins.pow(2)).sum(-1)
    var = e_x2 - e_x.pow(2)
    return var.mean(-1)  # (bs, na)


# ──────────────────────────────────────────────────────────────────────────────
# Assigner factory
# ──────────────────────────────────────────────────────────────────────────────

def build_assigner(assigner_type: str, nc: int, assigner_kwargs: dict):
    """
    Instantiate an assigner by name, mirroring get_task_aligned_assigner().
    """
    base_kwargs = dict(
        topk=assigner_kwargs.pop("topk", 10),
        num_classes=nc,
        alpha=assigner_kwargs.pop("alpha", 1.0),
        beta=assigner_kwargs.pop("beta", 6.0),
        eps=assigner_kwargs.pop("eps", 1e-9),
    )
    if assigner_type == "TaskAlignedAssigner":
        return TaskAlignedAssigner(**base_kwargs)

    elif assigner_type == "TaskAlignedAssigner_dyab_dmetric_dscale":
        base_kwargs.update(assigner_kwargs)
        return TaskAlignedAssigner_dyab_dmetric_dscale(**base_kwargs)

    elif assigner_type == "TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea":
        # Pull RefineArea-specific kwargs
        base_kwargs["r_ref"] = assigner_kwargs.pop("r_ref", 32.0)
        base_kwargs["r_ref_type"] = assigner_kwargs.pop("r_ref_type", "pow")
        base_kwargs["r_ref_use_adaptive"] = assigner_kwargs.pop("r_ref_use_adaptive", False)
        # Pull dyab/dmetric/dscale kwargs
        base_kwargs["dyab_type"] = assigner_kwargs.pop("dyab_type", "DyabLinearFusion")
        base_kwargs["dyab_kwargs"] = assigner_kwargs.pop("dyab_kwargs", {})
        base_kwargs["overlap_iou_type"] = assigner_kwargs.pop("overlap_iou_type", "CIoU")
        base_kwargs["overlap_iou_kwargs"] = assigner_kwargs.pop("overlap_iou_kwargs", {})
        base_kwargs["align_iou_type"] = assigner_kwargs.pop("align_iou_type", "CIoU")
        base_kwargs["align_iou_kwargs"] = assigner_kwargs.pop("align_iou_kwargs", {})
        base_kwargs["score_iou_type"] = assigner_kwargs.pop("score_iou_type", "CIoU")
        base_kwargs["score_iou_kwargs"] = assigner_kwargs.pop("score_iou_kwargs", {})
        base_kwargs["dscale_func"] = assigner_kwargs.pop("dscale_func", "static")
        base_kwargs["scale_ratio"] = assigner_kwargs.pop("scale_ratio", 1.0)
        # Remainder
        base_kwargs.update(assigner_kwargs)
        return TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea(**base_kwargs)

    else:
        raise ValueError(f"Unknown assigner_type: {assigner_type}")


# ──────────────────────────────────────────────────────────────────────────────
# Core: run assignment for a single model
# ──────────────────────────────────────────────────────────────────────────────

def run_assignment(
    detection_model,
    assigner,
    img_letterbox: np.ndarray,       # (640, 640, 3) uint8
    gt_boxes_lb: list[dict],         # [{cls, x1,y1,x2,y2}] in letterbox coords
    device: torch.device,
    need_uncertainty: bool = False,
) -> dict:
    """
    Run the full forward + assign pipeline for one model/image pair.

    Returns a dict with:
        target_scores    : (1, total_anchors, nc)  — raw soft labels
        fg_mask          : (1, total_anchors) bool
        target_gt_idx    : (1, total_anchors) int
        anchor_points    : (total_anchors, 2)       — in letterbox pixels
        stride_tensor    : (total_anchors, 1)
        na_per_level     : [6400, 1600, 400]
    """
    head_info = get_head_info(detection_model)
    nc = head_info["nc"]
    reg_max = head_info["reg_max"]
    no = head_info["no"]
    stride = head_info["stride"]  # e.g. tensor([8., 16., 32.])

    # 1) Preprocess image tensor
    img_tensor = torch.from_numpy(img_letterbox).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device)

    # 2) Forward pass → head features
    feats = forward_features(detection_model, img_tensor)
    bs = feats[0].shape[0]

    # 3) Concatenate multi-scale features (same as loss.py)
    pred_distri, pred_scores = torch.cat(
        [xi.view(bs, no, -1) for xi in feats], dim=2
    ).split((reg_max * 4, nc), dim=1)

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()  # (bs, na, nc)
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()  # (bs, na, 4*reg_max)

    # 4) Compute anchors
    anchor_points, stride_tensor = make_anchors(feats, stride.tolist(), 0.5)
    # anchor_points: (na, 2) in feature-map coords (e.g. 0.5, 1.5, ...)
    # stride_tensor: (na, 1) values = 8, 16, 32

    # 5) Decode bboxes (DFL softmax → projection → dist2bbox, same as loss.bbox_decode)
    proj = torch.arange(reg_max, dtype=torch.float, device=device)
    _na = pred_distri.shape[1]
    _c = pred_distri.shape[2]
    decoded_bbox = pred_distri.view(bs, _na, 4, _c // 4).softmax(-1).matmul(proj.to(pred_distri.dtype))
    pred_bboxes = dist2bbox(decoded_bbox, anchor_points, xywh=False)  # (bs, na, 4) in feat-map coords

    # 6) Prepare targets
    # Build batch dict from single image
    n_gt = len(gt_boxes_lb)
    if n_gt == 0:
        # No GT — return zeros
        return {
            "target_scores": torch.zeros(1, anchor_points.shape[0], nc, device=device),
            "fg_mask": torch.zeros(1, anchor_points.shape[0], dtype=torch.bool, device=device),
            "target_gt_idx": torch.zeros(1, anchor_points.shape[0], dtype=torch.long, device=device),
            "anchor_points": anchor_points,
            "stride_tensor": stride_tensor,
            "na_per_level": [f.shape[2] * f.shape[3] for f in feats],
        }

    gt_labels_list = []
    gt_bboxes_list = []
    for gt in gt_boxes_lb:
        gt_labels_list.append(gt["cls"])
        gt_bboxes_list.append([gt["x1"], gt["y1"], gt["x2"], gt["y2"]])

    # Pad to uniform size (in loss.py, targets are padded to max number across batch)
    gt_labels = torch.tensor(gt_labels_list, dtype=torch.float32, device=device).view(1, n_gt, 1)
    gt_bboxes = torch.tensor(gt_bboxes_list, dtype=torch.float32, device=device).view(1, n_gt, 4)
    mask_gt = torch.ones(1, n_gt, 1, device=device)

    # 7) Compute uncertainty if needed
    u_raw = None
    if need_uncertainty:
        u_raw = compute_dfl_uncertainty(pred_distri, reg_max, proj)

    # 8) Build stride tensor for dscale (if assigner uses stride)
    # The USAA-series assigners expect stride as (bs, n_max_boxes, na) negative values
    _stride_tensor = stride_tensor.clone().squeeze(-1).unsqueeze(0).unsqueeze(0)  # (1, 1, na)
    _stride_tensor = _stride_tensor.expand(1, n_gt, -1).to(device=device, dtype=gt_bboxes.dtype)

    # 9) Call assigner
    # Prepare inputs in correct format
    pd_scores_sig = pred_scores.detach().sigmoid()
    pd_bboxes_scaled = (pred_bboxes.detach() * stride_tensor).to(gt_bboxes.dtype)
    anc_points_scaled = anchor_points * stride_tensor

    if need_uncertainty:
        # USAA assigner: forward(pd_scores, pd_bboxes, uncertainty, anc_points, gt_labels, gt_bboxes, mask_gt, stride=...)
        result = assigner(
            pd_scores_sig,
            pd_bboxes_scaled,
            u_raw,
            anc_points_scaled,
            gt_labels,
            gt_bboxes,
            mask_gt,
            stride=-_stride_tensor,
        )
    else:
        # Stock TAL: forward(pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, stride=None, **kwargs)
        result = assigner(
            pd_scores_sig,
            pd_bboxes_scaled,
            anc_points_scaled,
            gt_labels,
            gt_bboxes,
            mask_gt,
            stride=None,
        )

    target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = result

    return {
        "target_scores": target_scores,      # (1, na, nc)
        "fg_mask": fg_mask,                   # (1, na) bool
        "target_gt_idx": target_gt_idx,       # (1, na)
        "anchor_points": anchor_points,       # (na, 2) in feature-map coords
        "stride_tensor": stride_tensor,       # (na, 1)
        "na_per_level": [f.shape[2] * f.shape[3] for f in feats],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Level grid extraction (P3 / P4 / P5)
# ──────────────────────────────────────────────────────────────────────────────

def extract_level_grid(
    target_scores: torch.Tensor,  # (1, na, nc)
    na_per_level: list[int],
    level_idx: int,              # 0=P3, 1=P4, 2=P5
    grid_size: int,              # 80 / 40 / 20
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract one feature level's soft-label mask.

    Returns
    -------
    cls_map  : (grid, grid) int   — argmax class per cell (-1 for bg)
    val_map  : (grid, grid) float — max soft-label value per cell
    """
    total = sum(na_per_level)
    n_level = na_per_level[level_idx]
    assert n_level == grid_size * grid_size, \
        f"Level {level_idx}: anchors={n_level}, expected {grid_size*grid_size}"
    assert total == sum(EXPECTED_ANCHOR_SPLITS), \
        f"Total anchors={total}, expected {sum(EXPECTED_ANCHOR_SPLITS)} (splits={na_per_level})"

    # Offset into the concatenated anchor list
    start_idx = sum(na_per_level[:level_idx])
    end_idx = start_idx + n_level

    level_scores = target_scores[0, start_idx:end_idx, :]  # (n_level, nc)
    vals, cls_ids = level_scores.max(dim=-1)  # (n_level,)
    vals = vals.cpu().numpy().reshape(grid_size, grid_size)
    cls_ids = cls_ids.cpu().numpy().reshape(grid_size, grid_size)

    cls_map = np.where(vals > 0, cls_ids.astype(int), -1)
    val_map = vals

    return cls_map, val_map


# ──────────────────────────────────────────────────────────────────────────────
# Rendering
# ──────────────────────────────────────────────────────────────────────────────

def render_overlay_panel(
    ax: plt.Axes,
    img_640: np.ndarray,           # (640, 640, 3) uint8 BGR
    cls_map: np.ndarray,            # (grid, grid) int, -1 = bg
    val_map: np.ndarray,            # (grid, grid) float in [0, 1]
    gt_boxes_lb: list[dict],       # GT boxes in letterbox (640) coords
    class_hues: dict,              # cls_id → hue angle (0–360)
    crop_region: tuple,            # (x0, y0, x1, y1) in letterbox coords
    title: str = "",
    draw_grid: bool = True,
    draw_gt: bool = True,
    grid_size: int = 80,           # cells per dimension (80/40/20 for P3/P4/P5)
    cell_size: int = 8,            # pixel size of one cell (8/16/32)
    grid_style: dict | None = None,
    gt_style: dict | None = None,
):
    """
    Draw one panel: image + assignment overlay + GT boxes + grid.
    Renders at 640×640 then crops.

    Style dicts accept matplotlib Rectangle / axhline keywords, e.g.:
      grid_style = {"color": "gray", "alpha": 0.15, "linewidth": 0.3}
      gt_style   = {"edgecolor": "white", "linewidth": 1.2, "linestyle": "-"}
    """
    if grid_style is None:
        grid_style = DEFAULT_GRID_STYLE
    if gt_style is None:
        gt_style = DEFAULT_GT_STYLE

    x0, y0, x1, y1 = crop_region

    # Convert BGR to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)

    # Start with the image
    ax.imshow(img_rgb, extent=(0, 640, 640, 0), interpolation="bilinear")  # top-left origin

    # ── Overlay assignment mask ───────────────────────────────────────────
    # Create an RGBA overlay of same size as image
    overlay_rgba = np.zeros((640, 640, 4), dtype=np.float32)
    for gy in range(grid_size):
        for gx in range(grid_size):
            c = cls_map[gy, gx]
            v = float(val_map[gy, gx])
            if c < 0 or v <= 0:
                continue
            c_str = str(int(c))
            hue = class_hues.get(c_str, (int(c) * 37) % 360)
            rgba = hsv_to_rgba(hue, s=0.85, v=0.95, alpha=v)
            y_start = gy * cell_size
            y_end = y_start + cell_size
            x_start = gx * cell_size
            x_end = x_start + cell_size
            overlay_rgba[y_start:y_end, x_start:x_end] = rgba

    ax.imshow(overlay_rgba, extent=(0, 640, 640, 0), interpolation="nearest")

    # ── Draw grid lines ────────────────────────────────────────────────────
    if draw_grid:
        for i in range(0, 640 + 1, cell_size):
            ax.axhline(i, **grid_style)
            ax.axvline(i, **grid_style)

    # ── Draw GT boxes ──────────────────────────────────────────────────────
    if draw_gt:
        for gt in gt_boxes_lb:
            w = gt["x2"] - gt["x1"]
            h = gt["y2"] - gt["y1"]
            rect = plt.Rectangle(
                (gt["x1"], gt["y1"]), w, h,
                fill=False, **gt_style,
            )
            ax.add_patch(rect)

    # ── Crop ──────────────────────────────────────────────────────────────
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # flip y for image coordinates
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.axis("off")


def render_legend(
    ax: plt.Axes,
    class_hues: dict,
    active_classes: set | None = None,
    class_names: dict | None = None,
):
    """
    Draw a class → colour legend in the given axis.

    If ``active_classes`` is provided, only those class ids are shown
    (e.g. the classes actually present in the image GT).
    ``class_names``: cls_id → name string (optional, fallback to "cls_{id}").
    """
    ax.axis("off")
    ax.set_title("Class Legend", fontsize=8, fontweight="bold")

    items = sorted(class_hues.items(), key=lambda x: int(x[0]))
    if active_classes is not None:
        items = [(k, v) for k, v in items if int(k) in active_classes]

    n = len(items)
    if n == 0:
        ax.text(0.5, 0.5, "(no GT)", transform=ax.transAxes,
                fontsize=7, ha="center", va="center")
        return

    for i, (cls_str, hue) in enumerate(items):
        y = 1.0 - (i + 1) / (n + 1)
        rgba = hsv_to_rgba(hue, s=0.85, v=0.95)
        ax.add_patch(plt.Rectangle((0.05, y - 0.02), 0.1, 0.04,
                                     facecolor=rgba, edgecolor="gray", linewidth=0.5,
                                     transform=ax.transAxes))
        cls_id = int(cls_str)
        name = class_names.get(cls_id, f"cls_{cls_id}") if class_names else f"cls_{cls_id}"
        ax.text(0.18, y, name, transform=ax.transAxes, fontsize=6, verticalalignment="center")


def render_colorbar(fig, cmap_name: str = "viridis"):
    """Add a shared continuous 0→1 colorbar (for soft-label value reference)."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cm.get_cmap(cmap_name), norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("soft-label score (α)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Soft-label assignment overlay visualization for small-object detection papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Checkpoints
    parser.add_argument("--baseline_ckpt", type=str, required=True,
                        help="Path to baseline (stock TAL) checkpoint")
    parser.add_argument("--ours_ckpt", type=str, required=True,
                        help="Path to ours (calibration) checkpoint")
    # Image
    parser.add_argument("--image", type=str, required=True,
                        help="Path to input image (for visualization)")
    parser.add_argument("--labels", type=str, required=True,
                        help="Path to GT labels (YOLO txt format)")
    # Crop (pixel coords in letterbox 640 frame, multiples of 8)
    parser.add_argument("--crop", type=int, nargs=4, default=[0, 0, 640, 640],
                        metavar=("X0", "Y0", "X1", "Y1"),
                        help="Crop region in letterbox frame (default: 0 0 640 640)")
    # Assigner configuration
    parser.add_argument("--baseline_assigner_type", type=str,
                        default="TaskAlignedAssigner",
                        help="Assigner class name for baseline (default: TaskAlignedAssigner)")
    parser.add_argument("--ours_assigner_type", type=str,
                        default="TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea",
                        help="Assigner class name for ours (default: TaskAlignedAssigner_dyab_dmetric_dscale_RefineArea)")
    parser.add_argument("--baseline_assigner_kwargs", type=str, default="{}",
                        help="JSON string of extra kwargs for baseline assigner")
    parser.add_argument("--ours_assigner_kwargs", type=str, default="{}",
                        help="JSON string of extra kwargs for ours assigner")
    # Rendering
    parser.add_argument("--level", type=str, default="P3", choices=["P3", "P4", "P5"],
                        help="Feature level to visualize (default: P3)")
    parser.add_argument("--class_colors", type=str, default=None,
                        help="JSON string: cls_id → hue angle, e.g. '{\"0\":0,\"1\":60}'")
    # Grid line style
    parser.add_argument("--grid_color", type=str, default=DEFAULT_GRID_STYLE["color"],
                        help=f"Grid line color (default: {DEFAULT_GRID_STYLE['color']})")
    parser.add_argument("--grid_alpha", type=float, default=DEFAULT_GRID_STYLE["alpha"],
                        help=f"Grid line alpha (default: {DEFAULT_GRID_STYLE['alpha']})")
    parser.add_argument("--grid_lw", type=float, default=DEFAULT_GRID_STYLE["linewidth"],
                        help=f"Grid line width (default: {DEFAULT_GRID_STYLE['linewidth']})")
    # GT box style
    parser.add_argument("--gt_color", type=str, default=DEFAULT_GT_STYLE["edgecolor"],
                        help=f"GT box edge color (default: {DEFAULT_GT_STYLE['edgecolor']})")
    parser.add_argument("--gt_lw", type=float, default=DEFAULT_GT_STYLE["linewidth"],
                        help=f"GT box line width (default: {DEFAULT_GT_STYLE['linewidth']})")
    parser.add_argument("--gt_ls", type=str, default=DEFAULT_GT_STYLE["linestyle"],
                        help=f"GT box linestyle (default: {DEFAULT_GT_STYLE['linestyle']})")
    # Output
    parser.add_argument("--out", type=str, default="assign_overlay.pdf",
                        help="Output path (PDF recommended)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device: 'cuda', 'cpu', 'cuda:0', etc.")
    parser.add_argument("--dpi", type=int, default=200,
                        help="Output DPI (default: 200)")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ── Level config ─────────────────────────────────────────────────────────
    lvl = LEVEL_CONFIG[args.level]
    level_idx = lvl["idx"]
    grid_size = lvl["grid"]
    cell_size = lvl["cell"]
    print(f"[INFO] Feature level: {args.level}  (grid={grid_size}×{grid_size}, cell={cell_size}px)")

    # ── Style config ─────────────────────────────────────────────────────────
    grid_style = {"color": args.grid_color, "alpha": args.grid_alpha, "linewidth": args.grid_lw}
    # grid_style = {"color": "#1B1C3F", "alpha": args.grid_alpha, "linewidth": args.grid_lw}
    gt_style = {"edgecolor": args.gt_color, "linewidth": args.gt_lw, "linestyle": args.gt_ls}

    # ── Parse class colours ───────────────────────────────────────────────
    if args.class_colors:
        class_hues = json.loads(args.class_colors)
    else:
        class_hues = dict(DEFAULT_CLASS_HUES)
    # Convert string keys to match usage pattern
    class_hues = {str(k): float(v) for k, v in class_hues.items()}

    # ── Parse assigner kwargs ─────────────────────────────────────────────
    baseline_assigner_kwargs = json.loads(args.baseline_assigner_kwargs) if args.baseline_assigner_kwargs else {}
    ours_assigner_kwargs = json.loads(args.ours_assigner_kwargs) if args.ours_assigner_kwargs else {}

    # ── Load image ────────────────────────────────────────────────────────
    print(f"[INFO] Loading image: {args.image}")
    img_orig = cv2.imread(args.image)
    if img_orig is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    h_orig, w_orig = img_orig.shape[:2]

    # ── Parse GT labels ───────────────────────────────────────────────────
    print(f"[INFO] Parsing labels: {args.labels}")
    gt_boxes_orig = parse_yolo_labels(args.labels, h_orig, w_orig)
    print(f"[INFO] Found {len(gt_boxes_orig)} GT boxes in original frame")

    # ── LetterBox transform ───────────────────────────────────────────────
    img_lb, scale_r, pad_l, pad_t = letterbox_image(img_orig, (IMGSZ, IMGSZ))
    # Transform GT boxes to letterbox frame
    gt_boxes_lb = []
    for gt in gt_boxes_orig:
        gt_lb = {
            "cls": gt["cls"],
            "x1": gt["x1"] * scale_r + pad_l,
            "y1": gt["y1"] * scale_r + pad_t,
            "x2": gt["x2"] * scale_r + pad_l,
            "y2": gt["y2"] * scale_r + pad_t,
        }
        gt_boxes_lb.append(gt_lb)

    # ── Active classes (only those present in the GT of this image) ──────────
    active_classes = {gt["cls"] for gt in gt_boxes_lb}
    print(f"[INFO] Active classes in this image: {sorted(active_classes)}")

    # ── Validate crop region ──────────────────────────────────────────────
    x0, y0, x1, y1 = args.crop
    for v, name in [(x0, "x0"), (y0, "y0"), (x1, "x1"), (y1, "y1")]:
        if v % 8 != 0:
            print(f"[WARN] Crop {name}={v} is not a multiple of 8 — snapping to nearest")
    x0 = (x0 // 8) * 8
    y0 = (y0 // 8) * 8
    x1 = ((x1 + 7) // 8) * 8
    y1 = ((y1 + 7) // 8) * 8
    x0 = max(0, min(x0, IMGSZ))
    y0 = max(0, min(y0, IMGSZ))
    x1 = max(0, min(x1, IMGSZ))
    y1 = max(0, min(y1, IMGSZ))
    if x1 <= x0:
        x1 = x0 + 8
    if y1 <= y0:
        y1 = y0 + 8
    crop_region = (x0, y0, x1, y1)
    print(f"[INFO] Crop region: {crop_region}")

    # ── Load models ───────────────────────────────────────────────────────
    print(f"[INFO] Loading baseline model: {args.baseline_ckpt}")
    model_base = load_detection_model(args.baseline_ckpt, str(device))
    head_info_base = get_head_info(model_base)
    nc = head_info_base["nc"]
    reg_max_base = head_info_base["reg_max"]
    print(f"[INFO] Baseline: nc={nc}, reg_max={reg_max_base}, stride={head_info_base['stride'].tolist()}")

    print(f"[INFO] Loading ours model: {args.ours_ckpt}")
    model_ours = load_detection_model(args.ours_ckpt, str(device))
    head_info_ours = get_head_info(model_ours)
    nc_ours = head_info_ours["nc"]
    reg_max_ours = head_info_ours["reg_max"]
    print(f"[INFO] Ours:     nc={nc_ours}, reg_max={reg_max_ours}, stride={head_info_ours['stride'].tolist()}")
    assert nc == nc_ours, f"Class count mismatch: {nc} vs {nc_ours}"

    # ── Build assigners ───────────────────────────────────────────────────
    print(f"[INFO] Baseline assigner: {args.baseline_assigner_type}")
    assigner_base = build_assigner(args.baseline_assigner_type, nc, baseline_assigner_kwargs)
    assigner_base.to(device)
    assigner_base.eval()

    print(f"[INFO] Ours assigner: {args.ours_assigner_type}")
    assigner_ours = build_assigner(args.ours_assigner_type, nc, ours_assigner_kwargs)
    assigner_ours.to(device)
    assigner_ours.eval()

    need_uncertainty_base = "TaskAlignedAssigner_dyab_dmetric_dscale" in args.baseline_assigner_type
    need_uncertainty_ours = "TaskAlignedAssigner_dyab_dmetric_dscale" in args.ours_assigner_type

    # ── Run assignment for both models ────────────────────────────────────
    print(f"[INFO] Running baseline assignment...")
    result_base = run_assignment(
        model_base, assigner_base, img_lb, gt_boxes_lb, device,
        need_uncertainty=need_uncertainty_base,
    )

    print(f"[INFO] Running ours assignment...")
    result_ours = run_assignment(
        model_ours, assigner_ours, img_lb, gt_boxes_lb, device,
        need_uncertainty=need_uncertainty_ours,
    )

    # ── SANITY CHECK: anchor splits ───────────────────────────────────────
    na_base = result_base["na_per_level"]
    na_ours = result_ours["na_per_level"]
    total_base = sum(na_base)
    total_ours = sum(na_ours)
    print(f"\n{'='*60}")
    print(f"[SANITY] Anchor splits:")
    print(f"  Baseline: P3={na_base[0]}, P4={na_base[1]}, P5={na_base[2]}  total={total_base}")
    print(f"  Ours:     P3={na_ours[0]}, P4={na_ours[1]}, P5={na_ours[2]}  total={total_ours}")
    print(f"  Expected: P3={EXPECTED_ANCHOR_SPLITS[0]}, P4={EXPECTED_ANCHOR_SPLITS[1]}, P5={EXPECTED_ANCHOR_SPLITS[2]}  total={sum(EXPECTED_ANCHOR_SPLITS)}")

    if na_base != EXPECTED_ANCHOR_SPLITS:
        print(f"[ERROR] Baseline anchor split does NOT match expected {EXPECTED_ANCHOR_SPLITS}! Stopping so you can verify fork differences.")
        sys.exit(1)
    if na_ours != EXPECTED_ANCHOR_SPLITS:
        print(f"[ERROR] Ours anchor split does NOT match expected {EXPECTED_ANCHOR_SPLITS}! Stopping so you can verify fork differences.")
        sys.exit(1)
    print(f"[SANITY] Anchor split check PASSED.")

    # ── Extract level grids ─────────────────────────────────────────────────
    cls_map_base, val_map_base = extract_level_grid(
        result_base["target_scores"], result_base["na_per_level"], level_idx, grid_size,
    )
    cls_map_ours, val_map_ours = extract_level_grid(
        result_ours["target_scores"], result_ours["na_per_level"], level_idx, grid_size,
    )

    # ── SANITY CHECK: per-GT stats ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[SANITY] Per-GT soft-label peak and positive cell counts")
    print(f"         (showing ALL anchors, not just {args.level}):")
    fg_base = result_base["fg_mask"][0]  # (na,)
    fg_ours = result_ours["fg_mask"][0]

    # Per-GT best positive soft-label peak (over ALL anchors, all levels)
    ts_base = result_base["target_scores"][0]  # (na, nc)
    ts_ours = result_ours["target_scores"][0]

    # Also compute per-level fg counts
    na_splits_base = result_base["na_per_level"]
    na_splits_ours = result_ours["na_per_level"]
    start = sum(na_splits_base[:level_idx])
    n_level_anchors = na_splits_base[level_idx]
    fg_level_base = fg_base[start:start + n_level_anchors].sum().item()
    fg_level_ours = fg_ours[start:start + n_level_anchors].sum().item()

    for i, gt in enumerate(gt_boxes_lb):
        gt_idx_base = result_base["target_gt_idx"][0]
        gt_idx_ours = result_ours["target_gt_idx"][0]

        fg_for_gt_base = (gt_idx_base == i) & fg_base
        fg_for_gt_ours = (gt_idx_ours == i) & fg_ours

        n_pos_base = fg_for_gt_base.sum().item()
        n_pos_ours = fg_for_gt_ours.sum().item()

        peak_base = ts_base[fg_for_gt_base].max().item() if n_pos_base > 0 else 0.0
        peak_ours = ts_ours[fg_for_gt_ours].max().item() if n_pos_ours > 0 else 0.0

        gt_w = gt["x2"] - gt["x1"]
        gt_h = gt["y2"] - gt["y1"]
        gt_area = gt_w * gt_h
        print(f"  GT[{i}] cls={gt['cls']}, size=({gt_w:.0f},{gt_h:.0f}) area={gt_area:.0f}: "
              f"baseline peak={peak_base:.4f} n_pos={n_pos_base}, "
              f"ours peak={peak_ours:.4f} n_pos={n_pos_ours}")

    print(f"\n  Total foreground cells ({args.level}): baseline={fg_level_base}, ours={fg_level_ours}")
    print(f"  Total foreground cells (all levels):  baseline={fg_base.sum().item()}, ours={fg_ours.sum().item()}")

    # ── SANITY: alpha mapping function ────────────────────────────────────
    print(f"\n[SANITY] Alpha mapping: both baseline and ours use identical linear clamp [0,1] (no floor).")
    print(f"{'='*60}\n")

    # ── Render ────────────────────────────────────────────────────────────
    print(f"[INFO] Rendering figure...")
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.25], wspace=0.04)

    # Panel 1: Image crop only
    ax_img = fig.add_subplot(gs[0])
    render_overlay_panel(
        ax_img, img_lb, cls_map_base, np.zeros_like(val_map_base),
        gt_boxes_lb, class_hues, crop_region,
        title="Image (crop)", draw_grid=False, draw_gt=True,
        grid_size=grid_size, cell_size=cell_size,
        grid_style=grid_style, gt_style=gt_style,
    )

    # Panel 2: Baseline overlay
    ax_base = fig.add_subplot(gs[1])
    render_overlay_panel(
        ax_base, img_lb, cls_map_base, val_map_base,
        gt_boxes_lb, class_hues, crop_region,
        title=f"Baseline (Stock TAL) [{args.level}]", draw_grid=True, draw_gt=True,
        grid_size=grid_size, cell_size=cell_size,
        grid_style=grid_style, gt_style=gt_style,
    )

    # Panel 3: Ours overlay
    ax_ours = fig.add_subplot(gs[2])
    render_overlay_panel(
        ax_ours, img_lb, cls_map_ours, val_map_ours,
        gt_boxes_lb, class_hues, crop_region,
        title=f"Ours (Calibration) [{args.level}]", draw_grid=True, draw_gt=True,
        grid_size=grid_size, cell_size=cell_size,
        grid_style=grid_style, gt_style=gt_style,
    )

    # Legend (only classes present in this image)
    ax_legend = fig.add_subplot(gs[3])
    render_legend(ax_legend, class_hues, active_classes=active_classes)

    # Colorbar for soft-label value reference
    render_colorbar(fig)

    # Save
    out_path = args.out
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.1)
    print(f"[INFO] Saved to: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
