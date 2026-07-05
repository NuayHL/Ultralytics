"""
Generate GT-area vs prediction-score scatter plots during YOLO validation.

Usage:
    python myutils/myval_area_statics.py

    Or import and use programmatically:
    from myutils.myval_area_statics import val_and_plot
    val_and_plot(model_path="...", data_path="...")
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

from ultralytics import YOLO
from area_score_val import get_area_score_validator


def val_area_score(
    model_path,
    data_path,
    name="val/area_score",
    batch=16,
    imgsz=640,
    conf=0.001,
    iou=0.6,
    device=None,
):
    """
    Run validation using the appropriate AreaScoreValidator (YOLO or RT-DETR)
    and return collected area-score data.

    Args:
        model_path: Path to the model weights (.pt file) or config (.yaml).
        data_path: Path to the dataset YAML config.
        name: Name for the validation run (save directory).
        batch: Batch size.
        imgsz: Input image size.
        conf: Confidence threshold for NMS.
        iou: IoU threshold for NMS.
        device: Device to run on (None for auto).

    Returns:
        tuple: (area_score_data, validator) where area_score_data is a list of
               dicts with keys: gt_area, pred_score, status, gt_class, pred_class.
    """
    model = YOLO(model_path)

    # Auto-select the correct validator class (DetectionValidator vs RTDETRValidator)
    ValidatorCls = get_area_score_validator(model)

    metrics = model.val(
        validator=ValidatorCls,
        data=data_path,
        name=name,
        batch=batch,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        split="val",
    )

    # model.val() doesn't expose the validator instance directly.
    # Access it via the class-level reference set during __init__.
    validator = ValidatorCls.last_instance
    data = validator.area_score_data

    # Auto-save pickle for later comparison / re-plot
    result_dir = Path(validator.save_dir)
    pkl_path = result_dir / "area_score_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump({"data": data, "names": getattr(validator, "names", None)}, f)
    print(f"Data saved to {pkl_path}")

    return data, validator


AREA_LABELS = {
    "original": ("GT Box Area (original image pixels)", "area_original", "pred_area_original"),
    "input": ("GT Box Area (model input pixels)", "area_input", "pred_area_input"),
    "pct": ("GT Box Area (% of image)", "area_pct", "pred_area_pct"),
}


def _get_area_key(area_key):
    """Resolve user-friendly shortcuts to the full dict key."""
    aliases = {"orig": "original", "inp": "input", "%": "pct"}
    area_key = aliases.get(area_key, area_key)
    if area_key not in AREA_LABELS:
        raise ValueError(f"Unknown area_key '{area_key}'. Choose from: {list(AREA_LABELS.keys())}")
    return area_key


def _apply_pub_style():
    """Apply publication-quality rcParams so plots are suitable for paper figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "legend.framealpha": 0.6,
        "legend.edgecolor": "0.5",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
    })


# Fixed axis limits — use the same range across all models for fair comparison
X_LIM_LOG = (1, 10**4)      # for "original" / "input" (pixel areas, log scale)
X_LIM_PCT = (0, 100)        # for "pct" (percentage, linear scale)
Y_LIM = (-0.02, 1.02)

# Object-size ranges for per-range calibration analysis (resized / input pixels)
# Based on COCO convention with extra granularity at the small end.
AREA_RANGES = [
    ("verytiny",  0,     8**2),     # [0, 64)
    ("tiny",      8**2,  16**2),    # [64, 256)
    ("small",     16**2, 32**2),    # [256, 1024)
    ("medium",    32**2, 96**2),    # [1024, 9216)
    ("large",     96**2, float("inf")),  # [9216, ∞)
]

# AITOD-style object-size ranges (side-length based, squared for area)
AREA_RANGES_AITOD = [
    ("0-8²",      0,     8**2),     # [0, 64)
    ("8²-16²",    8**2,  16**2),    # [64, 256)
    ("16²-32²",   16**2, 32**2),    # [256, 1024)
    ("32²+",      32**2, float("inf")),  # [1024, ∞)
]


def generate_geometric_ranges(start=10, base=2, n_bins=None, max_area=640**2):
    """
    Generate area ranges with geometrically increasing side-length thresholds.

    Side-length edges:  start,  start*base,  start*base²,  ...
    Area thresholds:    edge²

    Args:
        start:    First side-length edge in pixels (default 10).
        base:     Geometric ratio between successive side lengths (default 2).
        n_bins:   Number of bins. If None, auto-derive from max_area.
        max_area: Upper bound (pixels²) for the last edge.

    Returns:
        List of (label, lo_area, hi_area) tuples.
    """
    edges = [start]
    while edges[-1] ** 2 < max_area:
        edges.append(edges[-1] * base)
    if n_bins is not None:
        edges = edges[:n_bins + 1]

    ranges = []
    for i in range(len(edges) - 1):
        lo_side = edges[i]
        hi_side = edges[i + 1]
        ranges.append((f"{lo_side:.0f}-{hi_side:.0f}", lo_side ** 2, hi_side ** 2))
    return ranges


def compute_per_range_calibration(area_score_data, area_ranges=None, n_bins=10,
                                  area_field="area_input"):
    """
    Compute ECE / MCE / precision for each object-size range.

    Args:
        area_score_data: List of dict records.
        area_ranges: List of (name, lo, hi) tuples. Defaults to AREA_RANGES.
        n_bins: Bins for ECE within each range.
        area_field: Which area field to use for partitioning (default: area_input).

    Returns:
        list of dicts with keys: name, n_preds, n_tp, precision, ece, mce.
        Plus an "all" entry at the end.
    """
    if area_ranges is None:
        area_ranges = AREA_RANGES

    results = []
    for name, lo, hi in area_ranges:
        subset = [d for d in area_score_data
                  if d["status"] in ("TP", "FP")
                  and lo <= d.get(area_field, 0) < hi]
        n_preds = len(subset)
        n_tp = sum(1 for d in subset if d["status"] == "TP")
        precision = n_tp / n_preds if n_preds > 0 else 0.0

        cal = compute_calibration(subset, n_bins=n_bins)
        ece = cal["ece"] if cal else 0.0
        mce = cal["mce"] if cal else 0.0

        results.append({
            "name": name, "lo": lo, "hi": hi,
            "n_preds": n_preds, "n_tp": n_tp,
            "precision": precision, "ece": ece, "mce": mce,
        })

    # "all" aggregate
    all_preds = [d for d in area_score_data if d["status"] in ("TP", "FP")]
    n_all = len(all_preds)
    n_all_tp = sum(1 for d in all_preds if d["status"] == "TP")
    cal_all = compute_calibration(area_score_data, n_bins=n_bins)
    results.append({
        "name": "all", "lo": 0, "hi": float("inf"),
        "n_preds": n_all, "n_tp": n_all_tp,
        "precision": n_all_tp / n_all if n_all > 0 else 0.0,
        "ece": cal_all["ece"] if cal_all else 0.0,
        "mce": cal_all["mce"] if cal_all else 0.0,
    })

    return results


def print_per_range_calibration(per_range_results):
    """Print a per-area-range calibration table."""
    print(f"\n  --- Per-area-range breakdown (resized area) ---")
    header = f"  {'Range':>10s}  {'Preds':>8s}  {'TP':>7s}  {'Precision':>9s}  {'ECE':>8s}  {'MCE':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in per_range_results:
        print(f"  {r['name']:>10s}  {r['n_preds']:>8d}  {r['n_tp']:>7d}  "
              f"{r['precision']:>9.4f}  {r['ece']:>8.4f}  {r['mce']:>8.4f}")


def plot_area_vs_score(
    area_score_data,
    save_path=None,
    title=None,
    figsize=(6, 5),
    model_names=None,
    area_key="original",
    box_source="gt",
    show_fp_fn=True,
):
    """
    Generate area vs prediction-score scatter plot (publication style).

    Args:
        area_score_data: List of dicts from val_area_score().
        save_path: Path to save the figure (if None, display only).
        title: Plot title (auto-generated if None).
        figsize: Figure size tuple.
        model_names: Dict mapping class indices to class names.
        area_key: Which area metric — "original", "input", or "pct".
        box_source: "gt" for GT-box area (default), "pred" for prediction-box area.
        show_fp_fn: Whether to include FP/FN markers.

    Returns:
        matplotlib Figure.
    """
    _apply_pub_style()
    area_key = _get_area_key(area_key)
    xlabel, gt_field, pred_field = AREA_LABELS[area_key]

    if box_source == "pred":
        area_field = pred_field
        source_label = "Pred-box"
    else:
        area_field = gt_field
        box_source = "gt"
        source_label = "GT-box"

    if title is None:
        title = f"{source_label} Area vs Prediction Score"

    if not area_score_data:
        print("No area-score data to plot.")
        return None

    tp_data = [d for d in area_score_data if d["status"] == "TP"]
    fp_data = [d for d in area_score_data if d["status"] == "FP"]
    fn_data = [d for d in area_score_data if d["status"] == "FN"]

    fig, ax = plt.subplots(figsize=figsize)

    xlog = area_key in ("original", "input")

    if len(tp_data) > 0:
        tp_areas = np.array([d[area_field] for d in tp_data])
        tp_scores = np.array([d["pred_score"] for d in tp_data])
        ax.scatter(tp_areas, tp_scores, alpha=0.35, s=14, c="#1f77b4", edgecolors="none",
                   label=f"TP (n={len(tp_data)})")

    if show_fp_fn:
        if len(fp_data) > 0:
            fp_scores_arr = np.array([d["pred_score"] for d in fp_data])
            if box_source == "pred":
                fp_areas_arr = np.array([d[area_field] for d in fp_data])
                ax.scatter(fp_areas_arr, fp_scores_arr, alpha=0.2, s=14, c="#d62728",
                           edgecolors="none", label=f"FP (n={len(fp_data)})")
            else:
                fp_x = 1 if xlog else 0.001
                ax.scatter(np.full_like(fp_scores_arr, fp_x), fp_scores_arr,
                           alpha=0.2, s=14, c="#d62728", edgecolors="none",
                           label=f"FP (n={len(fp_data)})")
        if len(fn_data) > 0:
            fn_areas_arr = np.array([d[area_field] for d in fn_data])
            ax.scatter(fn_areas_arr, np.zeros_like(fn_areas_arr), alpha=0.3, s=14,
                       c="#ff7f0e",
                       label=f"FN (n={len(fn_data)})", marker="x")

    # Fixed axis limits for fair comparison across models
    if xlog:
        ax.set_xscale("log")
        ax.set_xlim(*X_LIM_LOG)
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlim(*X_LIM_PCT)
        ax.set_xlabel(xlabel)
    ax.set_ylim(*Y_LIM)
    ax.set_ylabel("Confidence Score")
    ax.set_title(title)
    ax.legend(markerscale=2, loc="best")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


def _print_tp_by_size_ranges(tp_areas, tp_scores, area_ranges, title, area_key="original"):
    """
    Print TP count / mean score / median score broken down by area ranges.

    Args:
        tp_areas:  np.array of TP box areas.
        tp_scores: np.array of TP confidence scores.
        area_ranges: List of (label, lo, hi) tuples (area thresholds).
        title: Section title string.
        area_key: "original"/"input" or "pct" — if "pct", lo/hi are percentages
                  and the label "px²" is replaced by "%".
    """
    unit = "%" if area_key == "pct" else "px²"
    print(f"\n  --- {title} ---")
    for label, lo, hi in area_ranges:
        mask = (tp_areas >= lo) & (tp_areas < hi)
        n = mask.sum()
        if n > 0:
            print(f"  {label} {unit}: {n:6d}  mean score={tp_scores[mask].mean():.4f}  "
                  f"median score={np.median(tp_scores[mask]):.4f}")
        else:
            print(f"  {label} {unit}: {n:6d}")


def print_summary(area_score_data, model_names=None, area_key="original", box_source="gt"):
    """Print summary statistics from area-score data.

    Args:
        area_score_data: List of dicts from val_area_score().
        model_names: Dict mapping class indices to class names.
        area_key: Which area metric — "original", "input", or "pct".
        box_source: "gt" for GT-box area, "pred" for prediction-box area.
    """
    area_key = _get_area_key(area_key)
    _, gt_field, pred_field = AREA_LABELS[area_key]
    area_field = pred_field if box_source == "pred" else gt_field
    source_label = "pred-box" if box_source == "pred" else "GT-box"

    tp_data = [d for d in area_score_data if d["status"] == "TP"]
    fp_data = [d for d in area_score_data if d["status"] == "FP"]
    fn_data = [d for d in area_score_data if d["status"] == "FN"]

    print("\n" + "=" * 60)
    print(f"Area-Score Analysis Summary  (area={area_key}  source={source_label})")
    print("=" * 60)
    print(f"  Total predictions:  {len(tp_data) + len(fp_data)}")
    print(f"  True Positives:     {len(tp_data)}")
    print(f"  False Positives:    {len(fp_data)}")
    print(f"  False Negatives:    {len(fn_data)}")

    if tp_data:
        tp_areas = np.array([d[area_field] for d in tp_data])
        tp_scores = np.array([d["pred_score"] for d in tp_data])

        if area_key in ("original", "input"):
            small_mask = tp_areas < 32 ** 2
            medium_mask = (tp_areas >= 32 ** 2) & (tp_areas < 96 ** 2)
            large_mask = tp_areas >= 96 ** 2

            print(f"\n  --- TP by object size (COCO definition, in pixels) ---")
            for label, mask in [("Small (<32²)", small_mask), ("Medium (32²-96²)", medium_mask), ("Large (>96²)", large_mask)]:
                n = mask.sum()
                if n > 0:
                    print(f"  {label}: {n:6d}  mean score={tp_scores[mask].mean():.4f}  "
                          f"median score={np.median(tp_scores[mask]):.4f}")
                else:
                    print(f"  {label}: {n:6d}")

            # AITOD-style breakdown: 0-8², 8²-16², 16²-32², 32²+
            _print_tp_by_size_ranges(tp_areas, tp_scores, AREA_RANGES_AITOD,
                                     "TP by object size (AITOD definition)", area_key)

            # Geometric progression breakdown: edges at 10×2ⁿ pixels side-length
            geo_ranges = generate_geometric_ranges(start=10, base=2, max_area=640**2)
            _print_tp_by_size_ranges(tp_areas, tp_scores, geo_ranges,
                                     "TP by object size (geometric 10×2ⁿ side-length edges)", area_key)

        if len(tp_areas) > 10:
            print(f"\n  --- Mean score by area percentile (TP only) ---")
            area_percentiles = np.percentile(tp_areas, [0, 25, 50, 75, 100])
            for i in range(len(area_percentiles) - 1):
                lo, hi = area_percentiles[i], area_percentiles[i + 1]
                mask = (tp_areas >= lo) & (tp_areas < hi)
                n = mask.sum()
                if n > 0:
                    print(f"  [{lo:7.1f}, {hi:7.1f}]: n={n:6d}  mean score={tp_scores[mask].mean():.4f}")
    print("=" * 60)

    # Calibration summary
    print_calibration(area_score_data)

    # Per-area-range calibration
    per_range = compute_per_range_calibration(area_score_data, area_field=area_field)
    print_per_range_calibration(per_range)


# ---------------------------------------------------------------------------
#  Calibration analysis
# ---------------------------------------------------------------------------


def compute_calibration(area_score_data, n_bins=10):
    """
    Compute calibration curve data from area-score records.

    Groups all predictions (TP + FP) by confidence score into equal-width bins
    and computes precision per bin.

    Args:
        area_score_data: List of dicts from val_area_score().
        n_bins: Number of confidence bins.

    Returns:
        dict with keys:
            bin_edges:   (n_bins+1,) bin boundaries in [0, 1]
            bin_centers: (n_bins,) bin midpoints
            precision:   (n_bins,) precision = TP/(TP+FP) per bin
            avg_conf:    (n_bins,) mean confidence score per bin
            count:       (n_bins,) total predictions per bin
            ece:         float, Expected Calibration Error
            mce:         float, Maximum Calibration Error
    """
    preds = [d for d in area_score_data if d["status"] in ("TP", "FP")]
    if not preds:
        return None

    scores = np.array([d["pred_score"] for d in preds])
    is_tp = np.array([d["status"] == "TP" for d in preds], dtype=float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    precision = np.zeros(n_bins)
    avg_conf = np.zeros(n_bins)
    count = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1])
        count[i] = mask.sum()
        if count[i] > 0:
            precision[i] = is_tp[mask].mean()
            avg_conf[i] = scores[mask].mean()
        else:
            precision[i] = 0.0
            avg_conf[i] = bin_centers[i]

    total = len(preds)
    ece = float(np.sum(count / total * np.abs(precision - avg_conf)))

    # MCE: only consider bins that actually contain predictions.
    # Empty bins have precision=0, avg_conf=bin_center (e.g. 0.95),
    # which would produce a spurious gap if included.
    non_empty = count > 0
    mce = float(np.max(np.abs(precision[non_empty] - avg_conf[non_empty]))) if non_empty.any() else 0.0

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "precision": precision,
        "avg_conf": avg_conf,
        "count": count,
        "ece": ece,
        "mce": mce,
    }


def plot_calibration_curve(
    area_score_data,
    save_path=None,
    n_bins=10,
    title="Calibration Curve",
    figsize=(6, 5),
):
    """
    Plot calibration curve: confidence score vs observed precision (single model).

    A perfectly calibrated model has precision == confidence in every bin
    (points lie on the diagonal).

    Args:
        area_score_data: List of dicts from val_area_score().
        save_path: Path to save figure.
        n_bins: Number of confidence bins.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    _apply_pub_style()
    cal = compute_calibration(area_score_data, n_bins=n_bins)
    if cal is None:
        print("No prediction data for calibration.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # --- Left: calibration curve ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    sizes = np.maximum(cal["count"] / max(cal["count"].max(), 1) * 120, 20)
    ax.scatter(cal["avg_conf"], cal["precision"], s=sizes, c="#1f77b4", alpha=0.85,
               edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title}  (ECE={cal['ece']:.3f})", fontsize=12)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    for i in range(n_bins):
        if cal["count"][i] > 0:
            ax.annotate(str(cal["count"][i]), (cal["avg_conf"][i], cal["precision"][i]),
                        textcoords="offset points", xytext=(0, 8), ha="center",
                        fontsize=8, color="gray")

    # --- Right: gap per bin ---
    ax2 = axes[1]
    gap = cal["precision"] - cal["avg_conf"]
    colors = ["#d62728" if g < 0 else "#2ca02c" for g in gap]
    ax2.bar(range(n_bins), gap, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("Confidence Bin")
    ax2.set_ylabel("Precision − Confidence")
    ax2.set_title("Gap per Bin")
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f"{cal['bin_edges'][i]:.1f}" for i in range(n_bins)], fontsize=9)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Calibration plot saved to {save_path}")

    plt.show()
    return fig


def print_calibration(area_score_data, n_bins=10):
    """Print calibration statistics including ECE and per-bin precision."""
    cal = compute_calibration(area_score_data, n_bins=n_bins)
    if cal is None:
        print("No prediction data for calibration.")
        return

    print("\n" + "=" * 60)
    print(f"Calibration Analysis  (bins={n_bins})")
    print("=" * 60)
    print(f"  ECE (Expected Calibration Error): {cal['ece']:.5f}")
    print(f"  MCE (Maximum Calibration Error):  {cal['mce']:.5f}")
    print(f"  Total predictions evaluated:      {sum(cal['count'])}")
    print(f"\n  {'Bin':>5s}  {'Count':>6s}  {'AvgConf':>8s}  {'Prec':>8s}  {'Gap':>8s}")
    print(f"  {'-'*5}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*8}")
    for i in range(n_bins):
        if cal["count"][i] > 0:
            print(f"  [{cal['bin_edges'][i]:.1f},{cal['bin_edges'][i+1]:.1f})  "
                  f"{cal['count'][i]:6d}  {cal['avg_conf'][i]:8.4f}  "
                  f"{cal['precision'][i]:8.4f}  {cal['precision'][i] - cal['avg_conf'][i]:+8.4f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
#  Comparison mode (--load)
# ---------------------------------------------------------------------------

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]


def load_pickle_data(paths, labels=None):
    """
    Load one or more pickle files saved by val_area_score().

    Args:
        paths: List of Path/str to .pkl files.
        labels: Optional list of display names (default: derived from path).

    Returns:
        list of (data_list, label_str) tuples.
    """
    datasets = []
    for i, p in enumerate(paths):
        p = Path(p)
        with open(p, "rb") as f:
            obj = pickle.load(f)
        data = obj["data"] if isinstance(obj, dict) else obj
        if labels and i < len(labels):
            lbl = labels[i]
        else:
            lbl = p.parent.name
        datasets.append((data, lbl))
        print(f"Loaded {p}  →  {len(data)} records  label='{lbl}'")
    return datasets


def plot_compare_calibration(
    datasets,
    save_path=None,
    n_bins=10,
    title="Calibration Comparison",
    figsize=(12, 5),
):
    """
    Overlaid calibration curves + grouped gap bars for multiple models
    (publication style).
    """
    _apply_pub_style()
    cals = []
    labels = []
    for data, lbl in datasets:
        cal = compute_calibration(data, n_bins=n_bins)
        if cal is not None:
            cals.append(cal)
            labels.append(lbl)

    if not cals:
        print("No calibration data for any dataset.")
        return None

    n_models = len(cals)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left: overlaid calibration curves ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
    for i, (cal, lbl) in enumerate(zip(cals, labels)):
        color = COLORS[i % len(COLORS)]
        ax.plot(cal["bin_centers"], cal["precision"], "-", color=color, alpha=0.8, linewidth=1.5)
        sizes = np.maximum(cal["count"] / max(cal["count"].max(), 1) * 100, 25)
        ax.scatter(cal["bin_centers"], cal["precision"], s=sizes, color=color, alpha=0.85,
                   edgecolors="k", linewidths=0.5, zorder=4,
                   label=f"{lbl}  (ECE={cal['ece']:.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # --- Right: grouped gap bars ---
    ax2 = axes[1]
    bar_width = 0.8 / n_models
    bin_edges = cals[0]["bin_edges"]
    for i, (cal, lbl) in enumerate(zip(cals, labels)):
        color = COLORS[i % len(COLORS)]
        gap = cal["precision"] - cal["avg_conf"]
        x = np.arange(n_bins) + (i - (n_models - 1) / 2) * bar_width
        ax2.bar(x, gap, width=bar_width, color=color, alpha=0.85, edgecolor="k",
                linewidth=0.5, label=lbl)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("Confidence Bin")
    ax2.set_ylabel("Precision − Confidence")
    ax2.set_title("Gap per Bin")
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f"{bin_edges[i]:.1f}" for i in range(n_bins)], fontsize=10)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Calibration comparison saved to {save_path}")

    plt.show()
    return fig


def plot_compare_area_vs_score(
    datasets,
    save_path=None,
    area_key="original",
    box_source="gt",
    title=None,
    figsize=(6, 5),
    show_fp_fn=False,
):
    """
    Overlaid area-vs-score scatter for comparing multiple models.
    Publication-ready single-panel figure.
    """
    _apply_pub_style()
    area_key = _get_area_key(area_key)
    xlabel, gt_field, pred_field = AREA_LABELS[area_key]
    area_field = pred_field if box_source == "pred" else gt_field
    source_label = "pred-box" if box_source == "pred" else "GT-box"

    if title is None:
        title = f"Area-Score Comparison  ({source_label})"

    xlog = area_key in ("original", "input")

    fig, ax = plt.subplots(figsize=figsize)

    for i, (data, lbl) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        tp_data = [d for d in data if d["status"] == "TP"]
        if tp_data:
            a = np.array([d[area_field] for d in tp_data])
            s = np.array([d["pred_score"] for d in tp_data])
            ax.scatter(a, s, alpha=0.35, s=6, color=color, edgecolors="none",
                       label=f"{lbl} TP (n={len(tp_data)})")
        if show_fp_fn:
            if len([d for d in data if d["status"] == "FP"]) > 0:
                fp_data = [d for d in data if d["status"] == "FP"]
                a = np.array([d[area_field] for d in fp_data])
                s = np.array([d["pred_score"] for d in fp_data])
                ax.scatter(a, s, alpha=0.2, s=14, color=color,
                           marker="x", label=f"{lbl} FP (n={len(fp_data)})")

    if xlog:
        ax.set_xscale("log")
        ax.set_xlim(*X_LIM_LOG)
    else:
        ax.set_xlim(*X_LIM_PCT)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Confidence Score")
    ax.set_title(title)
    ax.set_ylim(*Y_LIM)
    ax.legend(markerscale=2, loc="upper left")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        print(f"Area-score comparison saved to {save_path}")

    plt.show()
    return fig


def print_compare_summary(datasets, area_key="original", box_source="gt", n_bins=10):
    """Print side-by-side summary for multiple datasets."""
    area_key = _get_area_key(area_key)
    _, gt_field, pred_field = AREA_LABELS[area_key]
    area_field = pred_field if box_source == "pred" else gt_field
    source_label = "pred-box" if box_source == "pred" else "GT-box"

    print("\n" + "=" * 70)
    print(f"Comparison Summary  (area={area_key}  source={source_label})")
    print("=" * 70)

    header = f"{'Model':>20s}  {'TP':>8s}  {'FP':>8s}  {'FN':>8s}  {'ECE':>8s}  {'MCE':>8s}"
    print(header)
    print("-" * 70)

    for data, lbl in datasets:
        tp = sum(1 for d in data if d["status"] == "TP")
        fp = sum(1 for d in data if d["status"] == "FP")
        fn = sum(1 for d in data if d["status"] == "FN")
        cal = compute_calibration(data, n_bins=n_bins)
        ece = cal["ece"] if cal else float("nan")
        mce = cal["mce"] if cal else float("nan")
        print(f"{lbl:>20s}  {tp:>8d}  {fp:>8d}  {fn:>8d}  {ece:>8.4f}  {mce:>8.4f}")

    # Per-model calibration detail
    for data, lbl in datasets:
        print(f"\n{'─' * 60}")
        print(f"  Model: {lbl}")
        print(f"{'─' * 60}")

        # TP-by-size breakdowns
        tp_data = [d for d in data if d["status"] == "TP"]
        if tp_data:
            tp_areas = np.array([d[area_field] for d in tp_data])
            tp_scores = np.array([d["pred_score"] for d in tp_data])

            # COCO
            small_mask = tp_areas < 32 ** 2
            medium_mask = (tp_areas >= 32 ** 2) & (tp_areas < 96 ** 2)
            large_mask = tp_areas >= 96 ** 2
            print(f"\n  --- TP by object size (COCO definition, in pixels) ---")
            for label, mask in [("Small (<32²)", small_mask), ("Medium (32²-96²)", medium_mask), ("Large (>96²)", large_mask)]:
                n = mask.sum()
                if n > 0:
                    print(f"  {label}: {n:6d}  mean score={tp_scores[mask].mean():.4f}  "
                          f"median score={np.median(tp_scores[mask]):.4f}")
                else:
                    print(f"  {label}: {n:6d}")

            # AITOD
            _print_tp_by_size_ranges(tp_areas, tp_scores, AREA_RANGES_AITOD,
                                     "TP by object size (AITOD definition)", area_key)
            # Geometric
            geo_ranges = generate_geometric_ranges(start=10, base=2, max_area=640**2)
            _print_tp_by_size_ranges(tp_areas, tp_scores, geo_ranges,
                                     "TP by object size (geometric 10×2ⁿ side-length edges)", area_key)

        print_calibration(data, n_bins=n_bins)
        per_range = compute_per_range_calibration(data, area_field=area_field, n_bins=n_bins)
        print_per_range_calibration(per_range)

    print("=" * 70)


if __name__ == "__main__":
    import argparse
    import sys
    from contextlib import redirect_stdout

    class _Tee:
        """Write to both a file and the original stdout."""
        def __init__(self, file, stdout):
            self.file = file
            self.stdout = stdout
        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)
        def flush(self):
            self.file.flush()
            self.stdout.flush()

    parser = argparse.ArgumentParser(description="GT Area vs Prediction Score Analysis")

    # --- Single val mode ---
    parser.add_argument("--model", type=str, default=None, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, default=None, help="Path to dataset YAML")
    parser.add_argument("--name", type=str, default="val/area_score", help="Val run name")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g., '0', 'cpu')")

    # --- Compare mode ---
    parser.add_argument("--load", type=str, nargs="+", default=None,
                        help="Load pickle file(s) for comparison (skips val)")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Display names for each loaded pickle (same order as --load)")

    # --- Plot options (shared) ---
    parser.add_argument("--area", type=str, default="original",
                        choices=["original", "input", "pct", "orig", "inp", "%"],
                        help="Which area metric to plot (default: original)")
    parser.add_argument("--box-source", type=str, default="gt",
                        choices=["gt", "pred"],
                        help="gt = GT-box area on x-axis (default), pred = prediction-box area")
    parser.add_argument("--n-bins", type=int, default=10, help="Number of confidence bins for calibration")
    parser.add_argument("--show-fp-fn", action="store_true", default=False,
                        help="Show FP/FN markers on scatter plots (default: TP only)")
    parser.add_argument("--save", type=str, default=None, help="Save plot to path (default: auto)")

    opt = parser.parse_args()

    area_slug = _get_area_key(opt.area)

    # =====================================================================
    #  Compare mode: load pickles and overlay
    # =====================================================================
    if opt.load:
        datasets = load_pickle_data(opt.load, labels=opt.labels)

        # Output dir
        if opt.save:
            result_dir = Path(opt.save).parent
        else:
            result_dir = Path("runs/detect") / opt.name
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save text report + console output
        report_path = result_dir / "report.txt"
        with open(report_path, "w", encoding="utf-8") as rf:
            tee = _Tee(rf, sys.stdout)
            with redirect_stdout(tee):
                print_compare_summary(datasets, area_key=area_slug, box_source=opt.box_source,
                                      n_bins=opt.n_bins)
        print(f"Report saved to {report_path}")

        # Plots
        cal_path = str(result_dir / f"compare_calibration_{area_slug}.png")
        plot_compare_calibration(datasets, save_path=cal_path, n_bins=opt.n_bins)

        area_path = str(result_dir / f"compare_area_vs_score_{opt.box_source}_{area_slug}.png")
        plot_compare_area_vs_score(datasets, save_path=area_path,
                                   area_key=area_slug, box_source=opt.box_source,
                                   show_fp_fn=opt.show_fp_fn)

    # =====================================================================
    #  Single val mode (original behaviour)
    # =====================================================================
    else:
        if not opt.model or not opt.data:
            parser.error("--model and --data are required for val mode (or use --load for compare mode)")

        data, validator = val_area_score(
            model_path=opt.model,
            data_path=opt.data,
            name=opt.name,
            batch=opt.batch,
            imgsz=opt.imgsz,
            conf=opt.conf,
            iou=opt.iou,
            device=opt.device,
        )

        result_dir = Path(validator.save_dir)

        # Save text report + console output
        report_path = result_dir / "report.txt"
        with open(report_path, "w", encoding="utf-8") as rf:
            tee = _Tee(rf, sys.stdout)
            with redirect_stdout(tee):
                print_summary(data, model_names=getattr(validator, "names", None),
                              area_key=area_slug, box_source=opt.box_source)
        print(f"Report saved to {report_path}")

        # Plots
        save_path = opt.save or str(result_dir / f"area_vs_score_{opt.box_source}_{area_slug}.png")
        plot_area_vs_score(
            data,
            save_path=str(save_path),
            model_names=getattr(validator, "names", None),
            area_key=area_slug,
            box_source=opt.box_source,
            show_fp_fn=True,
        )

        # Calibration plot
        cal_path = str(result_dir / f"calibration_{area_slug}.png")
        plot_calibration_curve(
            data,
            save_path=cal_path,
            n_bins=opt.n_bins,
        )
