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
from area_score_val import AreaScoreValidator


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
    Run validation using AreaScoreValidator and return collected area-score data.

    Args:
        model_path: Path to the model weights (.pt file).
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

    metrics = model.val(
        validator=AreaScoreValidator,
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
    validator = AreaScoreValidator.last_instance
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


def plot_area_vs_score(
    area_score_data,
    save_path=None,
    title=None,
    figsize=(12, 6),
    model_names=None,
    area_key="original",
    box_source="gt",
    show_fp_fn=True,
):
    """
    Generate area vs prediction-score scatter plots.

    Args:
        area_score_data: List of dicts from val_area_score().
        save_path: Path to save the figure (if None, display only).
        title: Plot title (auto-generated if None).
        figsize: Figure size tuple.
        model_names: Dict mapping class indices to class names.
        area_key: Which area metric — "original", "input", or "pct".
        box_source: "gt" for GT-box area (default), "pred" for prediction-box area.
        show_fp_fn: Whether to include FP/FN markers on the left scatter plot.

    Returns:
        matplotlib Figure.
    """
    area_key = _get_area_key(area_key)
    xlabel, gt_field, pred_field = AREA_LABELS[area_key]

    # Select which area field to use based on box_source
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

    tp_areas = np.array([d[area_field] for d in tp_data])
    tp_scores = np.array([d["pred_score"] for d in tp_data])

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- Left plot: scatter ---
    ax = axes[0]
    if len(tp_data) > 0:
        ax.scatter(tp_areas, tp_scores, alpha=0.25, s=4, c="#1f77b4", edgecolors="none",
                   label=f"TP (n={len(tp_data)})")
    if show_fp_fn:
        if len(fp_data) > 0:
            fp_scores_arr = np.array([d["pred_score"] for d in fp_data])
            if box_source == "pred":
                # FP predictions have real pred-box areas — plot at their actual x
                fp_areas_arr = np.array([d[area_field] for d in fp_data])
                ax.scatter(fp_areas_arr, fp_scores_arr, alpha=0.2, s=4, c="#d62728",
                           edgecolors="none", label=f"FP (n={len(fp_data)})")
            else:
                # GT area: draw FP at sentinel x (no GT box)
                fp_x = 1 if area_key in ("original", "input") else 0.001
                ax.scatter(np.full_like(fp_scores_arr, fp_x), fp_scores_arr,
                           alpha=0.2, s=4, c="#d62728", edgecolors="none",
                           label=f"FP (n={len(fp_data)})")
        if len(fn_data) > 0:
            fn_areas_arr = np.array([d[area_field] for d in fn_data])
            ax.scatter(fn_areas_arr, np.zeros_like(fn_areas_arr), alpha=0.3, s=4,
                       c="#ff7f0e", edgecolors="none",
                       label=f"FN (n={len(fn_data)})", marker="x")

    xlog = area_key in ("original", "input")
    if xlog:
        ax.set_xscale("log")
        ax.set_xlabel(f"{xlabel} (log scale)")
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Prediction Confidence Score")
    ax.set_title(f"{title}\narea={area_key}  source={box_source}")
    ax.legend(markerscale=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    if xlog:
        ax.set_xlim(xmin=0.5)

    # --- Right plot: 2D histogram for TP + FP (both have pred boxes) ---
    ax2 = axes[1]
    if box_source == "pred":
        # Include FP since they have real pred-box areas
        hist_data = tp_data + fp_data
    else:
        hist_data = tp_data

    if hist_data:
        areas_arr = np.array([d[area_field] for d in hist_data])
        scores_arr = np.array([d["pred_score"] for d in hist_data])
        p99 = np.percentile(areas_arr, 99) or 1
        x_range = [0, p99] if area_key in ("original", "input") else [0, 100]
        h = ax2.hist2d(areas_arr, scores_arr, bins=(80, 40), cmap="viridis",
                       range=[x_range, [0, 1]])
        plt.colorbar(h[3], ax=ax2, label="Count")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Prediction Confidence Score")
    ax2.set_title(f"{source_label} Density (area={area_key})")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()
    return fig


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
    mce = float(np.max(np.abs(precision - avg_conf)))

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
    figsize=(8, 6),
):
    """
    Plot calibration curve: confidence score vs observed precision.

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
    cal = compute_calibration(area_score_data, n_bins=n_bins)
    if cal is None:
        print("No prediction data for calibration.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

    # --- Left: calibration curve ---
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    sizes = np.maximum(cal["count"] / max(cal["count"].max(), 1) * 120, 15)
    ax.scatter(cal["avg_conf"], cal["precision"], s=sizes, c="#1f77b4", alpha=0.8,
               edgecolors="k", linewidths=0.5, zorder=3)
    ax.set_xlabel("Mean Confidence Score per Bin")
    ax.set_ylabel("Observed Precision (TP / (TP+FP))")
    ax.set_title(f"{title}\nECE={cal['ece']:.4f}  MCE={cal['mce']:.4f}  bins={n_bins}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Annotate bins with count
    for i in range(n_bins):
        if cal["count"][i] > 0:
            ax.annotate(str(cal["count"][i]), (cal["avg_conf"][i], cal["precision"][i]),
                        textcoords="offset points", xytext=(0, 8), ha="center", fontsize=6, color="gray")

    # --- Right: gap (precision - confidence) per bin ---
    ax2 = axes[1]
    gap = cal["precision"] - cal["avg_conf"]
    colors = ["#d62728" if g < 0 else "#2ca02c" for g in gap]
    bars = ax2.bar(range(n_bins), gap, color=colors, alpha=0.8, edgecolor="k", linewidth=0.5)
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("Confidence Bin Index")
    ax2.set_ylabel("Precision − Confidence")
    ax2.set_title("Gap per Bin (negative = overconfident)")
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f"{cal['bin_edges'][i]:.1f}" for i in range(n_bins)], fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
    figsize=(14, 6),
):
    """
    Overlaid calibration curves + grouped gap bars for multiple models.

    Args:
        datasets: List of (area_score_data, label) from load_pickle_data().
        save_path: Path to save figure.
        n_bins: Number of confidence bins.
        title: Plot title.
        figsize: Figure size (width, height).

    Returns:
        matplotlib Figure.
    """
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
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect")
    for i, (cal, lbl) in enumerate(zip(cals, labels)):
        color = COLORS[i % len(COLORS)]
        ax.plot(cal["avg_conf"], cal["precision"], "-", color=color, alpha=0.7, linewidth=1.5)
        sizes = np.maximum(cal["count"] / max(cal["count"].max(), 1) * 100, 20)
        ax.scatter(cal["avg_conf"], cal["precision"], s=sizes, color=color, alpha=0.85,
                   edgecolors="k", linewidths=0.4, zorder=4,
                   label=f"{lbl}  ECE={cal['ece']:.4f}")
    ax.set_xlabel("Mean Confidence per Bin")
    ax.set_ylabel("Observed Precision")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8, markerscale=0.8)
    ax.grid(True, alpha=0.3)
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
        ax2.bar(x, gap, width=bar_width, color=color, alpha=0.8, edgecolor="k",
                linewidth=0.3, label=lbl)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xlabel("Confidence Bin")
    ax2.set_ylabel("Precision − Confidence")
    ax2.set_title("Gap per Bin (negative = overconfident)")
    ax2.set_xticks(range(n_bins))
    ax2.set_xticklabels([f"{bin_edges[i]:.1f}" for i in range(n_bins)], fontsize=7)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Calibration comparison saved to {save_path}")

    plt.show()
    return fig


def plot_compare_area_vs_score(
    datasets,
    save_path=None,
    area_key="original",
    box_source="gt",
    title=None,
    figsize=None,
):
    """
    Overlaid area-vs-score scatter + aligned 2D histograms for multiple models.

    Args:
        datasets: List of (area_score_data, label).
        save_path: Path to save figure.
        area_key: "original", "input", or "pct".
        box_source: "gt" or "pred".
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    area_key = _get_area_key(area_key)
    xlabel, gt_field, pred_field = AREA_LABELS[area_key]
    area_field = pred_field if box_source == "pred" else gt_field
    source_label = "pred-box" if box_source == "pred" else "GT-box"

    if title is None:
        title = f"Area-Score Comparison  ({source_label}, area={area_key})"

    n_models = len(datasets)
    if figsize is None:
        figsize = (12, 4 + 3 * n_models)

    # Compute shared axis limits from all data
    all_areas = []
    all_scores = []
    for data, _ in datasets:
        preds = [d for d in data if d["status"] in ("TP", "FP") and d[area_field] > 0]
        if preds:
            all_areas.extend([d[area_field] for d in preds])
            all_scores.extend([d["pred_score"] for d in preds])
    all_areas = np.array(all_areas) if all_areas else np.array([0, 1])
    all_scores = np.array(all_scores) if all_scores else np.array([0, 1])

    xlog = area_key in ("original", "input")
    if xlog:
        x_lim = (max(all_areas[all_areas > 0].min() * 0.5 if (all_areas > 0).any() else 0.5, 0.5),
                 all_areas.max() * 1.1 or 1)
    else:
        x_lim = (0, max(all_areas.max() * 1.05, 1))
    y_lim = (-0.02, 1.02)

    # Layout: left = scatter overlay, right = small-multiple 2D histograms
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_models + 1, 2, height_ratios=[1] * n_models + [0.06],
                          hspace=0.3, wspace=0.25)
    ax_scatter = fig.add_subplot(gs[:-1, 0])

    # --- Left: overlaid scatter ---
    for i, (data, lbl) in enumerate(datasets):
        color = COLORS[i % len(COLORS)]
        tp_data = [d for d in data if d["status"] == "TP"]
        if tp_data:
            a = np.array([d[area_field] for d in tp_data])
            s = np.array([d["pred_score"] for d in tp_data])
            ax_scatter.scatter(a, s, alpha=0.2, s=4, color=color, edgecolors="none",
                               label=f"{lbl} TP (n={len(tp_data)})")
        if box_source == "pred":
            fp_data = [d for d in data if d["status"] == "FP"]
            if fp_data:
                a = np.array([d[area_field] for d in fp_data])
                s = np.array([d["pred_score"] for d in fp_data])
                ax_scatter.scatter(a, s, alpha=0.1, s=4, color=color, edgecolors="none",
                                   marker="x", label=f"{lbl} FP (n={len(fp_data)})")
    if xlog:
        ax_scatter.set_xscale("log")
        ax_scatter.set_xlabel(f"{xlabel} (log scale)")
    else:
        ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel("Prediction Confidence Score")
    ax_scatter.set_title(title)
    ax_scatter.set_xlim(x_lim)
    ax_scatter.set_ylim(y_lim)
    ax_scatter.legend(markerscale=3, fontsize=7)
    ax_scatter.grid(True, alpha=0.3)

    # --- Right: small-multiple 2D histograms ---
    for i, (data, lbl) in enumerate(datasets):
        ax_h = fig.add_subplot(gs[i, 1])
        hist_data = [d for d in data if d["status"] in ("TP", "FP") and d[area_field] > 0]
        if hist_data:
            a = np.array([d[area_field] for d in hist_data])
            s = np.array([d["pred_score"] for d in hist_data])
            h = ax_h.hist2d(a, s, bins=(60, 30), cmap="plasma",
                            range=[list(x_lim), list(y_lim)])
        ax_h.set_xlabel(xlabel, fontsize=8)
        ax_h.set_ylabel("Score", fontsize=8)
        ax_h.set_title(f"{lbl}", fontsize=9)
        ax_h.set_xlim(x_lim)
        ax_h.set_ylim(y_lim)
        if xlog:
            ax_h.set_xscale("log")
        ax_h.grid(True, alpha=0.2)

    # Shared colorbar
    cbar_ax = fig.add_subplot(gs[-1, 1])
    # Use a dummy histogram to get colorbar
    dummy_ax = fig.add_subplot(gs[0, 1])
    h_dummy = dummy_ax.hist2d([0], [0], bins=(1, 1), cmap="plasma", range=[[0, 1], [0, 1]])
    dummy_ax.remove()
    cbar = plt.colorbar(h_dummy[3], cax=cbar_ax, orientation="horizontal", label="Count")
    cbar_ax.set_xlabel("Count")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
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
        print_calibration(data, n_bins=n_bins)

    print("=" * 70)


if __name__ == "__main__":
    import argparse

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
    parser.add_argument("--save", type=str, default=None, help="Save plot to path (default: auto)")

    opt = parser.parse_args()

    area_slug = _get_area_key(opt.area)

    # =====================================================================
    #  Compare mode: load pickles and overlay
    # =====================================================================
    if opt.load:
        datasets = load_pickle_data(opt.load, labels=opt.labels)

        print_compare_summary(datasets, area_key=area_slug, box_source=opt.box_source,
                              n_bins=opt.n_bins)

        # Output dir: use the directory of the first pickle
        result_dir = Path(opt.load[0]).parent

        # Calibration comparison
        cal_path = opt.save or str(result_dir / f"compare_calibration_{area_slug}.png")
        plot_compare_calibration(datasets, save_path=cal_path, n_bins=opt.n_bins)

        # Area-score comparison
        area_path = str(result_dir / f"compare_area_vs_score_{opt.box_source}_{area_slug}.png")
        plot_compare_area_vs_score(datasets, save_path=area_path,
                                   area_key=area_slug, box_source=opt.box_source)

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

        print_summary(data, model_names=getattr(validator, "names", None),
                      area_key=area_slug, box_source=opt.box_source)

        result_dir = Path(validator.save_dir)

        # Area-vs-score scatter plot
        save_path = opt.save or str(result_dir / f"area_vs_score_{opt.box_source}_{area_slug}.png")
        plot_area_vs_score(
            data,
            save_path=str(save_path),
            model_names=getattr(validator, "names", None),
            area_key=area_slug,
            box_source=opt.box_source,
        )

        # Calibration plot
        cal_path = str(result_dir / f"calibration_{area_slug}.png")
        plot_calibration_curve(
            data,
            save_path=cal_path,
            n_bins=opt.n_bins,
        )
