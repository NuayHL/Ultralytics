import pickle
import os
from typing import List, Tuple, Dict

import torch
import csv
import math
import matplotlib.pyplot as plt
from ultralytics.utils.tal import TaskAlignedAssigner
from tqdm import tqdm

import pandas as pd
import numpy as np
import yaml


# ============================================================================
# Classes
# ============================================================================

class Assignment_Analyzer:
    """Analyzer for loading assignment data from pickle files."""
    load_keyset = ['pd_scores', 'pd_bboxes', 'anc_points', 'gt_labels', 'gt_bboxes', 'mask_gt']
    
    def __init__(self):
        pass

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return {key: torch.tensor(data[key]) for key in self.load_keyset}


class Assigner(TaskAlignedAssigner):
    """Extended TaskAlignedAssigner with additional analysis methods."""
    
    def __init__(self, topk: int = 10, num_classes: int = 10, alpha: float = 0.5, beta: float = 6.0):
        super().__init__(topk=topk, num_classes=num_classes, alpha=alpha, beta=beta)

    def _forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        """
        Compute the task-aligned assignment.

        Args:
            pd_scores (torch.Tensor): Predicted classification scores with shape (bs, num_total_anchors, num_classes).
            pd_bboxes (torch.Tensor): Predicted bounding boxes with shape (bs, num_total_anchors, 4).
            anc_points (torch.Tensor): Anchor points with shape (num_total_anchors, 2).
            gt_labels (torch.Tensor): Ground truth labels with shape (bs, n_max_boxes, 1).
            gt_bboxes (torch.Tensor): Ground truth boxes with shape (bs, n_max_boxes, 4).
            mask_gt (torch.Tensor): Mask for valid ground truth boxes with shape (bs, n_max_boxes, 1).

        Returns:
            target_labels (torch.Tensor): Target labels with shape (bs, num_total_anchors).
            target_bboxes (torch.Tensor): Target bounding boxes with shape (bs, num_total_anchors, 4).
            target_scores (torch.Tensor): Target scores with shape (bs, num_total_anchors, num_classes).
            fg_mask (torch.Tensor): Foreground mask with shape (bs, num_total_anchors).
            target_gt_idx (torch.Tensor): Target ground truth indices with shape (bs, num_total_anchors).
        """
        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )

        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # Normalize: max score assigned with max overlap score, rest using propotions with the max score.
        align_metric *= mask_pos
        pos_align_metrics = align_metric.amax(dim=-1, keepdim=True)  # b, max_num_obj
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)  # b, max_num_obj
        norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align_metric

        return target_labels, target_bboxes, target_scores, fg_mask.bool(), target_gt_idx

    @torch.no_grad()
    def compute_candidate_mask(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, alpha: float, beta: float):
        old_alpha, old_beta = self.alpha, self.beta
        self.alpha, self.beta = alpha, beta
        try:
            mask_pos, align_metric, overlaps = self.get_pos_mask(pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt)
        finally:
            self.alpha, self.beta = old_alpha, old_beta
        return mask_pos, align_metric, overlaps

    @torch.no_grad()
    def compute_align_scores_on_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_pos, alpha: float, beta: float):
        old_alpha, old_beta = self.alpha, self.beta
        self.alpha, self.beta = alpha, beta
        try:
            # Reuse get_box_metrics but restrict to candidate mask
            align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_pos.bool())
        finally:
            self.alpha, self.beta = old_alpha, old_beta
        return align_metric, overlaps


# ============================================================================
# Utility Functions
# ============================================================================

def compute_box_area_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Compute box areas from xyxy format boxes.
    
    Args:
        boxes: (b, n, 4) tensor in xyxy format
    
    Returns:
        areas: (b, n) tensor of box areas
    """
    wh = (boxes[..., 2:4] - boxes[..., 0:2]).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def bin_by_area(areas: torch.Tensor, thresholds: List[float]) -> torch.Tensor:
    """
    Bin areas into categories based on thresholds.
    
    Args:
        areas: (b, n) tensor of box areas
        thresholds: List of area thresholds in ascending order. 
                   E.g., [1024, 9216] creates 3 bins: [0, 1024), [1024, 9216), [9216, inf)
    
    Returns:
        bins: (b, n) tensor with bin indices from 0 to len(thresholds)
    """
    bins = torch.zeros_like(areas, dtype=torch.long)
    for i, thr in enumerate(thresholds):
        bins = torch.where(areas >= thr, torch.full_like(bins, i + 1), bins)
    return bins


def iter_pkl_files(root_dir: str) -> List[str]:
    """Iterate through all pickle files in a directory tree.
    
    Args:
        root_dir: Root directory to search
    
    Returns:
        List of file paths, sorted
    """
    files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.endswith('.pkl'):
                files.append(os.path.join(dirpath, fn))
    files.sort()
    return files


def _unique_sorted(values):
    """Get unique sorted values from an iterable."""
    vals = sorted(set(values))
    return vals


# ============================================================================
# Hyperparameter Loading Functions
# ============================================================================

def load_hyper_pair(hyper_test_yaml, folder_path):
    """Load hyperparameter ranges from YAML config file.
    
    Args:
        hyper_test_yaml: Path to YAML config file
        folder_path: Unused, kept for compatibility
    
    Returns:
        alpha_range: List of alpha values as strings
        beta_range: List of beta values as strings
    """
    print(f"INFO: Loading configuration from {hyper_test_yaml}")
    with open(hyper_test_yaml, 'r') as f:
        search_space = yaml.safe_load(f)['search_space']
    alpha_cfg = search_space['alpha']
    beta_cfg = search_space['beta']
    alpha_range_np = np.arange(alpha_cfg['min'], alpha_cfg['max'] + 1e-2, alpha_cfg['step'])
    beta_range_np = np.arange(beta_cfg['min'], beta_cfg['max'] + 1e-2, beta_cfg['step'])

    alpha_range = [f"{alpha:.1f}" for alpha in alpha_range_np]
    beta_range = [f"{beta:.1f}" for beta in beta_range_np]
    return alpha_range, beta_range


def load_hyper_result(folder_path, hyper_test_yaml):
    """Load hyperparameter evaluation results from .npy files.
    
    Args:
        folder_path: Directory containing .npy result files
        hyper_test_yaml: Path to YAML config file for hyperparameter ranges
    
    Returns:
        3D numpy array of shape (num_alphas, num_betas, stats_length)
    """
    print(f"INFO: Loading configuration from {hyper_test_yaml}")
    with open(hyper_test_yaml, 'r') as f:
        search_space = yaml.safe_load(f)['search_space']
    alpha_cfg = search_space['alpha']
    beta_cfg = search_space['beta']
    alpha_range_np = np.arange(alpha_cfg['min'], alpha_cfg['max'] + 1e-2, alpha_cfg['step'])
    beta_range_np = np.arange(beta_cfg['min'], beta_cfg['max'] + 1e-2, beta_cfg['step'])

    alpha_range = [f"{alpha:.1f}" for alpha in alpha_range_np]
    beta_range = [f"{beta:.1f}" for beta in beta_range_np]

    print("\n--- Loading data from files... ---")
    # First, try to load one file to determine the depth (stats_length)
    try:
        first_a, first_b = alpha_range[0], beta_range[0]
        first_filename = f"{first_a}_{first_b}.npy"
        first_filepath = os.path.join(folder_path, first_filename)
        first_stats = np.load(first_filepath)
        stats_length = len(first_stats)
        print(f"Detected stats vector length from file: {stats_length}")
    except FileNotFoundError:
        print(f"ERROR: Could not find the first file to determine data shape.")
        print(f"Looked for: {first_filepath}")
        exit()

    # Create an empty 3D numpy array to hold all the results
    hyper_eval_result = np.zeros((len(alpha_range), len(beta_range), stats_length))
    valid_mask = np.full((len(alpha_range), len(beta_range)), fill_value=True, dtype=bool)

    # Iterate through each combination of alpha and beta
    for i, a_value in enumerate(alpha_range):
        for j, b_value in enumerate(beta_range):
            filename = f"{a_value}_{b_value}.npy"
            filepath = os.path.join(folder_path, filename)

            try:
                stats = np.load(filepath)
                if stats[0] <= 1e-5:
                    hyper_eval_result[i, j, :] = np.nan
                else:
                    hyper_eval_result[i, j, :] = stats
            except FileNotFoundError:
                print(f"Warning: File not found for alpha={a_value}, beta={b_value}. Path: {filepath}")
                hyper_eval_result[i, j, :] = np.nan

    print("--- Data loading complete. ---")
    return hyper_eval_result


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_gap(
    pkl_dir: str,
    align_alpha: float,
    align_beta: float,
    score_alphas: List[float],
    score_betas: List[float],
    area_thresholds: List[float] = None,
    topk: int = 10,
    num_classes: int = 80,
    save_csv: str = None,
    device: str = None,
):
    """
    Analyze gap metrics across different score alpha/beta combinations and area bins.
    
    Args:
        pkl_dir: Directory containing pickle files
        align_alpha: Alpha parameter for candidate selection
        align_beta: Beta parameter for candidate selection
        score_alphas: List of alpha values for scoring
        score_betas: List of beta values for scoring
        area_thresholds: List of area thresholds in ascending order. 
                        E.g., [1024, 9216] creates 3 bins.
                        Default: [32^2, 96^2] for small/medium/large objects
        topk: Top-k parameter for assignment
        num_classes: Number of object classes
        save_csv: Path to save CSV results
        device: Device to use ('cuda', 'cpu', or None for auto)
    
    Returns:
        List of result rows
    """
    if area_thresholds is None:
        area_thresholds = [32.0 * 32.0, 96.0 * 96.0]
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    num_bins = len(area_thresholds) + 1
    
    assigner = Assigner(topk=topk, num_classes=num_classes, alpha=align_alpha, beta=align_beta)
    analyzer = Assignment_Analyzer()

    # Aggregators: {(a_s, b_s, bin): [sum_gap, sum_ratio, count]}
    agg: Dict[Tuple[float, float, int], List[float]] = {}

    files = iter_pkl_files(pkl_dir)
    for fpath in tqdm(files):
        data = analyzer.load_pickle(fpath)
        pd_scores = data['pd_scores'].to(device)
        pd_bboxes = data['pd_bboxes'].to(device)
        anc_points = data['anc_points'].to(device)
        gt_labels = data['gt_labels'].to(device)
        gt_bboxes = data['gt_bboxes'].to(device)
        mask_gt = data['mask_gt'].to(device)

        target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx = assigner.forward(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt
        )

        # Candidate mask with fixed align (alpha, beta)
        mask_pos, _, _ = assigner.compute_candidate_mask(
            pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt, align_alpha, align_beta
        )

        # Compute GT areas and bins
        areas = compute_box_area_xyxy(gt_bboxes)  # (b, n)
        area_bins = bin_by_area(areas, area_thresholds)  # (b, n)

        bsz, n_max, num_anchors = mask_pos.shape
        valid_gt_mask = mask_gt.squeeze(-1).bool()  # (b, n)

        # Pre-expand masks for indexing per gt
        for a_s in score_alphas:
            for b_s in score_betas:
                align_scores, _ = assigner.compute_align_scores_on_mask(
                    pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_pos, a_s, b_s
                )  # (b, n, A)

                # For each valid gt, compute gap among its candidate anchors
                # set non-candidates to -inf so they don't affect topk
                scores = align_scores.masked_fill(~mask_pos.bool(), float('-inf'))

                # top-2 per gt
                top2, _ = torch.topk(scores, k=min(2, num_anchors), dim=-1)
                # If a gt has <2 candidates (rare), skip
                has_two = (top2[..., 1] > float('-inf'))

                # Compute gap and ratio where defined
                gap = (top2[..., 0] - top2[..., 1]).masked_fill(~has_two, 0.0)
                ratio = (top2[..., 0] / (top2[..., 1] + 1e-12)).masked_fill(~has_two, 0.0)

                # Aggregate by bins
                for bin_id in range(num_bins):
                    sel = valid_gt_mask & (area_bins == bin_id) & has_two
                    if sel.any():
                        key = (float(a_s), float(b_s), int(bin_id))
                        gsum = gap[sel].sum().item()
                        rsum = ratio[sel].sum().item()
                        cnt = int(sel.sum().item())
                        if key not in agg:
                            agg[key] = [0.0, 0.0, 0]
                        agg[key][0] += gsum
                        agg[key][1] += rsum
                        agg[key][2] += cnt

    # Prepare CSV output
    rows = []
    for (a_s, b_s, bin_id), (gsum, rsum, cnt) in agg.items():
        if cnt > 0:
            rows.append({
                'score_alpha': a_s,
                'score_beta': b_s,
                'align_alpha': float(align_alpha),
                'align_beta': float(align_beta),
                'area_bin': int(bin_id),
                'count': int(cnt),
                'avg_gap': gsum / cnt,
                'avg_ratio': rsum / cnt,
            })

    rows.sort(key=lambda r: (r['area_bin'], r['score_alpha'], r['score_beta']))

    if save_csv:
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        with open(save_csv, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=list(rows[0].keys()) if rows else [
                    'score_alpha', 'score_beta', 'align_alpha', 'align_beta', 
                    'area_bin', 'count', 'avg_gap', 'avg_ratio'
                ]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)

    return rows


def visualize_gap_csv(
    csv_path: str, 
    out_dir: str = None, 
    min_count: int = 10, 
    show_ratio: bool = False, 
    area_thresholds: List[float] = None
):
    """
    Visualize gap analysis results from CSV.
    
    Args:
        csv_path: Path to CSV file
        out_dir: Output directory for plots
        min_count: Minimum sample count threshold
        show_ratio: Whether to show ratio heatmaps
        area_thresholds: List of area thresholds used for binning (for generating bin names)
                        If None, will use generic names like "Bin 0", "Bin 1", etc.
    
    Returns:
        List of saved file paths
    """
    # Read CSV
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]
    if not rows:
        raise ValueError("CSV has no rows")

    # Determine grids and bins
    score_alphas = _unique_sorted(float(r['score_alpha']) for r in rows)
    score_betas = _unique_sorted(float(r['score_beta']) for r in rows)
    bins = _unique_sorted(int(r['area_bin']) for r in rows)

    # Prepare output dir
    if out_dir is None:
        out_dir = os.path.dirname(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    # Generate bin names dynamically
    def get_bin_name(bin_id: int, thresholds: List[float] = None) -> str:
        if thresholds is None:
            # Fallback to simple names if thresholds not provided
            bin_name_map = {0: 'Small', 1: 'Medium', 2: 'Large'}
            return bin_name_map.get(bin_id, f"Bin {bin_id}")
        
        if bin_id == 0:
            return f"Area < {thresholds[0]:.0f}"
        elif bin_id == len(thresholds):
            return f"Area >= {thresholds[-1]:.0f}"
        else:
            return f"{thresholds[bin_id-1]:.0f} <= Area < {thresholds[bin_id]:.0f}"
    
    bin_name = {bin_id: get_bin_name(bin_id, area_thresholds) for bin_id in bins}

    def build_matrix(metric_key: str, bin_id: int):
        h, w = len(score_betas), len(score_alphas)
        mat = [[math.nan for _ in range(w)] for _ in range(h)]
        cnt = [[0 for _ in range(w)] for _ in range(h)]
        for r in rows:
            if int(r['area_bin']) != bin_id:
                continue
            a = float(r['score_alpha'])
            b = float(r['score_beta'])
            i = score_betas.index(b)
            j = score_alphas.index(a)
            c = int(r['count']) if r['count'] != '' else 0
            val = float(r[metric_key]) if r[metric_key] != '' else math.nan
            mat[i][j] = val
            cnt[i][j] = c
        # mask by min_count
        for i in range(h):
            for j in range(w):
                if cnt[i][j] < min_count:
                    mat[i][j] = math.nan
        return mat, cnt

    def plot_heatmap(mat, title, filename):
        fig, ax = plt.subplots(figsize=(max(6, len(score_alphas) * 0.5), max(4, len(score_betas) * 0.5)))
        im = ax.imshow(mat, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xticks(range(len(score_alphas)))
        ax.set_xticklabels([str(a) for a in score_alphas], rotation=45, ha='right')
        ax.set_yticks(range(len(score_betas)))
        ax.set_yticklabels([str(b) for b in score_betas])
        ax.set_xlabel('score alpha')
        ax.set_ylabel('score beta')
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('avg_gap')
        fig.tight_layout()
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return out_path

    saved = []
    # Plot avg_gap heatmaps per bin
    for bin_id in bins:
        mat_gap, _ = build_matrix('avg_gap', bin_id)
        title = f"Avg Gap Heatmap - {bin_name.get(bin_id, str(bin_id))} (min_count={min_count})"
        fname = f"gap_heatmap_bin{bin_id}.png"
        saved.append(plot_heatmap(mat_gap, title, fname))

    # Optionally plot ratio heatmaps
    if show_ratio:
        for bin_id in bins:
            mat_ratio, _ = build_matrix('avg_ratio', bin_id)
            title = f"Avg Ratio Heatmap - {bin_name.get(bin_id, str(bin_id))} (min_count={min_count})"
            fname = f"ratio_heatmap_bin{bin_id}.png"
            saved.append(plot_heatmap(mat_ratio, title, fname))

    # Also export the best (alpha,beta) per bin by avg_gap
    best_rows = []
    for bin_id in bins:
        best = None
        for r in rows:
            if int(r['area_bin']) != bin_id:
                continue
            if r['avg_gap'] == '' or r['count'] == '':
                continue
            if int(r['count']) < min_count:
                continue
            val = float(r['avg_gap'])
            if (best is None) or (val > best['avg_gap']):
                best = {
                    'area_bin': bin_id,
                    'score_alpha': float(r['score_alpha']),
                    'score_beta': float(r['score_beta']),
                    'avg_gap': val,
                    'count': int(r['count'])
                }
        if best is not None:
            best_rows.append(best)

    best_txt = os.path.join(out_dir, 'best_ab_per_bin.txt')
    with open(best_txt, 'w') as f:
        for br in best_rows:
            f.write(
                f"bin={br['area_bin']}({bin_name.get(br['area_bin'], 'Unknown')}) "
                f"best(alpha,beta)=({br['score_alpha']},{br['score_beta']}) "
                f"avg_gap={br['avg_gap']:.6f} count={br['count']}\n"
            )
    saved.append(best_txt)

    return saved


def plot_gap_vs_area(
    csv_path: str,
    align_alpha: float,
    align_beta: float,
    area_thresholds: list,
    save_path: str = None,
):
    """
    Visualize avg_gap vs area relationship for specific align_alpha/align_beta.
    
    Args:
        csv_path: CSV file path from analyze_gap()
        align_alpha: align_alpha to filter
        align_beta: align_beta to filter
        area_thresholds: List of area thresholds in ascending order
        save_path: Optional, image save path (e.g., "output/gap_area.png")
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Filter target alpha/beta
    df_sel = df[(df["score_alpha"] == align_alpha) & (df["score_beta"] == align_beta)]
    if df_sel.empty:
        raise ValueError(f"No entries found for align_alpha={align_alpha}, align_beta={align_beta}")

    # Define area bin boundaries and representative values
    area_bins = [0] + area_thresholds + [640 * 640]
    area_centers = []
    for i in range(len(area_bins) - 1):
        # Logarithmic center point is more appropriate for large area spans
        area_centers.append(np.sqrt(area_bins[i] * area_bins[i + 1]))
    area_centers = np.array(area_centers)

    # Plot
    plt.figure(figsize=(8, 5))
    for (a_s, b_s), subdf in df_sel.groupby(["score_alpha", "score_beta"]):
        subdf = subdf.sort_values("area_bin")
        y = subdf["avg_gap"].to_numpy()
        plt.plot(area_centers[: len(y)], y, marker="o", label=f"αs={a_s}, βs={b_s}")

    plt.xscale("log")
    plt.xlabel("GT Area (log scale)")
    plt.ylabel("Average Gap (Top1 - Top2)")
    plt.title(f"Gap vs Area (align α={align_alpha}, β={align_beta})")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure saved to {save_path}")
    else:
        plt.show()

