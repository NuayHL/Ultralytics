import os
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

from assignment_utils import (
    load_hyper_pair,
)


def get_bin_name(bin_id: int, thresholds: list) -> str:
    """Generate bin name from bin_id and thresholds."""
    if bin_id == 0:
        return f"Area < {thresholds[0]:.0f}"
    elif bin_id == len(thresholds):
        return f"Area >= {thresholds[-1]:.0f}"
    else:
        return f"{thresholds[bin_id-1]:.0f} <= Area < {thresholds[bin_id]:.0f}"


def load_and_filter_csv(csv_path: str):
    """Load CSV and filter rows where align_alpha == score_alpha and align_beta == score_beta.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        List of filtered rows (dicts)
    """
    if not os.path.isfile(csv_path):
        return []
    
    filtered_rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            align_alpha = float(row['align_alpha'])
            align_beta = float(row['align_beta'])
            score_alpha = float(row['score_alpha'])
            score_beta = float(row['score_beta'])
            
            # Only keep rows where align and score match
            if abs(align_alpha - score_alpha) < 1e-6 and abs(align_beta - score_beta) < 1e-6:
                filtered_rows.append({
                    'alpha': align_alpha,
                    'beta': align_beta,
                    'area_bin': int(row['area_bin']),
                    'count': int(row['count']),
                    'avg_gap': float(row['avg_gap']),
                    'avg_ratio': float(row['avg_ratio']),
                })
    
    return filtered_rows


def plot_heatmap_for_bin(
    data_dict: dict,
    bin_id: int,
    alpha_range: list,
    beta_range: list,
    area_thresholds: list,
    out_dir: str,
    min_count: int = 10,
):
    """Plot heatmap for a specific area bin.
    
    Args:
        data_dict: Dictionary mapping (alpha, beta) -> avg_gap
        bin_id: Area bin ID
        alpha_range: List of alpha values
        beta_range: List of beta values
        area_thresholds: List of area thresholds
        out_dir: Output directory
        min_count: Minimum count threshold (not used here but kept for consistency)
    """
    # Build matrix
    h, w = len(beta_range), len(alpha_range)
    mat = [[math.nan for _ in range(w)] for _ in range(h)]
    
    for i, beta in enumerate(beta_range):
        for j, alpha in enumerate(alpha_range):
            key = (alpha, beta)
            if key in data_dict:
                mat[i][j] = data_dict[key] if data_dict[key] > 1e-3 else math.nan
    
    # Plot
    fig, ax = plt.subplots(figsize=(max(6, len(alpha_range) * 0.5), max(4, len(beta_range) * 0.5)))
    im = ax.imshow(mat, aspect='auto', origin='lower', cmap='viridis')
    
    ax.set_xticks(range(len(alpha_range)))
    ax.set_xticklabels([str(a) for a in alpha_range], rotation=45, ha='right')
    ax.set_yticks(range(len(beta_range)))
    ax.set_yticklabels([str(b) for b in beta_range])
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')
    
    bin_name_str = get_bin_name(bin_id, area_thresholds)
    ax.set_title(f'Avg Gap Heatmap - {bin_name_str}')
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('avg_gap')
    
    fig.tight_layout()
    
    # Save
    out_path = os.path.join(out_dir, f'gap_heatmap_bin{bin_id}.png')
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    
    return out_path


def analyze_hyper_csvs(
    csv_dir: str,
    hyper_test_yaml: str,
    area_thresholds: list,
    out_dir: str,
    min_count: int = 10,
):
    """Analyze hyperparameter CSV files and generate heatmaps.
    
    Args:
        csv_dir: Directory containing CSV files
        hyper_test_yaml: Path to YAML config file
        area_thresholds: List of area thresholds
        out_dir: Output directory for heatmaps
        min_count: Minimum count threshold (for filtering)
    """
    # Load hyperparameter ranges
    alpha_range_str, beta_range_str = load_hyper_pair(hyper_test_yaml, "")
    alpha_range = [float(a) for a in alpha_range_str]
    beta_range = [float(b) for b in beta_range_str]
    
    # Get all CSV files
    csv_files = []
    for alpha_str in alpha_range_str:
        for beta_str in beta_range_str:
            csv_path = os.path.join(csv_dir, f'yolo12n_a{alpha_str}_b{beta_str}.csv')
            if os.path.exists(csv_path):
                csv_files.append(csv_path)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Collect all data, organized by area_bin
    # Structure: {bin_id: {(alpha, beta): avg_gap}}
    bin_data = {}
    
    for csv_path in csv_files:
        print(f"Processing {os.path.basename(csv_path)}...")
        rows = load_and_filter_csv(csv_path)
        
        for row in rows:
            bin_id = row['area_bin']
            alpha = row['alpha']
            beta = row['beta']
            avg_gap = row['avg_gap']
            count = row['count']
            
            # Filter by min_count
            if count < min_count:
                continue
            
            if bin_id not in bin_data:
                bin_data[bin_id] = {}
            
            key = (alpha, beta)
            # If multiple entries exist, we could average them, but typically there should be only one
            if key not in bin_data[bin_id]:
                bin_data[bin_id][key] = avg_gap
            else:
                # Average if duplicate (shouldn't happen, but just in case)
                bin_data[bin_id][key] = (bin_data[bin_id][key] + avg_gap) / 2
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot heatmaps for each bin
    saved_files = []
    for bin_id in sorted(bin_data.keys()):
        print(f"Plotting heatmap for bin {bin_id}...")
        saved_path = plot_heatmap_for_bin(
            bin_data[bin_id],
            bin_id,
            alpha_range,
            beta_range,
            area_thresholds,
            out_dir,
            min_count,
        )
        saved_files.append(saved_path)
        print(f"  Saved to {saved_path}")
    
    # Also save a summary text file
    summary_path = os.path.join(out_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Hyperparameter Gap Analysis Summary\n")
        f.write(f"=====================================\n\n")
        f.write(f"Alpha range: {alpha_range}\n")
        f.write(f"Beta range: {beta_range}\n")
        f.write(f"Area thresholds: {area_thresholds}\n")
        f.write(f"Number of bins: {len(bin_data)}\n\n")
        
        for bin_id in sorted(bin_data.keys()):
            f.write(f"\nBin {bin_id} ({get_bin_name(bin_id, area_thresholds)}):\n")
            f.write(f"  Number of data points: {len(bin_data[bin_id])}\n")
            if bin_data[bin_id]:
                gaps = list(bin_data[bin_id].values())
                f.write(f"  Min gap: {min(gaps):.6f}\n")
                f.write(f"  Max gap: {max(gaps):.6f}\n")
                f.write(f"  Mean gap: {np.mean(gaps):.6f}\n")
    
    saved_files.append(summary_path)
    print(f"\nSummary saved to {summary_path}")
    
    return saved_files


if __name__ == "__main__":
    folder_path = 'myutils/hyper_result'  # <--- CHANGE THIS (not used but kept for compatibility)
    hyper_test_yaml = 'hyper_ab/config.yaml'  # <--- CHANGE THIS
    
    input_size = 640 * 640
    area_thresholds = list()
    i = 10
    while i < input_size:
        area_thresholds.append(i)
        i *= 2
    
    csv_dir = 'assign_detail/hyper_ab_csv'  # <--- CHANGE THIS
    out_dir = 'assign_detail/hyper_ab_heatmaps'  # <--- CHANGE THIS: output directory for heatmaps
    
    saved_files = analyze_hyper_csvs(
        csv_dir=csv_dir,
        hyper_test_yaml=hyper_test_yaml,
        area_thresholds=area_thresholds,
        out_dir=out_dir,
        min_count=10,  # Minimum count threshold
    )
    
    print(f"\n✅ Generated {len(saved_files)} files in {out_dir}")
