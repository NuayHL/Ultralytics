import os

from assignment_utils import (
    analyze_gap,
    visualize_gap_csv,
)

if __name__ == "__main__":
    # Default analysis run for VisDrone
    PKL_DIR = 'assign_detail'
    EXP_NAME = 'v12s_a1_b4_visdrone'

    pkl_dir = os.path.join(os.path.dirname(__file__), PKL_DIR, EXP_NAME)
    result_dir = os.path.join(os.path.dirname(__file__), PKL_DIR, f'{EXP_NAME}_results')
    save_csv_path = os.path.join(result_dir, 'gap_analysis.csv')
    os.makedirs(result_dir, exist_ok=True)
    # Grids
    score_alphas = [round(0.5 + 0.2 * i, 2) for i in range(int((2.5 - 0.5) / 0.2) + 1)]
    score_betas = [round(0.5 + 0.5 * i, 2) for i in range(int((5.0 - 0.5) / 0.5) + 1)]

    # Convert tuple thresholds to list for compatibility
    area_thresholds_list = [32.0 * 32.0, 96.0 * 96.0]
    
    results = analyze_gap(
        pkl_dir=pkl_dir,
        align_alpha=0.5,
        align_beta=6.0,
        score_alphas=score_alphas,
        score_betas=score_betas,
        area_thresholds=area_thresholds_list,
        topk=10,
        num_classes=10,
        save_csv=save_csv_path,
    )
    print(f"Wrote {len(results)} aggregated rows to {save_csv_path}")

    saved = visualize_gap_csv(
        csv_path=save_csv_path,
        out_dir=result_dir,
        min_count=10,          # 小样本屏蔽阈值，避免噪声
        show_ratio=False,      # 若想看 ratio 的热力图，改为 True
        area_thresholds=area_thresholds_list  # Pass thresholds for better bin naming
    )
    print(saved)

