import os

from assignment_utils import (
    load_hyper_result,
    analyze_gap,
    plot_gap_vs_area,
)


if __name__ == "__main__":

    folder_path = 'myutils/hyper_result'  # <--- CHANGE THIS
    hyper_test_yaml = 'hyper_ab/config.yaml'
    
    hyper_eval_result = load_hyper_result(folder_path, hyper_test_yaml)

    # Define area thresholds - can be any number of thresholds
    # Example 1: 3 bins (small, medium, large) - default
    # area_thresholds = [32.0 * 32.0, 96.0 * 96.0]

    # Example 2: 4 bins (tiny, small, medium, large)
    # area_thresholds = [16.0 * 16.0, 32.0 * 32.0, 96.0 * 96.0]

    # Example 3: 5 bins (extra fine-grained)
    # area_thresholds = [16.0 * 16.0, 32.0 * 32.0, 64.0 * 64.0, 128.0 * 128.0]

    input_size = 640 * 640
    area_thresholds = list()
    i = 10
    while i < input_size:
        area_thresholds.append(i)
        i *= 2

    # Default analysis run for VisDrone
    PKL_DIR = 'assign_detail'
    EXP_NAME = 'v12s_visdrone'

    pkl_dir = os.path.join(os.path.dirname(__file__), PKL_DIR, EXP_NAME)
    result_dir = os.path.join(os.path.dirname(__file__), PKL_DIR, f'{EXP_NAME}_{len(area_thresholds)}bin_results')
    os.makedirs(result_dir, exist_ok=True)
    save_csv_path = os.path.join(result_dir, 'gap_analysis.csv')
    # Grids
    score_alphas = [round(0.5 + 0.2 * i, 2) for i in range(int((2.5 - 0.5) / 0.2) + 1)]
    score_betas = [round(0.5 + 0.5 * i, 2) for i in range(int((6.0 - 0.5) / 0.5) + 1)]

    # ---------------------------------------------- get csv -----------------------------------------------------------

    results = analyze_gap(
        pkl_dir=pkl_dir,
        align_alpha=0.5,
        align_beta=6.0,
        score_alphas=score_alphas,
        score_betas=score_betas,
        area_thresholds=area_thresholds,
        topk=10,
        num_classes=10,
        save_csv=save_csv_path,
        device='cpu',  # Explicitly set to CPU as in original
    )
    print(f"Wrote {len(results)} aggregated rows to {save_csv_path}")

    # -------------------------------------------- end -----------------------------------------------------------------

    # saved = visualize_gap_csv(
    #     csv_path=save_csv_path,
    #     out_dir=result_dir,
    #     min_count=10,
    #     show_ratio=False,
    #     area_thresholds=area_thresholds  # Pass thresholds for better bin naming
    # )
    # print(saved)

    plot_gap_vs_area(csv_path=save_csv_path, align_alpha=0.5, align_beta=6.0,
                     area_thresholds=area_thresholds, save_path=os.path.join(result_dir, 'gap_area.png'))

