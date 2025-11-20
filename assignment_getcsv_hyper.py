import os

from assignment_utils import (
    load_hyper_pair,
    analyze_gap,
)

if __name__ == "__main__":

    folder_path = 'myutils/hyper_result'  # <--- CHANGE THIS
    hyper_test_yaml = 'hyper_ab/config.yaml'  # <--- CHANGE THIS
    
    input_size = 640 * 640
    area_thresholds = list()
    i = 10
    while i < input_size:
        area_thresholds.append(i)
        i *= 2

    alpha_range, beta_range = load_hyper_pair(hyper_test_yaml, folder_path)
    score_alphas = [float(alpha) for alpha in alpha_range]
    score_betas = [float(beta) for beta in beta_range]

    pkl_dir = 'assign_detail/hyper_ab'  # <--- CHANGE THIS

    csv_dir = 'assign_detail/hyper_ab_csv'  # <--- CHANGE THIS
    os.makedirs(csv_dir, exist_ok=True)

    for alpha in alpha_range:
        for beta in beta_range:
            print(f"INFO: Processing alpha={alpha}, beta={beta}")
            pkl_dir_path = os.path.join(pkl_dir, f'yolo12n_a{alpha}_b{beta}')   # <--- CHANGE THIS
            if not os.path.exists(pkl_dir_path):
                print(f"INFO: {pkl_dir_path} not found, skipping")
                continue
            csv_path = os.path.join(csv_dir, f'yolo12n_a{alpha}_b{beta}.csv')   # <--- CHANGE THIS
            analyze_gap(
                pkl_dir=pkl_dir_path,
                align_alpha=float(alpha),
                align_beta=float(beta),
                score_alphas=score_alphas,
                score_betas=score_betas,
                area_thresholds=area_thresholds,
                topk=10,
                save_csv=csv_path,
            )
            print(f"INFO: Analyzed gap and saved to {csv_path}")

