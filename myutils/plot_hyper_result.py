import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import os  # Import the 'os' module to handle file paths
import yaml

# ==============================================================================
# 1. Configuration - !!! PLEASE MODIFY THESE VALUES !!!
# ==============================================================================

# --- Path to the folder containing your .npy files ---
# IMPORTANT: Replace this with the actual path to your folder.
# Example for Windows: 'C:/Users/YourUser/Desktop/results'
# Example for Linux/macOS: '/home/YourUser/project/results'
def load_config(config_path="config.yaml"):
    """Loads the master configuration file."""
    print(f"INFO: Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
folder_path = './hyper_result'  # <--- CHANGE THIS
hyper_test_yaml = '../hyper_ab/config.yaml'

search_space = load_config(hyper_test_yaml)['search_space']
alpha_cfg = search_space['alpha']
beta_cfg = search_space['beta']
alpha_range_np = np.arange(alpha_cfg['min'], alpha_cfg['max'] + 1e-2, alpha_cfg['step'])
beta_range_np = np.arange(beta_cfg['min'], beta_cfg['max'] + 1e-2, beta_cfg['step'])

alpha_range = [f"{alpha:.1f}" for alpha in alpha_range_np]
beta_range = [f"{beta:.1f}" for beta in beta_range_np]

# ==============================================================================
# 3. Load data from individual .npy files into a single 3D array
# ==============================================================================

print("\n--- Loading data from files... ---")
# First, try to load one file to determine the depth (stats_length)
try:
    # Construct the filename for the first (a, b) pair
    first_a, first_b = alpha_range[0], beta_range[0]
    first_filename = f"{first_a}_{first_b}.npy"
    first_filepath = os.path.join(folder_path, first_filename)
    # Load the first file to get the length of the stats array
    first_stats = np.load(first_filepath)
    stats_length = len(first_stats)
    print(f"Detected stats vector length from file: {stats_length}")
except FileNotFoundError:
    print(f"ERROR: Could not find the first file to determine data shape.")
    print(f"Looked for: {first_filepath}")
    # Exit or raise an exception if the first file is not found
    exit()

# Create an empty 3D numpy array to hold all the results
# The shape is (num_alphas, num_betas, stats_length)
hyper_eval_result = np.zeros((len(alpha_range), len(beta_range), stats_length))
valid_mask = np.full((len(alpha_range), len(beta_range)),fill_value=True, dtype=bool)

# Iterate through each combination of alpha and beta
for i, a_value in enumerate(alpha_range):
    for j, b_value in enumerate(beta_range):
        # Construct the expected filename
        filename = f"{a_value}_{b_value}.npy"
        filepath = os.path.join(folder_path, filename)

        try:
            # Load the data from the .npy file
            stats = np.load(filepath)
            if stats[0]<=1e-5:
                hyper_eval_result[i, j, :] = np.nan
                # Place the loaded 1D array into the correct slice of our 3D array
            else:
                hyper_eval_result[i, j, :] = stats
        except FileNotFoundError:
            print(f"Warning: File not found for alpha={a_value}, beta={b_value}. Path: {filepath}")
            # The corresponding entry in hyper_eval_result will remain zeros
            # You could also fill it with np.nan if you prefer
            hyper_eval_result[i, j, :] = np.nan

print("--- Data loading complete. ---")


# ==============================================================================
# 4. Select the layer and plot the heatmap (This part is unchanged)
# ==============================================================================

def plot_heatmap_for_layer(data, layer_index, alphas, betas, folder='hyper_result_plot'):
    """
    Plots a heatmap for a specific layer of the 3D data array.

    Args:
        data (np.ndarray): The 3D numpy array of shape (num_alphas, num_betas, num_metrics).
        layer_index (int): The index of the layer (metric) to visualize.
        alphas (np.ndarray): The array of alpha values (for the y-axis).
        betas (np.ndarray): The array of beta values (for the x-axis).
    """
    os.makedirs(folder, exist_ok=True)
    num_alphas, num_betas, num_metrics = data.shape
    if not (0 <= layer_index < num_metrics):
        raise ValueError(
            f"layer_index must be between 0 and {num_metrics - 1}, but got {layer_index}"
        )

    data_slice = data[:, :, layer_index]

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        data_slice,
        ax=ax,
        xticklabels=[f"{b:.2f}" for b in betas],
        yticklabels=[f"{a:.2f}" for a in alphas],
        cmap='viridis',
        annot=True,
        fmt=".3f",
        linewidths=.5,
    )

    ax.set_title(f'Heatmap of Metric at Layer Index: {layer_index}', fontsize=16)
    ax.set_xlabel('Beta Values', fontsize=12)
    ax.set_ylabel('Alpha Values', fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{folder}/heatmap_{layer_index}.png")
    plt.close()
    #plt.show()

import math
# --- Dynamic Hyperparameter Functions ---
# def get_dynamic_alpha(area: float) -> float:
#     log_area = np.log(area + 1e-6)
#
#     alpha = math.exp(-log_area * 0.3) * 3 + 0.3
#     return alpha
def get_dynamic_alpha(x, a = 0.3, b = 2.5, c = 2.0, d = 1.7):
    """
    y = a + (b-a) / (1 + exp(c*(log10(x)-d)))
    a: 下渐近线 (≈0.3)
    b: 上渐近线 (≈2.5)
    c: 下降斜率
    d: 拐点位置
    """
    return a + (b-a) / (1 + np.exp(c*(np.log10(x)-d)))

def get_dynamic_beta(x, a = 0.3, b = 5, c = 2.0, d = 1.7):
    return b + (a-b) / (1 + np.exp(c*(np.log10(x)-d)))


def plot_topk_ab(alpha_range, beta_range, topk, hyper_eval_result, eval_metric_indices, area_representatives):
    optimal_alphas = []
    optimal_betas = []
    area_representatives = np.array(area_representatives)
    area_mask = np.full_like(area_representatives, True, dtype=bool)
    # Step 2: Iterate through each area range
    for i, area_rep in enumerate(area_representatives):
        metric_index = eval_metric_indices[i]

        # 步骤 3: 提取当前面积区间的评估结果
        # (Step 3: Extract the evaluation results for the current area)
        area_eval_results = hyper_eval_result[:, :, metric_index]

        if area_eval_results[0,0] < 0:
            area_mask[i] = False
            continue

        area_eval_results_copy = area_eval_results.copy()
        nan_mask = np.isnan(area_eval_results_copy)
        if np.any(nan_mask):
            area_eval_results_copy[nan_mask] = -1.0

        # 步骤 4: 找到 top-k 的结果
        # (Step 4: Find the top-k results)
        # 将二维评估结果展平，并找到 top-k 分数对应的索引
        # (Flatten the 2D evaluation results and find the indices of the top-k scores)
        flat_indices = np.argsort(area_eval_results_copy.flatten())[-topk:]

        # 将一维索引转换回二维的 (alpha_index, beta_index)
        # (Convert the flat indices back to 2D indices)
        top_k_indices = np.unravel_index(flat_indices, area_eval_results.shape)

        # 步骤 5: 统计最优超参数
        # (Step 5: Aggregate the optimal hyperparameters)
        # 获取 top-k 结果对应的 alpha 和 beta 值
        # (Get the alpha and beta values corresponding to the top-k results)
        top_k_alphas = alpha_range[top_k_indices[0]]
        top_k_betas = beta_range[top_k_indices[1]]

        # 计算 top-k 超参数的均值作为该面积下的最优值
        # (Calculate the mean of the top-k hyperparameters as the optimal value for this area)
        optimal_alphas.append(top_k_alphas)
        optimal_betas.append(top_k_betas)

    optimal_alphas = np.array(optimal_alphas).flatten()
    optimal_betas = np.array(optimal_betas).flatten()
    area_representatives = area_representatives[area_mask]

    # ---------------------------------------------------------
    areas = np.logspace(0, 6, 1000)  # 从10^0到10^6

    # 计算对应的alpha和beta值
    alphas = [get_dynamic_alpha(area) for area in areas]
    betas = [get_dynamic_beta(area) for area in areas]
    # ---------------------------------------------------------

    # 步骤 6: 绘制散点图
    # (Step 6: Plot the scatter plots)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 2D 散点图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # (area_representative, optimal_alpha)
    ax1.scatter(np.repeat(area_representatives, topk), optimal_alphas, s=100, c='royalblue', alpha=0.8, edgecolors='w')
    ax1.plot(areas, alphas, 'b-', linewidth=2, label='Dynamic Alpha')
    ax1.set_xscale('log')
    ax1.set_xlabel('Area Representative (sqrt scale)', fontsize=12)
    ax1.set_ylabel('Optimal Alpha', fontsize=12)
    ax1.set_title('Optimal Alpha vs. Area', fontsize=14)

    # (area_representative, optimal_beta)
    ax2.scatter(np.repeat(area_representatives, topk), optimal_betas, s=100, c='seagreen', alpha=0.8, edgecolors='w')
    ax2.plot(areas, betas, 'g-', linewidth=2, label='Dynamic Beta')
    ax2.set_xscale('log')
    ax2.set_xlabel('Area Representative (sqrt scale)', fontsize=12)
    ax2.set_ylabel('Optimal Beta', fontsize=12)
    ax2.set_title('Optimal Beta vs. Area', fontsize=14)

    plt.tight_layout()
    plt.savefig('optimal_hyperparameters_vs_area_2d.png', dpi=300)
    print("2D scatter plot saved as optimal_hyperparameters_vs_area_2d.png")

    # ---------------------------------------------------------------
    x_data = np.repeat(area_representatives, topk)
    alpha_data_to_save = np.column_stack((x_data, optimal_alphas))
    alpha_file_name = f'optimal_alpha_vs_area_topk{topk}.csv'
    np.savetxt(alpha_file_name, alpha_data_to_save, delimiter=',')

    beta_data_to_save = np.column_stack((x_data, optimal_betas))
    beta_file_name = f'optimal_beta_vs_area_topk{topk}.csv'
    np.savetxt(beta_file_name, beta_data_to_save, delimiter=',')
    # ----------------------------------------------------------------

    # 3D 散点图
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # scatter = ax.scatter(np.repeat(area_representatives, topk),
    #                      optimal_betas, optimal_alphas, c=area_representatives, cmap='viridis', s=100,
    #                      depthshade=True)
    #
    # ax.set_xlabel('Area Representative (sqrt scale)', fontsize=10, labelpad=10)
    # ax.set_xscale('log')
    # ax.set_ylabel('Optimal Beta', fontsize=10, labelpad=10)
    # ax.set_zlabel('Optimal Alpha', fontsize=10, labelpad=10)
    # ax.set_title('Optimal Alpha and Beta vs. Area', fontsize=14)
    #
    # # 添加颜色条 (Add a color bar)
    # cbar = fig.colorbar(scatter, shrink=0.5, aspect=10)
    # cbar.set_label('Area Representative', fontsize=10)
    #
    # plt.savefig('optimal_hyperparameters_vs_area_3d.png', dpi=300)
    # print("3D scatter plot saved as optimal_hyperparameters_vs_area_3d.png")

if __name__ == '__main__':
    input_size = 640 * 640
    _partial = [0]
    i = 10
    while i < input_size:
        _partial.append(i)
        i *= 2
    _partial.append(input_size)

    # 您的 custom_area_rng 包含了全部范围和分段范围
    # 我们只分析分段范围
    custom_area_rng_full = [[_partial[0], _partial[-1]], ]
    for i in range(len(_partial) - 1):
        custom_area_rng_full.append([_partial[i], _partial[i + 1]])

    # 我们只使用分段进行分析
    custom_areas = custom_area_rng_full[1:]
    print("Analyzing area ranges:")
    print(custom_areas)

    area_representatives = []
    for i, area in enumerate(custom_areas):
        start, end = area
        if start == 0:
            # 特殊处理第一个区间 [0, X]，使用算术平均数
            # (Special handling for the first interval [0, X])
            rep = (start + end) / 2.0
        else:
            # 其他指数区间，使用几何平均数
            # (For other exponential intervals, use geometric mean)
            rep = np.sqrt(start * end)
        area_representatives.append(rep)

    print("\nCalculated Area Representatives:")
    print(np.array(area_representatives))

    assert stats_length == (len(custom_area_rng_full) - 1) * 2 + 6

    plot_topk_ab(alpha_range_np, beta_range_np,
                 topk=1,
                 hyper_eval_result=hyper_eval_result,
                 eval_metric_indices=[i+3 for i in range(len(custom_areas))],
                 area_representatives=area_representatives)

# for i in range(stats_length):
#         plot_heatmap_for_layer(hyper_eval_result, i, alpha_range_np, beta_range_np)
