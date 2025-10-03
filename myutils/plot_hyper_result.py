import numpy as np
import matplotlib.pyplot as plt
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
hyper_test_yaml = '../ab_hyper/config.yaml'

search_space = load_config(hyper_test_yaml)['search_space']
alpha_cfg = search_space['alpha']
beta_cfg = search_space['beta']
alpha_range_np = np.arange(alpha_cfg['min'], alpha_cfg['max'] + alpha_cfg['step'], alpha_cfg['step'])
beta_range_np = np.arange(beta_cfg['min'], beta_cfg['max'] + beta_cfg['step'], beta_cfg['step'])

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


# --- USAGE ---
# Choose the index of the layer (metric) you want to visualize.
for i in range(stats_length):
    plot_heatmap_for_layer(hyper_eval_result, i, alpha_range_np, beta_range_np)
