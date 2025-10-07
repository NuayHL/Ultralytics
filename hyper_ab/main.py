# main_experiment.py
import os
import yaml
import json
import subprocess
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import sys


def load_config(config_path="config.yaml"):
    """Loads the master configuration file."""
    print(f"INFO: Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_parameter_grid(search_space):
    """Generates a grid of (alpha, beta) pairs from the search space."""
    alpha_cfg = search_space['alpha']
    beta_cfg = search_space['beta']

    alpha_range = np.arange(alpha_cfg['min'], alpha_cfg['max'] + alpha_cfg['step'], alpha_cfg['step'])
    beta_range = np.arange(beta_cfg['min'], beta_cfg['max'] + beta_cfg['step'], beta_cfg['step'])

    grid = []
    for alpha in alpha_range:
        for beta in beta_range:
            # Round to handle potential float precision issues
            grid.append((round(alpha, 4), round(beta, 4)))
    print(f"INFO: Generated {len(grid)} total experiments.")
    return grid


def initialize_progress(progress_file, grid):
    """Initializes or loads the progress tracking file."""
    if os.path.exists(progress_file):
        print(f"INFO: Found existing progress file: {progress_file}. Resuming experiment.")
        with open(progress_file, 'r') as f:
            pre_process = json.load(f)
        progress = {f"a{a}_b{b}": {"alpha": a, "beta": b, "status": "pending"} for a, b in grid}
        progress.update(pre_process)
    else:
        print("INFO: No progress file found. Starting a new experiment.")
        progress = {f"a{a}_b{b}": {"alpha": a, "beta": b, "status": "pending"} for a, b in grid}
        save_progress(progress_file, progress)
        return progress


def save_progress(progress_file, progress_data):
    """Saves the current progress to the JSON file."""
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=4)


def generate_model_config(base_config_path, output_path, alpha, beta):
    """Generates a specific YOLO model YAML with the given alpha and beta."""
    with open(base_config_path, 'r') as f:
        model_config = yaml.safe_load(f)

    # --- IMPORTANT ---
    # This part MUST match the structure of your base_yolo12n.yaml
    # We navigate to the 'assigner' dictionary and update the values.
    # The last element of head is usually the detection layer with TAL
    model_config['assigner_type'] = 'TaskAlignedAssigner'
    model_config['topk'] = 10
    model_config['alpha'] = alpha
    model_config['beta'] = beta
    # assigner_type: 'TaskAlignedAssigner'
    # topk: 10
    # alpha: 1.0
    # beta: 1.0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(model_config, f, sort_keys=False)

def main():
    # 1. Load Configurations
    config = load_config()
    paths = config['paths']
    train_params = config['train_params']

    # 2. Setup Experiment Grid and Progress Tracking
    param_grid = generate_parameter_grid(config['search_space'])
    progress_data = initialize_progress(paths['progress_file'], param_grid)
    os.makedirs(paths['log_dir'], exist_ok=True)

    # 3. Main Experiment Loop
    for exp_key, details in tqdm(progress_data.items(), desc="Total Experiments"):
        if details['status'] == 'completed':
            tqdm.write(f"SKIP: Experiment {exp_key} is already completed.")
            continue

        if details['status'] == 'running':
            tqdm.write(f"WARN: Experiment {exp_key} was 'running'. Resetting to 'pending' for a retry.")
            details['status'] = 'pending'

        alpha, beta = float(details['alpha']), float(details['beta'])

        # Lock the current experiment
        details['status'] = 'running'
        save_progress(paths['progress_file'], progress_data)

        try:
            # A. Generate the specific model config for this run
            exp_run_name = f"{train_params['run_name_prefix']}_a{alpha}_b{beta}"
            exp_run_name_size = (f"{train_params['run_name_prefix']}{train_params['model_size']}"
                                 f"_a{alpha}_b{beta}")
            generated_config_path = os.path.join(paths['generated_configs_dir'], f"{exp_run_name}.yaml")
            tqdm.write(f"\n---> STARTING: {exp_key} | alpha={alpha}, beta={beta}")
            tqdm.write(f"  - Generating config: {generated_config_path}")
            generate_model_config(paths['base_model_config'], generated_config_path, alpha, beta)
            terminal_log_file = os.path.join(paths['log_dir'], f"{exp_run_name}.log")

            # B. Initialize and Train the model
            tqdm.write(f"  - Starting training for {exp_run_name}...")

            with open(terminal_log_file, 'w') as f:
                subprocess.run(
                [
                    "yolo", "detect", "train",
                    f"data={paths['dataset_config']}",
                    f"model={generated_config_path}",
                    f"name={train_params['project_name']}/{exp_run_name_size}",
                    f"epochs={train_params['epochs']}",
                    f"imgsz={train_params['imgsz']}",
                    f"batch={train_params['batch']}",
                    f"seed={train_params['seed']}",
                    "save=True"
                ],
                    stdout=f,
                    stderr=f,
                    check=True
                )
            model = YOLO(f"../runs/detect/{train_params['project_name']}/"
                         f"{exp_run_name_size}/weights/best.pt")
            model.val(data=paths['dataset_config'],
                      name=f"{train_params['project_name']}_val/{exp_run_name_size}",
                      batch=train_params['batch'],
                      imgsz=train_params['imgsz'],
                      save_json=True,)

            # C. Mark as completed
            details['status'] = 'completed'
            tqdm.write(f"<--- SUCCESS: Experiment {exp_key} finished.")

        except KeyboardInterrupt:
            # If user presses Ctrl+C
            details['status'] = 'pending'  # Reset status to be picked up next time
            save_progress(paths['progress_file'], progress_data)
            print("\n\nINFO: Keyboard interrupt detected. Saving progress and exiting gracefully.")
            sys.exit(0)

        except Exception as e:
            # Handle other errors (e.g., CUDA out of memory)
            details['status'] = 'failed'
            details['error'] = str(e)
            tqdm.write(f"!!!! ERROR: Experiment {exp_key} failed with error: {e}")

        # Save progress after every run
        save_progress(paths['progress_file'], progress_data)

    print("\n\n========================================")
    print("All experiments have been completed!")
    print("========================================")


if __name__ == "__main__":
    main()