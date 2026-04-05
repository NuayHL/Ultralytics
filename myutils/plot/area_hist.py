import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def npy_check(file_path):
    try:
        print(f"--- Analyzing: {file_path} ---")
        data = np.load(file_path)

        if data.size == 0:
            print("The file is empty.")
        else:
            print(f"Total number of entries: {len(data)}")
            print(f"Min value: {data.min()}")
            print(f"Max value: {data.max()}")
            print(f"Mean value: {data.mean():.2f}")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def plot_histogram_from_npy(file_path, num_bins=50):
    """
    Loads data from a .npy file and plots its histogram.

    Args:
        file_path (str): The path to the .npy file.
        num_bins (int): The number of bins for the histogram.
    """
    try:
        # Load the data from the specified .npy file
        data = np.load(file_path)

        # Check if the loaded data is a 1D array
        if data.ndim != 1:
            print(f"Warning: Data is not 1-dimensional (shape: {data.shape}). Flattening the array.")
            data = data.flatten()

        # Create a figure and an axes object for the plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Generate the histogram
        ax.hist(data, bins=num_bins, color='steelblue', edgecolor='black', alpha=0.8)

        # Set plot titles and labels
        ax.set_title('Data Histogram', fontsize=16)
        ax.set_xlabel('Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        # Add a grid for better readability
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Improve layout
        plt.tight_layout()

        # Display the plot
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

import matplotlib.ticker as mticker
def plot_quantile_histogram(file_path, percentiles_to_use):
    """
    Loads data and plots a histogram with bins defined by quantiles.
    """
    try:
        # 1. 加载数据
        data = np.load(file_path)
        # 确保数据是正数，以便用于对数坐标轴
        data = data[data > 0]
        if data.size == 0:
            print("No positive data found to plot.")
            return

        # 2. 根据指定的百分位数计算bin的边界
        # np.unique确保边界值不重复
        bin_edges = np.unique(np.percentile(data, percentiles_to_use))

        print("--- Quantile-Based Bin Edges ---")
        print("Calculated bin edges based on data distribution:")
        # 为了可读性，我们打印一些整数值
        print([int(edge) for edge in bin_edges])
        print(f"Total bins created: {len(bin_edges) - 1}")

        # 3. 绘制直方图
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(data, bins=bin_edges, color='darkcyan', edgecolor='black', alpha=0.8)

        # 4. 美化图表
        ax.set_title('Histogram with Quantile-Based Bins', fontsize=16)
        ax.set_xlabel('Area (pixels)', fontsize=12)
        ax.set_ylabel('Frequency (Number of GT Boxes)', fontsize=12)

        # 使用对数坐标轴能更好地展示长尾分布
        ax.set_xscale('log')
        # ax.set_yscale('log')

        # 优化坐标轴刻度显示，防止科学计数法
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())

        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

        plt.xticks(rotation=45)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def plot_log_histogram(file_path, num_bins=20):
    try:
        data = np.load(file_path)
        data = data[data > 0]
        if data.size == 0:
            return

        # 在对数空间中创建均匀的bin边界
        # np.log10(data.min()) 是起始点
        # np.log10(data.max()) 是结束点
        # bin_edges = np.logspace(np.log10(data.min()), np.log10(data.max()), num_bins)
        bin_edges = np.logspace(np.log10(data.min()), np.log10(640*640), num_bins)

        print("\n--- Log-Spaced Bin Edges ---")
        print([int(edge) for edge in bin_edges])

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(data, bins=bin_edges, color='purple', edgecolor='black', alpha=0.8)

        ax.set_title('Histogram with Log-Spaced Bins', fontsize=16)
        ax.set_xlabel('Area (pixels)', fontsize=12)
        ax.set_ylabel('Frequency (Number of GT Boxes)', fontsize=12)

        # 必须使用对数坐标轴，否则无法看清分布
        ax.set_xscale('log')
        # ax.set_yscale('log')

        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.yaxis.get_major_formatter().set_scientific(False)

        plt.xticks(rotation=45)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


def plot_custom_histogram(file_path, custom_bin_edges):
    """
    使用用户提供的自定义列表作为bin边界来绘制直方图。

    Args:
        file_path (str): .npy文件的路径。
        custom_bin_edges (list or np.array): 用于划分bin的边界值列表。
    """
    try:
        # 1. 加载数据
        data = np.load(file_path)
        data = data[data > 0]
        if data.size == 0:
            print("No positive data found to plot.")
            return

        # 确保边界是排序好的，并且移除重复值
        bin_edges = sorted(list(set(custom_bin_edges)))

        print("\n--- Custom Bin Edges ---")
        print(f"Using {len(bin_edges) - 1} custom bins with the following edges:")
        print([int(edge) for edge in bin_edges])

        # 2. 绘制直方图
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.hist(data, bins=bin_edges, color='olivedrab', edgecolor='black', alpha=0.8)

        # 3. 美化图表
        ax.set_title('Histogram with Custom-Defined Bins', fontsize=16)
        ax.set_xlabel('Area (pixels)', fontsize=12)
        ax.set_ylabel('Frequency (Number of GT Boxes)', fontsize=12)

        # 对于类似 [10, 20, 40, 80...] 这种指数增长的区间，
        # 使用对数坐标轴能让bin在视觉上更均匀
        ax.set_xscale('log')
        # 根据我们之前的讨论，我们注释掉y轴的对数缩放以避免错误
        # ax.set_yscale('log')

        # 优化坐标轴刻度显示
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.xaxis.get_major_formatter().set_scientific(False)
        plt.xticks(rotation=45)

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig('label_temp')

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    path_to_npy_file = 'aitodv2_coco_val_area_letterbox.npy'
    npy_check(path_to_npy_file)

    plot_histogram_from_npy(path_to_npy_file, num_bins=50)
    percentiles_list = np.arange(0, 101, 10)
    # plot_quantile_histogram(path_to_npy_file, percentiles_list)
    # plot_log_histogram(path_to_npy_file)
    plot_custom_histogram(path_to_npy_file, [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240,
                                             20480, 40960, 81920, 163840, 327680, 409600])
