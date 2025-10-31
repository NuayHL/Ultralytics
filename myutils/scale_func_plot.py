import numpy as np
import matplotlib.pyplot as plt
import math

def compute_scale_ratio_from_ratio_scalar(r, r_max=1.0):
    """标量版本，用于画图"""
    if 0.25 <= r < 1:
        return r_max * (r - 0.25) / 0.75
    elif 1 <= r < 2:
        return r_max
    elif 2 <= r < 2.5:
        return r_max * (2.5 - r)
    else:
        return 0

def compute_scale_ratio_from_ratio_log_gaussian(r, r_max=1.5, sigma=0.5):
    log_r = np.log(r + 1e-9)
    peak = math.log(1.5)
    return r_max * np.exp(-((log_r - peak) ** 2) / (2 * sigma ** 2))

def smooth_plateau_function(r, r_max=1.0, a=3.0, b=1.0):
    """平滑上凸 + 高台 + 缓降曲线"""
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    # 上升段 (控制上凸程度)
    rise = sigmoid(a * (r - 0.5))
    # 缓降段 (控制下降速度)
    fall = 1 - 0.25 * sigmoid(b * (r - 2.0))
    return r_max * rise * fall

# 生成输入区间
r = np.linspace(0, 3, 500)
# s = np.array([compute_scale_ratio_from_ratio_scalar(x) for x in r])
# s = np.array([compute_scale_ratio_from_ratio_log_gaussian(x) for x in r])
s = np.array([smooth_plateau_function(x) for x in r])

# 绘图
plt.figure(figsize=(7, 4))
plt.plot(r, s, label=r"$s(r)$", color="royalblue", linewidth=2)
plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
plt.axvline(1, color="gray", linestyle=":", linewidth=0.8)
plt.axvline(2, color="gray", linestyle=":", linewidth=0.8)
plt.axvline(0.25, color="gray", linestyle=":", linewidth=0.8)
plt.axvline(2.5, color="gray", linestyle=":", linewidth=0.8)

plt.title("Piecewise Scale Ratio Function", fontsize=13)
plt.xlabel(r"$r$ (stride / w or stride / h)")
plt.ylabel(r"$s(r)$")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
