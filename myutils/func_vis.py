import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Union


def configure_professional_style():
    """
    Configures matplotlib params to mimic professional math software (e.g., Maple/Mathematica).
    """
    plt.rcParams.update({
        # Font settings: Use serif for a formal academic look
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',  # Computer Modern (LaTeX style)

        # Font sizes: Smaller, concise labels as requested
        'font.size': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,

        # Line and Axes aesthetics
        'lines.linewidth': 1.5,
        'axes.linewidth': 1.0,  # Slightly thicker border
        'axes.grid': True,  # Enable grid
        'grid.alpha': 0.3,  # Subtle grid
        'grid.linestyle': '--',
        'grid.color': 'gray',

        # Ticks: Inward facing ticks look more "scientific"
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
    })


def plot_functions(
        func_list: List[List[Union[str, Callable]]],
        x_range: Tuple[float, float],
        stride: float,
        log_x: bool = False,
        log_y: bool = False,
        title: str = "Function Visualization"
):
    """
    Plots a list of univariate functions over a specified range.

    Args:
        func_list: List of [name, func], e.g., [['sin(x)', np.sin], ...].
        x_range: Tuple (start, end).
        stride: Step size for the x-axis sampling.
        log_x: Boolean, enable logarithmic scale for X-axis.
        log_y: Boolean, enable logarithmic scale for Y-axis.
        title: Title of the plot.
    """

    # Apply the aesthetic style
    configure_professional_style()

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)

    start, end = x_range

    # Generate Domain X
    if log_x:
        # If log scale is requested, linear stride might result in poor resolution
        # at the lower end. We generate points geometrically but respect bounds.
        # However, to strictly follow "stride" logic implies linear steps.
        # Here we prioritize visual smoothness for log plots.
        if start <= 0:
            raise ValueError("Range start must be > 0 for log_x scale.")
        num_points = int((end - start) / stride)
        # Use logspace for smoother curves in log-x view
        x = np.geomspace(start, end, max(num_points, 200))
    else:
        x = np.arange(start, end, stride)

    # Color cycle for high contrast (Maple-like: Red, Blue, Green, Black)
    colors = ['#D32F2F', '#1976D2', '#388E3C', '#000000', '#7B1FA2']

    # Plotting Loop
    for i, item in enumerate(func_list):
        label_name = item[0]
        func = item[1]

        try:
            y = func(x)

            # Handle log_y domain errors (mask <= 0 values)
            if log_y:
                y = np.ma.masked_less_equal(y, 0)

            color = colors[i % len(colors)]
            ax.plot(x, y, label=label_name, color=color)

        except Exception as e:
            print(f"Error plotting function '{label_name}': {e}")

    # Axis Scaling
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')

    # Labeling
    ax.set_xlabel('x', style='italic')  # Mathematical convention
    ax.set_ylabel('f(x)', style='italic')
    ax.set_title(title, weight='bold')

    # Legend settings (frame, location)
    if len(func_list) > 1:
        ax.legend(frameon=True, fancybox=False, edgecolor='black', framealpha=0.8)

    # Layout adjustment
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import numpy as np

    ls = 30
    d0 = 10
    p = 0.5

    alpha = lambda x: 0.5 + 0.5 * np.exp(-x / ls)
    test1 = lambda x: x ** alpha1(x)
    alpha1 = lambda d: 0.9 + 0.1 / (1.0 + (d / d0) ** p)

    eps = 1e-6
    # func_list = [
    #     ['test', lambda area: (1 - area**p / (area**p + 36**p)) * area \
    #         + (area**p / (area**p + 36**p)) * np.sqrt(area + eps) ],
    #     ['1/2', lambda area: np.sqrt(area + eps)],
    #     ['x', lambda x: x],
    #     ['test1', test1 ],
    #     ['alpha', alpha],
    #     ['alpha1', alpha1]
    # ]

    def modified_activation(x):
        # Use np.where for vectorized conditional logic
        # Condition: x < 5
        # True:  1.8 * x + 16
        # False: x ** 2
        return np.where(x < 5, 1.8 * x + 16, x ** 2)


    class SmoothLSEActivation(nn.Module):
        def __init__(self):
            super().__init__()
            # We can make the linear parameters learnable if needed,
            # but here we fix them as per your constraints.
            self.slope = 0.1
            self.bias = 20.0

        def forward(self, x):
            """
            Implements f(x) = SoftMax(x^2, 1.5x + 16)
            Using logaddexp for numerical stability:
            log(exp(x^2) + exp(1.5x + 16))
            """
            x = torch.tensor(x)
            quad_part = x ** 2
            linear_part = self.slope * x ** 2 + self.bias

            # torch.logaddexp avoids overflow when exponentials are large
            return torch.logaddexp(quad_part, linear_part).numpy()


    def designed_function(x):
        """
        Implements the shifted Hill function based on user constraints.
        Formula: f(x) = y_min + (y_max - y_min) * (x^n / (x^n + k^n))
        """
        y_min = 0.3
        y_max = 16.0
        k = 16.0  # Semi-saturation point (controls where the curve is at 50% rise)
        n = 3.0  # Hill coefficient (controls steepness)

        # Calculate the ratio part
        ratio = np.power(x, n) / (np.power(x, n) + np.power(k, n))

        return y_min + (y_max - y_min) * ratio

    func_list = [

        # ['x^2+buff', lambda x: x**2 + 20 * (1 - np.sqrt(x + 2)/5)],
        # ['x^2', lambda x: x ** 2],
        ['test', designed_function],
        # ['x^2+ext1', SmoothLSEActivation()]
    ]

    # plot_functions(func_list, (eps, 100), 1, log_x=True)
    plot_functions(func_list, (eps, 72), 0.1,)
