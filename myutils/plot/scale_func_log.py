import torch
import matplotlib.pyplot as plt

def visualize_log_functions(
        func_objects,
        x_range=(1, 1e7),
        num_points=500,
        shared_plot=True,
        y_clip_min=None,
        title="Log-based Function Visualization"
    ):
    """
    Visualize log-based functions defined inside CreateFunc-like class.

    Args:
        func_objects (list): List of `CreateFunc` objects
        x_range (tuple): (min_x, max_x), must be > 0
        num_points (int): number of sampling points in log scale
        shared_plot (bool): True: plot all curves on one figure; False: separate figures
        y_clip_min (float or None): if not None, clip y < y_clip_min to this value
        title (str): shared plot title
    """

    assert x_range[0] > 0, "x_range must be positive for log scale"
    x = torch.logspace(
        torch.log10(torch.tensor(x_range[0])),
        torch.log10(torch.tensor(x_range[1])),
        steps=num_points
    )

    if shared_plot:
        plt.figure(figsize=(8, 6))

    for func_obj in func_objects:
        y = func_obj(x)

        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        # 防止对数函数中 nan 或 -inf 溢出
        if y_clip_min is not None:
            y = torch.clamp(y, min=y_clip_min)

        if not shared_plot:
            plt.figure(figsize=(8, 6))

        plt.plot(x.cpu().numpy(), y.cpu().numpy(), label=func_obj.func_name)
        plt.xscale("log")

        if not shared_plot:
            plt.title(func_obj.func_name)
            plt.xlabel("x (log scale)")
            plt.ylabel("y")
            plt.grid(True, which="both")
            plt.legend()
            plt.show()

    if shared_plot:
        plt.title(title)
        plt.xlabel("x (log scale)")
        plt.ylabel("y")
        plt.grid(True, which="both")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    from ultralytics.utils.mla_dab import CreateFunc
    f2 = CreateFunc("log_sigmoid", a=0.2, b=1.0, c=-5, d=3.0)
    # f3 = CreateFunc("static", a=0.5)

    visualize_log_functions(
        func_objects=[CreateFunc("log_linear", -0.1, 1.3),
                      CreateFunc("log_linear", 0.33, 3.0),
                      CreateFunc("log_sigmoid", 0.3, 2.5, 2.0, 1.7),],
        x_range=(1, 1e7),
        shared_plot=True,
    )
