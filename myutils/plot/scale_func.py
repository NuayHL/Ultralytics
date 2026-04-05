import torch
import inspect
import matplotlib.pyplot as plt
from ultralytics.utils.mla_scale import TaskAlignedAssigner_dScale

def visualize_scale_functions(
        func_names,
        r_range=(0, 3),
        num_points=500,
        shared_plot=False,
        save_name=None,
        **func_kwargs
    ):
    """
    可视化 TaskAlignedAssigner_dScale 内的 scale 函数。

    Args:
        cls: 包含静态方法的类（例如 TaskAlignedAssigner_dScale）
        func_names (list[str]): 要可视化的函数名
        r_range (tuple): r 的取值范围
        num_points (int): r 的采样点数
        shared_plot (bool): True 画在一张图里；False 每个函数一张图
        **func_kwargs: 每个函数对应的参数，例如：
            visualize_scale_functions(..., func_1={"r_max": 2.0})
    """
    cls = TaskAlignedAssigner_dScale
    # 生成 r 值 (1,1,N)
    r = torch.linspace(r_range[0], r_range[1], num_points).view(1, 1, -1)

    plt.figure(figsize=(8, 6)) if shared_plot else None

    for func_name in func_names:
        if not hasattr(cls, func_name):
            raise ValueError(f"{func_name} not found in {cls.__name__}")

        func = getattr(cls, func_name)
        if not callable(func):
            raise ValueError(f"{func_name} is not a callable function")

        # 自动解析参数
        sig = inspect.signature(func)
        valid_kwargs = {}

        # 对该函数的参数进行过滤与匹配
        if func_name in func_kwargs:
            for k, v in func_kwargs[func_name].items():
                if k in sig.parameters:
                    valid_kwargs[k] = v
                else:
                    raise ValueError(f"Function {func_name} does not accept argument {k}")

        # 调用函数
        with torch.no_grad():
            y = func(r, **valid_kwargs)  # r shape is (1,1,N)

        # squeeze 成 (N,)
        y = y.view(-1).cpu()

        # 决定是否单独开图
        if not shared_plot:
            plt.figure(figsize=(7, 5))

        plt.plot(r.view(-1).cpu(), y, label=func_name)

        if not shared_plot:
            plt.title(func_name)
            plt.xlabel("r")
            plt.ylabel("scale_ratio")
            plt.grid(True)
            plt.legend()
            plt.show()
            if save_name:
                plt.savefig(save_name + f"_{func_name}.png", dpi=300)
                plt.close()

    if shared_plot:
        plt.title("Scale Ratio Functions")
        plt.xlabel("r")
        plt.ylabel("scale_ratio")
        plt.grid(True)
        plt.legend()
        plt.show()
    if save_name:
        plt.savefig(save_name + ".png", dpi=300)
        plt.close()

if __name__ == "__main__":
    visualize_scale_functions(["func_1",
                               "func_2",
                               "func_smooth_1",
                               "func_gaussian_dip",
                               "func_exp_saturate",
                               "func_inverse_smooth",
                               "func_scale_adaptive"],
                              # save_name="scale_func_plot",
                              r_range=(0, 3),
                              shared_plot=True,
                              func_1={"r_max": 1.0},
                              func_2={"r_max": 1.0},
                              func_smooth_1={"r_max": 1.0},
                              func_gaussian_dip={"r_max": 1.0, "r_ideal": 0.25, "sigma": 0.1},
                              func_exp_saturate={"r_max": 1.0, "a": 1.0},
                              func_inverse_smooth={"r_min": 1.0, "r_max": 1.5, "k": 12},
                              func_scale_adaptive={"scale_base":0.5, "scale_boost": 0.5, "gamma":2.5})