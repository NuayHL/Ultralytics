import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import YAML
from ultralytics.models.yolo.detect import DetectionTrainerWithDynamicAssigner

class LogLogger:
    """
    Context manager to redirect low-level file descriptors (stdout/stderr).
    This captures output from C libraries, tqdm, and pre-initialized loggers.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.log_file = None
        self.original_stdout_fd = None
        self.original_stderr_fd = None

    def __enter__(self):
        # 1. Flush python buffers to ensure previous outputs are written
        sys.stdout.flush()
        sys.stderr.flush()

        # 2. Open the log file
        self.log_file = open(self.filepath, 'w', encoding='utf-8')
        
        # 3. Save original file descriptors (1 for stdout, 2 for stderr)
        self.original_stdout_fd = os.dup(1)
        self.original_stderr_fd = os.dup(2)

        # 4. Redirect standard file descriptors to the log file's descriptor
        # This is the "Nuclear" option: it affects the entire process space
        os.dup2(self.log_file.fileno(), 1)
        os.dup2(self.log_file.fileno(), 2)

        return self.log_file

    def __exit__(self, exc_type, exc_val, exc_tb):
        # 1. Flush any pending output in the redirected stream
        sys.stdout.flush()
        sys.stderr.flush()

        # 2. Restore original file descriptors
        os.dup2(self.original_stdout_fd, 1)
        os.dup2(self.original_stderr_fd, 2)

        # 3. Close the duplicated backup descriptors
        os.close(self.original_stdout_fd)
        os.close(self.original_stderr_fd)

        # 4. Close the log file
        if self.log_file:
            self.log_file.close()


def run_experiment(exp_name, extra_tags, exp_prefix, data_yaml, model_yaml, log_root, trainer=None,
                   other_train_kwargs={}, no_files_upload=False):
    """
    Runs a single training experiment and triggers the post-processing script.
    
    Args:
        exp_name: 实验名称
        extra_tags: 额外的标签列表
        exp_prefix: 实验前缀
        data_yaml: 数据配置文件路径
        model_yaml: 模型配置文件路径
        log_root: 日志根目录
        trainer: Trainer 类，如果为 None 则使用默认 Trainer
        other_train_kwargs: 其它训练参数
    """

    log_file = log_root / f"{exp_name}.log"
    project_path = f"{exp_prefix}/{exp_name}"
    print(f"Log file: {log_file}")
    print(f"Project path: {project_path}")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Experiment: {exp_name}...")

    # 1. Train with log redirection
    # Everything inside this 'with' block goes to the file, not the screen.
    with LogLogger(log_file):
        print(f"Experiment: {exp_name}")
        print(f"Start Time: {datetime.now()}")
        
        # Initialize model
        model = YOLO(model_yaml)
        
        # 准备训练参数
        train_kwargs = {
            'data': data_yaml,
            'epochs': 150,
            'batch': 16,
            'imgsz': 640,
            'save': True,
            'project': "runs/detect",
            'name': project_path,
            'amp': True,
            'save_period': -1,
            'exist_ok': True,
            'trainer': trainer
        }

        train_kwargs.update(other_train_kwargs)
          
        # Train
        model.train(**train_kwargs)
        print("Training completed.")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Training finished. Log saved to {log_file}")

    # 2. Run external post-processing script
    # We use subprocess to call your existing script exactly as before.
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Running post-processing...")

    cmd = [
        "python", "myutils/main.py",
        "--exp-path", f"runs/detect/{exp_prefix}/{exp_name}",
        "--data-path", data_yaml,
    ]

    if no_files_upload:
        cmd.append("--no-files")

    # Only add the flag if the list is not empty
    if extra_tags:
        cmd.append("--extra-tags")
        cmd.extend(extra_tags)
    
    # Run the subprocess. 
    # capturing_output=False ensures this script's output still shows on your screen 
    # (or you can redirect this too if you want).
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Post-processing finished.\n")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Post-processing failed with error code {result.returncode}.\n")

def main(exp_prefix="hituav", data_yaml="ultralytics/cfg/datasets/hit-uav.yaml", no_files_upload=False,
         exp_list=[{"exp_name": "v12s_record", "extra_tags": ["v12", "v12s", "baseline"]},],):

    log_root = Path("terminal_log") / f"terminal_log_{exp_prefix}"
    os.makedirs(log_root, exist_ok=True)
    
    print('Starts!')
    for exp in exp_list:
        run_experiment(
            exp_name=exp["exp_name"],
            extra_tags=exp["extra_tags"],
            exp_prefix=exp_prefix,
            data_yaml=data_yaml,
            model_yaml=exp["model_yaml"],
            log_root=log_root,
            trainer=exp.get("trainer", None),
            other_train_kwargs=exp.get("other_train_kwargs", {}),
            no_files_upload=no_files_upload
        )


if __name__ == "__main__":
    EXP_LIST = [
        # dict(exp_name="v12s_topk7_no_amp",
        #      extra_tags=["v12", "v12s", "baseline", "no_amp"],
        #      model_yaml="cfg/yolo12s_topk7.yaml",
        #      trainer=None, other_train_kwargs=dict(amp=False)),
        dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_l2_pow4_7",
             extra_tags=["v12", "v12s"],
             model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_pow4_7.yaml",
             trainer=None),

        dict(exp_name="v12n_assign4ciou_align_hausdorff_ext_l2_pow4_7",
             extra_tags=["v12", "v12n"],
             model_yaml="cfg/assign_iou/yolo12n_assign4ciou_align_hausdorff_ext_l2_pow4_7.yaml",
             trainer=None),
        dict(exp_name="v12n",
             extra_tags=["v12", "v12n", "baseline"],
             model_yaml="cfg/yolo12n.yaml",
             trainer=None),

        dict(exp_name="v12m_assign4ciou_align_hausdorff_ext_l2_pow4_7",
             extra_tags=["v12", "v12m"],
             model_yaml="cfg/assign_iou/yolo12m_assign4ciou_align_hausdorff_ext_l2_pow4_7.yaml",
             trainer=None),

        dict(exp_name="v12l_assign4ciou_align_hausdorff_ext_l2_pow4_7",
             extra_tags=["v12", "v12l"],
             model_yaml="cfg/assign_iou/yolo12l_assign4ciou_align_hausdorff_ext_l2_pow4_7.yaml",
             trainer=None),
    ]

    # EXP_PREFIX = "visdrone"
    # DATA_YAML = "ultralytics/cfg/datasets/VisDrone.yaml"
    # main(
    #     exp_prefix=EXP_PREFIX,
    #     data_yaml=DATA_YAML,
    #     exp_list=EXP_LIST
    # )

    # EXP_LIST = [
    #     dict(exp_name="v12s_topk7_no_amp",
    #          extra_tags=["v12", "v12s", "baseline", "no_amp"],
    #          model_yaml="cfg/yolo12s_topk7.yaml",
    #          trainer=None, other_train_kwargs=dict(amp=False)),
    #     dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_l2_fix_pow4_12_topk7_no_amp",
    #          extra_tags=["v12", "v12s", "no_amp"],
    #          model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_fix_pow4_12_topk7.yaml",
    #          trainer=None, other_train_kwargs=dict(amp=False)),
    # ]
    #
    EXP_PREFIX = "hituav"
    DATA_YAML = "ultralytics/cfg/datasets/hit-uav.yaml"
    main(
        exp_prefix=EXP_PREFIX,
        data_yaml=DATA_YAML,
        exp_list=EXP_LIST
    )
    #
    # EXP_PREFIX = "aitodv2"
    # DATA_YAML = "ultralytics/cfg/datasets/ai-todv2.yaml"
    # main(
    #     exp_prefix=EXP_PREFIX,
    #     data_yaml=DATA_YAML,
    #     exp_list=EXP_LIST
    # )

