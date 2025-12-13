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


def run_experiment(exp_name, extra_tags, exp_prefix, data_yaml, model_yaml, log_root, trainer=None):
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
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Experiment: {exp_name}...")
    
    log_file = log_root / f"{exp_name}.log"
    project_path = f"{exp_prefix}/{exp_name}"
    
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
            # 'save_period': 1,
            'exist_ok': True,
            'trainer': trainer
        }
          
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
        # "--no-files",
        "--extra-tags"
    ] + extra_tags
    
    # Run the subprocess. 
    # capturing_output=False ensures this script's output still shows on your screen 
    # (or you can redirect this too if you want).
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Post-processing finished.\n")
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Post-processing failed with error code {result.returncode}.\n")

def main(exp_prefix="hituav", data_yaml="ultralytics/cfg/datasets/hit-uav.yaml", 
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
            trainer=exp.get("trainer", None))


if __name__ == "__main__":

    EXP_LIST = [
                dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_l2_7_topk7",
                     extra_tags=["v12", "v12s"],
                     model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_7_topk7.yaml",
                     trainer=None),
                dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_l2_10_topk7",
                     extra_tags=["v12", "v12s"],
                     model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_10_topk7.yaml",
                     trainer=None),
                dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_IoU_topk7",
                     extra_tags=["v12", "v12s"],
                     model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_IoU_topk7.yaml",
                     trainer=None),
                dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_l2_pow4_7_topk7",
                     extra_tags=["v12", "v12s"],
                     model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_pow4_7_topk7.yaml",
                     trainer=None),
                dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_l2_pow3_7_topk7",
                     extra_tags=["v12", "v12s"],
                     model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_pow3_7_topk7.yaml",
                     trainer=None),
    ]

    EXP_PREFIX = "hituav"
    DATA_YAML = "ultralytics/cfg/datasets/hit-uav.yaml"

    main(
        exp_prefix=EXP_PREFIX,
        data_yaml=DATA_YAML,
        exp_list=EXP_LIST
    )


    EXP_PREFIX = "aitodv2"
    DATA_YAML = "ultralytics/cfg/datasets/ai-todv2.yaml"

    EXP_LIST = [
                dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_l2_7_topk7",
                     extra_tags=["v12", "v12s"],
                     model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_7_topk7.yaml",
                     trainer=None),
                dict(exp_name="v12s_assign4ciou_align_hausdorff_ext_IoU_topk7",
                     extra_tags=["v12", "v12s"],
                     model_yaml="cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_IoU_topk7.yaml",
                     trainer=None),
    ]
    
    main(
        exp_prefix=EXP_PREFIX,
        data_yaml=DATA_YAML,
        exp_list=EXP_LIST
    )

    EXP_PREFIX = "visdrone"
    DATA_YAML = "ultralytics/cfg/datasets/VisDrone.yaml"

    main(
        exp_prefix=EXP_PREFIX,
        data_yaml=DATA_YAML,
        exp_list=EXP_LIST
    )