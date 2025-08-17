from ultralytics import YOLO
import torch
import platform
import datetime
from decimal import Decimal, ROUND_HALF_UP

def quantize_to_2dp(value):
    if isinstance(value, (int, float)):
        value_decimal = Decimal(str(value))
    elif isinstance(value, Decimal):
        value_decimal = value
    else:
        raise TypeError(f"Unsupported type for quantization: {type(value)}")

    two_places = Decimal('0.01')
    return str(value_decimal.quantize(two_places, rounding=ROUND_HALF_UP))

def format_metrics(model_path, data_path, name='val/exp', batch=8, imgsz=640):
    model = YOLO(model_path)
    metrics = model.val(data=data_path, name=name, batch=batch, imgsz=imgsz)

    return_dicts = dict()

    return_dicts['mAP50'] = map50 = metrics.box.map50
    return_dicts['mAP75'] = map75 = metrics.box.map75
    return_dicts['mAP'] = map = metrics.box.map

    total_images = metrics.total_images

    headers = ['Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP75','mAP50-95']
    lines = ["|" + "|".join(headers) + "|",
             "|" + "|".join(["---"] * len(headers)) + "|",
              f"|all|{total_images}|{metrics.nt_per_class.sum()}|"
         f"{quantize_to_2dp(metrics.box.mp * 100)}|{quantize_to_2dp(metrics.box.mr * 100)}|"
         f"{quantize_to_2dp(map50 * 100)}|{quantize_to_2dp(map75 * 100)}|{quantize_to_2dp(map * 100)}|"]

    for class_name, images, instances, p, r, ap50, ap75, ap in zip(
        metrics.names.values(),
        metrics.nt_per_image,
        metrics.nt_per_class,
        metrics.box.p,
        metrics.box.r,
        metrics.box.ap50,
        metrics.box.all_ap[:, 5],
        metrics.box.ap,
    ):
        lines.append(f"|{class_name}|{images}|{instances}|"
                     f"{quantize_to_2dp(p * 100)}|{quantize_to_2dp(r * 100)}|"
                     f"{quantize_to_2dp(ap50 * 100)}|{quantize_to_2dp(ap75 * 100)}|{quantize_to_2dp(ap * 100)}|")

    sys_info = {
        "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model": model_path,
        "Data": data_path,
        "mAP50": quantize_to_2dp(map50 * 100),
        "mAP75": quantize_to_2dp(map75 * 100),
        "mAP50-95": quantize_to_2dp(map * 100),
        "Device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "CUDA": torch.version.cuda if torch.cuda.is_available() else "N/A",
        "PyTorch": torch.__version__,
        "Python": platform.python_version(),
    }

    return_dicts['sys_info'] = "\n".join([f"{k}: {v}" for k, v in sys_info.items()])
    return_dicts['metrics_table'] = "\n".join(lines)
    return return_dicts

def print_info(metrics_dicts):
    print("\n# === Summary ===")
    print(metrics_dicts['sys_info'])

    print("\n# === Class-wise Metrics (Markdown Table) ===")
    print(metrics_dicts['metrics_table'])

if __name__ == "__main__":
    print_info(
    format_metrics(
        model_path='../runs/detect/visdrone/v12s_pre_assign_bce1/weights/best.pt',
        data_path='../ultralytics/cfg/datasets/VisDrone.yaml',
        name='val/test',
        batch=8,
        imgsz=640
    ))
