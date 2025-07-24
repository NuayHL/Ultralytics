from ultralytics import YOLO
#
model = YOLO('runs/detect/iou/weights/best.pt')

# metrics = model.val(data="ultralytics/cfg/datasets/VisDrone.yaml", imgsz=640, batch=8, save_json=True)
metrics = model.val(data="ultralytics/cfg/datasets/VisDrone.yaml", name='val/test', batch=8, imgsz=640)
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.map)  # mAP50-95
