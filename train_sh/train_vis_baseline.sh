echo 'Starts!'
echo 'yolov12s'
yolo detect train \
    data=ultralytics/cfg/datasets/VisDrone.yaml \
    model=cfg/yolo12s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=visdrone/v12s > terminal_log/yolo12s_vis.log 2>&1

echo 'yolov11s'
yolo detect train \
    data=ultralytics/cfg/datasets/VisDrone.yaml \
    model=cfg/yolo11s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=visdrone/v11s > terminal_log/yolo11s_vis.log 2>&1

echo 'yolov8s'
yolo detect train \
    data=ultralytics/cfg/datasets/VisDrone.yaml \
    model=cfg/yolov8s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=visdrone/v8s > terminal_log/yolov8s_vis.log 2>&1
