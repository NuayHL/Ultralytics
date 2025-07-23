echo 'Starts!'
echo 'yolov12s'
yolo detect train \
    data=ultralytics/cfg/datasets/VisDrone.yaml \
    model=mycfg/yolo12s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=visdrone/v12s > terminal_log/yolov12.log 2>&1

echo 'yolov11s'
yolo detect train \
    data=ultralytics/cfg/datasets/VisDrone.yaml \
    model=mycfg/yolo11s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=visdrone/v11s > terminal_log/yolov11.log 2>&1

echo 'yolov8s'
yolo detect train \
    data=ultralytics/cfg/datasets/VisDrone.yaml \
    model=mycfg/yolov8s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=visdrone/v8s > terminal_log/yolov8.log 2>&1
