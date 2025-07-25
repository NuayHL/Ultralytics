echo 'Starts!'
echo 'yolov12s'
yolo detect train \
    data=ultralytics/cfg/datasets/VOC.yaml \
    model=mycfg/yolo12s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=voc/v12s > terminal_log/yolo12s_voc.log 2>&1

echo 'yolov11s'
yolo detect train \
    data=ultralytics/cfg/datasets/VOC.yaml \
    model=mycfg/yolo11s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=voc/v11s > terminal_log/yolo11s_voc.log 2>&1

echo 'yolov8s'
yolo detect train \
    data=ultralytics/cfg/datasets/VOC.yaml \
    model=mycfg/yolov8s.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=voc/v8s > terminal_log/yolov8s_voc.log 2>&1
