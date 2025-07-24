echo 'Starts!'
echo 'yolov12s_assign_record'
yolo detect train \
    data=ultralytics/cfg/datasets/VisDrone.yaml \
    model=cfg/yolo12s_assign_visdrone.yaml \
    epochs=150 \
    batch=16 \
    imgsz=640 \
    device=0 \
    save=True \
    name=assign_record/v12s_vis > terminal_log/yolov12s_assign_record.log 2>&1