EXP_PREFIX="visdrone"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
#LOG_DIR="terminal_log/terminal_log_${EXP_PREFIX}_${TIMESTAMP}"
LOG_DIR="terminal_log/terminal_log_${EXP_PREFIX}_${TIMESTAMP}"
if [ -d "$LOG_DIR" ]; then
    LOG_DIR="${LOG_DIR}_$(date +%s)"
fi
mkdir -p "$LOG_DIR"

echo 'Starts!'

EXP_NAME="vis_v12s_bb_cbb"
echo "${EXP_NAME}"
yolo detect train data=ultralytics/cfg/datasets/VisDrone.yaml \
                  model=cfg/yolo12s_cbb.yaml \
                  epochs=150 batch=16 imgsz=640 device=0 save=True \
                  name="${EXP_PREFIX}/${EXP_NAME}" pretrained=yolov8m.pt \
                  > "$LOG_DIR/${EXP_NAME}.log" 2>&1

EXP_NAME="vis_v12s_bb_ca"
echo "${EXP_NAME}"
yolo detect train data=ultralytics/cfg/datasets/VisDrone.yaml \
                  model=cfg/yolo12s_ca.yaml \
                  epochs=150 batch=16 imgsz=640 device=0 save=True \
                  name="${EXP_PREFIX}/${EXP_NAME}" pretrained=yolov8m.pt \
                  > "$LOG_DIR/${EXP_NAME}.log" 2>&1

EXP_NAME="vis_v12s_bb_se"
echo "${EXP_NAME}"
yolo detect train data=ultralytics/cfg/datasets/VisDrone.yaml \
                  model=cfg/yolo12s_se.yaml \
                  epochs=150 batch=16 imgsz=640 device=0 save=True \
                  name="${EXP_PREFIX}/${EXP_NAME}" pretrained=yolov8m.pt \
                  > "$LOG_DIR/${EXP_NAME}.log" 2>&1

