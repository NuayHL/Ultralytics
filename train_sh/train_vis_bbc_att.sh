EXP_PREFIX="visdrone"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
#LOG_DIR="terminal_log/terminal_log_${EXP_PREFIX}_${TIMESTAMP}"
LOG_DIR="terminal_log/terminal_log_${EXP_PREFIX}"
if [ -d "$LOG_DIR" ]; then
    LOG_DIR="${LOG_DIR}_$(date +%s)"
fi
mkdir -p "$LOG_DIR"

echo 'Starts!'

EXP_NAME="vis_v12s_bb_cbase"
echo "${EXP_NAME}"
yolo detect train data=ultralytics/cfg/datasets/VisDrone.yaml \
                  model=cfg/yolo12s_cbase.yaml \
                  epochs=150 batch=16 imgsz=640 device=0 save=True \
                  name="${EXP_PREFIX}/${EXP_NAME}" \
                  > "$LOG_DIR/${EXP_NAME}.log" 2>&1

EXP_NAME="vis_v12s_bb_cbcb"
echo "${EXP_NAME}"
yolo detect train data=ultralytics/cfg/datasets/VisDrone.yaml \
                  model=cfg/yolo12s_cbcb.yaml \
                  epochs=150 batch=16 imgsz=640 device=0 save=True \
                  name="${EXP_PREFIX}/${EXP_NAME}" \
                  > "$LOG_DIR/${EXP_NAME}.log" 2>&1

EXP_NAME="vis_v12s_bb_cbb_c"
echo "${EXP_NAME}"
yolo detect train data=ultralytics/cfg/datasets/VisDrone.yaml \
                  model=cfg/yolo12s_cbb_c.yaml \
                  epochs=150 batch=16 imgsz=640 device=0 save=True \
                  name="${EXP_PREFIX}/${EXP_NAME}" \
                  > "$LOG_DIR/${EXP_NAME}.log" 2>&1



