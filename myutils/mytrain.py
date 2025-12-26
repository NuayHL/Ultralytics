from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainerWithDynamicAssigner

# Load a model
model = YOLO("cfg/assign_iou/yolo12s_assign4ciou_align_hausdorff_ext_l2_rfix_pow4_7.yaml")  # build a new model from YAML
# model = YOLO("cfg/yolo12s.yaml")  # build a new model from YAML

# Train the model
# results = model.train(data="VisDrone.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True)
results = model.train(data="hit-uav.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True, seed=0)
# results = model.train(data="ai-todv2.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True, seed=0)
