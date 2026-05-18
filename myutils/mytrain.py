from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainerWithDynamicAssigner

other_train_kwargs = dict(optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=False, mosaic=0.5, close_mosaic=15)
aitod_train_kwargs = dict(optimizer="SGD", lr0=0.01, lrf=0.01, cos_lr=False, mosaic=1.0, close_mosaic=15)

# Load a model
# model = YOLO("../runs/detect/visdrone/v12s_record/weights/epoch130.pt")  # build a new model from YAML
model = YOLO("cfg/usaa/yolo12s_usaa_raw_a1b4_ra32_rtadd.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="ai-todv2.yaml", 
                    epochs=150, 
                    imgsz=640, 
                    batch=16, 
                    name='test/test', 
                    save_json=True, 
                    **aitod_train_kwargs)
# results = model.train(data="hit-uav.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True, seed=0)
# results = model.train(data="ai-todv2.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True, seed=0)
# results = model.train(data="VisDrone.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True, seed=0)
