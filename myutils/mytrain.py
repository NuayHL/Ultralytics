from ultralytics import YOLO

# Load a model
model = YOLO("cfg/yolo12s.yaml")  # build a new model from YAML

# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, batch=16, name='test/test', save_json=True)
