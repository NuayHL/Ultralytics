from ultralytics import YOLO

# Load a model
# model = YOLO("../cfg/yolo12s_assign_mix.yaml")  # build a new model from YAML
# model = YOLO("../cfg/yolo12s_hbg.yaml")  # build a new model from YAML
model = YOLO("../cfg/ab_hyper/yolo12s_dab.yaml")  # build a new model from YAML
# model = YOLO("../cfg/yolo12s_permute_random.yaml")  # build a new model from YAML

# Train the model
# results = model.train(data="VisDrone.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True, pretrained='../yolo12s.pt')
results = model.train(data="VOC.yaml", epochs=150, imgsz=640, batch=16, name='test/test', save_json=True, )
