from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n_custom.pt")  # load a custom model

# Validate the model
metrics = model.val(data="data.yaml", imgsz=640, batch=16, conf=0.09, iou=0.6, plots=True)  # no arguments needed, dataset and settings remembered
# mAP, precision, recall
# For single-class dataset
print(metrics.to_json())
#print(metrics.box.map)  # map50-95
#metrics.box.map50  # map50
#metrics.box.map75  # map75
#metrics.box.maps  # a list contains map50-95 of each category