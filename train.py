from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/yolov8.yaml")  
# Train the model
results = model.info(verbose=True, detailed=False)
