from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/v8/modified/yolov8-C2f-Star-CAA.yaml")  

# Train the model
results = model.info(verbose=True)
